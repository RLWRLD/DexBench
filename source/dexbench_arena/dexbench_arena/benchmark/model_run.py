# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""A model's run of a benchmark cell or sweep: the checked-in file with the model's policy block in place."""

from __future__ import annotations

import copy
import datetime
import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import yaml
from isaaclab_arena.evaluation.arena_experiment_config_loader import load_arena_experiment_from_config_file

import dexbench_arena.environments  # noqa: F401  -- registers the environment types the cell files name
from dexbench_arena.environments.manipulation.scene_environment_base import environment_cfg_snapshot
from dexbench_arena.benchmark.models import MODEL_ENVIRONMENT, ModelEntry, model_record, plain, resolve_model
from dexbench_arena.benchmark.protocol import (
    DEFAULT_EVAL_VERSION,
    RENDER_VARIABLES,
    PolicyExecution,
    cell_rows,
    launcher_hook_variables,
)
from dexbench_arena.benchmark.spec import TASKS_BY_NAME
from dexbench_arena.benchmark.frozen import load_frozen_set, pinned_path, pinned_version_rows, write_rows
from dexbench_arena.evaluation.artifacts import publish_new, recorded, redact_api_tokens, replace_atomic, sha256_bytes
from dexbench_arena.tasks.object_axis import asset_major_problem
from dexbench_arena.benchmark.reference_settings import evaluation_contract_sha256
from dexbench_arena.benchmark.run_manifest import ARENA_OUTPUT, COMPOSED, RUN_MANIFEST, RUN_MANIFEST_VERSION
from dexbench_arena.evaluation.evaluation_signature import code_revision, declaration_problems

DEPLOYMENT_FIELDS = ("remote_host", "remote_port", "remote_url", "policy_config_yaml_path", "policy_device",
                     "cost_budget_usd", "cost_ledger_path")
"""What a deployment may change (``--set``): where the server is, the client's contract file and device, and a
planner's cost cap and the ledger it sums. The server's token is not one: it comes from ``DEXBENCH_POLICY_API_TOKEN``
in the runner's environment, so no file a run writes holds it (``artifacts.redact_api_tokens``). Everything else in a
block is the model: its type, key, horizon and action representation
(``relative_joint_slices``, ``relative_action_groups``, which would otherwise turn ``cosmos_rel`` or ``gr00t_rel``
into an absolute run under its label)."""


def model_policy(model: ModelEntry, deployment: Mapping[str, Any] | None = None, task: str | None = None) -> dict[str, Any]:
    """The model's policy block, with its fields for ``task`` over it, and ``deployment`` (fields the block declares,
    e.g. ``remote_port``) changed."""
    if not model.available:
        raise ValueError(f"{model.key} has no adapter in this repo yet ({model.note})")
    policy = {**plain(model.policy), **plain(dict(model.task_policy.get(task, {}) if task else {}))}
    for key, value in (deployment or {}).items():
        if key not in DEPLOYMENT_FIELDS or key not in model.policy:
            raise ValueError(f"{key!r} is not a deployment field of {model.key}'s policy block; a deployment changes "
                             f"{', '.join(sorted(set(model.policy) & set(DEPLOYMENT_FIELDS)))}")
        policy[key] = value
    return policy


def compose(experiment: Mapping[str, Any], model: ModelEntry, policy: Mapping[str, Any],
            baseline_policy: Mapping[str, Any], record_dataset: bool = False,
            execution: PolicyExecution | None = None, scoring: Mapping[str, Any] | None = None,
            submission: bool = False) -> dict[str, Any]:
    """``experiment`` (a loaded cell or sweep file) with each Run's baseline policy replaced by ``policy`` and
    the model's environment fields over the Run's; refuses a Run that does not hold the baseline policy.
    ``execution`` (the task's ``spec.policy_execution``) puts its policy rate over the model's block and its control
    rate over the Run's environment; a trial (a planner choosing its own re-plan points, no horizon) keeps the
    task's own control rate, asked every step. ``scoring`` (the task's ``spec.scoring_environment``) goes over every
    Run's environment, a trial's too: how its episodes are judged, which the cell files leave to the run.
    ``record_dataset`` sets the Runs' ``record_dataset``: the per-step state HDF5 (no camera frames), which a
    typed experiment takes from its config (experiment_runner.py has no such flag). It is one of
    evaluation_signature.RECORDING_FIELDS, outside the cell's declaration, so it scores the same.
    ``submission`` also enables upstream trajectory terms, including episode identity, without
    changing the ordinary recording layout or policy execution."""
    if submission and not record_dataset:
        raise ValueError("a submission requires record_dataset")
    rated = execution is not None and model.horizon is not None
    rate_policy = execution.policy_fields() if rated else {}
    rate_environment = execution.environment_fields() if rated else {}
    composed = copy.deepcopy(plain(experiment))
    for name, run in composed["runs"].items():
        if run.get("policy") != plain(baseline_policy):
            raise ValueError(f"Run {name!r} holds {run.get('policy')!r}, not the baseline policy a cell file declares")
        run["policy"] = {**policy, **rate_policy}
        run["environment"] = {**run.get("environment", {}), **plain(MODEL_ENVIRONMENT), **rate_environment,
                              **plain(scoring or {})}
        if record_dataset:
            run["environment"]["record_dataset"] = True
        if submission:
            run.setdefault("environment_builder", {})["record_trajectories"] = True
    return composed


def _composed_text(composed: Mapping[str, Any], source: str, model: ModelEntry) -> str:
    header = (f"# {source} for {model.label} ({model.key}), composed by scripts/evaluation/run_benchmark_cell.py\n"
              "# (dexbench_arena.benchmark.model_run); run_manifest.json beside it says what ran. Not checked in.\n")
    return header + yaml.safe_dump(composed, sort_keys=False, width=120, allow_unicode=True)


def composed_problems(composed_path: Path, declared: Mapping[str, Any], rows: Mapping[str, Any],
                      model: ModelEntry, policy_execution: Mapping[str, Any] | None = None) -> list[str]:
    """What keeps the composed Runs from building what the cell declares for ``model``
    (``evaluation_signature.declaration_problems`` on each Run as Arena's loader builds it, with the manifest's
    ``policy_execution``)."""
    experiment = load_arena_experiment_from_config_file(str(composed_path), device="cpu")
    policy_fields = ("policy_variant", "action_chunk_length", "policy_hz", "policy_action_interpolation")
    return [f"{name}: builds {problem}"
            for name, run in experiment.runs.items()
            for problem in declaration_problems(
                environment_cfg_snapshot(run.environment), {**declared, **rows.get(name, {}).get("environment", {})},
                {field: getattr(run.policy, field, None) for field in policy_fields}, model.key, model.horizon,
                policy_execution)]


def environment_problems(environ: Mapping[str, str] | None = None) -> None:
    """Refuse the render variables (``RENDER_VARIABLES``) that would change a cell's camera images without the
    episode rows saying so, and the launcher's teleop hooks (``launcher_hook_variables``), which no cell declares.
    ``DEXBENCH_CONTROL_HZ`` needs no check: every cell file states its rate."""
    environ = os.environ if environ is None else environ
    problems = [f"{name}={environ[name]!r} changes the camera images and no cell declares it" for name in RENDER_VARIABLES
                if environ.get(name) is not None]
    problems += [f"{name}={environ[name]!r} is a teleop hook of the launcher, not a cell's setting"
                 for name in launcher_hook_variables(environ)]
    if problems:
        raise ValueError("unset these for a benchmark run: " + "; ".join(problems))


def prepare(task: str, cell: str | None, model_key: str, output: str | Path, *, manifest_path: str | Path,
            configs: str | Path, deployment: Mapping[str, Any] | None = None, checkpoint: str | None = None,
            record_dataset: bool = True, record_camera_video: bool = True,
            shard: tuple[int, int] | None = None, slices: Mapping[str, Sequence[int]] | None = None,
            version: str = DEFAULT_EVAL_VERSION, argv: Sequence[str] = (),
            model_entry: Mapping[str, Any] | None = None, submission: bool = False,
            checkpoint_sha256: str | None = None) -> dict[str, Any]:
    """Check and write a model's run of ``task``'s ``cell`` (None: its sweep) into ``output``; returns the run manifest."""
    if submission and (not record_dataset or not record_camera_video or not checkpoint or cell is None):
        raise ValueError("a submission requires a cell, checkpoint label, dataset and camera recording")
    model = resolve_model(model_key if model_entry is None else model_entry)
    if model.key != model_key:
        raise ValueError(f"--model {model_key!r} and its entry's key {model.key!r} differ")
    if checkpoint_sha256 is not None and (len(checkpoint_sha256) != 64 or any(c not in "0123456789abcdef" for c in checkpoint_sha256)):
        raise ValueError("checkpoint_sha256 must be a lowercase SHA-256 digest")
    if submission and model_entry is not None and checkpoint_sha256 is None:
        raise ValueError("a config-model submission requires --checkpoint-sha256")
    if not model.runs(task, cell):
        raise ValueError(f"{model.key} runs {dict(model.cells or {})} only, not {task} {cell or 'sweep'}")
    source_code = code_revision(refresh=submission)
    if submission and source_code.get("dirty") is not False:
        raise ValueError("a submission requires a recorded clean evaluator checkout")
    policy = model_policy(model, deployment, task)
    manifest_bytes = Path(manifest_path).read_bytes()
    manifest = json.loads(manifest_bytes)
    entry = manifest["tasks"][task]
    section = "headline" if cell is not None else "sensitivity"
    if cell is not None and cell not in entry["cells"]:
        raise ValueError(f"{task} has no cell {cell!r}; its cells are {sorted(entry['cells'])}")
    cell_entry = entry["cells"][cell] if cell is not None else entry["sensitivity"]
    environment_problems()
    source = cell_entry["eval"]
    cell_bytes = (Path(configs) / source).read_bytes()
    frozen_file = pinned_path(task, section)
    if not frozen_file.is_file():
        raise ValueError(f"{frozen_file}: no pinned frozen set; a model runs only frozen rows "
                         f"(score_benchmark_cells.py freeze --task {task} --section {section}, then commit the candidate)")
    frozen_sha256, _rows = load_frozen_set(frozen_file, manifest, task, section=section)
    contracts = entry.get("asset_major_contracts", {})
    source_paths = [row["path"] for row in cell_entry["rows"].values()] if cell is not None else cell_entry["paths"]
    for path in source_paths:
        rows = _rows[path]
        for index, row in enumerate(rows):
            task_object = recorded(row, "task_object") or {}
            required_major = contracts.get(task_object.get("axis"), {}).get(task_object.get("key"))
            contract = entry.get("asset_version_contracts", {}).get(task_object.get("axis"), {}).get(task_object.get("key"))
            versions = {"frozen source": task_object.get("asset_version")}
            if contract is not None:
                versions["manifest sidecar"] = contract["version"]
            problem = asset_major_problem(required_major, versions,
                                          rename_only=contract["rename_only"] if contract else ())
            if problem:
                raise ValueError(f"{path}:{index}: {problem}; record/filter/freeze a compatible epoch before launching")
    selection = _selection(cell_entry, cell, shard, model.rows, version, slices,
                           partial(pinned_version_rows, manifest, task, cell))

    output = Path(output)
    if (output / ARENA_OUTPUT).exists():
        raise FileExistsError(f"{output / ARENA_OUTPUT} exists: a run's directory holds one run")
    spec = TASKS_BY_NAME[task]
    composed = compose(yaml.safe_load(cell_bytes), model, policy, spec.baseline_policy, record_dataset,
                       spec.policy_execution, spec.scoring_environment, submission=submission)
    if selection is not None:
        composed["runs"] = {name: run for name, run in composed["runs"].items() if name in selection}
    output.mkdir(parents=True, exist_ok=True)
    row_files = []
    for name, rows in (selection or {}).items():
        if rows == tuple(range(cell_entry["rows"][name]["rows"])):
            continue
        target = output / "rows" / f"{name}.jsonl"
        row_files.append((cell_entry["rows"][name]["path"], rows, target))
        composed["runs"][name].setdefault("environment_builder", {})["recorded_variation_samples_path"] = str(target)
    composed_text = _composed_text(redact_api_tokens(composed), source, model)
    with tempfile.TemporaryDirectory(dir=output) as scratch:
        candidate = Path(scratch) / COMPOSED
        candidate.write_text(composed_text)
        problems = composed_problems(candidate, cell_entry["declared"] if cell is not None else {},
                                     cell_entry["rows"] if cell is not None else {}, model, entry.get("policy_execution"))
    if problems:
        raise ValueError(f"{task} {cell or 'sweep'} {model.key}: the composed Runs depart from the declaration:\n  - "
                         + "\n  - ".join(problems[:20]))
    run_manifest = {
        "version": RUN_MANIFEST_VERSION,
        "task": task,
        "section": section,
        "cell": cell,
        "model": model.key,
        "label": model.label,
        "group": model.group,
        "model_entry": redact_api_tokens(model_record(model)),
        "policy": redact_api_tokens(policy),
        "environment": plain(MODEL_ENVIRONMENT),
        "horizon": policy.get("action_chunk_length"),
        "deployment": redact_api_tokens(plain(dict(deployment or {}))),
        "checkpoint": checkpoint,
        "record_dataset": record_dataset,
        "record_camera_video": record_camera_video,
        **({"submission": True} if submission else {}),
        "eval_version": None if model.rows is not None or cell is None else version,
        "row_indices": None if selection is None else {name: list(rows) for name, rows in selection.items()},
        "rows": model.rows,
        "code": source_code,
        "evaluation_contract_sha256": evaluation_contract_sha256(manifest, task),
        "checkpoint_sha256": checkpoint_sha256,
        "cell_config": source,
        "cell_config_sha256": sha256_bytes(cell_bytes),
        "composed_sha256": sha256_bytes(composed_text.encode()),
        "benchmark_manifest_sha256": sha256_bytes(manifest_bytes),
        "frozen_set": frozen_file.name,
        "frozen_sha256": frozen_sha256,
        "arena_output": ARENA_OUTPUT,
        "argv": redact_api_tokens(list(argv)),
        "prepared_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    manifest_path_out = output / RUN_MANIFEST
    prepared = json.loads(manifest_path_out.read_text()) if manifest_path_out.exists() else None
    if prepared is not None and not _same_run(prepared, run_manifest):
        raise FileExistsError(f"{manifest_path_out} exists with other content: {output} holds another run (another "
                              "model, deployment, version, shard or slices); a run's directory holds one run")
    for source_path, rows, target in row_files:
        write_rows(source_path, rows, target)
    if prepared is not None:
        publish_new(output / COMPOSED, composed_text.encode())
        replace_atomic(manifest_path_out, (json.dumps(run_manifest, indent=1) + "\n").encode())
        return run_manifest
    publish_new(output / COMPOSED, composed_text.encode())
    publish_new(manifest_path_out, (json.dumps(run_manifest, indent=1) + "\n").encode())
    return run_manifest


PREPARED_FIELDS = ("argv", "prepared_at")
"""Run-manifest fields that say how and when a run was prepared, not what it runs (:func:`prepare`)."""


def _same_run(prepared: Mapping[str, Any], run_manifest: Mapping[str, Any]) -> bool:
    return ({key: value for key, value in prepared.items() if key not in PREPARED_FIELDS}
            == {key: value for key, value in run_manifest.items() if key not in PREPARED_FIELDS})


def parse_slices(text: str) -> dict[str, Any]:
    """``--slices``' JSON, which must be an object ``{run: [start, end], ...}``: ``null`` would read as "no slices"
    and run the whole cell, and a list has no Run names."""
    slices = json.loads(text)
    if not isinstance(slices, dict):
        raise ValueError(f'slices are a JSON object {{"<run>": [start, end], ...}}, not {text!r}')
    return slices


def _selection(cell_entry: Mapping[str, Any], cell: str | None, shard: tuple[int, int] | None, pilot_rows: int | None,
               version: str, explicit: Mapping[str, Sequence[int]] | None = None,
               pinned: Callable[[str], Mapping[str, Sequence[int]]] | None = None) -> dict[str, tuple[int, ...]] | None:
    """``{run: frozen row indices}`` a run of a cell replays (the version's rows as ``pinned(version)`` lists them, a pilot's
    subset, a shard or ``explicit`` slices of them), or None for the whole cell (``v1-plus``, or the sweep)."""
    if cell is None:
        if shard is not None or explicit is not None:
            raise ValueError("a sweep is not split: its rows are per build (recorded_variation_samples_paths)")
        if pilot_rows is not None:
            raise ValueError("a pilot runs cells, not the sweep")
        return None
    quota = {name: row["rows"] for name, row in cell_entry["rows"].items()}
    if pilot_rows is not None:
        if shard is not None:
            raise ValueError("a pilot's subset is not split further into shards")
        if explicit is not None:
            raise ValueError("explicit slices are not combined with a shard or a pilot's subset")
        return cell_rows(quota, pilot_rows, version, pinned)
    rows = cell_rows(quota, None, version, pinned)
    if explicit is not None:
        if shard is not None:
            raise ValueError("explicit slices are not combined with a shard or a pilot's subset")
        if not isinstance(explicit, Mapping):
            raise ValueError(f"explicit slices map a Run to [start, end], not {explicit!r}")
        if not explicit:
            raise ValueError("explicit slices name at least one Run")
        chosen = {}
        for name, bound in explicit.items():
            if name not in rows:
                raise ValueError(f"slice of Run {name!r}: not a Run of the cell at {version} ({len(rows)} Runs)")
            if (not isinstance(bound, Sequence) or len(bound) != 2
                    or not all(isinstance(index, int) and not isinstance(index, bool) for index in bound)):
                raise ValueError(f"slice of Run {name!r}: {bound!r} is not [start, end] of two whole row indices")
            start, end = bound
            if not 0 <= start < end <= len(rows[name]):
                raise ValueError(f"slice {name} {start}:{end}: needs 0 <= start < end <= {len(rows[name])} "
                                 f"(the Run's rows at {version})")
            chosen[name] = rows[name][start:end]
        return chosen
    if shard is not None:
        index, parts = shard
        if not 0 <= index < parts or parts > min(len(kept) for kept in rows.values()):
            raise ValueError(f"shard {index}/{parts}: needs 0 <= i < n <= {min(len(kept) for kept in rows.values())} "
                             "(the smallest Run's rows)")
        return {name: kept[len(kept) * index // parts: len(kept) * (index + 1) // parts] for name, kept in rows.items()}
    return None if rows == {name: tuple(range(count)) for name, count in quota.items()} else rows


__all__ = [
    "DEPLOYMENT_FIELDS",
    "compose",
    "composed_problems",
    "model_policy",
    "parse_slices",
    "prepare",
]
