#!/usr/bin/env python3
# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Score a task's benchmark cells, and freeze the rows they replay (docs/Eval_Protocol.md §4).

``freeze`` writes a candidate frozen set (the SHA-256 and row count of every row file the manifest names
for a task) and says whether it matches the set checked in under ``benchmark/frozen/``; a pin changes
only by committing the candidate there. For the headline set it also writes the rows v1 replays of each row file
(``frozen.version_pin``), committed with the set as ``benchmark/frozen/versions_db_<task>.json``.
``score`` reads each model run's directory (``run_benchmark_cell.py --output``) and reports, per model and
cell, successes k of n with the 95% equal-tailed interval of Beta(k+1, n-k+1) (a uniform Beta(1, 1) prior)
and the macro-average. It refuses missing or incorrect frozen rows, unknown settings and evaluation
signatures that differ among the headline models; known departures from the cell's approved settings are
scored and marked non-headline. An ablation or a trial (the manifest model's ``group``) is scored the same way and
reported apart; a trial (a tool-calling planner) is held to no chunk horizon. So is a model an eval config registered
instead of ``models.py`` (its run manifests' ``model_entry``), under ``community``,
with the entry in the report. An eval launch's directories come with what
it accepted of each (``--accepted``): one a simulator crash ended counts only its episodes the launch accepted, and each
cell reports how many episodes came from such attempts and how many rows needed more than one. A task that declares a challenge track
(the manifest's ``challenge``, DB-L1) gets it per model and cell beside the headline, from each row's
``challenge_success``, by the same interval convention and never mixed into the headline. Needs no simulator (scipy)::

    /isaac-sim/python.sh scripts/evaluation/score_benchmark_cells.py freeze --task r2
    /isaac-sim/python.sh scripts/evaluation/score_benchmark_cells.py score --task r2 \\
        --results train=outputs/db_r2_eval_train_gr00t train=outputs/db_r2_eval_train_cosmos ga_o=... ga_s=... ga_os=...
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from collections.abc import Collection, Mapping
from functools import partial
from pathlib import Path

from scipy.stats import beta

from dexbench_arena.benchmark import Refusal
from dexbench_arena.benchmark.models import MODELS_BY_KEY, plain
from dexbench_arena.benchmark.protocol import CONTROL_HZ, DEFAULT_EVAL_VERSION, EVAL_VERSIONS, cell_rows
from dexbench_arena.benchmark.frozen import (candidate_path, freeze, load_frozen_set, pinned_path,
                                             pinned_version_rows, version_pin, version_pin_bytes, version_pin_path)
from dexbench_arena.benchmark.run_manifest import read_run_manifest
from dexbench_arena.benchmark.reference_settings import check_reference, load_reference
from dexbench_arena.benchmark.leaderboard_models import with_registered_models
from dexbench_arena.evaluation.artifacts import publish_new, recorded
from dexbench_arena.tasks.object_axis import asset_major_problem
from dexbench_arena.evaluation.evaluation_signature import (
    current_environment_cfg,
    declaration_problems,
    evaluation_signature,
    same_value,
    settings_problems,
)

CONFIGS = Path(__file__).resolve().parent / "experiment_configs" / "benchmark"
RESULT_FILE = "arena_experiment_result.json"
CONFIDENCE = 0.95
PRIOR = (1.0, 1.0)
"""The uniform Beta(1, 1) prior (RoboLab's rule), not Jeffreys' Beta(1/2, 1/2): the posterior is Beta(k+1, n-k+1)."""
INTERVAL = {
    "prior": "Beta(1, 1), uniform",
    "posterior": "Beta(k+1, n-k+1)",
    "interval": "equal-tailed 95%: the 2.5% and 97.5% posterior quantiles (not highest-density)",
    "pooling": "k and n summed over the cell's objects (each object's frozen quota); no per-object interval is averaged",
    "macro": "unweighted mean of the cells' success rates",
}


def beta_interval(successes: int, trials: int, confidence: float = CONFIDENCE) -> tuple[float, float]:
    """Equal-tailed interval of the posterior Beta(k + 1, n - k + 1) under the uniform prior :data:`PRIOR`."""
    tail = (1.0 - confidence) / 2.0
    posterior = beta(successes + PRIOR[0], trials - successes + PRIOR[1])
    return float(posterior.ppf(tail)), float(posterior.ppf(1.0 - tail))


NON_HEADLINE_SECTIONS = {"ablation": "ablations", "trial": "trials", "community": "community"}
"""Where the report puts each non-headline group (``ModelEntry.group``): apart from the headline, never in it."""


def with_config_models(manifest: dict, task: str, results: list[tuple[str, Path]]) -> dict:
    """``manifest`` with every model an eval config registered (``models.config_model``) among ``results`` added to
    ``task``'s models, as its run manifests record it (``model_entry`` with ``registration: config``): its label,
    group and policy block, and the whole entry, which the report keeps. A config model reuses no registered key, and
    every directory of it records the same entry; anything else is refused. A run of a model neither the manifest nor
    a config entry names is left for ``collect`` to refuse."""
    models = manifest["tasks"][task]["models"]
    found: dict[str, tuple[dict, Path]] = {}
    for _cell, directory in results:
        run_manifest = read_run_manifest(directory)
        model, entry = run_manifest.get("model"), run_manifest.get("model_entry") or {}
        if entry.get("registration") != "config":
            continue
        if model in MODELS_BY_KEY or model in models:
            raise Refusal(f"{directory}: a config model named {model!r}, a key of benchmark/models.py; it is never "
                          "scored as the registered model")
        if entry.get("group") == "outside":
            raise Refusal(f"{directory}: its model_entry names group outside, the name the community group had "
                          "before it was renamed; record the run again")
        if entry.get("key") != model or entry.get("group") != "community":
            raise Refusal(f"{directory}: its model_entry is not {model!r}'s or names a group other than community, the "
                          "one a config model takes")
        first = found.setdefault(model, (entry, directory))
        if first[0] != entry:
            raise Refusal(f"{directory}: {model}'s entry differs from {first[1]}'s; one model's score is one entry")
    for _cell, directory in results:
        run_manifest = read_run_manifest(directory)
        model = run_manifest.get("model")
        if model in found and run_manifest.get("model_entry") != found[model][0]:
            raise Refusal(f"{directory}: a run of {model!r} that does not record {found[model][1]}'s config entry; "
                          "one model's score is the runs of one entry")
    if not found:
        return manifest
    manifest = json.loads(json.dumps(manifest))
    for model, (entry, directory) in found.items():
        missing = [field for field in ("label", "policy") if field not in entry]
        if missing:
            raise Refusal(f"{directory}: {model}'s model_entry lacks {missing}")
    manifest["tasks"][task]["models"].update({model: {
        "label": entry["label"], "available": True, "group": entry["group"], "policy": entry["policy"],
        "registration": "config", "entry": entry} for model, (entry, _) in found.items()})
    return manifest


def model_group(declared: dict) -> str:
    """A manifest model's group; a manifest without ``group`` says ``ablation`` or nothing."""
    return declared.get("group") or ("ablation" if declared.get("ablation") else "headline")


def _accepted_runs(directory: Path, run_manifest: dict, listed: Mapping[str, list[int]]) -> dict[str, dict]:
    """A directory's Runs as the experiment result holds them, read from each Run's episode result files (Arena appends
    one line as each episode ends) and kept to the episodes ``listed`` (``{run: [replay_source_record_index, ...]}``):
    an attempt a simulator crash ended has no experiment result, and its episodes past the ones the launch accepted are
    not scored. Each listed episode must be there exactly once; a line the crash tore is no episode."""
    arena = directory / run_manifest["arena_output"]
    runs = {}
    for run_name, indices in listed.items():
        records = []
        for path in sorted((arena / run_name).glob("episode_results*.jsonl")):
            for line in path.read_text().splitlines():
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if isinstance(record, dict):
                    records.append(record)
        counts = Counter(record.get("replay_source_record_index") for record in records)
        for index in indices:
            if counts[index] != 1:
                raise Refusal(f"{directory}: Run {run_name!r} holds {counts[index]} episodes replaying {index}, which "
                              f"the launch accepted from it once")
        runs[run_name] = {"policy_variant": (run_manifest.get("policy") or {}).get("policy_variant"), "status": "completed",
                          "rebuilds": [{"index": 0, "episodes": [record for record in records
                                                                 if record.get("replay_source_record_index") in set(indices)]}]}
    return runs


def collect(manifest: dict, task: str, results: list[tuple[str, Path]],
            frozen_sha256: str | None = None, accepted: Mapping[str, dict] | None = None) -> dict[str, dict[str, list[dict]]]:
    """``{cell: {model key: [episode, ...]}}`` from every ``(cell, model run directory)``."""
    accepted = {str(Path(key)): value for key, value in (accepted or {}).items()}
    entry = manifest["tasks"][task]
    episodes: dict[str, dict[str, list[dict]]] = {}
    seen: dict[tuple[str, str, str], list[tuple[set[int], Path]]] = {}
    checkpoints: dict[str, tuple[str | None, Path]] = {}
    for cell, directory in results:
        if cell not in entry["cells"]:
            raise Refusal(f"{task} has no cell {cell!r}; its cells are {sorted(entry['cells'])}")
        run_manifest = read_run_manifest(directory)
        made_for = (run_manifest.get("task"), run_manifest.get("section"), run_manifest.get("cell"))
        if made_for != (task, "headline", cell):
            raise Refusal(f"{directory}: a run of {made_for}, given for {task} {cell}")
        model = run_manifest.get("model")
        if model not in entry["models"]:
            raise Refusal(f"{directory}: model {model!r} is not one the benchmark requires ({sorted(entry['models'])}), "
                          "and its run manifest records no config model entry (with_config_models)")
        if frozen_sha256 is not None and run_manifest.get("frozen_sha256") != frozen_sha256:
            raise Refusal(f"{directory}: ran against frozen set {str(run_manifest.get('frozen_sha256'))[:12]}, "
                             f"scoring {frozen_sha256[:12]}")
        if run_manifest.get("checkpoint") is None:
            raise Refusal(f"{directory}: {model} records no checkpoint (run_benchmark_cell.py --checkpoint), so its "
                             f"score is no one checkpoint's")
        checkpoint, first = checkpoints.setdefault(model, (run_manifest.get("checkpoint"), directory))
        if run_manifest.get("checkpoint") != checkpoint:
            raise Refusal(f"{directory}: {model} ran checkpoint {run_manifest.get('checkpoint')!r}, {first} ran "
                             f"{checkpoint!r}; one model's score is one checkpoint (score another as its own model entry)")
        if run_manifest.get("rows") != entry["models"][model].get("rows"):
            raise Refusal(f"{directory}: made for {run_manifest.get('rows') or 'every'} rows of the cell, {model} "
                             f"declares {entry['models'][model].get('rows') or 'every row'}")
        provenance = {key: run_manifest.get(key) for key in ("checkpoint", "deployment", "code", "cell_config_sha256")}
        rows = entry["cells"][cell]["rows"]
        launch = accepted.get(str(Path(directory)))
        listed = None if launch is None else launch.get("rows")
        if listed is not None:
            result = {"runs": _accepted_runs(directory, run_manifest, listed)}
        else:
            result_path = directory / run_manifest["arena_output"] / RESULT_FILE
            if not result_path.is_file():
                raise Refusal(f"{directory}: no {run_manifest['arena_output']}/{RESULT_FILE}, so the cell has not run "
                                 f"(or has not finished) in this directory")
            result = json.loads(result_path.read_text())
        for run_name, run in result["runs"].items():
            if run_name not in rows:
                raise Refusal(f"{directory}: Run {run_name!r} is not a row of {task} {cell}")
            replayed = run_rows(run_manifest, run_name, rows[run_name]["rows"])
            scored = set(replayed) if listed is None else {replayed[index] for index in listed[run_name]
                                                           if 0 <= index < len(replayed)}
            for other_rows, other in seen.setdefault((cell, model, run_name), []):
                if other_rows & scored:
                    raise Refusal(f"{task} {cell}: {model}'s Run {run_name!r} rows {sorted(other_rows & scored)[:5]} "
                                  f"are in both {other} and {directory}")
            seen[cell, model, run_name].append((scored, directory))
            retried = set(((launch or {}).get("retried") or {}).get(run_name) or ())
            if run["policy_variant"] != model:
                raise Refusal(f"{run_name}: recorded policy_variant {run['policy_variant']!r}, its run manifest names {model!r}")
            if run["status"] != "completed":
                raise Refusal(f"{run_name}: status {run['status']}")
            for episode in (episode for rebuild in run["rebuilds"] for episode in rebuild["episodes"]):
                index = episode.get("replay_source_record_index")
                if isinstance(index, int):
                    if not 0 <= index < len(replayed):
                        raise Refusal(f"{directory}: Run {run_name!r} replayed index {index}, outside its {len(replayed)} rows")
                    attempt = None if launch is None else {"salvaged": bool(launch.get("salvaged")), "retried": index in retried}
                    episode = {**episode, "replay_source_record_index": replayed[index],
                               **({} if attempt is None else {"_attempt": attempt})}
                episodes.setdefault(cell, {}).setdefault(model, []).append({**episode, "_row": run_name, "_run": provenance})
    return episodes


def run_rows(run_manifest: dict, run_name: str, count: int) -> tuple[int, ...]:
    """The frozen rows of a Run of ``count`` rows that a run replayed, in the order it replayed them."""
    if (run_manifest.get("row_indices") or {}).get(run_name) is not None:
        return tuple(run_manifest["row_indices"][run_name])
    start, end = (run_manifest.get("slices") or {}).get(run_name, (0, count))
    return tuple(range(start, end))


def _other_version_row(episode: dict, wanted: dict[str, tuple[int, ...]], quota: dict[str, int]) -> bool:
    """Whether ``episode`` replayed a frozen row of its Run that the version scored leaves out. An episode without a
    valid index is not one: ``_row_draws`` refuses it."""
    index = episode.get("replay_source_record_index")
    return (isinstance(index, int) and 0 <= index < quota.get(episode["_row"], 0)
            and index not in wanted.get(episode["_row"], ()))


def _same_draws(replayed, source) -> bool:
    """Whether two ``variations`` mappings hold the same keys and, per key, the same value (``same_value``)."""
    replayed = {} if replayed is None else replayed
    source = {} if source is None else source
    if not isinstance(replayed, dict) or not isinstance(source, dict) or set(replayed) != set(source):
        return False
    return all(same_value(replayed[key], value) for key, value in source.items())


def _source_object(record: dict) -> tuple | None:
    task_object = recorded(record, "task_object")
    return (task_object.get("axis"), task_object.get("key")) if isinstance(task_object, dict) else None


def episode_problems(episode: dict, declared: dict, backend: str, model: str, horizon: int | None,
                     legacy_values: dict | None = None, other_task_fields: Collection[str] = (),
                     policy_execution: dict | None = None, scoring_environment: dict | None = None,
                     block: dict | None = None) -> list[str]:
    """How ``episode`` departs from the approved settings its cell declares for ``model``: any field of the
    environment config but the other tasks' (``other_task_fields``; ``evaluation_signature.declaration_problems``,
    with the model and horizon as the policy stamped them; a trial's horizon is None), the cap the env ran to,
    the physics backend; the policy rate the task's cells ask for is the manifest's ``policy_execution``.
    Raises ValueError for an episode whose settings are unknown, or that ran a field of the manifest's
    ``scoring_environment`` (DB-L1's ``l1_headline_bags``) at another value: that field decides what its ``success``
    means, so a deviation there would count another rule's verdict toward the headline k. ``block`` is the model's
    policy block when the registry does not hold it (a config model's, from its run manifests).
    """
    environment_cfg, settings = recorded(episode, "environment_cfg"), recorded(episode, "evaluation_settings")
    if not isinstance(environment_cfg, dict):
        raise Refusal("records no environment_cfg, so its settings are unknown")
    environment_cfg = current_environment_cfg(environment_cfg, legacy_values, other_task_fields)
    for field, value in (scoring_environment or {}).items():
        if not same_value(environment_cfg.get(field), value):
            raise Refusal(f"ran {field}={environment_cfg.get(field)!r}, and its cell judges success at {value!r} "
                          f"(scoring_environment), so its success is another rule's; run the cell through "
                          f"run_benchmark_cell.py, which sets it, or rescore the episode")
    unknown = settings_problems(settings)
    if unknown:
        raise Refusal("; ".join(unknown))
    policy = settings.get("policy") or {}
    problems = [f"ran {problem}" for problem in declaration_problems(environment_cfg, declared, policy, model, horizon,
                                                                     policy_execution)]
    control = settings.get("control") or {}
    rate_environment = (policy_execution or {}).get("environment") or {} if horizon is None else {}
    approved_hz = float((None if "rby1_control_hz" in rate_environment else declared.get("rby1_control_hz")) or CONTROL_HZ)
    if not (control.get("sim_dt") and control.get("decimation")
            and abs(1.0 / (control["sim_dt"] * control["decimation"]) - approved_hz) <= 1e-3):
        problems.append(f"ran at control {control}, its cell runs at {approved_hz:g} Hz")
    if not same_value(settings.get("episode_length_s"), declared.get("episode_length_s")):
        problems.append(f"ran to a {settings.get('episode_length_s')} s cap, its cell declares {declared.get('episode_length_s')} s")
    block = block if block is not None else MODELS_BY_KEY[model].policy if model in MODELS_BY_KEY else None
    expected = plain((block or {}).get("relative_joint_slices") or [])
    if plain(policy.get("relative_joint_slices") or []) != expected:
        problems.append(f"ran relative_joint_slices={policy.get('relative_joint_slices')!r}, model {model} declares {expected!r}")
    if "relative_action_groups" in (block or {}):
        if "relative_action_groups" not in policy:
            raise Refusal(f"records no served relative_action_groups, so it may be a {model} run from before "
                             f"{model}'s action representation was stamped")
        if sorted(policy["relative_action_groups"]) != sorted(block["relative_action_groups"]):
            problems.append(f"ran relative_action_groups={policy['relative_action_groups']!r}, model {model} declares "
                            f"{list(block['relative_action_groups'])!r}")
    if "denoising_timesteps" in (block or {}):
        if policy.get("denoising_timesteps") is None:
            raise Refusal(f"records no denoising_timesteps, so it may be a {model} run served at another step count")
        if plain(policy["denoising_timesteps"]) != plain(block["denoising_timesteps"]):
            problems.append(f"ran denoising_timesteps={policy['denoising_timesteps']!r}, model {model} declares "
                            f"{list(block['denoising_timesteps'])!r}")
    physics = str((settings.get("physics") or {}).get("type", ""))
    if not physics.endswith(backend):
        problems.append(f"ran on {physics or 'an unnamed backend'}, its cell declares {backend}")
    return problems


def _row_draws(model: str, cell: str, task: str, rows: list[dict], expected_indices: dict[str, tuple[int, ...]],
               sources: dict[str, list[dict]], declared: dict, horizon: int | None, backend: str, deviations: set,
               legacy_values: dict | None = None, other_task_fields: Collection[str] = (),
               policy_execution: dict | None = None, scoring_environment: dict | None = None,
               asset_major_contracts: dict | None = None, asset_version_contracts: dict | None = None,
               block: dict | None = None) -> dict:
    """``{(row, source index): draws}``; refuses a missing, duplicate or extra source row, or an unscored episode."""
    seen: dict[tuple[str, int], str] = {}
    for episode in rows:
        index = episode.get("replay_source_record_index")
        if not isinstance(index, int):
            raise Refusal(f"{task} {cell} {model}: an episode of {episode['_row']} records no replay_source_record_index")
        if episode.get("success") is None:
            raise Refusal(f"{task} {cell} {model}: row {episode['_row']}:{index} has no success value")
        key = (episode["_row"], index)
        if key in seen:
            raise Refusal(f"{task} {cell} {model}: row {episode['_row']}:{index} ran twice")
        try:
            problems = episode_problems(episode, declared[episode["_row"]], backend, model, horizon, legacy_values,
                                        other_task_fields, policy_execution, scoring_environment, block)
        except Refusal as error:
            raise Refusal(f"{task} {cell} {model}: row {episode['_row']}:{index} {error}") from None
        deviations.update(problems)
        source_rows = sources[episode["_row"]]
        if not 0 <= index < len(source_rows):
            raise Refusal(f"{task} {cell} {model}: row {episode['_row']}:{index} is not in its frozen file")
        source = source_rows[index]
        if "placement" in episode or "placement" in source:
            raise Refusal(f"{task} {cell} {model}: row {episode['_row']}:{index} carries a relation placement, "
                             "which the benchmark does not admit")
        source_fit = (recorded(source, "task_object") or {}).get("fit")
        if source_fit is not None and not _same_draws((recorded(episode, "task_object") or {}).get("fit"), source_fit):
            raise Refusal(f"{task} {cell} {model}: row {episode['_row']}:{index} ran another object fit "
                             "than its frozen row records")
        if _source_object(episode) != _source_object(source):
            raise Refusal(f"{task} {cell} {model}: row {episode['_row']}:{index} ran object "
                             f"{_source_object(episode)}, its frozen row is {_source_object(source)}")
        source_object = recorded(source, "task_object") or {}
        required_major = (asset_major_contracts or {}).get(source_object.get("axis"), {}).get(source_object.get("key"))
        contract = (asset_version_contracts or {}).get(source_object.get("axis"), {}).get(source_object.get("key"))
        versions = {"frozen source": source_object.get("asset_version"),
                    "scored episode": (recorded(episode, "task_object") or {}).get("asset_version")}
        if contract is not None:
            versions["manifest sidecar"] = contract["version"]
        major_problem = asset_major_problem(required_major, versions,
                                            rename_only=contract["rename_only"] if contract else ())
        if major_problem:
            raise Refusal(f"{task} {cell} {model}: row {episode['_row']}:{index} {major_problem}")
        if not _same_draws(episode.get("variations"), source.get("variations")):
            raise Refusal(f"{task} {cell} {model}: row {episode['_row']}:{index} did not replay its frozen row's draws")
        signature = evaluation_signature(episode, legacy_values, with_success_criteria=False,
                                         other_task_fields=other_task_fields)
        with_criteria = evaluation_signature(episode, legacy_values, other_task_fields=other_task_fields)
        seen[key] = (json.dumps(episode.get("variations") or {}, sort_keys=True), signature,
                     with_criteria if with_criteria != signature else None, json.dumps(_source_object(episode)))
    expected = {(row, index) for row, indices in expected_indices.items() for index in indices}
    if set(seen) != expected:
        missing, extra = sorted(expected - set(seen)), sorted(set(seen) - expected)
        raise Refusal(f"{task} {cell} {model}: {len(seen)} episodes, the cell has {len(expected)} rows "
                         f"(missing {missing[:5]}, extra {extra[:5]})")
    return seen


def _rows_and_settings(draws: dict, part: int) -> str:
    """One model's ``_row_draws`` as text: its rows' draws (``part`` 0) or their evaluation signatures (1)."""
    return json.dumps(sorted((key, value[part]) for key, value in draws.items()))


def _success_criteria_differ(draws: dict[str, dict]) -> bool:
    """Whether the models' ``_row_draws`` record two different success criteria anywhere in one Run, for one object
    under one signature: any two rows that record them, of one model or of two, at one index or at two (shards with and
    without them may interleave). A row without them (evaluation_signature.SUCCESS_CRITERIA_KEY) agrees with any."""
    by_group: dict[tuple[str, str, str], set] = {}
    for rows in draws.values():
        for (run, _index), (_draws, signature, criteria, task_object) in rows.items():
            if criteria is not None:
                by_group.setdefault((run, task_object, signature), set()).add(criteria)
    return any(len(criteria) > 1 for criteria in by_group.values())


def _summarise(data: dict, cells) -> None:
    """A model's completeness and macro-average (the unweighted mean of its cells), in place."""
    data["missing_cells"] = sorted(set(cells) - set(data["cells"]))
    data["complete"] = not data["missing_cells"]
    data["macro_sr"] = (sum(cell["sr"] for cell in data["cells"].values()) / len(data["cells"])
                        if data["complete"] else None)


def score(manifest: dict, task: str, episodes: dict[str, dict[str, list[dict]]],
          sources: dict[str, list[dict]], frozen_sha256: str | None = None, version: str = DEFAULT_EVAL_VERSION) -> dict:
    """Per model: per cell ``{k, n, sr, ci95}`` and the macro-average."""
    entry = manifest["tasks"][task]
    challenge = entry.get("challenge")
    challenge_tracks: dict[str, dict] = {}
    groups = {model: model_group(declared) for model, declared in entry["models"].items()}
    registered = {model for model, group in groups.items() if group == "registered"}
    if registered:
        _, reference_rows = load_reference(manifest, task, frozen_sha256)
        for model in registered:
            selected = {cell: {model: rows[model]} for cell, rows in episodes.items() if model in rows}
            check_reference(manifest, task, model, selected, reference_rows)
    ablation = {model for model, group in groups.items() if group not in {"headline", "registered"}}
    report: dict[str, dict] = {}
    for cell, cell_entry in entry["cells"].items():
        quota = {name: row["rows"] for name, row in cell_entry["rows"].items()}
        by_model = episodes.get(cell, {})
        cell_sources = {name: sources[row["path"]] for name, row in cell_entry["rows"].items()}
        declared = {name: {**cell_entry["declared"], **row["environment"]} for name, row in cell_entry["rows"].items()}
        deviations: dict[str, set] = {model: set() for model in by_model}
        pinned = partial(pinned_version_rows, manifest, task, cell)
        wanted = {model: cell_rows(quota, entry["models"][model].get("rows"), version, pinned) for model in by_model}
        by_model = {model: rows if entry["models"][model].get("rows") else
                    [row for row in rows if not _other_version_row(row, wanted[model], quota)]
                    for model, rows in by_model.items()}
        draws = {model: _row_draws(model, cell, task, rows, wanted[model], cell_sources, declared,
                                   None if groups[model] == "trial" else manifest["execution_horizon"],
                                   cell_entry["backend"], deviations[model], entry.get("legacy_values"),
                                   entry.get("other_task_fields", ()), entry.get("policy_execution"),
                                   entry.get("scoring_environment"), entry.get("asset_major_contracts"),
                                   entry.get("asset_version_contracts"), entry["models"][model].get("policy"))
                 for model, rows in by_model.items()}
        headline = {model: rows for model, rows in draws.items() if model not in ablation}
        if len({_rows_and_settings(rows, 0) for rows in headline.values()}) > 1:
            raise Refusal(f"{task} {cell}: the models did not run the same rows")
        if len({_rows_and_settings(rows, 1) for rows in headline.values()}) > 1 or _success_criteria_differ(headline):
            raise Refusal(f"{task} {cell}: the models ran under different evaluation settings "
                             "(success definition or criteria, cap, visibility, backend, solver or reset settings; evaluation_signature)")
        reference = next(iter(headline.values()), None)
        for model in set(draws) & ablation:
            if reference is not None and not set(draws[model]) <= set(reference):
                deviations[model].add("ran rows the headline models did not")
                continue
            same_rows = None if reference is None else {key: reference[key] for key in draws[model]}
            if reference is not None and _rows_and_settings(draws[model], 0) != _rows_and_settings(same_rows, 0):
                deviations[model].add("did not run the headline models' rows")
            elif reference is not None and (_rows_and_settings(draws[model], 1) != _rows_and_settings(same_rows, 1)
                                            or _success_criteria_differ({**headline, model: draws[model]})):
                deviations[model].add("ran under other evaluation settings than the headline models (evaluation_signature)")
        signatures = {model: sorted({value[1][:12] for value in rows.values()}) for model, rows in draws.items()}
        revisions = {model: sorted({json.dumps((recorded(row, "evaluation_settings") or {}).get("code"), sort_keys=True)
                                    for row in rows}) for model, rows in by_model.items()}
        runs = {model: sorted({json.dumps(row.get("_run"), sort_keys=True) for row in rows}) for model, rows in by_model.items()}
        for model, rows in by_model.items():
            k, n = sum(bool(row["success"]) for row in rows), sum(len(indices) for indices in wanted[model].values())
            low, high = beta_interval(k, n)
            objects: dict[str, dict] = {}
            for row in rows:
                counts = objects.setdefault(str((_source_object(row) or (None, None))[1]), {"k": 0, "n": 0})
                counts["k"] += bool(row["success"])
                counts["n"] += 1
            if challenge is not None:
                unjudged = next((row for row in rows if row.get("challenge_success") is None), None)
                if unjudged is not None:
                    raise Refusal(f"{task} {cell} {model}: row {unjudged['_row']}:{unjudged.get('replay_source_record_index')} "
                                  f"records no challenge_success, so its challenge track ({challenge}) is unknown; an "
                                  f"episode run before the track existed is rescored, not scored as it is")
                ck = sum(bool(row["challenge_success"]) for row in rows)
                challenge_tracks.setdefault(model, {"cells": {}})["cells"][cell] = {
                    "k": ck, "n": n, "sr": ck / n, "ci95": list(beta_interval(ck, n)),
                    "deviations": sorted(deviations[model])[:20]}
            attempts = [row["_attempt"] for row in rows if "_attempt" in row]
            salvage = ({"attempts": {"salvaged_episodes": sum(a["salvaged"] for a in attempts),
                                     "retried_rows": sum(a["retried"] for a in attempts)}} if attempts else {})
            report.setdefault(model, {"cells": {}})["cells"][cell] = {"k": k, "n": n, "sr": k / n, "ci95": [low, high],
                                                                      **salvage,
                                                                      "objects": dict(sorted(objects.items())),
                                                                      "evaluation_signatures": signatures[model],
                                                                      "code": [json.loads(item) for item in revisions[model]],
                                                                      "runs": [json.loads(item) for item in runs[model]],
                                                                      "deviations": sorted(deviations[model])[:20]}
    if not report:
        raise Refusal(f"{task}: no model has a scored episode in any cell; nothing to report")
    apart = {section: {} for section in NON_HEADLINE_SECTIONS.values()}
    for model in sorted(set(report) & ablation):
        data = apart[NON_HEADLINE_SECTIONS[groups[model]]][model] = report.pop(model)
        _summarise(data, entry["models"][model].get("cells") or entry["cells"])
        data["headline"] = False
        if "entry" in entry["models"][model]:
            data["entry"] = entry["models"][model]["entry"]
    for model, declared in entry["models"].items():
        if model in ablation or groups[model] == "registered":
            continue
        data = report.setdefault(model, {"cells": {}})
        data["available"] = declared["available"]
        _summarise(data, entry["cells"])
    for model, data in report.items():
        if groups[model] == "registered":
            data["available"] = entry["models"][model]["available"]
            _summarise(data, entry["cells"])
        data["group"] = groups[model]
        data["rank_eligible"] = data["complete"] and not any(cell["deviations"] for cell in data["cells"].values())
        data["headline"] = data["rank_eligible"] and groups[model] == "headline"
    required = [data for model, data in report.items() if groups[model] == "headline"]
    deviated = any(cell["deviations"] for data in required for cell in data["cells"].values())
    complete = all(data["complete"] for data in required)
    status = "non-headline" if deviated else ("complete" if complete else "partial")
    tracks = {}
    if challenge is not None:
        for model, data in challenge_tracks.items():
            _summarise(data, entry["models"][model].get("cells") or entry["cells"])
            data["group"] = groups[model]
        tracks["challenge"] = {"rule": challenge, "models": challenge_tracks}
    return {"status": status, "version": version, "rows_per_cell": EVAL_VERSIONS[version], "frozen_sha256": frozen_sha256,
            "models": report, **apart, **tracks}


def score_submission(manifest: dict, task: str, model: str, episodes: dict,
                     sources: dict, frozen_sha256: str, version: str = DEFAULT_EVAL_VERSION, *, reference_rows: dict) -> dict:
    """Score one entrant with the existing row and settings checks, without requiring other baselines."""
    scoped = copy.deepcopy(manifest)
    entry = scoped["tasks"][task]
    declared = entry["models"].get(model)
    if (declared is None or model_group(declared) not in {"headline", "community", "registered"} or declared.get("rows") is not None
            or declared.get("cells") is not None):
        raise Refusal(f"{model}: submission requires a full-cell baseline or config model")
    if any(set(models) != {model} for models in episodes.values()):
        raise Refusal("a submission contains results for another model")
    check_reference(manifest, task, model, episodes, reference_rows)
    group = model_group(declared)
    entry["models"] = {model: {**declared, "group": "headline"}}
    report = score(scoped, task, episodes, sources, frozen_sha256, version)
    if report["status"] != "complete" or not report["models"][model]["headline"]:
        raise Refusal(f"{task}: submission is incomplete or has evaluation deviations")
    report["models"][model].update(group=group, rank_eligible=True, headline=group == "headline")
    return report


def format_report(manifest: dict, task: str, report: dict) -> str:
    entry = manifest["tasks"][task]

    def cells(data: dict) -> str:
        return "  ".join(f"{entry['cells'][cell]['label']} {value['k']}/{value['n']} {value['sr']:.1%} "
                         f"[{value['ci95'][0]:.1%}, {value['ci95'][1]:.1%}]{salvage(value)}"
                         for cell, value in data["cells"].items())

    def salvage(value: dict) -> str:
        attempts = value.get("attempts") or {}
        if not any(attempts.values()):
            return ""
        return f" ({attempts['salvaged_episodes']} salvaged, {attempts['retried_rows']} rerun)"

    lines = [f"{entry['name']} {report['version']} ({report['rows_per_cell']} rows per cell; {report['status']}; frozen set sha256 {str(report['frozen_sha256'])[:12]}): success rate k/n pooled over each cell's objects, 95% equal-tailed "
             "interval of Beta(k+1, n-k+1) (uniform Beta(1, 1) prior); macro = unweighted mean over the cells"]
    for model, data in report["models"].items():
        if any(cell["deviations"] for cell in data["cells"].values()):
            status = "NON-HEADLINE: " + "; ".join(sorted({d for cell in data["cells"].values() for d in cell["deviations"]})[:3])
        elif not data["available"]:
            status = "missing: no adapter yet"
        elif data["missing_cells"]:
            status = "missing cells " + ", ".join(data["missing_cells"])
        else:
            status = f"macro {data['macro_sr']:.1%}"
        lines.append(f"  {model}: {cells(data)}  {status}".rstrip())
    for section in NON_HEADLINE_SECTIONS.values():
        if report.get(section):
            lines.append(f"  {section} (never in the headline):")
        for model, data in report.get(section, {}).items():
            deviations = sorted({d for cell in data["cells"].values() for d in cell["deviations"]})
            status = ("deviations: " + "; ".join(deviations[:3]) if deviations else
                      f"macro {data['macro_sr']:.1%}" if data["complete"] else "missing cells " + ", ".join(data["missing_cells"]))
            lines.append(f"    {model}: {cells(data)}  {status}".rstrip())
    if "challenge" in report:
        lines.append(f"  challenge track (never in the headline): {report['challenge']['rule']}")
        for model, data in report["challenge"]["models"].items():
            deviations = sorted({d for cell in data["cells"].values() for d in cell["deviations"]})
            status = ("NON-HEADLINE: " + "; ".join(deviations[:3]) if deviations else
                      f"macro {data['macro_sr']:.1%}" if data["complete"] else "missing cells " + ", ".join(data["missing_cells"]))
            lines.append(f"    {model}: {cells(data)}  {status}".rstrip())
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--manifest", type=Path, default=CONFIGS / "benchmark_manifest.json")
    commands = parser.add_subparsers(dest="command", required=True)
    frozen_parser = commands.add_parser("freeze")
    frozen_parser.add_argument("--task", required=True)
    frozen_parser.add_argument("--root", type=Path, default=Path("."), help="What the manifest's row paths are relative to.")
    frozen_parser.add_argument("--section", choices=("headline", "sensitivity"), default="headline",
                               help="The cells' rows (the score) or the sweep's, frozen and checked apart.")
    frozen_parser.add_argument("--output", type=Path,
                               help="The candidate; default db_<task>[_sensitivity]_frozen_candidate.json beside the rows.")
    score_parser = commands.add_parser("score")
    score_parser.add_argument("--task", required=True)
    score_parser.add_argument("--results", nargs="+", required=True, metavar="CELL=DIR",
                              help="Each cell's model run directories (run_benchmark_cell.py --output), e.g. train=outputs/db_r2_eval_train_gr00t.")
    score_parser.add_argument("--root", type=Path, default=Path("."))
    score_parser.add_argument("--version", choices=sorted(EVAL_VERSIONS), default=DEFAULT_EVAL_VERSION,
                              help="The rows scored: v1 a fixed half of each cell (the default), v1-plus every row.")
    score_parser.add_argument("--json", type=Path, help="Also write the report here.")
    score_parser.add_argument("--model-registry", type=Path, help="Trusted DexBench-leaderboard/models checkout")
    score_parser.add_argument("--accepted", type=Path, metavar="JSON",
                              help="What an eval launch accepted of each --results directory: the "
                                   "score_<task>.accepted.json that scripts/evaluation/eval_launch/launch.sh <config> "
                                   "--score writes into the launch's output, and passes here itself. A directory listed "
                                   "with rows is scored on those episodes only, read from its Runs' episode result files.")
    args = parser.parse_args(argv)
    try:
        return _run(args, parser)
    except Refusal as refusal:
        print(f"refused: {refusal}", file=sys.stderr)
        return 2


def _run(args, parser) -> int:
    manifest = json.loads(args.manifest.read_text())
    if args.task not in manifest["tasks"]:
        raise Refusal(f"no task {args.task!r} in {args.manifest.name}; its tasks are {sorted(manifest['tasks'])}")
    if args.command == "freeze":
        output = args.output or args.root / candidate_path(manifest, args.task, args.section)
        frozen = freeze(manifest, args.task, args.root, args.section)
        text = (json.dumps(frozen, indent=1) + "\n").encode()
        publish_new(output, text)
        pinned = pinned_path(args.task, args.section)
        matches = pinned.is_file() and json.loads(pinned.read_text()) == frozen
        print(f"froze {len(frozen)} row files -> {output}: " + ("matches the pinned set" if matches else
              f"differs from {pinned}; a benchmark run replays only the pinned set, so commit the candidate there "
              "on purpose (every result scored against the old set then stops matching)"))
        if args.section == "headline":
            versions = version_pin(manifest, args.task, text)
            versions_output = output.with_name(output.stem + "_versions.json")
            publish_new(versions_output, version_pin_bytes(versions))
            pinned_versions = version_pin_path(args.task)
            same = pinned_versions.is_file() and json.loads(pinned_versions.read_text()) == versions
            print(f"version rows -> {versions_output}: " + ("match the pinned rows" if same else
                  f"differ from {pinned_versions}; commit them there with the set, on purpose (every result of a "
                  "version scored against the old rows then stops matching)"))
        return 0
    frozen_sha256, sources = load_frozen_set(pinned_path(args.task), manifest, args.task, args.root)
    results = []
    for item in args.results:
        cell, separator, directory = item.partition("=")
        if not separator:
            parser.error(f"--results takes CELL=DIR, got {item!r}")
        results.append((cell, Path(directory)))
    accepted = json.loads(args.accepted.read_text()) if args.accepted is not None else None
    manifest = with_config_models(manifest, args.task, results)
    manifest = with_registered_models(manifest, args.task, results, args.model_registry)
    report = score(manifest, args.task, collect(manifest, args.task, results, frozen_sha256, accepted), sources,
                   frozen_sha256, args.version)
    print(format_report(manifest, args.task, report))
    if args.json is not None:
        args.json.write_text(json.dumps({"convention": INTERVAL, **report}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
