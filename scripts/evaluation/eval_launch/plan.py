# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Plan an evaluation launch: the work units, their expected cost, and the time estimate.

Kit-free: everything comes from the benchmark manifest (``write_benchmark_experiments.py``), the frozen pins and the
launch config, so this runs anywhere the package imports. No task is named here; a task reaches a plan the moment
its spec is in the manifest (design doc, "Adding tasks").

A unit is one ``run_benchmark_cell.py`` call: a task, a cell (or the sweep), the config's ``version`` (``v1``, the
default, or ``v1-plus``; ``protocol.EVAL_VERSIONS``) and ``slices``, ``{run: [start, end)}`` of each Run's rows at that
version, the form ``model_run.prepare`` replays and the scorer joins. Small Runs are grouped into one
unit; a Run longer than a unit is cut into row slices, each paying its own environment build, but never into slices
shorter than ``min_slice_builds`` builds. Units are ordered longest first, so the last units to start are the short
ones and the launch's tail stays near one unit.

Costs, best source first, per task: a measured slowdown and build time (``timings``), else the median of the
measured tasks', else the config's. An episode costs its cap times the slowdown (wall time over simulated time): one
that fails runs to its cap, so this is the worst case.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import re
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import yaml

from dexbench_arena import REPO_ROOT
from dexbench_arena.benchmark import BENCHMARK_CONFIGS, MANIFEST as MANIFEST_NAME
from dexbench_arena.benchmark.frozen import pinned_path, pinned_version_rows
from dexbench_arena.benchmark.models import CONFIG_MODEL_FIELDS, ModelEntry, resolve_model
from dexbench_arena.benchmark.protocol import DEFAULT_EVAL_VERSION, EVAL_VERSIONS, PRE_VERSION_EVAL_VERSION, cell_rows

from eval_launch import PACKAGE_DIR
from eval_launch.layout import gpu_count
from eval_launch.site import apply_site

MANIFEST = REPO_ROOT / BENCHMARK_CONFIGS / MANIFEST_NAME
PLAN_VERSION = 1
SWEEP = "sweep"
DEFAULT_COSTS = {
    "unit_minutes": 15.0,
    "slowdown": 5.0,
    "build_s": 120.0,
    "kit_start_s": 60.0,
    "min_slice_builds": 3.0,
    "units_per_gpu": 4.0,
}
"""Planning defaults: a 15 min unit; 5x wall time over simulated time (one task's measurement, GR00T on one L40S);
2 min per environment build; 1 min to start Kit with a seeded cache, paid by every unit (each is its own runner
process); a Run is not cut into slices shorter than three builds; about four units per GPU, so on few GPUs the units
grow past 15 min and pay fewer starts and builds, while on many the tail stays short. Measured costs come only from
the config's ``plan.timings`` (a file such as ``timings/l40s_example.json``, measured on one GPU type and layout);
without it the plan says every cost is a default."""
CONFIG_KEYS = {"model", "tasks", "cells", "sweep", "version", "submission", "servers", "plan", "backend", "output", "trees"}
RETIRED_KEYS = {
    "instructions": "the launcher no longer checks episodes' instructions against the training set's task texts (each "
                    "unit's outcome still lists the instructions it saw)",
}
"""Keys a launch config once had: read, so a launch planned with one resumes, and refused on a new launch, where they
would claim a setting that does nothing (``launch.main``)."""


@dataclass(frozen=True)
class Costs:
    unit_s: float
    slowdown: float
    build_s: float
    kit_start_s: float
    min_slice_builds: float
    units_per_gpu: float
    timings: Mapping[str, Mapping[str, float]]

    def _per_task(self, task: str, field: str, default: float) -> tuple[float, str]:
        """The task's measured value, else the median of the measured tasks', else the config's."""
        if self.timings.get(task, {}).get(field):
            return float(self.timings[task][field]), "measured"
        measured = sorted(float(entry[field]) for entry in self.timings.values() if entry.get(field))
        if measured:
            return measured[len(measured) // 2], "median of measured tasks"
        return default, "config default"

    def episode_s(self, task: str, cap_s: float) -> tuple[float, str]:
        """Worst-case seconds of wall time per episode (the cap times the slowdown) and where the slowdown comes from."""
        slowdown, source = self._per_task(task, "slowdown", self.slowdown)
        return cap_s * slowdown, f"cap {cap_s:g} s x slowdown {slowdown:g}, {source}"

    def build(self, task: str) -> float:
        return self._per_task(task, "build_s", self.build_s)[0]


def task_list(text: str | None) -> list[str] | None:
    """``--tasks``: comma-separated task names, each stripped (``"r2, r3"`` is r2 and r3), or None when not given."""
    return [task.strip() for task in text.split(",") if task.strip()] if text else None


def load_config(path: str | Path) -> dict[str, Any]:
    """The launch config (YAML), checked for the fields a plan needs."""
    config = apply_site(yaml.safe_load(Path(path).read_text()) or {}, Path(path))
    unknown = sorted(set(config) - CONFIG_KEYS - set(RETIRED_KEYS))
    if unknown:
        raise ValueError(f"{path}: unknown fields {unknown}; a launch config has {sorted(CONFIG_KEYS)}")
    if type(config.get("submission", False)) is not bool:
        raise ValueError("submission must be a boolean")
    model = config.get("model") or {}
    if "key" not in model:
        raise ValueError(f"{path}: model.key names an entry of benchmark/models.py, or the model this config registers")
    unknown = sorted(set(model) - {"checkpoint", "checkpoints", "checkpoint_sha256", *CONFIG_MODEL_FIELDS})
    if unknown:
        raise ValueError(f"{path}: model has key, checkpoint or checkpoints, and for a model of your own "
                         f"{', '.join(field for field in CONFIG_MODEL_FIELDS if field != 'key')}; not {unknown}")
    described = sorted(set(model) & {"label", "note", "group"})
    if described and "policy" not in model:
        raise ValueError(f"{path}: model.{', model.'.join(described)} describe a model this config registers, which "
                         f"needs model.policy (its client's block); without one, {model['key']!r} runs as the registered "
                         "model of that key, under its own label and group")
    try:
        model_entry(config)
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from None
    if ("checkpoint" in model) == ("checkpoints" in model):
        raise ValueError(f"{path}: give model.checkpoint (one for every task) or model.checkpoints (task -> checkpoint)")
    checkpoint_hash = model.get("checkpoint_sha256")
    if checkpoint_hash is not None and (not isinstance(checkpoint_hash, str) or re.fullmatch(r"[a-f0-9]{64}", checkpoint_hash) is None):
        raise ValueError("model.checkpoint_sha256 must be a lowercase SHA-256 digest")
    if config.get("submission", False) and ("checkpoints" in model or checkpoint_hash is None):
        raise ValueError("submission needs one model.checkpoint and model.checkpoint_sha256 for every task")
    backend = config.get("backend") or {}
    if config.get("output") and not Path(str(config["output"])).is_absolute() and not (
            backend.get("kind") == "k8s" or "k8s" in backend):
        config["output"] = str(Path(str(config["output"])).absolute())
    if config.get("sweep"):
        raise ValueError(f"{path}: sweep: true is not runnable through the launcher yet (a sweep unit has no cell or "
                         "row slices for run_benchmark_cell.py); run the sweep with run_benchmark_cell.py --sweep")
    return config


def model_spec(config: Mapping[str, Any]) -> str | dict[str, Any]:
    """What the workers pass to ``run_benchmark_cell.py`` for the config's model: its key, for an entry of
    benchmark/models.py, or the entry the config registers (``model`` with a ``policy`` block: its key, label, note,
    group and policy; ``models.config_model``), which goes as ``--model-entry``."""
    model = config["model"]
    if "policy" not in model:
        return str(model["key"])
    return {field: model[field] for field in CONFIG_MODEL_FIELDS if field in model}


def model_entry(config: Mapping[str, Any]) -> ModelEntry:
    """The config's model: a registered entry by its key, or the one the config registers."""
    return resolve_model(model_spec(config))


RECIPE_FIELDS = ("checkpoint", "checkpoint_id", "port", "token_file", "worker_dir", "launcher")
"""What ``eval_worker.PolicyServer`` fills into a recipe's steps and ``client_set`` with ``str.format``."""


def check_recipe(config: Mapping[str, Any]) -> None:
    """Refuse a config whose workers could not start its server: no ``servers.recipe.command`` (an example that is
    only planned), or a step that does not format with the worker's fields. Each step is ``str.format``-ed before
    ``bash -c``, so a brace the shell should see is written doubled, ``${{HOME}}``; a bare ``${HOME}`` was a
    KeyError at every worker's first server start. The fields' values are shell-quoted already."""
    recipe = (config.get("servers") or {}).get("recipe") or {}
    if not recipe.get("command"):
        raise ValueError("servers.recipe.command is missing: the workers could not start a policy server, so this "
                         "config can only be planned (--plan-only)")
    fields = {name: "/x" for name in RECIPE_FIELDS} | {"port": 5810}
    steps = [(step, recipe[step]) for step in ("setup", "command", "probe") if recipe.get(step)]
    steps += [(f"client_set[{index}]", item) for index, item in enumerate(recipe.get("client_set", []))]
    for step, text in steps:
        try:
            str(text).format(**fields)
        except (KeyError, IndexError, ValueError) as error:
            raise ValueError(f"servers.recipe.{step}: {error!r} when filled in with {{{'}, {'.join(RECIPE_FIELDS)}}}; "
                             "write a brace the shell should see doubled ({{ and }}, e.g. ${{HOME}})") from None


def load_timings(config: Mapping[str, Any], override: str | Path | None = None) -> dict[str, Any] | None:
    """The measured costs to plan with: ``override`` (the CLI's ``--timings``), else the config's ``plan.timings``
    (relative to this directory, so the same config works in any checkout), else None (every cost a default)."""
    path = override or (config.get("plan") or {}).get("timings")
    if not path:
        return None
    path = Path(path) if Path(path).is_absolute() else PACKAGE_DIR / path
    return json.loads(path.read_text())


def costs_from(config: Mapping[str, Any], timings: Mapping[str, Any] | None = None) -> Costs:
    plan = {**DEFAULT_COSTS, **{k: v for k, v in (config.get("plan") or {}).items() if k in DEFAULT_COSTS}}
    return Costs(unit_s=float(plan["unit_minutes"]) * 60.0, slowdown=float(plan["slowdown"]),
                 build_s=float(plan["build_s"]), kit_start_s=float(plan["kit_start_s"]),
                 min_slice_builds=float(plan["min_slice_builds"]), units_per_gpu=float(plan["units_per_gpu"]),
                 timings=(timings or {}).get("tasks", {}))


def _pinned(task: str, section: str) -> bool:
    return pinned_path(task, section).is_file()


def eval_version(config: Mapping[str, Any]) -> str:
    """The config's ``version``, the rows of each cell a launch runs; ``v1`` when it names none."""
    version = config.get("version", DEFAULT_EVAL_VERSION)
    if version not in EVAL_VERSIONS:
        raise ValueError(f"version {version!r} is not one of {sorted(EVAL_VERSIONS)}")
    return version


def plan_eval_version(plan: Mapping[str, Any]) -> str:
    """The version a plan runs: its ``eval_version``, or ``v1-plus`` for a plan made before versions."""
    return plan.get("eval_version", PRE_VERSION_EVAL_VERSION)


def unit_eval_version(unit: Mapping[str, Any]) -> str:
    """The version a unit runs: its ``version``, or ``v1-plus`` for a unit planned before versions."""
    return unit.get("version", PRE_VERSION_EVAL_VERSION)


def version_counts(manifest: Mapping[str, Any], task: str, cell: str, version: str) -> dict[str, int]:
    """``{run: rows}`` of a cell at ``version`` (``protocol.cell_rows``: every row, or the rows pinned beside the frozen
    set), in the manifest's Run order; a Run the version gives none is left out."""
    quota = {run: int(row["rows"]) for run, row in manifest["tasks"][task]["cells"][cell]["rows"].items()}
    return {run: len(rows) for run, rows in cell_rows(quota, None, version, partial(pinned_version_rows, manifest, task, cell)).items()}


def task_states(manifest: Mapping[str, Any], config: Mapping[str, Any],
                pinned: Callable[[str, str], bool] = _pinned) -> dict[str, dict[str, Any]]:
    """Every task the config asks for: its checkpoint, tree and cells, or the reason it cannot run yet."""
    model = model_entry(config)
    model_key = model.key
    if model.rows is not None:
        raise ValueError(f"model {model_key} runs the first {model.rows} rows of each cell (a pilot); the launcher "
                         "plans whole cells, so run it with run_benchmark_cell.py")
    client_set = {item.split("=", 1)[0] for item in ((config.get("servers") or {}).get("recipe") or {}).get("client_set", [])}
    if model.policy is not None and not {"remote_url", "remote_port"} & (set(model.policy) | client_set):
        raise ValueError(f"model {model_key}'s policy block has no remote_url or remote_port, and the recipe's "
                         "client_set sets neither: the workers could not point its client at their servers")
    wanted = config.get("tasks", "all")
    names = list(manifest["tasks"]) if wanted == "all" else list(wanted)
    cells_wanted = config.get("cells", "all")
    if not (cells_wanted == "all" or isinstance(cells_wanted, (list, dict))):
        raise ValueError(f"cells is all, a list for every task or a map of task to list, not {cells_wanted!r}")
    checkpoints = config["model"].get("checkpoints")
    trees = config.get("trees") or {}
    default_tree = (config.get("backend") or {}).get("tree")
    states = {}
    for task in names:
        entry = manifest["tasks"].get(task)
        state: dict[str, Any] = {"runnable": False}
        states[task] = state
        if entry is None:
            state["reason"] = "not on the benchmark spec (no manifest entry)"
            continue
        if entry.get("rows_pending"):
            state["reason"] = f"rows pending: {entry['rows_pending']}"
            continue
        if not pinned(task, "headline"):
            state["reason"] = f"no pinned frozen set ({pinned_path(task).name})"
            continue
        if checkpoints is not None and task not in checkpoints:
            raise ValueError(f"model.checkpoints names no checkpoint for {task}; list it or drop the task")
        if cells_wanted == "all":
            cells = list(entry["cells"])
        else:
            cells = list(cells_wanted if isinstance(cells_wanted, list) else cells_wanted.get(task, []))
        unknown = sorted(set(cells) - set(entry["cells"]))
        if unknown:
            raise ValueError(f"{task} has no cells {unknown}; its cells are {sorted(entry['cells'])}")
        cells = [cell for cell in cells if model.runs(task, cell)]
        if not cells:
            state["reason"] = f"model {model_key} runs none of its requested cells"
            continue
        state.update(runnable=True, cells=cells, cap_s=float(entry["cap_s"]),
                     checkpoint=checkpoints[task] if checkpoints is not None else config["model"]["checkpoint"],
                     tree=trees.get(task, default_tree))
        if config.get("sweep") and entry.get("sensitivity") and model.runs(task, None) and pinned(task, "sensitivity"):
            state["sweep"] = True
    return states


ROWS_COMMAND = "scripts/download_datasets.sh --eval-conditions"
"""What installs every cell's frozen rows (``outputs/db_conditions/``) into a tree, each checked against its pin."""


def missing_inputs(manifest: Mapping[str, Any], states: Mapping[str, Mapping[str, Any]],
                   root: Path = REPO_ROOT) -> dict[str, str]:
    """``{path: the command that installs it}`` for every file a planned task's units read from the tree at ``root``
    that is not there: each planned cell's row files, and the task's ``runtime_inputs`` (the spec's, through the
    manifest: DB-H3's input package). A unit on such a tree would fail at its start, every one alike."""
    missing: dict[str, str] = {}
    for task, state in states.items():
        if not state.get("runnable"):
            continue
        entry = manifest["tasks"][task]
        for cell in state["cells"]:
            for row in entry["cells"][cell]["rows"].values():
                if not (root / row["path"]).is_file():
                    missing[row["path"]] = ROWS_COMMAND
        for path, command in (entry.get("runtime_inputs") or {}).items():
            if not (root / path).is_file():
                missing[path] = command
    return missing


def check_inputs(manifest: Mapping[str, Any], states: Mapping[str, Mapping[str, Any]], root: Path = REPO_ROOT) -> None:
    """Refuse a plan whose units the tree at ``root`` cannot run (:func:`missing_inputs`), naming what is missing
    and the command that installs it, run in the tree."""
    missing = missing_inputs(manifest, states, root)
    if not missing:
        return
    by_command: dict[str, list[str]] = {}
    for path, command in missing.items():
        by_command.setdefault(command, []).append(path)
    raise ValueError(f"{root} lacks what the planned tasks read: " + "; ".join(
        f"{len(paths)} files ({', '.join(sorted(paths)[:3])}{', ...' if len(paths) > 3 else ''}): run {command} "
        "there" for command, paths in by_command.items()))


def _cell_units(task: str, cell: str, rows: Mapping[str, int], episode_s: float, build_s: float,
                costs: Costs, unit_s: float) -> list[dict[str, Any]]:
    """The cell's units, in the manifest's Run order: big Runs cut into slices, small ones grouped."""
    units: list[dict[str, Any]] = []
    group: dict[str, list[int]] = {}
    group_s = 0.0

    def close() -> None:
        nonlocal group, group_s
        if group:
            units.append({"slices": group, "runs": len(group), "episodes": sum(e - s for s, e in group.values()),
                          "expected_s": costs.kit_start_s + group_s,
                          "overhead_s": costs.kit_start_s + len(group) * build_s})
        group, group_s = {}, 0.0

    for run, count in rows.items():
        work = count * episode_s
        parts = min(count, math.ceil(work / unit_s), max(1, math.floor(work / (costs.min_slice_builds * build_s))))
        if parts > 1:
            close()
            for index in range(parts):
                start, end = count * index // parts, count * (index + 1) // parts
                units.append({"slices": {run: [start, end]}, "runs": 1, "episodes": end - start,
                              "expected_s": costs.kit_start_s + build_s + (end - start) * episode_s,
                              "overhead_s": costs.kit_start_s + build_s})
            continue
        if group and group_s + build_s + work > unit_s:
            close()
        group[run] = [0, count]
        group_s += build_s + work
    close()
    for index, unit in enumerate(units):
        unit.update(id=f"{task}.{cell}.{index:03d}", task=task, section="headline", cell=cell)
    return units


def _sweep_unit(task: str, entry: Mapping[str, Any], episode_s: float, build_s: float, costs: Costs) -> dict[str, Any]:
    """The sweep as one unit: its rows are per build (``recorded_variation_samples_paths``), not yet sliced."""
    sweep = entry["sensitivity"]
    builds = len(sweep["paths"])
    episodes = builds * int(sweep["rows_per_path"])
    return {"id": f"{task}.{SWEEP}.000", "task": task, "section": "sensitivity", "cell": None, "slices": None,
            "runs": builds, "episodes": episodes,
            "expected_s": costs.kit_start_s + builds * build_s + episodes * episode_s,
            "overhead_s": costs.kit_start_s + builds * build_s}


DEADLINE_FACTOR = 3.0
MIN_DEADLINE_S = 1800.0
"""A unit is stopped after max(MIN_DEADLINE_S, DEADLINE_FACTOR x its planned worst case) unless the config's ``plan``
says otherwise (``deadline_factor``, ``min_deadline_s``): the plan already assumes every episode runs to its cap."""


def deadline_factor(plan_config: Mapping[str, Any] | None = None) -> float:
    return float((plan_config or {}).get("deadline_factor", DEADLINE_FACTOR))


def deadline_s(unit: Mapping[str, Any], plan_config: Mapping[str, Any] | None = None) -> float:
    """How long ``unit`` may run (the worker's and the drivers' deadline): here, Kit-free, so the smoke check reads
    the same rule."""
    return max(float((plan_config or {}).get("min_deadline_s", MIN_DEADLINE_S)),
               deadline_factor(plan_config) * float(unit.get("expected_s", 0)))


def plan_units(manifest: Mapping[str, Any], states: Mapping[str, Mapping[str, Any]], costs: Costs,
               unit_s: float | None = None, version: str = DEFAULT_EVAL_VERSION) -> list[dict[str, Any]]:
    unit_s = costs.unit_s if unit_s is None else unit_s
    units = []
    for task, state in states.items():
        if not state["runnable"]:
            continue
        entry = manifest["tasks"][task]
        episode_s, _ = costs.episode_s(task, state["cap_s"])
        build_s = costs.build(task)
        for cell in state["cells"]:
            rows = version_counts(manifest, task, cell, version)
            units += [{**unit, "version": version} for unit in _cell_units(task, cell, rows, episode_s, build_s, costs, unit_s)]
        if state.get("sweep"):
            units.append(_sweep_unit(task, entry, episode_s, build_s, costs))
    for unit in units:
        unit["checkpoint"] = states[unit["task"]]["checkpoint"]
        unit["tree"] = states[unit["task"]]["tree"]
    return sorted(units, key=lambda unit: (-unit["expected_s"], unit["id"]))


def check_coverage(manifest: Mapping[str, Any], units: list[dict[str, Any]]) -> None:
    """Every planned cell's every row at its units' version in exactly one unit, as the scorer requires of the parts
    of one run."""
    covered: dict[tuple[str, str, str], list[tuple[int, int]]] = {}
    for unit in units:
        for run, (start, end) in (unit["slices"] or {}).items():
            covered.setdefault((unit["task"], unit["cell"], run), []).append((start, end))
    versions = {(unit["task"], unit["cell"]): unit_eval_version(unit) for unit in units if unit["cell"]}
    for (task, cell, run), parts in covered.items():
        count = version_counts(manifest, task, cell, versions[task, cell])[run]
        parts.sort()
        if parts[0][0] != 0 or parts[-1][1] != count or any(a[1] != b[0] for a, b in zip(parts, parts[1:])):
            raise AssertionError(f"{task} {cell} {run}: slices {parts} do not cover rows 0..{count} exactly once")
    for task in {unit["task"] for unit in units}:
        for cell in {unit["cell"] for unit in units if unit["task"] == task and unit["cell"]}:
            runs = set(version_counts(manifest, task, cell, versions[task, cell]))
            planned = {run for (t, c, run) in covered if (t, c) == (task, cell)}
            if planned != runs:
                raise AssertionError(f"{task} {cell}: Runs {sorted(runs - planned)} are in no unit")


def estimate(units: list[dict[str, Any]], gpus: int) -> dict[str, float]:
    """List-schedule the units in plan order onto ``gpus`` workers: the launch's wall time and GPU hours, and the share
    of the GPU hours spent starting Kit and building environments rather than running episodes."""
    finish = [0.0] * max(1, gpus)
    heapq.heapify(finish)
    for unit in units:
        heapq.heappush(finish, heapq.heappop(finish) + unit["expected_s"])
    total = sum(unit["expected_s"] for unit in units)
    wall = max(finish) if units else 0.0
    overhead = sum(unit.get("overhead_s", 0.0) for unit in units)
    return {"gpus": gpus, "gpu_hours": total / 3600.0, "wall_hours": wall / 3600.0,
            "overhead_share": overhead / total if total else 0.0,
            "utilisation": total / (wall * len(finish)) if wall else 0.0,
            "longest_unit_hours": (units[0]["expected_s"] / 3600.0) if units else 0.0}


def make_plan(config: Mapping[str, Any], manifest_path: str | Path = MANIFEST,
              timings: Mapping[str, Any] | None = None,
              pinned: Callable[[str, str], bool] = _pinned) -> dict[str, Any]:
    """The plan: per-task states, the units in launch order and the estimate on the config's GPUs."""
    manifest_bytes = Path(manifest_path).read_bytes()
    manifest = json.loads(manifest_bytes)
    costs = costs_from(config, timings)
    states = task_states(manifest, config, pinned)
    version = eval_version(config)
    gpus = gpu_count((config.get("backend") or {}).get("gpus", 1))
    work = sum(unit["expected_s"] for unit in plan_units(manifest, states, costs, costs.unit_s, version))
    unit_s = max(costs.unit_s, work / (gpus * costs.units_per_gpu))
    units = plan_units(manifest, states, costs, unit_s, version)
    floor = unit_s / 8
    while len(units) < gpus and unit_s > floor:
        unit_s /= 2
        smaller = plan_units(manifest, states, costs, unit_s, version)
        if len(smaller) == len(units):
            break
        units = smaller
    if config.get("submission", False):
        for unit in units:
            unit["submission"] = True
            unit["checkpoint_sha256"] = config["model"].get("checkpoint_sha256")
    check_coverage(manifest, units)
    for task, state in states.items():
        if state["runnable"]:
            state["episode_s"], state["cost_source"] = costs.episode_s(task, state["cap_s"])
    return {
        "version": PLAN_VERSION,
        "config": config,
        "eval_version": version,
        "benchmark_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "costs": {"unit_s": unit_s, "slowdown": costs.slowdown, "build_s": costs.build_s,
                  "kit_start_s": costs.kit_start_s, "min_slice_builds": costs.min_slice_builds,
                  "units_per_gpu": costs.units_per_gpu},
        "tasks": states,
        "units": units,
        "estimate": estimate(units, gpus),
    }


def add_to_plan(plan: Mapping[str, Any], config: Mapping[str, Any], tasks: list[str],
                manifest_path: str | Path = MANIFEST, timings: Mapping[str, Any] | None = None,
                pinned: Callable[[str, str], bool] = _pinned) -> dict[str, Any]:
    """``plan`` with ``tasks`` planned and their units appended behind the existing ones (a running launch's queue
    takes them as they come). A task already in the plan is refused: its units may be running."""
    if config.get("submission", False) != plan["config"].get("submission", False):
        raise ValueError("cannot change submission recording within a launch")
    present = sorted(set(tasks) & {unit["task"] for unit in plan["units"]})
    if present:
        raise ValueError(f"{present} are already in the plan")
    if eval_version(config) != plan_eval_version(plan):
        raise ValueError(f"the plan runs {plan_eval_version(plan)}, the config asks for {eval_version(config)}; "
                         "one launch scores one version")
    added = make_plan({**config, "tasks": tasks}, manifest_path, timings, pinned)
    if added["benchmark_manifest_sha256"] != plan["benchmark_manifest_sha256"]:
        added_note = "the manifest changed since the plan was made; the new tasks are planned on the new one"
    else:
        added_note = None
    units = list(plan["units"]) + added["units"]
    return {**plan, "tasks": {**plan["tasks"], **added["tasks"]}, "units": units,
            "estimate": estimate(units, added["estimate"]["gpus"]),
            "additions": [*plan.get("additions", []), {"tasks": tasks, "manifest_sha256": added["benchmark_manifest_sha256"],
                                                        "note": added_note}]}


def summary(plan: Mapping[str, Any]) -> str:
    """The plan as the table ``--plan-only`` prints."""
    lines = [f"version {plan_eval_version(plan)}: {EVAL_VERSIONS[plan_eval_version(plan)]} rows per cell",
             f"{'task':<8}{'state':<10}{'cells':>6}{'units':>7}{'episodes':>10}{'GPU-h':>8}  cost per episode"]
    for task, state in plan["tasks"].items():
        units = [unit for unit in plan["units"] if unit["task"] == task]
        if not state["runnable"]:
            lines.append(f"{task:<8}{'skipped':<10}  {state['reason']}")
            continue
        hours = sum(unit["expected_s"] for unit in units) / 3600.0
        cells = len(state["cells"]) + (1 if state.get("sweep") else 0)
        lines.append(f"{task:<8}{'runs':<10}{cells:>6}{len(units):>7}{sum(u['episodes'] for u in units):>10}"
                     f"{hours:>8.1f}  {state['episode_s']:.0f} s ({state['cost_source']})")
    est = plan["estimate"]
    lines += ["", f"{len(plan['units'])} units, {sum(u['episodes'] for u in plan['units'])} episodes, "
                  f"{est['gpu_hours']:.1f} GPU-hours; on {est['gpus']} GPUs about {est['wall_hours']:.1f} h "
                  f"(utilisation {est['utilisation']:.0%}, longest unit {est['longest_unit_hours']:.2f} h, "
                  f"{est['overhead_share']:.0%} of GPU time on Kit starts and builds)"]
    if any(unit["cell"] is None for unit in plan["units"]):
        lines.append("note: each sweep is one unit (its rows are per build and not sliced yet), so it can set the tail")
    return "\n".join(lines)
