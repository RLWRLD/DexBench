#!/usr/bin/env python3
# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Keep the settled conditions of a benchmark cell's condition recording: the rows its evaluation replays.

A condition recording (``db_<task>_conditions_<cell>.yaml``) runs each Run with the hold-pose policy and
``record_settle``, so every episode row says whether the scene came to rest and what ended the episode.
This keeps the rows whose scene settled and whose episode ran to its cap (ended by ``time_out`` alone),
in recorded order, and writes each Run's quota of them unchanged to the row file Arena's replay
(``environment_builder.recorded_variation_samples_path``) reads::

    python3 scripts/evaluation/filter_condition_records.py \\
        --recorded outputs/db_r2_conditions_train --task r2 --cell train

``--task <t> --cell <c>`` (or ``--task <t> --sensitivity``, or ``--task <t> --collection --batch <b>`` for a
batch of a training collection's start-state dump, ``db_<t>_collect_conditions_batch<b>.yaml``) takes everything from
``benchmark_manifest.json``: the exact set of Runs the cell records (a missing Run is refused, and so is
any other directory under the recording, or a rebuild file a cell does not record), each Run's quota,
its object (every kept row must record it in ``task_object``), the Run's seed (every kept row of a cell or
a collection Run must record it), that no two kept rows of a cell repeat a draw (``repeated_draws``), the box a beyond range keeps rows outside of, and the output files. A sweep keeps each build's rows in its own file (its build-time draws differ
per build). Every kept draw must be finite. A Run with fewer settled rows than its quota is refused
with its counts, and nothing is written then. An output that exists with other content is refused; the
same content is accepted (``dexbench_arena.evaluation.artifacts.publish_new``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dexbench_arena.evaluation.artifacts import is_finite, jsonl_bytes, publish_new, read_jsonl, recorded

RESULTS_FILE = "episode_results_rebuild0.jsonl"
MANIFEST = Path(__file__).resolve().parent / "experiment_configs" / "benchmark" / "benchmark_manifest.json"


def is_settled_condition(row: dict) -> bool:
    """The row's scene came to rest and its episode ran to the cap; refuses a row of no condition recording."""
    settle = recorded(row, "settle")
    if not isinstance(settle, dict):
        raise ValueError("a row has no `settle` field: this is not a condition recording (record_settle: true)")
    return bool(settle.get("settled")) and settle.get("terminated_by") == ["time_out"]


def outside_box(row: dict, key: str, low: list[float], high: list[float]) -> bool:
    """Whether ``row``'s ``variations[key]`` draw lies outside ``[low, high]`` in at least one component."""
    draw = (row.get("variations") or {}).get(key)
    if draw is None:
        raise ValueError(f"a row has no draw for {key!r}; its variations are {sorted(row.get('variations') or {})}")
    draw = [draw] if isinstance(draw, (int, float)) else list(draw)
    if not len(draw) == len(low) == len(high):
        raise ValueError(f"{key!r} draws {len(draw)} values; the box has {len(low)} lows and {len(high)} highs")
    return any(value < lo or value > hi for value, lo, hi in zip(draw, low, high))


def select(rows: list[dict], count: int, outside: tuple[str, list[float], list[float]] | None = None,
           task_object: dict | None = None, seed: int | None = None) -> list[dict]:
    """The first ``count`` settled conditions of ``rows`` (outside the box, if given); ValueError when too few."""
    if count <= 0:
        raise ValueError(f"asked for {count} rows; a quota is positive")
    kept = [row for row in rows if is_settled_condition(row)]
    settled = len(kept)
    if outside is not None:
        kept = [row for row in kept if outside_box(row, *outside)]
    if len(kept) < count:
        where = f", {len(kept)} of them outside the box" if outside is not None else ""
        raise ValueError(f"{len(rows)} rows, {settled} settled{where}: fewer than the {count} asked for")
    objects = {json.dumps(recorded(row, "task_object"), sort_keys=True) for row in kept[:count]}
    if len(objects) > 1:
        raise ValueError(f"the rows name {len(objects)} different objects; a condition file holds one")
    if task_object is not None:
        for row in kept[:count]:
            ran = recorded(row, "task_object") or {}
            if (ran.get("axis"), ran.get("key")) != (task_object["axis"], task_object["key"]):
                raise ValueError(f"a row records task_object {ran or None}; this Run's is {task_object}")
    if seed is not None:
        seeds = sorted({row.get("seed") for row in kept[:count]}, key=str)
        if seeds != [seed]:
            raise ValueError(f"the rows were recorded at seed(s) {seeds}; this Run records at {seed}")
    if not all(is_finite(row.get("variations")) for row in kept[:count]):
        raise ValueError("a kept row records a non-finite draw")
    return kept[:count]


def repeated_draws(selected: dict[str, list[dict]]) -> list[str]:
    """Draws the kept rows of one cell repeat, by two rules. Within a Run, no two rows may record the same
    ``variations``, whatever their values: a Run whose rows did not draw replays one start state. Across Runs, no
    two rows may agree on every coordinate that varies over the cell where one of them is real-valued: that is two
    Runs drawing at one seed (each then replays the other's offsets at its own setting). Draws of integers only
    (a lane, an index) may repeat across Runs."""
    rows = [(run, row) for run, kept in selected.items() for row in kept]
    problems = []
    for run, kept in selected.items():
        seen: dict[str, str] = {}
        for row in kept:
            recorded = json.dumps(row.get("variations") or {}, sort_keys=True)
            where = f"{run} row {row.get('episode_in_env')}"
            if recorded in seen:
                problems.append(f"{where} repeats the draws of {seen[recorded]}")
            else:
                seen[recorded] = where
    values: dict[str, set[str]] = {}
    for _run, row in rows:
        for key, value in (row.get("variations") or {}).items():
            values.setdefault(key, set()).add(json.dumps(value, sort_keys=True))
    varying = sorted(key for key, seen in values.items() if len(seen) > 1)
    first: dict[str, tuple[str, str]] = {}
    for run, row in rows:
        draws = {key: (row.get("variations") or {}).get(key) for key in varying}
        if not any(isinstance(value, float) or (isinstance(value, list) and any(isinstance(v, float) for v in value))
                   for value in draws.values()):
            continue
        vector = json.dumps(draws, sort_keys=True)
        where = f"{run} row {row.get('episode_in_env')}"
        if vector in first and first[vector][0] != run:
            problems.append(f"{where} repeats the draws of {first[vector][1]}")
        else:
            first.setdefault(vector, (run, where))
    return problems


def _runs(recorded: Path) -> dict[str, Path]:
    if recorded.is_file():
        return {recorded.stem: recorded}
    runs = {path.parent.name: path for path in sorted(recorded.glob(f"*/{RESULTS_FILE}"))}
    if not runs:
        raise ValueError(f"no */{RESULTS_FILE} under {recorded}")
    return runs


def _rebuilds(recorded: Path) -> dict[str, Path]:
    """``{<run>_rebuild<i>: path}`` for every rebuild of every Run, in rebuild order."""
    rebuilds = {}
    for path in sorted(recorded.glob("*/episode_results_rebuild*.jsonl")):
        index = int(path.stem.removeprefix("episode_results_rebuild"))
        rebuilds[(path.parent.name, index)] = path
    if not rebuilds:
        raise ValueError(f"no */episode_results_rebuild*.jsonl under {recorded}")
    return {f"{run}_rebuild{index}": path for (run, index), path in sorted(rebuilds.items())}


def manifest_plan(manifest: dict, task: str, cell: str | None,
                  batch: int | None = None) -> tuple[dict, tuple | None, bool]:
    """``({run: (quota, output path, task_object, seed)}, outside, per_rebuild)`` of a benchmark cell, the sweep
    (``cell`` None) or batch ``batch`` of the training collection's dump; ``seed`` is the Run's own (a cell's or a
    collection's), None for the sweep's builds."""
    entry = manifest["tasks"][task]
    if batch is not None:
        dump = entry.get("collection")
        if dump is None:
            raise ValueError(f"{task} declares no collection dump (its training starts do not come from one)")
        batches = {declared["batch"]: declared for declared in dump["batches"]}
        if batch not in batches:
            raise ValueError(f"{task} declares collection batches {sorted(batches)}, not {batch}")
        plan = {name: (row["rows"], Path(row["path"]), {"axis": dump["axis"], "key": row["object"]}, row["seed"])
                for name, row in batches[batch]["rows"].items()}
        return plan, None, False
    if cell is None:
        sweep = entry["sensitivity"]
        task_object = {"axis": sweep["axis"], "key": sweep["object"]}
        plan = {Path(path).stem: (sweep["rows_per_path"], Path(path), task_object, None) for path in sweep["paths"]}
        return plan, None, True
    if cell not in entry["cells"]:
        raise ValueError(f"{task} has no cell {cell!r}; its cells are {sorted(entry['cells'])}")
    cell_entry = entry["cells"][cell]
    if cell_entry["conditions"] is None or cell_entry["frozen"]:
        raise ValueError(f"{task} {cell} replays frozen rows; nothing of it is recorded to filter")
    plan = {name: (row["rows"], Path(row["path"]), {"axis": cell_entry["axis"], "key": row["object"]}, row["seed"])
            for name, row in cell_entry["rows"].items()}
    outside = cell_entry.get("outside")
    return plan, (tuple(outside) if outside else None), False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--recorded", type=Path, required=True, help="The condition recording's runner output directory.")
    parser.add_argument("--task", required=True, help="The benchmark task: the Runs, quotas, box and outputs come from the manifest.")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--cell", help="The cell whose recording this is.")
    target.add_argument("--sensitivity", action="store_true", help="The sweep's recording.")
    target.add_argument("--collection", action="store_true",
                        help="A batch of the training collection's dump (db_<task>_collect_conditions_batch<b>.yaml).")
    parser.add_argument("--batch", type=int, help="With --collection: the batch the recording is of.")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args(argv)
    if args.collection != (args.batch is not None):
        parser.error("--batch goes with --collection, and --collection names its --batch")

    try:
        plan, outside, per_rebuild = manifest_plan(json.loads(args.manifest.read_text()), args.task, args.cell,
                                                   args.batch)
        runs = _rebuilds(args.recorded) if per_rebuild else _runs(args.recorded)
    except ValueError as error:
        print(f"Refused, nothing written: {error}", file=sys.stderr)
        return 1
    planned_dirs = {name.rsplit("_rebuild", 1)[0] for name in plan} if per_rebuild else set(plan)
    directories = {path.name for path in args.recorded.iterdir() if path.is_dir()} if args.recorded.is_dir() else set()
    stray = sorted(directories - planned_dirs)
    extra_rebuilds = [] if per_rebuild else sorted(
        str(path) for path in args.recorded.glob("*/episode_results_rebuild*.jsonl") if path.name != RESULTS_FILE)
    if stray or extra_rebuilds:
        print(f"Refused, nothing written: directories that are no Run of the cell {stray}, "
              f"or rebuild files a cell does not record {extra_rebuilds[:3]}", file=sys.stderr)
        return 1
    missing, extra = sorted(set(plan) - set(runs)), sorted(set(runs) - set(plan))
    if missing or extra:
        print(f"Refused, nothing written: the recording's Runs are not the cell's (missing {missing}, extra {extra})",
              file=sys.stderr)
        return 1
    selected, problems = {}, []
    for run, (count, output, task_object, seed) in plan.items():
        try:
            selected[run] = select(read_jsonl(runs[run])[1], count, outside, task_object, seed)
            if output.exists() and output.read_bytes() != jsonl_bytes(selected[run]):
                raise ValueError(f"{output} exists with other content")
        except ValueError as error:
            problems.append(f"{run} ({runs[run]}): {error}")
    if not problems and args.cell is not None:
        repeats = repeated_draws(selected)
        if repeats:
            problems = [f"the cell's rows repeat {len(repeats)} draws (every Run draws at its own seed): "
                        + "; ".join(repeats[:3])]
    if problems:
        print("Refused, nothing written:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    for run, rows in selected.items():
        output = plan[run][1]
        written = publish_new(output, jsonl_bytes(rows))
        print(f"{run}: {len(rows)} of {len(read_jsonl(runs[run])[1])} rows -> {output}{'' if written else ' (unchanged)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
