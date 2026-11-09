# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""A condition recording filtered to the settled rows an evaluation replays."""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/evaluation/filter_condition_records.py"
_spec = importlib.util.spec_from_file_location("filter_condition_records", SCRIPT)
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)
MANIFEST = Path(__file__).resolve().parents[3] / "scripts/evaluation/experiment_configs/benchmark/benchmark_manifest.json"


def _row(index, settled=True, ended=("time_out",), base=(0.0, 0.0, 0.0)):
    return {
        "episode_in_env": index,
        "settle": {"settled": settled, "unsettled": [] if settled else ["r2_target_drink"], "terminated_by": list(ended)},
        "variations": {"dexbench_rby1_wuji2_joint.base_pose": list(base)},
        "task_object": {"axis": "r2_target_sku", "key": "TC1"},
    }


def _recording(directory, run, rows):
    path = directory / run / tool.RESULTS_FILE
    path.parent.mkdir(parents=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_the_first_settled_rows_are_kept_unchanged_and_a_beyond_range_keeps_draws_outside_the_box():
    rows = [_row(0), _row(1, settled=False), _row(2, ended=("object_dropped",)), _row(3), _row(4)]
    assert tool.select(rows, 2) == [_row(0), _row(3)]
    with pytest.raises(ValueError, match="4 rows, 2 settled: fewer than the 3 asked for"):
        tool.select(rows[:4], 3)
    with pytest.raises(ValueError, match="a quota is positive"):
        tool.select(rows, 0)

    ring = [_row(0, base=(0.01, 0.0, 0.0)), _row(1, base=(0.045, -0.01, 0.0)), _row(2, base=(0.0, -0.04, 0.0))]
    box = ("dexbench_rby1_wuji2_joint.base_pose", [-0.03, -0.03, -1], [0.03, 0.03, 1])
    assert [row["episode_in_env"] for row in tool.select(ring, 2, box)] == [1, 2]
    with pytest.raises(ValueError, match="2 of them outside the box"):
        tool.select(ring, 3, box)


def test_a_benchmark_cell_is_filtered_to_its_declared_runs_quotas_and_files(tmp_path, monkeypatch, capsys):
    """--task/--cell: the manifest's Runs exactly, each to its quota, object and seed, and no draw repeated."""
    cell = json.loads(MANIFEST.read_text())["tasks"]["r2"]["cells"]["ga_o"]
    names = list(cell["rows"])
    monkeypatch.chdir(tmp_path)

    def recording(label, change=lambda directory: None, runs=(), draws=None):
        """The cell's recording with two rows to spare per Run, ``runs`` ({name: (object, rows, row fields)})
        in place of some, then ``change``; ``draws(run, i)`` is row i's base offset. Returns the tool's arguments."""
        directory = tmp_path / label
        for run, (name, row) in enumerate(cell["rows"].items()):
            key, count, fields = dict(runs).get(name, (row["object"], row["rows"] + 2, {}))
            base = draws or (lambda run, i: (0.001 * run, 0.0001 * i, 0.0))
            _recording(directory, name, [{**_row(i, base=base(run, i)), "seed": row["seed"],
                                          "task_object": {"axis": "r2_target_sku", "key": key}, **fields}
                                         for i in range(count)])
        change(directory)
        return ["--recorded", str(directory), "--task", "r2", "--cell", "ga_o", "--manifest", str(MANIFEST)]

    assert tool.main(recording("good")) == 0
    for row in cell["rows"].values():
        assert len(Path(row["path"]).read_text().splitlines()) == row["rows"]

    nan_base = {"variations": {"dexbench_rby1_wuji2_joint.base_pose": [float("nan"), 0, 0]}}
    for argv, reason in (
        (recording("stale", lambda d: _recording(d, "stale_run", [_row(0)])), "no Run of the cell ['stale_run']"),
        (recording("failed", lambda d: (d / "failed_run").mkdir()), "no Run of the cell ['failed_run']"),
        (recording("rebuild", lambda d: (d / names[0] / "episode_results_rebuild1.jsonl").write_text("")),
         "episode_results_rebuild1.jsonl']"),
        (recording("missing", lambda d: shutil.rmtree(d / names[1])), f"missing ['{names[1]}']"),
        (recording("object", runs={names[1]: ("TC1", 40, {})}), "'key': 'TC1'}; this Run's is"),
        (recording("nan", runs={names[1]: (cell["rows"][names[1]]["object"], 40, nan_base)}), "non-finite draw"),
        (recording("seed", runs={names[1]: (cell["rows"][names[1]]["object"], 40, {"seed": 9100})}),
         "recorded at seed(s) [9100]; this Run records at"),
    ):
        capsys.readouterr()
        assert tool.main(argv) == 1 and reason in capsys.readouterr().err, reason

    (tmp_path / "fresh").mkdir()
    monkeypatch.chdir(tmp_path / "fresh")
    for argv, reason in (
        (recording("shared_seed", draws=lambda run, i: (0.001 * min(run, 1), 0.0001 * i, 0.0)),
         f"{names[2]} row 0 repeats the draws of {names[1]} row 0"),
        (recording("no_draw", draws=lambda run, i: (0.001 * run, 0.0001 * i * (run != 1), 0.0)),
         f"{names[1]} row 1 repeats the draws of {names[1]} row 0"),
    ):
        capsys.readouterr()
        assert tool.main(argv) == 1, reason
        error = capsys.readouterr().err
        assert "the cell's rows repeat" in error and reason in error, error
        assert not any(Path(row["path"]).exists() for row in cell["rows"].values())


def test_a_sweep_keeps_the_settled_rows_of_each_build_in_its_own_file(tmp_path, monkeypatch):
    """A sensitivity recording's builds differ in their build-time draws, so each replays from its own file."""
    sweep = json.loads(MANIFEST.read_text())["tasks"]["r2"]["sensitivity"]
    monkeypatch.chdir(tmp_path)
    recorded = tmp_path / "recorded"
    key = {"axis": "r2_target_sku", "key": sweep["object"]}
    for index in range(len(sweep["paths"])):
        rows = [{**_row(i, settled=i != 1), "task_object": key} for i in range(sweep["rows_per_path"] + 1)]
        path = recorded / sweep["row"] / f"episode_results_rebuild{index}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    argv = ["--recorded", str(recorded), "--task", "r2", "--sensitivity", "--manifest", str(MANIFEST)]
    assert tool.main(argv) == 0
    kept = [json.loads(line)["episode_in_env"] for line in Path(sweep["paths"][0]).read_text().splitlines()]
    assert kept == [0, *range(2, sweep["rows_per_path"] + 1)]



def test_a_collection_batch_keeps_each_runs_quota_at_its_seed_and_refuses_too_few(tmp_path, monkeypatch, capsys):
    """--collection: each Run of the dump keeps its quota of settled rows (DB-R2's 28 or 26), recorded at its own
    seed; a Run with too few settled rows or rows of another seed is refused and nothing is written."""
    manifest = json.loads(MANIFEST.read_text())
    collection = manifest["tasks"]["r2"]["collection"]
    dump = {**collection["batches"][0], "axis": collection["axis"]}
    monkeypatch.chdir(tmp_path)

    def recording(label, spare=2, unsettled=(), seeds=None):
        directory = tmp_path / label
        for name, row in dump["rows"].items():
            seed = (seeds or {}).get(name, row["seed"])
            _recording(directory, name, [
                {**_row(i, settled=i not in unsettled), "seed": seed, "task_object": {"axis": dump["axis"], "key": row["object"]}}
                for i in range(row["rows"] + spare)])
        return ["--recorded", str(directory), "--task", "r2", "--collection", "--batch", "0", "--manifest", str(MANIFEST)]

    first = next(iter(dump["rows"]))
    for argv, reason in (
        (recording("short", unsettled=(0, 1, 2)), "settled: fewer than the"),
        (recording("seed", seeds={first: 9000}), "recorded at seed(s) [9000]; this Run records at"),
    ):
        capsys.readouterr()
        assert tool.main(argv) == 1 and reason in capsys.readouterr().err, reason
        assert not any(Path(row["path"]).exists() for row in dump["rows"].values())

    assert tool.main(recording("good", unsettled=(1,))) == 0
    for row in dump["rows"].values():
        kept = [json.loads(line) for line in Path(row["path"]).read_text().splitlines()]
        assert [line["episode_in_env"] for line in kept] == [0, *range(2, row["rows"] + 1)]
        assert {line["seed"] for line in kept} == {row["seed"]}
    assert sorted({row["rows"] for row in dump["rows"].values()}) == [26, 28]

    capsys.readouterr()
    no_dump = ["--recorded", str(tmp_path / "good"), "--task", "m1", "--collection", "--batch", "0", "--manifest", str(MANIFEST)]
    assert tool.main(no_dump) == 1 and "declares no collection dump" in capsys.readouterr().err
    other = ["--recorded", str(tmp_path / "good"), "--task", "r2", "--collection", "--batch", "7", "--manifest", str(MANIFEST)]
    assert tool.main(other) == 1 and "declares collection batches [0, 1], not 7" in capsys.readouterr().err
