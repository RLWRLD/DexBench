# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Teleop collection from a recorded start-state dump: a row is held until a demo of it is kept."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

pytest.importorskip("isaaclab_arena.variations.variation_replay_scheduler", reason="needs IsaacLab-Arena importable")

from isaaclab_arena.variations.recorded_variation_samples import load_rebuild_variation_record  # noqa: E402
from isaaclab_arena.variations.variation_replay_scheduler import VariationReplayScheduler  # noqa: E402

from dexbench_arena.evaluation.artifacts import file_sha256  # noqa: E402
from dexbench_arena.variations.collection_rows import (  # noqa: E402
    DATASET_METADATA_KEY,
    CollectionRows,
    CollectionRowsExhausted,
    collection_conditions,
    embodiment_mismatches,
    hold_rows,
    install_collection_recording,
    record_collection_row,
)

REPO = Path(__file__).resolve().parents[3]
BASE = "dexbench_rby1_wuji2_joint.base_pose"
CARTON = "target_carton.pose"


def _rows_file(path: Path, count: int = 3, seed: int = 8101) -> Path:
    rows = [{"episode_in_env": index, "seed": seed, "variations": {BASE: [0.01 * index, 0.0, 0.0], CARTON: [0.0, 0.001 * index, 0.0]},
             "task_object": {"axis": "r3_target_asset", "key": "dexbench_retail_e1_fc_r3t1"}} for index in range(count)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


class _Episode:
    success = None


class _Recorder:
    """Isaac Lab's recorder manager as record_demos.py uses it: an export writes env_args, then the demo if it succeeded."""

    active_terms = ["actions"]

    def __init__(self):
        self.exported_successful_episode_count = 0
        self.episode = _Episode()
        self.env_args: dict = {}

    def get_ep_meta(self) -> dict:
        return {}

    def get_episode(self, env_id):
        return self.episode

    def export_episodes(self, env_ids=None):
        self.env_args.update(self.get_ep_meta())
        if self.episode.success:
            self.exported_successful_episode_count += 1
        self.episode = _Episode()


class _Env:
    def __init__(self, scheduler):
        self.variation_replay_scheduler = scheduler
        self.recorder_manager = _Recorder()
        self.rows_started: list[int] = []
        self.takes: list[dict] = []

    def reset(self):
        """Arena's _reset_idx for env 0: the finishing take's row is recorded, then the next take is assigned."""
        scheduler = self.variation_replay_scheduler
        if scheduler.num_assignments_started > scheduler.num_assignments_completed:
            self.takes.append(record_collection_row(self, 0))
            scheduler.complete_episodes([0])
        scheduler.assign_new_episodes([0])
        self.rows_started.append(scheduler.runtime_sample_for(BASE, [0])[0][0])

    def succeed(self):
        """process_success_condition: mark the take a success and export it."""
        self.recorder_manager.episode.success = True
        self.recorder_manager.export_episodes([0])


def test_a_row_is_held_through_discarded_reset_and_timed_out_takes_and_moves_on_only_when_a_demo_is_kept(tmp_path):
    path = _rows_file(tmp_path / "r3t1.jsonl")
    rows = CollectionRows.read(path)
    scheduler = VariationReplayScheduler(load_rebuild_variation_record(path, build_time_variation_keys=set()))
    assert hold_rows(scheduler, rows) is scheduler
    env = _Env(scheduler)
    assert install_collection_recording(env, scheduler)

    env.reset()
    env.recorder_manager.export_episodes([0])
    env.reset()
    env.reset()
    env.reset()
    assert env.rows_started == [0.0, 0.0, 0.0, 0.0]
    env.succeed()
    env.reset()
    env.reset()
    env.succeed()
    env.reset()
    assert [round(x, 3) for x in env.rows_started] == [0.0, 0.0, 0.0, 0.0, 0.01, 0.01, 0.02]
    sha = file_sha256(path)
    assert [take["collection_row"]["row"] for take in env.takes] == [0, 0, 0, 0, 1, 1]
    assert {take["collection_row"]["rows_sha256"] for take in env.takes} == {sha}

    env.succeed()
    stamp = env.recorder_manager.env_args[DATASET_METADATA_KEY]
    assert stamp["rows_sha256"] == sha and stamp["rows_file"] == str(path) and stamp["rows"] == 3
    assert stamp["demo_rows"] == {"demo_0": 0, "demo_1": 1, "demo_2": 2}
    with pytest.raises(CollectionRowsExhausted, match="every row"):
        env.reset()


def test_a_row_the_console_requeues_is_held_again_before_the_next_one_in_file_order(tmp_path):
    """A kept demo the operator throws away afterwards gives its row back: the console appends it to the requeue
    file, and the session holds it after the row it is on, then goes on in file order; a row requeued before the
    start comes first, and is not offered again when file order reaches it. An entry is taken once."""
    path = _rows_file(tmp_path / "r3t1.jsonl", count=6)
    requeue = tmp_path / "requeue.txt"
    requeue.write_text("4\n")
    scheduler = VariationReplayScheduler(load_rebuild_variation_record(path, build_time_variation_keys=set()))
    hold_rows(scheduler, CollectionRows.read(path, first_row=1, requeue_path=requeue))
    env = _Env(scheduler)
    assert install_collection_recording(env, scheduler)
    held = []
    for appended in (None, "1\n", None, None, "x\n"):
        if appended:
            requeue.write_text(requeue.read_text() + appended)
        env.reset()
        held.append(scheduler.held_row)
        env.succeed()
    env.reset()
    held.append(scheduler.held_row)
    assert held == [4, 1, 1, 2, 3, 5]
    assert env.recorder_manager.env_args[DATASET_METADATA_KEY]["demo_rows"] == {
        "demo_0": 4, "demo_1": 1, "demo_2": 1, "demo_3": 2, "demo_4": 3}


def test_a_session_resumes_from_a_row_and_runs_one_env(tmp_path):
    path = _rows_file(tmp_path / "r3t1.jsonl")
    scheduler = VariationReplayScheduler(load_rebuild_variation_record(path, build_time_variation_keys=set()))
    hold_rows(scheduler, CollectionRows.read(path, first_row=2))
    scheduler.assign_new_episodes([0])
    assert scheduler.source_record_index_for_env(0) == 2
    with pytest.raises(ValueError, match="runs one env"):
        scheduler.assign_new_episodes([1])
    with pytest.raises(ValueError, match="outside"):
        CollectionRows.read(path, first_row=3)
    started = VariationReplayScheduler(load_rebuild_variation_record(path, build_time_variation_keys=set()))
    started.assign_new_episodes([0])
    with pytest.raises(ValueError, match="before the first reset"):
        hold_rows(started, CollectionRows.read(path))


def test_a_teleop_session_without_a_recording_holds_its_row(tmp_path):
    path = _rows_file(tmp_path / "r3t1.jsonl")
    scheduler = VariationReplayScheduler(load_rebuild_variation_record(path, build_time_variation_keys=set()))
    env = _Env(hold_rows(scheduler, CollectionRows.read(path)))
    env.recorder_manager.active_terms = []
    assert not install_collection_recording(env, scheduler)


def test_rows_of_another_robot_are_refused_and_a_manifest_runs_fields_are_held(tmp_path):
    path = _rows_file(tmp_path / "tc1_mf3.jsonl")
    robots = ["dexbench_rby1_wuji2", "dexbench_rby1_wuji2_joint"]
    (problem,) = embodiment_mismatches(path, "dexbench_rby1_wuji2", robots)
    assert f"['{BASE}'], another robot's draws; this build runs dexbench_rby1_wuji2" in problem
    assert embodiment_mismatches(path, "dexbench_rby1_wuji2_joint", robots) == []

    manifest = {"tasks": {"r2": {"collection": {"batches": [{"batch": 1, "rows": {"tc1_mf3": {
        "path": str(path), "rows": 3, "environment": {"r2_target_sku": "TC1", "r2_target_lane": 3}}}}]}}}}
    rows = CollectionRows.read(path, manifest=manifest)
    assert rows.manifest_run == "r2/batch_1/tc1_mf3"

    class Build:
        r2_target_sku, r2_target_lane = "TC1", 4

    assert rows.build_differences(Build()) == ["r2_target_lane: the rows' Run r2/batch_1/tc1_mf3 sets 3, this build 4"]
    manifest["tasks"]["r2"]["collection"]["batches"][0]["rows"]["tc1_mf3"]["rows"] = 26
    with pytest.raises(ValueError, match="keeps 26"):
        CollectionRows.read(path, manifest=manifest)


def test_a_dumps_run_fields_are_held_wherever_the_file_is_read_from(tmp_path, monkeypatch):
    """The manifest's paths are repo-relative: the Run is found from another working directory, a copy of the dump
    elsewhere is matched by its rows' environment_cfg, and collection rows that match no Run are refused."""
    from dexbench_arena.variations import collection_rows

    def dump(path, lane):
        rows = [{"episode_in_env": index, "seed": 8211, "variations": {BASE: [0.01 * index, 0.0, 0.0]},
                 "environment_cfg": {"r2_target_sku": "TC1", "r2_target_lane": lane}} for index in range(3)]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return path

    monkeypatch.setattr(collection_rows, "REPO_ROOT", tmp_path / "repo")
    relative = "outputs/db_conditions/db_r2_collect/batch_1/tc1_mf3.jsonl"
    dump(tmp_path / "repo" / relative, 3)
    manifest = {"tasks": {"r2": {"collection": {"batches": [{"batch": 1, "rows": {"tc1_mf3": {
        "path": relative, "rows": 3, "environment": {"r2_target_sku": "TC1", "r2_target_lane": 3}}}}]}}}}
    monkeypatch.chdir(tmp_path)
    assert CollectionRows.read(tmp_path / "repo" / relative, manifest=manifest).manifest_run == "r2/batch_1/tc1_mf3"

    copy = CollectionRows.read(dump(tmp_path / "elsewhere.jsonl", 3), manifest=manifest)
    assert (copy.manifest_run, dict(copy.environment)) == ("r2/batch_1/tc1_mf3", {"r2_target_sku": "TC1", "r2_target_lane": 3})
    with pytest.raises(ValueError, match="matches no collection Run"):
        CollectionRows.read(dump(tmp_path / "other_lane.jsonl", 5), manifest=manifest)


def test_the_wrapper_takes_the_rows_and_records_one_demo_per_row_left(tmp_path):
    helper = runpy.run_path(str(REPO / "scripts/imitation_learning/_dexbench_arena_script.py"))
    take = helper["_take_collection_rows"]
    path = str(_rows_file(tmp_path / "r3t1.jsonl", seed=42))
    argv, rows = take("record_demos.py", ["--dataset_file", str(tmp_path / "d.hdf5"), "--collection_rows", path,
                                         "--collection_first_row", "1", "dexbench_retail_v1_environment"])
    assert argv == ["--num_demos", "2", "--dataset_file", str(tmp_path / "d.hdf5"), "dexbench_retail_v1_environment"]
    assert (rows.path, rows.first_row, rows.rows) == (path, 1, 3)
    for argv, reason in ((["--num_demos", "0"], "1..3"), (["--num_demos", "4"], "1..3"),
                         (["--collection_rows", "other.jsonl"], "one file")):
        with pytest.raises(SystemExit, match=reason):
            take("record_demos.py", ["--collection_rows", path, *argv, "dexbench_retail_v1_environment"])
    with pytest.raises(SystemExit, match="does not take --collection_rows"):
        take("replay_demos.py", ["--collection_rows", path])
    assert take("teleop.py", ["dexbench_retail_v1_environment"]) == (["dexbench_retail_v1_environment"], None)


def test_condition_rows_collection_starts_each_demo_from_the_row_its_session_recorded():
    env_args = {"dexbench_collection_rows": {"demo_rows": {"demo_0": 7, "demo_1": 3}}}
    conditions = {f"demo_{row}": {"variations": {"seed": row}} for row in range(5)}
    chosen = collection_conditions(env_args, ["demo_0", "demo_1"], conditions)
    assert chosen == {"demo_1": {"variations": {"seed": 3}, "condition_row": "demo_3"}}
    with pytest.raises(ValueError, match="demo_2"):
        collection_conditions(env_args, ["demo_0", "demo_2"], conditions)
