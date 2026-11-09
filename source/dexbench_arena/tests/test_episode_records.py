# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""A recording session's episode records: Arena's JSONL, one row per take, with the string draws."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("isaaclab_arena.recording.episode_recorder_manager", reason="needs IsaacLab-Arena on the path")

from isaaclab_arena.recording.common_terms import (  # noqa: E402
    CoreEpisodeRecorderTermCfg,
    VariationEpisodeRecorderTermCfg,
)
from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderManager, EpisodeRecorderTermCfg  # noqa: E402
from isaaclab_arena.utils.configclass import make_configclass  # noqa: E402
from isaaclab_arena.variations.variation_recorder import VariationRecorder  # noqa: E402

from dexbench_arena.layouts.db_r2_layout import (  # noqa: E402
    fc_260821_r2_stocking,
)
from dexbench_arena.environments.manipulation.scene_environment_base import (  # noqa: E402
    exported_demos,
    record_exported_demo,
    record_terminated_by,
)
from dexbench_arena.variations.db_r2_lane_variation import R2LaneVariation  # noqa: E402

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts/imitation_learning"
sys.path.insert(0, str(SCRIPTS))


def _load(name):
    spec = importlib.util.spec_from_file_location(name.lstrip("_"), SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


metadata = importlib.import_module("dexbench_arena.recording.rerecord_metadata")
wrapper = _load("_dexbench_arena_script")


class _RecorderManager:
    """Isaac Lab's recorder manager as record_demos.py drives it (EXPORT_SUCCEEDED_ONLY)."""

    def __init__(self):
        self.exported_successful_episode_count = 0
        self.success = False

    def export_episodes(self, env_ids=None, demo_ids=None):
        if self.success:
            self.exported_successful_episode_count += 1


class _Env:
    """The parts of an Arena env the episode recorder's terms and the record_demos hook read."""

    num_envs = 1

    def __init__(self, variation_recorder, dataset_dir="."):
        recorders = SimpleNamespace(dataset_export_dir_path=str(dataset_dir), dataset_filename="session_001")
        self.cfg = SimpleNamespace(seed=7, recorders=recorders)
        self.recorder_manager = _RecorderManager()
        self.sim = SimpleNamespace(is_playing=lambda: True)
        self.termination_manager = SimpleNamespace(active_terms=[])
        self.variation_recorder = variation_recorder
        self.variation_replay_scheduler = None
        self.episode_length_buf = torch.zeros(1, dtype=torch.long)
        self.episode = 0

    def get_episode_index(self, env_id):
        return self.episode

    def get_language_instruction(self):
        return "retrieve the drink"

    def close(self):
        self.closed = getattr(self, "closed", 0) + 1


def test_a_session_writes_one_row_per_take_with_its_string_draw(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    variation = R2LaneVariation(fc_260821_r2_stocking(lane_draw=[5, 6]), "r2_target_drink")
    recorder = VariationRecorder()
    recorder.attach({"r2_target_drink": [variation]})
    env = _Env(recorder, tmp_path)
    recorder.bind_env(env)
    cfg = make_configclass("EpisodeRecorderManagerCfg", [
        ("core", EpisodeRecorderTermCfg, CoreEpisodeRecorderTermCfg()),
        ("variations", EpisodeRecorderTermCfg, VariationEpisodeRecorderTermCfg()),
    ])()
    env.episode_recorder = EpisodeRecorderManager(cfg, env)

    wrapper._write_episode_records_of_recording(env)
    path = tmp_path / "session_001.episodes.jsonl"

    drawn = []
    for length in (12, 30):
        drawn += variation.sampler.sample(num_samples=1, choices=variation.plan.labels, env_ids=torch.tensor([0]))
        env.episode_length_buf[0] = length
        if length == 12:
            env.episode_recorder.record_pre_reset([0])
            env.episode += 1
    env.close()
    env.close()

    assert env.closed == 2
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["episode_in_env"] for row in rows] == [0, 1]
    assert [row["episode_length"] for row in rows] == [12, 30]
    assert [row["variations"]["r2_target_drink.lane"] for row in rows] == list(drawn)
    assert all(isinstance(label, str) for label in drawn)
    assert {row["job_name"] for row in rows} == {"record_demos"}
    assert {row["success"] for row in rows} == {None}


def test_each_row_names_the_demo_its_take_was_exported_as(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    env = _Env(None, tmp_path)
    cfg = make_configclass("EpisodeRecorderManagerCfg", [
        ("core", EpisodeRecorderTermCfg, CoreEpisodeRecorderTermCfg()),
        ("exported_demo", EpisodeRecorderTermCfg, EpisodeRecorderTermCfg(func=record_exported_demo)),
    ])()
    env.episode_recorder = EpisodeRecorderManager(cfg, env)
    wrapper._write_episode_records_of_recording(env)

    for success in (False, True):
        env.recorder_manager.success = success
        env.recorder_manager.export_episodes([0])
        env.episode_recorder.record_pre_reset([0])
        env.episode += 1
    env.close()

    rows = [json.loads(line) for line in (tmp_path / "session_001.episodes.jsonl").read_text().splitlines()]
    assert [row["exported_demo"] for row in rows] == [None, "demo_0", None]
    assert [row["episode_in_env"] for row in rows] == [0, 1, 2]
    assert exported_demos(env) == {}
    assert record_exported_demo(_Env(None), 0) == {}


def test_each_row_names_the_terms_that_ended_its_episode():
    """Env 0 ran to its cap, env 1 dropped a sheet, env 2 is still running: each row names only its own terms."""
    terms = {"time_out": torch.tensor([True, False, False]), "sheet_dropped": torch.tensor([False, True, False])}
    env = SimpleNamespace(termination_manager=SimpleNamespace(active_terms=list(terms), get_term=terms.__getitem__))
    assert [record_terminated_by(env, env_id) for env_id in (0, 1, 2)] == [
        {"terminated_by": ["time_out"]}, {"terminated_by": ["sheet_dropped"]}, {"terminated_by": []}]
