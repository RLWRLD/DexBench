# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Stored subtask boundaries drive ``get_subtask_term_signals`` during ``--auto`` replay."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from dexbench_arena.embodiments.common.mimic_boundaries import ReplaysSubtaskBoundaries  # noqa: E402


class _LabMimicBase:
    """What the mixin relies on from Isaac Lab's Mimic env, without a simulator."""

    num_envs = 1
    device = "cpu"

    def __init__(self, cfg):
        self.cfg = cfg
        self.common_step_counter = 0

    def reset_to(self, state, env_ids=None, seed=None, is_relative=False):
        return "reset"

    def get_subtask_term_signals(self, env_ids=None):
        raise NotImplementedError


class _Env(ReplaysSubtaskBoundaries, _LabMimicBase):
    pass


def _flags(env):
    return {name: bool(value) for name, value in env.get_subtask_term_signals().items()}


def test_without_a_table_the_refusal_names_the_flag():
    with pytest.raises(NotImplementedError, match="--mimic_boundaries"):
        _Env(SimpleNamespace()).get_subtask_term_signals()


def test_an_env_with_its_own_signal_answers_without_a_table():
    class _OwnSignal(_Env):
        def get_subtask_term_signals(self, env_ids=None):
            if self.subtask_boundaries is not None:
                return super().get_subtask_term_signals(env_ids)
            return {"own": torch.ones(1, dtype=torch.bool)}

    assert _flags(_OwnSignal(SimpleNamespace())) == {"own": True}


def test_marks_are_replayed_by_episode_and_step_from_the_reset(tmp_path):
    table = tmp_path / "boundaries.json"
    table.write_text(json.dumps([{"grasp": 2, "extracted": 4}, None, {"grasp": 1}]))
    env = _Env(SimpleNamespace(subtask_boundaries_path=str(table), subtask_boundaries_episode_names=["demo_0", "demo_1", "demo_2"]))

    env.common_step_counter = 37
    env.reset_to(None)
    seen = []
    for _ in range(5):
        seen.append(_flags(env))
        _flags(env)
        env.common_step_counter += 1
    assert [f["grasp"] for f in seen] == [False, False, True, True, True]
    assert [f["extracted"] for f in seen] == [False, False, False, False, True]

    env.reset_to(None)
    assert _flags(env) == {"grasp": False, "extracted": False}

    env.reset_to(None)
    env.common_step_counter += 1
    assert _flags(env) == {"grasp": True, "extracted": False}


def test_each_replayed_demo_gets_its_own_row_past_ten_demos(tmp_path):
    """annotate_demos.py replays demo_0, demo_1, demo_10, demo_11, demo_2, ... (h5py's order);
    the table tools write row i for the i-th demo in numeric order."""
    table = tmp_path / "boundaries.json"
    table.write_text(json.dumps([{"grasp": row} for row in range(12)]))
    replay_order = sorted(f"demo_{n}" for n in range(12))
    env = _Env(SimpleNamespace(subtask_boundaries_path=str(table), subtask_boundaries_episode_names=replay_order))

    marks = {}
    for name in replay_order:
        env.reset_to(None)
        start = env.common_step_counter
        while not _flags(env)["grasp"]:
            env.common_step_counter += 1
        marks[name] = env.common_step_counter - start
        env.common_step_counter += 1
    assert marks == {f"demo_{n}": n for n in range(12)}

    renamed = _Env(SimpleNamespace(subtask_boundaries_path=str(table),
                                   subtask_boundaries_episode_names=[*replay_order[:-1], "take_7"]))
    with pytest.raises(ValueError, match="demo_<int>"):
        renamed.get_subtask_term_signals()
