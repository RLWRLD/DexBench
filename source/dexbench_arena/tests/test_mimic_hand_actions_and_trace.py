# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dexbench_arena.embodiments.hand_actions import (
    merge_bimanual_hand_actions,
    split_bimanual_hand_actions,
)
from dexbench_arena.embodiments.rby1_wuji2.constants import (
    RBY1_WUJI2_HAND_JOINT_ACTION_NAMES,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_script(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("hand_offset", (14, 16), ids=("openxr", "joint-space"))
def test_mimic_round_trips_stage_major_layout(hand_offset):
    hand_names = RBY1_WUJI2_HAND_JOINT_ACTION_NAMES
    actions = torch.arange(hand_offset + len(hand_names)).reshape(1, -1)

    split = split_bimanual_hand_actions(actions, hand_names, hand_offset)

    for side, prefixes in (("left", "left_"), ("right", "right_")):
        expected = [
            hand_offset + index
            for index, name in enumerate(hand_names)
            if name.startswith(prefixes)
        ]
        assert split[side].tolist() == [expected]

    rebuilt_hand = merge_bimanual_hand_actions(split["left"], split["right"], hand_names)
    rebuilt_action = torch.cat((actions[:, :hand_offset], rebuilt_hand), dim=-1)
    torch.testing.assert_close(rebuilt_action, actions)


@pytest.mark.parametrize("inactive_side", ("left", "right"))
def test_single_arm_rebuild_preserves_inactive_hand(inactive_side):
    hand_names = RBY1_WUJI2_HAND_JOINT_ACTION_NAMES
    current_action = torch.arange(len(hand_names)).reshape(1, -1)
    current_hands = split_bimanual_hand_actions(current_action, hand_names, 0)
    active_side = "right" if inactive_side == "left" else "left"
    generated_hands = {
        active_side: torch.full_like(current_hands[active_side], -1),
        inactive_side: current_hands[inactive_side],
    }

    rebuilt = merge_bimanual_hand_actions(
        generated_hands["left"],
        generated_hands["right"],
        hand_names,
    )
    rebuilt_hands = split_bimanual_hand_actions(rebuilt, hand_names, 0)

    torch.testing.assert_close(rebuilt_hands[inactive_side], current_hands[inactive_side])
    torch.testing.assert_close(
        rebuilt_hands[active_side],
        torch.full_like(current_hands[active_side], -1),
    )


def test_native_joint_csv_records_the_episode_step(tmp_path):
    """The real writer exposes reset boundaries to the smoothness analyzer."""
    wrapper = _load_script(
        "dexbench_script_joint_csv_for_test",
        "scripts/imitation_learning/_dexbench_arena_script.py",
    )
    robot = SimpleNamespace(
        joint_names=["joint_0"],
        data=SimpleNamespace(
            joint_pos=torch.zeros((1, 1)),
            joint_vel=torch.zeros((1, 1)),
            joint_pos_target=torch.zeros((1, 1)),
        ),
    )
    env = SimpleNamespace(
        scene={"robot": robot},
        step_dt=1.0 / 30.0,
        common_step_counter=0,
        episode_length_buf=torch.zeros(1, dtype=torch.long),
    )
    path = tmp_path / "joints.csv"
    logger = wrapper._JointCsvLogger(env, str(path))

    for global_step, episode_step in enumerate((1, 2, 0, 1), start=1):
        env.common_step_counter = global_step
        env.episode_length_buf[0] = episode_step
        logger.publish()
    logger.close()

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames[:3] == ["t", "step", "episode_step"]
        assert [int(row["episode_step"]) for row in reader] == [1, 2, 0, 1]
