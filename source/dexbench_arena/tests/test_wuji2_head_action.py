# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""head_action gives dexbench_rby1_wuji2 a direct head term after its IK columns and keeps waist IK."""

from types import SimpleNamespace

import pytest

from dexbench_arena.embodiments.rby1_wuji2.constants import RBY1_WUJI2_HEAD_JOINT_NAMES, RBY1_WUJI2_TORSO_JOINT_NAMES


def test_head_action_moves_the_head_out_of_pink(simulation_app):
    from dexbench_arena.embodiments.rby1_wuji2.rby1_wuji2 import DexBenchRby1Wuji2Embodiment

    plain = DexBenchRby1Wuji2Embodiment(waist_ik=True)
    head = DexBenchRby1Wuji2Embodiment(waist_ik=True, head_action=True)
    assert not hasattr(plain.action_config, "head")
    assert set(RBY1_WUJI2_HEAD_JOINT_NAMES) <= set(plain.action_config.upper_body_ik.pink_controlled_joint_names)
    pink = head.action_config.upper_body_ik.pink_controlled_joint_names
    assert not set(RBY1_WUJI2_HEAD_JOINT_NAMES) & set(pink)
    assert set(RBY1_WUJI2_TORSO_JOINT_NAMES) <= set(pink)
    assert list(head.action_config.head.joint_names) == list(RBY1_WUJI2_HEAD_JOINT_NAMES)
    assert set(pink) | set(RBY1_WUJI2_HEAD_JOINT_NAMES) == set(plain.action_config.upper_body_ik.pink_controlled_joint_names)
    for value in ("false", "true", 1):
        with pytest.raises(ValueError, match="head_action must be a bool"):
            DexBenchRby1Wuji2Embodiment(head_action=value)
    assert DexBenchRby1Wuji2Embodiment(head_action=None).head_action is False


def test_mimic_refuses_head_action_columns(simulation_app):
    import torch

    from dexbench_arena.embodiments.rby1_wuji2.rby1_wuji2 import Rby1Wuji2MimicEnv

    env = SimpleNamespace(action_manager=SimpleNamespace(active_terms=["upper_body_ik", "head"]))
    env._refuse_head_action = lambda: Rby1Wuji2MimicEnv._refuse_head_action(env)
    actions = torch.zeros((1, 56))
    for read in (Rby1Wuji2MimicEnv.action_to_target_eef_pose, Rby1Wuji2MimicEnv.actions_to_gripper_actions):
        with pytest.raises(ValueError, match="head_action"):
            read(env, actions)
    env.action_manager.active_terms = ["upper_body_ik"]
    assert set(Rby1Wuji2MimicEnv.actions_to_gripper_actions(env, torch.zeros((1, 54)))) == {"left", "right"}
