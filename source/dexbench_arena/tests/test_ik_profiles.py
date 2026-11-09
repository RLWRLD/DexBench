# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""The RBY1-M IK profile reaches the Pink controller through the embodiment constructor."""

from __future__ import annotations

import pytest

pytest.importorskip("isaaclab.sim", exc_type=ImportError)

from dexbench_arena.embodiments.rby1_wuji2.rby1_wuji2 import (  # noqa: E402
    DexBenchRby1Wuji2Embodiment,
    DexBenchRby1Wuji2JointEmbodiment,
)
from dexbench_arena.embodiments.rby1m.ik import RBY1M_IK_GENERATION, RBY1M_IK_TELEOP  # noqa: E402


def _wrist_tasks(embodiment):
    controller = embodiment.action_config.upper_body_ik.controller
    return [task for task in controller.variable_input_tasks if hasattr(task, "frame")]


@pytest.mark.parametrize(("name", "profile"), [("teleop", RBY1M_IK_TELEOP), ("generation", RBY1M_IK_GENERATION)])
def test_the_named_profile_sets_both_wrist_tasks(name, profile):
    embodiment = DexBenchRby1Wuji2Embodiment(ik_profile=name)
    assert embodiment.ik_profile == profile
    tasks = _wrist_tasks(embodiment)
    assert len(tasks) == 2
    for task in tasks:
        assert (task.gain, task.lm_damping, task.position_cost) == (profile.gain, profile.lm_damping, profile.position_cost)


def test_the_default_is_the_teleop_profile_and_unknown_names_are_named():
    assert DexBenchRby1Wuji2Embodiment().ik_profile == RBY1M_IK_TELEOP
    with pytest.raises(ValueError, match="generation"):
        DexBenchRby1Wuji2Embodiment(ik_profile="fast")


def test_a_joint_embodiment_does_not_take_a_profile():
    with pytest.raises(TypeError, match="ik_profile"):
        DexBenchRby1Wuji2JointEmbodiment(ik_profile="generation")

