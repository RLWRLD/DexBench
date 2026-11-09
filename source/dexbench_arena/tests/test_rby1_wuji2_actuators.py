# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Default RBY1-Wuji2 gains, the explicit saved-grasp override and a task's power-driver gains remain isolated."""

from __future__ import annotations

import dataclasses

import pytest

pytest.importorskip("isaaclab.sim", exc_type=ImportError)

from isaaclab_arena.utils.physics_backend import PhysicsBackend  # noqa: E402

from dexbench_arena.embodiments.rby1_wuji2.constants import RBY1_WUJI2_RETIRED_GAIN_VARIABLES  # noqa: E402
from dexbench_arena.embodiments.rby1_wuji2.rby1_wuji2 import (  # noqa: E402
    DexBenchRby1Wuji2Embodiment,
    DexBenchRby1Wuji2JointEmbodiment,
)

NEWTON_AT_FF4A93BE = {
    "head": {"damping": 110.0, "effort_limit_sim": 50.0, "stiffness": 300.0, "velocity_limit_sim": 10.0},
    "torso": {"damping": 200.0, "effort_limit": 100.0, "effort_limit_sim": 10000.0, "stiffness": 800.0,
              "velocity_limit": 100.0, "velocity_limit_sim": 1000000.0},
    "right_arm": {"armature": 0.0, "damping": 15.0, "effort_limit": 100.0, "stiffness": 400.0,
                  "velocity_limit": 100.0, "velocity_limit_sim": 1000000.0},
    "left_arm": {"armature": 0.0, "damping": 15.0, "effort_limit": 100.0, "stiffness": 400.0,
                 "velocity_limit": 100.0, "velocity_limit_sim": 1000000.0},
    "right_hand": {"armature": 0.01, "damping": 5.0, "effort_limit_sim": 5.0, "stiffness": 100.0, "velocity_limit_sim": 12.0},
    "left_hand": {"armature": 0.01, "damping": 5.0, "effort_limit_sim": 5.0, "stiffness": 100.0, "velocity_limit_sim": 12.0},
    "wheels": {"damping": 0.0, "effort_limit_sim": 1.0, "stiffness": 0.0, "velocity_limit_sim": 0.0},
}

EMBODIMENTS = [DexBenchRby1Wuji2JointEmbodiment, DexBenchRby1Wuji2Embodiment]


@pytest.fixture
def no_gain_variables(monkeypatch):
    for name in RBY1_WUJI2_RETIRED_GAIN_VARIABLES:
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("embodiment_class", EMBODIMENTS, ids=lambda cls: cls.__name__)
def test_the_one_profile_is_what_newton_ran(no_gain_variables, embodiment_class):
    actuators = embodiment_class().scene_config.robot.actuators
    assert set(actuators) == set(NEWTON_AT_FF4A93BE)
    for group, expected in NEWTON_AT_FF4A93BE.items():
        cfg = actuators[group]
        assert type(cfg).__name__ == "ImplicitActuatorCfg", group
        values = {
            field.name: getattr(cfg, field.name)
            for field in dataclasses.fields(cfg)
            if field.name not in ("class_type", "joint_names_expr") and getattr(cfg, field.name) is not None
        }
        assert values == expected, group


def test_a_retired_gain_variable_is_refused_even_when_empty(no_gain_variables, monkeypatch):
    monkeypatch.setenv("DEXBENCH_ARM_STIFFNESS", "")
    with pytest.raises(ValueError, match="DEXBENCH_ARM_STIFFNESS set, but the RBY1-Wuji2 actuator gains are not read"):
        DexBenchRby1Wuji2JointEmbodiment()


def test_saved_support_grasp_changes_only_left_finger_drive_and_preserves_action_order(no_gain_variables):
    default = DexBenchRby1Wuji2JointEmbodiment()
    supported = DexBenchRby1Wuji2JointEmbodiment(support_hand_mode="init_q")
    action = supported.action_config.joint_pos
    assert action.joint_names == default.action_config.joint_pos.joint_names
    assert len(action.joint_names) == 62
    assert len(action.grasp_joint_names) == 20
    assert all(name.startswith("left_") for name in action.grasp_joint_names)
    thumb = action.grasp_joint_names.index("left_thumb_cmc_abd")
    abduction = action.grasp_joint_names.index("left_index_finger_mcp_abd")
    flexion = action.grasp_joint_names.index("left_index_finger_pip")
    assert [action.closing_directions[i] for i in (thumb, abduction, flexion)] == [-1, 0, 1]
    assert [action.finger_columns[i] for i in (thumb, abduction, flexion)] == [0, 1, 1]
    assert action.grasp_profile.anchor_to_reset
    assert action.grasp_profile.preload_rad == .003
    assert action.grasp_profile.friction == 1.1
    assert action.grasp_profile.rolling_friction_m == .006
    assert action.adaptive_profile.hold_preload_rad == .001
    expected = dict(NEWTON_AT_FF4A93BE["left_hand"], stiffness=8., damping=.3, effort_limit_sim=.25)
    for field, value in expected.items():
        assert getattr(supported.scene_config.robot.actuators["left_hand"], field) == value
    for group in default.scene_config.robot.actuators:
        if group != "left_hand":
            assert dataclasses.asdict(supported.scene_config.robot.actuators[group]) == dataclasses.asdict(default.scene_config.robot.actuators[group])
    assert DexBenchRby1Wuji2JointEmbodiment().scene_config.robot.actuators["left_hand"].stiffness == 100.
    with pytest.raises(ValueError, match="Unknown support_hand_mode"):
        DexBenchRby1Wuji2JointEmbodiment(support_hand_mode="current_q")


@pytest.mark.parametrize("embodiment_type", [DexBenchRby1Wuji2Embodiment, DexBenchRby1Wuji2JointEmbodiment])
def test_power_driver_gains_do_not_mutate_other_robot_instances(embodiment_type):
    loaded = embodiment_type()
    loaded.configure_physics_backend(PhysicsBackend.NEWTON)
    loaded.configure_power_driver_actuators(loaded.scene_config.robot)
    actuators = loaded.scene_config.robot.actuators
    assert (actuators["right_arm"].stiffness, actuators["right_arm"].damping) == (400., 40.)
    assert (actuators["torso"].stiffness, actuators["torso"].damping) == (5000., 400.)
    assert (actuators["right_hand"].stiffness, actuators["right_hand"].damping) == (100., 5.)
    default = embodiment_type()
    default.configure_physics_backend(PhysicsBackend.NEWTON)
    assert default.scene_config.robot.actuators["right_arm"].damping == 15.
    assert default.scene_config.robot.actuators["torso"].stiffness == 800.
    with pytest.raises(TypeError):
        embodiment_type(unknown_actuator_setting=True)


@pytest.mark.parametrize("embodiment_class", EMBODIMENTS, ids=lambda cls: cls.__name__)
def test_a_build_on_another_backend_than_newton_is_refused(no_gain_variables, embodiment_class):
    """Arena tells the embodiment the builder's backend on every build path; an upstream environment
    (PhysX by default) built with this robot is refused there, as a DexBench environment refuses it itself."""
    from isaaclab_arena.utils.physics_backend import PhysicsBackend

    embodiment_class().configure_physics_backend(PhysicsBackend.NEWTON)
    with pytest.raises(ValueError, match="runs on Newton only, and this build resolved physx"):
        embodiment_class().configure_physics_backend(PhysicsBackend.PHYSX)
