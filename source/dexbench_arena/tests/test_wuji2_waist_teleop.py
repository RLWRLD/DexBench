# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Core contracts for opt-in Wuji2 waist teleop."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest


def test_waist_constructor_and_shared_cli(simulation_app):
    from dexbench_arena.embodiments.rby1_wuji2.rby1_wuji2 import DexBenchRby1Wuji2Embodiment
    from dexbench_arena.embodiments.rby1_wuji2.constants import RBY1_WUJI2_TORSO_JOINT_NAMES
    from dexbench_arena.environments.manipulation.e2_logistics_environment import (
        DexBenchLogisticsCenterE2Environment,
    )

    default = DexBenchRby1Wuji2Embodiment()
    factory = DexBenchLogisticsCenterE2Environment()
    parser = argparse.ArgumentParser()
    factory.add_cli_args(parser)
    for flag in ("--waist_ik", "--waist-ik"):
        args = parser.parse_args([
            "--embodiment=dexbench_rby1_wuji2", "--anchor", "A1_CAGE",
            flag, "--waist_ik_max_lead_rad", "0.075",
        ])
        enabled = factory.get_env(args).embodiment
        ik = enabled.action_config.upper_body_ik
        assert ik.waist_ik_enabled and enabled.waist_ik
        assert ik.waist_ik_max_lead_rad == 0.075
        assert ik.pink_controlled_joint_names[-6:] == list(RBY1_WUJI2_TORSO_JOINT_NAMES)
        assert ik.controller.num_hand_joints + 14 == 54
    assert not default.action_config.upper_body_ik.waist_ik_enabled
    assert len(ik.pink_controlled_joint_names) == len(default.action_config.upper_body_ik.pink_controlled_joint_names) + 6
    for value in (float("nan"), 0, -0.1, "0.05", True):
        with pytest.raises(ValueError, match="waist_ik_max_lead_rad"):
            DexBenchRby1Wuji2Embodiment(waist_ik=True, waist_ik_max_lead_rad=value)
    with pytest.raises(ValueError, match="waist_ik must be a bool"):
        DexBenchRby1Wuji2Embodiment(waist_ik="false")
    defaulted = DexBenchRby1Wuji2Embodiment(waist_ik=None, waist_ik_max_lead_rad=None)
    assert not defaulted.waist_ik and defaulted.waist_ik_max_lead_rad == 0.05
    zero = parser.parse_args(["--embodiment=dexbench_rby1_wuji2", "--anchor", "A1_CAGE", "--waist_ik", "--waist_ik_max_lead_rad", "0"])
    with pytest.raises(ValueError, match="waist_ik_max_lead_rad must be finite and positive"):
        factory.get_env(zero)
    args.embodiment = "dexbench_rby1_wuji2_joint"
    with pytest.raises(ValueError, match="does not take --waist_ik"):
        factory.get_env(args)


def test_waist_ik_limits_command_speed_and_tracking_error(monkeypatch):
    pytest.importorskip("isaaclab")
    import torch
    from isaaclab.envs.mdp.actions.pink_task_space_actions import PinkInverseKinematicsAction

    from dexbench_arena.embodiments.rby1_wuji2.constants import RBY1_WUJI2_TORSO_JOINT_NAMES
    from dexbench_arena.embodiments.rby1_wuji2.rby1_wuji2 import Rby1Wuji2PinkInverseKinematicsAction

    for hook in ("_compute_ik_solutions", "_initialize_joint_info"):
        assert hasattr(PinkInverseKinematicsAction, hook), f"Upstream Pink action removed {hook}"

    desired = {"value": 0.2}
    monkeypatch.setattr(
        PinkInverseKinematicsAction, "_compute_ik_solutions", lambda self: torch.full((2, 6), desired["value"])
    )
    measured = torch.zeros((2, 6))

    def upstream_init(self, cfg, env):
        """What upstream's constructor leaves for the body action once the joints are resolved."""
        self.cfg = cfg
        self._env = env
        self._sim_dt = 1 / 60
        self._asset = SimpleNamespace(data=SimpleNamespace(joint_pos=SimpleNamespace(torch=measured)))
        self._isaaclab_controlled_joint_names = list(cfg.pink_controlled_joint_names)
        self._isaaclab_controlled_joint_ids = list(range(len(cfg.pink_controlled_joint_names)))

    monkeypatch.setattr(PinkInverseKinematicsAction, "__init__", upstream_init)
    env = SimpleNamespace(device="cpu")

    def action_cfg(joint_names, waist_ik_enabled):
        return SimpleNamespace(
            pink_controlled_joint_names=joint_names, waist_ik_enabled=waist_ik_enabled, waist_ik_max_lead_rad=0.05
        )

    fixed_torso = Rby1Wuji2PinkInverseKinematicsAction(action_cfg(["left_arm_0"], False), env)
    assert torch.allclose(fixed_torso._compute_ik_solutions(), torch.full((2, 6), desired["value"]))

    action = Rby1Wuji2PinkInverseKinematicsAction(action_cfg(list(RBY1_WUJI2_TORSO_JOINT_NAMES), True), env)

    max_step = 0.8 / 60
    assert torch.allclose(action._compute_ik_solutions(), torch.full((2, 6), max_step))
    measured.fill_(0.002)
    assert torch.allclose(action._compute_ik_solutions(), torch.full((2, 6), 2 * max_step))

    for _ in range(5):
        limited = action._compute_ik_solutions()
    assert torch.allclose(limited, torch.full((2, 6), 0.052))

    desired["value"] = 0.052
    measured.fill_(0.103)
    assert torch.allclose(action._compute_ik_solutions(), torch.full((2, 6), 0.053))

    desired["value"] = 0.2
    measured.fill_(0.2)
    assert torch.allclose(action._compute_ik_solutions(), torch.full((2, 6), 0.15))

    monkeypatch.setattr(PinkInverseKinematicsAction, "reset", lambda self, env_ids=None: None)
    action.reset([0])
    limited = action._compute_ik_solutions()
    assert torch.allclose(limited[0], measured[0])
    assert torch.allclose(limited[1], torch.full((6,), 0.15 + max_step))
    action.reset()
    assert torch.allclose(action._compute_ik_solutions(), measured)

    action.reset()
    desired["value"] = 0.3
    with torch.inference_mode():
        action._compute_ik_solutions()
    action.reset([1])
    limited = action._compute_ik_solutions()
    assert torch.allclose(limited[0], torch.full((6,), 0.2 + 2 * max_step))
    assert torch.allclose(limited[1], torch.full((6,), 0.2 + max_step))


def test_waist_survives_generation_profile(simulation_app):
    """The waist settings land on the IK term the profile produced, not the default one."""
    from dexbench_arena.embodiments.rby1_wuji2.rby1_wuji2 import DexBenchRby1Wuji2Embodiment
    from dexbench_arena.embodiments.rby1_wuji2.constants import RBY1_WUJI2_TORSO_JOINT_NAMES

    ik = DexBenchRby1Wuji2Embodiment(ik_profile="generation", waist_ik=True).action_config.upper_body_ik
    assert ik.waist_ik_enabled
    assert set(RBY1_WUJI2_TORSO_JOINT_NAMES) <= set(ik.pink_controlled_joint_names)
    assert ik.controller.variable_input_tasks[0].gain == 1.0
    assert ik.controller.urdf_path


@pytest.mark.parametrize("arm_cost", [0.5, 8.0, 0.0])
def test_waist_costs_follow_joint_order_and_arm_cost(arm_cost):
    pin = pytest.importorskip("pinocchio")
    pink = pytest.importorskip("pink")
    pytest.importorskip("isaaclab")
    from dexbench_arena.embodiments.rby1m.waist_ik import _joint_costs

    model = pin.Model()
    names = ["torso_5", "arm_0", "torso_3", "torso_0", "torso_4", "arm_1", "torso_2", "torso_1"]
    for name in names:
        model.addJoint(0, pin.JointModelRZ(), pin.SE3.Identity(), name)
    config = pink.Configuration(model, model.createData(), pin.neutral(model))
    costs = _joint_costs(config, arm_cost, 16.0)
    assert costs.tolist() == [
        arm_cost * (16 if name.startswith("torso_") and name != "torso_5" else 1) for name in names
    ]


def test_waist_costs_reject_floating_base():
    pin = pytest.importorskip("pinocchio")
    pink = pytest.importorskip("pink")
    pytest.importorskip("isaaclab")
    from dexbench_arena.embodiments.rby1m.waist_ik import _joint_costs

    model = pin.Model()
    model.addJoint(0, pin.JointModelFreeFlyer(), pin.SE3.Identity(), "root_joint")
    config = pink.Configuration(model, model.createData(), pin.neutral(model))
    with pytest.raises(ValueError, match="fixed-base"):
        _joint_costs(config, 0.5, 16)
