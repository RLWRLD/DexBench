# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""The Wuji2 fingertip pads' contact force observation: a pressed pad reads a force, a free one none (Kit tier)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.usefixtures("simulation_app")

R2 = ["dexbench_retail_store_e1_environment", "--task", "retrieve_beverage_from_clutter", "--anchor", "Fridge"]
EMBODIMENT = ["--embodiment", "dexbench_rby1_wuji2_joint", "--arm_mode", "dual_arm"]
PRESSED = "right_index_finger_tip_sensor_frame"


@pytest.fixture
def restore_builder(monkeypatch):
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    monkeypatch.setattr(ArenaEnvBuilder, "compose_manager_cfg", ArenaEnvBuilder.compose_manager_cfg)
    monkeypatch.setattr(ArenaEnvBuilder, "make_registered", ArenaEnvBuilder.make_registered)
    monkeypatch.setattr(ArenaEnvBuilder, "get_entry_point", ArenaEnvBuilder.get_entry_point)


def test_a_pressed_pad_reads_its_contact_and_a_free_hand_reads_none(restore_builder):
    import gymnasium as gym
    import torch
    import warp as wp

    from dexbench_arena.cli import get_arena_builder_from_cli, get_dexbench_arena_cli_parser
    from dexbench_arena.embodiments.rby1_wuji2.constants import RBY1_WUJI2_FINGERTIP_PAD_BODY_NAMES

    builder = get_arena_builder_from_cli(get_dexbench_arena_cli_parser().parse_args([*R2, *EMBODIMENT]))
    name, cfg, kwargs = builder.build_registered()
    cfg.actions.joint_pos.use_default_offset = True
    cfg.recorders = cfg.metrics = None
    for key, term in vars(cfg.terminations).items():
        if hasattr(term, "func"):
            setattr(cfg.terminations, key, None)
    env = gym.make(name, cfg=cfg, **kwargs).unwrapped
    try:
        assert [key for key in env.scene.sensors if "contact" in key] == ["fingertip_contact"]
        env.reset()
        hold = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
        for _ in range(5):
            obs, *_ = env.step(hold)
        free = obs["policy"]["fingertip_contact_force"]
        assert free.shape == (env.num_envs, 10)
        assert float(free.max()) < 0.5, dict(zip(RBY1_WUJI2_FINGERTIP_PAD_BODY_NAMES, free[0].tolist()))

        def as_torch(value):
            return value.torch if hasattr(value, "torch") else wp.to_torch(value)

        robot, can = env.scene["robot"], env.scene["r2_target_drink"]
        pad = robot.body_names.index(PRESSED)
        pressed = []
        for _ in range(4):
            pose = as_torch(can.data.root_pose_w).clone()
            pose[:, :3] = as_torch(robot.data.body_pos_w)[:, pad] + torch.tensor([0.0, 0.0, -0.03], device=env.device)
            can.write_root_pose_to_sim_index(root_pose=pose)
            can.write_root_velocity_to_sim_index(root_velocity=torch.zeros(env.num_envs, 6, device=env.device))
            obs, *_ = env.step(hold)
            pressed.append(obs["policy"]["fingertip_contact_force"][0].clone())
        pressed = torch.stack(pressed).max(dim=0).values
        readings = dict(zip(RBY1_WUJI2_FINGERTIP_PAD_BODY_NAMES, pressed.tolist()))
        assert readings[PRESSED] > 1.0, readings
        assert all(value < 0.5 for key, value in readings.items() if key.startswith("left_")), readings
    finally:
        env.close()
