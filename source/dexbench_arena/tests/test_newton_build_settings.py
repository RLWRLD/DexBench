# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""An environment built the normal way runs the Newton settings collection ran with (Kit tier)."""

from __future__ import annotations

import pytest

from dexbench_arena.teleop.newton_solver import PHYSICS_OPTIONS

pytestmark = pytest.mark.usefixtures("simulation_app")

EMBODIMENT = ["--embodiment", "dexbench_rby1_wuji2_joint", "--arm_mode", "dual_arm"]
SCENES = {
    "r2": ["dexbench_retail_v1_environment", "--task", "retrieve_beverage_from_clutter", "--anchor", "Fridge"],
    "l2": ["dexbench_logistics_center_e2_environment", "--task", "lift_to_conveyor", "--anchor", "A1_CAGE"],
    "h1": ["dexbench_hotel_fb_e4_environment", "--task", "stack_plates"],
}


def _builder(presets, scene):
    from dexbench_arena.cli import get_arena_builder_from_cli, get_dexbench_arena_cli_parser

    argv = (["--presets", presets] if presets else []) + SCENES[scene] + EMBODIMENT
    return get_arena_builder_from_cli(get_dexbench_arena_cli_parser().parse_args(argv))


def _compose(builder):
    composed = builder.compose_manager_cfg()
    return composed[0] if isinstance(composed, tuple) else composed


@pytest.fixture
def restore_builder(monkeypatch):
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    monkeypatch.setattr(ArenaEnvBuilder, "compose_manager_cfg", ArenaEnvBuilder.compose_manager_cfg)
    monkeypatch.setattr(ArenaEnvBuilder, "make_registered", ArenaEnvBuilder.make_registered)
    monkeypatch.setattr(ArenaEnvBuilder, "get_entry_point", ArenaEnvBuilder.get_entry_point)


@pytest.mark.parametrize("scene", sorted(SCENES))
@pytest.mark.parametrize("presets", [None, "newton"], ids=["default", "presets-newton"])
def test_the_robot_and_the_scene_get_their_newton_profiles_on_every_scene(restore_builder, presets, scene):
    from dexbench_arena.embodiments.rby1_wuji2.rby1_wuji2 import RBY1_WUJI2_CFG

    builder = _builder(presets, scene)
    env_cfg = _compose(builder)
    assert type(env_cfg.sim.physics).__name__ == "NewtonCfg"
    assert env_cfg.scene.robot.actuators == RBY1_WUJI2_CFG.actuators
    if scene != "l2":
        assert env_cfg.sim.physics.num_substeps >= 8
    settings = builder.arena_env.dexbench_settings
    options = settings.factory.newton_solver_options_for(settings.cfg)
    for key, value in options.items():
        owner = env_cfg.sim.physics if key in PHYSICS_OPTIONS else env_cfg.sim.physics.solver_cfg
        assert getattr(owner, key) == value, key


def test_r2_on_newton_sits_through_reset_and_settling_without_terminating(restore_builder):
    """R2 held still on Newton: no termination fires through reset and settling, and the target drink comes to rest."""
    import gymnasium as gym
    import torch
    import warp as wp

    builder = _builder(None, "r2")
    name, cfg, kwargs = builder.build_registered()
    cfg.actions.joint_pos.use_default_offset = True
    cfg.recorders = cfg.metrics = None
    env = gym.make(name, cfg=cfg, **kwargs).unwrapped
    try:
        env.reset()
        actions = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
        fired = []
        for step in range(90):
            env.step(actions)
            for term in env.termination_manager.active_terms:
                if term != "time_out" and bool(env.termination_manager.get_term(term).any()):
                    fired.append((step, term))
        assert not fired, f"terminations while holding still on Newton: {fired[:5]}"
        speed = env.scene["r2_target_drink"].data.root_lin_vel_w
        speed = speed if isinstance(speed, torch.Tensor) else wp.to_torch(speed)
        assert float(speed.norm(dim=-1).max()) < 0.05
    finally:
        env.close()
