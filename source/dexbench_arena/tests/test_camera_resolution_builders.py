# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Exercise camera resolution through the evaluation and recording builders."""

from pathlib import Path

import pytest


@pytest.mark.parametrize("route", [
    "policy_runner", "scene_cli", "experiment_runner", "record_demos", "record_experiment",
])
@pytest.mark.parametrize("scale", [None, 0.25, 1.0])
@pytest.mark.parametrize("backend", [None, "newton"], ids=["default", "presets-newton"])
def test_camera_resolution_reaches_scene_cfg(simulation_app, monkeypatch, tmp_path, route, scale, backend):
    import gymnasium as gym

    from dexbench_arena import isaaclab_interop
    from dexbench_arena.cli import (
        ensure_dexbench_environments_registered,
        get_arena_builder_from_cli,
        get_dexbench_arena_cli_parser,
    )
    from dexbench_arena.environments.manipulation.e4_hotel_fb_environment import DexBenchHotelFBE4EnvironmentCfg
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.evaluation.arena_run import ArenaRunCfg
    from isaaclab_arena.evaluation.policy_runner_cli import add_policy_runner_arguments
    from isaaclab_arena.evaluation.run_execution import build_arena_builder_from_run_cfg
    from isaaclab_arena.policy.policy_base import PolicyCfg
    from isaaclab_arena.utils.physics_backend import PhysicsBackend

    ensure_dexbench_environments_registered()
    environment = "dexbench_hotel_fb_e4_environment"
    recording = route in ("record_demos", "record_experiment")
    embodiment = "dexbench_rby1_wuji2" if recording else "dexbench_rby1_wuji2_joint"
    tail = ["--task", "stack_plates", "--embodiment", embodiment, "--enable_cameras"]
    if recording:
        tail += ["--teleop_device", "openxr"]
    if scale is not None:
        tail += ["--camera_scale", str(scale)]
    if route == "experiment_runner":
        run = ArenaRunCfg(
            name="camera_resolution",
            environment=DexBenchHotelFBE4EnvironmentCfg(
                task="stack_plates", embodiment="dexbench_rby1_wuji2_joint",
                enable_cameras=True, camera_scale=scale,
            ),
            environment_builder=ArenaEnvBuilderCfg(presets=PhysicsBackend(backend) if backend else None),
            policy=PolicyCfg(),
        )
        builder = build_arena_builder_from_run_cfg(run)
        cfg, _ = builder.compose_manager_cfg()
    elif route in ("record_demos", "record_experiment"):
        head = ["--presets", backend] if backend else []
        if route == "record_experiment":
            monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "scripts/imitation_learning"))
            from _dexbench_arena_script import _compute_lab_argv, expand_experiment_run

            experiment = tmp_path / "camera.yaml"
            experiment.write_text(
                "shared:\n  environment:\n"
                f"    type: {environment}\n    task: stack_plates\n"
                f"    embodiment: {embodiment}\n    enable_cameras: true\n"
                + (f"    camera_scale: {scale}\n" if scale is not None else "")
                + (f"  environment_builder:\n    presets: {backend.upper()}\n" if backend else "")
                + "runs:\n  camera: {}\n"
            )
            _, head, environment, tail = _compute_lab_argv(
                "record_demos.py", expand_experiment_run([
                    "--experiment", str(experiment), "--run", "camera", "--teleop_device", "openxr",
                ])
            )
        monkeypatch.setattr(isaaclab_interop, "HEAD_ARGV", head)
        monkeypatch.setattr(isaaclab_interop, "ENVIRONMENT_NAME", environment)
        monkeypatch.setattr(isaaclab_interop, "TAIL_ARGV", tail)
        monkeypatch.setattr(isaaclab_interop, "IM_SCRIPT_NAME", "record_demos.py")
        monkeypatch.setattr(isaaclab_interop, "BUILDER", None)
        assert isaaclab_interop.environment_registration_callback() == []
        cfg = gym.spec(environment).kwargs["env_cfg_entry_point"]
    else:
        parser = get_dexbench_arena_cli_parser()
        if route == "policy_runner":
            add_policy_runner_arguments(parser)
        presets = ["--presets", backend] if backend else []
        builder = get_arena_builder_from_cli(parser.parse_args([*presets, environment, *tail]))
        cfg, _ = builder.compose_manager_cfg()
    expected = {None: (640, 360), 0.25: (320, 180), 1.0: (1280, 720)}[scale]
    for name in ("robot_pov_left_cam", "robot_pov_right_cam"):
        camera = getattr(cfg.scene, name)
        assert (camera.width, camera.height) == expected
        assert camera.renderer_cfg.global_settings.antialiasing_mode == "DLAA"
    assert cfg.sim.dt * cfg.decimation == pytest.approx(1 / 30)
    assert type(cfg.sim.physics).__name__ == "NewtonCfg"
