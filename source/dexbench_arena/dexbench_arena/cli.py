# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""CLI helpers for DexBench-native Arena environments."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable

from isaaclab_arena.assets.registries import EnvironmentRegistry
from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena_environments.cli import add_environment_cli_args

from dexbench_arena.evaluation.dataset_recording import apply_dataset_recording
from dexbench_arena.teleop.newton_solver import apply_solver_options


def ensure_dexbench_environments_registered() -> None:
    """Import DexBench native environments to fire ``@register_environment`` decorators."""

    import dexbench_arena.environments  # noqa: F401


def parse_args_before_app_launch(
    args_parser: argparse.ArgumentParser,
    add_environment_args: Callable[[argparse.ArgumentParser], object] | None = None,
) -> argparse.Namespace:
    """Parse what the app launcher needs, leaving the environment's arguments unregistered."""

    argv = sys.argv[1:]
    if "-h" in argv or "--help" in argv:
        (add_environment_args or add_dexbench_environments_cli_args)(args_parser)
        args_parser.parse_args()

    args_cli, _deferred = args_parser.parse_known_args()
    return args_cli


def add_dexbench_environments_cli_args(args_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add DexBench-native environment subcommands to an Arena parser."""

    ensure_dexbench_environments_registered()
    env_registry = EnvironmentRegistry()
    subparsers = args_parser.add_subparsers(
        dest="example_environment",
        required=True,
        help="DexBench native Arena environment to run",
    )
    for env_name in sorted(env_registry.get_all_keys()):
        if not env_name.startswith("dexbench_"):
            continue
        env_cls = env_registry.get_component_by_name(env_name)
        subparser = subparsers.add_parser(env_cls.name)
        env_cls.add_cli_args(subparser)
    return args_parser


def add_named_environment_cli_args(args_parser: argparse.ArgumentParser, environment_name: str) -> argparse.ArgumentParser:
    """Add one DexBench environment's own arguments to ``args_parser``, without a subcommand."""

    ensure_dexbench_environments_registered()
    add_environment_cli_args(args_parser, EnvironmentRegistry().get_component_by_name(environment_name))
    return args_parser


def get_dexbench_arena_cli_parser(args_parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    """Return an Arena parser with DexBench-native environment subcommands."""

    if args_parser is None:
        args_parser = get_isaaclab_arena_cli_parser()
    return add_dexbench_environments_cli_args(args_parser)


DEXBENCH_PHYSICS_HZ = 60.0
DEXBENCH_PHYSICS_DT = 1.0 / DEXBENCH_PHYSICS_HZ
DEXBENCH_DEFAULT_CONTROL_HZ = 30.0


def resolve_control_hz(args_cli=None) -> float:
    """Control rate for this run, highest precedence first."""
    flag = getattr(args_cli, "rby1_control_hz", None) if args_cli is not None else None
    if flag:
        return float(flag)
    from_env = os.environ.get("DEXBENCH_CONTROL_HZ")
    if from_env:
        return float(from_env)
    return DEXBENCH_DEFAULT_CONTROL_HZ


def resolve_sim_rate(args_cli=None) -> tuple[float, int]:
    """``(physics_dt, decimation)`` for the resolved control rate."""
    control_hz = resolve_control_hz(args_cli)
    decimation = int(round(DEXBENCH_PHYSICS_HZ / control_hz))
    if decimation < 1 or abs(DEXBENCH_PHYSICS_HZ / decimation - control_hz) > 1e-6:
        raise ValueError(
            f"Control rate {control_hz} Hz does not divide the {DEXBENCH_PHYSICS_HZ} Hz physics rate; "
            f"use one of {[DEXBENCH_PHYSICS_HZ / d for d in (1, 2, 3, 4)]}."
        )
    return DEXBENCH_PHYSICS_DT, decimation


DEXBENCH_CONTROL_HZ = resolve_control_hz()

NEWTON_NCONMAX = 4000
NEWTON_NJMAX = 3000


def set_rtx_global_settings(cfg, **settings) -> int:
    """Apply process-global Isaac RTX settings to a composed env cfg."""

    from isaaclab_tasks.utils.presets import set_isaac_rtx_global_settings

    cameras = 0
    scene = getattr(cfg, "scene", None)
    for entry in vars(scene).values() if scene is not None else ():
        renderer_cfg = getattr(entry, "renderer_cfg", None)
        if renderer_cfg is None:
            continue
        set_isaac_rtx_global_settings(renderer_cfg, **settings)
        cameras += 1

    try:
        from isaaclab_physx.renderers.isaac_rtx_renderer_cfg import IsaacRtxRendererGlobalSettingsCfg
        from isaaclab_physx.renderers.isaac_rtx_renderer_utils import apply_isaac_rtx_global_settings

        apply_isaac_rtx_global_settings(IsaacRtxRendererGlobalSettingsCfg(**settings))
    except Exception as exc:  # noqa: BLE001 - never fail a run over a render knob
        print(f"[dexbench] could not apply RTX global settings ({type(exc).__name__}: {exc}); {settings}")

    return cameras


def set_camera_update_period(cfg, control_hz: float, camera_hz: float) -> int:
    """Let every camera on the scene take a new frame only every ``control_hz / camera_hz`` control steps: the
    steps a policy asked at ``camera_hz`` reads (``policy.policy_rate``). Returns how many cameras were set."""
    ratio = control_hz / camera_hz
    if round(ratio) < 1 or abs(ratio - round(ratio)) > 1e-6:
        raise ValueError(f"camera_hz {camera_hz} does not divide the {control_hz:g} Hz control rate")
    period = (round(ratio) - 0.5) / control_hz
    cameras = 0
    scene = getattr(cfg, "scene", None)
    for entry in vars(scene).values() if scene is not None else ():
        if getattr(entry, "renderer_cfg", None) is not None and hasattr(entry, "update_period"):
            entry.update_period = period
            cameras += 1
    return cameras


def apply_delivery_render_settings(cfg, settings: dict[str, int | float], source: str):
    """Apply a delivery's render settings, given in their USD spelling, as RTX global settings."""

    carb_settings = {"/" + key.replace(":", "/"): value for key, value in settings.items()}
    cameras = set_rtx_global_settings(cfg, carb_settings=carb_settings)
    print(
        f"[dexbench] {len(carb_settings)} render settings from {source} "
        f"({cameras} camera renderer cfg{'' if cameras == 1 else 's'})"
    )
    return cfg


def apply_dlss_render_settings(cfg, dlss_mode: int) -> None:
    """Turn DLSS on at ``dlss_mode``, keeping the DL denoiser."""

    cameras = set_rtx_global_settings(
        cfg, antialiasing_mode="DLSS", dlss_mode=dlss_mode, enable_dl_denoiser=True
    )
    print(
        f"[dexbench] DLSS execMode {dlss_mode} (0=Performance..2=Quality), DL denoiser on "
        f"({cameras} camera renderer cfg{'' if cameras == 1 else 's'})"
    )


def apply_dexbench_env_settings(cfg, environment_cfg, *, embodiment=None, output_base_dir: str | None = None) -> None:
    """Pin an Arena env cfg to DexBench's control rate, render settings and dataset recording."""

    physics_dt, decimation = resolve_sim_rate(environment_cfg)
    cfg.sim.dt = physics_dt
    cfg.decimation = decimation
    print(
        f"[dexbench] control {1.0 / (physics_dt * decimation):.1f} Hz "
        f"(physics {1.0 / physics_dt:.0f} Hz x decimation {decimation})"
    )
    camera_hz = getattr(environment_cfg, "camera_hz", None)
    if camera_hz:
        cameras = set_camera_update_period(cfg, 1.0 / (physics_dt * decimation), camera_hz)
        print(f"[dexbench] cameras take a new frame at {camera_hz} Hz ({cameras} camera cfg{'' if cameras == 1 else 's'})")
    cfg.sim.render_interval = max(1, int(os.environ.get("DEXBENCH_RENDER_INTERVAL", decimation)))
    if getattr(environment_cfg, "enable_cameras", False):
        cfg.num_rerenders_on_reset = max(1, getattr(cfg, "num_rerenders_on_reset", 0) or 0)
    dlss_mode = os.environ.get("DEXBENCH_DLSS_MODE")
    if dlss_mode is not None:
        apply_dlss_render_settings(cfg, int(dlss_mode))
    apply_dataset_recording(
        cfg,
        environment_cfg,
        enable_cameras=bool(getattr(embodiment, "enable_cameras", False)),
        output_base_dir=output_base_dir,
    )


def apply_embodiment_recorders(cfg, embodiment) -> None:
    """The embodiment's own recorders, once DexBench's settings and solver are on the cfg."""

    configure_env_recorders = getattr(embodiment, "configure_env_recorders", None)
    if configure_env_recorders is not None:
        configure_env_recorders(cfg)


def apply_dexbench_physics_settings(cfg, environment_cfg, *, environment=None) -> None:
    """DexBench's solver settings, applied to the Newton physics cfg Arena's preset left."""

    from isaaclab_newton.physics.newton_manager_cfg import NewtonCfg

    physics = getattr(cfg.sim, "physics", None)
    if not isinstance(physics, NewtonCfg):
        raise TypeError(f"DexBench builds on Newton only; the composed env cfg carries {type(physics).__name__}")
    if "DEXBENCH_SOLVER_POSITION_ITER" in os.environ:
        print(
            "[dexbench] DEXBENCH_SOLVER_POSITION_ITER is ignored: it is PhysX's position iteration count, and "
            "DexBench runs on Newton only. MuJoCo-Warp's settings are the environment's "
            "newton_solver_options and the --newton_* flags."
        )
    from dexbench_arena.assets.newton_root_reset import install_newton_fixed_root_reset_compat

    install_newton_fixed_root_reset_compat()

    solver = getattr(cfg.sim.physics, "solver_cfg", None)
    if solver is not None and hasattr(solver, "nconmax") and hasattr(solver, "njmax"):
        solver.nconmax = max(int(solver.nconmax), int(getattr(environment_cfg, "newton_nconmax", None) or NEWTON_NCONMAX))
        solver.njmax = max(int(solver.njmax), int(getattr(environment_cfg, "newton_njmax", None) or NEWTON_NJMAX))
    if getattr(environment_cfg, "newton_cuda_graph", True) is False:
        cfg.sim.physics.use_cuda_graph = False
        print("[dexbench] Newton CUDA graph capture off (--no-newton_cuda_graph)")
    hook = getattr(environment, "apply_newton_sim_settings", None)
    if hook is not None:
        hook(cfg, environment_cfg)
    run_options = {
        key: value
        for key, value in (
            ("solver", getattr(environment_cfg, "newton_solver", None)),
            ("num_substeps", getattr(environment_cfg, "newton_num_substeps", None)),
            ("tolerance", getattr(environment_cfg, "newton_solver_tolerance", None)),
        )
        if value is not None
    }
    if run_options:
        print(f"[dexbench] Newton run settings over the environment's profile: {apply_solver_options(cfg.sim.physics, run_options)}")
    solver = getattr(cfg.sim.physics, "solver_cfg", None)
    if solver is not None and hasattr(solver, "nconmax") and hasattr(solver, "njmax"):
        print(f"[dexbench] Newton MJWarp contact budget nconmax={solver.nconmax} njmax={solver.njmax}")


def parse_args_with_hydra_overrides(
    args_parser: argparse.ArgumentParser,
) -> tuple[argparse.Namespace, list[str]]:
    """Parse a DexBench script's command line, keeping the Hydra variation overrides."""

    from isaaclab_arena.utils.hydra_overrides import assert_hydra_overrides

    from dexbench_arena.assets.physics_variants import apply_newton_usd_parse_guard

    args_cli, hydra_overrides = args_parser.parse_known_args()
    assert_hydra_overrides(hydra_overrides, args_parser)
    apply_newton_usd_parse_guard()
    return args_cli, hydra_overrides


def get_arena_builder_from_cli(
    args_cli: argparse.Namespace,
    hydra_overrides: list[str] | None = None,
):
    """Build an ArenaEnvBuilder for a DexBench-native environment."""

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    if hydra_overrides:
        print(f"[dexbench] variation overrides: {list(hydra_overrides)}")

    ensure_dexbench_environments_registered()
    env_registry = EnvironmentRegistry()

    assert hasattr(args_cli, "example_environment"), "Example environment must be specified"
    assert env_registry.is_registered(args_cli.example_environment), (
        f"Example environment type {args_cli.example_environment} is not registered"
    )
    example_env = env_registry.get_component_by_name(args_cli.example_environment)()
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse

    builder_cfg = arena_env_builder_cfg_from_argparse(args_cli)
    builder = ArenaEnvBuilder(example_env.get_env(args_cli), builder_cfg, hydra_overrides=hydra_overrides)
    builder.dexbench_environment = example_env
    return builder
