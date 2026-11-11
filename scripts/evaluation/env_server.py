#!/usr/bin/env python3
# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Serve a DexBench environment to remote policy clients over ZeroMQ.

The environment (and mounted embodiment) is swappable through the same CLI
subcommands as the other DexBench scripts; the policy process connects with
``dexbench_arena.evaluation.env_client.DexBenchEnvClient`` and drives
``reset``/``step`` remotely — the server rolls the received actions out on the
sim robot.

Examples (inside the DexBench container):

    /isaac-sim/python.sh scripts/evaluation/env_server.py \
        --viz none --enable_cameras --port 5560 \
        dexbench_retail_v1_environment \
        --scene_layout fc_260821 --task place_box_in_slot --anchor Gondola \
        --embodiment dexbench_rby1_wuji2_joint

    /isaac-sim/python.sh scripts/evaluation/env_server.py \
        --viz none --port 5560 \
        dexbench_hotel_fb_e4_environment --task stack_plates \
        --embodiment dexbench_rby1_wuji2_joint --arm_mode dual_arm

For the GA Wuji2 palm-camera asset, add ``--wrist-cameras`` after the environment
subcommand and keep ``--enable_cameras`` before it. Omit it or pass
``--no-wrist-cameras`` for ego-only observations; physical mounts remain.

By default the environment's own termination terms are stripped so episodes
never auto-reset under the client's feet; the server reports truncated=True
after --max_episode_steps (by default the task's own episode cap) and the client
decides when to call reset. Pass
--env_managed_episodes to keep the env's terminations active instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "source" / "dexbench_arena"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser  # noqa: E402
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext  # noqa: E402

from dexbench_arena.app_launch import stop_unneeded_viewport_rendering, with_dexbench_kit_args  # noqa: E402
from dexbench_arena.assets.physics_variants import apply_newton_usd_parse_guard  # noqa: E402


def add_env_server_cli_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("DexBench Env Server Arguments")
    group.add_argument("--port", type=int, default=5560, help="ZMQ REP port to bind.")
    group.add_argument("--bind_address", type=str, default="*", help="Interface to bind (default: all).")
    group.add_argument(
        "--max_episode_steps",
        type=int,
        default=None,
        help=(
            "Steps per episode before the server reports truncated=True (stabilization excluded). "
            "Default: the environment's own episode cap (the task's timeout) in control steps."
        ),
    )
    group.add_argument(
        "--reset_stabilization_steps",
        type=int,
        default=10,
        help="Physics-settling steps executed inside reset() before the first observation is returned.",
    )
    group.add_argument(
        "--reset_rerenders",
        type=int,
        default=3,
        help=(
            "RTX-only renders after reset before observations are computed (default: 3). "
            "This refreshes cameras without advancing physics; camera runs use at least one render."
        ),
    )
    group.add_argument(
        "--stabilization_mode",
        type=str,
        default="hold",
        choices=("hold", "zero", "none"),
        help="Action used during stabilization: hold current state, zero action, or skip.",
    )
    group.add_argument(
        "--terminate_on_success",
        action="store_true",
        help="Fold the env's success termination term (when present) into terminated.",
    )
    group.add_argument(
        "--env_managed_episodes",
        action="store_true",
        help=(
            "Keep the environment's own termination terms active (episodes auto-reset in-sim). "
            "By default terminations are stripped and the client owns episode boundaries."
        ),
    )
    group.add_argument(
        "--wire_image_resize_height",
        type=int,
        default=0,
        help=(
            "Area-downscale camera observations to this height on the wire, the GR00T"
            " training transform; the aspect ratio must match the render. 0 keeps native size."
        ),
    )
    group.add_argument(
        "--wire_image_resize_width",
        type=int,
        default=0,
        help="Width for --wire_image_resize_height; 0 keeps native size.",
    )


def main() -> None:
    from dexbench_arena.contact_materials import validate_legacy_hand_friction_environment

    validate_legacy_hand_friction_environment()
    args_parser = get_isaaclab_arena_cli_parser()
    add_env_server_cli_args(args_parser)
    args_cli, _ = args_parser.parse_known_args()
    apply_newton_usd_parse_guard()
    args_cli.kit_args = with_dexbench_kit_args(args_cli.kit_args)
    launcher_enable_cameras = bool(getattr(args_cli, "enable_cameras", False))

    with SimulationAppContext(args_cli) as sim_app_context:
        stop_unneeded_viewport_rendering()
        import gymnasium as gym
        import isaaclab_tasks  # noqa: F401

        from isaaclab_arena.utils.hydra_overrides import assert_hydra_overrides
        from isaaclab_arena.utils.isaaclab_utils.simulation_app import reapply_viewer_cfg

        from dexbench_arena.cli import get_arena_builder_from_cli, get_dexbench_arena_cli_parser
        from dexbench_arena.evaluation.env_server import DexBenchEnvServer, DexBenchEnvServerConfig, episode_cap_steps

        args_parser_with_envs = get_dexbench_arena_cli_parser(args_parser)
        args_cli, hydra_overrides = args_parser_with_envs.parse_known_args()
        assert_hydra_overrides(hydra_overrides, args_parser_with_envs)
        args_cli.enable_cameras = bool(getattr(args_cli, "enable_cameras", False)) or launcher_enable_cameras

        arena_builder = get_arena_builder_from_cli(args_cli, hydra_overrides=hydra_overrides)
        if args_cli.list_variations:
            print(arena_builder.get_variations_catalogue_as_string())
            return

        name, env_cfg, env_kwargs = arena_builder.build_registered()

        if args_cli.reset_rerenders < 0:
            args_parser_with_envs.error("--reset_rerenders must be non-negative")
        env_cfg.num_rerenders_on_reset = max(
            env_cfg.num_rerenders_on_reset, args_cli.reset_rerenders
        )

        terminations_cfg = getattr(env_cfg, "terminations", None)
        success_term = getattr(terminations_cfg, "success", None) if terminations_cfg is not None else None

        if args_cli.max_episode_steps is None:
            args_cli.max_episode_steps = episode_cap_steps(env_cfg)
            print(f"[INFO] Episode budget: the task's cap, {args_cli.max_episode_steps} control steps")
        if args_cli.max_episode_steps <= 0:
            args_parser_with_envs.error("--max_episode_steps must be positive")

        if not args_cli.env_managed_episodes:
            total_steps = args_cli.max_episode_steps + args_cli.reset_stabilization_steps
            env_cfg.episode_length_s = total_steps * env_cfg.decimation * env_cfg.sim.dt
            env_cfg.terminations = {}
            if getattr(env_cfg, "rewards", None) is not None:
                print("[INFO] Stripping env rewards together with terminations for server-managed episodes")
                env_cfg.rewards = None
            if getattr(env_cfg, "recorders", None) is not None:
                from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg

                print("[INFO] Stripping env recorders together with terminations for server-managed episodes")
                env_cfg.recorders = RecorderManagerBaseCfg()

        env = gym.make(name, cfg=env_cfg, **env_kwargs).unwrapped
        reapply_viewer_cfg(env)

        config = DexBenchEnvServerConfig(
            port=args_cli.port,
            bind_address=args_cli.bind_address,
            max_episode_steps=args_cli.max_episode_steps,
            reset_stabilization_steps=args_cli.reset_stabilization_steps,
            stabilization_mode=args_cli.stabilization_mode,
            terminate_on_success=args_cli.terminate_on_success,
            wire_image_resize_height=(
                args_cli.wire_image_resize_height if args_cli.wire_image_resize_height > 0 else None
            ),
            wire_image_resize_width=(
                args_cli.wire_image_resize_width if args_cli.wire_image_resize_width > 0 else None
            ),
            render_on_idle=not bool(getattr(sim_app_context.app_launcher, "_headless", True)),
        )
        server = DexBenchEnvServer(
            config,
            env,
            sim_app_context.app_launcher.app,
            success_term=success_term,
        )
        try:
            server.run()
        finally:
            print("[INFO] Closing environment")
            env.close()


if __name__ == "__main__":
    main()
