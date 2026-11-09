# SPDX-License-Identifier: Apache-2.0
"""Start Isaac Sim the way Arena does, with DexBench's pre-start environment settings."""

from __future__ import annotations

import argparse
import traceback
from collections.abc import Callable

from isaaclab_arena.utils.isaaclab_utils.simulation_app import get_app_launcher

from dexbench_arena.assets.physics_variants import apply_newton_usd_parse_guard

DEXBENCH_KIT_ARGS = ("--/persistent/UJITSO/geometry=false",)


def with_dexbench_kit_args(kit_args: str | None) -> str:
    """``DEXBENCH_KIT_ARGS`` followed by the caller's ``--kit_args`` value, so a setting the caller names wins."""

    return " ".join((*DEXBENCH_KIT_ARGS, *([kit_args] if kit_args else [])))


ACTIVE_VIEWPORT_SETTING = "/isaaclab/render/active_viewport"
OFFSCREEN_RENDER_SETTING = "/isaaclab/render/offscreen"


def stop_unneeded_viewport_rendering() -> None:
    """Restore IsaacLab#851's headless offscreen viewport optimization."""
    import carb

    settings = carb.settings.get_settings()
    if settings.get(ACTIVE_VIEWPORT_SETTING) is not False or settings.get(OFFSCREEN_RENDER_SETTING) is not True:
        return

    from omni.kit.viewport.utility import get_active_viewport

    viewport = get_active_viewport()
    if viewport is not None:
        viewport.updates_enabled = False


def launch_isaac_app(args_cli: argparse.Namespace):
    """Set the Newton USD parse guard, then start the app through Arena's ``get_app_launcher``."""

    apply_newton_usd_parse_guard()
    launch_args = argparse.Namespace(**vars(args_cli))
    launch_args.kit_args = with_dexbench_kit_args(getattr(args_cli, "kit_args", None))
    launcher = get_app_launcher(launch_args)
    stop_unneeded_viewport_rendering()
    return launcher


def visualizers_disabled(args_cli: argparse.Namespace) -> bool:
    """Return True when the caller explicitly asked for no visualizer (``--viz none``)."""

    return getattr(args_cli, "visualizer", None) is None and bool(
        getattr(args_cli, "visualizer_explicit", False)
    )


def run_then_close_app(simulation_app, main: Callable[[], object]) -> int:
    """Run ``main``, then close Kit with an exit status that says whether it raised."""
    exit_code = 1
    try:
        main()
        exit_code = 0
    except BaseException:  # noqa: BLE001 -- every failure, SystemExit too, must leave a nonzero status
        traceback.print_exc()
    finally:
        simulation_app.close(exit_code=exit_code)
    return exit_code
