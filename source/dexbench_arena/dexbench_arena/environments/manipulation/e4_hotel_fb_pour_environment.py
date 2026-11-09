# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""DB-H2 (Lift and Pour) in the E4 banquet hall: the E4 scene with the pour station, and the task."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentFactory

from dexbench_arena.assets.e4_hotel_fb_pour import POUR_JUG_OBJECT_AXIS
from dexbench_arena.layouts.db_h1_layout import POUR_RENDER_SETTINGS
from dexbench_arena.environments.manipulation.e4_hotel_fb_environment import (
    DexBenchHotelFBE4Environment,
    DexBenchHotelFBE4EnvironmentCfg,
    build_e4_hotel_fb_scene,
)

if TYPE_CHECKING:
    from isaaclab_arena.scene.scene import Scene

LIFT_AND_POUR_TASK_NAME = "lift_and_pour"


@dataclass
class DexBenchHotelFBE4PourEnvironmentCfg(DexBenchHotelFBE4EnvironmentCfg):
    """Typed config for DB-H2 in the E4 banquet hall: the E4 fields, the pour station and the start draws."""

    idealisk_pour: bool = True
    """Add the water-pour arrangement at the serving station: the jug and the target goblet. On by default: the
    environment's one task, ``lift_and_pour``, refuses a scene without it."""
    pour_decor_goblets: bool = False
    """Spawn the layout's decor goblets beside the target; off keeps the counter as the room has it."""
    pour_jug: str = "idealisk"
    """Which jug stands at the pour station: ``idealisk``, ``d00527_x15`` or one of the Trinix ``steel_*`` packages."""
    pour_jug_jitter_xy: float = 0.0
    """Half-extent [m] of the jug's per-reset uniform X and Y offset, recorded as ``<jug>.pour_pose``; 0 = off."""
    pour_jug_jitter_yaw_deg: float = 0.0
    """Half-extent [deg] of the jug's per-reset uniform turn about its axis; 0 = off."""
    pour_goblet_jitter_xy: float = 0.0
    """Half-extent [m] of the target goblet's per-reset uniform X and Y offset, recorded as ``pour_goblet.pour_pose``;
    0 = off."""
    pour_jug_offset_x: float = 0.0
    """Fixed X offset [m] of the jug from its spot, the centre of its ``pour_pose`` draw (a GA-S fixed point); every
    draw is the offset plus the jitter, recorded as one draw. 0 = none."""
    pour_jug_offset_y: float = 0.0
    """Fixed Y offset [m] of the jug, as ``pour_jug_offset_x``."""
    pour_jug_offset_yaw_deg: float = 0.0
    """Fixed turn [deg] of the jug about its axis, the centre of its yaw draw, as ``pour_jug_offset_x``."""
    pour_goblet_offset_x: float = 0.0
    """Fixed X offset [m] of the target goblet, the centre of its ``pour_pose`` draw, as ``pour_jug_offset_x``."""
    pour_goblet_offset_y: float = 0.0
    """Fixed Y offset [m] of the target goblet, as ``pour_goblet_offset_x``."""
    pour_station_offset_y: float = 0.05
    """How far the station (the jug, the target goblet, the decor goblets) stands from the layout's spots along +y
    [m]. DB-H2 stands it 0.05 m behind the layout's spots (the jug at y 2.44, the goblet at 2.47), where its teleop
    grasps were recorded, with the base moved the same 0.05 m (``base_offset_y`` 0.15, y 1.90): at the 1.85 m base
    that spot lies at the arm's reach. The grasps are replayed relative to the jug, so they serve at either spot."""


def make_lift_and_pour_task(environment, cfg: DexBenchHotelFBE4PourEnvironmentCfg):  # noqa: ARG001
    """Lift the jug and pour its 100 mL of water into the wine glass (the task owns the water). A condition
    recording (``record_settle``) builds no water: the hold-pose policy only checks that the start comes to rest."""
    if not cfg.idealisk_pour:
        raise ValueError("--task lift_and_pour needs the pour station (--idealisk_pour)")
    from dexbench_arena.tasks.lift_and_pour import DexBenchLiftAndPourTask

    return DexBenchLiftAndPourTask(cfg.pour_jug, episode_length_s=cfg.episode_length_s, water=not cfg.record_settle)


@register_environment
class DexBenchHotelFBE4PourEnvironment(
    DexBenchHotelFBE4Environment, ArenaEnvironmentFactory[DexBenchHotelFBE4PourEnvironmentCfg]
):
    """DB-H2: the E4 banquet hall staged for lifting the jug and pouring into the wine glass."""

    name = "dexbench_hotel_fb_e4_pour_environment"
    _legacy_argparse_cfg_type = DexBenchHotelFBE4PourEnvironmentCfg
    available_tasks = {LIFT_AND_POUR_TASK_NAME: make_lift_and_pour_task}
    object_axes = {LIFT_AND_POUR_TASK_NAME: POUR_JUG_OBJECT_AXIS}

    def pour_station_dressing(self, cfg: DexBenchHotelFBE4PourEnvironmentCfg) -> dict | None:
        """The station's glass render settings, when the station is built."""
        return POUR_RENDER_SETTINGS if cfg.idealisk_pour else None

    def build_scene(self, cfg: DexBenchHotelFBE4PourEnvironmentCfg) -> Scene:
        return build_e4_hotel_fb_scene(
            self.asset_registry,
            dish_variant=cfg.plate_variant,
            plate_diameter_m=cfg.plate_diameter_m,
            source_plate_count=int(cfg.source_plate_count),
            dest_plate_count=int(cfg.dest_plate_count),
            source_stack_offset_xy_m=(float(cfg.source_stack_offset_x), float(cfg.source_stack_offset_y)),
            dest_stack_offset_xy_m=(float(cfg.dest_stack_offset_x), float(cfg.dest_stack_offset_y)),
            stack_jitter_xy_m=float(cfg.stack_jitter_xy),
            stack_jitter_yaw_deg=float(cfg.stack_jitter_yaw_deg),
            idealisk_pour=bool(cfg.idealisk_pour),
            pour_jug=str(cfg.pour_jug),
            pour_decor_goblets=bool(cfg.pour_decor_goblets),
            pour_jug_jitter_xy_m=float(cfg.pour_jug_jitter_xy),
            pour_jug_jitter_yaw_deg=float(cfg.pour_jug_jitter_yaw_deg),
            pour_goblet_jitter_xy_m=float(cfg.pour_goblet_jitter_xy),
            pour_station_offset_xy_m=(0.0, float(cfg.pour_station_offset_y)),
            pour_jug_offset_xy_m=(float(cfg.pour_jug_offset_x), float(cfg.pour_jug_offset_y)),
            pour_jug_offset_yaw_deg=float(cfg.pour_jug_offset_yaw_deg),
            pour_goblet_offset_xy_m=(float(cfg.pour_goblet_offset_x), float(cfg.pour_goblet_offset_y)),
        )

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        super().add_cli_args(parser)

        def _add(*options, **kwargs):
            if any(option in parser._option_string_actions for option in options):
                return
            parser.add_argument(*options, **kwargs)

        _add(
            "--idealisk_pour",
            "--idealisk-pour",
            dest="idealisk_pour",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Add the water-pour arrangement at the serving station (A_h2): the jug (--pour_jug) and the "
            "target goblet, lit by the room (see --pour_decor_goblets). On by default; lift_and_pour needs it. "
            "The water is lift_and_pour's; a teleop session without the task attaches its own with DEXBENCH_WATER=1.",
        )
        _add(
            "--pour_jug",
            "--pour-jug",
            dest="pour_jug",
            choices=("idealisk", "steel_water_jug_tall", "steel_pitcher_large", "steel_pitcher_footed", "d00527_x15"),
            default=os.environ.get("DEXBENCH_E4_POUR_JUG", "idealisk"),
            help="Which jug stands at the pour station. IDEALISK keeps the scene name idealisk_jug; the "
            "others spawn as pour_jug. The water finds the station's jug by itself.",
        )
        _add(
            "--pour_decor_goblets",
            "--pour-decor-goblets",
            dest="pour_decor_goblets",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Spawn the two decor goblets in the counter's back row beside the target goblet. Off by default: "
            "main's room has no glassware on the counter.",
        )
        if not any(option in parser._option_string_actions for option in ("--pour_station_offset_y",)):
            parser.add_argument("--pour_station_offset_y", "--pour-station-offset-y", dest="pour_station_offset_y",
                                type=float, default=0.05,
                                help="How far the station (jug, target goblet, decor goblets) stands from the delivered "
                                "spots along +y [m]; DB-H2 stands it 0.05 m behind them (jug y 2.44, goblet 2.47).")
        for flag, unit, what in (("pour_jug_jitter_xy", "m", "the jug's per-reset uniform X and Y offset"),
                                 ("pour_jug_jitter_yaw_deg", "deg", "the jug's per-reset uniform turn about its axis"),
                                 ("pour_goblet_jitter_xy", "m", "the target goblet's per-reset uniform X and Y offset")):
            options = (f"--{flag}", f"--{flag.replace('_', '-')}")
            if not any(option in parser._option_string_actions for option in options):
                parser.add_argument(*options, dest=flag, type=float, default=0.0,
                                    help=f"Half-extent [{unit}] of {what} (DB-H2's start draws, recorded per episode). 0 = off.")
        for flag, unit, what in (("pour_jug_offset_x", "m", "the jug's X"), ("pour_jug_offset_y", "m", "the jug's Y"),
                                 ("pour_jug_offset_yaw_deg", "deg", "the jug's turn about its axis"),
                                 ("pour_goblet_offset_x", "m", "the target goblet's X"),
                                 ("pour_goblet_offset_y", "m", "the target goblet's Y")):
            options = (f"--{flag}", f"--{flag.replace('_', '-')}")
            if not any(option in parser._option_string_actions for option in options):
                parser.add_argument(*options, dest=flag, type=float, default=0.0,
                                    help=f"Fixed offset [{unit}] of {what} that its start draw is centred on (a GA-S "
                                    "fixed point, recorded with the draw). 0 = none.")
