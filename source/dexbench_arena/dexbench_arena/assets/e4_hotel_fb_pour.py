# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""The E4 water-pour props: the IDEALISK jug the robot pours from and the goblet it pours into."""

from __future__ import annotations

from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.assets.object_library import LibraryObject
from isaaclab_arena.assets.register import register_asset

from dexbench_arena import ASSETS_DIR
from dexbench_arena.assets.scene_bundles import runtime_usd
from dexbench_arena.tasks.object_axis import ObjectAxis, TaskObject

__all__ = [
    "DexBenchGoblet30cl",
    "DexBenchIdealiskJug",
    "DexBenchSteelWaterJugTall",
    "DexBenchSteelPitcherLarge",
    "DexBenchSteelPitcherFooted",
    "POUR_JUGS",
    "POUR_JUG_OBJECT_AXIS",
]


@register_asset
class DexBenchGoblet30cl(LibraryObject):
    """PSL 30 cl goblet as a registered RIGID object (the E4 props layer's copies are placement
    decor, not scene entities). Authored rigid body and collision ship in the package; z-up,
    base at origin."""

    name = "dexbench_goblet_30cl"
    tags = ["object", "hotel_fb_e4", "tableware", "pour"]
    object_type = ObjectType.RIGID
    usd_path = runtime_usd(ASSETS_DIR / "object_library" / "prop_goblet_30cl" / "usd" / "prop_goblet_30cl.usd")


@register_asset
class DexBenchIdealiskJug(LibraryObject):
    """1 l stainless steel jug (z-up, base at origin, handle +X)."""

    name = "dexbench_idealisk_jug"
    tags = ["object", "hotel_fb_e4", "pour"]
    object_type = ObjectType.RIGID
    usd_path = str(ASSETS_DIR / "object_library" / "steel_jug_1l" / "usd" / "steel_jug_1l.usd")


class _TrinixJug(LibraryObject):
    """A Trinix cad2sim serveware jug (z-up, base at origin, handle +X like IDEALISK)."""

    tags = ["object", "hotel_fb_e4", "pour"]
    object_type = ObjectType.RIGID
    package: str

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.usd_path = str(ASSETS_DIR / "object_library" / cls.package / "usd" / f"{cls.package}.usd")


@register_asset
class DexBenchSteelWaterJugTall(_TrinixJug):
    """Brushed stainless tall water jug: 4.7 cm bore, 21.6 cm deep, 1.50 L."""

    name = "dexbench_steel_water_jug_tall"
    package = "steel_water_jug_tall"


@register_asset
class DexBenchSteelPitcherLarge(_TrinixJug):
    """Hammered stainless pitcher: 6.9 cm belly, 22.8 cm deep, 2.41 L, spout at -X."""

    name = "dexbench_steel_pitcher_large"
    package = "steel_pitcher_large"


@register_asset
class DexBenchSteelPitcherFooted(_TrinixJug):
    """Copper-plated footed pitcher: 5.5 cm belly on a 3.7 cm foot, 0.92 L, spout at -X."""

    name = "dexbench_steel_pitcher_footed"
    package = "steel_pitcher_footed"


@register_asset
class DexBenchD00527x15(LibraryObject):
    """D00527 milk pitcher scaled 1.5x (pool package milk_pitcher_d00527_x15, water shell and profile beside the USD)."""

    name = "dexbench_d00527_x15"
    tags = ["object", "hotel_fb_e4", "pour"]
    object_type = ObjectType.RIGID
    usd_path = str(ASSETS_DIR / "object_library" / "milk_pitcher_d00527_x15" / "usd" / "milk_pitcher_d00527_x15.usd")


POUR_JUGS: dict[str, tuple[str, str]] = {
    "d00527_x15": ("dexbench_d00527_x15", "pour_jug"),
    "idealisk": ("dexbench_idealisk_jug", "idealisk_jug"),
    "steel_water_jug_tall": ("dexbench_steel_water_jug_tall", "pour_jug"),
    "steel_pitcher_large": ("dexbench_steel_pitcher_large", "pour_jug"),
    "steel_pitcher_footed": ("dexbench_steel_pitcher_footed", "pour_jug"),
}
"""``--pour_jug`` choice -> (registered asset name, scene instance name). IDEALISK keeps the
instance name the pour tools and the teleop water (``DEXBENCH_WATER``) address; the steel jugs share ``pour_jug``."""


POUR_JUG_OBJECT_AXIS = ObjectAxis(
    field="pour_jug",
    default="idealisk",
    objects=(
        TaskObject("idealisk", "dexbench_idealisk_jug", "train", "the jug", (0.144, 0.136)),
        TaskObject("d00527_x15", "dexbench_d00527_x15", "train", "the jug", (0.162, 0.192)),
        TaskObject("steel_water_jug_tall", "dexbench_steel_water_jug_tall", "stand-in", "the jug", (0.099, 0.276)),
        TaskObject("steel_pitcher_large", "dexbench_steel_pitcher_large", "eval", "the jug", (0.149, 0.227)),
        TaskObject("steel_pitcher_footed", "dexbench_steel_pitcher_footed", "eval", "the jug", (0.135, 0.181)),
    ),
    size_axes="inner diameter, rim height",
)
"""DB-H2's jugs, selected by ``--pour_jug``."""
