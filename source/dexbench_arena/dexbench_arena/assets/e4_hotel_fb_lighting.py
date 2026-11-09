# SPDX-License-Identifier: Apache-2.0
"""The E4 hall's delivered light layer and the one edit a lighting draw makes to it."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from dexbench_arena import ASSETS_DIR
from dexbench_arena.assets.scene_bundles import runtime_usd

if TYPE_CHECKING:
    from pxr import Usd

E4_LIGHTING_LAYER = Path(runtime_usd(ASSETS_DIR / "env" / "e4_hotel_fb" / "banquet" / "lighting.usda"))
E4_LIGHTING_ASSET = "dexbench_hotel_fb_e4_lighting"
COLOR_TEMPERATURE_LIMITS_K = (1000.0, 10000.0)
"""The colour temperatures Isaac Lab accepts on a light [K]."""


def check_light_edits(intensity_scale: float = 1.0, color_temperature_k: float | None = None) -> None:
    """Raise ``ValueError`` for an edit :func:`apply_light_edits` cannot make."""
    if intensity_scale < 0.0:
        raise ValueError(f"intensity_scale must be non-negative, got {intensity_scale}")
    low, high = COLOR_TEMPERATURE_LIMITS_K
    if color_temperature_k is not None and not low <= color_temperature_k <= high:
        raise ValueError(f"colour temperature must be in [{low:g}, {high:g}] K, got {color_temperature_k}")


def apply_light_edits(root: Usd.Prim, intensity_scale: float = 1.0, color_temperature_k: float | None = None) -> int:
    """Scale every light under ``root`` and optionally set one colour temperature on all of them."""
    from pxr import Usd, UsdLux

    check_light_edits(intensity_scale, color_temperature_k)
    count = 0
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdLux.LightAPI):
            continue
        light = UsdLux.LightAPI(prim)
        if intensity_scale != 1.0:
            attr = light.GetIntensityAttr()
            attr.Set(float(attr.Get()) * float(intensity_scale))
        if color_temperature_k is not None:
            light.CreateEnableColorTemperatureAttr(True)
            light.CreateColorTemperatureAttr(float(color_temperature_k))
        count += 1
    return count
