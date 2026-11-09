# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Compose M3 into E3 through live USD overrides; shipped asset bytes stay unchanged."""

from pxr import UsdLux

from dexbench_arena.assets.db_m1_station import apply_m1_station_edits
from dexbench_arena.layouts.db_m3_layout import E3_M3_DEACTIVATED_SLOTS


def apply_m3_station_edits(stage, cell_root: str) -> None:
    """Deactivate the occupied slots and conveyor graph, validating before edits."""
    paths = [f"{cell_root}/FixtureSlots/{name}" for name in E3_M3_DEACTIVATED_SLOTS]
    paths.append(f"{cell_root}/FixtureSlots/Slot_Conveyor/Asset/ConveyorBeltGraph")
    prims = [stage.GetPrimAtPath(path) for path in paths]
    missing = [path for path, prim in zip(paths, prims) if not prim.IsValid()]
    if missing:
        raise RuntimeError(f"E3 M3 composition is missing required prims: {missing}")
    apply_m1_station_edits(stage, cell_root)
    for prim in prims:
        prim.SetActive(False)

    fill = UsdLux.DomeLight.Define(stage, "/World/M3CollectionFill")
    fill.CreateIntensityAttr(700.)
