# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""DexBench's edits to the E3 assembly line for the DB-M4 leveling-leg station, authored at spawn."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dexbench_arena.assets.e3_manufacturing_cell import edit_e3_cell
from dexbench_arena.layouts.db_m4_layout import (
    E3_M4_DEACTIVATED_DRESSING,
    E3_M4_DEACTIVATED_SLOTS,
    E3_M4_WORKBENCH_SLOT,
    E3_M4_WORKBENCH_SLOT_XYZ,
    E3_M4_WORKBENCH_SLOT_YAW_DEG,
    E3_M4_WORKSTATION_SLOT,
    E3_M4_WORKSTATION_SLOT_XYZ,
)

if TYPE_CHECKING:
    from pxr import Usd


def apply_m4_station_edits(stage: Usd.Stage, cell_root: str) -> None:
    """Author the DB-M4 edits over the assembly line composed under ``cell_root`` (``e3_manufacturing_cell.edit_e3_cell``)."""
    edit_e3_cell(
        stage,
        cell_root,
        slot_poses={
            E3_M4_WORKSTATION_SLOT: (E3_M4_WORKSTATION_SLOT_XYZ, None),
            E3_M4_WORKBENCH_SLOT: (E3_M4_WORKBENCH_SLOT_XYZ, E3_M4_WORKBENCH_SLOT_YAW_DEG),
        },
        deactivated=(*E3_M4_DEACTIVATED_SLOTS, *E3_M4_DEACTIVATED_DRESSING),
    )
