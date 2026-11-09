# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""The DB-M1 tabletop start-state envelope: the box Train's rows are drawn in, and GA-S's scale of it."""

from __future__ import annotations

import math

from dexbench_arena.embodiments.rby1_wuji2.constants import RBY1_WUJI2_JOINT_EMBODIMENT

TABLETOP_TRAIN_BOUNDS_M_RAD = {
    f"{RBY1_WUJI2_JOINT_EMBODIMENT}.base_pose": (0.03, 0.03, math.radians(5)),
    "drawer.pose": (0.05, 0.05, math.radians(5)),
    "washer.pose": (0.05, 0.05, math.radians(5)),
}
"""Half-extents ``(dx m, dy m, dyaw rad)`` of each recorded pose draw in the Train cell, keyed as the rows record it."""

TABLETOP_GA_S_SCALE = 1.5
"""GA-S draws from this multiple of the Train box, keeping only rows outside Train on at least one coordinate."""
