# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""DB-R1 task layout on the reviewed 260821 E1 retail scene (produce display, Fruit station)."""

from __future__ import annotations


FC_260821_R1_ANCHOR = "Fruit"
FC_260821_R1_DISPLAY_INSTANCE_NAME = "retail_e1_fc_produce_display"
FC_260821_R1_BASKET_INSTANCE_NAME = "retail_e1_fc_fruit_basket"
FC_260821_R1_CART_INSTANCE_NAME = "retail_e1_fc_cart_r1"
FC_260821_R1_FIXTURE_INSTANCE_NAMES = (
    FC_260821_R1_DISPLAY_INSTANCE_NAME, FC_260821_R1_CART_INSTANCE_NAME, FC_260821_R1_BASKET_INSTANCE_NAME,
)
"""The fixtures DB-R1 touches: what ``r1_minimal_scene`` keeps."""

FC_260821_R1_DISPLAY_ORIGIN_XY_M = (3.32, 0.70)
FC_260821_R1_TRAY_FRONT_PLANE_X_M = 3.0423

FC_260821_R1_PILE_TRAY_BIN = (2, 1)
"""The pile compartment as the display dressing numbers it (tray, bin); its dressing gives way to the pile."""

FC_260821_R1_BASKET_ORIGIN_XYZ_M = (2.60, 0.25, 0.60)
FC_260821_R1_BASKET_INTERIOR_HALF_EXTENTS_XY_M = (0.1775, 0.1175)
FC_260821_R1_BASKET_FLOOR_TOP_Z_M = 0.603
FC_260821_R1_BASKET_WALL_TOP_Z_M = 0.830
FC_260821_R1_BASKET_BOUNDS = (
    (FC_260821_R1_BASKET_ORIGIN_XYZ_M[0] - FC_260821_R1_BASKET_INTERIOR_HALF_EXTENTS_XY_M[0],
     FC_260821_R1_BASKET_ORIGIN_XYZ_M[1] - FC_260821_R1_BASKET_INTERIOR_HALF_EXTENTS_XY_M[1],
     FC_260821_R1_BASKET_FLOOR_TOP_Z_M),
    (FC_260821_R1_BASKET_ORIGIN_XYZ_M[0] + FC_260821_R1_BASKET_INTERIOR_HALF_EXTENTS_XY_M[0],
     FC_260821_R1_BASKET_ORIGIN_XYZ_M[1] + FC_260821_R1_BASKET_INTERIOR_HALF_EXTENTS_XY_M[1],
     FC_260821_R1_BASKET_WALL_TOP_Z_M),
)
"""The basket's interior as a world AABB: inside the walls, from the floor top to the wall top."""
FC_260821_R1_BASKET_MARGIN_M = 0.004
"""How far a placed fruit's extent may reach past the interior (wall and mesh tolerance)."""


FC_260821_R1_WORKSPACE_XY_BOUNDS = ((2.00, -0.20), (3.70, 1.60))
FC_260821_R1_OBJECT_DROPPED_MIN_Z_M = 0.45
FC_260821_R1_HOLD_S = 1.0
FC_260821_R1_RELEASE_WEIGHT_FRACTION = 0.25
"""How long the placed fruit stays at rest and let go in the basket, the others on the tray."""
FC_260821_R1_EPISODE_LENGTH_S = 60.0


__all__ = [name for name in dir() if name.startswith("FC_260821_R1_")]
