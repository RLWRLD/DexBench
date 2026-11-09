# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""The DB-M1 layout and seat geometry, shared by the environment, its task and its tools."""

from __future__ import annotations

import math
from dataclasses import replace

from dexbench_arena.tasks.object_axis import ObjectAxis, TaskObject


ENV_FROM_RIG_OFFSET_M = (0.0, 0.0, 0.25)
"""Added to a position in the rig's N.Light washer frame to give the env frame (floor at z 0)."""


def env_from_rig(xyz) -> tuple[float, float, float]:
    """A rig-frame position in this environment's frame."""
    return tuple(float(value) + offset for value, offset in zip(xyz, ENV_FROM_RIG_OFFSET_M))



WASHER_SHIFT_M = (0.0766, 0.0643, 0.0)
WASHER_ROOT_XYZ = env_from_rig(WASHER_SHIFT_M)
"""Where the washer asset's root goes (its body stands on the pallet top, z 0.25)."""

SEAT_IN_WASHER_XYZ = (-0.2264, -0.3877, 0.9092)
_RAIL_ENTRY_Y_IN_WASHER = -0.7377
_RAILS_ENGAGED_Y_IN_WASHER = -0.6877


def _in_env(point_in_washer) -> tuple[float, float, float]:
    return env_from_rig(tuple(p + s for p, s in zip(point_in_washer, WASHER_SHIFT_M)))


SEAT_XYZ = _in_env(SEAT_IN_WASHER_XYZ)
"""Drawer origin at N.Light's DrawerSeated, in the env frame. The drawer is seated upright (identity rotation)."""
RAIL_ENTRY_Y_M = _in_env((0.0, _RAIL_ENTRY_Y_IN_WASHER, 0.0))[1]
RAILS_ENGAGED_Y_M = _in_env((0.0, _RAILS_ENGAGED_Y_IN_WASHER, 0.0))[1]

SEAT_PROXY_HOME_DY_M = -0.009

SEAT_DEPTH_WINDOW_M = (-0.012, 0.002)
SEAT_MAX_LATERAL_ERROR_M = 0.002
SEAT_MAX_TILT_DEG = 1.5
SEAT_HOLD_S = 0.5

PROGRESS_RAIL_ENTRY_DEPTH_M = _RAIL_ENTRY_Y_IN_WASHER - SEAT_IN_WASHER_XYZ[1]
PROGRESS_RAILS_ENGAGED_DEPTH_M = _RAILS_ENGAGED_Y_IN_WASHER - SEAT_IN_WASHER_XYZ[1]
PROGRESS_HALF_DEPTH_M = PROGRESS_RAIL_ENTRY_DEPTH_M / 2
PROGRESS_ALIGN_LEAD_M = 0.05
PROGRESS_ALIGN_MAX_LATERAL_ERROR_M = 0.020
PROGRESS_ALIGN_MAX_TILT_DEG = 3.0

DRAWER_DROP_M = 0.10

EPISODE_TIMEOUT_S = 68.2
DRAWER_DROPPED_Z_M = SEAT_XYZ[2] - DRAWER_DROP_M

START_DEPTH_M = -0.25
DRAWER_NOMINAL_START_XYZ = (SEAT_XYZ[0], RAIL_ENTRY_Y_M + START_DEPTH_M, SEAT_XYZ[2])

TABLETOP_WORKTOP_Z_M = 0.75
TABLETOP_DRAWER_START_XYZ = (-0.43, -1.15, TABLETOP_WORKTOP_Z_M + 0.056)
TABLETOP_DRAWER_YAW_DEG = 90.0
TABLETOP_DRAWER_DROPPED_Z_M = 0.20
TABLETOP_EPISODE_TIMEOUT_S = 90.0
TABLETOP_SEAT_DEPTH_WINDOW_M = (-0.001, SEAT_DEPTH_WINDOW_M[1])
TABLETOP_SEAT_MAX_TILT_DEG = 1.0
TABLETOP_SEAT_MAX_LINEAR_SPEED_M_S = 0.01

TABLETOP_SUPPORT_BOUNDS_XY_M = ((-1.2, -0.4), (-1.9757, -0.5757))
TABLETOP_DRAWER_LOCAL_BOUNDS_XY_M = ((-0.093, 0.093), (0.0, 0.5392))
TABLETOP_WASHER_LOCAL_BOUNDS_XY_M = ((-0.4833, 0.4833), (-0.4129, 0.4129))
TABLETOP_SPATIAL_PILOT_BOUNDS = {
    "drawer": ((-0.02, -0.02, -math.radians(5)), (0.02, 0.02, math.radians(5))),
    "washer": ((-0.03, -0.03, -math.radians(0.5)), (0.03, 0.002, math.radians(0.5))),
    "base": ((-0.03, -0.03, -math.radians(3)), (0.03, 0.03, math.radians(3))),
}


ROBOT_BASE_YAW_DEG = 130.0
ROBOT_BASE_ANCHORS_XYZ = {"A_m1": env_from_rig((0.15, -0.9757, -0.25))}
ANCHOR_FACING_TARGETS_XY = {
    "A_m1": (
        ROBOT_BASE_ANCHORS_XYZ["A_m1"][0] + math.cos(math.radians(ROBOT_BASE_YAW_DEG)),
        ROBOT_BASE_ANCHORS_XYZ["A_m1"][1] + math.sin(math.radians(ROBOT_BASE_YAW_DEG)),
    )
}


WASHER_LINE_SIZE_M = (2.4, 0.96, 0.25)
WASHER_LINE_CENTER_XYZ = env_from_rig((0.0, 0.33 - 0.3257, -0.125))


E3_CELL_XYZ = env_from_rig((3.8, 0.5243, -0.25))
E3_CELL_YAW_DEG = 180.0

E3_M1_WORKSTATION_SLOT = "FixtureSlots/Slot_Workstation"
E3_M1_WORKSTATION_SLOT_XYZ = (4.2, 1.8, 0.0)
E3_M1_DEACTIVATED_SLOTS = (
    "FixtureSlots/Slot_StationWorkpiece",
    "FixtureSlots/Slot_StationPcTower",
    "FixtureSlots/Slot_StationHangerRail",
)
E3_M1_DEACTIVATED_DRESSING = (
    "Background/BG_S/Geom/DR_S01_0",
    *(
        f"Background/BG_FLOOR/Geom/DR_F01__A_station_{corner}_{axis}_a0_mark"
        for corner in ("NE", "NW", "SE", "SW")
        for axis in ("x", "y")
    ),
    "Background/BG_FLOOR/Geom/DR_F02__A_station_label_a0_plate",
    "Background/BG_FLOOR/Geom/DR_F02__A_station_label_a1_num",
)

INTERIOR_LIGHT_XYZ = (0.0, -0.75, 2.6)


DRAWER_NAME = "drawer"
WASHER_NAME = "washer"

DRAWER_SENSING_COLLIDERS = ("Body/CollisionProxy/.*", "Body/CollisionTray/.*")
WASHER_COUNTERPART_COLLIDERS = ("Body/CollisionProxy/.*", "Body/CollisionSlot/.*")
SLOT_CONTACT_REGIONS = {
    "panel": ('bottom_plate', 'front_lip_low', 'front_wall', 'hollow_fin', 'panel_left', 'panel_right', 'pocket_back', 'pocket_floor_0', 'pocket_floor_1', 'pocket_floor_2', 'pocket_floor_3', 'top_shell'),
    "left_wall": ('tray_00', 'tray_05', 'tray_06', 'tray_09', 'tray_14', 'tray_24', 'tray_27', 'tray_59'),
    "bottom": ('tray_01', 'tray_03', 'tray_04', 'tray_11', 'tray_13', 'tray_15', 'tray_16', 'tray_19', 'tray_20', 'tray_26', 'tray_34', 'tray_36', 'tray_37', 'tray_40', 'tray_43', 'tray_44', 'tray_48', 'tray_49', 'tray_50', 'tray_51', 'tray_52', 'tray_53'),
    "top": ('tray_02', 'tray_17', 'tray_23', 'tray_28', 'tray_46', 'tray_56'),
    "rear_lead": ('tray_07', 'tray_08', 'tray_18', 'tray_21', 'tray_29', 'tray_31', 'tray_32', 'tray_35', 'tray_38', 'tray_45', 'tray_47', 'tray_54', 'tray_55', 'tray_58'),
    "right_wall": ('tray_10', 'tray_12', 'tray_30', 'tray_33', 'tray_39', 'tray_42'),
    "tray_mid": ('tray_22', 'tray_25', 'tray_41', 'tray_57'),
}


M1_DRAWER_OBJECT_AXIS = ObjectAxis(
    field="m1_drawer",
    default="lg_fh25eae_detergent_drawer",
    objects=(
        TaskObject(
            key="lg_fh25eae_detergent_drawer",
            asset="lg_fh25eae_detergent_drawer",
            split="train",
            label="detergent drawer",
            size_m=(0.20, 0.54, 0.09),
        ),
    ),
    size_axes="width, depth, height",
)

M1_TABLETOP_DRAWER_OBJECT_AXIS = replace(
    M1_DRAWER_OBJECT_AXIS, scene_layout="tabletop", unlabelled_scene_layout="tabletop",
)
