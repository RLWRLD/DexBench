# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""The DB-M4 layout, thread geometry and hand-start window, shared by the environment, its task and its tools."""

from __future__ import annotations

import math
from dataclasses import dataclass

from dexbench_arena.layouts.db_m1_layout import E3_M1_WORKSTATION_SLOT_XYZ
from dexbench_arena.tasks.object_axis import ObjectAxis, TaskObject


WORKBENCH_TOP_Z_M = 0.74
"""The E3 workbench's worktop (``assembly_workbench_fc``'s ``WbFC_Worktop``), which the pallet lies on."""
PALLET_SIZE_M = (0.70, 0.68, 0.04)
PALLET_TOP_Z_M = WORKBENCH_TOP_Z_M + PALLET_SIZE_M[2]
PALLET_CENTER_XYZ = (0.0, 0.0, PALLET_TOP_Z_M - PALLET_SIZE_M[2] / 2)
"""The pallet is a proxy box under the cabinet (the spec's is 0.80 x 0.80 x 0.03 m on the conveyor), spawned as a
kinematic cuboid, cut to the cabinet's footprint so it stays on the workbench and 40 mm thick so the cabinet's
bottom stays at 0.78, where the spec's pallet on the belt puts it."""
BASE_CABINET_ROOT_XYZ = (0.0, 0.0, PALLET_TOP_Z_M)
"""Where the cabinet package's root goes: its frame is the env frame raised by the pallet top."""



TARGET_HOLE_IN_CABINET_XYZ = (-0.32299856533375987, -0.31000200830714114, 0.07410001718111969)
"""``Frames/Hole_1``: the target hole's axis on the bracket top face, the hole mouth. The leg's
origin (its stud tip) is at this point at ``Frames/Leg_Entry``."""
LEG_SEATED_IN_CABINET_Z_M = 0.031100022170402086
"""``Frames/Leg_Seated``: the leg origin's height when fully fastened (the nut runner's end state)."""
HOLE_INSERTION_AXIS_IN_CABINET = (0.0, 0.0, -1.0)
"""The direction the stud advances in, in the cabinet frame (``Leg_Seated`` customData ``axis``)."""
LEG_STUD_AXIS_IN_LEG = (0.0, 0.0, -1.0)
"""The same direction in the leg's frame (the leg's customData ``insertionAxis``: stud pointing down)."""
THREAD_PITCH_M = 0.00125
"""M10 x 1.25 right hand: advance per turn (``Leg_Seated`` customData ``pitch_m``, measured by N.Light
on the CAD thread)."""

THREAD_CATCH_DEPTH_M = 0.00808
"""Depth of the stud tip below the hole mouth where the thread catches it: a leg dropped onto the
hole, not turned, comes to rest here. Pushed down with 20 N and not turned, it stays."""

HAND_START_TURNS = 1
"""Full turns past the catch that make the hand-start. The spec (§4) asks for 360 deg; the real line hand-starts
2 to 3 turns before the nut runner (§2). One turn is one pitch of advance, since the thread does not let the
stud advance without turning."""
HAND_START_DEPTH_M = THREAD_CATCH_DEPTH_M + HAND_START_TURNS * THREAD_PITCH_M
"""Success depth of the stud tip below the hole mouth: 9.33 mm."""

AXIS_MAX_TILT_DEG = 2.0
AXIS_MAX_OFFSET_M = 0.0006
HAND_START_HOLD_S = 0.5
"""The window has to hold this long in sim time, as DB-M1's seat window does."""


LEG_NAME = "leg"
BASE_CABINET_NAME = "base_cabinet"

HOLDER_CENTER_XY = (-0.72, -0.67)
WORKTOP_Z_M = 0.75
HOLDER_FLOOR_Z_M = 0.820
HOLDER_TOP_Z_M = 0.845
HOLDER_HEIGHT_M = HOLDER_TOP_Z_M - WORKTOP_Z_M
HOLDER_OUTER_M = 0.028
HOLDER_BORE_M = 0.0108
_WALL = (HOLDER_OUTER_M - HOLDER_BORE_M) / 2
_WALL_OFFSET = (HOLDER_BORE_M + _WALL) / 2
_HOLDER_Z = HOLDER_TOP_Z_M - HOLDER_HEIGHT_M / 2
HOLDER_WALLS = {
    "leg_holder_south": ((HOLDER_OUTER_M, _WALL, HOLDER_HEIGHT_M), (HOLDER_CENTER_XY[0], HOLDER_CENTER_XY[1] - _WALL_OFFSET, _HOLDER_Z)),
    "leg_holder_north": ((HOLDER_OUTER_M, _WALL, HOLDER_HEIGHT_M), (HOLDER_CENTER_XY[0], HOLDER_CENTER_XY[1] + _WALL_OFFSET, _HOLDER_Z)),
    "leg_holder_west": ((_WALL, HOLDER_BORE_M, HOLDER_HEIGHT_M), (HOLDER_CENTER_XY[0] - _WALL_OFFSET, HOLDER_CENTER_XY[1], _HOLDER_Z)),
    "leg_holder_east": ((_WALL, HOLDER_BORE_M, HOLDER_HEIGHT_M), (HOLDER_CENTER_XY[0] + _WALL_OFFSET, HOLDER_CENTER_XY[1], _HOLDER_Z)),
    "leg_holder_floor": ((HOLDER_BORE_M, HOLDER_BORE_M, HOLDER_FLOOR_Z_M - WORKTOP_Z_M),
                         (*HOLDER_CENTER_XY, (HOLDER_FLOOR_Z_M + WORKTOP_Z_M) / 2)),
}
LEG_START_YAW_RAD = -math.pi / 4
"""The leg's yaw in the socket: its pad faces the hand at the side-pick start."""
LEG_START_XYZ = (*HOLDER_CENTER_XY, HOLDER_FLOOR_Z_M + 0.0001)
"""Stud down, the tip 0.1 mm above the socket floor so it spawns out of contact; it settles onto the floor."""

LEG_STATION_JITTER_M = 0.005
LEG_STATION_MAX_SHIFT_M = 0.080
"""The largest shift any override of that draw may reach; from the default centre it leaves 6 mm of worktop
in front of the holder (``LegStationDomain`` also keeps any other centre on the worktop)."""


class LegStationDomain:
    """The scene's limit on the leg-station draw ``[dx, dy, dyaw]`` (``object_pose_variation.PoseDomain``)."""

    reason = "the leg and its holder pieces move together, unturned, on the workstation top"

    def __init__(self, center_xy=HOLDER_CENTER_XY):
        x, y = center_xy
        if not all(math.isfinite(v) for v in (x, y)):
            raise ValueError("Leg station coordinates must be finite")
        (x_low, x_high), (y_low, y_high) = WORKTOP_X_RANGE_M, WORKTOP_Y_RANGE_M
        self.max_shift_m = min(x - x_low, x_high - x, y - y_low, y_high - y) - HOLDER_OUTER_M / 2
        if self.max_shift_m <= LEG_STATION_JITTER_M:
            raise ValueError("Leg station plus its +/-5 mm draw must fit on the workstation")
        self.max_shift_m = min(LEG_STATION_MAX_SHIFT_M, self.max_shift_m)

    def violation(self, low, high) -> str | None:
        if float(low[2]) != 0.0 or float(high[2]) != 0.0:
            return f"the draw turns the holder by up to {math.degrees(max(abs(float(low[2])), abs(float(high[2])))):.1f} deg"
        reach = max(abs(float(value)) for value in (*low[:2], *high[:2]))
        if reach > self.max_shift_m:
            return f"the draw shifts the holder by up to {reach * 1000:.0f} mm, past {self.max_shift_m * 1000:.0f} mm"
        return None

LEG_DROPPED_Z_M = 0.70

EPISODE_TIMEOUT_S = 60.0


M4_HAND_STATIC_FRICTION = 1.2
M4_HAND_DYNAMIC_FRICTION = 1.0

THREAD_DELIVERED_FRICTION = (0.4, 0.35, 0.005)
THREAD_FRICTION = (0.2, 0.15, 0.005)


@dataclass(frozen=True)
class M4ThreadContactMaterialCfg:
    """ThreadSteel's friction for a DB-M4 run: static, dynamic (dimensionless) and torsional (m)."""

    static_friction: float = THREAD_FRICTION[0]
    dynamic_friction: float = THREAD_FRICTION[1]
    torsional_friction: float = THREAD_FRICTION[2]

    def __post_init__(self) -> None:
        for name in ("static_friction", "dynamic_friction", "torsional_friction"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number, got {value!r}")
        if not 0.1 <= self.dynamic_friction <= self.static_friction <= 0.6:
            raise ValueError("Thread sliding friction must satisfy 0.1 <= dynamic <= static <= 0.6")
        if not 0.0 <= self.torsional_friction <= 0.005:
            raise ValueError("Thread torsional_friction must be within 0 to 0.005 m")



ROBOT_BASE_YAW_DEG = 90.0
ROBOT_BASE_ANCHORS_XYZ = {"A_m4": (-0.20, -0.72, 0.0)}
ANCHOR_FACING_TARGETS_XY = {
    "A_m4": (
        ROBOT_BASE_ANCHORS_XYZ["A_m4"][0] + math.cos(math.radians(ROBOT_BASE_YAW_DEG)),
        ROBOT_BASE_ANCHORS_XYZ["A_m4"][1] + math.sin(math.radians(ROBOT_BASE_YAW_DEG)),
    )
}
START_TORSO_YAW_RAD = math.pi / 4
START_TORSO_CROUCH_RAD = math.radians(15.0)


E3_CELL_XYZ = (3.58, 0.78, 0.0)
E3_CELL_YAW_DEG = 180.0


def env_from_cell(xyz) -> tuple[float, float, float]:
    """A position in the cell's frame in the env frame: turned by ``E3_CELL_YAW_DEG`` about Z, then moved by
    ``E3_CELL_XYZ``."""
    cos_yaw, sin_yaw = math.cos(math.radians(E3_CELL_YAW_DEG)), math.sin(math.radians(E3_CELL_YAW_DEG))
    return (
        E3_CELL_XYZ[0] + cos_yaw * xyz[0] - sin_yaw * xyz[1],
        E3_CELL_XYZ[1] + sin_yaw * xyz[0] + cos_yaw * xyz[1],
        E3_CELL_XYZ[2] + xyz[2],
    )


def cell_from_env(xyz) -> tuple[float, float, float]:
    """An env-frame position in the cell's frame, the inverse of :func:`env_from_cell`."""
    cos_yaw, sin_yaw = math.cos(math.radians(E3_CELL_YAW_DEG)), math.sin(math.radians(E3_CELL_YAW_DEG))
    dx, dy = xyz[0] - E3_CELL_XYZ[0], xyz[1] - E3_CELL_XYZ[1]
    return (cos_yaw * dx + sin_yaw * dy, -sin_yaw * dx + cos_yaw * dy, xyz[2] - E3_CELL_XYZ[2])


E3_M4_WORKSTATION_SLOT = "FixtureSlots/Slot_Workstation"
E3_M4_WORKSTATION_SLOT_XYZ = E3_M1_WORKSTATION_SLOT_XYZ
"""Where DB-M1 moves the workstation; its delivered turn of -90 deg stays."""
WORKSTATION_ROOT_XYZ = env_from_cell(E3_M4_WORKSTATION_SLOT_XYZ)
WORKSTATION_YAW_DEG = -90.0 + E3_CELL_YAW_DEG
WORKSTATION_WORKTOP_M = (0.80, 1.40)
"""The workstation's footprint (its asset's bounds): depth back from the front edge (-X) and length along Y,
centred on the slot. Those axes are the workstation's at :data:`WORKSTATION_YAW_DEG` 90 (its delivered -90 in the
half-turned cell); another turn of the cell turns the footprint too, and the ranges below would have to follow."""
WORKTOP_X_RANGE_M = (WORKSTATION_ROOT_XYZ[0] - WORKSTATION_WORKTOP_M[0], WORKSTATION_ROOT_XYZ[0])
WORKTOP_Y_RANGE_M = (
    WORKSTATION_ROOT_XYZ[1] - WORKSTATION_WORKTOP_M[1] / 2,
    WORKSTATION_ROOT_XYZ[1] + WORKSTATION_WORKTOP_M[1] / 2,
)
E3_M4_WORKBENCH_SLOT = "FixtureSlots/Slot_Workbench"
WORKBENCH_ROOT_XYZ = (0.30, -0.37, 0.0)
"""The workbench's front-edge centre: 30 mm in front of the pallet's -Y side, so the cabinet's draw keeps the pallet on
the worktop; 0.35 m ahead of the robot's base, whose front reaches 39 mm under the worktop."""
WORKBENCH_YAW_DEG = 0.0
WORKBENCH_WORKTOP_M = (1.80, 1.00)
"""The workbench's footprint (its asset's bounds): length along X centred on the slot, depth back from the front
edge (+Y)."""
WORKBENCH_WORKTOP_X_RANGE_M = (WORKBENCH_ROOT_XYZ[0] - WORKBENCH_WORKTOP_M[0] / 2, WORKBENCH_ROOT_XYZ[0] + WORKBENCH_WORKTOP_M[0] / 2)
WORKBENCH_WORKTOP_Y_RANGE_M = (WORKBENCH_ROOT_XYZ[1], WORKBENCH_ROOT_XYZ[1] + WORKBENCH_WORKTOP_M[1])
E3_M4_WORKBENCH_SLOT_XYZ = cell_from_env(WORKBENCH_ROOT_XYZ)
E3_M4_WORKBENCH_SLOT_YAW_DEG = WORKBENCH_YAW_DEG - E3_CELL_YAW_DEG
E3_M4_DEACTIVATED_SLOTS = (
    "FixtureSlots/Slot_StationWorkpiece",
    "FixtureSlots/Slot_StationPcTower",
    "FixtureSlots/Slot_StationHangerRail",
    "FixtureSlots/Slot_BenchJig",
    "FixtureSlots/Slot_BenchFeedTray",
    "FixtureSlots/Slot_ToolZone_DriverStand",
    "FixtureSlots/Slot_ToolZone_NutTray",
)
E3_M4_DEACTIVATED_DRESSING = (
    "Background/BG_S/Geom/DR_S01_0",
    "Background/BG_S/Geom/DR_S03_0",
    "BackgroundCollision/COLBO_S03",
    *(
        f"Background/BG_FLOOR/Geom/DR_F01__A_station_{corner}_{axis}_a0_mark"
        for corner in ("NE", "NW", "SE", "SW")
        for axis in ("x", "y")
    ),
    "Background/BG_FLOOR/Geom/DR_F02__A_station_label_a0_plate",
    "Background/BG_FLOOR/Geom/DR_F02__A_station_label_a1_num",
)


CABINET_STATION_JITTER_M = 0.02
CABINET_STATION_MAX_ABS_XY_M = (
    min(-PALLET_SIZE_M[0] / 2 - WORKBENCH_WORKTOP_X_RANGE_M[0], WORKBENCH_WORKTOP_X_RANGE_M[1] - PALLET_SIZE_M[0] / 2),
    min(-PALLET_SIZE_M[1] / 2 - WORKBENCH_WORKTOP_Y_RANGE_M[0], WORKBENCH_WORKTOP_Y_RANGE_M[1] - PALLET_SIZE_M[1] / 2),
)
ROBOT_BASE_JITTER_M = 0.02

INTERIOR_LIGHT_XYZ = (-0.3, -0.6, 2.6)


M4_LEG_OBJECT_AXIS = ObjectAxis(
    field="m4_leg",
    default="lg_fx25_leveling_leg",
    objects=(
        TaskObject(
            key="lg_fx25_leveling_leg",
            asset="lg_fx25_leveling_leg",
            split="train",
            label="leveling leg",
            size_m=(0.034, 0.034, 0.059),
            benchmark_asset_major_renames=((1, 2,
                "assets/object_library/lg_fx25_leveling_leg/CHANGELOG.md"),),
        ),
    ),
    size_axes="width, depth, height",
)
