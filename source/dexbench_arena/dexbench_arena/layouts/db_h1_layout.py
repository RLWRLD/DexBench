"""The DB-H1 layout and plate-stack geometry, shared by the environment, its task and its tools."""

from __future__ import annotations

import math

from dexbench_arena.embodiments.rby1m.constants import rby1m_torso_z_fold


PLATE_SPAWN_GAP_M = 0.0


ROUND_TABLE_XY_M = (2.49, 1.37)
ROUND_TABLE_RADIUS_M = 0.610
ROUND_TABLE_TOP_Z_M = 0.740
CLEARING_SIZE_XY_M = (0.900, 0.600)
CLEARING_TOP_Z_M = 0.800
CLEARING_ROBOT_EDGE_Y_M = 1.768
CLEARING_XY_M = (1.48, 2.068)

ROBOT_BASE_ANCHORS_XYZ = {
    "A_h1": (1.48, 1.37, 0.0),
    "A_h2": (3.55, 1.75, 0.0),
    "A_h3": (3.55, 1.00, 0.0),
}

ANCHOR_FACING_TARGETS_XY = {
    "A_h1": (2.48, 1.37),
    "A_h2": (3.55, 2.75),
    "A_h3": (3.55, 0.00),
}

STACK_RIM_INSET_M = 0.132
SOURCE_STACK_XY_M = (2.012, 1.370)
DEST_STACK_XY_M = (1.480, 1.900)
DEFAULT_SOURCE_PLATE_COUNT = 3
DEFAULT_DEST_PLATE_COUNT = 2

POUR_JUG_XYZ_M = (3.71, 2.39, 0.805)
POUR_GOBLET_XYZ_M = (3.52, 2.42, 0.803)
POUR_DECOR_GOBLETS_XY_M = {"decor_goblet_a": (3.52, 2.60), "decor_goblet_b": (3.73, 2.56)}

H1_TORSO_CROUCH_RAD = 0.8029

H1_ARM_START_POSE_RAD = {
    "right_arm_0": 0.5236,
    "right_arm_1": -0.4887,
    "right_arm_2": 0.0,
    "right_arm_3": -1.8675,
    "right_arm_4": -0.0524,
    "right_arm_5": -0.5236,
    "right_arm_6": -0.5061,
    "left_arm_0": 0.4887,
    "left_arm_1": 0.4189,
    "left_arm_2": 0.0,
    "left_arm_3": -1.8151,
    "left_arm_4": -0.0524,
    "left_arm_5": -0.5236,
    "left_arm_6": 0.3840,
}

E4_RENDER_SETTINGS: dict[str, int | float] = {
    "rtx:pathtracing:maxBounces": 8,
    "rtx:pathtracing:maxSpecularAndTransmissionBounces": 16,
    "rtx:post:tonemap:exposureTime": 0.75264,
    "rtx:post:tonemap:fNumber": 2.8,
    "rtx:post:tonemap:filmIso": 1600.0,
    "rtx:post:tonemap:irayReinhard:burnHighlights": 0.0,
    "rtx:post:tonemap:irayReinhard:crushBlacks": 0.0,
    "rtx:post:tonemap:irayReinhard:saturation": 0.7,
    "rtx:post:tonemap:op": 7,
    "rtx:rtpt:maxSpecularAndTransmissionBounces": 16,
    "rtx:translucency:maxRefractionBounces": 16,
}

POUR_RENDER_SETTINGS: dict[str, bool] = {
    "rtx:rtpt:fireflyFilter:enabled": False,
}

LIGHT_INTENSITY_SCALE_RANGE = (0.25, 1.5)
LIGHT_COLOR_TEMPERATURE_RANGE_K = (3000.0, 9000.0)


RIM_OFFSET_BUDGET_MM = 30.0
LEAN_BUDGET_DEG = 30.0
SEAT_HEIGHT_BUDGET_MM = 10.0
HANDS_CLEAR_MARGIN_M = 0.03
HANDS_CLEAR_HEIGHT_M = 0.03



BASE_REACH_WINDOW_M = 0.07

PLATE_MAX_OVERHANG_FRACTION = 0.30

H1_HAND_STATIC_FRICTION = 1.2
H1_HAND_DYNAMIC_FRICTION = 1.0


def initial_stack_height_error_m(heights: dict[str, float], initial_heights: dict[str, float]) -> float:
    """Largest settling displacement; missing or non-finite plates invalidate a smoke run."""
    if not initial_heights or heights.keys() != initial_heights.keys():
        return math.inf
    if not all(math.isfinite(z) for z in (*heights.values(), *initial_heights.values())):
        return math.inf
    return max(abs(heights[name] - z) for name, z in initial_heights.items())


def torso_crouch_joint_pos(base_joint_pos: dict[str, float], theta_rad: float) -> dict[str, float]:
    """Copy of ``base_joint_pos`` with the torso folded into a Z-fold crouch."""

    pose = dict(base_joint_pos)
    for name, value in rby1m_torso_z_fold(theta_rad).items():
        if name not in pose:
            raise KeyError(f"torso crouch expects {name} in the start pose (got {sorted(pose)[:8]}...)")
        pose[name] = value
    return pose


def stack_region_radius_m(plate_diameter_m: float) -> float:
    """Membership radius around a stack axis, for deciding which plates are in it: one plate radius."""

    return plate_radius_m(plate_diameter_m)


def plate_radius_m(diameter_m: float) -> float:
    return float(diameter_m) / 2.0
