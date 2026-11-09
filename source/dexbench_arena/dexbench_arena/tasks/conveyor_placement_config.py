# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Shared E2 placement geometry and thresholds; safe to import before Kit starts."""

CONVEYOR_PLACE_XY_RECTS_M = (
    ((1.0745, 0.425), (2.9545, 1.175)),
    ((3.0045, 0.350), (3.8325, 1.250)),
    ((3.8325, 0.425), (4.1925, 1.175)),
)
CONVEYOR_PLACE_TOP_Z_RANGE_M = (0.800, 0.821)
CONVEYOR_PLACE_BOTTOM_HEIGHT_TOLERANCE_M = 0.015
CONVEYOR_PLACE_MAX_FACE_TILT_DEG = 20.0
CONVEYOR_PLACE_LINEAR_SPEED_M_S = 0.05
CONVEYOR_PLACE_ANGULAR_SPEED_RAD_S = 0.25
CONVEYOR_PLACE_HOLD_TIME_S = 0.5

CONVEYOR_SET_DOWN_XY_RECT_M = ((CONVEYOR_PLACE_XY_RECTS_M[0][0][0], 0.45), (2.10, 1.15))
CONVEYOR_SET_DOWN_TOP_Z_M = 0.812
CONVEYOR_SET_DOWN_MARGIN_M = 0.010


def conveyor_placement_params() -> dict:
    """Fresh keyword arguments shared by the environment and offline geometry checks."""
    return {
        "surface_xy_rects": CONVEYOR_PLACE_XY_RECTS_M,
        "surface_top_z_range_m": CONVEYOR_PLACE_TOP_Z_RANGE_M,
        "bottom_height_tolerance_m": CONVEYOR_PLACE_BOTTOM_HEIGHT_TOLERANCE_M,
        "max_face_tilt_deg": CONVEYOR_PLACE_MAX_FACE_TILT_DEG,
        "velocity_threshold_m_s": CONVEYOR_PLACE_LINEAR_SPEED_M_S,
        "angular_velocity_threshold_rad_s": CONVEYOR_PLACE_ANGULAR_SPEED_RAD_S,
    }


def set_down_x_m(depth_along_x_m: float) -> float:
    """x of a box's centre set down on the station's stretch, its upstream end 15 mm inside the stretch's inset upstream edge."""
    upstream_x = CONVEYOR_SET_DOWN_XY_RECT_M[0][0] + CONVEYOR_SET_DOWN_MARGIN_M
    return upstream_x + 0.015 + depth_along_x_m / 2.0
