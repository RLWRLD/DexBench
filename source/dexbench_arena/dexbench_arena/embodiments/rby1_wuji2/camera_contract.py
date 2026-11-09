# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Dataset/policy image names and their Wuji2 simulator observations (Kit-free)."""

EGO_CAMERAS = {
    "camera_ego_left": "robot_pov_left_cam_rgb",
    "camera_ego_right": "robot_pov_right_cam_rgb",
}
WRIST_CAMERAS = {
    "camera_wrist_left": "wrist_left_cam_rgb",
    "camera_wrist_right": "wrist_right_cam_rgb",
}
CAMERA_OBSERVATIONS = {**EGO_CAMERAS, **WRIST_CAMERAS}


def camera_mapping(keys):
    """Resolve a declared camera set; unknown, empty or repeated keys are errors."""
    keys = list(keys)
    if not keys or len(keys) != len(set(keys)) or set(keys) - CAMERA_OBSERVATIONS.keys():
        raise ValueError(f"Unsupported Wuji2 camera keys: {keys}")
    return {key: CAMERA_OBSERVATIONS[key] for key in keys}
