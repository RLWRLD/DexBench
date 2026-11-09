# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""IsaacTeleop pipeline builder for RBY1-M Wuji Hand 2 OpenXR teleoperation."""

from __future__ import annotations

from pathlib import Path

from dexbench_arena.embodiments.rby1m.openxr_teleop import build_rby1m_openxr_pipeline
from dexbench_arena.embodiments.rby1_wuji2.constants import (
    RBY1_WUJI2_HAND_JOINT_ACTION_NAMES,
    RBY1_WUJI2_LEFT_RETARGET_URDF_PATH,
    RBY1_WUJI2_LEFT_WRIST_IDLE_ROOT_POSE,
    RBY1_WUJI2_OPENXR_ACTION_ORDER,
    RBY1_WUJI2_OPERATOR_TO_MANO_LEFT,
    RBY1_WUJI2_OPERATOR_TO_MANO_RIGHT,
    RBY1_WUJI2_RIGHT_RETARGET_URDF_PATH,
    RBY1_WUJI2_RIGHT_WRIST_IDLE_ROOT_POSE,
    RBY1_WUJI2_XR_WRIST_OFFSET_RPY_LEFT,
    RBY1_WUJI2_XR_WRIST_OFFSET_RPY_RIGHT,
)

_DATA_DIR = Path(__file__).resolve().parent / "data" / "dex_retargeting"
_LEFT_HAND_CONFIG = _DATA_DIR / "wujihand2_left_dexpilot.yml"
_RIGHT_HAND_CONFIG = _DATA_DIR / "wujihand2_right_dexpilot.yml"


def build_rby1_wuji2_openxr_pipeline(
    arm_mode: object | None = None,
    idle_wrist_poses: dict[str, tuple[float, ...]] | None = None,
):
    """Build the RBY1-M Wuji Hand 2 OpenXR pipeline."""

    for path in (_LEFT_HAND_CONFIG, _RIGHT_HAND_CONFIG):
        if not path.exists():
            raise FileNotFoundError(f"wujihand2 dex-retargeting config missing: {path}")
    for path in (RBY1_WUJI2_LEFT_RETARGET_URDF_PATH, RBY1_WUJI2_RIGHT_RETARGET_URDF_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"wujihand2 retarget URDF missing: {path}. "
                "It ships with the robot asset under assets/, so this checkout is incomplete."
            )

    return build_rby1m_openxr_pipeline(
        arm_mode=arm_mode,
        idle_wrist_poses=idle_wrist_poses,
        action_order=RBY1_WUJI2_OPENXR_ACTION_ORDER,
        left_hand_config=_LEFT_HAND_CONFIG,
        right_hand_config=_RIGHT_HAND_CONFIG,
        left_hand_urdf=RBY1_WUJI2_LEFT_RETARGET_URDF_PATH,
        right_hand_urdf=RBY1_WUJI2_RIGHT_RETARGET_URDF_PATH,
        default_left_idle_pose=RBY1_WUJI2_LEFT_WRIST_IDLE_ROOT_POSE,
        default_right_idle_pose=RBY1_WUJI2_RIGHT_WRIST_IDLE_ROOT_POSE,
        operator_to_mano_left=RBY1_WUJI2_OPERATOR_TO_MANO_LEFT,
        operator_to_mano_right=RBY1_WUJI2_OPERATOR_TO_MANO_RIGHT,
        wrist_offset_rpy_left=RBY1_WUJI2_XR_WRIST_OFFSET_RPY_LEFT,
        wrist_offset_rpy_right=RBY1_WUJI2_XR_WRIST_OFFSET_RPY_RIGHT,
    )
