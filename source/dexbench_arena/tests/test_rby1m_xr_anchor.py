# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""The RBY1-M XR anchor faces the operator along the base's current heading."""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("isaaclab_teleop", exc_type=ImportError)

import torch  # noqa: E402
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz  # noqa: E402
from isaaclab_teleop.xr_cfg import XrAnchorRotationMode  # noqa: E402

from dexbench_arena.embodiments.rby1m.xr import rby1m_anchor_rotation, rby1m_xr_cfg  # noqa: E402


def _base_pose(yaw_deg: float, roll_deg: float = 0.0, pitch_deg: float = 0.0) -> np.ndarray:
    quat = quat_from_euler_xyz(*(torch.tensor([math.radians(a)]) for a in (roll_deg, pitch_deg, yaw_deg)))
    return np.array([2.63, 1.86, 0.0, *quat[0].tolist()])


def _facing_deg(anchor_xyzw: np.ndarray) -> float:
    yaw = euler_xyz_from_quat(torch.as_tensor(anchor_xyzw, dtype=torch.float64).unsqueeze(0))[2]
    return math.degrees(float(yaw[0])) + 90.0


@pytest.mark.parametrize("base_yaw_deg", [0.0, 90.0, 180.0, -90.0])
def test_the_operator_faces_the_base_heading_whatever_the_base_yaw(base_yaw_deg):
    facing = _facing_deg(rby1m_anchor_rotation(np.zeros(7), _base_pose(base_yaw_deg)))
    assert math.isclose((facing - base_yaw_deg + 180.0) % 360.0 - 180.0, 0.0, abs_tol=1e-2)


def test_a_tilted_base_still_faces_its_heading():
    facing = _facing_deg(rby1m_anchor_rotation(np.zeros(7), _base_pose(90.0, roll_deg=15.0, pitch_deg=-10.0)))
    assert math.isclose(facing, 90.0, abs_tol=1e-2)


def test_the_family_cfg_is_anchored_to_the_base_with_the_custom_rotation():
    cfg = rby1m_xr_cfg("/World/envs/env_.*/Robot/base")
    assert cfg.anchor_prim_path == "/World/envs/env_.*/Robot/base"
    assert cfg.anchor_rotation_mode is XrAnchorRotationMode.CUSTOM
    assert cfg.anchor_rotation_custom_func is rby1m_anchor_rotation
    assert cfg.fixed_anchor_height
