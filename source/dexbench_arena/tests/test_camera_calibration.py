# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Calibration remains usable without the private recording integration or Kit."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from dexbench_arena.embodiments import camera_catalog as calibration


def test_stereo_calibration_from_runtime_sensors():
    def sensor(x):
        return SimpleNamespace(
            cfg=SimpleNamespace(width=1280, height=720, offset=SimpleNamespace(pos=(x, 0, 0))),
            data=SimpleNamespace(intrinsic_matrices=np.array([[
                [600, 0, 640], [0, 601, 360], [0, 0, 1],
            ]])),
        )
    env = SimpleNamespace(scene=SimpleNamespace(sensors={
        "robot_pov_left_cam": sensor(0),
        "robot_pov_right_cam": sensor(0.063),
        "wrist_left_cam": sensor(0.2),
        "wrist_right_cam": sensor(-0.2),
        "unrelated_sensor": object(),
    }))
    result = calibration.parse_runtime_camera_calibration(env)
    assert set(result) == {"robot_pov_left_cam_rgb", "robot_pov_right_cam_rgb", "wrist_left_cam_rgb", "wrist_right_cam_rgb"}
    for name, values in result.items():
        if name.startswith("wrist_"):
            assert "baseline_m" not in values
            assert values["fx"] == 600
            continue
        assert values == {
            "model": "pinhole", "width": 1280, "height": 720,
            "fx": 600, "fy": 601, "cx": 640, "cy": 360,
            "baseline_m": pytest.approx(0.063),
        }


def test_no_cameras():
    assert calibration.parse_runtime_camera_calibration(SimpleNamespace()) == {}
