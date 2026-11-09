# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""DB-M4's hand-start window is measured in the cabinet's frame, along the hole axis, and leaves the leg's turn free."""

import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("isaaclab_arena", reason="needs isaaclab_arena on the path")

from isaaclab.utils.math import quat_apply, quat_from_angle_axis, quat_mul  # noqa: E402

from dexbench_arena.layouts.db_m4_layout import (  # noqa: E402
    AXIS_MAX_OFFSET_M,
    AXIS_MAX_TILT_DEG,
    HAND_START_DEPTH_M,
    HOLE_INSERTION_AXIS_IN_CABINET,
    LEG_STUD_AXIS_IN_LEG,
    TARGET_HOLE_IN_CABINET_XYZ,
)
from dexbench_arena.tasks.predicates.threaded_fastening import stud_in_hand_start_window  # noqa: E402

CABINET_LAYOUT = (0.0, 0.0, 0.78, 0.0, 0.0, 0.0, 1.0)
CABINET_MOVED = (0.4, -1.1, 0.8, 0.0, 0.0, 0.3826834, 0.9238795)


def _env(cabinet_pose, depth_m, offset_xy_m=(0.0, 0.0), spin_deg=0.0, tilt_deg=0.0):
    """A stub env with the leg's tip ``depth_m`` down the target hole, offset in the cabinet's x/y, spun and tilted."""
    cabinet = torch.tensor([cabinet_pose], dtype=torch.float64)
    tip_in_cabinet = torch.tensor(
        [[TARGET_HOLE_IN_CABINET_XYZ[0] + offset_xy_m[0], TARGET_HOLE_IN_CABINET_XYZ[1] + offset_xy_m[1],
          TARGET_HOLE_IN_CABINET_XYZ[2] - depth_m]],
        dtype=torch.float64,
    )
    spin = quat_from_angle_axis(torch.tensor([math.radians(spin_deg)], dtype=torch.float64), torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64))
    tilt = quat_from_angle_axis(torch.tensor([math.radians(tilt_deg)], dtype=torch.float64), torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64))
    leg_q = quat_mul(cabinet[:, 3:], quat_mul(tilt, spin))
    leg = torch.cat((cabinet[:, :3] + quat_apply(cabinet[:, 3:], tip_in_cabinet), leg_q), -1)
    poses = {"leg": leg, "base_cabinet": cabinet}
    return SimpleNamespace(num_envs=1, arena_world=SimpleNamespace(get_pose_w=poses.__getitem__))


def _in_window(env) -> bool:
    return bool(
        stud_in_hand_start_window(
            env, "leg", "base_cabinet", TARGET_HOLE_IN_CABINET_XYZ, HOLE_INSERTION_AXIS_IN_CABINET, LEG_STUD_AXIS_IN_LEG,
            HAND_START_DEPTH_M, AXIS_MAX_OFFSET_M, AXIS_MAX_TILT_DEG,
        )[0]
    )


@pytest.mark.parametrize("cabinet", [CABINET_LAYOUT, CABINET_MOVED], ids=["layout", "moved_and_turned_45deg"])
def test_the_window_follows_the_cabinet(cabinet):
    assert _in_window(_env(cabinet, HAND_START_DEPTH_M + 0.0002))
    assert not _in_window(_env(cabinet, HAND_START_DEPTH_M - 0.0002))
    assert _in_window(_env(cabinet, 0.02))
    assert _in_window(_env(cabinet, 0.01, offset_xy_m=(0.0004, 0.0004)))
    assert not _in_window(_env(cabinet, 0.01, offset_xy_m=(0.0005, 0.0005)))
    assert _in_window(_env(cabinet, 0.01, tilt_deg=1.9))
    assert not _in_window(_env(cabinet, 0.01, tilt_deg=2.1))
    assert _in_window(_env(cabinet, 0.01, spin_deg=-95.0))
