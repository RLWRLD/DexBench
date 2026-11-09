# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""DB-M1's seat window is measured in the washer's frame, so it follows the washer."""

import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("isaaclab_arena", reason="needs isaaclab_arena on the path")

from isaaclab.utils.math import (  # noqa: E402
    euler_xyz_from_quat,
    quat_apply,
    quat_from_angle_axis,
    quat_from_euler_xyz,
    quat_mul,
)

from dexbench_arena.layouts.db_m1_layout import (  # noqa: E402
    SEAT_DEPTH_WINDOW_M,
    SEAT_IN_WASHER_XYZ,
    SEAT_MAX_LATERAL_ERROR_M,
    SEAT_MAX_TILT_DEG,
)
from dexbench_arena.tasks.predicates.drawer_insertion import drawer_in_seat_window  # noqa: E402


def _in_window(washer_pose, drawer_offset_in_washer, drawer_rotation_in_washer=(0.0, 0.0, 0.0)):
    washer = torch.tensor([washer_pose], dtype=torch.float64)
    roll, pitch, yaw = (torch.tensor([v], dtype=torch.float64) for v in drawer_rotation_in_washer)
    drawer_q = quat_mul(washer[:, 3:], quat_from_euler_xyz(roll, pitch, yaw))
    seat = torch.tensor([SEAT_IN_WASHER_XYZ], dtype=torch.float64) + torch.tensor([drawer_offset_in_washer], dtype=torch.float64)
    drawer = torch.cat((washer[:, :3] + quat_apply(washer[:, 3:], seat), drawer_q), -1)
    poses = {"drawer": drawer, "washer": washer}
    env = SimpleNamespace(num_envs=1, arena_world=SimpleNamespace(get_pose_w=poses.__getitem__))
    return bool(
        drawer_in_seat_window(
            env, "drawer", "washer", SEAT_IN_WASHER_XYZ, SEAT_DEPTH_WINDOW_M, SEAT_MAX_LATERAL_ERROR_M, SEAT_MAX_TILT_DEG
        )[0]
    )


@pytest.mark.parametrize(
    "washer_pose",
    [(0.0766, 0.0643, 0.25, 0.0, 0.0, 0.0, 1.0), (1.0, -2.0, 0.25, 0.0, 0.0, 0.3826834, 0.9238795)],
    ids=["layout", "moved_and_turned_45deg"],
)
def test_seat_window_follows_the_washer(washer_pose):
    assert _in_window(washer_pose, (0.0, -0.009, 0.0))
    assert not _in_window(washer_pose, (0.0, -0.013, 0.0))
    assert not _in_window(washer_pose, (0.0, 0.003, 0.0))
    assert _in_window(washer_pose, (0.0019, -0.009, -0.0019))
    assert not _in_window(washer_pose, (0.0021, -0.009, 0.0))
    assert not _in_window(washer_pose, (0.0, -0.009, 0.0021))
    assert _in_window(washer_pose, (0.0, -0.009, 0.0), (0.0, 0.0, 0.025))
    assert not _in_window(washer_pose, (0.0, -0.009, 0.0), (0.0, 0.0, 0.0275))


def test_the_tilt_bound_is_the_rotation_angle_not_each_axis():
    """1.8 deg about the cube diagonal tilts no single axis past 1.5 deg, and still fails the window."""
    washer = (0.0766, 0.0643, 0.25, 0.0, 0.0, 0.0, 1.0)
    axis = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float64)
    q = quat_from_angle_axis(torch.tensor([math.radians(1.8)], dtype=torch.float64), axis)
    roll, pitch, yaw = euler_xyz_from_quat(q)
    assert not _in_window(washer, (0.0, -0.009, 0.0), (float(roll[0]), float(pitch[0]), float(yaw[0])))


def test_tabletop_finish_restarts_the_hold_when_drawer_moves(monkeypatch):
    from dexbench_arena.tasks.predicates import drawer_insertion as predicates

    monkeypatch.setattr(predicates, 'drawer_in_seat_window', lambda *args: torch.tensor([True]))
    velocity = torch.zeros(1, 3)
    env = SimpleNamespace(step_dt=1 / 30, common_step_counter=0,
                          episode_length_buf=torch.zeros(1, dtype=torch.long),
                          arena_world=SimpleNamespace(get_root_linear_velocity_w=lambda name: velocity))
    for step in range(1, 31):
        env.common_step_counter = step
        env.episode_length_buf[:] = step
        velocity[0, 0] = .02 if step == 15 else 0.
        ok = predicates.drawer_seated(env, 'drawer', 'washer', SEAT_IN_WASHER_XYZ,
                                      (-.001, .002), .002, 1., .5, max_linear_speed_m_s=.01)
        assert bool(ok[0]) == (step == 30)
