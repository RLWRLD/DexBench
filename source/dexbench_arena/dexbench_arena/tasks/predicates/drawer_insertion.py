# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""DB-M1 predicates: a drawer seated in its slot (held for a span of sim time) and the washer's push on it."""

from __future__ import annotations

import math

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import quat_error_magnitude
from isaaclab_arena.tasks.predicates.spatial import depth_in_range, object_is_moving_slowly

from dexbench_arena.tasks.predicates.hold import sustained


def drawer_in_slot_window(
    env: ManagerBasedRLEnv,
    object_name: str,
    receiver_name: str,
    seat_in_receiver_xyz: tuple[float, float, float],
    min_depth_m: float,
    max_depth_m: float | None = None,
    max_lateral_error_m: float | None = None,
    max_tilt_deg: float | None = None,
) -> torch.Tensor:
    """True where the drawer origin is inside a window along the slot now (no hold), in the washer's frame."""
    seat = tuple(seat_in_receiver_xyz)
    ok = depth_in_range(env, object_name, receiver_name, seat, min_depth_m, max_depth_m, receiver_axis=(0.0, 1.0, 0.0))
    if max_lateral_error_m is not None:
        for axis in ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)):
            ok &= depth_in_range(
                env, object_name, receiver_name, seat, -max_lateral_error_m, max_lateral_error_m, receiver_axis=axis
            )
    if max_tilt_deg is not None:
        world = env.arena_world
        tilt = quat_error_magnitude(world.get_pose_w(object_name)[:, 3:], world.get_pose_w(receiver_name)[:, 3:])
        ok &= tilt < math.radians(max_tilt_deg)
    return ok


def drawer_in_seat_window(
    env: ManagerBasedRLEnv,
    object_name: str,
    receiver_name: str,
    seat_in_receiver_xyz: tuple[float, float, float],
    depth_window_m: tuple[float, float],
    max_lateral_error_m: float,
    max_tilt_deg: float,
) -> torch.Tensor:
    """True where the drawer origin is inside the seat window now (no hold): :func:`drawer_in_slot_window` with all bounds."""
    return drawer_in_slot_window(
        env, object_name, receiver_name, seat_in_receiver_xyz, depth_window_m[0], depth_window_m[1],
        max_lateral_error_m, max_tilt_deg,
    )


def drawer_seated(
    env: ManagerBasedRLEnv,
    object_name: str,
    receiver_name: str,
    seat_in_receiver_xyz: tuple[float, float, float],
    depth_window_m: tuple[float, float],
    max_lateral_error_m: float,
    max_tilt_deg: float,
    hold_time_s: float,
    max_linear_speed_m_s: float | None = None,
) -> torch.Tensor:
    """True once :func:`drawer_in_seat_window` has held continuously for ``hold_time_s`` of sim time."""
    ok = drawer_in_seat_window(
        env, object_name, receiver_name, seat_in_receiver_xyz, depth_window_m, max_lateral_error_m, max_tilt_deg
    )
    if max_linear_speed_m_s is not None:
        if not math.isfinite(max_linear_speed_m_s) or max_linear_speed_m_s <= 0:
            raise ValueError("max_linear_speed_m_s must be positive and finite")
        ok &= object_is_moving_slowly(env.arena_world.get_root_linear_velocity_w(object_name), max_linear_speed_m_s)
    return sustained(env, ok, hold_time_s, key=f"drawer_seated:{object_name}")


class SlotContactForce(ManagerTermBase):
    """Observation term: the washer's force on the drawer summed per drawer region, world frame [N]."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        sensor = env.scene.sensors[cfg.params["sensor_name"]]
        names = [str(name).rsplit("/", 1)[-1] for name in sensor.sensor_names]
        self.region_ids = []
        for region, members in cfg.params["regions"].items():
            ids = [index for index, name in enumerate(names) if name in set(members)]
            assert ids, f"slot contact region {region!r}: none of its colliders is a sensing shape of {cfg.params['sensor_name']}"
            self.region_ids.append(ids)

    def __call__(self, env: ManagerBasedRLEnv, sensor_name: str, regions: dict) -> torch.Tensor:  # noqa: ARG002
        forces = env.scene.sensors[sensor_name].data.force_matrix_w.torch
        per_shape = forces.sum(dim=2)
        return torch.cat([per_shape[:, ids, :].sum(dim=1) for ids in self.region_ids], dim=-1)
