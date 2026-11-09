# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""DB-M2 predicates: a harness plug seated in its mate (held for a span of sim time), dropped, or knocked out."""

from __future__ import annotations

import math

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_error_magnitude

from dexbench_arena.tasks.predicates.hold import sustained


def _constant(values, like: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(values, dtype=like.dtype, device=like.device)


def plug_insertion_state(
    env: ManagerBasedRLEnv,
    plug_name: str,
    seat_xyz: tuple[float, float, float],
    pull_out_axis: tuple[float, float, float],
    seated_quat_xyzw: tuple[float, float, float, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(depth, lateral, tilt)`` per env: metres along the pull-out axis, metres off it, radians off square."""
    pose = env.arena_world.get_pose_e(plug_name)
    offset = pose[:, :3] - _constant(seat_xyz, pose)
    axis = _constant(pull_out_axis, pose)
    axis = axis / torch.linalg.vector_norm(axis)
    depth = offset @ axis
    lateral = torch.linalg.vector_norm(offset - depth[:, None] * axis, dim=-1)
    tilt = quat_error_magnitude(pose[:, 3:], _constant(seated_quat_xyzw, pose).expand(pose.shape[0], -1))
    return depth, lateral, tilt


def plug_in_seat_window(
    env: ManagerBasedRLEnv,
    plug_name: str,
    seat_xyz: tuple[float, float, float],
    pull_out_axis: tuple[float, float, float],
    seated_quat_xyzw: tuple[float, float, float, float],
    max_depth_m: float,
    max_lateral_m: float,
    max_tilt_deg: float,
) -> torch.Tensor:
    """True where the plug is in its seat window now (no hold): along the axis within ``max_depth_m`` of the seat
    point, off the axis at most ``max_lateral_m``, and square to it within ``max_tilt_deg``."""
    depth, lateral, tilt = plug_insertion_state(env, plug_name, seat_xyz, pull_out_axis, seated_quat_xyzw)
    return (depth.abs() <= max_depth_m) & (lateral <= max_lateral_m) & (tilt < math.radians(max_tilt_deg))


def plug_seated(
    env: ManagerBasedRLEnv,
    plug_name: str,
    seat_xyz: tuple[float, float, float],
    pull_out_axis: tuple[float, float, float],
    seated_quat_xyzw: tuple[float, float, float, float],
    max_depth_m: float,
    max_lateral_m: float,
    max_tilt_deg: float,
    hold_time_s: float,
) -> torch.Tensor:
    """True once :func:`plug_in_seat_window` has held continuously for ``hold_time_s`` of sim time."""
    ok = plug_in_seat_window(env, plug_name, seat_xyz, pull_out_axis, seated_quat_xyzw, max_depth_m, max_lateral_m,
                             max_tilt_deg)
    return sustained(env, ok, hold_time_s, key=f"plug_seated:{plug_name}")


def hand_plug_force(env: ManagerBasedRLEnv, sensor_names: tuple[str, ...]) -> torch.Tensor:
    """``(num_envs, len(sensor_names))``: each listed hand sensor's total contact force on the plug [N]."""
    columns = []
    for name in sensor_names:
        sensor = env.scene.sensors.get(name) if hasattr(env.scene, "sensors") else None
        if sensor is None:
            columns.append(torch.zeros(env.num_envs, device=env.device))
            continue
        forces = sensor.data.force_matrix_w
        forces = forces if isinstance(forces, torch.Tensor) else forces.torch
        columns.append(torch.linalg.vector_norm(forces, dim=-1).flatten(1).sum(dim=-1))
    return torch.stack(columns, dim=-1)


def plug_along_axis(
    env: ManagerBasedRLEnv,
    plug_name: str,
    seat_xyz: tuple[float, float, float],
    pull_out_axis: tuple[float, float, float],
    seated_quat_xyzw: tuple[float, float, float, float],
    max_depth_m: float,
    max_lateral_error_m: float,
    max_tilt_deg: float,
) -> torch.Tensor:
    """True where the plug is at most ``max_depth_m`` out of its seat, on the axis and square to it now."""
    depth, lateral, tilt = plug_insertion_state(env, plug_name, seat_xyz, pull_out_axis, seated_quat_xyzw)
    return (depth <= max_depth_m) & (lateral <= max_lateral_error_m) & (tilt < math.radians(max_tilt_deg))


def plug_above(env: ManagerBasedRLEnv, plug_name: str, min_z_m: float) -> torch.Tensor:
    """True where the plug origin is above ``min_z_m`` (env frame): lifted off where it rested."""
    return env.arena_world.get_pose_e(plug_name)[:, 2] > min_z_m


def plug_below(env: ManagerBasedRLEnv, plug_name: str, min_z_m: float) -> torch.Tensor:
    """True where the plug origin is below ``min_z_m`` (env frame): it has left the liner."""
    return env.arena_world.get_pose_e(plug_name)[:, 2] < min_z_m


def plug_displaced(
    env: ManagerBasedRLEnv, plug_name: str, seat_xyz: tuple[float, float, float], max_displacement_m: float
) -> torch.Tensor:
    """True where the plug origin is more than ``max_displacement_m`` from its seat point: knocked out of its mate."""
    pose = env.arena_world.get_pose_e(plug_name)
    return torch.linalg.vector_norm(pose[:, :3] - _constant(seat_xyz, pose), dim=-1) > max_displacement_m


def plug_insertion_observation(
    env: ManagerBasedRLEnv,
    plug_name: str,
    seat_xyz: tuple[float, float, float],
    pull_out_axis: tuple[float, float, float],
    seated_quat_xyzw: tuple[float, float, float, float],
) -> torch.Tensor:
    """Observation term: ``(depth m, lateral m, tilt rad)`` of the task plug, ``(num_envs, 3)``."""
    return torch.stack(plug_insertion_state(env, plug_name, seat_xyz, pull_out_axis, seated_quat_xyzw), dim=-1)
