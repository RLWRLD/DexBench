# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""DB-M4 predicates: a threaded stud started in its hole, held for a span of sim time, and how far it is in."""

from __future__ import annotations

import math

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply, quat_apply_inverse
from isaaclab_arena.tasks.predicates.spatial import depth_in_range, lateral_in_proximity, tilt_axis_aligned

from dexbench_arena.tasks.predicates.hold import sustained


def stud_in_hand_start_window(
    env: ManagerBasedRLEnv,
    object_name: str,
    receiver_name: str,
    hole_in_receiver_xyz: tuple[float, float, float],
    hole_axis_in_receiver: tuple[float, float, float],
    stud_axis_in_object: tuple[float, float, float],
    min_depth_m: float,
    max_offset_m: float,
    max_tilt_deg: float,
) -> torch.Tensor:
    """True where the stud is in the hand-start window now (no hold), measured in the receiver's frame."""
    hole = tuple(hole_in_receiver_xyz)
    axis = tuple(hole_axis_in_receiver)
    ok = depth_in_range(env, object_name, receiver_name, hole, min_depth_m, None, receiver_axis=axis)
    ok &= lateral_in_proximity(env, object_name, receiver_name, hole, max_offset_m, receiver_axis=axis)
    ok &= tilt_axis_aligned(
        env, object_name, receiver_name, math.radians(max_tilt_deg), subject_axis=tuple(stud_axis_in_object), receiver_axis=axis
    )
    return ok


def stud_hand_started(
    env: ManagerBasedRLEnv,
    object_name: str,
    receiver_name: str,
    hole_in_receiver_xyz: tuple[float, float, float],
    hole_axis_in_receiver: tuple[float, float, float],
    stud_axis_in_object: tuple[float, float, float],
    min_depth_m: float,
    max_offset_m: float,
    max_tilt_deg: float,
    hold_time_s: float,
) -> torch.Tensor:
    """True once :func:`stud_in_hand_start_window` has held continuously for ``hold_time_s`` of sim time."""
    ok = stud_in_hand_start_window(
        env, object_name, receiver_name, hole_in_receiver_xyz, hole_axis_in_receiver, stud_axis_in_object,
        min_depth_m, max_offset_m, max_tilt_deg,
    )
    return sustained(env, ok, hold_time_s, key=f"stud_hand_started:{object_name}")


def stud_engagement(
    env: ManagerBasedRLEnv,
    object_name: str,
    receiver_name: str,
    hole_in_receiver_xyz: tuple[float, float, float],
    hole_axis_in_receiver: tuple[float, float, float],
    stud_axis_in_object: tuple[float, float, float],
) -> torch.Tensor:
    """Observation term: ``(depth [m], offset [m], tilt [rad])`` of the stud in the hole, ``(num_envs, 3)``."""
    world = env.arena_world
    T_W_S = world.get_pose_w(object_name)
    T_W_R = world.get_pose_w(receiver_name)
    hole_R = torch.as_tensor(hole_in_receiver_xyz, dtype=T_W_R.dtype, device=T_W_R.device).expand(env.num_envs, -1)
    axis_R = torch.as_tensor(hole_axis_in_receiver, dtype=T_W_R.dtype, device=T_W_R.device).expand(env.num_envs, -1)
    axis_R = axis_R / torch.linalg.vector_norm(axis_R, dim=-1, keepdim=True)
    tip_R = quat_apply_inverse(T_W_R[:, 3:], T_W_S[:, :3] - T_W_R[:, :3]) - hole_R
    depth = torch.sum(tip_R * axis_R, dim=-1)
    offset = torch.linalg.vector_norm(tip_R - depth[:, None] * axis_R, dim=-1)
    stud_S = torch.as_tensor(stud_axis_in_object, dtype=T_W_S.dtype, device=T_W_S.device).expand(env.num_envs, -1)
    stud_W = quat_apply(T_W_S[:, 3:], stud_S / torch.linalg.vector_norm(stud_S, dim=-1, keepdim=True))
    hole_W = quat_apply(T_W_R[:, 3:], axis_R)
    tilt = torch.acos(torch.clamp(torch.sum(stud_W * hole_W, dim=-1), -1.0, 1.0))
    return torch.stack((depth, offset, tilt), dim=-1)
