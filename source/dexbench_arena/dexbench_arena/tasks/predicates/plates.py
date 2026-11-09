# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""H1 stack seating, release clearance and pose-window settling."""

from __future__ import annotations

import torch
import warp as wp
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import matrix_from_quat


def plates_seated_in_stack(
    env: ManagerBasedRLEnv,
    plate_names: list[str],
    stack_center_xy: tuple[float, float] | torch.Tensor,
    surface_top_z_m: float,
    region_radius_m: float,
    expected_count: int,
    max_rim_offset_m: float,
    min_up_alignment: float,
    nest_pitch_m: float,
    velocity_threshold_m_s: float = 0.10,
    surface_z_tolerance_m: float = 0.02,
    max_seat_height_error_m: float = 0.010,
    robot_asset_name: str | None = None,
    clear_radius_m: float = 0.0,
    clear_height_m: float = 0.0,
    settle_window_s: float = 0.0,
    rim_offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    stack_center_offset=None,
) -> torch.Tensor:
    """True when every plate has been seated into one stack over ``stack_center_xy``."""
    origins = env.unwrapped.scene.env_origins

    positions = torch.stack([wp.to_torch(env.unwrapped.scene[name].data.root_pos_w) for name in plate_names], dim=1)
    quats = torch.stack(
        [wp.to_torch(env.unwrapped.scene[name].data.root_quat_w) for name in plate_names], dim=1
    )
    speeds = torch.stack(
        [torch.norm(wp.to_torch(env.unwrapped.scene[name].data.root_lin_vel_w), dim=-1) for name in plate_names],
        dim=1,
    )
    if settle_window_s > 0:
        from dexbench_arena.tasks._pose_motion_window import PoseMotionWindow

        owner = env.unwrapped
        key = (tuple(plate_names), float(settle_window_s), float(owner.step_dt))
        if not hasattr(owner, "_plate_pose_motion_windows"):
            owner._plate_pose_motion_windows = {}
        if key not in owner._plate_pose_motion_windows:
            owner._plate_pose_motion_windows[key] = PoseMotionWindow(settle_window_s, owner.step_dt)
        speeds = owner._plate_pose_motion_windows[key].update(positions, owner.episode_length_buf)
    positions_env = positions - origins.unsqueeze(1)

    center = torch.as_tensor(stack_center_xy, dtype=positions_env.dtype, device=positions_env.device)
    if center.ndim == 2:
        if center.shape != (positions_env.shape[0], 2):
            raise ValueError("Destination centers must have shape (num_envs, 2)")
    if stack_center_offset is not None:
        center = center + stack_center_offset.xy(positions_env.shape[0], center.device, center.dtype)
    if center.ndim == 2:
        center = center.unsqueeze(1)
    in_region = (torch.norm(positions_env[..., :2] - center, dim=-1) <= region_radius_m) & (
        positions_env[..., 2] >= surface_top_z_m - surface_z_tolerance_m
    )
    transferred = in_region.sum(dim=1) >= expected_count

    up = matrix_from_quat(quats)[..., :, 2]
    up_z = up[..., 2]
    upright = ((up_z >= min_up_alignment) | ~in_region).all(dim=1)
    at_rest = ((speeds < velocity_threshold_m_s) | ~in_region).all(dim=1)

    sort_key = torch.where(in_region, positions_env[..., 2], torch.full_like(positions_env[..., 2], 1e6))
    order = torch.argsort(sort_key, dim=1)
    rim_offset = stack_lateral_offset(positions_env, quats, order, in_region, rim_offset_xyz)
    member = torch.gather(in_region, 1, order)
    seated = rim_offset < max_rim_offset_m

    z_sorted = torch.gather(positions_env[..., 2], 1, order)
    slot = surface_top_z_m + nest_pitch_m * torch.arange(
        z_sorted.shape[1], dtype=z_sorted.dtype, device=z_sorted.device
    )
    height_error = (z_sorted - slot).abs()
    at_slot_height = ((height_error <= max_seat_height_error_m) | ~member).all(dim=1)

    released = torch.ones_like(transferred)
    if robot_asset_name is not None and clear_radius_m > 0.0:
        robot = env.unwrapped.scene[robot_asset_name]
        body_pos_env = wp.to_torch(robot.data.body_pos_w) - origins.unsqueeze(1)
        top_z = (
            torch.where(in_region, positions_env[..., 2], torch.full_like(positions_env[..., 2], -1e6))
            .max(dim=1)
            .values
        )
        radial = torch.norm(body_pos_env[:, :, None, :2] - positions_env[:, None, :, :2], dim=-1)
        near_plate = ((radial < clear_radius_m) & in_region[:, None, :]).any(dim=-1)
        inside = near_plate & (body_pos_env[..., 2] < (top_z + clear_height_m).unsqueeze(1))
        released = ~inside.any(dim=1)

    return transferred & upright & at_rest & seated & at_slot_height & released


def stack_lateral_offset(positions, quaternions, order, in_region, offset_xyz):
    """Cumulative local-feature separation, shared by live success and offline audit."""
    offset = torch.as_tensor(offset_xyz, dtype=positions.dtype, device=positions.device)
    centers = positions + (matrix_from_quat(quaternions) @ offset[..., None]).squeeze(-1)
    xy = torch.gather(centers[..., :2], 1, order.unsqueeze(-1).expand(-1, -1, 2))
    members = torch.gather(in_region, 1, order)
    return (torch.linalg.vector_norm(xy[:, 1:] - xy[:, :-1], dim=-1) * (members[:, 1:] & members[:, :-1])).sum(dim=1)
