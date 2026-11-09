# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Placement predicates: the object is seated where the task wants it, upright and at rest."""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply
from dexbench_arena.tasks.predicates.geometry import object_center_env, per_env_rows, root_quat_w
from dexbench_arena.tasks.predicates.rest import SETTLED_ANG_VEL_THRESHOLD_RAD_S, object_settled


def _up_alignment(env, name: str, object_local_up_xyz) -> torch.Tensor:
    quat = root_quat_w(env, name)
    local_up = per_env_rows(object_local_up_xyz, quat.shape[0], 1, quat)
    return quat_apply(quat, local_up)[:, 2]


def _upright(env, name: str, object_local_up_xyz, min_up_alignment: float) -> torch.Tensor:
    return _up_alignment(env, name, object_local_up_xyz) > min_up_alignment


def object_up_misalignment(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg, object_local_up_xyz=(0.0, 0.0, 1.0)) -> torch.Tensor:
    """``1 - cos(lean)`` of the object's local up axis from world +Z: 0 upright, 0.015 at 10 degrees (a refine measure)."""
    return 1.0 - _up_alignment(env, object_cfg.name, object_local_up_xyz)


def object_standing_in_slot(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    object_center_offset_xyz,
    slot_center_xy,
    slot_center_halfsize_xy,
    shelf_top_z_m: float,
    center_height_range_m,
    object_local_up_xyz=(0.0, 0.0, 1.0),
    min_up_alignment: float = 0.90,
    velocity_threshold_m_s: float = 0.10,
    ang_vel_threshold_rad_s: float = SETTLED_ANG_VEL_THRESHOLD_RAD_S,
) -> torch.Tensor:
    """True when the object stands upright at rest inside a rectangular shelf slot (DB-R3)."""
    center = object_center_env(env, object_cfg.name, object_center_offset_xyz)
    num_envs = center.shape[0]
    slot_center = per_env_rows(slot_center_xy, num_envs, 1, center)
    halfsize = per_env_rows(slot_center_halfsize_xy, num_envs, 1, center)
    in_rect = ((center[:, :2] - slot_center).abs() <= halfsize).all(dim=-1)
    height_range = per_env_rows(center_height_range_m, num_envs, 1, center)
    height = center[:, 2] - shelf_top_z_m
    resting = (height > height_range[:, 0]) & (height < height_range[:, 1])
    upright = _upright(env, object_cfg.name, object_local_up_xyz, min_up_alignment)
    slow = object_settled(env, object_cfg.name, velocity_threshold_m_s, ang_vel_threshold_rad_s)
    return in_rect & resting & upright & slow


def object_resting_on_disk(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    object_center_offset_xyz: tuple[float, float, float],
    disk_center_xy: tuple[float, float],
    disk_top_z_m: float,
    disk_radius_m: float,
    center_height_range_m: tuple[float, float],
    xy_margin_m: float = 0.0,
    velocity_threshold_m_s: float = 0.10,
    ang_vel_threshold_rad_s: float = SETTLED_ANG_VEL_THRESHOLD_RAD_S,
) -> torch.Tensor:
    """True when the object's centre rests over a disk: inside ``disk_radius_m + xy_margin_m``, in the height window, at rest."""
    center = object_center_env(env, object_cfg.name, object_center_offset_xyz)
    disk_center = torch.tensor(disk_center_xy, dtype=center.dtype, device=center.device)
    over_disk = torch.norm(center[:, :2] - disk_center, dim=-1) < (disk_radius_m + xy_margin_m)
    height = center[:, 2] - disk_top_z_m
    resting = (height > center_height_range_m[0]) & (height < center_height_range_m[1])
    slow = object_settled(env, object_cfg.name, velocity_threshold_m_s, ang_vel_threshold_rad_s)
    return over_disk & resting & slow
