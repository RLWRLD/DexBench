# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Geometry predicates: where an object's centre is, and whether it is upright."""

from __future__ import annotations

import torch
import warp as wp
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply
from isaaclab_arena.tasks.predicates.spatial import object_bounds_center_over_destination
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

from dexbench_arena.tasks.predicates.object_state import get_env, get_rigid_object, get_root_pos_w


def per_env_rows(value, num_envs: int, row_shape_ndim: int, like: torch.Tensor) -> torch.Tensor:
    """Broadcast one shared parameter to every env, or check an explicit per-env stack."""
    t = torch.as_tensor(value, dtype=like.dtype, device=like.device)
    if t.ndim == row_shape_ndim:
        return t.unsqueeze(0).expand(num_envs, *t.shape)
    assert t.shape[0] == num_envs, f"per-env parameter has {t.shape[0]} rows for {num_envs} envs"
    return t


_per_env_rows = per_env_rows


def root_quat_w(env, name: str) -> torch.Tensor:
    """Root orientation (x, y, z, w) of a rigid object (``object_state`` has no quaternion getter)."""
    return wp.to_torch(get_rigid_object(env, name).data.root_quat_w)


_root_quat_w = root_quat_w


def object_center_env(env, object_name: str, object_center_offset_xyz) -> torch.Tensor:
    """The object's geometric centre in the env-local frame, shape ``(num_envs, 3)``."""
    root_pos_w = get_root_pos_w(env, object_name)
    offset = per_env_rows(object_center_offset_xyz, root_pos_w.shape[0], 1, root_pos_w)
    center_w = root_pos_w + quat_apply(root_quat_w(env, object_name), offset)
    return center_w - get_env(env).scene.env_origins


_object_center_env = object_center_env


def object_center_outside_xy_bounds(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    object_center_offset_xyz,
    xy_bounds,
) -> torch.Tensor:
    """True when the centre left the rectangle ``((min_x, min_y), (max_x, max_y))`` (workspace failure)."""
    center_xy = object_center_env(env, object_cfg.name, object_center_offset_xyz)[:, :2]
    bounds = per_env_rows(xy_bounds, center_xy.shape[0], 2, center_xy)
    return ((center_xy < bounds[:, 0]) | (center_xy > bounds[:, 1])).any(dim=-1)


def object_center_within_xy_bounds(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    object_center_offset_xyz,
    xy_bounds,
) -> torch.Tensor:
    """True when the centre is inside the rectangle (progress: "carried over the target")."""
    return ~object_center_outside_xy_bounds(env, object_cfg, object_center_offset_xyz, xy_bounds)


def object_center_over_region(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    object_center_offset_xyz,
    xy_bounds,
    min_z_m: float,
) -> torch.Tensor:
    """True when the centre is over the rectangle ``((min_x, min_y), (max_x, max_y))`` and not below ``min_z_m``."""
    center = object_center_env(env, object_cfg.name, object_center_offset_xyz)
    bounds = per_env_rows(xy_bounds, center.shape[0], 2, center)
    floor = torch.full_like(bounds[:, 0, :1], float(min_z_m))
    destination = AxisAlignedBoundingBox(
        min_point=torch.cat((bounds[:, 0], floor), dim=-1),
        max_point=torch.cat((bounds[:, 1], torch.full_like(floor, float("inf"))), dim=-1),
    )
    identity = torch.zeros(center.shape[0], 7, dtype=center.dtype, device=center.device)
    identity[:, 6] = 1.0
    return object_bounds_center_over_destination(center, identity, destination)


def object_center_in_box(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    object_center_offset_xyz,
    box_center_xy,
    box_center_halfsize_xy,
    box_floor_z_m: float,
    center_height_range_m,
) -> torch.Tensor:
    """True when the centre is inside an axis-aligned box: within the XY half-sizes and the height window above its floor."""
    center = object_center_env(env, object_cfg.name, object_center_offset_xyz)
    num_envs = center.shape[0]
    center_xy = per_env_rows(box_center_xy, num_envs, 1, center)
    half = per_env_rows(box_center_halfsize_xy, num_envs, 1, center)
    in_rect = ((center[:, :2] - center_xy).abs() <= half).all(dim=-1)
    height_range = per_env_rows(center_height_range_m, num_envs, 1, center)
    height = center[:, 2] - box_floor_z_m
    return in_rect & (height > height_range[:, 0]) & (height < height_range[:, 1])


def object_center_distance_to_box(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    object_center_offset_xyz,
    box_center_xy,
    box_center_halfsize_xy,
    box_floor_z_m: float,
    center_height_range_m,
) -> torch.Tensor:
    """Euclidean distance (m) from the centre to the region :func:`object_center_in_box` accepts; 0 inside."""
    center = object_center_env(env, object_cfg.name, object_center_offset_xyz)
    num_envs = center.shape[0]
    center_xy = per_env_rows(box_center_xy, num_envs, 1, center)
    half = per_env_rows(box_center_halfsize_xy, num_envs, 1, center)
    height_range = per_env_rows(center_height_range_m, num_envs, 1, center)
    excess_xy = ((center[:, :2] - center_xy).abs() - half).clamp(min=0.0)
    height = center[:, 2] - box_floor_z_m
    excess_z = (height_range[:, 0] - height).clamp(min=0.0) + (height - height_range[:, 1]).clamp(min=0.0)
    return torch.sqrt((excess_xy ** 2).sum(dim=-1) + excess_z ** 2)


def object_center_past_plane(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    object_center_offset_xyz,
    plane_y_m: float,
    clearance_m: float = 0.0,
    toward_negative_y: bool = True,
    plane_axis: int = 1,
) -> torch.Tensor:
    """True when the centre crossed the axis-aligned plane ``plane_axis = plane_y_m`` by ``clearance_m``."""
    center = object_center_env(env, object_cfg.name, object_center_offset_xyz)[:, plane_axis]
    if toward_negative_y:
        return center < (plane_y_m - clearance_m)
    return center > (plane_y_m + clearance_m)


object_center_past_y_plane = object_center_past_plane
"""Alias of :func:`object_center_past_plane` (the plane axis is a parameter)."""


def any_object_root_past_plane(
    env: ManagerBasedRLEnv,
    object_names: list[str],
    plane_y_m: float,
    toward_negative_y: bool = True,
    plane_axis: int = 1,
) -> torch.Tensor:
    """True when ANY named object's root crossed the plane (a neighbour was carried out)."""
    base = get_env(env)
    result = torch.zeros(base.num_envs, dtype=torch.bool, device=base.device)
    for name in object_names:
        root = (get_root_pos_w(env, name) - base.scene.env_origins)[:, plane_axis]
        result = result | (root < plane_y_m if toward_negative_y else root > plane_y_m)
    return result


any_object_root_past_y_plane = any_object_root_past_plane
"""Alias of :func:`any_object_root_past_plane`."""


def object_toppled(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    max_up_alignment: float = 0.70,
    object_local_up_xyz=(0.0, 0.0, 1.0),
) -> torch.Tensor:
    """True when the object's local up axis (``object_local_up_xyz``, +Z by default) leans past the bound
    (0.70 is about 45 degrees from vertical)."""
    quat = root_quat_w(env, object_cfg.name)
    local_up = per_env_rows(object_local_up_xyz, quat.shape[0], 1, quat)
    return quat_apply(quat, local_up)[:, 2] < max_up_alignment


def any_object_toppled(
    env: ManagerBasedRLEnv,
    object_names: list[str],
    max_up_alignment: float = 0.70,
    object_local_up_xyz=None,
) -> torch.Tensor:
    """True when ANY named object has toppled (see ``object_toppled``); ``object_local_up_xyz`` gives each
    object's local up axis in the order of ``object_names`` (None: +Z for all)."""
    base = get_env(env)
    result = torch.zeros(base.num_envs, dtype=torch.bool, device=base.device)
    ups = object_local_up_xyz if object_local_up_xyz is not None else [(0.0, 0.0, 1.0)] * len(object_names)
    if len(ups) != len(object_names):
        raise ValueError(f"{len(ups)} local up axes for {len(object_names)} objects")
    for name, up in zip(object_names, ups):
        result = result | object_toppled(env, SceneEntityCfg(name), max_up_alignment=max_up_alignment,
                                         object_local_up_xyz=up)
    return result
