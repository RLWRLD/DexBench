# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Moving-kit-box placement geometry and success for the E3 M5 kitting cell."""

from __future__ import annotations

import itertools

import torch

from dexbench_arena.tasks.conveyor_placement import rotate_xyzw
from dexbench_arena.tasks.predicates.hold import sustained

KIT_BOX_PLACE_MAX_FLOOR_GAP_M = 0.005
KIT_BOX_PLACE_MIN_CENTER_Z_LOCAL = 0.0
KIT_BOX_PLACE_MAX_OVERHANG_FRACTION = 0.15


def rotate_xyzw_inverse(quat: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    """Apply the inverse of unit xyzw quaternions (rotation by the conjugate)."""
    conjugate = torch.cat((-quat[..., :3], quat[..., 3:]), dim=-1)
    return rotate_xyzw(conjugate, vectors)


def parts_in_compartment(
    box_pos: torch.Tensor,
    box_quat: torch.Tensor,
    part_pos: torch.Tensor,
    part_quat: torch.Tensor,
    *,
    compartment_min_xy_local,
    compartment_max_xy_local,
    compartment_floor_z_local: float,
    part_center_local,
    part_extents_local,
    part_end_points_local,
    max_floor_gap_m: float = KIT_BOX_PLACE_MAX_FLOOR_GAP_M,
    max_overhang_fraction: float = KIT_BOX_PLACE_MAX_OVERHANG_FRACTION,
    min_center_z_local: float = KIT_BOX_PLACE_MIN_CENTER_Z_LOCAL,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Which parts are in the compartment, and where their ends sit in the box."""

    like = box_pos

    def as_tensor(values) -> torch.Tensor:
        return torch.as_tensor(values, dtype=like.dtype, device=like.device)

    center_local = as_tensor(part_center_local)
    extents_local = as_tensor(part_extents_local)
    ends_local = as_tensor(part_end_points_local)
    lower = as_tensor(compartment_min_xy_local)
    upper = as_tensor(compartment_max_xy_local)
    if lower.dim() == 2:
        lower, upper = lower[:, None, :], upper[:, None, :]
    signs = as_tensor(list(itertools.product((-1.0, 1.0), repeat=3)))

    def to_box(part_points_local: torch.Tensor) -> torch.Tensor:
        world = part_pos + rotate_xyzw(part_quat, part_points_local.expand_as(part_pos))
        return rotate_xyzw_inverse(box_quat[:, None, :], world - box_pos[:, None, :])

    center3 = to_box(center_local)
    center = center3[..., :2]
    corners = torch.stack([to_box(center_local + sign * extents_local / 2) for sign in signs], dim=2)
    over_rectangle = ((center >= lower) & (center <= upper)).all(dim=-1)
    on_floor = (corners[..., 2].amin(dim=-1) <= compartment_floor_z_local + max_floor_gap_m) & (
        center3[..., 2] > min_center_z_local
    )
    overhang = torch.maximum(lower[..., None, :] - corners[..., :2], corners[..., :2] - upper[..., None, :]).amax(
        dim=(-2, -1)
    )
    contained = overhang < max_overhang_fraction * torch.linalg.vector_norm(extents_local)
    ends = torch.stack([to_box(point) for point in ends_local], dim=2)
    return over_rectangle & on_floor & contained, ends


def kit_box_seated_on_belt(
    box_pos: torch.Tensor,
    box_quat: torch.Tensor,
    *,
    half_xy,
    belt_y_range,
    belt_top_z: float,
    max_gap: float,
) -> torch.Tensor:
    """Whether every bottom corner of the kit box sits between the side frames and on the belt top."""

    hx, hy = half_xy
    corners = torch.tensor(
        [[hx, hy, 0.0], [hx, -hy, 0.0], [-hx, hy, 0.0], [-hx, -hy, 0.0]], dtype=box_pos.dtype, device=box_pos.device
    )
    num_envs = box_pos.shape[0]
    world = box_pos[:, None, :] + rotate_xyzw(
        box_quat[:, None, :].expand(num_envs, 4, 4), corners.expand(num_envs, 4, 3)
    )
    y, z = world[..., 1], world[..., 2]
    low, high = belt_y_range
    seated = (y > low) & (y < high) & ((z - belt_top_z).abs() <= max_gap)
    return seated.all(dim=-1)


def _root_pose(base, name: str):
    import warp as wp

    data = base.scene[name].data
    return wp.to_torch(data.root_pos_w) - base.scene.env_origins, wp.to_torch(data.root_quat_w)


def kit_box_empty_compartment(env, default: int) -> torch.Tensor:
    """``(N,)`` index of each env's empty compartment: the one the kit box's ``empty_compartment`` variation drew at
    this episode's reset, or ``default`` where it drew none (the variation is off, or the env has not reset yet)."""

    from dexbench_arena.variations.kit_box_compartment_variation import empty_compartment_event

    base = env.unwrapped
    fallback = torch.full((base.num_envs,), int(default), dtype=torch.long, device=base.device)
    event = empty_compartment_event(base)
    if event is None:
        return fallback
    drawn = event[0].empty_compartment.to(fallback.device)
    return torch.where(drawn >= 0, drawn, fallback)


GRAVITY_M_S2 = 9.81


def hand_release_limit_n(mass_kg: torch.Tensor, max_force_n: float, max_weight_fraction: float) -> torch.Tensor:
    """The most force [N] the hands may still put on a part of ``mass_kg`` that they have let go of:
    ``max_weight_fraction`` of its weight, and never more than ``max_force_n``. DB-R1's rule
    (``r1_rules.evaluate``): a hand that carries a light part pushes on it with less than 1 N."""

    return torch.clamp(max_weight_fraction * mass_kg * GRAVITY_M_S2, max=max_force_n)


def part_placed_in_kit_box_compartment(
    env,
    kit_box_name: str,
    part_names,
    compartments_min_xy_local,
    compartments_max_xy_local,
    default_empty_compartment: int,
    compartment_floor_z_local: float,
    part_center_local,
    part_extents_local,
    part_end_points_local,
    *,
    kit_box_half_xy,
    belt_y_range,
    belt_top_z: float,
    max_belt_gap_m: float,
    hand_sensor_names=None,
    max_hand_force_n: float | None = None,
    max_hand_weight_fraction: float | None = None,
    hold_time_s: float = 0.0,
    hold_key: str = "db_m5_placed",
) -> torch.Tensor:
    """Any one of ``part_names`` in the empty compartment and let go of, with the box seated on the belt,
    held for ``hold_time_s``."""

    base = env.unwrapped
    box_pos, box_quat = _root_pose(base, kit_box_name)
    poses = [_root_pose(base, name) for name in part_names]
    rectangles = torch.as_tensor(
        (compartments_min_xy_local, compartments_max_xy_local), dtype=box_pos.dtype, device=box_pos.device
    )
    lower, upper = rectangles[:, kit_box_empty_compartment(base, default_empty_compartment).to(box_pos.device)]
    part_pos = torch.stack([pos for pos, _ in poses], dim=1)
    inside, _ = parts_in_compartment(
        box_pos,
        box_quat,
        part_pos,
        torch.stack([quat for _, quat in poses], dim=1),
        compartment_min_xy_local=lower,
        compartment_max_xy_local=upper,
        compartment_floor_z_local=compartment_floor_z_local,
        part_center_local=part_center_local,
        part_extents_local=part_extents_local,
        part_end_points_local=part_end_points_local,
    )
    seated = kit_box_seated_on_belt(
        box_pos,
        box_quat,
        half_xy=kit_box_half_xy,
        belt_y_range=belt_y_range,
        belt_top_z=belt_top_z,
        max_gap=max_belt_gap_m,
    )
    if hand_sensor_names is not None:
        from dexbench_arena.tasks.predicates.release import HAND_RELEASED_MAX_FORCE_N, hand_force_on_object

        if len(hand_sensor_names) != len(part_names):
            raise ValueError(f"{len(hand_sensor_names)} hand sensors for {len(part_names)} parts; give one per part")
        limit = HAND_RELEASED_MAX_FORCE_N if max_hand_force_n is None else max_hand_force_n
        force = torch.stack([hand_force_on_object(base, name) for name in hand_sensor_names], dim=1)
        if max_hand_weight_fraction is not None:
            mass = torch.stack([base.scene[name].data.body_mass.torch.reshape(-1) for name in part_names], dim=1)
            limit = hand_release_limit_n(mass.to(force), limit, max_hand_weight_fraction)
        inside = inside & (force.to(inside.device) <= torch.as_tensor(limit).to(inside.device))
    return sustained(base, inside.any(dim=1) & seated, hold_time_s, hold_key)


def kit_box_reached_belt_end(env, kit_box_name: str, leading_edge_local_x: float, belt_end_x: float) -> torch.Tensor:
    """The kit box's leading (+X) edge has reached the belt's downstream end."""

    base = env.unwrapped
    pos, quat = _root_pose(base, kit_box_name)
    edge = torch.zeros_like(pos)
    edge[:, 0] = leading_edge_local_x
    return (pos + rotate_xyzw(quat, edge))[:, 0] >= belt_end_x
