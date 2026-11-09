# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""E1 retail composites: DB-R2 success and wrong pick, DB-R3 placed and released (both tiers), and
the neighbour-knocked gates of DB-R2 and DB-R3."""

from __future__ import annotations

import os

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply

from dexbench_arena.tasks.predicates.object_state import get_root_lin_vel_w, get_root_pos_w

from dexbench_arena.tasks.predicates.geometry import (
    any_object_root_past_plane,
    any_object_toppled,
    object_center_env,
    object_center_in_box,
    per_env_rows,
    root_quat_w,
)
from dexbench_arena.tasks.predicates.hold import sustained, sustained_count
from dexbench_arena.tasks.predicates.object_state import get_env
from dexbench_arena.tasks.predicates.placement import object_standing_in_slot
from dexbench_arena.tasks.predicates.release import (
    HAND_BACK_OUT_M,
    HAND_RELEASED_MAX_FORCE_N,
    body_past_plane,
    hand_force_on_object,
)
from dexbench_arena.tasks.predicates.rest import (
    SETTLED_ANG_VEL_THRESHOLD_RAD_S,
    RestGates,
    any_object_fell_below_rest,
    any_object_lifted_from_rest,
    any_object_shifted_from_rest,
    max_rest_displacement,
    object_settled,
    objects_within_rest_displacement,
    rest_pos_w,
)
from dexbench_arena.tasks.predicates.status import publish


def db_r2_wrong_pick(
    env: ManagerBasedRLEnv,
    neighbor_names: list[str],
    plane_y_m: float,
    min_lift_m: float,
    toward_negative_y: bool = True,
    plane_axis: int = 1,
    rest_gates: RestGates | None = None,
) -> torch.Tensor:
    """True when a watched neighbour left its station: crossed the opening plane or rose ``min_lift_m`` off its rest."""
    return any_object_root_past_plane(env, neighbor_names, plane_y_m, toward_negative_y, plane_axis) | (
        any_object_lifted_from_rest(env, neighbor_names, min_lift_m, rest_gates)
    )
def db_r2_neighbor_knocked(
    env: ManagerBasedRLEnv,
    neighbor_names: list[str],
    max_up_alignment: float,
    min_lift_m: float,
    plane_y_m: float,
    max_lateral_m: float,
    toward_negative_y: bool = True,
    plane_axis: int = 1,
    lateral_axis: int = 0,
    rest_gates: RestGates | None = None,
) -> torch.Tensor:
    """True when a watched neighbour was knocked: toppled, risen ``min_lift_m`` off its rest, out past the
    opening plane (:func:`db_r2_wrong_pick`), or ``max_lateral_m`` sideways along ``lateral_axis`` from its rest."""
    return (
        any_object_toppled(env, neighbor_names, max_up_alignment)
        | db_r2_wrong_pick(env, neighbor_names, plane_y_m, min_lift_m, toward_negative_y, plane_axis, rest_gates)
        | any_object_shifted_from_rest(env, neighbor_names, max_lateral_m, lateral_axis, rest_gates)
    )


def db_r2_target_placed(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    object_center_offset_xyz,
    basket_center_xy,
    basket_center_halfsize_xy,
    basket_floor_z_m: float,
    center_height_range_m,
    neighbor_names: list[str],
    max_neighbor_displacement_m: float | None,
    velocity_threshold_m_s: float = 0.10,
    hold_time_s: float = 0.0,
    hold_key: str = "db_r2_strict",
    ang_vel_threshold_rad_s: float = SETTLED_ANG_VEL_THRESHOLD_RAD_S,
    rest_gates: RestGates | None = None,
    hold_share: float = 1.0,
    neighbor_knock: dict | None = None,
) -> torch.Tensor:
    """DB-R2 success: target centre inside the basket, target at rest, neighbours undisturbed.
    Uprightness is not required (a can on its side is a valid retrieval). "Released" is
    approximated by "at rest". ``hold_time_s`` > 0 is the sustained strict tier, held for
    ``hold_share`` of that window (1.0: every step of it)."""
    contained = object_center_in_box(
        env, object_cfg, object_center_offset_xyz, basket_center_xy, basket_center_halfsize_xy,
        basket_floor_z_m, center_height_range_m,
    )
    settled = object_settled(env, object_cfg.name, velocity_threshold_m_s, ang_vel_threshold_rad_s)
    undisturbed = torch.ones_like(contained)
    if max_neighbor_displacement_m is not None:
        undisturbed = objects_within_rest_displacement(env, neighbor_names, max_neighbor_displacement_m, rest_gates)
    if neighbor_knock is not None:
        undisturbed = undisturbed & ~db_r2_neighbor_knocked(env, neighbor_names, **neighbor_knock, rest_gates=rest_gates)
    ok = contained & settled & undisturbed
    if os.environ.get("DEXBENCH_R2_DIAG"):
        _print_r2_diag(env, object_cfg, object_center_offset_xyz, neighbor_names, contained, settled, undisturbed)
    return sustained(env, ok, hold_time_s, hold_key, hold_share)


def db_r3_neighbor_toppled_or_fallen(
    env: ManagerBasedRLEnv,
    neighbor_names: list[str],
    neighbor_local_up_xyz,
    max_up_alignment: float,
    max_drop_m: float,
    rest_gates: RestGates | None = None,
) -> torch.Tensor:
    """DB-R3's hard neighbour failure: a shelf neighbour toppled (its up axis, ``neighbor_local_up_xyz`` in the
    order of the names, past ``max_up_alignment``) or fallen more than ``max_drop_m`` below its rest."""
    return any_object_toppled(env, neighbor_names, max_up_alignment, neighbor_local_up_xyz) | (
        any_object_fell_below_rest(env, neighbor_names, max_drop_m, rest_gates)
    )


def db_r3_neighbor_knocked(
    env: ManagerBasedRLEnv,
    neighbor_names: list[str],
    neighbor_local_up_xyz,
    max_up_alignment: float,
    max_drop_m: float,
    min_lift_m: float,
    max_along_m: float,
    max_depth_m: float,
    along_axis: int,
    depth_axis: int,
    rest_gates: RestGates | None = None,
) -> torch.Tensor:
    """True when a DB-R3 shelf neighbour was knocked: toppled or fallen (:func:`db_r3_neighbor_toppled_or_fallen`),
    risen ``min_lift_m`` off its rest, or moved from its rest more than ``max_along_m`` along the shelf or
    ``max_depth_m`` in depth (world axes ``along_axis`` / ``depth_axis``)."""
    return (
        db_r3_neighbor_toppled_or_fallen(env, neighbor_names, neighbor_local_up_xyz, max_up_alignment, max_drop_m, rest_gates)
        | any_object_lifted_from_rest(env, neighbor_names, min_lift_m, rest_gates)
        | any_object_shifted_from_rest(env, neighbor_names, max_along_m, along_axis, rest_gates)
        | any_object_shifted_from_rest(env, neighbor_names, max_depth_m, depth_axis, rest_gates)
    )


def db_r3_placed(
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
    neighbor_names: list[str] | None = None,
    max_neighbor_displacement_m: float | None = None,
    neighbor_reference_m: float | None = None,
    hand_sensor_name: str | None = None,
    wrist_body_name: str | None = None,
    front_plane_xy=None,
    outward_xy=None,
    max_hand_force_n: float = HAND_RELEASED_MAX_FORCE_N,
    min_back_out_m: float = HAND_BACK_OUT_M,
    hold_time_s: float = 0.0,
    hold_key: str = "db_r3_collection",
    ang_vel_threshold_rad_s: float = SETTLED_ANG_VEL_THRESHOLD_RAD_S,
    rest_gates: RestGates | None = None,
    publish_status: bool = True,
    neighbor_knock: dict | None = None,
) -> torch.Tensor:
    """DB-R3 success: the carton seated in the slot and the hand let go of it, held for ``hold_time_s``,
    and, given ``neighbor_knock`` (the keyword arguments of :func:`db_r3_neighbor_knocked` after the names),
    no neighbour knocked."""
    seated = object_standing_in_slot(
        env, object_cfg, object_center_offset_xyz, slot_center_xy, slot_center_halfsize_xy, shelf_top_z_m,
        center_height_range_m, object_local_up_xyz=object_local_up_xyz, min_up_alignment=min_up_alignment,
        velocity_threshold_m_s=velocity_threshold_m_s, ang_vel_threshold_rad_s=ang_vel_threshold_rad_s,
    )
    center = object_center_env(env, object_cfg.name, object_center_offset_xyz)
    error = (center[:, :2] - per_env_rows(slot_center_xy, center.shape[0], 1, center)).abs()
    window = per_env_rows(slot_center_halfsize_xy, center.shape[0], 1, center)
    quat = root_quat_w(env, object_cfg.name)
    up = quat_apply(quat, per_env_rows(object_local_up_xyz, quat.shape[0], 1, quat))[:, 2]
    values = {
        "seated": seated,
        "centred": (error <= window).all(dim=-1),
        "centre_error_depth_mm": error[:, 0] * 1000.0,
        "centre_error_along_mm": error[:, 1] * 1000.0,
        "window_depth_mm": window[:, 0] * 1000.0,
        "window_along_mm": window[:, 1] * 1000.0,
        "upright": up > min_up_alignment,
        "lean_deg": torch.rad2deg(torch.acos(up.clamp(-1.0, 1.0))),
    }
    ok = seated
    limit = neighbor_reference_m if max_neighbor_displacement_m is None else max_neighbor_displacement_m
    if neighbor_names and limit is not None:
        moved = max_rest_displacement(env, list(neighbor_names), rest_gates)
        undisturbed = objects_within_rest_displacement(env, list(neighbor_names), limit, rest_gates)
        values.update(neighbours_max_mm=moved * 1000.0, neighbours_limit_mm=limit * 1000.0, neighbours_ok=undisturbed,
                      neighbours_gate=max_neighbor_displacement_m is not None)
        if max_neighbor_displacement_m is not None:
            ok = ok & undisturbed
    if neighbor_names and neighbor_knock is not None:
        knocked = db_r3_neighbor_knocked(env, list(neighbor_names), **neighbor_knock, rest_gates=rest_gates)
        values["neighbours_knocked"] = knocked
        ok = ok & ~knocked
    if hand_sensor_name is not None:
        force = hand_force_on_object(env, hand_sensor_name)
        back_out = body_past_plane(env, wrist_body_name, front_plane_xy, outward_xy)
        released = (force <= max_hand_force_n) & (back_out >= min_back_out_m)
        values.update(hand_force_n=force, hand_force_limit_n=max_hand_force_n, back_out_cm=back_out * 100.0,
                      back_out_min_cm=min_back_out_m * 100.0, released=released)
        ok = ok & released
    done = sustained(env, ok, hold_time_s, hold_key)
    count = sustained_count(env, hold_key) if hold_time_s > 0 else None
    values.update(all_ok=ok, success=done, hold_s=hold_time_s)
    if count is not None:
        values["held_s"] = count.float() * get_env(env).step_dt
    if publish_status:
        publish(env, "db_r3", values)
    return done


def _print_r2_diag(env, object_cfg, object_center_offset_xyz, neighbor_names, contained, settled, undisturbed) -> None:
    """``DEXBENCH_R2_DIAG=1``: per-env line with the three success factors, the target centre and neighbour drift."""
    center = object_center_env(env, object_cfg.name, object_center_offset_xyz)
    speed = torch.norm(get_root_lin_vel_w(env, object_cfg.name), dim=-1)
    drift = {}
    for name in neighbor_names:
        rest, _recorded = rest_pos_w(env, name)
        drift[name] = torch.norm(get_root_pos_w(env, name) - rest, dim=-1)
    for i in range(center.shape[0]):
        worst = max(drift, key=lambda n: float(drift[n][i])) if drift else None
        print(
            f"[r2diag] env{i} contained={bool(contained[i])} settled={bool(settled[i])} undisturbed={bool(undisturbed[i])} "
            f"target_center=({float(center[i, 0]):.3f}, {float(center[i, 1]):.3f}, {float(center[i, 2]):.3f}) speed={float(speed[i]):.3f}"
            + (f" worst_neighbor={worst} {float(drift[worst][i]) * 1000:.1f} mm" if worst else "")
        )
