# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Conveyor placement geometry and sustained success, usable without Kit for dataset audits."""

from __future__ import annotations

import math

import torch

from dexbench_arena.evaluation.replay_guards import deterministic_reset
from dexbench_arena.tasks.conveyor_placement_config import (
    CONVEYOR_PLACE_ANGULAR_SPEED_RAD_S,
    CONVEYOR_PLACE_BOTTOM_HEIGHT_TOLERANCE_M,
    CONVEYOR_PLACE_HOLD_TIME_S,
    CONVEYOR_PLACE_LINEAR_SPEED_M_S,
    CONVEYOR_PLACE_MAX_FACE_TILT_DEG,
)
from dexbench_arena.tasks.predicates.hold import reset_sustained, sustained


def box_sizes_per_env(box_sizes_xyz, like: torch.Tensor) -> torch.Tensor:
    """Expand a member-size table to the actual box in each environment."""
    sizes = torch.as_tensor(box_sizes_xyz, dtype=like.dtype, device=like.device).reshape(-1, 3)
    return sizes[torch.arange(like.shape[0], device=like.device) % len(sizes)]


def rotate_xyzw(quat: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    """Apply unit xyzw quaternions, including a broadcast corner dimension."""
    xyz = quat[..., :3].expand_as(vectors)
    cross = 2.0 * torch.linalg.cross(xyz, vectors, dim=-1)
    return vectors + quat[..., 3:] * cross + torch.linalg.cross(xyz, cross, dim=-1)


def box_corners(root_pos: torch.Tensor, root_quat: torch.Tensor, box_sizes_xyz) -> torch.Tensor:
    """The eight corners of each environment's box, (num_envs, 8, 3); the first four are the bottom face."""
    sizes = box_sizes_per_env(box_sizes_xyz, root_pos)
    unit_corners = root_pos.new_tensor(
        [
            [-0.5, -0.5, 0.0],
            [-0.5, 0.5, 0.0],
            [0.5, -0.5, 0.0],
            [0.5, 0.5, 0.0],
            [-0.5, -0.5, 1.0],
            [-0.5, 0.5, 1.0],
            [0.5, -0.5, 1.0],
            [0.5, 0.5, 1.0],
        ]
    )
    return root_pos[:, None, :] + rotate_xyzw(root_quat[:, None, :], sizes[:, None, :] * unit_corners)


def placement_measurements(
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    lin_vel: torch.Tensor,
    ang_vel: torch.Tensor,
    box_sizes_xyz,
) -> dict[str, torch.Tensor]:
    """What box_placement_checks compares with its thresholds, per environment."""
    sizes = box_sizes_per_env(box_sizes_xyz, root_pos)
    corners = box_corners(root_pos, root_quat, box_sizes_xyz)
    eye = torch.eye(3, dtype=root_pos.dtype, device=root_pos.device)
    axes_z = torch.stack([rotate_xyzw(root_quat, eye[i].expand_as(root_pos))[:, 2] for i in range(3)], dim=-1)
    half_height = torch.zeros_like(root_pos)
    half_height[:, 2] = sizes[:, 2] / 2.0
    long_side_axis = (sizes[:, 1] * sizes[:, 2] < sizes[:, 0] * sizes[:, 2]).long()
    return {
        "corners": corners,
        "centre": root_pos + rotate_xyzw(root_quat, half_height),
        "axes_z": axes_z,
        "down_axis": axes_z.abs().argmax(dim=-1),
        "long_side_axis": long_side_axis,
        "speed": torch.linalg.vector_norm(lin_vel, dim=-1),
        "spin": torch.linalg.vector_norm(ang_vel, dim=-1),
    }


def over_surfaces(centre_xy: torch.Tensor, surface_xy_rects) -> torch.Tensor:
    """Where ``centre_xy`` (num_envs, 2) lies over one of the ``((min_x, min_y), (max_x, max_y))`` rectangles."""
    rects = centre_xy.new_tensor(surface_xy_rects)
    xy = centre_xy[:, None, :]
    return ((xy >= rects[:, 0]) & (xy <= rects[:, 1])).all(dim=-1).any(dim=-1)


def box_placement_checks(
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    lin_vel: torch.Tensor,
    ang_vel: torch.Tensor,
    box_sizes_xyz,
    surface_xy_rects,
    surface_top_z_range_m,
    bottom_height_tolerance_m: float,
    max_face_tilt_deg: float,
    velocity_threshold_m_s: float,
    angular_velocity_threshold_rad_s: float,
    measured: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Instantaneous checks on env-local poses that the box rests on the conveyor; each (num_envs,)."""
    if measured is None:
        measured = placement_measurements(root_pos, root_quat, lin_vel, ang_vel, box_sizes_xyz)
    low, high = (float(value) for value in surface_top_z_range_m)
    lowest = measured["corners"][..., 2].min(dim=-1).values
    return {
        "over": over_surfaces(measured["centre"][:, :2], surface_xy_rects),
        "seated": (lowest >= low - bottom_height_tolerance_m) & (lowest <= high + bottom_height_tolerance_m),
        "flat": measured["axes_z"].abs().max(dim=-1).values >= math.cos(math.radians(max_face_tilt_deg)),
        "not_on_long_side": measured["down_axis"] != measured["long_side_axis"],
        "linear_slow": measured["speed"] < velocity_threshold_m_s,
        "angular_slow": measured["spin"] < angular_velocity_threshold_rad_s,
    }


def describe_placement_failures(root_pos, measured, checks, params, envs) -> list[str]:
    """For each environment index in ``envs``, the checks it failed, each with its value ("" if none)."""
    if not envs:
        return []
    host = {name: value[envs].cpu() for name, value in (*measured.items(), *checks.items())}
    low, high = (float(value) for value in params["surface_top_z_range_m"])
    tolerance = params["bottom_height_tolerance_m"]
    band = (round(low - tolerance, 4), round(high + tolerance, 4))
    lines = []
    for row in range(len(envs)):
        reasons = []
        if not host["over"][row]:
            x, y = (float(v) for v in host["centre"][row, :2])
            reasons.append(f"over: centre ({x:.3f}, {y:.3f}) over none of the conveyor's surfaces")
        if not host["seated"][row]:
            lowest = float(host["corners"][row, :, 2].min())
            reasons.append(f"seated: lowest corner z {lowest:.3f} not in [{band[0]}, {band[1]}]")
        if not host["flat"][row]:
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(host["axes_z"][row].abs().max())))))
            reasons.append(f"flat: {tilt:.1f} deg off its nearest face > {params['max_face_tilt_deg']:.1f}")
        if not host["not_on_long_side"][row]:
            reasons.append("not_on_long_side: lying on a long side face")
        if not host["linear_slow"][row]:
            limit = params["velocity_threshold_m_s"]
            reasons.append(f"linear_slow: {float(host['speed'][row]):.3f} m/s >= {limit:.3f}")
        if not host["angular_slow"][row]:
            limit = params["angular_velocity_threshold_rad_s"]
            reasons.append(f"angular_slow: {float(host['spin'][row]):.2f} rad/s >= {limit:.2f}")
        lines.append("; ".join(reasons))
    return lines


def placement_hold_key(object_name: str) -> str:
    """Key of the object's placement hold in ``predicates.hold`` (read it with ``sustained_count``)."""
    return f"conveyor_placement/{object_name}"


@deterministic_reset
def reset_conveyor_placement(env, env_ids, object_name: str) -> None:
    """Reset event paired with object_settled_on_conveyor (also used by recording): clears its hold count."""
    reset_sustained(env, env_ids, placement_hold_key(object_name))
    log = getattr(getattr(env, "unwrapped", env), _PLACEMENT_LOG_ATTR, {}).get(object_name)
    if log is not None:
        for index in range(len(log["state"])) if env_ids is None else torch.as_tensor(env_ids).tolist():
            log["state"][index] = log["printed_at"][index] = None


def object_settled_on_conveyor(
    env,
    object_cfg,
    box_sizes_xyz,
    surface_xy_rects,
    surface_top_z_range_m,
    bottom_height_tolerance_m: float = CONVEYOR_PLACE_BOTTOM_HEIGHT_TOLERANCE_M,
    max_face_tilt_deg: float = CONVEYOR_PLACE_MAX_FACE_TILT_DEG,
    velocity_threshold_m_s: float = CONVEYOR_PLACE_LINEAR_SPEED_M_S,
    angular_velocity_threshold_rad_s: float = CONVEYOR_PLACE_ANGULAR_SPEED_RAD_S,
    hold_time_s: float = CONVEYOR_PLACE_HOLD_TIME_S,
    log_checks: bool = False,
) -> torch.Tensor:
    """The box at rest on the conveyor (box_placement_checks) held for 0.5 s, shared by evaluation and recording."""
    import warp as wp

    base = env.unwrapped
    data = base.scene[object_cfg.name].data
    root_pos = wp.to_torch(data.root_pos_w) - base.scene.env_origins
    root_quat = wp.to_torch(data.root_quat_w)
    lin_vel, ang_vel = wp.to_torch(data.root_lin_vel_w), wp.to_torch(data.root_ang_vel_w)
    params = {
        "box_sizes_xyz": box_sizes_xyz,
        "surface_xy_rects": surface_xy_rects,
        "surface_top_z_range_m": surface_top_z_range_m,
        "bottom_height_tolerance_m": bottom_height_tolerance_m,
        "max_face_tilt_deg": max_face_tilt_deg,
        "velocity_threshold_m_s": velocity_threshold_m_s,
        "angular_velocity_threshold_rad_s": angular_velocity_threshold_rad_s,
    }
    measured = placement_measurements(root_pos, root_quat, lin_vel, ang_vel, box_sizes_xyz)
    checks = box_placement_checks(root_pos, root_quat, lin_vel, ang_vel, **params, measured=measured)
    ok = torch.stack(tuple(checks.values())).all(dim=0)
    placed = sustained(base, ok, hold_time_s, placement_hold_key(object_cfg.name))
    if log_checks:
        _log_placement(base, object_cfg.name, root_pos, measured, checks, ok, placed, params, hold_time_s)
    return placed


_PLACEMENT_LOG_ATTR = "_dexbench_placement_logs"
_LOG_NEAR_DECK_Z_M = 0.05
_LOG_NEAR_DECK_XY_M = 0.40
_LOG_REPEAT_S = 1.0
_LOG_MIN_GAP_S = 0.2


def _log_placement(base, object_name, root_pos, measured, checks, ok, placed, params, hold_time_s) -> None:
    """Print each environment's placement state when it changes while the box is at the decks."""
    log = getattr(base, _PLACEMENT_LOG_ATTR, None)
    if log is None:
        log = {}
        setattr(base, _PLACEMENT_LOG_ATTR, log)
    entry = log.get(object_name)
    num_envs = ok.shape[0]
    if entry is None or len(entry["state"]) != num_envs:
        entry = log[object_name] = {"step": None, "state": [None] * num_envs, "printed_at": [None] * num_envs}
    step = int(base.common_step_counter)
    if entry["step"] == step:
        return
    entry["step"] = step
    rects = params["surface_xy_rects"]
    area_low = [min(rect[0][axis] for rect in rects) - _LOG_NEAR_DECK_XY_M for axis in (0, 1)]
    area_high = [max(rect[1][axis] for rect in rects) + _LOG_NEAR_DECK_XY_M for axis in (0, 1)]
    top = float(params["surface_top_z_range_m"][1]) + float(params["bottom_height_tolerance_m"])
    x, y = root_pos[:, 0], root_pos[:, 1]
    near = (
        (measured["corners"][:, :4, 2].min(dim=-1).values <= top + _LOG_NEAR_DECK_Z_M)
        & (x >= area_low[0]) & (x <= area_high[0]) & (y >= area_low[1]) & (y <= area_high[1])
    ).tolist()
    envs = [env for env, at_deck in enumerate(near) if at_deck]
    for env, at_deck in enumerate(near):
        if not at_deck:
            entry["state"][env] = entry["printed_at"][env] = None
    if not envs:
        return
    placed_now, ok_now = placed[envs].tolist(), ok[envs].tolist()
    passed = {name: value[envs].tolist() for name, value in checks.items()}
    seconds = (base.episode_length_buf[envs] * base.step_dt).tolist()
    failing = [row for row, good in enumerate(ok_now) if not good]
    described = describe_placement_failures(root_pos, measured, checks, params, [envs[row] for row in failing])
    reasons = dict(zip(failing, described))
    repeat_steps = max(1, round(_LOG_REPEAT_S / base.step_dt))
    gap_steps = max(1, round(_LOG_MIN_GAP_S / base.step_dt))
    for row, env in enumerate(envs):
        if placed_now[row]:
            current, line = "placed", f"placed: every check held {hold_time_s:.2f} s"
        elif ok_now[row]:
            current, line = "holding", f"every check passes; holding for {hold_time_s:.2f} s"
        else:
            current = tuple(name for name, values in passed.items() if not values[row])
            line = "not placed -- " + reasons[row]
        since = None if entry["printed_at"][env] is None else step - entry["printed_at"][env]
        if current == entry["state"][env]:
            if current in ("placed", "holding") or since < repeat_steps:
                continue
        elif current != "placed" and isinstance(entry["state"][env], tuple) and since < gap_steps:
            continue
        entry["state"][env] = current
        entry["printed_at"][env] = step
        print(f"[lift_to_conveyor] env {env} t={seconds[row]:.1f}s {line}", flush=True)


def box_over_conveyor(env, object_cfg, box_sizes_xyz, surface_xy_rects, min_z_m: float) -> torch.Tensor:
    """The partial-credit chain's "over the conveyor": the box's centre over one of the surfaces that
    carry it (box_placement_checks' ``over``) and its lowest corner not below ``min_z_m``."""
    import warp as wp

    base = env.unwrapped
    data = base.scene[object_cfg.name].data
    root_pos = wp.to_torch(data.root_pos_w) - base.scene.env_origins
    still = torch.zeros_like(root_pos)
    measured = placement_measurements(root_pos, wp.to_torch(data.root_quat_w), still, still, box_sizes_xyz)
    lowest = measured["corners"][..., 2].min(dim=-1).values
    return over_surfaces(measured["centre"][:, :2], surface_xy_rects) & (lowest >= min_z_m)


def conveyor_box_outside_workspace(env, object_cfg, box_sizes_xyz, xy_bounds) -> torch.Tensor:
    """Existing workspace failure, using each member's actual centre offset."""
    import warp as wp

    base = env.unwrapped
    data = base.scene[object_cfg.name].data
    pos = wp.to_torch(data.root_pos_w) - base.scene.env_origins
    sizes = box_sizes_per_env(box_sizes_xyz, pos)
    offset = torch.zeros_like(pos)
    offset[:, 2] = sizes[:, 2] / 2.0
    center = pos + rotate_xyzw(wp.to_torch(data.root_quat_w), offset)
    bounds = pos.new_tensor(xy_bounds)
    return ((center[:, :2] < bounds[0]) | (center[:, :2] > bounds[1])).any(dim=-1)
