# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Rest-pose predicates: at rest, lifted, fell, displaced."""

from __future__ import annotations

import torch
import warp as wp
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab_arena.relations.physics_settle_params import PhysicsSettleParams
from isaaclab_arena.tasks.predicates.object_settling import get_object_initial_rest_state, objects_settled

from dexbench_arena.tasks.predicates.object_state import get_env, get_root_lin_vel_w, get_root_pos_w
from isaaclab_arena.tasks.predicates.spatial import object_is_above_height

RestGates = tuple[float, float]
"""``(linear m/s, angular rad/s)`` speed gates below which an object counts as at rest."""

REST_GATES_DEFAULT: RestGates = (PhysicsSettleParams().lin_vel_thresh, PhysicsSettleParams().ang_vel_thresh)
"""Gates for recording rest poses: upstream's settle-check defaults (0.1 m/s, 0.1 rad/s)."""
REST_LIN_VEL_THRESHOLD_M_S, REST_ANG_VEL_THRESHOLD_RAD_S = REST_GATES_DEFAULT
GRASP_HEIGHT_GAIN_M = 0.02
"""Height above its rest pose at which a progress chain counts an object as grasped: only a held object rises."""
SETTLED_ANG_VEL_THRESHOLD_RAD_S = REST_GATES_DEFAULT[1]
"""Angular speed gate paired with a predicate's ``velocity_threshold_m_s``."""


def record_rest_poses(env, object_names: list[str], rest_gates: RestGates | None = None) -> torch.Tensor:
    """Let the recorder capture each object's rest pose; True per env when all are at rest now."""
    lin, ang = rest_gates or REST_GATES_DEFAULT
    if not object_names:
        base = get_env(env)
        return torch.ones(base.num_envs, dtype=torch.bool, device=base.device)
    settled = objects_settled(env, list(object_names), lin, ang)
    if len(object_names) > 1 and not bool(settled.all()):
        for name in object_names:
            objects_settled(env, [name], lin, ang)
    return settled


def rest_pos_w(env, name: str, rest_gates: RestGates | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """``(rest position (num_envs, 3) in the world frame, recorded mask (num_envs,))`` of an object."""
    _pos, recorded = get_object_initial_rest_state(env, name)
    if not bool(recorded.all()):
        record_rest_poses(env, [name], rest_gates)
    return get_object_initial_rest_state(env, name)
def object_settled(
    env, name: str, velocity_threshold_m_s: float, ang_vel_threshold_rad_s: float = SETTLED_ANG_VEL_THRESHOLD_RAD_S
) -> torch.Tensor:
    """True when the object is slower than both gates (upstream ``objects_settled`` for one object)."""
    return objects_settled(env, [name], velocity_threshold_m_s, ang_vel_threshold_rad_s)


def object_lifted(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    min_height_gain_m: float = 0.10,
    eef_body_names: tuple[str, ...] | None = None,
    max_eef_distance_m: float = 0.25,
    rest_gates: RestGates | None = None,
) -> torch.Tensor:
    """True when the object is ``min_height_gain_m`` above its rest height, optionally held near an EEF.
    The height test is upstream's ``object_is_above_height(use_settled_state=True)``. With
    ``eef_body_names`` one of the named robot bodies must also be within ``max_eef_distance_m``
    of the object root, which separates a grasp-and-lift from an object flung upward; names
    missing on the current robot are ignored.
    """
    record_rest_poses(env, [object_cfg.name], rest_gates)
    lifted = object_is_above_height(env, object_cfg.name, use_settled_state=True, distance=min_height_gain_m)
    if eef_body_names:
        robot = get_env(env).scene["robot"]
        body_ids = [robot.body_names.index(name) for name in eef_body_names if name in robot.body_names]
        if body_ids:
            root_pos_w = get_root_pos_w(env, object_cfg.name)
            body_pos_w = wp.to_torch(robot.data.body_pos_w)
            distances = torch.norm(body_pos_w[:, body_ids] - root_pos_w.unsqueeze(1), dim=-1)
            lifted = lifted & (distances.min(dim=1).values < max_eef_distance_m)
    return lifted


def object_fell_below_rest(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    drop_height_m: float = 0.15,
    rest_gates: RestGates | None = None,
) -> torch.Tensor:
    """True when the object is ``drop_height_m`` below its rest height (knocked off its support)."""
    rest, recorded = rest_pos_w(env, object_cfg.name, rest_gates)
    height_w = get_root_pos_w(env, object_cfg.name)[:, 2]
    return recorded & ((height_w - rest[:, 2]) < -drop_height_m)


object_fell_below_initial = object_fell_below_rest
"""Alias of :func:`object_fell_below_rest`."""


def any_object_fell_below_rest(
    env: ManagerBasedRLEnv,
    object_names: list[str],
    drop_height_m: float,
    rest_gates: RestGates | None = None,
) -> torch.Tensor:
    """True when ANY named object is ``drop_height_m`` below its rest height (see ``object_fell_below_rest``)."""
    base = get_env(env)
    result = torch.zeros(base.num_envs, dtype=torch.bool, device=base.device)
    for name in object_names:
        result = result | object_fell_below_rest(env, SceneEntityCfg(name), drop_height_m, rest_gates)
    return result


def object_descending_faster_than(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    max_descent_speed_m_s: float = 1.0,
) -> torch.Tensor:
    """True when the object falls faster than ``max_descent_speed_m_s`` (it left the grasp)."""
    return get_root_lin_vel_w(env, object_cfg.name)[:, 2] < -abs(max_descent_speed_m_s)


def max_rest_displacement(
    env: ManagerBasedRLEnv,
    object_names: list[str],
    rest_gates: RestGates | None = None,
) -> torch.Tensor:
    """Largest recorded-rest displacement per env; unrecorded objects contribute zero."""
    base = get_env(env)
    result = torch.zeros(base.num_envs, device=base.device)
    record_rest_poses(env, list(object_names), rest_gates)
    for name in object_names:
        rest, recorded = get_object_initial_rest_state(env, name)
        moved = torch.norm(get_root_pos_w(env, name) - rest, dim=-1)
        result = torch.maximum(result, torch.where(recorded, moved, torch.zeros_like(moved)))
    return result


def objects_within_rest_displacement(
    env: ManagerBasedRLEnv,
    object_names: list[str],
    max_displacement_m: float = 0.005,
    rest_gates: RestGates | None = None,
) -> torch.Tensor:
    """True while every named object is within ``max_displacement_m`` of its rest pose."""
    base = get_env(env)
    result = torch.ones(base.num_envs, dtype=torch.bool, device=base.device)
    record_rest_poses(env, list(object_names), rest_gates)
    for name in object_names:
        rest, recorded = get_object_initial_rest_state(env, name)
        moved = torch.norm(get_root_pos_w(env, name) - rest, dim=-1)
        result = result & recorded & (moved <= max_displacement_m)
    return result


objects_within_spawn_displacement = objects_within_rest_displacement
"""Alias of :func:`objects_within_rest_displacement`; the reference is the recorded rest pose."""


def any_object_lifted_from_rest(
    env: ManagerBasedRLEnv,
    object_names: list[str],
    min_height_gain_m: float,
    rest_gates: RestGates | None = None,
) -> torch.Tensor:
    """True when ANY named object's root is ``min_height_gain_m`` above its rest height."""
    base = get_env(env)
    result = torch.zeros(base.num_envs, dtype=torch.bool, device=base.device)
    record_rest_poses(env, list(object_names), rest_gates)
    for name in object_names:
        rest, recorded = get_object_initial_rest_state(env, name)
        result = result | (recorded & ((get_root_pos_w(env, name)[:, 2] - rest[:, 2]) > min_height_gain_m))
    return result


any_object_lifted_from_spawn = any_object_lifted_from_rest
"""Alias of :func:`any_object_lifted_from_rest`."""


def any_object_shifted_from_rest(
    env: ManagerBasedRLEnv,
    object_names: list[str],
    max_shift_m: float,
    axis: int = 0,
    rest_gates: RestGates | None = None,
) -> torch.Tensor:
    """True when ANY named object's root is more than ``max_shift_m`` from its rest position along world ``axis``."""
    base = get_env(env)
    result = torch.zeros(base.num_envs, dtype=torch.bool, device=base.device)
    record_rest_poses(env, list(object_names), rest_gates)
    for name in object_names:
        rest, recorded = get_object_initial_rest_state(env, name)
        result = result | (recorded & ((get_root_pos_w(env, name)[:, axis] - rest[:, axis]).abs() > max_shift_m))
    return result
