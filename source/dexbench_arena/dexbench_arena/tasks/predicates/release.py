# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Release predicates: a hand has let go of an object and drawn back from where it put it."""

from __future__ import annotations

import torch
import warp as wp

from dexbench_arena.tasks.predicates.geometry import per_env_rows
from dexbench_arena.tasks.predicates.object_state import get_env

HAND_RELEASED_MAX_FORCE_N = 1.0
"""At most this much summed contact force [N] from the hand on the object counts as let go. Contact that
holds or presses a carton is several newtons (R3-T2, the lightest, weighs 1.2 N); a fingertip resting on it
reads tenths of a newton."""
HAND_BACK_OUT_M = 0.05
"""How far past the object's front plane [m] the wrist must be, toward the robot, to count as backed out."""

_BODY_IDS_ATTR = "_dexbench_release_body_ids"


def hand_force_on_object(env, sensor_name: str) -> torch.Tensor:
    """Summed contact force [N] the bodies of ``sensor_name`` exert on its filter partners, shape ``(num_envs,)``."""
    data = get_env(env).scene.sensors[sensor_name].data
    forces = data.force_matrix_w
    if forces is None:
        raise ValueError(f"contact sensor {sensor_name!r} has no filter partners, so it cannot tell what the hand touches")
    return forces.torch.norm(dim=-1).sum(dim=(1, 2))


def _body_id(env, robot_name: str, body_name: str) -> int:
    cache = env.__dict__.setdefault(_BODY_IDS_ATTR, {})
    key = (robot_name, body_name)
    if key not in cache:
        ids, _ = env.scene[robot_name].find_bodies(body_name)
        if len(ids) != 1:
            raise ValueError(f"{body_name!r} names {len(ids)} bodies of {robot_name!r}, not one")
        cache[key] = ids[0]
    return cache[key]


def body_past_plane(env, body_name: str, plane_point_xy, outward_xy, robot_name: str = "robot") -> torch.Tensor:
    """How far [m] a robot body is past a vertical plane, along its horizontal outward normal; negative behind it."""
    env = get_env(env)
    body_pos_w = wp.to_torch(env.scene[robot_name].data.body_pos_w)[:, _body_id(env, robot_name, body_name)]
    position = body_pos_w[:, :2] - env.scene.env_origins[:, :2]
    point = per_env_rows(plane_point_xy, position.shape[0], 1, position)
    outward = per_env_rows(outward_xy, position.shape[0], 1, position)
    outward = outward / outward.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    return ((position - point) * outward).sum(dim=-1)


def hand_released(
    env,
    hand_sensor_name: str,
    wrist_body_name: str,
    plane_point_xy,
    outward_xy,
    max_force_n: float = HAND_RELEASED_MAX_FORCE_N,
    min_back_out_m: float = HAND_BACK_OUT_M,
    robot_name: str = "robot",
) -> torch.Tensor:
    """True when the hand presses on the object with at most ``max_force_n`` and its wrist is at least
    ``min_back_out_m`` past the plane, toward the robot."""
    force = hand_force_on_object(env, hand_sensor_name)
    out = body_past_plane(env, wrist_body_name, plane_point_xy, outward_xy, robot_name)
    return (force <= max_force_n) & (out >= min_back_out_m)
