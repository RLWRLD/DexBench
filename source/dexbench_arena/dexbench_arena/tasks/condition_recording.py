# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Episode recorder term for a condition recording: did the scene come to rest, and how did the episode end."""

from __future__ import annotations

import torch
from isaaclab_arena.tasks.predicates.object_settling import (
    DEFAULT_ANGULAR_VELOCITY_THRESHOLD,
    DEFAULT_LINEAR_VELOCITY_THRESHOLD,
    compute_objects_settled_mask,
)

from dexbench_arena.environments.manipulation.scene_environment_base import terminating_terms

SETTLE_RECORD_KEY = "settle"


class _BeltFrameWorld:
    """``arena_world`` seen from a belt surface moving at ``velocity_w`` (world frame, m/s): every linear
    velocity less the belt's, everything else as the world reads it."""

    def __init__(self, arena_world, velocity_w: tuple[float, float, float]):
        self._world = arena_world
        self._velocity_w = tuple(float(v) for v in velocity_w)

    def _less_belt(self, velocity: torch.Tensor) -> torch.Tensor:
        return velocity - torch.as_tensor(self._velocity_w, dtype=velocity.dtype, device=velocity.device)

    def get_root_linear_velocity_w(self, scene_key: str) -> torch.Tensor:
        return self._less_belt(self._world.get_root_linear_velocity_w(scene_key))

    def get_nodal_velocities_w(self, scene_key: str) -> torch.Tensor:
        return self._less_belt(self._world.get_nodal_velocities_w(scene_key))

    def __getattr__(self, name: str):
        return getattr(self._world, name)


def record_settle(
    env,
    env_id: int,
    lin_vel_threshold: float = DEFAULT_LINEAR_VELOCITY_THRESHOLD,
    ang_vel_threshold: float = DEFAULT_ANGULAR_VELOCITY_THRESHOLD,
    belt_carried: dict[str, tuple[float, float, float]] | None = None,
) -> dict:
    """``{"settle": {"settled", "unsettled", "terminated_by", ...}}`` for ``env_id``'s finishing episode."""
    names = sorted(env.scene.rigid_objects.keys())
    belt_carried = dict(belt_carried or {})
    missing = sorted(set(belt_carried) - set(names))
    if missing:
        raise ValueError(f"belt-carried objects {missing} are no rigid object of the scene ({names})")

    def frame(name: str):
        velocity_w = belt_carried.get(name)
        return env.arena_world if velocity_w is None else _BeltFrameWorld(env.arena_world, velocity_w)

    unsettled = [
        name
        for name in names
        if not bool(
            compute_objects_settled_mask(frame(name), env.scene, [name], lin_vel_threshold, ang_vel_threshold)[env_id]
        )
    ]
    terminated_by = terminating_terms(env, env_id)
    record = {
        "settled": not unsettled,
        "unsettled": unsettled,
        "terminated_by": terminated_by,
        "lin_vel_threshold": lin_vel_threshold,
        "ang_vel_threshold": ang_vel_threshold,
    }
    if belt_carried:
        record["relative_to_belt"] = {name: [float(v) for v in belt_carried[name]] for name in sorted(belt_carried)}
    return {SETTLE_RECORD_KEY: record}


__all__ = ["SETTLE_RECORD_KEY", "record_settle"]
