# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
"""Object-state getters the DexBench predicates read through.

These getters are retained from Arena's ``isaaclab_arena.tasks.predicates.predicate_utils`` as of
c5826c87a, kept here because the pin that followed removed the module: Arena
79959e4e0 (isaac-sim/IsaacLab-Arena#1195, "Use ArenaWorld in task predicates")
moved its own predicates onto ``ArenaWorld`` and dropped the env-based helpers.
The DexBench predicates still take the env first -- that is what makes them
usable as ``TerminationTermCfg`` funcs -- so the getters live on unchanged.
The unused upstream ``select`` helper is omitted; this module docstring is local.
"""

from __future__ import annotations

import torch
import warp as wp
from isaaclab.assets import RigidObject


def get_env(env):
    """Resolve to the unwrapped manager-based env regardless of wrapper depth."""
    seen = set()
    while hasattr(env, "unwrapped") and env.unwrapped is not env and id(env) not in seen:
        seen.add(id(env))
        env = env.unwrapped
    return env


def get_rigid_object(env, name: str) -> RigidObject:
    """Get a rigid object from the env's scene."""
    return get_env(env).scene[name]


def get_root_pos_w(env, name: str) -> torch.Tensor:
    """Get the root position of a rigid object in the world frame."""
    return wp.to_torch(get_rigid_object(env, name).data.root_pos_w)


def get_root_lin_vel_w(env, name: str) -> torch.Tensor:
    """Get the root linear velocity of a rigid object in the world frame."""
    return wp.to_torch(get_rigid_object(env, name).data.root_lin_vel_w)


def get_root_ang_vel_w(env, name: str) -> torch.Tensor:
    """Get the root angular velocity of a rigid object in the world frame."""
    return wp.to_torch(get_rigid_object(env, name).data.root_ang_vel_w)
