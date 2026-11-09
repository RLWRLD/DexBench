# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Shared reset event terms for DexBench embodiments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from dexbench_arena.evaluation.replay_guards import deterministic_reset

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


@deterministic_reset
def reset_all_articulation_joints_and_targets(env: ManagerBasedEnv, env_ids: torch.Tensor):
    """Reset articulation joints to the initial state AND pin their position targets."""
    for articulation_asset in env.scene.articulations.values():
        default_root_state = wp.to_torch(articulation_asset.data.default_root_state)[env_ids].clone()
        default_root_state[:, 0:3] += env.scene.env_origins[env_ids]
        articulation_asset.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
        articulation_asset.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)
        default_joint_pos = wp.to_torch(articulation_asset.data.default_joint_pos)[env_ids].clone()
        default_joint_vel = wp.to_torch(articulation_asset.data.default_joint_vel)[env_ids].clone()
        articulation_asset.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)
        articulation_asset.set_joint_position_target_index(target=default_joint_pos, env_ids=env_ids)
