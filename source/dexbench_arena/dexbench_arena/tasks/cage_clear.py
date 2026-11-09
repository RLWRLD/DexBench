# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""DB-L2's cage-clear boundary: whether the parcel is out of the cage (no simulator imports)."""

import math

import torch

from dexbench_arena.assets.e2_logistics_cell import E2_PARCEL_CAGE_SLOT_XY_M, E2_PARCEL_CAGE_SLOT_YAW_DEG
from dexbench_arena.evaluation.replay_guards import deterministic_reset
from dexbench_arena.tasks.conveyor_placement import box_corners


def cage_front_plane(cage_xy=E2_PARCEL_CAGE_SLOT_XY_M, cage_yaw_deg=E2_PARCEL_CAGE_SLOT_YAW_DEG):
    """The parcel cage's open face, env-local: a point on it and its outward unit normal."""
    yaw = math.radians(cage_yaw_deg)
    return tuple(cage_xy), (math.sin(yaw), -math.cos(yaw))


E2_CAGE_CLEAR_MIN_Z_M = 0.70


def box_clear_of_cage(root_pose, box_sizes_xyz, front=None, margin=0.02, min_z=E2_CAGE_CLEAR_MIN_Z_M):
    """All eight corners beyond the cage front and, unless ``min_z`` is None, above drop height ``min_z``."""
    (px, py), (nx, ny) = front if front is not None else cage_front_plane()
    world = box_corners(root_pose[:, :3], root_pose[:, 3:7], box_sizes_xyz)
    beyond = (world[..., 0] - px) * nx + (world[..., 1] - py) * ny
    clear = beyond.amin(-1) > margin
    return clear if min_z is None else clear & (world[..., 2].amin(-1) > min_z)


@deterministic_reset
def reset_cage_clear(env, env_ids, box_sizes_xyz, success_params):
    """Reset event: clear the Mimic cage-clear signal of the resetting envs (no draw)."""
    env._e2_mimic_box_sizes = box_sizes_xyz
    env._e2_mimic_success_params = success_params
    if not hasattr(env, "_e2_cage_clear"):
        env._e2_cage_clear = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    env._e2_cage_clear[env_ids if env_ids is not None else slice(None)] = False


def package_clear_of_cage(env, object_name, box_sizes_xyz, min_z=E2_CAGE_CLEAR_MIN_Z_M):
    """True where the parcel ``object_name`` is out of the cage now: :func:`box_clear_of_cage` on its env-local pose."""
    base = getattr(env, "unwrapped", env)
    pose = base.arena_world.get_pose_w(object_name).clone()
    pose[:, :3] -= base.scene.env_origins
    return box_clear_of_cage(pose, box_sizes_xyz, min_z=min_z)


def cage_clear_signal(env, env_ids=None):
    env._e2_cage_clear[env.episode_length_buf == 0] = False
    env._e2_cage_clear |= package_clear_of_cage(env, "cage_l2_package", env._e2_mimic_box_sizes)
    return {"cage_clear": env._e2_cage_clear[env_ids if env_ids is not None else slice(None)].clone()}
