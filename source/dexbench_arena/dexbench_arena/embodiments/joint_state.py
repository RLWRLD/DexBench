# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Joint-state observation term with a cached name-to-index resolution."""

from __future__ import annotations

from collections.abc import Sequence

import torch

_CACHE_ATTR = "_dexbench_joint_state_cache"


def as_torch(joint_pos):
    """Normalise the backend's joint-position buffer to a torch tensor."""
    if isinstance(joint_pos, torch.Tensor):
        return joint_pos
    import warp as wp

    return wp.to_torch(joint_pos)


def resolve_joint_order(asset, joint_names: Sequence[str]) -> tuple[list[int], list[str]]:
    """Resolve exact joint names in the requested public action/state order."""
    requested = list(joint_names)
    joint_ids, resolved = asset.find_joints(requested, preserve_order=True)
    if resolved != requested:
        missing = [name for name in requested if name not in resolved]
        unexpected = [name for name in resolved if name not in requested]
        raise ValueError(
            "Failed to resolve the exact joint order. "
            f"missing={missing}, unexpected={unexpected}, requested={requested}, resolved={resolved}"
        )
    return joint_ids, resolved


def resolve_joint_index(asset, joint_names: Sequence[str], device) -> torch.Tensor:
    """Return (and memoise on ``asset``) the index tensor for ``joint_names``."""
    requested = list(joint_names)
    key = (tuple(requested), str(device))
    cache = getattr(asset, _CACHE_ATTR, None)
    if cache is None:
        cache = {}
        try:
            setattr(asset, _CACHE_ATTR, cache)
        except AttributeError:
            cache = None
    if cache is not None and key in cache:
        return cache[key]
    joint_ids, _ = resolve_joint_order(asset, requested)
    index = torch.tensor(joint_ids, dtype=torch.long, device=device)
    if cache is not None:
        cache[key] = index
    return index


def get_joint_state(env, joint_names: Sequence[str]) -> torch.Tensor:
    """Joint positions for ``joint_names``, in the requested order."""
    robot = env.scene["robot"]
    joint_pos = as_torch(robot.data.joint_pos)
    index = resolve_joint_index(robot, joint_names, joint_pos.device)
    return joint_pos.index_select(1, index)
