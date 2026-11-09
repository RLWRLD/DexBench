# SPDX-License-Identifier: Apache-2.0
"""Isaac Lab's undecorated USD-file spawner, the one DexBench's own ``@clone`` spawners call."""

from __future__ import annotations

from isaaclab.sim.spawners.from_files import from_files
from dexbench_arena.assets.scene_bundles import runtime_usd


def spawn_usd_file(prim_path, usd_path, cfg, translation=None, orientation=None, **kwargs):
    """Spawn ``usd_path`` at ``prim_path`` with ``cfg``, as Isaac Lab's USD-file spawner does; returns the prim."""
    spawn = getattr(from_files, "spawn_from_usd_file", None) or from_files._spawn_from_usd_file
    return spawn(prim_path, runtime_usd(usd_path), cfg, translation, orientation, **kwargs)
