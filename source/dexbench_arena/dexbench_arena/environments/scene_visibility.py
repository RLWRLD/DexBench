# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Render-only hiding of scenery prims, by name, for the interactive and re-rendering scripts."""

from __future__ import annotations

import re

from dexbench_arena.evaluation.replay_guards import deterministic_reset


def hide_replay_prims_callback(previous, cfg, patterns: tuple[str, ...]):
    """Add the opt-in startup visibility event without replacing existing callbacks."""
    if not cfg.hide_replay_prims:
        return previous
    if not patterns:
        raise ValueError("hide_replay_prims: this environment declares no replay_hidden_prim_patterns")

    def callback(env_cfg):
        if previous is not None:
            env_cfg = previous(env_cfg) or env_cfg
        from isaaclab.managers import EventTermCfg

        env_cfg.events.hide_replay_prims = EventTermCfg(
            func=hide_prims_at_startup, mode="startup", params={"patterns": patterns}
        )
        return env_cfg

    return callback


def hide_scene_prims(arena_builder, attribute: str) -> int:
    """Make the prims named by the environment's ``attribute`` patterns invisible."""
    environment = getattr(arena_builder, "dexbench_environment", None)
    hidden = hide_prims_by_name(getattr(environment, attribute, ()))
    if hidden:
        print(f"[viz] hid {len(hidden)} scene prim(s) ({attribute}): {sorted(set(hidden))}")
    return len(hidden)


def hide_prims_by_name(patterns) -> list[str]:
    """Make every imageable prim on the current stage whose NAME matches a regex invisible; return their names."""
    from isaaclab.sim.utils import get_current_stage, set_prim_visibility
    from pxr import UsdGeom

    compiled = [re.compile(p) for p in patterns]
    if not compiled:
        return []
    hidden = []
    for prim in get_current_stage().Traverse():
        if prim.IsA(UsdGeom.Imageable) and any(p.search(prim.GetName()) for p in compiled):
            set_prim_visibility(prim, False)
            hidden.append(prim.GetName())
    return hidden


@deterministic_reset
def hide_prims_at_startup(env, env_ids, patterns: tuple[str, ...]) -> None:
    """Isaac Lab ``startup`` event: :func:`hide_prims_by_name` once the scene is built, before the first step."""
    del env, env_ids
    hidden = hide_prims_by_name(patterns)
    missing = [pattern for pattern in patterns if not any(re.search(pattern, name) for name in hidden)]
    if missing:
        raise RuntimeError(f"hide_replay_prims: no prim on the stage matches {missing}")
    print(f"[viz] hid {len(hidden)} scene prim(s) (replay_hidden_prim_patterns): {sorted(set(hidden))}")
