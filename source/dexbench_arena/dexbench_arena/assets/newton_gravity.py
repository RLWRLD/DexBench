# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Gravity compensation on the Newton backend for bodies PhysX excludes from gravity."""

from __future__ import annotations

import re


def newton_gravity_compensation_enabled(args_cli) -> bool:
    """Whether a Newton run compensates gravity (``--newton_gravity_compensation``, on unless switched off)."""
    return getattr(args_cli, "newton_gravity_compensation", True) is not False


def _spawn_then_compensate_gravity(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Spawn through the config's original spawner, then author ``mjc:gravcomp=1.0``."""

    from isaaclab.sim import schemas
    from isaaclab.sim.spawners._utils import subtree_carries_api
    from isaaclab.sim.utils import get_current_stage
    from isaaclab_newton.sim.schemas.schemas_cfg import MujocoRigidBodyCfg
    from pxr import UsdPhysics

    prim = cfg.__dict__["_dexbench_spawn_func_before_gravcomp"](
        prim_path, cfg, translation, orientation, **kwargs
    )
    fragments = [MujocoRigidBodyCfg(gravcomp=1.0)]
    if subtree_carries_api(prim_path, UsdPhysics.RigidBodyAPI, get_current_stage()):
        schemas.apply_rigid_body_properties(re.escape(prim_path) + "(/.*)?", fragments)
    else:
        schemas.apply_rigid_body_properties(re.escape(prim_path), fragments, create_if_missing=True)
    return prim


def _wants_gravity_compensation(spawn) -> bool:
    """Whether this spawn config asked for a body PhysX would exclude from gravity."""

    rigid_props = getattr(spawn, "rigid_props", None)
    if rigid_props is None or isinstance(rigid_props, (list, tuple)):
        return False
    return bool(getattr(rigid_props, "disable_gravity", False))


def _spawn_configs(scene, embodiment):
    """``(label, spawn cfg)`` for every rigid/articulated spawn a run can author."""

    for name, asset in getattr(scene, "assets", {}).items():
        object_type = getattr(getattr(asset, "object_type", None), "name", "")
        if object_type not in ("RIGID", "ARTICULATION"):
            continue
        spawn = getattr(getattr(asset, "object_cfg", None), "spawn", None)
        if spawn is not None:
            yield name, spawn
    robot = getattr(getattr(embodiment, "scene_config", None), "robot", None)
    spawn = getattr(robot, "spawn", None)
    if spawn is not None:
        yield getattr(embodiment, "name", "robot"), spawn


def apply_newton_gravity_compensation(scene, embodiment=None) -> list[str]:
    """Give every ``disable_gravity`` body full MuJoCo gravity compensation on Newton."""

    compensated: list[str] = []
    for name, spawn in _spawn_configs(scene, embodiment):
        if not _wants_gravity_compensation(spawn):
            continue
        if "_dexbench_spawn_func_before_gravcomp" in spawn.__dict__:
            continue
        spawn.__dict__["_dexbench_spawn_func_before_gravcomp"] = spawn.func
        spawn.func = _spawn_then_compensate_gravity
        compensated.append(name)
    return compensated
