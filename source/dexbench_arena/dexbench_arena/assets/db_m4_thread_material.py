# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""DB-M4's ThreadSteel friction, written on the spawned leg and cabinet before the physics model is built."""

from dataclasses import fields

from pxr import Usd, UsdPhysics, UsdShade

from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.utils import clone
from isaaclab.utils.configclass import configclass
from isaaclab.utils.string import string_to_callable

from dexbench_arena.layouts.db_m4_layout import (
    THREAD_DELIVERED_FRICTION,
    THREAD_FRICTION,
    M4ThreadContactMaterialCfg,
)


def set_m4_thread_material(root: Usd.Prim, material: M4ThreadContactMaterialCfg) -> list[str]:
    """Revalue every bound ThreadSteel within this instance; return collider paths."""
    materials, colliders = {}, []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        bound, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose="physics")
        if not bound or bound.GetPrim().GetName() != "ThreadSteel":
            continue
        target = bound.GetPrim()
        if not target.GetPath().HasPrefix(root.GetPath()) or target.IsInstanceProxy():
            raise RuntimeError(f"ThreadSteel must be editable inside {root.GetPath()}: {target.GetPath()}")
        if not target.HasAPI(UsdPhysics.MaterialAPI) or not target.GetAttribute("newton:torsionalFriction"):
            raise RuntimeError(f"ThreadSteel Newton material is missing at {target.GetPath()}; enable Newton physics variants")
        materials[str(target.GetPath())] = target
        colliders.append(str(prim.GetPath()))
    if not colliders:
        raise RuntimeError(f"No ThreadSteel-bound colliders under {root.GetPath()}")
    for prim in materials.values():
        api = UsdPhysics.MaterialAPI(prim)
        api.GetStaticFrictionAttr().Set(float(material.static_friction))
        api.GetDynamicFrictionAttr().Set(float(material.dynamic_friction))
        prim.GetAttribute("newton:torsionalFriction").Set(float(material.torsional_friction))
    return colliders


@clone
def spawn_m4_thread_usd(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Apply before cloning/import, retaining the original file-spawn behavior."""
    root = spawn_from_usd.__wrapped__(prim_path, cfg, translation, orientation, **kwargs)
    material = M4ThreadContactMaterialCfg(cfg.thread_static_friction, cfg.thread_dynamic_friction,
                                         cfg.thread_torsional_friction)
    set_m4_thread_material(root, material)
    return root


@configclass
class M4ThreadUsdFileCfg(UsdFileCfg):
    """Per-instance file-spawn cfg; selected only when the thread friction differs from the delivered one."""

    func = spawn_m4_thread_usd
    thread_static_friction: float = THREAD_FRICTION[0]
    thread_dynamic_friction: float = THREAD_FRICTION[1]
    thread_torsional_friction: float = THREAD_FRICTION[2]


def configure_m4_thread_material(asset, material: M4ThreadContactMaterialCfg) -> None:
    """Give ``asset`` a spawn cfg that writes ``material``; at the delivered friction, keep its own spawn cfg."""
    if material == M4ThreadContactMaterialCfg(*THREAD_DELIVERED_FRICTION):
        return
    old = asset.object_cfg.spawn
    spawn = string_to_callable(str(old.func)) if isinstance(old.func, str) else old.func
    if type(old) is not UsdFileCfg or spawn is not spawn_from_usd:
        raise TypeError("M4 thread material expects the library UsdFileCfg and its standard USD spawner")
    values = {field.name: getattr(old, field.name) for field in fields(old) if field.name != "func"}
    asset.object_cfg.spawn = M4ThreadUsdFileCfg(
        **values, thread_static_friction=material.static_friction,
        thread_dynamic_friction=material.dynamic_friction, thread_torsional_friction=material.torsional_friction,
    ).copy()
