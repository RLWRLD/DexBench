# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Spawn-time topology normalization for the RBY1 Wuji v2 asset."""

from __future__ import annotations

from collections.abc import Callable

from isaaclab.sim import schemas
from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.utils import clone
from isaaclab.utils.configclass import configclass
from pxr import Sdf, Usd, UsdGeom, UsdPhysics

from dexbench_arena.embodiments.rby1_wuji2.constants import RBY1_WUJI2_FINGERTIP_PAD_BODY_NAMES


_NORMALIZED_LAYERS: dict[str, Sdf.Layer] = {}

HAND_CONTACT_MATERIAL_NAME = "hand_contact"


def _is_task_contact_body(name: str) -> bool:
    """Keep detailed collision geometry only on manipulating arms and hands."""
    if name.startswith(("link_right_arm_", "link_left_arm_")):
        return True
    for side in ("right", "left"):
        if name == f"{side}_wrist" or name.startswith(tuple(
            f"{side}_{finger}_" for finger in
            ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky")
        )):
            return True
    return False


def _is_passive_wheel(name: str) -> bool:
    return name.startswith("wheel_")


def is_simready_build(root: Usd.Prim) -> bool:
    """The SimReady build of the robot (``embodiments/simready/<name>/v<x.y.z>/``) carries the
    Isaac robot schema on its default prim; the DEFT export it was built from does not."""
    return bool(root.GetAttribute("isaac:robotType"))


def _normalize_wuji2_layer(usd_path: str) -> Sdf.Layer:
    """Build an in-memory USD wrapper with one backend-neutral articulation tree."""
    if usd_path in _NORMALIZED_LAYERS:
        return _NORMALIZED_LAYERS[usd_path]

    source_stage = Usd.Stage.Open(usd_path)
    if source_stage is None or not source_stage.GetDefaultPrim().IsValid():
        raise RuntimeError(f"Could not open Wuji v2 USD default prim: {usd_path}")
    source_root_path = source_stage.GetDefaultPrim().GetPath()

    layer = Sdf.Layer.CreateAnonymous("rby1_wuji2_normalized.usda")
    stage = Usd.Stage.Open(layer)
    root = stage.DefinePrim(source_root_path, "Xform")
    root.GetReferences().AddReference(usd_path, source_root_path)
    stage.SetDefaultPrim(root)
    if not is_simready_build(root):
        raise RuntimeError(
            f"Not the SimReady Wuji v2 build (no isaac:robotType on the default prim): {usd_path}"
        )

    rigid_bodies = [prim for prim in Usd.PrimRange(root) if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    if not rigid_bodies:
        raise RuntimeError(f"Wuji v2 USD has no rigid bodies: {usd_path}")
    shape_count = 0
    wheels_off = 0
    for body in rigid_bodies:
        collisions = body.GetChild("collisions")
        visuals = body.GetChild("visuals")
        if not collisions.IsValid() or not visuals.IsValid():
            raise RuntimeError(f"Expected Wuji v2 visual/collision groups below {body.GetPath()}")
        for group in (collisions, visuals):
            if group.IsInstanceable():
                group.SetInstanceable(False)
        if _is_passive_wheel(body.GetName()):
            collisions.SetActive(False)
            wheels_off += 1
            continue
        shapes = sum(
            prim.IsA(UsdGeom.Gprim) and prim.HasAPI(UsdPhysics.CollisionAPI)
            for prim in Usd.PrimRange(collisions)
        )
        if not shapes:
            raise RuntimeError(f"Wuji v2 body has no collision shapes: {body.GetPath()}")
        shape_count += shapes
    print(
        f"[wuji2-spawn] normalized {len(rigid_bodies)} rigid bodies: "
        f"{shape_count} collision shapes, {wheels_off} wheel colliders off"
    )

    base = stage.GetPrimAtPath(source_root_path.AppendPath("joints/base_fixed"))
    if not (base.IsValid() and base.HasAPI(UsdPhysics.ArticulationRootAPI)):
        raise RuntimeError(f"Expected the Wuji v2 articulation root on joints/base_fixed under {source_root_path}")
    UsdPhysics.ArticulationRootAPI.Apply(root)
    if "PhysxArticulationAPI" not in root.GetAppliedSchemas():
        root.AddAppliedSchema("PhysxArticulationAPI")
    articulation_attrs = set(UsdPhysics.ArticulationRootAPI.GetSchemaAttributeNames())
    for source_attr in base.GetAttributes():
        attr_name = source_attr.GetName()
        if attr_name not in articulation_attrs and not attr_name.startswith("physxArticulation:"):
            continue
        target_attr = root.GetAttribute(attr_name)
        if not target_attr:
            target_attr = root.CreateAttribute(attr_name, source_attr.GetTypeName())
        target_attr.Set(source_attr.Get())
    base.RemoveAppliedSchema("PhysxArticulationAPI")
    base.RemoveAPI(UsdPhysics.ArticulationRootAPI)

    _NORMALIZED_LAYERS[usd_path] = layer
    return layer


def hand_contact_materials(root: Usd.Prim) -> list[Usd.Prim]:
    """The build's hand contact material prims under ``root`` (one per hand body)."""
    return [
        prim
        for prim in Usd.PrimRange(root)
        if prim.GetName() == HAND_CONTACT_MATERIAL_NAME and prim.HasAPI(UsdPhysics.MaterialAPI)
    ]


def set_hand_contact_material(
    root: Usd.Prim, static_friction: float, dynamic_friction: float, friction_combine_mode: str
) -> int:
    """Give every hand contact material under ``root`` these coefficients; returns how many."""
    materials = hand_contact_materials(root)
    if not materials:
        raise RuntimeError(
            f"No '{HAND_CONTACT_MATERIAL_NAME}' physics material under {root.GetPath()}: the robot build"
            " predates rby1m_wujihand2_beta2 v0.4.0, which authors it on the hand colliders"
        )
    for prim in materials:
        api = UsdPhysics.MaterialAPI(prim)
        api.GetStaticFrictionAttr().Set(float(static_friction))
        api.GetDynamicFrictionAttr().Set(float(dynamic_friction))
        prim.GetAttribute("physxMaterial:frictionCombineMode").Set(friction_combine_mode)
    return len(materials)


def set_fingertip_contact_dim(root: Usd.Prim, condim: int) -> list[str]:
    """Author Newton/MuJoCo contact dimension on sensor pads only, before cloning/model construction."""
    if condim not in (1, 3, 4, 6):
        raise ValueError(f"Invalid fingertip contact dimension: {condim}")
    paths = []
    for name in RBY1_WUJI2_FINGERTIP_PAD_BODY_NAMES:
        body = root.GetChild(name)
        colliders = [p for p in Usd.PrimRange(body) if p.HasAPI(UsdPhysics.CollisionAPI)] if body else []
        if not colliders:
            raise ValueError(f"No sensor-pad collider under {root.GetPath()}/{name}")
        for prim in colliders:
            prim.CreateAttribute("mjc:condim", Sdf.ValueTypeNames.Int, custom=False).Set(int(condim))
            paths.append(str(prim.GetPath()))
    return paths


def configure_bimanual_collision_pairs(root: Usd.Prim) -> dict[str, int]:
    """Enable opposing arms/hands while preserving the baseline's own-chain exclusions."""
    bodies = [p for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    sides = {}
    for body in bodies:
        name = body.GetName()
        sides[body.GetPath()] = (
            "left" if name.startswith(("left_", "link_left_arm_")) else "right"
        ) if _is_task_contact_body(name) else None
    filtered = 0
    enabled = 0
    for i, body in enumerate(bodies):
        targets = []
        for other in bodies[i + 1:]:
            a, b = sides[body.GetPath()], sides[other.GetPath()]
            if a is not None and b is not None and a != b:
                enabled += 1
            else:
                targets.append(other.GetPath())
                filtered += 1
        api = UsdPhysics.FilteredPairsAPI.Apply(body)
        relationship = api.CreateFilteredPairsRel()
        existing = relationship.GetTargets()
        relationship.SetTargets(list(dict.fromkeys([*existing, *targets])))
    return {"filtered_pairs": filtered, "opposing_limb_pairs": enabled}


def _spawn_normalized(prim_path, cfg, translation, orientation, **kwargs) -> Usd.Prim:
    articulation_props = cfg.articulation_props
    if articulation_props is None or articulation_props.fix_root_link is not None:
        raise ValueError("Wuji v2 topology normalization requires fix_root_link=None")
    normalized_layer = _normalize_wuji2_layer(cfg.usd_path)
    prim = spawn_from_usd(prim_path, cfg.replace(articulation_props=None), translation, orientation, **kwargs)
    references = prim.GetReferences()
    references.ClearReferences()
    references.AddReference(normalized_layer.identifier)
    schemas.modify_articulation_root_properties(prim_path, articulation_props)
    if getattr(cfg, "hand_static_friction", None) is not None:
        set_hand_contact_material(
            prim, cfg.hand_static_friction, cfg.hand_dynamic_friction, cfg.hand_friction_combine_mode
        )
    if getattr(cfg, "fingertip_contact_dim", None) is not None:
        set_fingertip_contact_dim(prim, cfg.fingertip_contact_dim)
    if getattr(cfg, "bimanual_collision_filter", False):
        configure_bimanual_collision_pairs(prim)
    return prim


@clone
def spawn_rby1_wuji2_usd(
    prim_path: str,
    cfg: Rby1Wuji2UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn Wuji v2 through its normalized, clone-safe in-memory USD layer."""
    return _spawn_normalized(prim_path, cfg, translation, orientation, **kwargs)



@configclass
class Rby1Wuji2UsdFileCfg(UsdFileCfg):
    """The robot's spawn config: the build's USD plus the task-selected hand contact material."""

    func: Callable = spawn_rby1_wuji2_usd
    hand_static_friction: float | None = None
    hand_dynamic_friction: float | None = None
    hand_friction_combine_mode: str = "max"
    fingertip_contact_dim: int | None = None
    bimanual_collision_filter: bool = False
    """Preserve collection self-collision exclusions while opposing limbs can contact."""
