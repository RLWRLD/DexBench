# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""A higher-friction physics material on the connector mates, authored on the live stage when the liner spawns."""

from __future__ import annotations

from collections.abc import Sequence

from isaaclab.sim.utils import clone
from pxr import Sdf, Usd, UsdPhysics, UsdShade

from dexbench_arena.assets.usd_file_spawn import spawn_usd_file
from dexbench_arena.assets.connector_harness import extend_harness, set_harness_collision

MATE_MATERIAL_NAME = "M2MateFriction"


def bind_mate_friction(stage: Usd.Stage, root_path: str, colliders: Sequence[str], friction: float | None,
                       torsional: float | None = None, name: str = MATE_MATERIAL_NAME) -> list[str]:
    """Bind a copy of each collider's physics material with static and dynamic friction ``friction`` (None keeps the
    source's) and torsional friction ``torsional`` [m] (``newton:torsionalFriction``; None keeps Newton's default)."""
    if friction is not None:
        friction = float(friction)
        if not friction > 0.0:
            raise ValueError(f"mate friction must be positive, got {friction}")
    if torsional is not None and not float(torsional) >= 0.0:
        raise ValueError(f"torsional friction must be non-negative, got {torsional}")
    bound, made = [], {}
    for relative in colliders:
        prim = stage.GetPrimAtPath(f"{root_path}/{relative}")
        if not prim.IsValid():
            raise ValueError(f"no collider {relative!r} under {root_path}")
        source = UsdShade.MaterialBindingAPI(prim).GetDirectBinding("physics").GetMaterial()
        key = str(source.GetPath()) if source else None
        if key not in made:
            path = Sdf.Path(f"{root_path}/PhysicsMaterials/{name}_{len(made)}")
            material = UsdShade.Material.Define(stage, path)
            if source:
                for attribute in source.GetPrim().GetAttributes():
                    if attribute.HasAuthoredValue():
                        copy = material.GetPrim().CreateAttribute(attribute.GetName(), attribute.GetTypeName(), custom=attribute.IsCustom())
                        copy.Set(attribute.Get())
            api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
            if friction is not None:
                api.CreateStaticFrictionAttr().Set(friction)
                api.CreateDynamicFrictionAttr().Set(friction)
            if torsional is not None:
                for attribute in ("newton:torsionalFriction", "mjc:torsionalfriction"):
                    material.GetPrim().CreateAttribute(attribute, Sdf.ValueTypeNames.Float, custom=False).Set(float(torsional))
            made[key] = material
        binding = UsdShade.MaterialBindingAPI.Apply(prim)
        binding.Bind(made[key], UsdShade.Tokens.weakerThanDescendants, "physics")
        bound.append(str(prim.GetPath()))
    return bound


def set_contact_dim(stage: Usd.Stage, root_path: str, colliders: Sequence[str], condim: int) -> list[str]:
    """Author ``mjc:condim`` (1, 3, 4 or 6) on each collider under ``root_path``; returns the collider paths."""
    if condim not in (1, 3, 4, 6):
        raise ValueError(f"condim must be 1, 3, 4 or 6, got {condim}")
    out = []
    for relative in colliders:
        prim = stage.GetPrimAtPath(f"{root_path}/{relative}")
        if not prim.IsValid():
            raise ValueError(f"no collider {relative!r} under {root_path}")
        prim.CreateAttribute("mjc:condim", Sdf.ValueTypeNames.Int, custom=False).Set(int(condim))
        out.append(str(prim.GetPath()))
    return out


def mate_friction_spawner(colliders: Sequence[str], friction: float | None, condim_colliders: Sequence[str] = (),
                          condim: int | None = None, torsional: float | None = None, *,
                          harness_length_m: float | None = None, harness_key: str = "S2", harness_pose=None,
                          harness_collision: str = "none", harness_joint_armature: float = 1e-6,
                          guide_colliders: Sequence[str] = (), guide_friction: float | None = None,
                          collision_variant: str | None = None):
    """A USD-file spawner that spawns the package, then binds ``friction`` on ``colliders``, and authors ``condim`` and
    binds a copy of their material with ``torsional`` friction on ``condim_colliders`` (each skipped when None).
    An optional harness is reshaped and anchored before cloning, at the configured initial plug pose.
    ``harness_collision='panel'`` enables only cable/panel contact, with the specified joint armature."""
    colliders, condim_colliders = tuple(colliders), tuple(condim_colliders)
    guide_colliders = tuple(guide_colliders)

    @clone
    def spawn(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
        prim = spawn_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation, **kwargs)
        if collision_variant is not None:
            from dexbench_arena.assets.connector_collision_variant import apply_connector_collision_variant
            apply_connector_collision_variant(prim.GetStage(), str(prim.GetPath()), collision_variant)
        if friction:
            bind_mate_friction(prim.GetStage(), str(prim.GetPath()), colliders, friction)
        if guide_friction is not None:
            bind_mate_friction(prim.GetStage(), str(prim.GetPath()), guide_colliders, guide_friction, name="M2GuideFriction")
        if condim is not None:
            set_contact_dim(prim.GetStage(), str(prim.GetPath()), condim_colliders, condim)
        if torsional is not None:
            bind_mate_friction(prim.GetStage(), str(prim.GetPath()), condim_colliders, None, torsional, name="M2PlugContact")
        if harness_length_m is not None:
            extend_harness(prim.GetStage(), str(prim.GetPath()), harness_length_m, harness_key, harness_pose)
        if harness_collision != "none":
            set_harness_collision(prim.GetStage(), str(prim.GetPath()), harness_key, harness_collision, harness_joint_armature)
        return prim

    return spawn
