# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""The task-selected hand contact material: spawn configuration, check and recording."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import asdict

from dexbench_arena.contact_materials import HandContactMaterialCfg, WUJI_GRASP_CONTACT_MATERIAL
from dexbench_arena.embodiments.episode_metadata import install_episode_metadata
from dexbench_arena.evaluation.replay_guards import deterministic_reset


class HandContactMaterialMixin:
    """Embodiment capability: the robot spawns with the selected hand material, an event records it."""

    default_hand_contact_material: HandContactMaterialCfg = WUJI_GRASP_CONTACT_MATERIAL
    hand_contact_material: HandContactMaterialCfg = WUJI_GRASP_CONTACT_MATERIAL
    model_hand_contact_material: HandContactMaterialCfg
    """What the hand carries with the task material disabled: the robot model's own contact
    material, declared by the embodiment (Wuji v2: its MJCF's 1.0 / 1.0, max)."""

    def configure_hand_contact_material(
        self, material: HandContactMaterialCfg, *, source: str, overrides: dict | None = None
    ) -> None:
        """Give this instance's robot spawn the material and its check event the provenance."""
        self.hand_contact_material = material
        applied = material if material.enabled else self.model_hand_contact_material
        spawn = self.scene_config.robot.spawn
        spawn.hand_static_friction = applied.static_friction
        spawn.hand_dynamic_friction = applied.dynamic_friction
        spawn.hand_friction_combine_mode = applied.friction_combine_mode
        term = self.event_config.hand_contact_material_check
        term.params = {**term.params, **asdict(material), "source": source, "overrides": dict(overrides or {})}


def record_hand_contact_material(env, env_id: int) -> dict:
    """Expose the applied material in Arena's evaluation episode records."""
    del env_id
    return {"hand_contact_material": deepcopy(env.cfg.dexbench_hand_contact_material)}


def install_hand_contact_recording_metadata(env, env_ids=None) -> None:
    """Install once the env's final recorder manager exists (``recorder_ready`` event); keep config data serializable."""
    del env_ids
    install_episode_metadata(env, "hand_contact_material", lambda: env.cfg.dexbench_hand_contact_material)


@deterministic_reset
def check_hand_contact_material(
    env,
    env_ids,
    hand_bodies: tuple[str, ...],
    static_friction: float,
    dynamic_friction: float,
    friction_combine_mode: str = "max",
    enabled: bool = True,
    source: str = "embodiment",
    overrides: dict | None = None,
    material_name: str = "hand_contact",
) -> None:
    """Startup event: check the hand colliders bind the material the run selected, and record it."""
    del env_ids
    from pxr import Usd, UsdPhysics, UsdShade

    materials, layouts, bound = {}, set(), 0
    for env_prim_path in env.scene.env_prim_paths:
        robot = env.sim.stage.GetPrimAtPath(f"{env_prim_path}/Robot")
        if not robot.IsValid():
            raise RuntimeError(f"No robot at {env_prim_path}/Robot to check the hand material on")
        below, hands = {}, {}
        for prim in Usd.PrimRange(robot):
            if prim.IsA(UsdPhysics.Joint):
                joint = UsdPhysics.Joint(prim)
                parent, child = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
                if parent and child and child[0].HasPrefix(robot.GetPath()):
                    below.setdefault(parent[0], []).append(child[0])
            elif prim.GetName() in hand_bodies and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                hands[prim.GetName()] = prim.GetPath()
        if missing := sorted(set(hand_bodies) - set(hands)):
            raise RuntimeError(f"No hand body {missing} under {env_prim_path}/Robot")
        hand, stack = set(), list(hands.values())
        while stack:
            body = stack.pop()
            if body not in hand:
                hand.add(body)
                stack.extend(below.get(body, ()))
        colliders = {
            prim.GetPath(): (prim, body)
            for body in hand
            for prim in Usd.PrimRange(env.sim.stage.GetPrimAtPath(body))
            if prim.HasAPI(UsdPhysics.CollisionAPI)
        }
        if not colliders:
            raise RuntimeError(f"No collider on the hand bodies {sorted(hands)} under {env_prim_path}/Robot")
        unbound = []
        for path, (prim, body) in sorted(colliders.items()):
            material = UsdShade.MaterialBindingAPI(prim).GetDirectBinding("physics").GetMaterial()
            applied = prim.HasAPI(UsdShade.MaterialBindingAPI) and material
            if applied and material.GetPrim().GetName() == material_name:
                materials[str(material.GetPath())] = material.GetPrim()
                layouts.add(f"*/{material.GetPath().MakeRelativePath(body)}")
            else:
                unbound.append(str(path))
        if unbound:
            raise RuntimeError(
                f"{len(unbound)} of {len(colliders)} hand colliders under {env_prim_path}/Robot do not bind a"
                f" '{material_name}' material: {unbound[:10]}"
            )
        bound += len(colliders)

    def coefficients(prim):
        api = UsdPhysics.MaterialAPI(prim)
        return (
            api.GetStaticFrictionAttr().Get(),
            api.GetDynamicFrictionAttr().Get(),
            prim.GetAttribute("physxMaterial:frictionCombineMode").Get(),
        )

    found = {coefficients(prim) for prim in materials.values()}
    if enabled:
        wanted = (static_friction, dynamic_friction, friction_combine_mode)
        wrong = {
            path: coefficients(prim)
            for path, prim in materials.items()
            if not (
                math.isclose(coefficients(prim)[0], wanted[0], abs_tol=1e-6)
                and math.isclose(coefficients(prim)[1], wanted[1], abs_tol=1e-6)
                and coefficients(prim)[2] == wanted[2]
            )
        }
        if wrong:
            raise RuntimeError(f"Hand materials do not carry the selected {wanted}: {wrong}")
    elif len(found) > 1:
        raise RuntimeError(f"Hand materials carry different coefficients: {sorted(found, key=str)}")
    else:
        static_friction, dynamic_friction, friction_combine_mode = next(iter(found))

    metadata = {
        **asdict(HandContactMaterialCfg(static_friction, dynamic_friction, friction_combine_mode, enabled)),
        "source": source,
        "overrides": dict(overrides or {}),
        "material_path": next(iter(layouts)) if len(layouts) == 1 else sorted(layouts),
        "bound_collision_count": bound,
        "physics_backend": type(env.cfg.sim.physics).__name__,
    }
    env.cfg.dexbench_hand_contact_material = metadata
    print(f"[dexbench] hand contact material: {metadata}")
