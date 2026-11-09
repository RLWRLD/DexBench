# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""PSL's conveyor (``object_library/logistics_conveyor``) as DexBench scenes stand it: scenery that blocks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dexbench_arena.assets.physics_variants import make_static

if TYPE_CHECKING:
    from pxr import Usd

LOGISTICS_CONVEYOR_PACKAGE = "logistics_conveyor"
LOGISTICS_CONVEYOR_BODIES = (
    "Belts/HB2_Belt/TransportBody",
    "Belts/FB1_Belt/TransportBody",
    "Rollers/RollerMount",
    "Rollers/Roller1",
    "Rollers/Roller2",
)
LOGISTICS_CONVEYOR_JOINTS = ("Rollers/Joint_Roller1", "Rollers/Joint_Roller2")
LOGISTICS_CONVEYOR_CONTROL = "Control"
LOGISTICS_CONVEYOR_BELT_BODIES = {
    "HB2": "Belts/HB2_Belt/TransportBody",
    "FB1": "Belts/FB1_Belt/TransportBody",
}
LOGISTICS_CONVEYOR_BELT_OVERLAPS = {
    "HB2": ("Belts/HB2_EdgeCover_F/HB2_EdgeCover_F", "Belts/HB2_EdgeCover_B/HB2_EdgeCover_B"),
    "FB1": (),
}
LOGISTICS_CONVEYOR_REPLACEMENT_COLLIDER = "DexBenchCollider"
LOGISTICS_CONVEYOR_REPLACEMENT_CLEARANCE_M = 0.002


def logistics_conveyor_frozen_bodies(moving_belts: tuple[str, ...] = ()) -> tuple[str, ...]:
    """The bodies ``freeze_logistics_conveyor`` takes the rigid body from: every one but the top runs of ``moving_belts``."""

    unknown = sorted(set(moving_belts) - set(LOGISTICS_CONVEYOR_BELT_BODIES))
    if unknown:
        raise ValueError(f"Conveyor has no belt {unknown}; its belts are {tuple(LOGISTICS_CONVEYOR_BELT_BODIES)}")
    moving = {LOGISTICS_CONVEYOR_BELT_BODIES[belt] for belt in moving_belts}
    return tuple(body for body in LOGISTICS_CONVEYOR_BODIES if body not in moving)


def freeze_logistics_conveyor(stage: Usd.Stage, asset_path: str, moving_belts: tuple[str, ...] = ()) -> None:
    """Stand the conveyor composed at ``asset_path`` still, as overs on the stage's edit target."""

    overlaps = tuple(path for belt in moving_belts for path in LOGISTICS_CONVEYOR_BELT_OVERLAPS.get(belt, ()))
    prims = {}
    for path in (*LOGISTICS_CONVEYOR_BODIES, *LOGISTICS_CONVEYOR_JOINTS, LOGISTICS_CONVEYOR_CONTROL, *overlaps):
        prim = stage.GetPrimAtPath(f"{asset_path}/{path}")
        if not prim.IsValid():
            raise RuntimeError(f"Conveyor at {asset_path} has no {path}; the {LOGISTICS_CONVEYOR_PACKAGE} package changed shape.")
        prims[path] = prim
    for path in logistics_conveyor_frozen_bodies(moving_belts):
        make_static(prims[path])
    for belt in moving_belts:
        top_run = prims[LOGISTICS_CONVEYOR_BELT_BODIES[belt]]
        for path in LOGISTICS_CONVEYOR_BELT_OVERLAPS[belt]:
            _replace_overlapping_collider(stage, prims[path], top_run)
    for path in (*LOGISTICS_CONVEYOR_JOINTS, LOGISTICS_CONVEYOR_CONTROL):
        prims[path].SetActive(False)


def _replace_overlapping_collider(stage: Usd.Stage, collider: Usd.Prim, top_run: Usd.Prim) -> None:
    """Take ``collider``'s collider away and put LOGISTICS_CONVEYOR_REPLACEMENT_COLLIDER over its bounds beside
    ``top_run``'s colliders across y, LOGISTICS_CONVEYOR_REPLACEMENT_CLEARANCE_M off them, with its physics material."""

    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.guide])
    cover = cache.ComputeWorldBound(collider).ComputeAlignedRange()
    belt = cache.ComputeWorldBound(top_run).ComputeAlignedRange()
    lo, hi = Gf.Vec3d(cover.GetMin()), Gf.Vec3d(cover.GetMax())
    if lo[1] < belt.GetMax()[1] < hi[1] and lo[1] >= belt.GetMin()[1]:
        lo[1] = belt.GetMax()[1] + LOGISTICS_CONVEYOR_REPLACEMENT_CLEARANCE_M
    elif lo[1] < belt.GetMin()[1] < hi[1] and hi[1] <= belt.GetMax()[1]:
        hi[1] = belt.GetMin()[1] - LOGISTICS_CONVEYOR_REPLACEMENT_CLEARANCE_M
    else:
        raise RuntimeError(f"{collider.GetPath()} does not lie to one side of {top_run.GetPath()} across y")

    collider.RemoveAPI(UsdPhysics.CollisionAPI)
    parent = collider.GetParent()
    box = UsdGeom.Cube.Define(stage, parent.GetPath().AppendChild(LOGISTICS_CONVEYOR_REPLACEMENT_COLLIDER))
    box.CreateSizeAttr(1.0)
    box.CreatePurposeAttr(UsdGeom.Tokens.guide)
    to_parent = UsdGeom.Xformable(parent).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse()
    turn = to_parent.ExtractRotationMatrix()
    if any(abs(turn[i][j] - (i == j)) > 1e-9 for i in range(3) for j in range(3)):
        raise RuntimeError(f"{parent.GetPath()} is turned or scaled in the world; its stand-in collider would be too")
    box.ClearXformOpOrder()
    box.AddTranslateOp().Set(to_parent.Transform((lo + hi) / 2.0))
    box.AddScaleOp().Set(Gf.Vec3f(*(hi[i] - lo[i] for i in range(3))))
    UsdPhysics.CollisionAPI.Apply(box.GetPrim())
    material = UsdShade.MaterialBindingAPI(collider).GetDirectBinding(materialPurpose="physics").GetMaterial()
    if material:
        UsdShade.MaterialBindingAPI.Apply(box.GetPrim()).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, materialPurpose="physics")
