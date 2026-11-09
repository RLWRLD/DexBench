# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""M3 fixtures, inactive CAD bolts and the housing geometry for the articulated driver."""

from dataclasses import replace
from functools import partial
import math

import numpy as np
import trimesh
from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade
import isaaclab.sim as sim_utils
from isaaclab.sim.utils import clone
from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.assets.object_library import LibraryObject
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.utils.usd.helpers import extract_trimesh_from_prim

from dexbench_arena.assets.object_library import KinematicObjectMixin
from dexbench_arena.assets.physics_variants import make_static
from dexbench_arena.assets.usd_file_spawn import spawn_usd_file
from dexbench_arena.layouts.db_m3_layout import PACKAGE, SEATED_HEADS, REFRIGERATOR_XYZ

FREEZER_DOORS = {"Door_Assembly_Freezer_Left_004": (-.445, -1), "Door_Assembly_Freezer_Right_004": (.445, 1)}
"""Each freezer door assembly, with the x of its hinge pivot and the sense its opening turns."""
ACCESS_CHANNEL_PARTS = (
    "Case_Foam_Assembly_003/Plate_Assembly_Center_003/Plate_Center_003",
    "Door_Set_Assembly_005/Hinge_Assembly_Center_003/Supporter_Hinge_003",
    "Door_Set_Assembly_005/Hinge_Assembly_Center_003_2/Supporter_Hinge_003",
)
"""The centre plate and the two outer hinge supporters in front of B3/B6, as the end of their prim paths."""


@clone
def spawn_accessible_fixture(prim_path, cfg, translation=None, orientation=None,
                             access_channels=True, explicit_visual_meshes=False, door_opening_deg=100.0, **kwargs):
    """Provide collision-only tool access to CAD screws hidden behind solid plates."""
    prim = spawn_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation, **kwargs)
    stage = prim.GetStage()
    found = set()
    for child in Usd.PrimRange(prim):
        for part, (x, sign) in FREEZER_DOORS.items():
            if child.GetName() == part:
                if part in found:
                    raise ValueError(f"Ambiguous M3 door assembly: {part}")
                found.add(part)
                xf = UsdGeom.Xformable(child)
                xf.AddTranslateOp(opSuffix="m3_hinge").Set(Gf.Vec3d(x, -.085, 0.))
                xf.AddRotateZOp(opSuffix="m3_opening").Set(sign*(door_opening_deg-100.))
                xf.AddTranslateOp(opSuffix="m3_hinge", isInverseOp=True)
    if found != set(FREEZER_DOORS):
        raise ValueError("M3 fixture is missing a freezer door assembly")
    channels_enabled = access_channels and not (
        cfg.collision_props is not None and cfg.collision_props.collision_enabled is False)
    for child in list(Usd.PrimRange(prim)) if channels_enabled else ():
        if not child.IsA(UsdGeom.Mesh) or not any(f"/{part}/" in f"{child.GetPath()}/" for part in ACCESS_CHANNEL_PARTS):
            continue
        surface = extract_trimesh_from_prim(stage, str(child.GetPath()))
        for head in (SEATED_HEADS[2], SEATED_HEADS[5]):
            center = np.array(head) - REFRIGERATOR_XYZ
            low, high = center - (.006, .03, .006), center + (.006, .015, .006)
            if np.any(surface.bounds[1] < low) or np.any(surface.bounds[0] > high):
                continue
            inside, outside = surface, []
            for axis in range(3):
                for bound, sign in ((low, 1), (high, -1)):
                    normal = np.eye(3)[axis] * sign
                    outside.append(trimesh.intersections.slice_mesh_plane(inside, -normal, bound))
                    inside = trimesh.intersections.slice_mesh_plane(inside, normal, bound)
            surface = trimesh.util.concatenate(outside)
        UsdPhysics.CollisionAPI(child).GetCollisionEnabledAttr().Set(False)
        proxy = UsdGeom.Mesh.Define(stage, child.GetPath().GetParentPath().AppendChild(child.GetName() + "_access_collision"))
        proxy.CreatePointsAttr(surface.vertices.tolist())
        proxy.CreateFaceVertexCountsAttr([3] * len(surface.faces))
        proxy.CreateFaceVertexIndicesAttr(surface.faces.flatten().tolist())
        proxy.CreateSubdivisionSchemeAttr("none")
        proxy.CreatePurposeAttr("guide")
        proxy.CreateVisibilityAttr("invisible")
        UsdPhysics.CollisionAPI.Apply(proxy.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(proxy.GetPrim()).CreateApproximationAttr("none")
        material = UsdShade.MaterialBindingAPI(child).ComputeBoundMaterial(materialPurpose="physics")[0]
        if material:
            UsdShade.MaterialBindingAPI.Apply(proxy.GetPrim()).Bind(material, materialPurpose="physics")
    if explicit_visual_meshes:
        for child in list(Usd.PrimRange(prim)):
            if not child.IsA(UsdGeom.Mesh) or not child.HasAPI(UsdPhysics.CollisionAPI):
                continue
            imageable = UsdGeom.Imageable(child)
            if (imageable.ComputeVisibility() == "invisible"
                    or imageable.ComputePurpose() not in ("default", "proxy")):
                continue
            enabled = UsdPhysics.CollisionAPI(child).GetCollisionEnabledAttr().Get()
            if enabled is False:
                continue
            path = child.GetPath().GetParentPath().AppendChild(child.GetName() + "_render")
            visual = stage.DefinePrim(path, "Mesh")
            visual.GetReferences().AddInternalReference(child.GetPath())
            visual.RemoveAPI(UsdPhysics.CollisionAPI)
            visual.RemoveAPI(UsdPhysics.MeshCollisionAPI)
    return prim


def extend_driver_bit(prim, extension_m):
    """Extend the spawned bit and selected backend's collision pieces together."""
    bit = UsdGeom.Mesh(prim.GetStage().GetPrimAtPath(prim.GetPath().AppendPath("Power_Drill/chuck_assembly/bit")))
    points = bit.GetPointsAttr().Get()
    front, back = min(p[1] for p in points), max(p[1] for p in points)
    cache = UsdGeom.XformCache()
    bit_world = cache.GetLocalToWorldTransform(bit.GetPrim())
    meshes = [bit]
    assembly = bit.GetPrim().GetParent()
    for child in assembly.GetChildren():
        if child.GetName() == "bit_newton_collision":
            meshes.extend(UsdGeom.Mesh(p) for p in Usd.PrimRange(child) if p.IsA(UsdGeom.Mesh))
    for mesh in meshes:
        world = cache.GetLocalToWorldTransform(mesh.GetPrim())
        to_bit, from_bit = world*bit_world.GetInverse(), bit_world*world.GetInverse()
        result = []
        for point in mesh.GetPointsAttr().Get():
            local = to_bit.Transform(Gf.Vec3d(point))
            weight = max(0., min(1., (back-local[1])/(back-front)))
            local[1] -= extension_m*weight
            result.append(Gf.Vec3f(from_bit.Transform(local)))
        mesh.GetPointsAttr().Set(result)
        mesh.GetExtentAttr().Set(UsdGeom.PointBased.ComputeExtent(result))


@clone
def spawn_single_body_driver(prim_path, cfg, translation=None, orientation=None, bit_extension_m=0.08, **kwargs):
    """Reuse the shared tool geometry/materials and rigidly lock its moving parts."""
    raw = replace(cfg, rigid_props=None, articulation_props=None, mass_props=None, activate_contact_sensors=False)
    prim = spawn_usd_file(prim_path, cfg.usd_path, raw, translation, orientation, **kwargs)
    extend_driver_bit(prim, bit_extension_m)
    for child in list(Usd.PrimRange(prim)):
        if child.IsA(UsdPhysics.Joint):
            child.SetActive(False)
        make_static(child)
        if child.HasAPI(UsdPhysics.ArticulationRootAPI):
            child.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(1.006853)
    sim_utils.modify_rigid_body_properties(prim_path, cfg.rigid_props)
    sim_utils.activate_contact_sensors(prim_path, threshold=0.0)
    return prim


@register_asset
class CenterHingeFixture(LibraryObject):
    name = "center_hinge_fixture"
    tags = ["manufacturing", "fixture"]
    object_type = ObjectType.BASE
    usd_path = str(PACKAGE / "usd/center_hinge_station.usd")
    spawn_cfg_addon = {"func": spawn_accessible_fixture}

    def __init__(self, initial_pose: Pose, access_channels: bool = False, door_opening_deg: float = 140.0):
        if not math.isfinite(door_opening_deg) or not 100 <= door_opening_deg <= 170:
            raise ValueError("door_opening_deg must be in [100, 170]")
        super().__init__(initial_pose=initial_pose)
        self.object_cfg.spawn.func = partial(spawn_accessible_fixture, access_channels=access_channels,
                                             door_opening_deg=door_opening_deg)


class CenterHingeBolt(KinematicObjectMixin, Object):
    def __init__(self, bolt_id: str, initial_pose: Pose):
        super().__init__(
            name=bolt_id, tags=["manufacturing", "constrained_bolt"],
            usd_path=str(PACKAGE / bolt_id.lower() / f"center_hinge_station_{bolt_id.lower()}.usd"),
            object_type=ObjectType.RIGID, initial_pose=initial_pose,
            spawn_cfg_addon={
                "rigid_props": sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                "activate_contact_sensors": True,
                "collision_props": sim_utils.CollisionPropertiesCfg(contact_offset=0.0002, rest_offset=0.0),
            },
        )
