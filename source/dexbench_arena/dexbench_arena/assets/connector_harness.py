# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Author a longer, anchored DB-M2 harness on the live USD stage before Newton imports it."""
import math

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

PANEL_COLLIDER_ROOTS = (
    "Liner/Collision/P_Case_Inner_001_Mesh_8655",
    "Liner/Collision/S_Case_Inner_001_Mesh_8655",
)


def set_harness_collision(stage, root: str, key: str, mode: str, joint_armature: float = 1e-6) -> dict:
    """Enable cable capsules against the panel and its walls, using standard USD collision groups."""
    if mode not in ("none", "panel"):
        raise ValueError("Harness collision mode must be 'none' or 'panel'")
    if not math.isfinite(joint_armature) or joint_armature < 0.0:
        raise ValueError("Cable joint armature must be finite and nonnegative")
    chain = stage.GetPrimAtPath(f"{root}/Task/Wires/{key}_0")
    if not chain:
        raise ValueError(f"Missing harness {key} under {root}")
    capsules = [p for p in Usd.PrimRange(chain) if p.IsA(UsdGeom.Capsule)]
    if not capsules:
        raise ValueError(f"Harness {key} has no cable capsules")
    if mode == "none":
        return {"mode": mode, "capsules": len(capsules)}
    panels = [stage.GetPrimAtPath(f"{root}/{path}") for path in PANEL_COLLIDER_ROOTS]
    if not all(panels):
        raise ValueError(f"Missing cable-contact panel geometry under {root}")
    cable_group = UsdPhysics.CollisionGroup.Define(stage, f"{root}/CollisionGroups/Harness_{key}")
    liner_group = UsdPhysics.CollisionGroup.Define(stage, f"{root}/CollisionGroups/HarnessLiner")
    cable_group.GetCollidersCollectionAPI().CreateIncludesRel().SetTargets([p.GetPath() for p in capsules])
    liner_group.GetCollidersCollectionAPI().CreateIncludesRel().SetTargets([p.GetPath() for p in panels])
    cable_group.CreateInvertFilteredGroupsAttr().Set(True)
    cable_group.CreateFilteredGroupsRel().SetTargets([liner_group.GetPath()])
    for prim in capsules:
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr().Set(True)
        prim.CreateAttribute("mjc:condim", Sdf.ValueTypeNames.Int, custom=False).Set(3)
    if joint_armature:
        for prim in chain.GetChildren():
            if prim.GetName().startswith("J_") and prim.IsA(UsdPhysics.Joint):
                prim.CreateAttribute("newton:armature", Sdf.ValueTypeNames.Float, custom=False).Set(joint_armature)
    return {"mode": mode, "capsules": len(capsules), "allowed_surfaces": [str(p.GetPath()) for p in panels],
            "joint_armature": joint_armature}


def harness_arc(start, end, links: int, length_m: float) -> np.ndarray:
    """The ``links + 1`` nodes of a ``length_m`` bow from ``start`` to ``end``: a half sine bulging up (world +Z,
    orthogonal to the chord), its height bisected until the polyline's length is ``length_m``. Also the shape a reset
    at another tape spot sets the cable to (``connector_harness_shape``)."""
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    axis = (end - start) / np.linalg.norm(end - start)
    bow = np.array([0., 0., 1.])
    bow -= axis * (bow @ axis)
    bow /= np.linalg.norm(bow)
    u = np.linspace(0., 1., links + 1)

    def points(amplitude):
        return start + u[:, None] * (end - start) + amplitude * np.sin(np.pi * u)[:, None] * bow

    if np.linalg.norm(end - start) >= length_m:
        raise ValueError("Cable length does not span the authored endpoints")
    low, high = 0., length_m
    for _ in range(50):
        middle = (low + high) / 2
        actual = np.linalg.norm(np.diff(points(middle), axis=0), axis=1).sum()
        if actual < length_m:
            low = middle
        else:
            high = middle
    return points((low + high) / 2)


def extend_harness(stage, root: str, length_m: float, key: str = "S2", plug_pose=None) -> dict:
    """Reshape the existing chain to ``length_m`` and connect its final point to world."""
    if not math.isfinite(length_m) or not 0.16 <= length_m <= 0.4:
        raise ValueError("Harness length must be between 0.16 and 0.4 m")
    chain = stage.GetPrimAtPath(f"{root}/Task/Wires/{key}_0")
    links = sorted((p for p in chain.GetChildren() if p.GetName().startswith("Link_")), key=lambda p: p.GetName())
    cache = UsdGeom.XformCache()
    world = [cache.GetLocalToWorldTransform(p) for p in links]
    capsules = [next(p for p in link.GetChildren() if p.IsA(UsdGeom.Capsule)) for link in links]
    old_lengths = [float(p.GetAttribute("height").Get()) for p in capsules]
    start = np.array(world[0].ExtractTranslation())
    endpoint = world[-1].Transform(Gf.Vec3d(0., 0., old_lengths[-1]))
    attachment = UsdPhysics.Joint(stage.GetPrimAtPath(f"{chain.GetPath()}/ToConnector"))
    plug_prim = stage.GetPrimAtPath(str(attachment.GetBody0Rel().GetTargets()[0]))
    if plug_pose is not None:
        xyz, xyzw = plug_pose
        plug_transform = Gf.Matrix4d(Gf.Rotation(Gf.Quatd(xyzw[3], Gf.Vec3d(*xyzw[:3]))), Gf.Vec3d(*xyz))
        parent_inverse = cache.GetLocalToWorldTransform(plug_prim.GetParent()).GetInverse()
        UsdGeom.Xformable(plug_prim).MakeMatrixXform().Set(plug_transform * parent_inverse)
        start = np.array(plug_transform.Transform(Gf.Vec3d(attachment.GetLocalPos0Attr().Get())))
        cache.Clear()
    nodes = harness_arc(start, np.array(endpoint), len(links), length_m)
    lengths = np.linalg.norm(np.diff(nodes, axis=0), axis=1)
    poses = []
    for i, (link, capsule, length) in enumerate(zip(links, capsules, lengths)):
        rotation = Gf.Rotation(Gf.Vec3d(0., 0., 1.), Gf.Vec3d(*(nodes[i + 1] - nodes[i])).GetNormalized())
        transform = Gf.Matrix4d(rotation, Gf.Vec3d(*nodes[i]))
        parent_inverse = cache.GetLocalToWorldTransform(link.GetParent()).GetInverse()
        UsdGeom.Xformable(link).MakeMatrixXform().Set(transform * parent_inverse)
        poses.append(transform)
        radius = float(capsule.GetAttribute("radius").Get())
        capsule.GetAttribute("height").Set(float(length))
        capsule.GetAttribute("xformOp:translate").Set(Gf.Vec3d(0., 0., length / 2))
        capsule.GetAttribute("extent").Set(UsdGeom.Boundable.ComputeExtentFromPlugins(
            UsdGeom.Capsule(capsule), Usd.TimeCode.Default()))
        mass = float(link.GetAttribute("physics:mass").Get()) * length / old_lengths[i]
        link.GetAttribute("physics:mass").Set(float(mass))
        link.GetAttribute("physics:centerOfMass").Set(Gf.Vec3f(0., 0., length / 2))
        transverse = mass * (3 * radius**2 + length**2) / 12
        link.GetAttribute("physics:diagonalInertia").Set(Gf.Vec3f(transverse, transverse, mass * radius**2 / 2))
    for i in range(len(links) - 1):
        joint = UsdPhysics.Joint(stage.GetPrimAtPath(f"{chain.GetPath()}/J_{i:02d}"))
        joint.GetLocalPos0Attr().Set(Gf.Vec3f(0., 0., lengths[i]))
        joint.GetLocalPos1Attr().Set(Gf.Vec3f(0.))
        joint.GetLocalRot0Attr().Set(Gf.Quatf(1.))
        joint.GetLocalRot1Attr().Set(Gf.Quatf((poses[i] * poses[i + 1].GetInverse()).ExtractRotationQuat()))
    plug = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(str(attachment.GetBody0Rel().GetTargets()[0])))
    attachment.GetLocalRot0Attr().Set(Gf.Quatf((poses[0] * plug.GetInverse()).ExtractRotationQuat()))
    joint = UsdPhysics.Joint.Define(stage, str(chain.GetPath()) + "/AtTape")
    joint.GetPrim().AddAppliedSchema("MjcEqualityConnectAPI")
    joint.CreateBody0Rel().SetTargets([])
    joint.CreateBody1Rel().SetTargets([links[-1].GetPath()])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*endpoint))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0., 0., lengths[-1]))
    joint.CreateExcludeFromArticulationAttr().Set(True)
    return {"length_m": float(lengths.sum()), "target": list(endpoint),
            "tail_length": float(lengths[-1]), "tail_name": links[-1].GetName()}
