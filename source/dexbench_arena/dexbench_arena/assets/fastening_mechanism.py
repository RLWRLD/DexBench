# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Runtime-only joint and contact geometry for the M3 torque mechanism."""

import math

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics


def collision_box(stage, path, size, position, visible=False):
    """Author one box of the compound drive interface (metres, Y axis)."""
    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    box.AddTranslateOp().Set(Gf.Vec3d(*position))
    box.AddScaleOp().Set(Gf.Vec3f(*size))
    box.CreateDisplayColorAttr([(0.35, 0.37, 0.40)])
    if not visible:
        box.CreatePurposeAttr("guide")
        box.CreateVisibilityAttr("invisible")
    UsdPhysics.CollisionAPI.Apply(box.GetPrim())
    return box.GetPrim()


def rigid_body(prim, mass, inertia):
    body = UsdPhysics.RigidBodyAPI.Apply(prim)
    body.CreateRigidBodyEnabledAttr(True)
    body.CreateKinematicEnabledAttr(False)
    props = UsdPhysics.MassAPI.Apply(prim)
    props.CreateMassAttr(mass)
    props.CreateCenterOfMassAttr(Gf.Vec3f(0))
    props.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
    props.CreatePrincipalAxesAttr(Gf.Quatf(1))


def revolute_joint(stage, path, parent, child, pivot=(0.0, 0.0, 0.0)):
    joint = UsdPhysics.RevoluteJoint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([parent])
    joint.CreateBody1Rel().SetTargets([child])
    joint.CreateAxisAttr("Y")
    joint.CreateLocalPos0Attr(Gf.Vec3f(*pivot))
    joint.CreateLocalPos1Attr(Gf.Vec3f(*pivot))
    return joint


def add_screw_guide(root, *, travel_m, turns_to_seat, resistance_nm, locked=False):
    """Ground -> slider -> screw with solver-enforced pitch, no screw actuator."""
    stage, path = root.GetStage(), root.GetPath()
    anchor = UsdGeom.Xform.Define(stage, path.AppendChild("Anchor")).GetPrim()
    carrier = UsdGeom.Xform.Define(stage, path.AppendChild("Carrier")).GetPrim()
    screw = stage.GetPrimAtPath(path.AppendChild("Screw"))
    rigid_body(anchor, 0.01, (1e-5,) * 3)
    rigid_body(carrier, 0.001, (1e-7,) * 3)
    rigid_body(screw, 0.008, (2e-7, 1e-7, 2e-7))
    fixed = UsdPhysics.FixedJoint.Define(stage, path.AppendChild("WorldMount"))
    fixed.CreateBody1Rel().SetTargets([anchor.GetPath()])
    world = UsdGeom.XformCache().GetLocalToWorldTransform(root)
    fixed.CreateLocalPos0Attr(Gf.Vec3f(world.ExtractTranslation()))
    fixed.CreateLocalRot0Attr(Gf.Quatf(world.ExtractRotationQuat()))
    UsdPhysics.ArticulationRootAPI.Apply(root)
    slide = UsdPhysics.PrismaticJoint.Define(stage, path.AppendChild("ThreadTravel"))
    slide.CreateBody0Rel().SetTargets([anchor.GetPath()])
    slide.CreateBody1Rel().SetTargets([carrier.GetPath()])
    slide.CreateAxisAttr("Y")
    slide.CreateLowerLimitAttr(-0.001)
    slide.CreateUpperLimitAttr(travel_m)
    turn = revolute_joint(stage, path.AppendChild("ScrewRotation"), carrier.GetPath(), screw.GetPath())
    turn.CreateLowerLimitAttr(-360.0)
    turn.CreateUpperLimitAttr(360.0 * turns_to_seat)
    turn_prim = turn.GetPrim()
    turn_prim.AddAppliedSchema("NewtonMimicAPI")
    turn_prim.CreateRelationship("newton:mimicJoint").SetTargets([slide.GetPath()])
    turn_prim.CreateAttribute("newton:mimicCoef0", Sdf.ValueTypeNames.Float).Set(0.0)
    turn_prim.CreateAttribute("newton:mimicCoef1", Sdf.ValueTypeNames.Float).Set(2 * math.pi * turns_to_seat / travel_m)
    turn_prim.CreateAttribute("newton:friction", Sdf.ValueTypeNames.Float).Set(resistance_nm)
    turn_prim.CreateAttribute("mjc:damping", Sdf.ValueTypeNames.Float).Set(0.00002)
    solver_reference = [0.0025, 1.0]
    solver_impedance = [0.9999, 0.9999, 0.0001, 0.5, 2.0]
    turn_prim.AddAppliedSchema("MjcEqualityJointAPI")
    turn_prim.CreateAttribute("mjc:solref", Sdf.ValueTypeNames.DoubleArray).Set(solver_reference)
    turn_prim.CreateAttribute("mjc:solimp", Sdf.ValueTypeNames.DoubleArray).Set(solver_impedance)
    for joint in (turn, slide):
        for name in ("solreflimit", "solreffriction"):
            joint.GetPrim().CreateAttribute("mjc:" + name, Sdf.ValueTypeNames.DoubleArray).Set(solver_reference)
        for name in ("solimplimit", "solimpfriction"):
            joint.GetPrim().CreateAttribute("mjc:" + name, Sdf.ValueTypeNames.DoubleArray).Set(solver_impedance)
    if locked:
        turn.CreateLowerLimitAttr(-0.00001)
        turn.CreateUpperLimitAttr(0.00001)
    UsdPhysics.FilteredPairsAPI.Apply(screw).CreateFilteredPairsRel().SetTargets([anchor.GetPath(), carrier.GetPath()])
    return turn, slide


def add_cross_recess(screw, clearance_m=0.0003):
    """Four head quadrants leave a cross slot; floor carries axial feed load."""
    stage, path = screw.GetStage(), screw.GetPath()
    half_gap, outer = 0.00065 + clearance_m, 0.0055
    width = outer - half_gap
    for x in (-1, 1):
        for z in (-1, 1):
            collision_box(
                stage,
                path.AppendChild(f"DriveWall_{x+1}_{z+1}"),
                (width, 0.003, width),
                (x * (outer + half_gap) / 2, 0.0015, z * (outer + half_gap) / 2),
            )
    collision_box(stage, path.AppendChild("DriveFloor"), (0.011, 0.001, 0.011), (0.0, 0.0035, 0.0))


def add_cross_bit(spindle, tip_xyz, *, contact_enabled=True):
    """Matching bit extends 3 mm into the head at the nominal front-face site."""
    x, y, z = tip_xyz
    for name, size in (("CrossX", (0.005, 0.003, 0.0013)), ("CrossZ", (0.0013, 0.003, 0.005))):
        prim = collision_box(spindle.GetStage(), spindle.GetPath().AppendChild(name), size, (x, y + 0.0015, z), True)
        UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(contact_enabled)


def add_driver_trigger(root, housing):
    """Add the shared CAD trigger with its original +Y prismatic travel."""
    stage = root.GetStage()
    source = stage.GetPrimAtPath(housing.GetPath().AppendPath("Power_Drill/trigger/trigger"))
    cache = UsdGeom.XformCache()
    transform = cache.GetLocalToWorldTransform(source) * cache.GetLocalToWorldTransform(housing).GetInverse()
    trigger = UsdGeom.Xform.Define(stage, root.GetPath().AppendChild("Trigger")).GetPrim()
    rigid_body(trigger, .008548, (1e-6, 1e-6, 1e-6))
    UsdPhysics.MassAPI(trigger).CreateCenterOfMassAttr(Gf.Vec3f(0., .0353, .1344))
    visual = stage.DefinePrim(trigger.GetPath().AppendChild("Visual"), "Mesh")
    visual.GetReferences().AddInternalReference(source.GetPath())
    xf = UsdGeom.Xformable(visual)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(transform)
    UsdGeom.Imageable(visual).CreateVisibilityAttr("inherited")
    UsdPhysics.CollisionAPI.Apply(visual).CreateCollisionEnabledAttr(False)
    collision_box(stage, trigger.GetPath().AppendChild("Contact"), (.0232, .0200, .0253),
                  (0., .0353, .1344))
    for prim in Usd.PrimRange(source.GetParent()):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
    UsdGeom.Imageable(source).CreateVisibilityAttr("invisible")
    joint = UsdPhysics.PrismaticJoint.Define(stage, root.GetPath().AppendChild("TriggerPress"))
    joint.CreateBody0Rel().SetTargets([housing.GetPath()])
    joint.CreateBody1Rel().SetTargets([trigger.GetPath()])
    joint.CreateAxisAttr("Y")
    joint.CreateLowerLimitAttr(0.)
    joint.CreateUpperLimitAttr(.006)
    joint.GetPrim().CreateAttribute("mjc:solreflimit", Sdf.ValueTypeNames.DoubleArray).Set([.002, 1.])
    joint.CreateCollisionEnabledAttr(False)
    return joint
