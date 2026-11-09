# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Articulated runtime adapters for the shared M3 CAD assets."""

from dataclasses import replace
from functools import partial

from pxr import Gf, Usd, UsdGeom, UsdPhysics
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim.utils import clone
from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.utils.pose import Pose

from dexbench_arena.assets.center_hinge_station import spawn_single_body_driver
from dexbench_arena.assets.fastening_mechanism import (
    add_driver_trigger,
    add_cross_bit,
    add_cross_recess,
    add_screw_guide,
    rigid_body,
    revolute_joint,
)
from dexbench_arena.assets.usd_file_spawn import spawn_usd_file
from dexbench_arena.layouts.db_m3_layout import PACKAGE, TOOL_USD, TOOL_TIP_LOCAL


@clone
def spawn_motor_driver(prim_path, cfg, translation=None, orientation=None, bit_extension_m=0.12, **kwargs):
    stage = sim_utils.get_current_stage()
    root = UsdGeom.Xform.Define(stage, prim_path)
    root.AddTranslateOp().Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
    q = orientation or (0.0, 0.0, 0.0, 1.0)
    root.AddOrientOp().Set(Gf.Quatf(q[3], Gf.Vec3f(*q[:3])))
    housing = spawn_single_body_driver.__wrapped__(
        prim_path + "/Housing", cfg, bit_extension_m=bit_extension_m, **kwargs
    )
    UsdPhysics.MassAPI(housing).CreateMassAttr(0.968305)
    add_driver_trigger(root.GetPrim(), housing)
    spindle = UsdGeom.Xform.Define(stage, prim_path + "/Spindle").GetPrim()
    rigid_body(spindle, 0.03, (8e-5, 5e-6, 8e-5))
    pivot = (0.0, -0.016803, TOOL_TIP_LOCAL[2])
    UsdPhysics.MassAPI(spindle).CreateCenterOfMassAttr(Gf.Vec3f(*pivot))
    assembly = stage.GetPrimAtPath(housing.GetPath().AppendPath("Power_Drill/chuck_assembly"))
    cache = UsdGeom.XformCache()
    to_housing = cache.GetLocalToWorldTransform(housing).GetInverse()
    for mesh in list(Usd.PrimRange(assembly)):
        if not mesh.IsA(UsdGeom.Mesh):
            continue
        if mesh.GetName() in ("chuck", "bit_tip"):
            visual = stage.DefinePrim(spindle.GetPath().AppendChild(mesh.GetName()), "Mesh")
            visual.GetReferences().AddInternalReference(mesh.GetPath())
            xf = UsdGeom.Xformable(visual)
            xf.ClearXformOpOrder()
            xf.AddTransformOp(opSuffix="spindle_frame").Set(cache.GetLocalToWorldTransform(mesh) * to_housing)
            UsdGeom.Imageable(visual).CreateVisibilityAttr("inherited")
            UsdPhysics.CollisionAPI.Apply(visual).CreateCollisionEnabledAttr(False)
        if "bit" in str(mesh.GetPath()).split("chuck_assembly/", 1)[-1] and mesh.GetName() != "bit_tip":
            UsdPhysics.CollisionAPI.Apply(mesh).CreateCollisionEnabledAttr(False)
        if mesh.GetName() in ("chuck", "bit_tip", "bit"):
            UsdGeom.Imageable(mesh).CreateVisibilityAttr("invisible")
    tip = (TOOL_TIP_LOCAL[0], TOOL_TIP_LOCAL[1] - bit_extension_m, TOOL_TIP_LOCAL[2])
    shaft_back, shaft_front = -0.077803, tip[1] + 0.003
    shaft = UsdGeom.Cylinder.Define(stage, spindle.GetPath().AppendChild("Shaft"))
    shaft.CreateAxisAttr("Y")
    shaft.CreateRadiusAttr(0.002)
    shaft.CreateHeightAttr(shaft_back - shaft_front)
    shaft.AddTranslateOp().Set(Gf.Vec3d(0.0, (shaft_back + shaft_front) / 2, tip[2]))
    shaft.CreateDisplayColorAttr([(0.35, 0.37, 0.4)])
    UsdPhysics.CollisionAPI.Apply(shaft.GetPrim())
    add_cross_bit(spindle, tip)
    revolute_joint(stage, root.GetPath().AppendChild("SpindleMotor"), housing.GetPath(), spindle.GetPath(), pivot)
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    sim_utils.activate_contact_sensors(prim_path, threshold=0.0)
    return root.GetPrim()


@clone
def spawn_guided_screw(
    prim_path, cfg, translation=None, orientation=None, travel_m=0.005, turns_to_seat=4.0, resistance_nm=0.001, **kwargs
):
    stage = sim_utils.get_current_stage()
    root = UsdGeom.Xform.Define(stage, prim_path)
    root.AddTranslateOp().Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
    q = orientation or (0.0, 0.0, 0.0, 1.0)
    root.AddOrientOp().Set(Gf.Quatf(q[3], Gf.Vec3f(*q[:3])))
    raw = replace(cfg, rigid_props=None, articulation_props=None, mass_props=None, activate_contact_sensors=False)
    screw = spawn_usd_file(prim_path + "/Screw", cfg.usd_path, raw, **kwargs)
    for prim in Usd.PrimRange(screw):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
    add_cross_recess(screw)
    add_screw_guide(root.GetPrim(), travel_m=travel_m, turns_to_seat=turns_to_seat, resistance_nm=resistance_nm)
    fixture = stage.GetPrimAtPath(root.GetPath().GetParentPath().AppendChild("center_hinge_fixture"))
    rel = UsdPhysics.FilteredPairsAPI.Apply(screw).GetFilteredPairsRel()
    for prim in Usd.PrimRange(fixture):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            rel.AddTarget(prim.GetPath())
    sim_utils.activate_contact_sensors(prim_path, threshold=0.0)
    return root.GetPrim()


class MotorPowerDriver(Object):
    def __init__(self, initial_pose: Pose, bit_extension_m: float, motor_torque_nm: float, trigger_stiffness_n_m: float = 500.):
        super().__init__(
            name="power_driver",
            tags=["manufacturing", "hand-tool"],
            usd_path=str(TOOL_USD),
            object_type=ObjectType.ARTICULATION,
            initial_pose=initial_pose,
            spawn_cfg_addon={
                "func": partial(spawn_motor_driver, bit_extension_m=bit_extension_m),
                "rigid_props": sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=False),
                "activate_contact_sensors": True,
            },
        )
        self.object_cfg.actuators = {
            "trigger_return": ImplicitActuatorCfg(
                joint_names_expr=["TriggerPress"], stiffness=trigger_stiffness_n_m,
                damping=1., effort_limit_sim=10., velocity_limit_sim=.5,
            ),
        }
        if motor_torque_nm > 0:
            self.object_cfg.actuators["motor"] = ImplicitActuatorCfg(
                joint_names_expr=["SpindleMotor"],
                stiffness=0.0,
                damping=0.01,
                effort_limit_sim=motor_torque_nm,
                velocity_limit_sim=60.0,
            )


class GuidedScrew(Object):
    def __init__(self, bolt_id: str, initial_pose: Pose, turns_to_seat: float, resistance_nm: float):
        super().__init__(
            name=bolt_id,
            tags=["manufacturing", "constrained-fastener"],
            usd_path=str(PACKAGE / bolt_id.lower() / f"center_hinge_station_{bolt_id.lower()}.usd"),
            object_type=ObjectType.ARTICULATION,
            initial_pose=initial_pose,
            spawn_cfg_addon={
                "func": partial(spawn_guided_screw, turns_to_seat=turns_to_seat, resistance_nm=resistance_nm),
                "activate_contact_sensors": True,
            },
        )
