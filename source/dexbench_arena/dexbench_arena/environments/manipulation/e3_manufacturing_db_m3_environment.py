# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""DB-M3 staging for the shared E3 environment: free-driver pickup and native fastening."""

import argparse
import math
from dataclasses import dataclass, field
from functools import partial

from dexbench_arena.layouts.db_m3_layout import (
    BOLT_NAMES, E3_M3_CAD_MIN_Z_M, E3_M3_CONVEYOR_LENGTH_M, M3_E3_OBJECT_AXIS, M3_OBJECT_AXIS,
    REFRIGERATOR_XYZ, station_seated_heads,
)
from dexbench_arena.cli import resolve_sim_rate
from dexbench_arena.embodiments.lookup import embodiment_class, start_poses
from dexbench_arena.environments.manipulation.scene_environment_base import (
    TASKS,
)


POWER_DRIVER_TASK_NAME = "power_driver_fastening"
M3_ANCHOR = "A_m3"
_M3_ONLY = {TASKS: (POWER_DRIVER_TASK_NAME,)}
M3_NEWTON_SOLVER_OPTIONS = {"num_substeps": 4, "enable_multiccd": True}


@dataclass
class M3StationCfg:
    task: str = "none"
    m3_scene_layout: str = field(default="e3_assembly_line", metadata=_M3_ONLY)
    """Published isolated station, or ``e3_assembly_line`` for the E3 room."""
    m3_turns_to_seat: float = field(default=4.0, metadata=_M3_ONLY)
    m3_motor_rps: float = field(default=4.0, metadata=_M3_ONLY)
    m3_motor_enabled: bool = field(default=True, metadata=_M3_ONLY)
    m3_fastening_mode: str = field(default="contact_torque", metadata=_M3_ONLY)
    m3_trigger_press_m: float = field(default=.002, metadata=_M3_ONLY)
    m3_trigger_stiffness_n_m: float = field(default=500., metadata=_M3_ONLY)
    m3_motor_torque_nm: float = field(default=.03, metadata=_M3_ONLY)
    m3_thread_resistance_nm: float = field(default=.006, metadata=_M3_ONLY)
    m3_motor_direction: float = field(default=1., metadata=_M3_ONLY)
    m3_lateral_tolerance_m: float = field(default=0.002, metadata=_M3_ONLY)
    m3_axial_tolerance_m: float = field(default=0.004, metadata=_M3_ONLY)
    m3_alignment_tolerance_deg: float = field(default=10.0, metadata=_M3_ONLY)
    m3_contact_threshold_n: float = field(default=0.2, metadata=_M3_ONLY)
    m3_bit_extension_m: float = field(default=.02, metadata=_M3_ONLY)
    m3_fixture: str = field(default="center_hinge_fixture", metadata=_M3_ONLY)
    m3_mount_driver: bool = field(default=False, metadata=_M3_ONLY)
    m3_hand_grasp: bool = field(default=True, metadata=_M3_ONLY)
    m3_pickup_position_xyz: tuple[float, float, float] = field(default=(.3266, -.94, .82504999645), metadata=_M3_ONLY)
    m3_pickup_holder: bool = field(default=False, metadata=_M3_ONLY)
    m3_station_offset_y_m: float = field(default=-.45, metadata=_M3_ONLY)
    m3_station_offset_x_m: float = field(default=.0766, metadata=_M3_ONLY)
    m3_e3_conveyor_height_m: float = field(default=0.12, metadata=_M3_ONLY)
    """E3 conveyor top above the floor; cabinet bottom rests directly on it."""

    m3_pickup_pocket_height_m: float = field(default=0.03, metadata=_M3_ONLY)
    m3_workbench_rear_offset_m: float = field(default=.015, metadata=_M3_ONLY)
    m3_workbench_size_xy: tuple[float, float] = field(default=(0.9, 0.55), metadata=_M3_ONLY)
    m3_physics_substeps: int = field(default=4, metadata=_M3_ONLY)
    m3_newton_contact_stiffness: float = field(default=62500.0, metadata=_M3_ONLY)
    m3_newton_contact_damping: float = field(default=500.0, metadata=_M3_ONLY)
    m3_newton_contact_dimension: int = field(default=4, metadata=_M3_ONLY)
    m3_fastener_access_channels: bool = field(default=False, metadata=_M3_ONLY)
    m3_door_opening_deg: float = field(default=160., metadata=_M3_ONLY)

    def __post_init__(self):
        for name, expected in (("m3_fastening_mode", "contact_torque"),
                               ("m3_mount_driver", False), ("m3_hand_grasp", True),
                               ("m3_pickup_holder", False), ("m3_pickup_pocket_height_m", .03)):
            value = getattr(self, name)
            if value != expected or (isinstance(expected, bool) and type(value) is not bool):
                raise ValueError(f"{name} only supports {expected!r}; experimental modes were removed")

    @property
    def station_offset_z_m(self):
        return (self.m3_e3_conveyor_height_m - E3_M3_CAD_MIN_Z_M - REFRIGERATOR_XYZ[2]
                if self.m3_scene_layout == "e3_assembly_line" else 0.0)


def validate_m3_cfg(cfg):
    if cfg.m3_scene_layout not in ("station", "e3_assembly_line"):
        raise ValueError("m3_scene_layout must be station or e3_assembly_line")
    M3StationCfg.__post_init__(cfg)
    for name in ("m3_station_offset_x_m", "m3_station_offset_y_m"):
        if not math.isfinite(getattr(cfg, name)):
            raise ValueError(f"{name} must be finite")
    if not math.isfinite(cfg.m3_e3_conveyor_height_m) or not .06 <= cfg.m3_e3_conveyor_height_m <= .25:
        raise ValueError("m3_e3_conveyor_height_m must be in [.06, .25]")
    for name in ("m3_motor_torque_nm", "m3_thread_resistance_nm"):
        if not math.isfinite(getattr(cfg, name)) or getattr(cfg, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(cfg.m3_trigger_press_m) or not 0 < cfg.m3_trigger_press_m <= .006:
        raise ValueError("m3_trigger_press_m must be in (0, .006]")
    if not math.isfinite(cfg.m3_trigger_stiffness_n_m) or cfg.m3_trigger_stiffness_n_m <= 0:
        raise ValueError("m3_trigger_stiffness_n_m must be finite and positive")
    if cfg.m3_motor_direction not in (-1., 1.):
        raise ValueError("m3_motor_direction must be -1 or 1")
    if not math.isfinite(cfg.m3_door_opening_deg) or not 100 <= cfg.m3_door_opening_deg <= 170:
        raise ValueError("m3_door_opening_deg must be in [100, 170]")
    if (len(cfg.m3_pickup_position_xyz) != 3
            or not all(math.isfinite(v) for v in cfg.m3_pickup_position_xyz)
            or cfg.m3_pickup_position_xyz[2] <= 0):
        raise ValueError("m3_pickup_position_xyz requires three finite coordinates and positive height")
    if (len(cfg.m3_workbench_size_xy) != 2
            or any(not math.isfinite(v) or v < minimum
                   for v, minimum in zip(cfg.m3_workbench_size_xy, (.2, .25)))):
        raise ValueError("m3_workbench_size_xy must be at least 0.20 x 0.25 m to support the pickup fixture")
    if not math.isfinite(cfg.m3_workbench_rear_offset_m):
        raise ValueError("m3_workbench_rear_offset_m must be finite")
    if cfg.m3_hand_grasp and cfg.m3_pickup_position_xyz[2] <= .04:
        raise ValueError("m3_hand_grasp requires pickup height above the 0.04 m fixture")
    if cfg.task == POWER_DRIVER_TASK_NAME and cfg.m3_hand_grasp and getattr(embodiment_class(cfg.embodiment), "power_driver_grasp", None) is None:
        raise ValueError("m3_hand_grasp requires an embodiment declaring power_driver_grasp")
    if type(cfg.m3_physics_substeps) is not int or cfg.m3_physics_substeps < 1:
        raise ValueError("m3_physics_substeps must be a positive integer")
    for name in ("m3_newton_contact_stiffness", "m3_newton_contact_damping"):
        if not math.isfinite(getattr(cfg, name)) or getattr(cfg, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if type(cfg.m3_newton_contact_dimension) is not int or cfg.m3_newton_contact_dimension not in (3, 4, 6):
        raise ValueError("m3_newton_contact_dimension must be 3, 4 or 6")
    if not math.isfinite(cfg.base_jitter_radius) or not 0 <= cfg.base_jitter_radius <= .01:
        raise ValueError("M3 base_jitter_radius must be finite and in [0, .01] m")


def configure_m3_environment(environment, cfg):
    if cfg.task == POWER_DRIVER_TASK_NAME:
        from dexbench_arena.variations.db_m3_station_position_variation import M3StationPositionVariation
        from dexbench_arena.variations.object_pose_variation import (
            ObjectPoseVariation, ObjectPoseVariationCfg, PlanarPoseDomain,
        )
        from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg

        if cfg.base_jitter_radius:
            environment.embodiment.get_variation("base_pose").domain = PlanarPoseDomain(
                max_abs_xy_m=(.01, .01), reason="M3 pickup and low-conveyor approach corridor")
        assets = environment.scene.assets
        station = [asset for name, asset in assets.items()
                   if name in ("center_hinge_fixture", "Belt", *BOLT_NAMES)
                   or name.startswith(("Pallet_", "Frame_", "Guide_", "Leg_"))]
        assets["center_hinge_fixture"].add_variation(M3StationPositionVariation(station, environment.task))
        assets["power_driver"].add_variation(ObjectPoseVariation(
            "power_driver", name="pickup_position",
            cfg=ObjectPoseVariationCfg(sampler_cfg=UniformSamplerCfg(
                low=[-.01, -.01, 0.], high=[.01, .01, 0.])),
            domain=PlanarPoseDomain(max_abs_xy_m=(.01, .01),
                                   reason="Keep the battery on the bare pickup workbench"),
        ))
    previous = environment.env_cfg_callback

    def configure_physics(env_cfg):
        env_cfg = previous(env_cfg) if previous else env_cfg
        from isaaclab_newton.physics.newton_manager_cfg import NewtonCfg
        if not isinstance(env_cfg.sim.physics, NewtonCfg):
            raise ValueError("M3 contact_torque requires the newton preset and pinned Newton runtime")
        if environment.embodiment is not None:
            environment.embodiment.configure_power_driver_actuators(env_cfg.scene.robot)
            env_cfg.power_driver_grasp = environment.embodiment.power_driver_grasp
            env_cfg.power_driver_trigger_body_expr = environment.embodiment.power_driver_trigger_body_expr
        env_cfg.sim.dt /= cfg.m3_physics_substeps
        env_cfg.decimation *= cfg.m3_physics_substeps
        env_cfg.sim.render_interval = env_cfg.decimation
        if environment.embodiment is not None:
            from isaaclab.sensors import ContactSensorCfg
            body_expr = "|".join(
                embodiment_class(cfg.embodiment).held_object_contact_body_exprs.values()
            )
            env_cfg.scene.driver_grip_contact = ContactSensorCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/({body_expr})",
                filter_prim_paths_expr=["{ENV_REGEX_NS}/power_driver/(Housing|Trigger)"],
                history_length=env_cfg.decimation, update_period=0.0,
                track_contact_points=True,
            )
        return env_cfg

    environment.env_cfg_callback = configure_physics
    return environment


def apply_m3_newton_settings(env_cfg, cfg):
    """Select rigid grasp contacts through upstream Newton config fragments."""
    env_cfg.scene.center_hinge_fixture.spawn.func = partial(
        env_cfg.scene.center_hinge_fixture.spawn.func, explicit_visual_meshes=True)
    from isaaclab_newton.sim.schemas import MujocoCollisionCfg
    from isaaclab_physx.sim.schemas import PhysxCollisionCfg

    env_cfg.sim.physics.default_shape_cfg.ke = cfg.m3_newton_contact_stiffness
    env_cfg.sim.physics.default_shape_cfg.kd = cfg.m3_newton_contact_damping
    env_cfg.scene.power_driver.spawn.collision_props = {
        "/.*": [
            PhysxCollisionCfg(contact_offset=.0002, rest_offset=0.),
            MujocoCollisionCfg(condim=cfg.m3_newton_contact_dimension),
        ],
    }


def make_power_driver_task(environment, cfg):
    from dexbench_arena.tasks.power_driver_fastening import DexBenchPowerDriverFasteningTask
    from dexbench_arena.tasks.power_driver_model import FasteningCfg

    return DexBenchPowerDriverFasteningTask(
        FasteningCfg(
            turns_to_seat=cfg.m3_turns_to_seat, motor_rps=cfg.m3_motor_rps,
            lateral_tolerance_m=cfg.m3_lateral_tolerance_m, axial_tolerance_m=cfg.m3_axial_tolerance_m,
            alignment_tolerance_deg=cfg.m3_alignment_tolerance_deg, contact_threshold_n=cfg.m3_contact_threshold_n,
        ),
        motor_enabled=cfg.m3_motor_enabled,
        control_substeps=resolve_sim_rate(cfg)[1] * cfg.m3_physics_substeps,
        bit_extension_m=cfg.m3_bit_extension_m,
        fastening_mode=cfg.m3_fastening_mode, motor_direction=cfg.m3_motor_direction,
        seated_heads=station_seated_heads(cfg.m3_station_offset_y_m, cfg.m3_station_offset_x_m, cfg.station_offset_z_m),
        trigger_press_m=cfg.m3_trigger_press_m,
        trigger_body_expr=(embodiment_class(cfg.embodiment).power_driver_trigger_body_expr
                           if cfg.embodiment else None),
        robot_joint_pos_overrides=(start_poses(cfg.embodiment)["power_driver_low_clear"]
                                   if cfg.embodiment and cfg.m3_hand_grasp else None),
    )


def build_m3_scene(environment, cfg):
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose
    from dexbench_arena.assets.center_hinge_station import (
        CenterHingeBolt, CenterHingeFixture,
    )
    from dexbench_arena.assets.object_library import AssemblyWorkbench, DexBenchProceduralTable, DexBenchTubeLight
    from dexbench_arena.tasks.power_driver_model import FasteningCfg
    from dexbench_arena.assets.power_driver_torque import MotorPowerDriver, GuidedScrew

    if cfg.task not in ("none", POWER_DRIVER_TASK_NAME):
        raise ValueError(f"Unknown M3 task: {cfg.task}")
    if cfg.mimic:
        raise ValueError("M3 does not define Mimic subtask boundaries yet")
    environment.object_axis_for(cfg)
    M3_OBJECT_AXIS.resolve(cfg)
    in_e3 = cfg.m3_scene_layout == "e3_assembly_line"
    station_y = cfg.m3_station_offset_y_m
    refrigerator_xyz = (REFRIGERATOR_XYZ[0] + cfg.m3_station_offset_x_m,
                        REFRIGERATOR_XYZ[1] + station_y, REFRIGERATOR_XYZ[2] + cfg.station_offset_z_m)
    assets = [CenterHingeFixture(initial_pose=Pose(position_xyz=refrigerator_xyz),
                                access_channels=cfg.m3_fastener_access_channels,
                                door_opening_deg=cfg.m3_door_opening_deg)]
    if in_e3:
        from dexbench_arena.assets.background_library import DexBenchAssemblyLineE3M3Background

        assets.append(DexBenchAssemblyLineE3M3Background())
    fastening = FasteningCfg()
    for index, (name, head) in enumerate(zip(BOLT_NAMES, station_seated_heads(station_y, cfg.m3_station_offset_x_m, cfg.station_offset_z_m))):
        travel = fastening.travel_m if index in fastening.active_bolt_indices else 0.0
        pose = Pose(position_xyz=(head[0], head[1]-travel, head[2]))
        assets.append(GuidedScrew(name, pose, cfg.m3_turns_to_seat, cfg.m3_thread_resistance_nm)
                      if index in fastening.active_bolt_indices
                      else CenterHingeBolt(name, pose))
    tool_position = tuple(cfg.m3_pickup_position_xyz)
    tool_rotation = (0.0, 0.0, 1.0, 0.0)
    tool_pose = Pose(position_xyz=tool_position, rotation_xyzw=tool_rotation)
    assets.append(MotorPowerDriver(tool_pose, cfg.m3_bit_extension_m, cfg.m3_motor_torque_nm if cfg.m3_motor_enabled else 0., cfg.m3_trigger_stiffness_n_m))
    width, depth = cfg.m3_workbench_size_xy
    assets.append(AssemblyWorkbench(
        initial_pose=Pose(position_xyz=(tool_position[0] + width/2 - .10,
                                        tool_position[1] + cfg.m3_workbench_rear_offset_m - depth, 0.)),
        scale=(width/1.8, depth, tool_position[2]/.74),
    ))
    length = E3_M3_CONVEYOR_LENGTH_M if in_e3 else 7.0
    boxes = []
    if not in_e3:
        boxes.append(("Floor", (8.0, 4.0, 0.05), (0.0, 0.0, -0.025), (0.25, 0.25, 0.25)))
    if in_e3:
        height = cfg.m3_e3_conveyor_height_m
        boxes.append(("Belt", (length, .90, .02), (0., .40 + station_y, height-.01), (.15, .17, .19)))
        for side, y in enumerate((-.08 + station_y, .88 + station_y)):
            boxes.append((f"Frame_{side}", (length, .06, height-.02),
                          (0., y, (height-.02)/2), (.4, .4, .4)))
    else:
        boxes.append(("Belt", (length, 1.16, 0.03), (0.0, 0.40 + station_y, 0.165), (0.15, 0.17, 0.19)))
    for i, x in enumerate((-2.4, 0.0, 2.4)):
        if in_e3:
            continue
        boxes.append((f"Pallet_{i}", (1.0, 0.95, 0.10), (x, 0.40 + station_y, 0.23), (0.45, 0.34, 0.2)))
    for side, y in enumerate(() if in_e3 else (-0.21 + station_y, 1.01 + station_y)):
        boxes.append((f"Frame_{side}", (length, 0.06, 0.10), (0., y, 0.13), (0.4, 0.4, 0.4)))
        boxes.append((f"Guide_{side}", (length, 0.035, 0.03), (0., y, 0.195), (0.9, 0.65, 0.05)))
        for i, x in enumerate((-1.2, 0., 1.2) if in_e3 else (-3., -1.5, 0., 1.5, 3.)):
            boxes.append((f"Leg_{side}_{i}", (0.07, 0.07, 0.15), (x, y, 0.075), (0.35, 0.35, 0.35)))
    for name, size, xyz, color in boxes:
        if name == "Belt" or name.startswith(("Pallet_", "Frame_", "Guide_", "Leg_")):
            xyz = (xyz[0] + cfg.m3_station_offset_x_m, xyz[1], xyz[2])
        box = DexBenchProceduralTable(instance_name=name, prim_path=name, size=size, pos=xyz, color=color)
        assets.append(box)
    assets.append(DexBenchTubeLight(initial_pose=Pose(position_xyz=(0.0, -1.0, 2.5))))
    return Scene(assets=assets)


def add_m3_cli_args(parser: argparse.ArgumentParser):
    parser.add_argument("--m3_scene_layout", choices=("station", "e3_assembly_line"), default="e3_assembly_line",
                        help="Published standalone station or M3 inside the canonical E3 assembly line.")
    parser.add_argument("--m3_fastening_mode", choices=("contact_torque",), default="contact_torque")
    parser.add_argument("--m3_motor_torque_nm", type=float, default=.03)
    parser.add_argument("--m3_thread_resistance_nm", type=float, default=.006)
    parser.add_argument("--m3_motor_direction", type=float, choices=(-1.,1.), default=1.)
    parser.add_argument("--m3_fixture", choices=M3_OBJECT_AXIS.keys(), default=M3_OBJECT_AXIS.default)
    for field in ("turns_to_seat", "motor_rps", "lateral_tolerance_m", "axial_tolerance_m",
                  "alignment_tolerance_deg", "contact_threshold_n", "bit_extension_m",
                  "pickup_pocket_height_m", "station_offset_y_m", "station_offset_x_m", "e3_conveyor_height_m", "workbench_rear_offset_m", "door_opening_deg",
                  "newton_contact_stiffness", "newton_contact_damping", "trigger_press_m", "trigger_stiffness_n_m"):
        name = f"m3_{field}"
        parser.add_argument(f"--{name}", type=float, default=getattr(M3StationCfg, name))
    parser.add_argument("--m3_motor_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--m3_mount_driver", action=argparse.BooleanOptionalAction, default=False,
                        help="Recorded compatibility field; must remain false (free driver).")
    parser.add_argument("--m3_hand_grasp", action=argparse.BooleanOptionalAction, default=True,
                        help="Recorded compatibility field; must remain true (physical pickup).")
    parser.add_argument("--m3_pickup_holder", action=argparse.BooleanOptionalAction, default=False,
                        help="Recorded compatibility field; must remain false (bare workbench).")
    parser.add_argument("--m3_workbench_size_xy", type=float, nargs=2, default=(.9, .55),
                        help="Wood-top pickup workbench width and depth in metres.")
    parser.add_argument("--m3_pickup_position_xyz", type=float, nargs=3,
                        default=M3StationCfg.m3_pickup_position_xyz,
                        help="Workbench drill origin; pair changes with a reachable grasp_joint_waypoint.")
    parser.add_argument("--m3_physics_substeps", type=int, default=4,
                        help="Multiply the physics rate and contact history, preserving the control rate.")
    parser.add_argument("--m3_newton_contact_dimension", type=int, choices=(3, 4, 6), default=4)
    parser.add_argument("--m3_fastener_access_channels", action=argparse.BooleanOptionalAction, default=False,
                        help="Add collision-only access channels where CAD plates occlude B3/B6.")
