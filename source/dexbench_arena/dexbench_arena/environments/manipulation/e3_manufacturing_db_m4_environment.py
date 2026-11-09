# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""DB-M4: hand-start the LG FX25 leveling leg into the upturned base cabinet, in the E3 manufacturing cell, on Newton."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from dexbench_arena.contact_materials import HandContactMaterialCfg
from dexbench_arena.embodiments.lookup import embodiment_class, start_poses
from dexbench_arena.layouts.db_h1_layout import torso_crouch_joint_pos
from dexbench_arena.layouts.db_m4_layout import (
    AXIS_MAX_OFFSET_M,
    AXIS_MAX_TILT_DEG,
    BASE_CABINET_NAME,
    BASE_CABINET_ROOT_XYZ,
    CABINET_STATION_JITTER_M,
    CABINET_STATION_MAX_ABS_XY_M,
    EPISODE_TIMEOUT_S,
    HAND_START_DEPTH_M,
    HAND_START_HOLD_S,
    HOLDER_CENTER_XY,
    HOLDER_WALLS,
    HOLE_INSERTION_AXIS_IN_CABINET,
    INTERIOR_LIGHT_XYZ,
    LEG_DROPPED_Z_M,
    LEG_NAME,
    LEG_START_XYZ,
    LEG_START_YAW_RAD,
    LEG_STATION_JITTER_M,
    LegStationDomain,
    LEG_STUD_AXIS_IN_LEG,
    M4_HAND_DYNAMIC_FRICTION,
    M4_HAND_STATIC_FRICTION,
    M4_LEG_OBJECT_AXIS,
    M4ThreadContactMaterialCfg,
    PALLET_CENTER_XYZ,
    PALLET_SIZE_M,
    START_TORSO_CROUCH_RAD,
    START_TORSO_YAW_RAD,
    TARGET_HOLE_IN_CABINET_XYZ,
    THREAD_FRICTION,
)
from dexbench_arena.environments.manipulation.scene_environment_base import TASKS

if TYPE_CHECKING:
    from isaaclab_arena.scene.scene import Scene

HAND_START_LEG_TASK_NAME = "hand_start_leg"
M4_ANCHOR = "A_m4"
_M4_ONLY = {TASKS: (HAND_START_LEG_TASK_NAME,)}


M4_NEWTON_SOLVER_OPTIONS = {
    "num_substeps": 20,
    "collision_decimation": 10,
    "bvh_constructor_geometry": "lbvh",
}


@dataclass
class M4StationCfg:
    """DB-M4's settings: the fields the E3 config and DB-M4's own id's config take for this staging."""

    task: str = "none"
    """Task variant the scene builds: ``hand_start_leg``, or ``none`` for the scene shell."""
    task_start_pose: str = field(default="m4_side_pick", metadata=_M4_ONLY)
    """The embodiment's named pose the start is built from, before the waist yaw and crouch."""
    start_torso_yaw_rad: float = field(default=START_TORSO_YAW_RAD, metadata=_M4_ONLY)
    leg_start_yaw_rad: float = field(default=LEG_START_YAW_RAD, metadata=_M4_ONLY)
    leg_station_x_m: float = field(default=HOLDER_CENTER_XY[0], metadata=_M4_ONLY)
    leg_station_y_m: float = field(default=HOLDER_CENTER_XY[1], metadata=_M4_ONLY)
    thread_static_friction: float = field(default=THREAD_FRICTION[0], metadata=_M4_ONLY)
    thread_dynamic_friction: float = field(default=THREAD_FRICTION[1], metadata=_M4_ONLY)
    thread_torsional_friction: float = field(default=THREAD_FRICTION[2], metadata=_M4_ONLY)
    """ThreadSteel's friction (``M4ThreadContactMaterialCfg``); only ``hand_start_leg`` applies it."""
    m4_leg: str = field(default="lg_fx25_leveling_leg", metadata=_M4_ONLY)
    """The leg the task is run with (``M4_LEG_OBJECT_AXIS``; one)."""

    def __post_init__(self):
        material = M4ThreadContactMaterialCfg(self.thread_static_friction, self.thread_dynamic_friction,
                                             self.thread_torsional_friction)
        if material != M4ThreadContactMaterialCfg() and self.task != HAND_START_LEG_TASK_NAME:
            raise ValueError("A thread friction other than the default needs task hand_start_leg, which applies it")
        LegStationDomain((self.leg_station_x_m, self.leg_station_y_m))
        if not math.isfinite(self.start_torso_yaw_rad) or abs(self.start_torso_yaw_rad) > math.radians(135):
            raise ValueError("start_torso_yaw_rad must be within the waist's +/-135 degree range")
        if not math.isfinite(self.leg_start_yaw_rad):
            raise ValueError("leg_start_yaw_rad must be finite")


def make_hand_start_leg_task(environment, cfg: M4StationCfg):
    """Build DB-M4's task (``available_tasks``)."""
    del environment
    from dexbench_arena.tasks.hand_start_leg import DexBenchHandStartLegTask

    return DexBenchHandStartLegTask(
        object_name=LEG_NAME,
        receiver_name=BASE_CABINET_NAME,
        hole_in_receiver_xyz=TARGET_HOLE_IN_CABINET_XYZ,
        hole_axis_in_receiver=HOLE_INSERTION_AXIS_IN_CABINET,
        stud_axis_in_object=LEG_STUD_AXIS_IN_LEG,
        min_depth_m=HAND_START_DEPTH_M,
        max_offset_m=AXIS_MAX_OFFSET_M,
        max_tilt_deg=AXIS_MAX_TILT_DEG,
        hold_time_s=HAND_START_HOLD_S,
        dropped_below_z_m=LEG_DROPPED_Z_M,
        episode_length_s=EPISODE_TIMEOUT_S,
        task_description=(
            "Pick up the leveling leg from its holder with your left hand, put its stud on the free bracket hole "
            "of the base cabinet on your left and turn it in by hand until the thread has taken it."
        ),
        robot_joint_pos_overrides=m4_start_pose(cfg),
        hand_contact_material=HandContactMaterialCfg(
            static_friction=M4_HAND_STATIC_FRICTION,
            dynamic_friction=M4_HAND_DYNAMIC_FRICTION,
            friction_combine_mode="max",
        ),
    )


def m4_start_pose(cfg: M4StationCfg):
    """The embodiment's named task pose, with DB-M4 waist/crouch overrides; None without a robot."""
    if cfg.embodiment is None:
        return None
    poses = start_poses(cfg.embodiment)
    if cfg.task_start_pose not in poses:
        raise ValueError(f"Unknown DB-M4 task_start_pose {cfg.task_start_pose!r}; available: {tuple(poses)}")
    base = poses[cfg.task_start_pose]
    joint = getattr(embodiment_class(cfg.embodiment), "torso_yaw_joint_name", None)
    if not joint or joint not in base:
        raise ValueError(f"DB-M4 turns the waist on the embodiment's torso_yaw_joint_name; {cfg.embodiment!r} declares none")
    return {**torso_crouch_joint_pos(base, START_TORSO_CROUCH_RAD), joint: cfg.start_torso_yaw_rad}


def build_m4_scene(environment, cfg: M4StationCfg) -> Scene:
    """The cell staged for DB-M4, in the M4 frame; the thread material and the station draw only for ``hand_start_leg``."""
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_arena.utils.yaw import rotate_quat_by_yaw

    from dexbench_arena.assets import background_library as _background_library  # noqa: F401
    from dexbench_arena.assets import object_library as _object_library  # noqa: F401

    registry = environment.asset_registry
    background = registry.get_asset_by_name("dexbench_assembly_line_e3_m4")()
    light = registry.get_asset_by_name("dexbench_tube_light")(initial_pose=Pose(position_xyz=INTERIOR_LIGHT_XYZ))
    table = registry.get_asset_by_name("dexbench_table")
    pallet = table(
        instance_name="pallet", prim_path="pallet", size=PALLET_SIZE_M, pos=PALLET_CENTER_XYZ, color=(0.55, 0.42, 0.28)
    )
    shift = (cfg.leg_station_x_m - HOLDER_CENTER_XY[0], cfg.leg_station_y_m - HOLDER_CENTER_XY[1], 0.0)
    domain = LegStationDomain((cfg.leg_station_x_m, cfg.leg_station_y_m))
    holder = [
        table(instance_name=name, prim_path=name, size=size, pos=tuple(a + b for a, b in zip(center, shift)), color=(0.2, 0.35, 0.7))
        for name, (size, center) in HOLDER_WALLS.items()
    ]
    base_cabinet = registry.get_asset_by_name("lg_fx25_base_cabinet")(
        instance_name=BASE_CABINET_NAME, initial_pose=Pose(position_xyz=BASE_CABINET_ROOT_XYZ)
    )
    leg = registry.get_asset_by_name(M4_LEG_OBJECT_AXIS.selected(cfg).asset)(
        instance_name=LEG_NAME, initial_pose=Pose(
            position_xyz=tuple(a + b for a, b in zip(LEG_START_XYZ, shift)),
            rotation_xyzw=rotate_quat_by_yaw((0.0, 0.0, 0.0, 1.0), cfg.leg_start_yaw_rad),
        )
    )
    if getattr(cfg, "task", None) == HAND_START_LEG_TASK_NAME:
        from dexbench_arena.assets.db_m4_thread_material import configure_m4_thread_material

        material = M4ThreadContactMaterialCfg(cfg.thread_static_friction, cfg.thread_dynamic_friction,
                                             cfg.thread_torsional_friction)
        for asset in (leg, base_cabinet):
            configure_m4_thread_material(asset, material)
        _attach_leg_station_variation(leg, holder, domain)
        _attach_cabinet_station_variation(base_cabinet, pallet)
    return Scene(assets=[background, light, pallet, *holder, base_cabinet, leg])


def _attach_leg_station_variation(leg, holder, domain) -> None:
    """The +-5 mm draw on where the leg stands: one draw moves the leg and the five holder pieces."""
    from dexbench_arena.variations.object_pose_variation import make_planar_pose_variation

    leg.add_variation(
        make_planar_pose_variation(
            LEG_NAME,
            name="station_pose",
            xy_half_extent_m=LEG_STATION_JITTER_M,
            domain=domain,
            member_names=[LEG_NAME, *(wall.name for wall in holder)],
        )
    )


def _attach_cabinet_station_variation(base_cabinet, pallet) -> None:
    """Where the cabinet stands on the workbench: one draw moves the cabinet and its pallet together."""
    from dexbench_arena.variations.object_pose_variation import PlanarPoseDomain, make_planar_pose_variation

    base_cabinet.add_variation(
        make_planar_pose_variation(
            BASE_CABINET_NAME,
            name="station_pose",
            xy_half_extent_m=CABINET_STATION_JITTER_M,
            domain=PlanarPoseDomain(
                max_abs_xy_m=CABINET_STATION_MAX_ABS_XY_M,
                footprint_half_extents_xy_m=(PALLET_SIZE_M[0] / 2, PALLET_SIZE_M[1] / 2),
                reason="the cabinet's pallet stays on the workbench's worktop",
            ),
            member_names=[BASE_CABINET_NAME, pallet.name],
        )
    )


def add_m4_cli_args(parser: argparse.ArgumentParser) -> None:
    """DB-M4's flags (``M4StationCfg``)."""
    if "--m4_leg" not in parser._option_string_actions:
        parser.add_argument(
            "--m4_leg",
            choices=M4_LEG_OBJECT_AXIS.keys(),
            default=M4_LEG_OBJECT_AXIS.default,
            help="The leg the task is run with (one).",
        )
    for flag, default in zip(
        ("thread_static_friction", "thread_dynamic_friction", "thread_torsional_friction"), THREAD_FRICTION
    ):
        if "--" + flag not in parser._option_string_actions:
            parser.add_argument("--" + flag, type=float, default=default,
                                help="ThreadSteel friction for hand_start_leg; the torsional one in metres.")
    for flag, default in (("leg_station_x_m", HOLDER_CENTER_XY[0]), ("leg_station_y_m", HOLDER_CENTER_XY[1]),
                          ("start_torso_yaw_rad", START_TORSO_YAW_RAD), ("leg_start_yaw_rad", LEG_START_YAW_RAD)):
        if "--" + flag not in parser._option_string_actions:
            parser.add_argument("--" + flag, type=float, default=default, help="DB-M4 setup coordinate [m] or yaw [rad]; see the task card.")
    if "--task_start_pose" not in parser._option_string_actions:
        parser.add_argument("--task_start_pose", default="m4_side_pick", help="Embodiment pose before DB-M4's waist/crouch overrides.")
