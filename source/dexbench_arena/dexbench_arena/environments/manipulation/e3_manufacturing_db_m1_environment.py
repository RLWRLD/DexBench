# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""DB-M1: insert the LG FH25EAE detergent drawer into its washer, in the E3 manufacturing cell, on Newton."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from dexbench_arena.embodiments.lookup import embodiment_class, start_poses
from dexbench_arena.layouts.db_h1_layout import torso_crouch_joint_pos
from dexbench_arena.layouts.db_m1_layout import (
    TABLETOP_DRAWER_START_XYZ,
    TABLETOP_DRAWER_YAW_DEG,
    TABLETOP_DRAWER_DROPPED_Z_M,
    TABLETOP_EPISODE_TIMEOUT_S,
    TABLETOP_SEAT_DEPTH_WINDOW_M,
    TABLETOP_SEAT_MAX_TILT_DEG,
    TABLETOP_SEAT_MAX_LINEAR_SPEED_M_S,
    ROBOT_BASE_YAW_DEG,
    DRAWER_DROPPED_Z_M,
    DRAWER_NAME,
    DRAWER_NOMINAL_START_XYZ,
    DRAWER_SENSING_COLLIDERS,
    EPISODE_TIMEOUT_S,
    INTERIOR_LIGHT_XYZ,
    M1_DRAWER_OBJECT_AXIS,
    PROGRESS_ALIGN_LEAD_M,
    PROGRESS_ALIGN_MAX_LATERAL_ERROR_M,
    PROGRESS_ALIGN_MAX_TILT_DEG,
    PROGRESS_HALF_DEPTH_M,
    PROGRESS_RAIL_ENTRY_DEPTH_M,
    PROGRESS_RAILS_ENGAGED_DEPTH_M,
    SEAT_DEPTH_WINDOW_M,
    SEAT_HOLD_S,
    SEAT_IN_WASHER_XYZ,
    SEAT_MAX_LATERAL_ERROR_M,
    SEAT_MAX_TILT_DEG,
    SLOT_CONTACT_REGIONS,
    WASHER_COUNTERPART_COLLIDERS,
    WASHER_LINE_CENTER_XYZ,
    WASHER_LINE_SIZE_M,
    WASHER_NAME,
    WASHER_ROOT_XYZ,
)
from dexbench_arena.environments.manipulation.scene_environment_base import TASKS
from dexbench_arena.environments.manipulation.db_m1_physics import M1_PHYSICS_PROFILES, apply_m1_physics_profile

if TYPE_CHECKING:
    from isaaclab_arena.scene.scene import Scene

INSERT_DRAWER_TASK_NAME = "insert_drawer"
PICK_INSERT_DRAWER_TASK_NAME = "pick_insert_drawer"
M1_ANCHOR = "A_m1"
_M1_ONLY = {TASKS: (INSERT_DRAWER_TASK_NAME, PICK_INSERT_DRAWER_TASK_NAME)}
_M1_TABLETOP_ONLY = {TASKS: (PICK_INSERT_DRAWER_TASK_NAME,)}



@dataclass
class M1StationCfg:
    """DB-M1's settings: the fields the E3 config and DB-M1's own id's config take for this staging."""

    task: str = "none"
    """Task variant: ``insert_drawer``, ``pick_insert_drawer``, or ``none`` for the scene shell."""
    m1_drawer: str = field(default="lg_fh25eae_detergent_drawer", metadata=_M1_ONLY)
    """The drawer the task is run with (``M1_DRAWER_OBJECT_AXIS``; one for EA)."""
    m1_arm_armature: float | None = field(default=None, metadata=_M1_ONLY)
    """Arm joint armature on Newton for a comparison run (the EA rig's 0.05); None keeps the shared Newton profile."""
    m1_self_collision: bool = field(default=False, metadata=_M1_ONLY)
    """Robot self-collision for a comparison run (the EA rig had it on); False keeps the embodiment's (off)."""
    m1_hand_effort_limit_nm: float = field(default=5.0, metadata=_M1_ONLY)
    """Finger effort cap [N m]: the release robot's 5; 3 reproduces the EA rig the training set was recorded with."""
    m1_physics_profile: str = field(default="default", metadata=_M1_TABLETOP_ONLY)
    """Named contact profile: default or the collection's damped_contacts."""
    m1_grasp_diagnostics: bool = field(default=False, metadata=_M1_ONLY)
    """Keep release gates, poses, forces and wall time on the grasp event term's ``reset_result`` (``formed_grasp``)."""


    m1_tabletop_drawer_xyz: tuple[float, float, float] = field(default=TABLETOP_DRAWER_START_XYZ, metadata=_M1_TABLETOP_ONLY)
    """Flat pickup spawn in environment coordinates [m]."""
    m1_tabletop_drawer_mass_kg: float = field(default=0.55, metadata=_M1_TABLETOP_ONLY)
    """Measured drawer mass; reset scales inertia with the same ratio."""
    m1_tabletop_drawer_yaw_deg: float = field(default=TABLETOP_DRAWER_YAW_DEG, metadata=_M1_TABLETOP_ONLY)
    m1_tabletop_base_xy: tuple[float, float] | None = field(default=None, metadata=_M1_TABLETOP_ONLY)
    m1_tabletop_base_yaw_deg: float = field(default=ROBOT_BASE_YAW_DEG, metadata=_M1_TABLETOP_ONLY)
    m1_tabletop_crouch_deg: float | None = field(default=None, metadata=_M1_TABLETOP_ONLY)

    def __post_init__(self):
        if self.m1_physics_profile not in M1_PHYSICS_PROFILES:
            raise ValueError(f"m1_physics_profile must be one of {M1_PHYSICS_PROFILES}")
        if self.m1_physics_profile != "default" and self.task != PICK_INSERT_DRAWER_TASK_NAME:
            raise ValueError("m1_physics_profile is only configurable for pick_insert_drawer")
        if not math.isfinite(self.m1_tabletop_drawer_mass_kg) or self.m1_tabletop_drawer_mass_kg <= 0:
            raise ValueError("m1_tabletop_drawer_mass_kg must be positive and finite")


M1_NEWTON_SOLVER_OPTIONS = {"num_substeps": 8, "nconmax": 6000, "njmax": 6000}


def apply_m1_newton_robot_settings(env_cfg, cfg: M1StationCfg) -> None:
    """The finger effort, and the EA rig's robot settings where a run selects them."""
    robot = getattr(env_cfg.scene, "robot", None)
    if robot is None:
        apply_m1_physics_profile(env_cfg, cfg.m1_physics_profile)
        return
    if not math.isfinite(cfg.m1_hand_effort_limit_nm) or cfg.m1_hand_effort_limit_nm <= 0:
        raise ValueError("m1_hand_effort_limit_nm must be positive and finite")
    embodiment = embodiment_class(cfg.embodiment)
    hand_groups = getattr(embodiment, "hand_actuator_groups", ())
    if not hand_groups:
        raise ValueError(
            f"DB-M1 sets the finger effort on the embodiment's hand_actuator_groups; {cfg.embodiment!r} declares none"
        )
    for group_name in hand_groups:
        robot.actuators[group_name].effort_limit_sim = cfg.m1_hand_effort_limit_nm
    if cfg.m1_arm_armature is not None:
        if not math.isfinite(cfg.m1_arm_armature) or cfg.m1_arm_armature < 0:
            raise ValueError("m1_arm_armature must be non-negative and finite")
        arm_groups = getattr(embodiment, "arm_actuator_groups", ())
        if not arm_groups:
            raise ValueError(f"--m1_arm_armature needs the embodiment's arm_actuator_groups; {cfg.embodiment!r} declares none")
        for group_name in arm_groups:
            robot.actuators[group_name].armature = float(cfg.m1_arm_armature)
    if cfg.m1_self_collision:
        robot.spawn.articulation_props.enabled_self_collisions = True
    apply_m1_physics_profile(env_cfg, cfg.m1_physics_profile,
                             hand_body_pattern=getattr(embodiment, "hand_body_label_pattern", None))


def make_insert_drawer_task(environment, cfg: M1StationCfg):
    """Build DB-M1's task (``available_tasks``)."""
    del environment
    from dexbench_arena.tasks.insert_drawer import DexBenchInsertDrawerTask

    tabletop = cfg.task == PICK_INSERT_DRAWER_TASK_NAME
    return DexBenchInsertDrawerTask(
        object_name=DRAWER_NAME,
        washer_name=WASHER_NAME,
        seat_in_washer_xyz=SEAT_IN_WASHER_XYZ,
        depth_window_m=TABLETOP_SEAT_DEPTH_WINDOW_M if tabletop else SEAT_DEPTH_WINDOW_M,
        max_lateral_error_m=SEAT_MAX_LATERAL_ERROR_M,
        max_tilt_deg=TABLETOP_SEAT_MAX_TILT_DEG if tabletop else SEAT_MAX_TILT_DEG,
        hold_time_s=SEAT_HOLD_S,
        max_linear_speed_m_s=TABLETOP_SEAT_MAX_LINEAR_SPEED_M_S if tabletop else None,
        dropped_below_z_m=TABLETOP_DRAWER_DROPPED_Z_M if tabletop else DRAWER_DROPPED_Z_M,
        drawer_sensing_colliders=DRAWER_SENSING_COLLIDERS,
        washer_counterpart_colliders=WASHER_COUNTERPART_COLLIDERS,
        slot_contact_regions=SLOT_CONTACT_REGIONS,
        progress_stage_depths_m=(
            PROGRESS_RAIL_ENTRY_DEPTH_M - PROGRESS_ALIGN_LEAD_M,
            PROGRESS_RAILS_ENGAGED_DEPTH_M,
            PROGRESS_HALF_DEPTH_M,
        ),
        progress_align_max_lateral_error_m=PROGRESS_ALIGN_MAX_LATERAL_ERROR_M,
        progress_align_max_tilt_deg=PROGRESS_ALIGN_MAX_TILT_DEG,
        episode_length_s=TABLETOP_EPISODE_TIMEOUT_S if tabletop else EPISODE_TIMEOUT_S,
        task_description=("Pick up the detergent drawer from the workstation and insert it into the washer's slot until it is seated."
                          if tabletop else "Push the detergent drawer you are holding into the washer's slot until it is seated."),
    )


def build_m1_scene(environment, cfg: M1StationCfg) -> Scene:
    """The cell staged for DB-M1, in the M1 frame; the live grasp is attached only for ``insert_drawer``."""
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose

    from dexbench_arena.assets import background_library as _background_library  # noqa: F401
    from dexbench_arena.assets import object_library as _object_library  # noqa: F401

    registry = environment.asset_registry
    background = registry.get_asset_by_name("dexbench_assembly_line_e3_m1")()
    light = registry.get_asset_by_name("dexbench_tube_light")(initial_pose=Pose(position_xyz=INTERIOR_LIGHT_XYZ))
    washer = registry.get_asset_by_name("lg_fh25eae_washer")(
        instance_name=WASHER_NAME, initial_pose=Pose(position_xyz=WASHER_ROOT_XYZ)
    )
    tabletop = cfg.task == PICK_INSERT_DRAWER_TASK_NAME
    drawer_pose = Pose(position_xyz=DRAWER_NOMINAL_START_XYZ)
    if tabletop:
        import torch
        from isaaclab.utils.math import quat_from_euler_xyz
        from dexbench_arena.variations.db_m1_tabletop_variation import tabletop_pose_variations
        from isaaclab_arena.variations.object_mass_variation import ObjectMassVariationCfg
        from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg

        if len(cfg.m1_tabletop_drawer_xyz) != 3 or not all(math.isfinite(v) for v in cfg.m1_tabletop_drawer_xyz):
            raise ValueError("m1_tabletop_drawer_xyz must contain three finite coordinates")
        if not math.isfinite(cfg.m1_tabletop_drawer_yaw_deg):
            raise ValueError("m1_tabletop_drawer_yaw_deg must be finite")
        zero = torch.zeros(1)
        yaw = torch.tensor([math.radians(cfg.m1_tabletop_drawer_yaw_deg)])
        drawer_pose = Pose(position_xyz=tuple(cfg.m1_tabletop_drawer_xyz),
                           rotation_xyzw=tuple(quat_from_euler_xyz(zero, zero, yaw)[0].tolist()))
    drawer = registry.get_asset_by_name(M1_DRAWER_OBJECT_AXIS.selected(cfg).asset)(
        instance_name=DRAWER_NAME,
        initial_pose=drawer_pose,
    )
    if tabletop:
        drawer.get_variation("mass").apply_cfg(ObjectMassVariationCfg(
            enabled=True, sampler_cfg=UniformSamplerCfg(low=[cfg.m1_tabletop_drawer_mass_kg], high=[cfg.m1_tabletop_drawer_mass_kg]),
        ))
        drawer_variation, washer_variation = tabletop_pose_variations(cfg.m1_tabletop_drawer_xyz, cfg.m1_tabletop_drawer_yaw_deg)
        drawer.add_variation(drawer_variation)
        washer.add_variation(washer_variation)
    washer_line = registry.get_asset_by_name("dexbench_table")(
        prim_path="WasherLine",
        size=WASHER_LINE_SIZE_M,
        pos=WASHER_LINE_CENTER_XYZ,
        color=(0.25, 0.25, 0.27),
    )
    if getattr(cfg, "task", None) == INSERT_DRAWER_TASK_NAME:
        _attach_start_state(drawer, cfg)
    return Scene(assets=[background, light, washer_line, washer, drawer])


def _attach_start_state(drawer, cfg: M1StationCfg) -> None:
    """Attach the start-state grasp; Arena records and replays its six-coordinate draw."""
    from dexbench_arena.variations.db_m1_grasp_disturbance_variation import M1GraspDisturbance

    drawer.add_variation(M1GraspDisturbance(DRAWER_NAME, diagnostics=cfg.m1_grasp_diagnostics))


def add_m1_cli_args(parser: argparse.ArgumentParser) -> None:
    """DB-M1's flags (``M1StationCfg``)."""
    if "--m1_tabletop_drawer_xyz" not in parser._option_string_actions:
        parser.add_argument("--m1_tabletop_drawer_xyz", type=float, nargs=3, default=TABLETOP_DRAWER_START_XYZ,
                            help="Tabletop drawer spawn XYZ in environment coordinates [m].")
        parser.add_argument("--m1_tabletop_drawer_mass_kg", type=float, default=0.55,
                            help="Measured tabletop drawer mass [kg]; inertia scales with mass.")
        parser.add_argument("--m1_tabletop_drawer_yaw_deg", type=float, default=TABLETOP_DRAWER_YAW_DEG,
                            help="Tabletop drawer spawn yaw in environment coordinates [degrees].")
        parser.add_argument("--m1_tabletop_base_xy", type=float, nargs=2, default=None,
                            help="Override the tabletop robot base XY [m]; unset keeps the named anchor.")
        parser.add_argument("--m1_tabletop_base_yaw_deg", type=float, default=ROBOT_BASE_YAW_DEG,
                            help="Robot base heading [degrees], used with --m1_tabletop_base_xy.")
        parser.add_argument("--m1_tabletop_crouch_deg", type=float, default=None,
                            help="Tabletop torso crouch angle in (0, 65] degrees; unset keeps the selected pose.")
    if "--m1_physics_profile" not in parser._option_string_actions:
        parser.add_argument("--m1_physics_profile", choices=M1_PHYSICS_PROFILES, default="default",
                            help="Contact profile; damped_contacts reproduces the tabletop collection settings.")
    if "--m1_hand_effort_limit_nm" not in parser._option_string_actions:
        parser.add_argument("--m1_hand_effort_limit_nm", type=float, default=5.0,
                            help="Hand joint effort limit [N m] for M1.")
    if "--m1_arm_armature" not in parser._option_string_actions:
        parser.add_argument(
            "--m1_arm_armature",
            type=lambda value: None if value.lower() == "none" else float(value),
            default=None,
            help="Arm joint armature on Newton for a comparison run (the EA rig's 0.05); unset keeps the shared profile.",
        )
    if "--m1_self_collision" not in parser._option_string_actions:
        parser.add_argument(
            "--m1_self_collision",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Robot self-collision as the EA rig had it, for a comparison run; off keeps the embodiment's.",
        )
    if "--m1_drawer" not in parser._option_string_actions:
        parser.add_argument(
            "--m1_drawer",
            choices=M1_DRAWER_OBJECT_AXIS.keys(),
            default=M1_DRAWER_OBJECT_AXIS.default,
            help="The drawer the task is run with (one for EA).",
        )
    if "--m1_grasp_diagnostics" not in parser._option_string_actions:
        parser.add_argument(
            "--m1_grasp_diagnostics",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Expose live grasp release diagnostics for acceptance harnesses.",
        )


def m1_start_pose(cfg: M1StationCfg, pose):
    """Apply the tabletop crouch to the selected embodiment start pose."""
    if cfg.m1_tabletop_crouch_deg is None:
        return pose
    if cfg.task != PICK_INSERT_DRAWER_TASK_NAME:
        raise ValueError("m1_tabletop_crouch_deg is only for pick_insert_drawer")
    if not math.isfinite(cfg.m1_tabletop_crouch_deg) or not 0 < cfg.m1_tabletop_crouch_deg <= 65:
        raise ValueError("m1_tabletop_crouch_deg must be in (0, 65]")
    base = pose or start_poses(cfg.embodiment).get("teleop_ready")
    if base is None:
        raise ValueError("The tabletop crouch needs an embodiment with a teleop_ready start pose")
    return torso_crouch_joint_pos(base, math.radians(cfg.m1_tabletop_crouch_deg))

def m1_base_anchors(cfg: M1StationCfg, anchors):
    """Apply an explicitly configured tabletop base position."""
    if cfg.m1_tabletop_base_xy is None:
        return anchors
    if cfg.task != PICK_INSERT_DRAWER_TASK_NAME:
        raise ValueError("m1_tabletop_base_xy is only for pick_insert_drawer")
    if len(cfg.m1_tabletop_base_xy) != 2 or not all(math.isfinite(v) for v in cfg.m1_tabletop_base_xy):
        raise ValueError("m1_tabletop_base_xy must contain two finite coordinates")
    return {**anchors, cfg.anchor: (*cfg.m1_tabletop_base_xy, 0.0)}

def m1_facing_targets(cfg: M1StationCfg, targets):
    """Aim the tabletop base along its configured heading."""
    if cfg.m1_tabletop_base_xy is None:
        return targets
    if cfg.task != PICK_INSERT_DRAWER_TASK_NAME:
        raise ValueError("m1_tabletop_base_xy is only for pick_insert_drawer")
    if not math.isfinite(cfg.m1_tabletop_base_yaw_deg):
        raise ValueError("m1_tabletop_base_yaw_deg must be finite")
    x, y = cfg.m1_tabletop_base_xy
    yaw = math.radians(cfg.m1_tabletop_base_yaw_deg)
    return {**targets, cfg.anchor: (x + math.cos(yaw), y + math.sin(yaw))}
