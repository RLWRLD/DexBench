# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Arena-native RBY1-M + Wuji Hand 2 (wujiv2) embodiment."""

from __future__ import annotations

import functools
import os
from collections.abc import Sequence
from dataclasses import MISSING, asdict, field

import numpy as np
import torch
import warp as wp

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
import isaaclab.utils.math as PoseUtils
import isaaclab_tasks.contrib.pick_place.mdp as mdp
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.controllers.pink_ik import (
    DampingTaskCfg,
    FrameTaskCfg,
    NullSpacePostureTaskCfg,
    PinkIKController,
    PinkIKControllerCfg,
)
from isaaclab.controllers.pink_ik.pink_kinematics_configuration import PinkKinematicsConfiguration
from isaaclab.envs.mdp.actions import JointPositionActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.utils.configclass import configclass
from isaaclab_teleop import XrCfg

from isaaclab_arena.utils.cameras import ArenaCameraCfg
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.common.mimic_utils import get_rigid_and_articulated_object_poses
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLMimicEnv
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.variations.camera_intrinsics_variation import CameraIntrinsicsVariation

from dexbench_arena.contact_materials import WUJI_GRASP_CONTACT_MATERIAL
from dexbench_arena.embodiments.camera_catalog import (
    RBY1_HEAD_ZED_MINI,  # noqa: F401 - retained for the one-line opt-in below.
    ZED_MINI_HD720,
    ZED_MINI_SPEC_BASELINE_M,
    CameraMount,
    CameraRig,
    StereoRightEye,
)
from dexbench_arena.embodiments.common.recorder_ready_events import RECORDER_READY_MODE
from dexbench_arena.embodiments.common.compliant_grasp import AdaptiveGraspCfg, CompliantGraspCfg
from dexbench_arena.embodiments.common.grasp_action import SavedGraspJointPositionActionCfg
from dexbench_arena.embodiments.events import reset_all_articulation_joints_and_targets
from dexbench_arena.embodiments.fingertip_contact import (
    FINGERTIP_CONTACT_SENSOR_NAME,
    FingertipContactMixin,
    fingertip_contact_force,
    fingertip_contact_sensor_cfg,
)
from dexbench_arena.embodiments.hand_contact import (
    HandContactMaterialMixin,
    check_hand_contact_material,
    install_hand_contact_recording_metadata,
)
from dexbench_arena.embodiments.hand_actions import (
    merge_bimanual_hand_actions,
    split_bimanual_hand_actions,
)
from dexbench_arena.embodiments.joint_state import as_torch, get_joint_state, resolve_joint_index, resolve_joint_order
from dexbench_arena.embodiments.joint_torque import JointTorqueMixin

from dexbench_arena.embodiments.common.clean_action_labels import clean_action_batch, stash_clean_action
from dexbench_arena.embodiments.common.mimic_boundaries import ReplaysSubtaskBoundaries
from dexbench_arena.embodiments.common.xr import xr_enabled
from dexbench_arena.embodiments.rby1m.constants import RBY1M_BASE_MODEL
from dexbench_arena.embodiments.rby1m.ik import RBY1M_IK_TELEOP, Rby1mIkProfile, resolve_ik_profile
from dexbench_arena.embodiments.rby1m.waist_ik import (
    Rby1mPinkInverseKinematicsAction,
    Rby1mPinkInverseKinematicsActionCfg,
    enable_rby1m_waist_ik,
    resolve_waist_ik_args,
)
from dexbench_arena.embodiments.rby1m.xr import rby1m_xr_cfg

from dexbench_arena.embodiments.rby1_wuji2.constants import (
    RBY1_WUJI2_JOINT_EMBODIMENT,
    RBY1_WUJI2_PAPER_HAND,
    RBY1_WUJI2_BASE_LINK_NAME,
    RBY1_WUJI2_BASE_LINK_PRIM_PATH,
    RBY1_WUJI2_COLLISION_SPHERES,
    RBY1_WUJI2_FINGERTIP_DISTAL_BODY_NAMES,
    RBY1_WUJI2_FINGERTIP_PAD_BODY_NAMES,
    RBY1_WUJI2_HAND_JOINT_ACTION_NAMES,
    RBY1_WUJI2_HEAD_CAMERA_LINK_NAME,
    RBY1_WUJI2_HEAD_JOINT_NAMES,
    RBY1_WUJI2_HELD_OBJECT_CONTACT_BODY_EXPRS,
    RBY1_WUJI2_POWER_DRIVER_GRASP,
    RBY1_WUJI2_JOINT_ACTION_NAMES,
    RBY1_WUJI2_LEFT_ARM_JOINT_NAMES,
    RBY1_WUJI2_LEFT_EEF_LINK_NAME,
    RBY1_WUJI2_LEFT_HAND_COMMAND_JOINT_NAMES,
    RBY1_WUJI2_LEFT_WRIST_CONTROL_LINK_NAME,
    RBY1_WUJI2_LEFT_WRIST_IDLE_ROOT_POSE,
    RBY1_WUJI2_MJCF_HAND_CONTACT_MATERIAL,
    RBY1_WUJI2_NULL_SPACE_JOINT_NAMES,
    RBY1_WUJI2_PINK_CONTROLLED_JOINT_NAMES,
    RBY1_WUJI2_RETIRED_GAIN_VARIABLES,
    RBY1_WUJI2_RIGHT_ARM_JOINT_NAMES,
    RBY1_WUJI2_RIGHT_EEF_LINK_NAME,
    RBY1_WUJI2_RIGHT_WRIST_CONTROL_LINK_NAME,
    RBY1_WUJI2_RIGHT_WRIST_IDLE_ROOT_POSE,
    RBY1_WUJI2_ROBOT_PRIM_PATH,
    RBY1_WUJI2_CONTROLLER_HOME_JOINT_POS,
    RBY1_WUJI2_START_POSES,
    RBY1_WUJI2_TORSO_JOINT_NAMES,
    RBY1_WUJI2_TORSO_YAW_JOINT_NAME,
    RBY1_WUJI2_USD_PATH,
    RBY1_WUJI2_WHEEL_JOINT_NAMES,
    RBY1_WUJI2_ZED_MINI_LEFT_POS,
    RBY1_WUJI2_ZED_MINI_QUAT,
    WUJI2_FINGER_ORDER,
)
from dexbench_arena.embodiments.rby1_wuji2.mimic_source import (
    RBY1_WUJI2_JOINT_ACTION_DIMS,
    validate_mimic_source_dataset,
)
from dexbench_arena.embodiments.rby1_wuji2.pinocchio_urdf import get_rby1_wuji2_pinocchio_urdf_path
from dexbench_arena.embodiments.rby1_wuji2.spawn import HAND_CONTACT_MATERIAL_NAME, Rby1Wuji2UsdFileCfg
from dexbench_arena.embodiments.rby1_wuji2.wrist_cameras import PALM_CAMERA_RIGS
from dexbench_arena.evaluation.replay_guards import deterministic_reset
from dexbench_arena.variations.camera_mount_extrinsics_variation import MountedCameraExtrinsicsVariation
from dexbench_arena.embodiments.eef_arm_ik_controller import ArmIkLayout, ArmIkSettings, EefArmIkController

_WUJI2_HAND_JOINT_REGEX = "(thumb|index_finger|middle_finger|ring_finger|pinky)_.*"

_DEFAULT_RBY1_WUJI2_LEFT_CAMERA_OFFSET = Pose(
    position_xyz=RBY1_WUJI2_ZED_MINI_LEFT_POS, rotation_xyzw=RBY1_WUJI2_ZED_MINI_QUAT
)


RBY1_WUJI2_HEAD_RIG = CameraRig(
    model=ZED_MINI_HD720.name,
    left=CameraMount(
        obs_key="robot_pov_left_cam",
        prim_name="left_cam",
        parent_link=RBY1_WUJI2_HEAD_CAMERA_LINK_NAME,
        pos=RBY1_WUJI2_ZED_MINI_LEFT_POS,
        rot=RBY1_WUJI2_ZED_MINI_QUAT,
    ),
    right=StereoRightEye(
        obs_key="robot_pov_right_cam",
        prim_name="right_cam",
        baseline_m=ZED_MINI_SPEC_BASELINE_M,
        note="Shared ideal ZED Mini baseline for the RBY1 hand variants.",
    ),
    clipping_range=(0.01, 1.0e5),
    update_latest_camera_pose=True,
    default_scale=0.5,
    antialiasing_mode="DLAA",
)


_DEFAULT_RBY1_WUJI2_RIGHT_CAMERA_OFFSET = Pose(
    position_xyz=RBY1_WUJI2_HEAD_RIG.right_mount().pos,
    rotation_xyzw=RBY1_WUJI2_ZED_MINI_QUAT,
)


_get_rby1_wuji2_pinocchio_urdf_path = get_rby1_wuji2_pinocchio_urdf_path


def _set_rby1_wuji2_camera_offset(camera_cfg: CameraCfg, pose: Pose | None) -> None:
    """Apply an RBY1 Wuji2 camera offset through Arena's declared field."""
    if pose is None:
        return
    camera_cfg.offset = CameraCfg.OffsetCfg(
        pos=pose.position_xyz,
        rot=pose.rotation_xyzw,
        convention="ros",
    )


RBY1_WUJI2_UNLIMITED_JOINT_VELOCITY = 1.0e6


def check_retired_gain_environment() -> None:
    """Refuse the DEXBENCH_* actuator variables (``RBY1_WUJI2_RETIRED_GAIN_VARIABLES``), empty ones included."""
    present = [name for name in RBY1_WUJI2_RETIRED_GAIN_VARIABLES if name in os.environ]
    if present:
        raise ValueError(
            f"{', '.join(present)} set, but the RBY1-Wuji2 actuator gains are not read from the "
            "environment: the robot runs one profile, RBY1_WUJI2_CFG.actuators in "
            "embodiments/rby1_wuji2/rby1_wuji2.py. Unset them; to tune "
            "the gains, change that profile, or set scene_config.robot.actuators in a subclass's constructor."
        )


RBY1_WUJI2_FINGERTIP_CONTACT_BODY_NAMES = RBY1_WUJI2_FINGERTIP_PAD_BODY_NAMES + RBY1_WUJI2_FINGERTIP_DISTAL_BODY_NAMES

RBY1_WUJI2_CFG = ArticulationCfg(
    spawn=Rby1Wuji2UsdFileCfg(
        usd_path=str(RBY1_WUJI2_USD_PATH),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=None,
            solver_velocity_iteration_count=None,
        ),
    ),
    prim_path=RBY1_WUJI2_ROBOT_PRIM_PATH,
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos=dict(RBY1_WUJI2_CONTROLLER_HOME_JOINT_POS),
        joint_vel={".*": 0.0},
    ),
    actuators={
        "head": ImplicitActuatorCfg(
            joint_names_expr=list(RBY1_WUJI2_HEAD_JOINT_NAMES),
            effort_limit_sim=50.0,
            velocity_limit_sim=10.0,
            stiffness=300.0,
            damping=110.0,
        ),
        "torso": ImplicitActuatorCfg(
            joint_names_expr=list(RBY1_WUJI2_TORSO_JOINT_NAMES),
            effort_limit=100.0,
            effort_limit_sim=10000.0,
            velocity_limit=100.0,
            velocity_limit_sim=RBY1_WUJI2_UNLIMITED_JOINT_VELOCITY,
            stiffness=800.0,
            damping=200.0,
        ),
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=list(RBY1_WUJI2_RIGHT_ARM_JOINT_NAMES),
            effort_limit=100.0,
            velocity_limit=100.0,
            velocity_limit_sim=RBY1_WUJI2_UNLIMITED_JOINT_VELOCITY,
            stiffness=400.0,
            damping=15.0,
            armature=0.0,
        ),
        "left_arm": ImplicitActuatorCfg(
            joint_names_expr=list(RBY1_WUJI2_LEFT_ARM_JOINT_NAMES),
            effort_limit=100.0,
            velocity_limit=100.0,
            velocity_limit_sim=RBY1_WUJI2_UNLIMITED_JOINT_VELOCITY,
            stiffness=400.0,
            damping=15.0,
            armature=0.0,
        ),
        "right_hand": ImplicitActuatorCfg(
            joint_names_expr=[f"right_{_WUJI2_HAND_JOINT_REGEX}"],
            effort_limit_sim=5.0,
            velocity_limit_sim=12.0,
            stiffness=100.0,
            damping=5.0,
            armature=0.01,
        ),
        "left_hand": ImplicitActuatorCfg(
            joint_names_expr=[f"left_{_WUJI2_HAND_JOINT_REGEX}"],
            effort_limit_sim=5.0,
            velocity_limit_sim=12.0,
            stiffness=100.0,
            damping=5.0,
            armature=0.01,
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=list(RBY1_WUJI2_WHEEL_JOINT_NAMES),
            effort_limit_sim=1.0,
            velocity_limit_sim=0.0,
            stiffness=0.0,
            damping=0.0,
        ),
    },
)


class Rby1Wuji2EmbodimentBase(HandContactMaterialMixin, FingertipContactMixin, JointTorqueMixin, EmbodimentBase):
    """Base RBY1-M Wuji Hand 2 embodiment config shared by control variants."""

    name = "dexbench_rby1_wuji2"
    paper_hand = RBY1_WUJI2_PAPER_HAND
    default_arm_mode = ArmMode.DUAL_ARM
    rerecord_camera_defaults = {"wrist_cameras": True}
    base_model: str | None = RBY1M_BASE_MODEL
    start_poses = RBY1_WUJI2_START_POSES
    collision_spheres = RBY1_WUJI2_COLLISION_SPHERES
    model_hand_contact_material = RBY1_WUJI2_MJCF_HAND_CONTACT_MATERIAL
    fingertip_contact_body_names = RBY1_WUJI2_FINGERTIP_CONTACT_BODY_NAMES

    def configure_fingertip_contact_dim(self, condim: int) -> None:
        """Select soft-pad contact before spawning; every Wuji2 control variant inherits this capability."""
        if condim not in (1, 3, 4, 6):
            raise ValueError(f"Invalid fingertip contact dimension: {condim}")
        self.scene_config.robot.spawn.fingertip_contact_dim = condim

    held_object_contact_body_exprs: dict[str, str] = RBY1_WUJI2_HELD_OBJECT_CONTACT_BODY_EXPRS
    """Per side, a regex of the body names that can touch an object held in that hand."""
    hand_body_label_pattern: str = (
        r"/Robot/(?:.*/)?(?:left|right)_(?:wrist|thumb|index_finger|middle_finger|ring_finger|pinky)(?:_[^/]*)?$"
    )
    """A regex of the Newton body labels of both hands' links, the wrist (palm) and each thumb and finger link:
    what an environment's per-shape hand contact selects (DB-M2's Newton profile)."""
    hand_actuator_groups: tuple[str, ...] = ("left_hand", "right_hand")
    """The robot cfg's actuator groups that drive the fingers."""
    arm_actuator_groups: tuple[str, ...] = ("left_arm", "right_arm")
    """The robot cfg's actuator groups that drive the arms."""
    power_driver_grasp = RBY1_WUJI2_POWER_DRIVER_GRASP
    power_driver_trigger_body_expr = "right_index_finger_.*"
    power_driver_actuator_overrides = {"right_arm": {"damping": 40.},
                                      "torso": {"stiffness": 5000., "damping": 400.}}
    torso_yaw_joint_name: str = RBY1_WUJI2_TORSO_YAW_JOINT_NAME
    """The topmost torso joint, a yaw about the trunk."""

    def configure_power_driver_actuators(self, robot_cfg):
        """Apply the M3 loaded-tool gains to that task's composed robot cfg only."""
        for group, values in self.power_driver_actuator_overrides.items():
            for name, value in values.items():
                setattr(robot_cfg.actuators[group], name, value)

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        concatenate_observation_terms: bool = False,
        arm_mode: ArmMode | None = None,
        camera_scale: float | None = None,
        wrist_cameras: bool = False,
    ):
        check_retired_gain_environment()
        super().__init__(enable_cameras, initial_pose, concatenate_observation_terms, arm_mode)
        if wrist_cameras and not enable_cameras:
            raise ValueError("wrist_cameras requires enable_cameras=True")
        self.wrist_cameras = wrist_cameras
        self.scene_config = Rby1Wuji2SceneCfg()
        cameras = RBY1_WUJI2_HEAD_RIG.build(camera_scale)
        if wrist_cameras:
            for rig in PALM_CAMERA_RIGS:
                cameras.update(rig.build(camera_scale))
        self.camera_config = Rby1Wuji2CameraCfg(**cameras)
        if self.enable_cameras:
            self.add_camera_variations(self.camera_config)
        self.action_config = MISSING
        self.observation_config = Rby1Wuji2ObservationsCfg()
        self.observation_config.policy.concatenate_terms = self.concatenate_observation_terms
        self.event_config = Rby1Wuji2EventCfg()
        if not self.enable_cameras:
            self.event_config.hide_head_camera_body = None
        self.mimic_env = Rby1Wuji2MimicEnv

        self.xr: XrCfg = rby1m_xr_cfg(RBY1_WUJI2_BASE_LINK_PRIM_PATH)

    def add_camera_variations(self, camera_rig: ArenaCameraCfg) -> None:
        """Arena's per-camera extrinsics and intrinsics variations, the extrinsics realized per pose view
        (``MountedCameraExtrinsicsVariation``): the palm cameras sit under the asset's wrist attachment Xform, whose
        frame Arena's writer cannot serve."""
        for camera_name in camera_rig.camera_names():
            self.add_variation(MountedCameraExtrinsicsVariation(camera_name=camera_name))
            self.add_variation(CameraIntrinsicsVariation(camera_name=camera_name, camera_rig=camera_rig))

    def configure_env_recorders(self, env_cfg) -> None:
        """Record the full 62-joint command in demos and Mimic data, named (see recorders.py)."""

        from dexbench_arena.embodiments.rby1_wuji2.recorders import configure_wuji2_recorders

        configure_wuji2_recorders(env_cfg, embodiment=self.name)
        self.configure_fingertip_contact_recorders(env_cfg)
        self.configure_joint_torque_recorders(env_cfg)

    def _configure_physics_backend(self, backend) -> None:
        """Refuse a build on another backend than Newton."""
        if str(backend) != "newton":
            raise ValueError(
                f"{self.name} runs on Newton only, and this build resolved {backend}: pass --presets newton "
                "(environment_builder.presets: NEWTON in a run config)"
            )

    def get_openxr_controller_button_polling(self) -> bool:
        """Hand-tracking-only pipeline: no controller tracker."""

        return False

    def get_ee_frame_name(self, arm_mode: ArmMode) -> str:
        if arm_mode == ArmMode.LEFT:
            return RBY1_WUJI2_LEFT_EEF_LINK_NAME
        return RBY1_WUJI2_RIGHT_EEF_LINK_NAME

    def get_openxr_idle_wrist_poses(self) -> dict[str, tuple[float, ...]]:
        """World-frame single-arm idle wrist targets for the current placement."""

        if self.initial_pose is None:
            return {
                "left": RBY1_WUJI2_LEFT_WRIST_IDLE_ROOT_POSE,
                "right": RBY1_WUJI2_RIGHT_WRIST_IDLE_ROOT_POSE,
            }

        poses = {}
        for side, local in (("left", RBY1_WUJI2_LEFT_WRIST_IDLE_ROOT_POSE), ("right", RBY1_WUJI2_RIGHT_WRIST_IDLE_ROOT_POSE)):
            world = self.initial_pose.multiply(Pose(position_xyz=tuple(local[:3]), rotation_xyzw=tuple(local[3:7])))
            poses[side] = (*world.position_xyz, *world.rotation_xyzw)
        return poses


@register_asset
class DexBenchRby1Wuji2Embodiment(Rby1Wuji2EmbodimentBase):
    """RBY1-M Wuji Hand 2 with Pink IK upper-body control and dexterous hand joint actions."""

    name = "dexbench_rby1_wuji2"
    tags = ["embodiment", "ik"]
    ik_profile_by_source_origin = {"teleop": "teleop", "rrc": "generation", "scripted": "generation"}

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        concatenate_observation_terms: bool = False,
        arm_mode: ArmMode | None = None,
        left_camera_offset: Pose | None = _DEFAULT_RBY1_WUJI2_LEFT_CAMERA_OFFSET,
        right_camera_offset: Pose | None = _DEFAULT_RBY1_WUJI2_RIGHT_CAMERA_OFFSET,
        use_tiled_camera: bool = False,
        ik_profile: str | Rby1mIkProfile = "teleop",
        show_ik_warnings: bool = False,
        waist_ik: bool | None = False,
        waist_ik_max_lead_rad: float | None = 0.05,
        camera_scale: float | None = None,
        wrist_cameras: bool = False,
        head_action: bool | None = False,
    ):
        super().__init__(enable_cameras, initial_pose, concatenate_observation_terms, arm_mode,
                         camera_scale, wrist_cameras)
        self.ik_profile = resolve_ik_profile(ik_profile)
        self.action_config = Rby1Wuji2PinkActionsCfg()
        if self.ik_profile != RBY1M_IK_TELEOP or show_ik_warnings:
            self.action_config.upper_body_ik = rby1_wuji2_pink_ik_action_cfg(self.ik_profile, show_ik_warnings)
        self.waist_ik, self.waist_ik_max_lead_rad = resolve_waist_ik_args(waist_ik, waist_ik_max_lead_rad)
        self.action_config.upper_body_ik.waist_ik_max_lead_rad = self.waist_ik_max_lead_rad
        if self.waist_ik:
            enable_rby1m_waist_ik(self.action_config.upper_body_ik)
        self.action_config.upper_body_ik.controller.urdf_path = str(_get_rby1_wuji2_pinocchio_urdf_path())
        if head_action is None:
            head_action = False
        if not isinstance(head_action, (bool, np.bool_)):
            raise ValueError(f"head_action must be a bool, got {head_action!r}")
        self.head_action = bool(head_action)
        if self.head_action:
            remove_head_from_pink_ik(self.action_config.upper_body_ik)
            self.action_config = Rby1Wuji2PinkHeadActionsCfg(upper_body_ik=self.action_config.upper_body_ik)

        self.camera_config.set_use_tiled_camera(use_tiled_camera)
        _set_rby1_wuji2_camera_offset(self.camera_config.robot_pov_left_cam, left_camera_offset)
        _set_rby1_wuji2_camera_offset(self.camera_config.robot_pov_right_cam, right_camera_offset)


@register_asset
class DexBenchRby1Wuji2JointEmbodiment(Rby1Wuji2EmbodimentBase):
    """RBY1-M Wuji Hand 2 direct joint twin, with an optional saved left grasp."""

    name = RBY1_WUJI2_JOINT_EMBODIMENT
    tags = ["embodiment", "joint"]

    extraction_actuator_overrides = {
        'left_arm': dict(stiffness=3000., damping=100., armature=.01, effort_limit_sim=100., velocity_limit_sim=1e6),
        'right_arm': dict(stiffness=3000., damping=100., armature=.01, effort_limit_sim=100., velocity_limit_sim=1e6),
        'torso': dict(stiffness=800., damping=200., armature=10., effort_limit_sim=1000., velocity_limit_sim=1e6),
        'head': dict(stiffness=1000., damping=200., armature=10., effort_limit_sim=50., velocity_limit_sim=10.),
        'right_hand': dict(stiffness=350., damping=14., armature=.01, effort_limit_sim=3., velocity_limit_sim=12.),
    }

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        concatenate_observation_terms: bool = False,
        arm_mode: ArmMode | None = None,
        left_camera_offset: Pose | None = _DEFAULT_RBY1_WUJI2_LEFT_CAMERA_OFFSET,
        right_camera_offset: Pose | None = _DEFAULT_RBY1_WUJI2_RIGHT_CAMERA_OFFSET,
        use_tiled_camera: bool = True,
        camera_scale: float | None = None,
        wrist_cameras: bool = False,
        support_hand_mode: str = "none",
        extraction_drive_profile: str = "main_default",
    ):
        if support_hand_mode not in ("none", "init_q"):
            raise ValueError(f"Unknown support_hand_mode: {support_hand_mode!r}")
        super().__init__(enable_cameras, initial_pose, concatenate_observation_terms, arm_mode,
                         camera_scale, wrist_cameras)
        self.action_config = Rby1Wuji2JointPositionActionsCfg()
        self.support_hand_mode = support_hand_mode
        if extraction_drive_profile not in ('main_default', 'collection_v1'):
            raise ValueError(f'Unknown extraction drive profile: {extraction_drive_profile}')
        if extraction_drive_profile == 'collection_v1':
            for group, values in self.extraction_actuator_overrides.items():
                for name, value in values.items():
                    setattr(self.scene_config.robot.actuators[group], name, value)
        if support_hand_mode == "init_q":
            self._configure_saved_support_grasp()

        self.camera_config.set_use_tiled_camera(use_tiled_camera)
        _set_rby1_wuji2_camera_offset(self.camera_config.robot_pov_left_cam, left_camera_offset)
        _set_rby1_wuji2_camera_offset(self.camera_config.robot_pov_right_cam, right_camera_offset)

    def _configure_saved_support_grasp(self):
        """Select the collection's init_q action and left-finger drive profile."""
        profile = CompliantGraspCfg(
            anchor_to_reset=True, stiffness=8., damping=.3, effort_limit_nm=.25,
            friction=1.1, rolling_friction_m=.006, preload_rad=.003, travel_rad=.75,
        )
        original = self.action_config.joint_pos
        names = list(RBY1_WUJI2_LEFT_HAND_COMMAND_JOINT_NAMES)
        self.action_config.joint_pos = SavedGraspJointPositionActionCfg(
            asset_name=original.asset_name, joint_names=list(original.joint_names),
            scale=original.scale, use_default_offset=False, preserve_order=True,
            grasp_joint_names=names,
            closing_directions=[0 if n.endswith("mcp_abd") else -1 if n.endswith("thumb_cmc_abd") else 1 for n in names],
            finger_columns=[next(i for i, finger in enumerate(WUJI2_FINGER_ORDER)
                                 if n.startswith(f"left_{finger}_")) for n in names],
            grasp_profile=profile, adaptive_profile=AdaptiveGraspCfg(),
        )
        drive = self.scene_config.robot.actuators["left_hand"]
        drive.stiffness = profile.stiffness
        drive.damping = profile.damping
        drive.effort_limit_sim = profile.effort_limit_nm
        self.scene_config.robot.spawn.articulation_props.enabled_self_collisions = True
        self.scene_config.robot.spawn.bimanual_collision_filter = True
        self.support_hand_mode = "init_q"

    support_grasp_reference_body = RBY1_WUJI2_LEFT_EEF_LINK_NAME
    support_grasp_finger_body_patterns = tuple(f"left_{finger}_.*" for finger in WUJI2_FINGER_ORDER)
    support_grasp_body_pattern = r"/Robot/(?:.*/)?left_(?:wrist|thumb|index_finger|middle_finger|ring_finger|pinky)(?:_[^/]*)?$"
    support_grasp_shape_pattern = support_grasp_body_pattern.removesuffix("$") + "/"

    eef_arm_ik_layout = ArmIkLayout(
        arm_joint_names=(*RBY1_WUJI2_LEFT_ARM_JOINT_NAMES, *RBY1_WUJI2_RIGHT_ARM_JOINT_NAMES),
        held_joint_names=(),
        passthrough_joint_names=(*RBY1_WUJI2_HAND_JOINT_ACTION_NAMES, *RBY1_WUJI2_TORSO_JOINT_NAMES,
                                 *RBY1_WUJI2_HEAD_JOINT_NAMES),
        wrist_frames={"left": RBY1_WUJI2_LEFT_WRIST_CONTROL_LINK_NAME, "right": RBY1_WUJI2_RIGHT_WRIST_CONTROL_LINK_NAME},
    )
    """What a wrist-pose policy drives on this embodiment (:meth:`eef_arm_ik_controller`): the 14 arm joints are
    solved; the 40 hand joints, the torso (6) and the neck (2) take the policy's own targets, in that order; every
    column of ``joint_pos``."""

    @classmethod
    def eef_arm_ik_controller(cls, env, ik_profile: str = "generation") -> EefArmIkController:
        """Wrist-pose targets to this embodiment's 62-D joint action, for a policy that commands the MANO wrist frames
        the Pink embodiment (``dexbench_rby1_wuji2``) controls (a VLM goal policy; ``embodiments/eef_arm_ik_controller.py``)."""
        profile = resolve_ik_profile(ik_profile)
        action_cfg = rby1_wuji2_pink_ik_action_cfg(profile)
        remove_head_from_pink_ik(action_cfg)
        layout = cls.eef_arm_ik_layout
        if set(action_cfg.pink_controlled_joint_names) != set(layout.arm_joint_names):
            raise ValueError("the Pink action cfg without the head is not the layout's arm joints")
        robot = env.scene[action_cfg.asset_name]
        names = list(robot.joint_names)
        controller_cfg = action_cfg.controller
        controller_cfg.urdf_path = str(_get_rby1_wuji2_pinocchio_urdf_path())
        controller_cfg.joint_names = list(layout.arm_joint_names)
        controller_cfg.all_joint_names = names
        arm_ids = [names.index(name) for name in layout.arm_joint_names]
        solvers = [
            PinkIKController(cfg=controller_cfg.copy(), robot_cfg=env.scene.cfg.robot, device="cpu",
                             controlled_joint_indices=arm_ids)
            for _ in range(env.num_envs)
        ]
        settings = ArmIkSettings(layout=layout, ik_profile=ik_profile if isinstance(ik_profile, str) else "custom",
                                 profile=asdict(profile), step_dt=float(env.step_dt))
        term = env.action_manager.get_term("joint_pos")
        return EefArmIkController(settings, solvers, names, term, as_torch(robot.data.soft_joint_pos_limits))


@configclass
class Rby1Wuji2SceneCfg:
    """Robot-only scene contribution for RBY1-M Wuji Hand 2."""

    robot: ArticulationCfg = RBY1_WUJI2_CFG.copy()
    fingertip_contact: ContactSensorCfg | None = fingertip_contact_sensor_cfg(
        RBY1_WUJI2_ROBOT_PRIM_PATH, RBY1_WUJI2_FINGERTIP_CONTACT_BODY_NAMES
    )


def get_rby1_wuji2_joint_state(env, joint_names: list[str]) -> torch.Tensor:
    """Joint positions in the requested order on both PhysX and Newton (cached)."""

    return get_joint_state(env, joint_names)


@configclass
class Rby1Wuji2JointPositionActionsCfg:
    """Direct joint-position action over arms/head, the 40 v2 hand joints, and the torso."""

    joint_pos = JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(RBY1_WUJI2_JOINT_ACTION_NAMES),
        scale=1.0,
        use_default_offset=False,
        preserve_order=True,
    )


_WRIST_CONTROL_FRAMES = {
    "left": RBY1_WUJI2_LEFT_WRIST_CONTROL_LINK_NAME,
    "right": RBY1_WUJI2_RIGHT_WRIST_CONTROL_LINK_NAME,
}


def _pink_joint_indices(env) -> list[int]:
    """Robot joint indices in the Pinocchio model's joint order, resolved once per env."""
    cached = getattr(env, "_pink_joint_index_cache", None)
    if cached is None:
        robot_joints = env.scene["robot"].data.joint_names
        cached = [robot_joints.index(name) for name in _rby1_wuji2_pink_kinematics()._all_joint_names]
        env._pink_joint_index_cache = cached
    return cached


def rby1_wuji2_wrist_pose(env, eef_name: str, joint_pos: torch.Tensor) -> torch.Tensor:
    """Pink FK of the ``eef_name`` (``left``/``right``) MANO control frame for given robot joint positions (all
    envs), composed with the measured base-body pose, in env-local world coordinates: the frame Pink IK controls,
    in the coordinates its action takes."""
    robot = env.scene["robot"]
    config = _rby1_wuji2_pink_kinematics()
    frame = _WRIST_CONTROL_FRAMES[eef_name]
    order = _pink_joint_indices(env)

    positions, rotations = [], []
    for env_index in range(joint_pos.shape[0]):
        config.update(joint_pos[env_index, order].detach().cpu().numpy().astype(np.float64))
        pose = config.get_transform_frame_to_world(frame)
        positions.append(np.asarray(pose.translation))
        rotations.append(np.asarray(pose.rotation))

    device, dtype = joint_pos.device, joint_pos.dtype
    frame_pos = torch.tensor(np.stack(positions), device=device, dtype=dtype)
    frame_rot = torch.tensor(np.stack(rotations), device=device, dtype=dtype)

    base_index = robot.data.body_names.index(RBY1_WUJI2_BASE_LINK_NAME)
    base_pos = wp.to_torch(robot.data.body_pos_w)[:, base_index] - env.scene.env_origins
    base_quat = wp.to_torch(robot.data.body_quat_w)[:, base_index]
    base_rot = PoseUtils.matrix_from_quat(base_quat)

    world_pos = base_pos + torch.einsum("bij,bj->bi", base_rot, frame_pos)
    world_rot = torch.bmm(base_rot, frame_rot)

    return PoseUtils.make_pose(world_pos, world_rot)


def rby1_wuji2_recorded_wrist_poses(eef_name: str, joint_names: Sequence[str], joint_pos: torch.Tensor,
                                    root_pose: torch.Tensor) -> torch.Tensor:
    """``rby1_wuji2_wrist_pose`` of a recording, without an env: (T, 4, 4) the ``eef_name`` MANO control frame for
    recorded joint positions (T, J) named ``joint_names`` (the recording's ``state_joint_names``) and the robot's
    recorded root pose (T, 7; position, (x, y, z, w)), which is its base body's. On a hold-pose recording it equals
    the commanded wrist pose to 1e-7."""
    config = _rby1_wuji2_pink_kinematics()
    frame = _WRIST_CONTROL_FRAMES[eef_name]
    names = list(joint_names)
    order = [names.index(name) for name in config._all_joint_names]
    positions, rotations = [], []
    for row in joint_pos[:, order].detach().cpu().numpy().astype(np.float64):
        config.update(row)
        pose = config.get_transform_frame_to_world(frame)
        positions.append(np.asarray(pose.translation))
        rotations.append(np.asarray(pose.rotation))
    device, dtype = joint_pos.device, joint_pos.dtype
    local = PoseUtils.make_pose(torch.tensor(np.stack(positions), device=device, dtype=dtype),
                                torch.tensor(np.stack(rotations), device=device, dtype=dtype))
    base = PoseUtils.make_pose(root_pose[:, :3].to(dtype), PoseUtils.matrix_from_quat(root_pose[:, 3:7].to(dtype)))
    return base @ local


class Rby1Wuji2PinkInverseKinematicsAction(Rby1mPinkInverseKinematicsAction):
    """Pink action with a backend-independent Wuji Hand 2 action order."""

    def _initialize_joint_info(self) -> None:
        """Keep Pink joints native, but resolve the hand in the public 40D order."""
        self._isaaclab_controlled_joint_ids, self._isaaclab_controlled_joint_names = self._asset.find_joints(
            self.cfg.pink_controlled_joint_names
        )
        self.cfg.controller.joint_names = self._isaaclab_controlled_joint_names
        self._isaaclab_all_joint_ids = list(range(len(self._asset.data.joint_names)))
        self.cfg.controller.all_joint_names = self._asset.data.joint_names

        self._hand_joint_ids, self._hand_joint_names = resolve_joint_order(
            self._asset, self.cfg.hand_joint_names
        )

        self._controlled_joint_ids = self._isaaclab_controlled_joint_ids + self._hand_joint_ids
        self._controlled_joint_names = self._isaaclab_controlled_joint_names + self._hand_joint_names

    def hold_action(self, joint_pos: torch.Tensor) -> torch.Tensor:
        """The action that holds the robot at ``joint_pos`` (every robot joint, all envs): both wrist frames at
        their pose there (:func:`rby1_wuji2_wrist_pose`) and the hands at their joints, in the action's layout
        (left position and quaternion, right, then the 40 hand joints), as Mimic's ``target_eef_pose_to_action``
        builds it. ``HoldPosePolicy`` holds this embodiment's reset pose with it."""
        wrists = []
        for side in ("left", "right"):
            position, rotation = PoseUtils.unmake_pose(rby1_wuji2_wrist_pose(self._env, side, joint_pos))
            wrists += [position, PoseUtils.quat_from_matrix(rotation)]
        return torch.cat((*wrists, joint_pos[:, self._hand_joint_ids].to(wrists[0].dtype)), dim=1)


    clean_processed_actions: torch.Tensor | None = None
    """Pink's joint targets for the clean action, from the state of this step's last solve; ``None`` outside Mimic generation."""
    _clean_actions: torch.Tensor | None = None
    _tracks_clean: bool = False

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        self._take_clean_actions(actions)

    def _take_clean_actions(self, actions: torch.Tensor) -> None:
        """Note this step's clean action for the term's columns, when a Mimic env stashed one that differs."""
        self.clean_processed_actions = None
        self._clean_actions = None
        self._tracks_clean = False
        clean = clean_action_batch(self._env, self._env.action_manager.action)
        if clean is None:
            return
        self._tracks_clean = True
        start = self._action_offset()
        clean = clean[:, start : start + self.action_dim]
        if not torch.equal(clean, actions):
            self._clean_actions = clean.clone()

    def apply_actions(self) -> None:
        clean_solution = self._solve_for(self._clean_actions) if self._clean_actions is not None else None
        super().apply_actions()
        if not self._tracks_clean:
            return
        if clean_solution is None:
            self.clean_processed_actions = self._processed_actions.clone()
        else:
            self.clean_processed_actions = torch.cat((clean_solution, self._clean_actions[:, -self.hand_joint_dim :]), dim=1)

    def _action_offset(self) -> int:
        """Where this term's columns start in the env's action, in the action manager's term order."""
        manager = self._env.action_manager
        offset = 0
        for name in manager.active_terms:
            term = manager.get_term(name)
            if term is self:
                return offset
            offset += term.action_dim
        raise RuntimeError("the Pink action term is not one of the action manager's terms")

    def _solve_for(self, actions: torch.Tensor) -> torch.Tensor:
        """The IK solution for ``actions`` from the current state, leaving the term as it was."""
        assert hasattr(self, "_waist_ik_previous_target"), "the waist limiter's state moved; update _solve_for"
        waist = {name: getattr(self, name) for name in ("_waist_ik_previous_target", "_waist_ik_reset_mask")
                 if hasattr(self, name)}
        waist = {name: value.clone() if isinstance(value, torch.Tensor) else value for name, value in waist.items()}
        self._set_task_targets(self._transform_poses_to_base_link_frame(self._extract_controlled_frame_poses(actions)))
        try:
            return self._compute_ik_solutions()
        finally:
            for name, value in waist.items():
                setattr(self, name, value)
            self._set_task_targets(
                self._transform_poses_to_base_link_frame(self._extract_controlled_frame_poses(self._raw_actions))
            )


@configclass
class Rby1Wuji2PinkInverseKinematicsActionCfg(Rby1mPinkInverseKinematicsActionCfg):
    """RBY1 Wuji Hand 2 Pink action configuration with canonical hand-joint ordering."""

    class_type: type = Rby1Wuji2PinkInverseKinematicsAction


def rby1_wuji2_pink_ik_action_cfg(
    profile: Rby1mIkProfile, show_ik_warnings: bool = False
) -> Rby1Wuji2PinkInverseKinematicsActionCfg:
    """The Pink IK action term with ``profile``'s task costs and gains on both wrist frames."""
    return Rby1Wuji2PinkInverseKinematicsActionCfg(
        pink_controlled_joint_names=list(RBY1_WUJI2_PINK_CONTROLLED_JOINT_NAMES),
        hand_joint_names=list(RBY1_WUJI2_HAND_JOINT_ACTION_NAMES),
        target_eef_link_names={
            "left_wrist": RBY1_WUJI2_LEFT_WRIST_CONTROL_LINK_NAME,
            "right_wrist": RBY1_WUJI2_RIGHT_WRIST_CONTROL_LINK_NAME,
        },
        asset_name="robot",
        controller=PinkIKControllerCfg(
            articulation_name="robot",
            base_link_name=RBY1_WUJI2_BASE_LINK_NAME,
            num_hand_joints=len(RBY1_WUJI2_HAND_JOINT_ACTION_NAMES),
            show_ik_warnings=show_ik_warnings,
            fail_on_joint_limit_violation=False,
            variable_input_tasks=[
                FrameTaskCfg(
                    frame=RBY1_WUJI2_LEFT_WRIST_CONTROL_LINK_NAME,
                    position_cost=profile.position_cost,
                    orientation_cost=profile.orientation_cost,
                    lm_damping=profile.lm_damping,
                    gain=profile.gain,
                ),
                FrameTaskCfg(
                    frame=RBY1_WUJI2_RIGHT_WRIST_CONTROL_LINK_NAME,
                    position_cost=profile.position_cost,
                    orientation_cost=profile.orientation_cost,
                    lm_damping=profile.lm_damping,
                    gain=profile.gain,
                ),
                DampingTaskCfg(cost=profile.damping_cost),
                NullSpacePostureTaskCfg(
                    cost=profile.posture_cost,
                    lm_damping=1.0,
                    controlled_frames=[
                        RBY1_WUJI2_LEFT_WRIST_CONTROL_LINK_NAME,
                        RBY1_WUJI2_RIGHT_WRIST_CONTROL_LINK_NAME,
                    ],
                    controlled_joints=list(RBY1_WUJI2_NULL_SPACE_JOINT_NAMES),
                ),
            ],
            fixed_input_tasks=[],
            xr_enabled=xr_enabled(),
        ),
    )


def remove_head_from_pink_ik(upper_body_ik) -> None:
    """Take the head joints out of the IK term, the way Arena's G1 ``lock_waist`` takes the waist."""
    upper_body_ik.pink_controlled_joint_names = [
        n for n in upper_body_ik.pink_controlled_joint_names if n not in RBY1_WUJI2_HEAD_JOINT_NAMES
    ]


@configclass
class Rby1Wuji2PinkActionsCfg:
    """Pink IK action config matching the RBY1-M Wuji Hand 2 OpenXR action tensor layout."""

    upper_body_ik: Rby1Wuji2PinkInverseKinematicsActionCfg = rby1_wuji2_pink_ik_action_cfg(RBY1M_IK_TELEOP)


@configclass
class Rby1Wuji2PinkHeadActionsCfg(Rby1Wuji2PinkActionsCfg):
    """``dexbench_rby1_wuji2 head_action=True``: the Pink IK term (wrists 14, fingers 40), then the head (2)."""

    head = JointPositionActionCfg(
        asset_name="robot", joint_names=list(RBY1_WUJI2_HEAD_JOINT_NAMES),
        scale=1.0, use_default_offset=False, preserve_order=True,
    )


@configclass
class Rby1Wuji2CameraCfg(ArenaCameraCfg):
    """Head stereo plus optional GA palm cameras from the shared catalog."""

    robot_pov_left_cam: CameraCfg = field(
        default_factory=lambda: RBY1_WUJI2_HEAD_RIG.build()["robot_pov_left_cam"]
    )
    robot_pov_right_cam: CameraCfg = field(
        default_factory=lambda: RBY1_WUJI2_HEAD_RIG.build()["robot_pov_right_cam"]
    )
    wrist_left_cam: CameraCfg | None = None
    wrist_right_cam: CameraCfg | None = None


@configclass
class Rby1Wuji2ObservationsCfg:
    """Robot-only observation config for composition with Arena tasks/scenes."""

    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action)
        robot_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        robot_joint_vel = ObsTerm(
            func=base_mdp.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        robot_root_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_root_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_links_state = ObsTerm(func=mdp.get_all_robot_link_state)

        left_eef_pos = ObsTerm(func=mdp.get_eef_pos, params={"link_name": RBY1_WUJI2_LEFT_EEF_LINK_NAME})
        left_eef_quat = ObsTerm(func=mdp.get_eef_quat, params={"link_name": RBY1_WUJI2_LEFT_EEF_LINK_NAME})
        right_eef_pos = ObsTerm(func=mdp.get_eef_pos, params={"link_name": RBY1_WUJI2_RIGHT_EEF_LINK_NAME})
        right_eef_quat = ObsTerm(func=mdp.get_eef_quat, params={"link_name": RBY1_WUJI2_RIGHT_EEF_LINK_NAME})

        hand_joint_state = ObsTerm(
            func=get_rby1_wuji2_joint_state,
            params={"joint_names": list(RBY1_WUJI2_HAND_JOINT_ACTION_NAMES)},
        )
        head_joint_state = ObsTerm(
            func=get_rby1_wuji2_joint_state,
            params={"joint_names": list(RBY1_WUJI2_HEAD_JOINT_NAMES)},
        )
        fingertip_contact_force = ObsTerm(
            func=fingertip_contact_force,
            params={
                "sensor_cfg": SceneEntityCfg(
                    FINGERTIP_CONTACT_SENSOR_NAME,
                    body_names=list(RBY1_WUJI2_FINGERTIP_PAD_BODY_NAMES),
                    preserve_order=True,
                )
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


RBY1_WUJI2_CAMERA_BODY_MESH_RELATIVE_PATHS = (
    "ego_zed_m_camera/visuals/geom_0",
)


@deterministic_reset
def hide_rby1_wuji2_camera_body_mesh(env, env_ids=None):
    """Hide the ZED Mini housing mesh that otherwise occludes its camera pair."""

    del env_ids
    resolved_prims = []
    missing_paths = []
    for relative_path in RBY1_WUJI2_CAMERA_BODY_MESH_RELATIVE_PATHS:
        mesh_regex = f"{env.scene.env_regex_ns}/Robot/{relative_path}"
        matches = sim_utils.find_matching_prims(mesh_regex)
        if matches:
            resolved_prims.extend(matches)
        else:
            missing_paths.append(mesh_regex)

    if missing_paths:
        raise RuntimeError(f"RBY1 Wuji v2 ZED Mini housing meshes were not found: {missing_paths}")

    hidden_paths = []
    failures = []
    for prim in resolved_prims:
        try:
            sim_utils.set_prim_visibility(prim, False)
            hidden_paths.append(str(prim.GetPath()))
        except Exception as exc:  # noqa: BLE001 -- cloned instance proxies may be read-only
            failures.append((str(prim.GetPath()), repr(exc)))

    if not hidden_paths:
        raise RuntimeError(f"Failed to hide every matched RBY1 Wuji v2 ZED Mini housing mesh: {failures}")
    if failures:
        print(f"[rby1_wuji2] ignored read-only camera housing proxy prim(s): {failures}")
    print(f"[rby1_wuji2] hid ZED Mini housing mesh prim(s): {hidden_paths}")


@configclass
class Rby1Wuji2EventCfg:
    """Robot-only event config for RBY1-M Wuji Hand 2 (target-pinning reset)."""

    reset_all = EventTerm(func=reset_all_articulation_joints_and_targets, mode="reset")
    hide_head_camera_body = EventTerm(func=hide_rby1_wuji2_camera_body_mesh, mode="startup")
    hand_contact_recording_metadata = EventTerm(func=install_hand_contact_recording_metadata, mode=RECORDER_READY_MODE)
    hand_contact_material_check = EventTerm(
        func=check_hand_contact_material,
        mode="startup",
        params={
            **asdict(WUJI_GRASP_CONTACT_MATERIAL),
            "source": "embodiment",
            "material_name": HAND_CONTACT_MATERIAL_NAME,
            "hand_bodies": (RBY1_WUJI2_LEFT_EEF_LINK_NAME, RBY1_WUJI2_RIGHT_EEF_LINK_NAME),
        },
    )


@functools.lru_cache(maxsize=1)
def _rby1_wuji2_pink_kinematics():
    """Pinocchio model built from the URDF Pink IK itself loads (see the v1 twin)."""
    return PinkKinematicsConfiguration(
        controlled_joint_names=list(RBY1_WUJI2_PINK_CONTROLLED_JOINT_NAMES),
        urdf_path=str(_get_rby1_wuji2_pinocchio_urdf_path()),
    )


class Rby1Wuji2MimicEnv(ReplaysSubtaskBoundaries, IsaacLabArenaManagerBasedRLMimicEnv):
    """Mimic helpers for RBY1-M Wuji Hand 2 Pink IK actions (v1 twin, v2 names)."""

    validate_mimic_source_dataset = staticmethod(validate_mimic_source_dataset)

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Pose of the MANO frame Pink IK controls, read from Pink's own model."""
        if env_ids is None:
            env_ids = slice(None)

        robot = self.scene["robot"]
        joint_pos = as_torch(robot.data.joint_pos)
        return self._eef_pose_from_joint_positions(eef_name, joint_pos)[env_ids]

    def _eef_pose_from_joint_positions(self, eef_name: str, joint_pos: torch.Tensor) -> torch.Tensor:
        """Pink FK of the MANO control frame for given robot joint positions (:func:`rby1_wuji2_wrist_pose`)."""
        return rby1_wuji2_wrist_pose(self, eef_name, joint_pos)

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """The noisy action Mimic executes; the clean one is kept for the clean labels."""
        clean, played = self.clean_and_played_action(target_eef_pose_dict, gripper_action_dict, action_noise_dict, env_id)
        stash_clean_action(self, env_id, clean, played)
        return played

    def clean_and_played_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(clean, played)``: the action for the targets, and the same with Mimic's noise on the wrist poses."""
        target_eef_pose_dict = dict(target_eef_pose_dict)
        gripper_action_dict = dict(gripper_action_dict)
        for missing in ("left", "right"):
            if missing not in target_eef_pose_dict:
                target_eef_pose_dict[missing] = self.get_robot_eef_pose(missing, env_ids=[env_id])[0]
            if missing not in gripper_action_dict:
                hand = get_rby1_wuji2_joint_state(self, list(RBY1_WUJI2_HAND_JOINT_ACTION_NAMES))
                current_hands = split_bimanual_hand_actions(
                    hand,
                    RBY1_WUJI2_HAND_JOINT_ACTION_NAMES,
                    0,
                )
                gripper_action_dict[missing] = current_hands[missing][env_id]
        if action_noise_dict is not None:
            action_noise_dict = dict(action_noise_dict)
            for side in ("left", "right"):
                action_noise_dict.setdefault(side, 0.0)

        poses = {}
        for side in ("left", "right"):
            position, rotation = PoseUtils.unmake_pose(target_eef_pose_dict[side])
            poses[side] = (position, PoseUtils.quat_from_matrix(rotation))
        hands = merge_bimanual_hand_actions(
            gripper_action_dict["left"],
            gripper_action_dict["right"],
            RBY1_WUJI2_HAND_JOINT_ACTION_NAMES,
        )

        def action(wrists: dict) -> torch.Tensor:
            return torch.cat((*wrists["left"], *wrists["right"], hands), dim=0)

        clean = action(poses)
        if action_noise_dict is None:
            return clean, clean
        noisy = {
            side: (
                position + action_noise_dict[side] * torch.randn_like(position),
                quat + action_noise_dict[side] * torch.randn_like(quat),
            )
            for side, (position, quat) in poses.items()
        }
        return clean, action(noisy)

    def _refuse_head_action(self) -> None:
        """Refuse an env built with ``head_action``: its 56 columns are IK 54 then head 2, the width of the
        joint-space layout the two readers below tell apart by width, so they would read wrist poses as
        joint targets and the head as finger joints without an error."""
        if "head" in self.action_manager.active_terms:
            raise ValueError(
                "Mimic annotation and generation do not read dexbench_rby1_wuji2 head_action actions (wrists 14, "
                "fingers 40, head 2): drop --head_action, or annotate the recording a head_action replay was made from"
            )

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        self._refuse_head_action()
        if action.shape[-1] in RBY1_WUJI2_JOINT_ACTION_DIMS:
            names = (
                RBY1_WUJI2_JOINT_ACTION_NAMES
                if action.shape[-1] == len(RBY1_WUJI2_JOINT_ACTION_NAMES)
                else [*RBY1_WUJI2_PINK_CONTROLLED_JOINT_NAMES, *RBY1_WUJI2_HAND_JOINT_ACTION_NAMES]
            )
            robot = self.scene["robot"]
            joint_pos = as_torch(robot.data.joint_pos)
            commanded = joint_pos.clone()
            index = resolve_joint_index(robot, list(names), commanded.device)
            commanded[:, index] = action.to(commanded.device, commanded.dtype)
            return {side: self._eef_pose_from_joint_positions(side, commanded) for side in ("left", "right")}

        target_poses = {}

        target_left_wrist_position = action[:, 0:3]
        target_left_rot_mat = PoseUtils.matrix_from_quat(action[:, 3:7])
        target_poses["left"] = PoseUtils.make_pose(target_left_wrist_position, target_left_rot_mat)

        target_right_wrist_position = action[:, 7:10]
        target_right_rot_mat = PoseUtils.matrix_from_quat(action[:, 10:14])
        target_poses["right"] = PoseUtils.make_pose(target_right_wrist_position, target_right_rot_mat)

        return target_poses

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        self._refuse_head_action()
        if actions.shape[-1] in RBY1_WUJI2_JOINT_ACTION_DIMS:
            arm_dof = len(RBY1_WUJI2_PINK_CONTROLLED_JOINT_NAMES)
            return split_bimanual_hand_actions(actions, RBY1_WUJI2_HAND_JOINT_ACTION_NAMES, arm_dof)
        return split_bimanual_hand_actions(actions, RBY1_WUJI2_HAND_JOINT_ACTION_NAMES, 14)

    def get_object_poses(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)

        state = self.scene.get_state(is_relative=True)
        return get_rigid_and_articulated_object_poses(state, env_ids)
