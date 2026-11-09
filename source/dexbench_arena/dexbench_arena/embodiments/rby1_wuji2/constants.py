# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""RBY1-M + Wuji Hand 2 (wujiv2) constants for DexBench."""

from __future__ import annotations

from dexbench_arena import ASSETS_DIR, RBY1_WUJI2_BETA2_ASSETS_DIR
from dexbench_arena.contact_materials import HandContactMaterialCfg
from dexbench_arena.deformables.hand_spec import PaperHandSpec
from dexbench_arena.embodiments.common.collision_spheres import CollisionSphereModel

from dexbench_arena.embodiments.rby1m.constants import (
    RBY1M_DESK_READY_JOINT_POS,
    RBY1M_HEAD_JOINT_NAMES,
    RBY1M_LEFT_ARM_JOINT_NAMES,
    RBY1M_M4_SIDE_PICK_JOINT_POS,
    RBY1M_M5_KITTING_JOINT_POS,
    RBY1M_NULL_SPACE_JOINT_NAMES,
    RBY1M_REPLAY_HOME_JOINT_POS,
    RBY1M_RETAIL_READY_JOINT_POS,
    RBY1M_RIGHT_ARM_JOINT_NAMES,
    RBY1M_CONTROLLER_HOME_JOINT_POS,
    RBY1M_CONTROLLER_READY_JOINT_POS,
    RBY1M_TELEOP_HOME_JOINT_POS,
    RBY1M_TELEOP_READY_JOINT_POS,
    RBY1M_TORSO_JOINT_NAMES,
    RBY1M_TORSO_YAW_JOINT_NAME,
    RBY1M_WHEEL_JOINT_NAMES,
    RBY1M_ZED_MINI_BASELINE_M,
    RBY1M_ZED_MINI_LEFT_POS,
    RBY1M_ZED_MINI_QUAT,
)

RBY1_WUJI2_JOINT_EMBODIMENT = "dexbench_rby1_wuji2_joint"
"""The joint-position embodiment's registered name, which is also its robot asset's name: the prefix of the
robot's Arena variation keys (``<name>.base_pose``, ``<name>.camera_extrinsics_<camera>``)."""

RBY1_WUJI2_MJCF_HAND_CONTACT_MATERIAL = HandContactMaterialCfg(1.0, 1.0, "max")

RBY1_WUJI2_USD_PATH = ASSETS_DIR / "embodiments" / "simready" / "rby1m_wujihand2_beta2" / "v0.5.0" / "rby1m_wujihand2_beta2.usd"
RBY1_WUJI2_COLLISION_SPHERES = CollisionSphereModel(
    path=RBY1_WUJI2_USD_PATH.parent / "collision_spheres/right_axial_pinch.json",
    sha256="b0b43afc97ceeb5d6f4ca0e77a88701e8b2615a1dcdff2f373b1cdbd49cc5af7",
    base_box_m={"min": (-0.3533, -0.3036, 0.0142), "max": (0.3493, 0.3036, 0.3754)},
)
RBY1_WUJI2_MOUNT_COLLISION_SPHERES = CollisionSphereModel(
    path=RBY1_WUJI2_USD_PATH.parent / "collision_spheres/wrist_mounts.json",
    sha256="28d5cf64caaae95ec23866984b011953956137c64e4df4fde2f723b60c981d43",
)
RBY1_WUJI2_URDF_PATH = RBY1_WUJI2_BETA2_ASSETS_DIR / "rby1m_wujihand2_beta2.urdf"
RBY1_WUJI2_MJCF_PATH = RBY1_WUJI2_BETA2_ASSETS_DIR / "rby1m_wujihand2_beta2.xml"
RBY1_WUJI2_MESH_DIR = RBY1_WUJI2_BETA2_ASSETS_DIR
RBY1_WUJI2_LEFT_RETARGET_URDF_PATH = RBY1_WUJI2_BETA2_ASSETS_DIR / "urdf" / "wujihand2_left_retarget.urdf"
RBY1_WUJI2_RIGHT_RETARGET_URDF_PATH = RBY1_WUJI2_BETA2_ASSETS_DIR / "urdf" / "wujihand2_right_retarget.urdf"

RBY1_WUJI2_ROBOT_PRIM_PATH = "{ENV_REGEX_NS}/Robot"
RBY1_WUJI2_ENV0_ROBOT_PRIM_PATH = "/World/envs/env_0/Robot"

RBY1_WUJI2_BASE_LINK_NAME = "base"
RBY1_WUJI2_BASE_LINK_PRIM_PATH = f"{RBY1_WUJI2_ENV0_ROBOT_PRIM_PATH}/{RBY1_WUJI2_BASE_LINK_NAME}"

RBY1_WUJI2_LEFT_EEF_LINK_NAME = "left_wrist"
RBY1_WUJI2_RIGHT_EEF_LINK_NAME = "right_wrist"

RBY1_WUJI2_LEFT_WRIST_CONTROL_LINK_NAME = "left_wrist_frame"
RBY1_WUJI2_RIGHT_WRIST_CONTROL_LINK_NAME = "right_wrist_frame"

RBY1_WUJI2_XR_WRIST_OFFSET_RPY_LEFT = (0.0, 90.0, 0.0)
RBY1_WUJI2_XR_WRIST_OFFSET_RPY_RIGHT = (0.0, -90.0, 0.0)

RBY1_WUJI2_OPERATOR_TO_MANO_LEFT = (0, 0, 1, 0, 1, 0, -1, 0, 0)
RBY1_WUJI2_OPERATOR_TO_MANO_RIGHT = (0, 0, -1, 0, 1, 0, 1, 0, 0)

RBY1_WUJI2_HEAD_CAMERA_LINK_NAME = "link_head_2"


WUJI2_FINGER_ORDER = ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky")
_THUMB_STAGES = ("cmc_flex", "cmc_abd", "mcp", "ip")
_FINGER_STAGES = ("mcp_flex", "mcp_abd", "pip", "dip")
WUJI2_FINGER_STAGE_SUFFIXES = {
    "thumb": _THUMB_STAGES,
    "index_finger": _FINGER_STAGES,
    "middle_finger": _FINGER_STAGES,
    "ring_finger": _FINGER_STAGES,
    "pinky": _FINGER_STAGES,
}

WUJI2_LOCAL_HAND_JOINT_NAMES = tuple(
    f"{finger}_{suffix}"
    for finger in WUJI2_FINGER_ORDER
    for suffix in WUJI2_FINGER_STAGE_SUFFIXES[finger]
)

RBY1_WUJI2_RIGHT_HAND_COMMAND_JOINT_NAMES = tuple(
    f"right_{name}" for name in WUJI2_LOCAL_HAND_JOINT_NAMES
)
RBY1_WUJI2_LEFT_HAND_COMMAND_JOINT_NAMES = tuple(
    f"left_{name}" for name in WUJI2_LOCAL_HAND_JOINT_NAMES
)

RBY1_WUJI2_FINGERTIP_PAD_BODY_NAMES = tuple(
    f"{side}_{finger}_tip_sensor_frame" for side in ("left", "right") for finger in WUJI2_FINGER_ORDER
)
RBY1_WUJI2_FINGERTIP_DISTAL_BODY_NAMES = tuple(
    f"{side}_{finger}_distal" for side in ("left", "right") for finger in WUJI2_FINGER_ORDER
)

RBY1_WUJI2_HAND_JOINT_ACTION_NAMES = tuple(
    f"{side}_{finger}_{WUJI2_FINGER_STAGE_SUFFIXES[finger][stage]}"
    for stage in range(4)
    for side in ("left", "right")
    for finger in WUJI2_FINGER_ORDER
)

RBY1_WUJI2_LEFT_ARM_JOINT_NAMES = RBY1M_LEFT_ARM_JOINT_NAMES
RBY1_WUJI2_RIGHT_ARM_JOINT_NAMES = RBY1M_RIGHT_ARM_JOINT_NAMES
RBY1_WUJI2_HEAD_JOINT_NAMES = RBY1M_HEAD_JOINT_NAMES
RBY1_WUJI2_TORSO_JOINT_NAMES = RBY1M_TORSO_JOINT_NAMES
RBY1_WUJI2_TORSO_YAW_JOINT_NAME = RBY1M_TORSO_YAW_JOINT_NAME

RBY1_WUJI2_HELD_OBJECT_CONTACT_BODY_EXPRS = {
    "left": "link_left_arm_[3-6]|left_.*",
    "right": "link_right_arm_[3-6]|right_.*",
}
RBY1_WUJI2_WHEEL_JOINT_NAMES = RBY1M_WHEEL_JOINT_NAMES

RBY1_WUJI2_PINK_CONTROLLED_JOINT_NAMES = (
    *RBY1_WUJI2_LEFT_ARM_JOINT_NAMES,
    *RBY1_WUJI2_RIGHT_ARM_JOINT_NAMES,
    *RBY1_WUJI2_HEAD_JOINT_NAMES,
)

RBY1_WUJI2_NULL_SPACE_JOINT_NAMES = RBY1M_NULL_SPACE_JOINT_NAMES

RBY1_WUJI2_LEFT_WRIST_IDLE_ROOT_POSE = (
    0.4416,
    0.310724,
    0.998931,
    0.707107,
    -0.000002,
    0.000001,
    0.707107,
)
RBY1_WUJI2_RIGHT_WRIST_IDLE_ROOT_POSE = (
    0.4416,
    -0.310724,
    0.998931,
    -0.000002,
    0.707107,
    0.707107,
    0.000001,
)

RBY1_WUJI2_JOINT_ACTION_NAMES = (
    *RBY1_WUJI2_PINK_CONTROLLED_JOINT_NAMES,
    *RBY1_WUJI2_HAND_JOINT_ACTION_NAMES,
    *RBY1_WUJI2_TORSO_JOINT_NAMES,
)

RBY1_WUJI2_DATASET_ROBOT = "rby1_wuji2"

RBY1_WUJI2_OPENXR_ACTION_ORDER = (
    "left_wrist_pos_x",
    "left_wrist_pos_y",
    "left_wrist_pos_z",
    "left_wrist_quat_x",
    "left_wrist_quat_y",
    "left_wrist_quat_z",
    "left_wrist_quat_w",
    "right_wrist_pos_x",
    "right_wrist_pos_y",
    "right_wrist_pos_z",
    "right_wrist_quat_x",
    "right_wrist_quat_y",
    "right_wrist_quat_z",
    "right_wrist_quat_w",
    *RBY1_WUJI2_HAND_JOINT_ACTION_NAMES,
)

RBY1_WUJI2_OPENXR_ACTION_DIM = len(RBY1_WUJI2_OPENXR_ACTION_ORDER)

_WUJI2_HAND_ZERO_POSE = {name: 0.0 for name in RBY1_WUJI2_HAND_JOINT_ACTION_NAMES}


def _with_wuji2_hands(body_pose: dict[str, float]) -> dict[str, float]:
    """An RBY1-M body pose completed with all-zero v2 hand joints."""

    return {**body_pose, **_WUJI2_HAND_ZERO_POSE}


RBY1_WUJI2_REPLAY_HOME_JOINT_POS = _with_wuji2_hands(RBY1M_REPLAY_HOME_JOINT_POS)
RBY1_WUJI2_M5_KITTING_JOINT_POS = _with_wuji2_hands(RBY1M_M5_KITTING_JOINT_POS)
RBY1_WUJI2_TELEOP_HOME_JOINT_POS = _with_wuji2_hands(RBY1M_TELEOP_HOME_JOINT_POS)
RBY1_WUJI2_TELEOP_READY_JOINT_POS = _with_wuji2_hands(RBY1M_TELEOP_READY_JOINT_POS)
RBY1_WUJI2_CONTROLLER_HOME_JOINT_POS = _with_wuji2_hands(RBY1M_CONTROLLER_HOME_JOINT_POS)
RBY1_WUJI2_CONTROLLER_READY_JOINT_POS = _with_wuji2_hands(RBY1M_CONTROLLER_READY_JOINT_POS)
RBY1_WUJI2_RETAIL_READY_JOINT_POS = _with_wuji2_hands(RBY1M_RETAIL_READY_JOINT_POS)
RBY1_WUJI2_DESK_READY_JOINT_POS = _with_wuji2_hands(RBY1M_DESK_READY_JOINT_POS)

RBY1_WUJI2_R3_CARRY_JOINT_POS = {
    "wheel_fr": 0.0,
    "wheel_fl": 0.0,
    "wheel_rr": 0.0,
    "wheel_rl": 0.0,
    "torso_0": -1.7381343013767037e-06,
    "torso_1": 0.5235999822616577,
    "torso_2": -1.0471999645233154,
    "torso_3": 0.5235999822616577,
    "torso_4": -1.459909185541619e-06,
    "torso_5": 8.723135920263303e-07,
    "right_arm_0": 0.6162999868392944,
    "left_arm_0": 0.5878000259399414,
    "head_0": -0.0011783110676333308,
    "right_arm_1": -0.3831002116203308,
    "left_arm_1": 0.34119999408721924,
    "head_1": 0.8988216519355774,
    "right_arm_2": -0.02350040338933468,
    "left_arm_2": -0.028400424867868423,
    "right_arm_3": -2.1129000186920166,
    "left_arm_3": -2.0933001041412354,
    "right_arm_4": -0.012600029818713665,
    "left_arm_4": 0.001099946559406817,
    "right_arm_5": 0.004200040828436613,
    "left_arm_5": -0.0032000048086047173,
    "right_arm_6": -0.45410001277923584,
    "left_arm_6": 0.4372999966144562,
    "right_thumb_cmc_flex": 6.740926039583428e-09,
    "right_index_finger_mcp_flex": 3.825506578181148e-09,
    "right_middle_finger_mcp_flex": 4.1315573184874665e-09,
    "right_ring_finger_mcp_flex": 4.88746643156901e-09,
    "right_pinky_mcp_flex": 4.616727444783919e-09,
    "left_thumb_cmc_flex": -3.5385065988435827e-09,
    "left_index_finger_mcp_flex": -1.4149260652018825e-09,
    "left_middle_finger_mcp_flex": -1.7353669612418798e-09,
    "left_ring_finger_mcp_flex": -2.7211490838396912e-09,
    "left_pinky_mcp_flex": -2.9444824356517074e-09,
    "right_thumb_cmc_abd": 5.827636595512331e-09,
    "right_index_finger_mcp_abd": 1.083660183809343e-08,
    "right_middle_finger_mcp_abd": 1.1532139687631116e-08,
    "right_ring_finger_mcp_abd": 1.1583666470471599e-08,
    "right_pinky_mcp_abd": 9.61362900397944e-09,
    "left_thumb_cmc_abd": -4.035499934218478e-09,
    "left_index_finger_mcp_abd": -1.047561237754735e-08,
    "left_middle_finger_mcp_abd": -1.1430399737832886e-08,
    "left_ring_finger_mcp_abd": -1.174647223933789e-08,
    "left_pinky_mcp_abd": -9.875269491033123e-09,
    "right_thumb_mcp": 3.0314057930524996e-09,
    "right_index_finger_pip": 7.466577356396442e-10,
    "right_middle_finger_pip": 8.345170665613466e-10,
    "right_ring_finger_pip": 1.013779837855111e-09,
    "right_pinky_pip": 1.0021482532707182e-09,
    "left_thumb_mcp": -1.7539344421280134e-09,
    "left_index_finger_pip": -2.093415024662093e-10,
    "left_middle_finger_pip": -3.010520721602461e-10,
    "left_ring_finger_pip": -5.294220617457768e-10,
    "left_pinky_pip": -6.182723222281084e-10,
    "right_thumb_ip": 3.8558875536942594e-10,
    "right_index_finger_dip": 7.182685002327105e-11,
    "right_middle_finger_dip": 8.842680754295174e-11,
    "right_ring_finger_dip": 1.1540386007924397e-10,
    "right_pinky_dip": 1.337977645343713e-10,
    "left_thumb_ip": -1.7259160767224557e-10,
    "left_index_finger_dip": 1.6326987505033141e-12,
    "left_middle_finger_dip": -1.5335477679401244e-11,
    "left_ring_finger_dip": -4.783291471244233e-11,
    "left_pinky_dip": -7.258194045789423e-11,
}

WUJI2_M4_OPEN_FINGERS = {
    "thumb": (1.0490623218, -1.3221081018, 0.6408053324, 1.0),
    "index_finger": (1.0928712216, 0.4010723549, 0.559864581, 0.5),
    "middle_finger": (0.8389371648, 0.6980000138, 1.278158228, 0.0),
    "ring_finger": (1.4, 0.65, 1.6, 1.4),
    "pinky": (0.0, 0.0, 0.0, 0.0),
}
RBY1_WUJI2_M4_SIDE_PICK_JOINT_POS = {
    **_with_wuji2_hands(RBY1M_M4_SIDE_PICK_JOINT_POS),
    **{
        f"left_{finger}_{stage}": value
        for finger, values in WUJI2_M4_OPEN_FINGERS.items()
        for stage, value in zip(WUJI2_FINGER_STAGE_SUFFIXES[finger], values)
    },
}

RBY1_WUJI2_POWER_DRIVER_GRASP = {
    "body_name": "right_wrist",
    "grip_point_xyz": (0.0, 0.067, 0.087),
    "handle_bounds_xyz": ((-.027, .025, .034), (.027, .110, .149)),
    "wrap_pad_targets": {
        "thumb": (.010, .055, .100),
        "index_finger": (.007, .032, .132),
        "middle_finger": (.015, .067, .105),
        "ring_finger": (.015, .067, .080),
        "pinky": (.015, .067, .060),
    },
    "wrap_pad_offset": (0., .0094, -.0154),
    "wrap_thumb_pad_offset": (0., .0114, -.0154),
    "position_xyz": (-0.102, 0.040, -0.122),
    "rotation_xyzw": (0.5, 0.5, 0.5, 0.5),
    "finger_targets": dict(zip(RBY1_WUJI2_RIGHT_HAND_COMMAND_JOINT_NAMES, (
        0.8491520749521936, -0.6969420371372501, 1.1430747257170086, 0.3596795319549087,
        0.7250358763300301, -0.24271512369653928, 1.1418946889545238, 0.814775235027558,
        1.2338469178102558, -0.1396382607819128, 1.043094192171467, 0.7212419801102679,
        1.0513289366654879, 0.08207192816621128, 1.1593005789725526, 0.8009665921224555,
        0.809430613715352, 0.33200030909890926, 1.1542479722402985, 0.8140519940919824,
    ))),
    "open_finger_targets": dict(zip(RBY1_WUJI2_RIGHT_HAND_COMMAND_JOINT_NAMES, (
        0.6639729697147302, -1.4639999999999997, 0.2, 0.2,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    ))),
}


RBY1_WUJI2_POWER_DRIVER_PREGRASP_JOINT_POS = {
    **RBY1_WUJI2_REPLAY_HOME_JOINT_POS,
    "torso_1": .25, "torso_2": -.5, "torso_3": .25,
    **dict(zip(RBY1_WUJI2_LEFT_ARM_JOINT_NAMES, (0., .3, 0., -.35, 0., 0., .3))),
    **dict(zip((*RBY1_WUJI2_RIGHT_ARM_JOINT_NAMES, RBY1_WUJI2_TORSO_YAW_JOINT_NAME), (
        -0.3580048878001972, -0.1405432524109559, -0.4965259572458347, -1.2654409645556532, -1.507421017797923, 1.4583990002312597, -0.06942454363290802, -0.7915064944106356,
    ))),
    **RBY1_WUJI2_POWER_DRIVER_GRASP["open_finger_targets"],
}


RBY1_WUJI2_POWER_DRIVER_LOW_PREGRASP_JOINT_POS = {
    **RBY1_WUJI2_POWER_DRIVER_PREGRASP_JOINT_POS,
    "torso_1": .7684571525527198,
    "torso_2": -1.5369143051054397,
    "torso_3": .7684571525527198,
}


RBY1_WUJI2_POWER_DRIVER_LOW_CLEAR_JOINT_POS = {
    **RBY1_WUJI2_POWER_DRIVER_LOW_PREGRASP_JOINT_POS,
    "left_arm_0": 0.,
    "left_arm_3": -2.2,
    "right_arm_0": -0.18233554626124862,
    "right_arm_1": -0.6301297527137153,
    "right_arm_2": -0.9185798918533656,
    "right_arm_3": -0.8329833617021412,
    "right_arm_4": -1.7425599208874378,
    "right_arm_5": 1.724928598991293,
    "right_arm_6": 0.11973040266900994,
    RBY1_WUJI2_TORSO_YAW_JOINT_NAME: -0.45031037319503237,
    **dict(zip(RBY1_WUJI2_HEAD_JOINT_NAMES, (0., .65))),
}


RBY1_WUJI2_START_POSES = {
    "power_driver_pregrasp": RBY1_WUJI2_POWER_DRIVER_PREGRASP_JOINT_POS,
    "power_driver_low_pregrasp": RBY1_WUJI2_POWER_DRIVER_LOW_PREGRASP_JOINT_POS,
    "power_driver_low_clear": RBY1_WUJI2_POWER_DRIVER_LOW_CLEAR_JOINT_POS,
    "m4_side_pick": RBY1_WUJI2_M4_SIDE_PICK_JOINT_POS,
    "replay_home": RBY1_WUJI2_REPLAY_HOME_JOINT_POS,
    "m5_kitting": RBY1_WUJI2_M5_KITTING_JOINT_POS,
    "teleop_home": RBY1_WUJI2_TELEOP_HOME_JOINT_POS,
    "teleop_ready": RBY1_WUJI2_TELEOP_READY_JOINT_POS,
    "rrc_home": RBY1_WUJI2_CONTROLLER_HOME_JOINT_POS,
    "rrc_ready": RBY1_WUJI2_CONTROLLER_READY_JOINT_POS,
    "retail_ready": RBY1_WUJI2_RETAIL_READY_JOINT_POS,
    "desk_ready": RBY1_WUJI2_DESK_READY_JOINT_POS,
    "r3_carry": RBY1_WUJI2_R3_CARRY_JOINT_POS,
}

RBY1_WUJI2_RETIRED_GAIN_VARIABLES: tuple[str, ...] = (
    "DEXBENCH_ARM_STIFFNESS",
    "DEXBENCH_ARM_DAMPING",
    "DEXBENCH_HAND_STIFFNESS",
    "DEXBENCH_HAND_DAMPING",
    "DEXBENCH_HAND_EFFORT",
    "DEXBENCH_TORSO_STIFFNESS",
    "DEXBENCH_TORSO_DAMPING",
    "DEXBENCH_TORSO_EFFORT",
    "DEXBENCH_HEAD_STIFFNESS",
    "DEXBENCH_HEAD_DAMPING",
    "DEXBENCH_HEAD_EFFORT",
)

RBY1_WUJI2_ZED_MINI_BASELINE_M = RBY1M_ZED_MINI_BASELINE_M
RBY1_WUJI2_ZED_MINI_LEFT_POS = RBY1M_ZED_MINI_LEFT_POS
RBY1_WUJI2_ZED_MINI_QUAT = RBY1M_ZED_MINI_QUAT

RBY1_ARM_SHOULDER_FRAME = 0
RBY1_ARM_ELBOW_FRAME = 3
RBY1_ARM_DISTAL_FRAME = 6
RBY1_ARM_WRIST_FRAME = 7
RBY1_FOREARM_PROXY_SAMPLES = 12
RBY1_FOREARM_PROXY_RADIUS_M = 0.035
RBY1_UPPER_ARM_PROXY_RADIUS_M = 0.032
RBY1_PALM_PROXY_OFFSETS_M = ((0.0, 0.0, -0.045), (0.0, 0.0, -0.105))
RBY1_PALM_PROXY_RADIUS_M = 0.05
RBY1_OPEN_HAND_PROXY_OFFSETS_M = (*RBY1_PALM_PROXY_OFFSETS_M, (0.0, 0.0, -0.160), (0.060, 0.0, -0.070))
RBY1_OPEN_HAND_PROXY_RADII_M = (0.040, 0.045, 0.040, 0.030)
RBY1_TORSO_LINK_NAMES = tuple(f"link_torso_{i}" for i in range(6))
RBY1_PLANNING_MAX_ACCELERATION = 4.0

_WUJI2_HAND_LINK_STAGES = ("proximal", "proximal_abd", "middle", "distal", "tip_sensor_frame")
RBY1_WUJI2_PAPER_HAND = PaperHandSpec(
    palm_link="{side}_wrist",
    hull_link_pattern=(
        rf"{{side}}_wrist|{{side}}_({'|'.join(WUJI2_FINGER_ORDER)})_({'|'.join(_WUJI2_HAND_LINK_STAGES)})"
    ),
)
