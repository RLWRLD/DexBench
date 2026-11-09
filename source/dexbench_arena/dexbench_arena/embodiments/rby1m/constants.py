# SPDX-License-Identifier: Apache-2.0
"""The RBY1-M body: what every RBY1-M hand generation shares."""

from __future__ import annotations

from dexbench_arena.embodiments.camera_catalog import ZED_MINI_SPEC_BASELINE_M

RBY1M_BASE_MODEL = "rby1m"


RBY1M_LEFT_ARM_JOINT_NAMES = (
    "left_arm_0",
    "left_arm_1",
    "left_arm_2",
    "left_arm_3",
    "left_arm_4",
    "left_arm_5",
    "left_arm_6",
)

RBY1M_RIGHT_ARM_JOINT_NAMES = (
    "right_arm_0",
    "right_arm_1",
    "right_arm_2",
    "right_arm_3",
    "right_arm_4",
    "right_arm_5",
    "right_arm_6",
)

RBY1M_HEAD_JOINT_NAMES = (
    "head_0",
    "head_1",
)

RBY1M_TORSO_JOINT_NAMES = (
    "torso_0",
    "torso_1",
    "torso_2",
    "torso_3",
    "torso_4",
    "torso_5",
)
RBY1M_TORSO_YAW_JOINT_NAME = "torso_5"


def rby1m_torso_z_fold(theta_rad: float) -> dict[str, float]:
    """The torso's Z-fold by ``theta_rad``: ``(torso_1, torso_2, torso_3) = (theta, -2 theta, theta)``."""
    return {"torso_1": theta_rad, "torso_2": -2.0 * theta_rad, "torso_3": theta_rad}


RBY1M_WHEEL_JOINT_NAMES = (
    "wheel_fl",
    "wheel_fr",
    "wheel_rl",
    "wheel_rr",
)


RBY1M_NULL_SPACE_JOINT_NAMES = (
    "left_arm_0",
    "left_arm_1",
    "left_arm_2",
    "left_arm_3",
    "right_arm_0",
    "right_arm_1",
    "right_arm_2",
    "right_arm_3",
    "head_0",
    "head_1",
)


RBY1M_XR_ANCHOR_POS = (0.0, 0.0, 0.0)

RBY1M_XR_ANCHOR_ROT = (0.0, 0.0, -0.70711, 0.70711)



RBY1M_REPLAY_HOME_JOINT_POS = {
    "wheel_fr": 0.0,
    "wheel_fl": 0.0,
    "wheel_rr": 0.0,
    "wheel_rl": 0.0,
    "torso_0": 0.0,
    "torso_1": 0.5236,
    "torso_2": -1.0472,
    "torso_3": 0.5236,
    "torso_4": 0.0,
    "torso_5": 0.0,
    "right_arm_0": 0.6163,
    "right_arm_1": -0.3831,
    "right_arm_2": -0.0235,
    "right_arm_3": -2.1129,
    "right_arm_4": -0.0126,
    "right_arm_5": 0.0042,
    "right_arm_6": -0.4541,
    "left_arm_0": 0.5878,
    "left_arm_1": 0.3412,
    "left_arm_2": -0.0284,
    "left_arm_3": -2.0933,
    "left_arm_4": 0.0011,
    "left_arm_5": -0.0032,
    "left_arm_6": 0.4373,
    "head_0": 0.0,
    "head_1": 0.0,
}

RBY1M_M5_KITTING_JOINT_POS = {
    **RBY1M_REPLAY_HOME_JOINT_POS,
    "torso_1": 0.9644,
    "torso_2": -1.6107,
    "torso_3": 0.6579,
    "head_0": 0.9,
    "head_1": 1.0,
}


RBY1M_TELEOP_HOME_JOINT_POS = {
    "wheel_fr": 0.0,
    "wheel_fl": 0.0,
    "wheel_rr": 0.0,
    "wheel_rl": 0.0,
    "torso_0": 0.0,
    "torso_1": 0.5236,
    "torso_2": -1.0472,
    "torso_3": 0.5236,
    "torso_4": 0.0,
    "torso_5": 0.0,
    "right_arm_0": 0.6163,
    "right_arm_1": -0.3831,
    "right_arm_2": -0.0235,
    "right_arm_3": -2.1129,
    "right_arm_4": -0.0126,
    "right_arm_5": 0.0042,
    "right_arm_6": -0.4541,
    "left_arm_0": 0.5878,
    "left_arm_1": 0.3412,
    "left_arm_2": -0.0284,
    "left_arm_3": -2.0933,
    "left_arm_4": 0.0011,
    "left_arm_5": -0.0032,
    "left_arm_6": 0.4373,
    "head_0": 0.0,
    "head_1": 0.9,
}

RBY1M_M4_SIDE_PICK_JOINT_POS = {
    **RBY1M_TELEOP_HOME_JOINT_POS,
    "torso_1": 0.2617993877991494,
    "torso_2": -0.5235987755982988,
    "torso_3": 0.2617993877991494,
    "torso_5": 0.7853981633974483,
}


RBY1M_TELEOP_READY_JOINT_POS = {
    "wheel_fr": 0.0,
    "wheel_fl": 0.0,
    "wheel_rr": 0.0,
    "wheel_rl": 0.0,
    "torso_0": 0.0,
    "torso_1": 0.5236,
    "torso_2": -1.0472,
    "torso_3": 0.5236,
    "torso_4": 0.0,
    "torso_5": 0.0,
    "right_arm_0": 0.9,
    "right_arm_1": -0.55,
    "right_arm_2": -0.0235,
    "right_arm_3": -2.45,
    "right_arm_4": -0.0126,
    "right_arm_5": -0.35,
    "right_arm_6": -0.4541,
    "left_arm_0": 0.9,
    "left_arm_1": 0.55,
    "left_arm_2": -0.0284,
    "left_arm_3": -2.45,
    "left_arm_4": 0.0011,
    "left_arm_5": -0.35,
    "left_arm_6": 0.4373,
    "head_0": 0.0,
    "head_1": 0.9,
}


RBY1M_CONTROLLER_HOME_JOINT_POS = {
    "wheel_fr": 0.0,
    "wheel_fl": 0.0,
    "wheel_rr": 0.0,
    "wheel_rl": 0.0,
    "torso_0": 0.0,
    "torso_1": 0.5236,
    "torso_2": -1.0472,
    "torso_3": 0.5236,
    "torso_4": 0.0,
    "torso_5": 0.0,
    "right_arm_0": 0.0,
    "right_arm_1": -0.3,
    "right_arm_2": 0.0,
    "right_arm_3": -1.5708,
    "right_arm_4": 0.0,
    "right_arm_5": 0.0,
    "right_arm_6": -0.3,
    "left_arm_0": 0.0,
    "left_arm_1": 0.3,
    "left_arm_2": 0.0,
    "left_arm_3": -1.5708,
    "left_arm_4": 0.0,
    "left_arm_5": 0.0,
    "left_arm_6": 0.3,
    "head_0": 0.0,
    "head_1": 0.9,
}


RBY1M_CONTROLLER_READY_JOINT_POS = {
    "wheel_fr": 0.0,
    "wheel_fl": 0.0,
    "wheel_rr": 0.0,
    "wheel_rl": 0.0,
    "torso_0": 0.0,
    "torso_1": 0.5236,
    "torso_2": -1.0472,
    "torso_3": 0.5236,
    "torso_4": 0.0,
    "torso_5": 0.0,
    "right_arm_0": -0.7,
    "right_arm_1": -0.3,
    "right_arm_2": 0.0,
    "right_arm_3": -1.5708,
    "right_arm_4": 0.0,
    "right_arm_5": 0.0,
    "right_arm_6": -0.3,
    "left_arm_0": -0.7,
    "left_arm_1": 0.3,
    "left_arm_2": 0.0,
    "left_arm_3": -1.5708,
    "left_arm_4": 0.0,
    "left_arm_5": 0.0,
    "left_arm_6": 0.3,
    "head_0": 0.0,
    "head_1": 0.9,
}


RBY1M_RETAIL_READY_JOINT_POS = {
    "wheel_fr": 0.0,
    "wheel_fl": 0.0,
    "wheel_rr": 0.0,
    "wheel_rl": 0.0,
    "torso_0": 0.0,
    "torso_1": 0.5236,
    "torso_2": -1.0472,
    "torso_3": 0.5236,
    "torso_4": 0.0,
    "torso_5": 0.0,
    "right_arm_0": 0.45,
    "right_arm_1": -0.425,
    "right_arm_2": 0.0,
    "right_arm_3": -2.01,
    "right_arm_4": 0.0,
    "right_arm_5": -0.175,
    "right_arm_6": -0.3,
    "left_arm_0": 0.45,
    "left_arm_1": 0.425,
    "left_arm_2": 0.0,
    "left_arm_3": -2.01,
    "left_arm_4": 0.0,
    "left_arm_5": -0.175,
    "left_arm_6": 0.3,
    "head_0": 0.0,
    "head_1": 0.9,
}


RBY1M_DESK_READY_JOINT_POS = {
    "wheel_fr": 0.0,
    "wheel_fl": 0.0,
    "wheel_rr": 0.0,
    "wheel_rl": 0.0,
    "torso_0": 0.0,
    "torso_1": 0.5236,
    "torso_2": -1.0472,
    "torso_3": 0.5236,
    "torso_4": 0.0,
    "torso_5": 0.0,
    "right_arm_0": 0.9,
    "right_arm_1": -0.55,
    "right_arm_2": 0.0,
    "right_arm_3": -2.45,
    "right_arm_4": 0.0,
    "right_arm_5": -0.35,
    "right_arm_6": -0.3,
    "left_arm_0": 0.9,
    "left_arm_1": 0.55,
    "left_arm_2": 0.0,
    "left_arm_3": -2.45,
    "left_arm_4": 0.0,
    "left_arm_5": -0.35,
    "left_arm_6": 0.3,
    "head_0": 0.0,
    "head_1": 0.9,
}



RBY1M_ZED_MINI_LEFT_POS = (0.0276939720886537, 0.015823691002036552, 0.05795993357446777)

RBY1M_ZED_MINI_QUAT = (-0.49008310631483837, 0.5103343444683206, -0.4984392868061547, 0.5009348092607065)

RBY1M_ZED_MINI_BASELINE_M = ZED_MINI_SPEC_BASELINE_M
