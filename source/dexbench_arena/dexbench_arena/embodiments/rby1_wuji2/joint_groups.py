"""The RBY1-M Wuji2 joint groups: the named absolute-joint contract that policy adapters and dataset layouts
share, in the canonical action order (asserted below)."""

from dexbench_arena.embodiments.rby1_wuji2.constants import (
    RBY1_WUJI2_HAND_JOINT_ACTION_NAMES,
    RBY1_WUJI2_LEFT_HAND_COMMAND_JOINT_NAMES,
    RBY1_WUJI2_RIGHT_HAND_COMMAND_JOINT_NAMES,
    RBY1_WUJI2_JOINT_ACTION_NAMES,
    RBY1_WUJI2_LEFT_ARM_JOINT_NAMES,
    RBY1_WUJI2_RIGHT_ARM_JOINT_NAMES,
    RBY1_WUJI2_HEAD_JOINT_NAMES,
    RBY1_WUJI2_TORSO_JOINT_NAMES,
)

WUJI2_JOINT_GROUPS = {
    "left_arm_joints": tuple(RBY1_WUJI2_LEFT_ARM_JOINT_NAMES),
    "right_arm_joints": tuple(RBY1_WUJI2_RIGHT_ARM_JOINT_NAMES),
    "head_joints": tuple(RBY1_WUJI2_HEAD_JOINT_NAMES),
    "hands_joints": tuple(RBY1_WUJI2_HAND_JOINT_ACTION_NAMES),
    "torso_joints": tuple(RBY1_WUJI2_TORSO_JOINT_NAMES),
}

assert tuple(name for group in WUJI2_JOINT_GROUPS.values() for name in group) == tuple(
    RBY1_WUJI2_JOINT_ACTION_NAMES
), "Wuji2 group concatenation does not match the canonical action order"

WUJI2_RECORDED_JOINT_GROUPS = {
    "torso_joints": tuple(RBY1_WUJI2_TORSO_JOINT_NAMES),
    "right_arm_joints": tuple(RBY1_WUJI2_RIGHT_ARM_JOINT_NAMES),
    "right_hand_joints": tuple(RBY1_WUJI2_RIGHT_HAND_COMMAND_JOINT_NAMES),
    "left_arm_joints": tuple(RBY1_WUJI2_LEFT_ARM_JOINT_NAMES),
    "left_hand_joints": tuple(RBY1_WUJI2_LEFT_HAND_COMMAND_JOINT_NAMES),
    "neck_joints": tuple(RBY1_WUJI2_HEAD_JOINT_NAMES),
}
WUJI2_JOINT_LAYOUTS = {"wuji2_v1": WUJI2_JOINT_GROUPS, "wuji2_rrc_v1": WUJI2_RECORDED_JOINT_GROUPS}
