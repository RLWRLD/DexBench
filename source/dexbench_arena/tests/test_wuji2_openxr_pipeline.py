# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""The Wuji Hand 2 native OpenXR retargeting builds and runs without a headset."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("isaacteleop.retargeters", reason="needs the IsaacTeleop runtime in the container")

from isaacteleop.retargeters import DexHandRetargeter, DexHandRetargeterConfig  # noqa: E402
from isaacteleop.retargeting_engine.interface import TensorGroup  # noqa: E402
from isaacteleop.retargeting_engine.tensor_types import HandInput, HandInputIndex  # noqa: E402

from dexbench_arena.embodiments.rby1_wuji2.constants import (  # noqa: E402
    RBY1_WUJI2_HAND_JOINT_ACTION_NAMES,
    RBY1_WUJI2_LEFT_RETARGET_URDF_PATH,
    RBY1_WUJI2_OPERATOR_TO_MANO_LEFT,
    RBY1_WUJI2_OPERATOR_TO_MANO_RIGHT,
    RBY1_WUJI2_RIGHT_RETARGET_URDF_PATH,
)
from dexbench_arena.embodiments.rby1_wuji2.openxr_teleop import (  # noqa: E402
    _LEFT_HAND_CONFIG,
    _RIGHT_HAND_CONFIG,
    build_rby1_wuji2_openxr_pipeline,
)

_HAND = {
    "left": (_LEFT_HAND_CONFIG, RBY1_WUJI2_LEFT_RETARGET_URDF_PATH, RBY1_WUJI2_OPERATOR_TO_MANO_LEFT),
    "right": (_RIGHT_HAND_CONFIG, RBY1_WUJI2_RIGHT_RETARGET_URDF_PATH, RBY1_WUJI2_OPERATOR_TO_MANO_RIGHT),
}
_OPENXR_HAND_JOINT_COUNT = 26


def _tracked_hand() -> TensorGroup:
    """All 26 OpenXR joints valid, spread along -y like an open hand."""

    rng = np.random.default_rng(0)
    positions = rng.uniform(-0.01, 0.01, size=(_OPENXR_HAND_JOINT_COUNT, 3))
    positions[:, 1] -= 0.02 * (np.arange(_OPENXR_HAND_JOINT_COUNT) % 5)
    hand = TensorGroup(HandInput())
    hand[HandInputIndex.JOINT_POSITIONS] = positions.astype(np.float32)
    hand[HandInputIndex.JOINT_ORIENTATIONS] = np.tile(
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (_OPENXR_HAND_JOINT_COUNT, 1)
    )
    hand[HandInputIndex.JOINT_RADII] = np.full(_OPENXR_HAND_JOINT_COUNT, 0.01, dtype=np.float32)
    hand[HandInputIndex.JOINT_VALID] = np.ones(_OPENXR_HAND_JOINT_COUNT, dtype=np.uint8)
    return hand


@pytest.mark.parametrize("side", ("left", "right"))
def test_dexpilot_retargets_a_tracked_hand_onto_the_action_named_joints(side):
    hand_config, urdf, operator_to_mano = _HAND[side]
    names = [name for name in RBY1_WUJI2_HAND_JOINT_ACTION_NAMES if name.startswith(f"{side}_")]
    retargeter = DexHandRetargeter(
        DexHandRetargeterConfig(
            hand_retargeting_config=str(hand_config),
            hand_urdf=str(urdf),
            hand_joint_names=names,
            handtracking_to_baselink_frame_transform=operator_to_mano,
            hand_side=side,
        ),
        name=f"{side}_hand",
    )

    assert sorted(retargeter._dex_hand.optimizer.robot.dof_joint_names) == sorted(names)

    joints = retargeter({f"hand_{side}": _tracked_hand()})["hand_joints"]
    q = np.array([float(joints[i]) for i in range(len(names))])
    assert np.isfinite(q).all()
    assert np.abs(q).sum() > 0.0, "all zeros is what the retargeter emits when it has no hand to work on"


@pytest.mark.parametrize("arm_mode", ("dual_arm", "left", "right"))
def test_the_pipeline_builds_without_an_xr_session(arm_mode):
    pipeline = build_rby1_wuji2_openxr_pipeline(arm_mode=arm_mode)
    assert list(pipeline.output_types()) == ["action"]
