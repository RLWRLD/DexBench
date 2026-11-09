# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Tracking-gated wrist pose latch shared by the OpenXR teleop pipelines."""

from __future__ import annotations


def make_wrist_pose_latch(side: str, idle_pose, name: str | None = None):
    """Build a wrist-pose latch retargeter node (deferred IsaacTeleop imports)."""
    import numpy as np
    from isaacteleop.retargeters.dex_hand_retargeter import BaseRetargeter
    from isaacteleop.retargeting_engine.interface import OptionalType, TensorGroupType
    from isaacteleop.retargeting_engine.tensor_types import (
        DLDataType,
        HandInput,
        HandInputIndex,
        NDArrayType,
    )

    wrist_joint_index = 1

    class WristPoseLatchRetargeter(BaseRetargeter):
        """Hold the idle/last-valid wrist pose while hand tracking is absent."""

        def __init__(self) -> None:
            self._last_pose = np.asarray(idle_pose, dtype=np.float32).copy()
            assert self._last_pose.shape == (7,), f"idle_pose must be 7-dim, got {self._last_pose.shape}"
            super().__init__(name=name or f"{side}_wrist_latch")

        def _pose_type(self):
            return TensorGroupType(
                "ee_pose",
                [NDArrayType("pose", shape=(7,), dtype=DLDataType.FLOAT, dtype_bits=32)],
            )

        def input_spec(self):
            return {
                "ee_pose": self._pose_type(),
                f"hand_{side}": OptionalType(HandInput()),
            }

        def output_spec(self):
            return {"ee_pose": self._pose_type()}

        def _compute_fn(self, inputs, outputs, context) -> None:
            hand_group = inputs[f"hand_{side}"]
            if hand_group.is_none:
                outputs["ee_pose"][0] = self._last_pose
                return

            joint_valid = np.asarray(np.from_dlpack(hand_group[HandInputIndex.JOINT_VALID]), dtype=np.uint8)
            wrist_valid = (
                bool(joint_valid[wrist_joint_index])
                if joint_valid.size > wrist_joint_index
                else bool(joint_valid.any())
            )
            pose = np.asarray(inputs["ee_pose"][0], dtype=np.float32)
            if not wrist_valid or not np.all(np.isfinite(pose)) or float(np.linalg.norm(pose[3:7])) < 1.0e-6:
                outputs["ee_pose"][0] = self._last_pose
                return

            self._last_pose = pose.copy()
            outputs["ee_pose"][0] = pose

    return WristPoseLatchRetargeter()
