# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""DexBench retargeter registrations."""

from __future__ import annotations

from collections.abc import Callable

from isaaclab_arena.assets.register import register_retargeter
from isaaclab_arena.assets.retargeter_library import RetargetterBase


@register_retargeter
class DexBenchRby1Wuji2OpenXRRetargeter(RetargetterBase):
    """Quest/OpenXR pipeline builder for the native DexBench RBY1-M Wuji Hand 2 embodiment."""

    device = "openxr"
    embodiment = "dexbench_rby1_wuji2"

    def get_pipeline_builder(self, embodiment: object) -> Callable:
        from dexbench_arena.embodiments.rby1_wuji2.openxr_teleop import build_rby1_wuji2_openxr_pipeline

        def _build_pipeline():
            idle_wrist_poses = None
            if hasattr(embodiment, "get_openxr_idle_wrist_poses"):
                idle_wrist_poses = embodiment.get_openxr_idle_wrist_poses()
            return build_rby1_wuji2_openxr_pipeline(
                arm_mode=embodiment.get_arm_mode(),
                idle_wrist_poses=idle_wrist_poses,
            )

        return _build_pipeline


__all__ = [
    "DexBenchRby1Wuji2OpenXRRetargeter",
]
