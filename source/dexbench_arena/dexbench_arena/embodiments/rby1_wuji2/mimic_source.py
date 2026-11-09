# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Kit-free validation of stored Wuji2 Mimic source annotations."""

import h5py
import numpy as np

from dexbench_arena.embodiments.rby1_wuji2.constants import (
    RBY1_WUJI2_HAND_JOINT_ACTION_NAMES,
    RBY1_WUJI2_JOINT_ACTION_NAMES,
    RBY1_WUJI2_PINK_CONTROLLED_JOINT_NAMES,
)

RBY1_WUJI2_JOINT_ACTION_DIMS = {
    len(RBY1_WUJI2_JOINT_ACTION_NAMES),
    len(RBY1_WUJI2_PINK_CONTROLLED_JOINT_NAMES) + len(RBY1_WUJI2_HAND_JOINT_ACTION_NAMES),
}


def validate_mimic_source_dataset(path) -> None:
    """Refuse a joint-space demo whose stored target poses are achieved-pose labels."""
    with h5py.File(path, "r") as handle:
        for name, demo in handle["data"].items():
            if "actions" not in demo or demo["actions"].shape[-1] not in RBY1_WUJI2_JOINT_ACTION_DIMS:
                continue
            info = demo.get("obs/datagen_info")
            if info is None or "eef_pose" not in info or "target_eef_pose" not in info:
                continue
            achieved, targets = info["eef_pose"], info["target_eef_pose"]
            if len(targets) == 0 or set(targets) != set(achieved):
                continue
            if all(targets[side].size and np.array_equal(targets[side][:], achieved[side][:]) for side in targets):
                raise ValueError(
                    f"{path}: {name} has joint-space target_eef_pose equal to eef_pose at every step "
                    "for every eef (achieved-pose labels, not commanded targets). Re-annotate the original recording "
                    "to compute commanded-FK targets: the maintainers' boundary extraction, then "
                    "scripts/imitation_learning/annotate_demos.py --auto --mimic_boundaries <table>."
                )
