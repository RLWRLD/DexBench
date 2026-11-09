# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Build the matching GR00T modality contract for a Wuji2 camera set."""

from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

from dexbench_arena.embodiments.rby1_wuji2.camera_contract import EGO_CAMERAS, camera_mapping
from dexbench_arena_gr00t.policy.wuji2_gr00t_contract import ACTION_HORIZON
from dexbench_arena.embodiments.rby1_wuji2.joint_groups import WUJI2_JOINT_GROUPS


def modality_config(camera_keys=tuple(EGO_CAMERAS)):
    """The GR00T modality config for ``camera_keys``, in that order; the joint, action and language parts never vary."""
    groups = list(WUJI2_JOINT_GROUPS)
    return {
        "video": ModalityConfig(delta_indices=[0], modality_keys=list(camera_mapping(camera_keys))),
        "state": ModalityConfig(delta_indices=[0], modality_keys=groups),
        "action": ModalityConfig(
            delta_indices=list(range(ACTION_HORIZON)), modality_keys=groups,
            action_configs=[
                ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF,
                             format=ActionFormat.DEFAULT) for _ in groups
            ],
        ),
        "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.task_description"]),
    }
