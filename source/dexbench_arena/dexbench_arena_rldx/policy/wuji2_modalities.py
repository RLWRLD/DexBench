# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Native RLDX image inputs and the six-group absolute-joint layout."""

from rldx.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig

from dexbench_arena.embodiments.rby1_wuji2.camera_contract import EGO_CAMERAS, camera_mapping
from dexbench_arena_rldx.policy.dataset_contract import DATASET_JOINT_GROUPS


def modality_config(camera_keys=tuple(EGO_CAMERAS), action_horizon=16):
    """The RLDX modality config for ``camera_keys``, in that order, predicting ``action_horizon``-step chunks (the
    training run's ``--action-horizon``, which RLDX refuses when it disagrees); the joint and language parts never vary."""
    cameras = camera_mapping(camera_keys)
    if "camera_ego_left" not in cameras:
        raise ValueError("RLDX video contract must include the left ego camera")
    groups = list(DATASET_JOINT_GROUPS)
    return {
        "video": ModalityConfig(delta_indices=[0], modality_keys=list(cameras)),
        "state": ModalityConfig(delta_indices=[0], modality_keys=groups),
        "action": ModalityConfig(
            delta_indices=list(range(action_horizon)), modality_keys=groups,
            action_configs=[ActionConfig(rep=ActionRepresentation.ABSOLUTE,
                                         type=ActionType.NON_EEF, format=ActionFormat.DEFAULT)
                            for _ in groups],
        ),
        "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.task_description"]),
    }
