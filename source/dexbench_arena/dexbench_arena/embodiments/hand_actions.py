# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Helpers for splitting and rebuilding interleaved bimanual hand actions."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def _side_relative_indices(hand_joint_names: Sequence[str]) -> tuple[list[int], list[int]]:
    left_relative = [
        index for index, name in enumerate(hand_joint_names) if name.startswith("left_")
    ]
    right_relative = [
        index for index, name in enumerate(hand_joint_names) if name.startswith("right_")
    ]
    if len(left_relative) + len(right_relative) != len(hand_joint_names):
        raise ValueError("Every hand action name must identify its left or right side")
    return left_relative, right_relative


def split_bimanual_hand_actions(actions, hand_joint_names: Sequence[str], hand_offset: int) -> dict:
    """Gather left and right hand actions without assuming contiguous blocks."""

    left_relative, right_relative = _side_relative_indices(hand_joint_names)

    return {
        "left": actions[..., [hand_offset + index for index in left_relative]],
        "right": actions[..., [hand_offset + index for index in right_relative]],
    }


def merge_bimanual_hand_actions(left_actions, right_actions, hand_joint_names: Sequence[str]):
    """Rebuild the public stage-major layout from per-hand action vectors."""

    left_relative, right_relative = _side_relative_indices(hand_joint_names)
    if left_actions.shape[:-1] != right_actions.shape[:-1]:
        raise ValueError("Left and right hand actions must have matching leading dimensions")
    if left_actions.shape[-1] != len(left_relative) or right_actions.shape[-1] != len(right_relative):
        raise ValueError("Per-hand action widths do not match the hand joint names")

    left_index = 0
    right_index = 0
    ordered_actions = []
    for name in hand_joint_names:
        if name.startswith("left_"):
            ordered_actions.append(left_actions[..., left_index])
            left_index += 1
        else:
            ordered_actions.append(right_actions[..., right_index])
            right_index += 1
    return torch.stack(ordered_actions, dim=-1)
