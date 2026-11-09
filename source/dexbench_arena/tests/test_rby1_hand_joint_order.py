# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

from dexbench_arena.embodiments.joint_state import resolve_joint_order
from dexbench_arena.embodiments.rby1_wuji2.constants import RBY1_WUJI2_HAND_JOINT_ACTION_NAMES


class _FakeArticulation:
    def __init__(self, native_joint_names: list[str]):
        self.native_joint_names = native_joint_names

    def find_joints(self, joint_names: list[str], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        requested = set(joint_names)
        resolved = (
            [name for name in joint_names if name in self.native_joint_names]
            if preserve_order
            else [name for name in self.native_joint_names if name in requested]
        )
        return [self.native_joint_names.index(name) for name in resolved], list(resolved)


def _newton_hand_order() -> list[str]:
    """A different native enumeration of the same joints (Newton orders by finger, PhysX by joint)."""
    return list(reversed(RBY1_WUJI2_HAND_JOINT_ACTION_NAMES))


@pytest.mark.parametrize(
    "native_joint_names",
    [
        list(RBY1_WUJI2_HAND_JOINT_ACTION_NAMES),
        _newton_hand_order(),
    ],
    ids=["joint-major", "finger-major"],
)
def test_resolves_the_public_order_from_either_native_order(native_joint_names: list[str]):
    articulation = _FakeArticulation(native_joint_names)

    joint_ids, resolved_names = resolve_joint_order(articulation, RBY1_WUJI2_HAND_JOINT_ACTION_NAMES)

    assert resolved_names == list(RBY1_WUJI2_HAND_JOINT_ACTION_NAMES)
    assert [native_joint_names[index] for index in joint_ids] == list(RBY1_WUJI2_HAND_JOINT_ACTION_NAMES)


def test_rejects_incomplete_joint_resolution():
    articulation = _FakeArticulation(_newton_hand_order()[:-1])

    with pytest.raises(ValueError, match="Failed to resolve the exact joint order"):
        resolve_joint_order(articulation, RBY1_WUJI2_HAND_JOINT_ACTION_NAMES)
