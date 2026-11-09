# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Reject policy layouts that preserve width while permuting joints."""
import runpy
from pathlib import Path

import pytest

from dexbench_arena.embodiments.rby1_wuji2 import constants

SOURCE = Path(__file__).resolve().parents[1] / 'dexbench_arena/embodiments/rby1_wuji2/joint_groups.py'


def test_same_width_reordering_is_rejected(monkeypatch):
    names = constants.RBY1_WUJI2_HAND_JOINT_ACTION_NAMES
    monkeypatch.setattr(constants, 'RBY1_WUJI2_HAND_JOINT_ACTION_NAMES', names[::-1])
    with pytest.raises(AssertionError, match='canonical action order'):
        runpy.run_path(str(SOURCE))


def test_the_old_path_gives_the_same_groups():
    from dexbench_arena.embodiments.rby1_wuji2 import joint_groups
    from dexbench_arena.policy import wuji2_joint_groups

    for name in ("WUJI2_JOINT_GROUPS", "WUJI2_RECORDED_JOINT_GROUPS", "WUJI2_JOINT_LAYOUTS"):
        assert getattr(wuji2_joint_groups, name) is getattr(joint_groups, name)
    assert wuji2_joint_groups.WUJI2_RRC_JOINT_GROUPS is joint_groups.WUJI2_RECORDED_JOINT_GROUPS
