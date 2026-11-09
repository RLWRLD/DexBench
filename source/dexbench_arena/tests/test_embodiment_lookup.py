# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Base placement, named poses, the sphere model and the rerecord camera defaults come from the embodiment
class (embodiments/lookup.py); an explicit recorder choice always wins over a robot's default."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest


def test_every_wuji2_embodiment_stands_on_the_rby1m_base_with_the_wuji2_poses():
    pytest.importorskip("isaaclab_arena")
    from dexbench_arena.embodiments import lookup
    from dexbench_arena.embodiments.rby1_wuji2.constants import RBY1_WUJI2_COLLISION_SPHERES, RBY1_WUJI2_START_POSES
    from dexbench_arena.embodiments.rby1m.constants import RBY1M_BASE_MODEL

    for name in ("dexbench_rby1_wuji2", "dexbench_rby1_wuji2_joint"):
        assert lookup.base_model(name) == RBY1M_BASE_MODEL, name
        assert lookup.start_poses(name) is RBY1_WUJI2_START_POSES, name
        assert lookup.collision_spheres(name) is RBY1_WUJI2_COLLISION_SPHERES, name
    for name in (None, "dexbench_not_a_registered_robot"):
        assert lookup.base_model(name) is None
        assert lookup.start_poses(name) == {}
        assert lookup.collision_spheres(name) is None


@pytest.mark.parametrize("enabled,choice,declared,expected", [
    (True, None, True, True), (True, False, True, False),
    (True, True, True, True), (False, None, True, None), (True, None, False, None),
])
def test_rerecord_camera_defaults(monkeypatch, enabled, choice, declared, expected):
    lookup = pytest.importorskip("dexbench_arena.embodiments.lookup")
    robot = SimpleNamespace(rerecord_camera_defaults={"wrist_cameras": True} if declared else {})
    monkeypatch.setattr(lookup, "embodiment_class", lambda name: robot)
    args = argparse.Namespace(enable_cameras=enabled, embodiment="test", wrist_cameras=choice)
    lookup.apply_rerecord_camera_defaults(args)
    assert args.wrist_cameras is expected
