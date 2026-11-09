# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Scenes place and pose a robot from what its embodiment class declares, not from its name."""

from __future__ import annotations

import argparse

import pytest
from dexbench_arena.embodiments.rby1m.constants import RBY1M_BASE_MODEL

WUJI2_NAMES = ("dexbench_rby1_wuji2", "dexbench_rby1_wuji2_joint")


def test_start_pose_resolves_from_the_embodiment_class():
    pytest.importorskip("isaaclab_arena")
    from dexbench_arena.embodiments.rby1_wuji2.constants import RBY1_WUJI2_START_POSES
    from dexbench_arena.environments.manipulation.scene_environment_base import DexBenchSceneEnvironmentBase

    resolve = DexBenchSceneEnvironmentBase._start_pose_from_cli
    for name in WUJI2_NAMES:
        args = argparse.Namespace(start_pose="teleop_home", embodiment=name)
        assert resolve(args) is RBY1_WUJI2_START_POSES["teleop_home"], name
    with pytest.raises(ValueError, match="no named poses"):
        resolve(argparse.Namespace(start_pose="teleop_home", embodiment="franka_joint_pos"))


def test_e1_puts_wuji2_on_the_rby1m_station_spots_and_franka_on_the_anchor(simulation_app):
    from dexbench_arena.environments.manipulation.e1_retail_environment import (
        DexBenchRetailStoreE1Environment,
        DexBenchRetailStoreE1EnvironmentCfg,
    )

    factory = DexBenchRetailStoreE1Environment()
    spec = factory._layout_spec_for_args(DexBenchRetailStoreE1EnvironmentCfg())
    rby1_overrides = spec.anchor_overrides_by_base[RBY1M_BASE_MODEL]
    assert rby1_overrides, "the default E1 layout has RBY1-M station spots"

    def root_xy(name, anchor):
        embodiment, _ = factory._build_embodiment(DexBenchRetailStoreE1EnvironmentCfg(embodiment=name, anchor=anchor))
        return tuple(round(v, 4) for v in embodiment.initial_pose.position_xyz[:2])

    for anchor, station in rby1_overrides.items():
        anchor_xy = tuple(round(v, 4) for v in spec.robot_base_anchors_xyz[anchor][:2])
        station_xy = tuple(round(v, 4) for v in station[:2])
        assert station_xy != anchor_xy, anchor
        assert root_xy("franka_joint_pos", anchor) == anchor_xy, anchor
        for name in WUJI2_NAMES:
            assert root_xy(name, anchor) == station_xy, (name, anchor)


def test_an_anchor_with_a_stated_heading_keeps_it_under_a_base_offset(simulation_app):
    """DB-L1's A_L1 states its heading (180 deg): a base offset moves the robot and does not turn it."""
    import math

    from dexbench_arena.environments.manipulation.e2_logistics_l1_environment import (
        DexBenchLogisticsE2L1Environment,
        DexBenchLogisticsE2L1EnvironmentCfg,
    )
    from dexbench_arena.layouts.e2_logistics_l1_layout import (
        L1_ANCHOR_YAWS_DEG,
        L1_ROBOT_BASE_ANCHORS_XYZ,
    )

    factory = DexBenchLogisticsE2L1Environment()
    anchor_xy = L1_ROBOT_BASE_ANCHORS_XYZ["A_L1"][:2]
    for offset in ((0.0, 0.0), (0.3, -0.2)):
        cfg = DexBenchLogisticsE2L1EnvironmentCfg(
            embodiment="dexbench_rby1_wuji2_joint", anchor="A_L1", base_offset_x=offset[0], base_offset_y=offset[1]
        )
        embodiment, _ = factory._build_embodiment(cfg)
        pose = embodiment.initial_pose
        x, y, z, w = pose.rotation_xyzw
        turn = (math.degrees(2.0 * math.atan2(z, w)) - L1_ANCHOR_YAWS_DEG["A_L1"] + 180.0) % 360.0 - 180.0
        assert turn == pytest.approx(0.0, abs=1e-4), offset
        assert pose.position_xyz[:2] == pytest.approx((anchor_xy[0] + offset[0], anchor_xy[1] + offset[1])), offset
