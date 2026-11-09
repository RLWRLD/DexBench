# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""DB-H3's configuration: the collection drives its constructor selects, and its command-line defaults."""

import pytest

pytest.importorskip("isaaclab")

from dexbench_arena.embodiments.rby1_wuji2.rby1_wuji2 import DexBenchRby1Wuji2JointEmbodiment
from dexbench_arena.environments.manipulation.e4_h3_environment import H3Environment, H3EnvironmentCfg


def test_constructor_selects_collection_drives_without_mutating_default():
    factory = H3Environment()
    params = factory.embodiment_constructor_kwargs(H3EnvironmentCfg(arm_mode='dual_arm'))
    embodiment = DexBenchRby1Wuji2JointEmbodiment(**params)
    assert embodiment.support_hand_mode == 'init_q'
    assert embodiment.scene_config.robot.actuators['left_hand'].stiffness == 8.
    for group, values in embodiment.extraction_actuator_overrides.items():
        for field, expected in values.items():
            assert getattr(embodiment.scene_config.robot.actuators[group], field) == expected
    ordinary = DexBenchRby1Wuji2JointEmbodiment()
    assert ordinary.scene_config.robot.actuators['left_arm'].stiffness == 400.
    assert ordinary.scene_config.robot.actuators['left_hand'].stiffness == 100.
    with pytest.raises(ValueError, match='Unknown extraction drive profile'):
        DexBenchRby1Wuji2JointEmbodiment(extraction_drive_profile='typo')


def test_a_camera_rerecord_of_h3_gets_the_embodiments_wrist_cameras():
    """H3's command line leaves --wrist_cameras unset, as the typed config does, so a rerecord with cameras takes
    the Wuji2 rerecord default; an explicit flag still wins."""
    import argparse

    from dexbench_arena.embodiments.lookup import apply_rerecord_camera_defaults

    parser = argparse.ArgumentParser()
    H3Environment.add_cli_args(parser)
    assert parser.parse_args([]).wrist_cameras is None is H3EnvironmentCfg.wrist_cameras
    for argv, expected in (([], True), (['--no-wrist-cameras'], False)):
        args = parser.parse_args(argv)
        args.enable_cameras = True
        apply_rerecord_camera_defaults(args)
        assert args.wrist_cameras is expected
