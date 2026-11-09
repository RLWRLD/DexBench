# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""The Wuji2 embodiments add their wrist cameras only when asked, on the one shipped robot USD (Kit tier)."""

import pytest
from dexbench_arena.embodiments.rby1_wuji2.constants import RBY1_WUJI2_USD_PATH


@pytest.mark.parametrize("variant", ["pink", "joint"])
def test_optional_cameras_are_instance_local(simulation_app, variant):
    from dexbench_arena.embodiments.rby1_wuji2.rby1_wuji2 import (
        DexBenchRby1Wuji2Embodiment,
        DexBenchRby1Wuji2JointEmbodiment,
    )

    cls = {"pink": DexBenchRby1Wuji2Embodiment,
           "joint": DexBenchRby1Wuji2JointEmbodiment}[variant]
    enabled = cls(enable_cameras=True, wrist_cameras=True)
    disabled = cls(enable_cameras=True)
    camera_free = cls(enable_cameras=False)
    assert enabled.scene_config.robot.spawn.usd_path == str(RBY1_WUJI2_USD_PATH)
    assert disabled.scene_config.robot.spawn.usd_path == str(RBY1_WUJI2_USD_PATH)
    assert camera_free.scene_config.robot.spawn.usd_path == str(RBY1_WUJI2_USD_PATH)
    for side in ("left", "right"):
        assert getattr(disabled.camera_config, f"wrist_{side}_cam") is None
        assert getattr(camera_free.camera_config, f"wrist_{side}_cam") is None
        cfg = getattr(enabled.camera_config, f"wrist_{side}_cam")
        assert (cfg.width, cfg.height) == (640, 360)
    with pytest.raises(ValueError, match="enable_cameras"):
        cls(wrist_cameras=True)
