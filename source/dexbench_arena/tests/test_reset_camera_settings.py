# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Exercise shared simulator settings without requiring an AppLauncher."""
from types import SimpleNamespace as NS

import pytest


@pytest.mark.parametrize('cameras,existing,expected', [
    (True, 0, 1), (True, None, 1), (True, 1, 1), (True, 8, 8),
    (False, 0, 0), (False, 8, 8),
])
def test_common_camera_reset_minimum_preserves_warmup(monkeypatch, cameras, existing, expected):
    cli = pytest.importorskip('dexbench_arena.cli', reason='needs IsaacLab-Arena on the path')
    for name in ('DEXBENCH_CONTROL_HZ', 'DEXBENCH_RENDER_INTERVAL', 'DEXBENCH_DLSS_MODE'):
        monkeypatch.delenv(name, raising=False)
    cfg = NS(sim=NS(), num_rerenders_on_reset=existing)
    cli.apply_dexbench_env_settings(cfg, NS(enable_cameras=cameras))
    assert cfg.num_rerenders_on_reset == expected
    assert cfg.sim.dt * cfg.decimation == 1 / 30
    assert cfg.sim.render_interval == cfg.decimation

