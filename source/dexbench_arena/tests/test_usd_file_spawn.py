# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""usd_file_spawn finds Isaac Lab's USD-file spawner by name when a spawner runs, not at import."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

SHIM = Path(__file__).resolve().parents[1] / "dexbench_arena" / "assets" / "usd_file_spawn.py"


def test_the_public_name_is_preferred_and_a_missing_one_fails_only_the_spawn(monkeypatch):
    from_files = types.ModuleType("isaaclab.sim.spawners.from_files.from_files")
    for name in ("isaaclab", "isaaclab.sim", "isaaclab.sim.spawners", "isaaclab.sim.spawners.from_files"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["isaaclab.sim.spawners.from_files"].from_files = from_files
    monkeypatch.setitem(sys.modules, from_files.__name__, from_files)
    spec = importlib.util.spec_from_file_location("usd_file_spawn_under_test", SHIM)
    shim = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(shim)
    with pytest.raises(AttributeError):
        shim.spawn_usd_file("/A", "a.usd", None)
    from_files._spawn_from_usd_file = lambda *args, **kwargs: ("private", args)
    assert shim.spawn_usd_file("/A", "a.usd", "cfg") == ("private", ("/A", "a.usd", "cfg", None, None))
    from_files.spawn_from_usd_file = lambda *args, **kwargs: ("public", args)
    assert shim.spawn_usd_file("/A", "a.usd", "cfg", (1.0, 2.0, 3.0)) == ("public", ("/A", "a.usd", "cfg", (1.0, 2.0, 3.0), None))
