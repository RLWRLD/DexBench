# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
import os
import re
import runpy
import subprocess
import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
WRAPPER_HELPER = REPO_ROOT / "scripts" / "imitation_learning" / "_dexbench_arena_script.py"


def _resolved_arena_package_dir(*, arena_root: Path | None) -> Path:
    env = os.environ.copy()
    if arena_root is None:
        env.pop("ISAACLAB_ARENA_PATH", None)
    else:
        env["ISAACLAB_ARENA_PATH"] = str(arena_root)

    code = (
        "import runpy; "
        f"module = runpy.run_path({str(WRAPPER_HELPER)!r}); "
        "print(module['ARENA_PACKAGE_DIR'])"
    )
    output = subprocess.check_output([sys.executable, "-c", code], env=env, text=True)
    return Path(output.strip())


def test_wrapper_uses_arena_path_from_container_environment(tmp_path: Path):
    arena_root = tmp_path / "isaaclab_arena"

    assert _resolved_arena_package_dir(arena_root=arena_root) == arena_root / "isaaclab_arena"


def test_wrapper_defaults_to_repository_submodule():
    assert _resolved_arena_package_dir(arena_root=None) == (
        REPO_ROOT / "submodule" / "isaaclab_arena" / "isaaclab_arena"
    )


def _wrapper_script_names() -> set[str]:
    """Names every scripts/imitation_learning wrapper asks the helper to run."""

    pattern = re.compile(r"run_upstream_imitation_script\(\s*\"([^\"]+)\"")
    names = set()
    for path in (REPO_ROOT / "scripts" / "imitation_learning").glob("*.py"):
        if path == WRAPPER_HELPER:
            continue
        names.update(pattern.findall(path.read_text()))
    return names


def _helper_module() -> dict:
    return runpy.run_path(str(WRAPPER_HELPER))


def test_every_arena_routed_wrapper_is_accounted_for():
    """Every wrapper routed to Arena must resolve to a script shipped at the pin."""

    module = _helper_module()
    arena_script_dir = module["ARENA_IM_SCRIPT_DIR"]
    if not module["ARENA_PACKAGE_DIR"].exists():
        pytest.skip("IsaacLab-Arena submodule is not checked out")

    arena_routed = _wrapper_script_names() - set(module["ISAACLAB_SCRIPT_RELPATHS"])
    missing = {name for name in arena_routed if not (arena_script_dir / name).exists()}

    assert not missing, f"Arena-routed wrappers have no upstream implementation: {missing}"



def _launcher_module():
    """The helper as a real module, so its functions see attributes patched on it."""

    spec = importlib.util.spec_from_file_location("dexbench_arena_script_under_test", WRAPPER_HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_evaluation_module_launcher_keeps_argv0_on_the_wrapper(monkeypatch):
    """Upstream's --chunk_size re-spawns [sys.executable, *sys.argv]; argv[0] must stay the wrapper."""

    launcher = _launcher_module()
    monkeypatch.setattr(launcher, "_install_upstream_script_patches", lambda *args, **kwargs: None)
    seen: dict = {}
    stub = types.ModuleType("dexbench_test_stub_runner")
    stub.__file__ = str(launcher.ARENA_ROOT / "isaaclab_arena" / "evaluation" / "stub_runner.py")
    stub.main = lambda: seen.update(argv=list(sys.argv))
    monkeypatch.setitem(sys.modules, "dexbench_test_stub_runner", stub)
    monkeypatch.setattr(sys, "argv", ["/upstream/isaaclab_arena/evaluation/experiment_runner.py", "--chunk_size", "1"])
    monkeypatch.setenv("DEXBENCH_TELEOP_TRACE_TRUNCATE", "1")

    launcher.run_upstream_evaluation_module(
        "dexbench_test_stub_runner", wrapper_path="/dexbench/scripts/evaluation/experiment_runner.py"
    )

    assert seen["argv"] == ["/dexbench/scripts/evaluation/experiment_runner.py", "--chunk_size", "1"]
    assert os.environ["DEXBENCH_TELEOP_TRACE_TRUNCATE"] == "0"


def test_the_evaluation_module_launcher_refuses_a_module_outside_the_arena_root(monkeypatch, tmp_path):
    launcher = _launcher_module()
    monkeypatch.setattr(launcher, "_install_upstream_script_patches", lambda *args, **kwargs: None)
    stray = types.ModuleType("dexbench_test_stray_runner")
    stray.__file__ = str(tmp_path / "elsewhere" / "experiment_runner.py")
    stray.main = lambda: pytest.fail("a module outside ISAACLAB_ARENA_PATH must not run")
    monkeypatch.setitem(sys.modules, "dexbench_test_stray_runner", stray)
    monkeypatch.setattr(sys, "argv", ["wrapper.py"])
    monkeypatch.setenv("DEXBENCH_TELEOP_TRACE_TRUNCATE", "1")

    with pytest.raises(FileNotFoundError, match="outside the configured Arena root"):
        launcher.run_upstream_evaluation_module("dexbench_test_stray_runner", wrapper_path="wrapper.py")
