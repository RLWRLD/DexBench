# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""The launcher's teleop and recording hooks."""

import os
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dexbench_arena.benchmark import model_run
from dexbench_arena.benchmark.protocol import launcher_hook_variables

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts/imitation_learning"
HOOKS = ("DEXBENCH_WATER", "DEXBENCH_WATER_ML", "DEXBENCH_HIDE_PRIMS", "DEXBENCH_RENDER_TRANSLUCENCY",
         "DEXBENCH_RENDER_REFLECTIONS", "DEXBENCH_RECORD_EXPORT_ON_RESET")


def _helper(monkeypatch):
    for name in launcher_hook_variables(os.environ):
        monkeypatch.delenv(name)
    return runpy.run_path(str(SCRIPTS / "_dexbench_arena_script.py"))


def test_an_evaluation_refuses_the_teleop_hooks_and_a_session_installs_them(monkeypatch):
    helper = _helper(monkeypatch)
    names = helper["_install_upstream_script_patches"].__globals__
    for name in ("apply_newton_usd_parse_guard", "_after_app_launch", "_install_stack_dump_signal",
                 "_install_teleop_trace_step_patch", "_on_env_made"):
        monkeypatch.setitem(names, name, lambda *args, **kwargs: None)
    installed = []
    for name in ("_install_water_pour_sidecar_patch", "_install_render_cfg_patch", "_install_record_on_reset_patch"):
        monkeypatch.setitem(names, name, lambda name=name: installed.append(name))

    helper["_install_upstream_script_patches"]("experiment_runner", patch_environment_cli=False, evaluation=True)
    for hook in HOOKS:
        monkeypatch.setenv(hook, "1")
        with pytest.raises(RuntimeError, match=hook):
            helper["_install_upstream_script_patches"]("experiment_runner", patch_environment_cli=False,
                                                       evaluation=True)
        with pytest.raises(ValueError, match=hook):
            model_run.environment_problems()
        monkeypatch.delenv(hook)
    assert installed == []
    helper["_install_upstream_script_patches"]("teleop.py", patch_environment_cli=False, evaluation=False)
    assert installed == ["_install_water_pour_sidecar_patch", "_install_render_cfg_patch",
                         "_install_record_on_reset_patch"]


class _Sidecar:
    def __init__(self):
        self.calls = []

    def reset(self):
        self.calls.append("reseed")

    def step(self, dt):
        self.calls.append("step")


class _Env:
    """An RL env's surface: reset, reset_to, and a step that returns (obs, reward, terminated, truncated, extras)."""

    def __init__(self, num_envs=1):
        self.num_envs, self.step_dt, self.done = num_envs, 1 / 30, False

    def reset(self, *args, **kwargs):
        return {}, {}

    def reset_to(self, state, env_ids, seed=None, is_relative=False):
        return {}, {}

    def step(self, action):
        return {}, 0.0, np.array([self.done]), np.array([False]), {}


def test_the_teleop_water_runs_one_env_and_is_seeded_again_at_every_reset(monkeypatch):
    helper = _helper(monkeypatch)
    hooks = []
    monkeypatch.setitem(helper["_install_water_pour_sidecar_patch"].__globals__, "_on_env_made", hooks.append)
    monkeypatch.setenv("DEXBENCH_WATER", "1")
    helper["_install_water_pour_sidecar_patch"]()
    with pytest.raises(RuntimeError, match="--num_envs 1"):
        hooks[0](_Env(num_envs=2))

    sidecar = _Sidecar()
    built = []
    monkeypatch.setitem(sys.modules, "dexbench_arena.fluids.water_pour_setup", SimpleNamespace(
        build_water_pour_sidecar=lambda env: built.append(env) or sidecar))
    env = _Env()
    hooks[0](env)
    env.reset()
    env.step(None)
    env.done = True
    env.step(None)
    env.done = False
    env.reset_to({}, None)
    env.step(None)
    assert len(built) == 1 and sidecar.calls == ["step", "reseed", "reseed", "step"]


def test_an_operator_reset_exports_the_take_unsuccessful_and_marked(monkeypatch):
    helper = _helper(monkeypatch)
    monkeypatch.setitem(sys.modules, "isaaclab.managers.recorder_manager", SimpleNamespace(
        DatasetExportMode=SimpleNamespace(EXPORT_ALL="all", EXPORT_SUCCEEDED_ONLY="succeeded")))
    groups = {}
    handler = SimpleNamespace(get_num_episodes=lambda: len(groups), _hdf5_data_group=groups, flush=lambda: None)
    exports, resets = [], []
    take = SimpleNamespace(success=None, is_empty=lambda: False)

    def record_pre_reset(env_ids, force_export_or_skip=None):
        take.success = False

    def export_episodes(env_ids):
        exports.append((manager.cfg.dataset_export_mode, take.success))
        groups[f"demo_{len(groups)}"] = SimpleNamespace(attrs={"success": take.success})

    manager = SimpleNamespace(_episodes={0: take}, cfg=SimpleNamespace(dataset_export_mode="succeeded"),
                              record_pre_reset=record_pre_reset, export_episodes=export_episodes,
                              _dataset_file_handler=handler, reset=lambda env_ids=None: resets.append(env_ids))
    env = SimpleNamespace(num_envs=1, recorder_manager=manager, dexbench_exported_demos={})
    helper["_patch_recorder_reset_for_export"](env)
    manager.reset()
    assert exports == [("all", False)] and manager.cfg.dataset_export_mode == "succeeded"
    assert groups["demo_0"].attrs == {"success": False, "operator_reset": True}
    assert env.dexbench_exported_demos == {0: "demo_0"} and resets == [None]
