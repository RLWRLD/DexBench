# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""The ArenaEnvBuilder hooks check the build before compose and wrap the entry point (recorder-ready events, Newton sync)."""

from __future__ import annotations

import sys
import types

import pytest

pytest.importorskip("isaaclab_arena")
from isaaclab_arena.utils.physics_backend import PhysicsBackend

from dexbench_arena.assets.newton_episode_reset import SyncsNewtonSavedState
from dexbench_arena.embodiments.common.recorder_ready_events import RECORDER_READY_MODE, AppliesRecorderReadyEvents
from dexbench_arena.environments.arena_builder_hooks import install_builder_hooks

BUILDER_MODULE = "isaaclab_arena.environments.arena_env_builder"


class _Env:
    """Arena's env as far as the hooks see it: load_managers ends by replacing the recorder manager."""

    def load_managers(self):
        self.applied = []
        self.event_manager = types.SimpleNamespace(
            available_modes=["startup", RECORDER_READY_MODE],
            apply=lambda mode: self.applied.append((mode, self.recorder_manager)),
        )
        self.recorder_manager = "arena recorder manager"


class _Settings:
    def __init__(self):
        self.calls = []

    def before_compose(self, builder_cfg, backend):
        self.calls.append(("before_compose", builder_cfg, backend))

    def after_compose(self, variations, env_cfg):
        self.calls.append(("after_compose", variations, env_cfg))


def _install_with(monkeypatch, builder_cls):
    module = types.ModuleType(BUILDER_MODULE)
    module.ArenaEnvBuilder = builder_cls
    monkeypatch.setitem(sys.modules, BUILDER_MODULE, module)
    install_builder_hooks()


def _builder():
    class ArenaEnvBuilder:
        def __init__(self, arena_env, cfg, resolved_physics_backend=PhysicsBackend.NEWTON):
            self.arena_env, self.cfg, self.resolved_physics_backend = arena_env, cfg, resolved_physics_backend

        def compose_manager_cfg(self):
            return "env_cfg", {}

        def get_all_variations(self):
            return {"asset": ["variation"]}

        def get_entry_point(self):
            return _Env

    return ArenaEnvBuilder


def test_hooks_check_and_wrap_every_dexbench_build(monkeypatch):
    """The settings see the backend the builder resolved (they refuse anything but Newton) and a
    DexBench env gets the Newton sync; the recorder-ready events go on every env class."""
    builder_cls = _builder()
    _install_with(monkeypatch, builder_cls)
    _install_with(monkeypatch, builder_cls)
    settings = _Settings()
    backend = PhysicsBackend.NEWTON
    builder = builder_cls(types.SimpleNamespace(dexbench_settings=settings), "builder_cfg", backend)

    assert builder.compose_manager_cfg() == ("env_cfg", {})
    assert settings.calls == [("before_compose", "builder_cfg", backend), ("after_compose", {"asset": ["variation"]}, "env_cfg")]
    entry_point = builder.get_entry_point()
    assert issubclass(entry_point, _Env)
    assert issubclass(entry_point, SyncsNewtonSavedState)
    assert issubclass(entry_point, AppliesRecorderReadyEvents)
    env = object.__new__(entry_point)
    env.load_managers()
    assert env.applied == [(RECORDER_READY_MODE, "arena recorder manager")]
    plain = builder_cls(types.SimpleNamespace(), "builder_cfg", backend)
    assert plain.compose_manager_cfg() == ("env_cfg", {})
    plain_entry = plain.get_entry_point()
    assert issubclass(plain_entry, AppliesRecorderReadyEvents) and not issubclass(plain_entry, SyncsNewtonSavedState)
