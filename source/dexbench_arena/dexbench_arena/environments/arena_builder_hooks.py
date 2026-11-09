# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Two hooks on ``ArenaEnvBuilder`` that Arena has no extension point for."""

from __future__ import annotations

from dexbench_arena.assets.newton_episode_reset import newton_entry_point
from dexbench_arena.embodiments.common.recorder_ready_events import recorder_ready_entry_point
from dexbench_arena.evaluation.replay_guards import check_recorded_draws


def install_builder_hooks() -> None:
    """Check the build before Arena's ``compose_manager_cfg`` and wrap its entry point, on every builder."""

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    upstream = ArenaEnvBuilder.compose_manager_cfg
    if getattr(upstream, "_dexbench_builder_hooks", False):
        return

    def compose_manager_cfg(self, *args, **kwargs):
        settings = getattr(self.arena_env, "dexbench_settings", None)
        if settings is not None:
            settings.before_compose(self.cfg, self.resolved_physics_backend)
        rows = getattr(self.cfg, "recorded_variation_samples_path", None)
        if rows is not None:
            check_recorded_draws(self.get_all_variations(), rows)
        composed = upstream(self, *args, **kwargs)
        if settings is not None:
            settings.after_compose(self.get_all_variations(), composed[0])
        return composed

    upstream_entry_point = ArenaEnvBuilder.get_entry_point

    def get_entry_point(self):
        entry_point = recorder_ready_entry_point(upstream_entry_point(self))
        if getattr(self.arena_env, "dexbench_settings", None) is None:
            return entry_point
        return newton_entry_point(entry_point)

    compose_manager_cfg._dexbench_builder_hooks = True
    compose_manager_cfg.__wrapped__ = upstream
    get_entry_point.__wrapped__ = upstream_entry_point
    ArenaEnvBuilder.compose_manager_cfg = compose_manager_cfg
    ArenaEnvBuilder.get_entry_point = get_entry_point
