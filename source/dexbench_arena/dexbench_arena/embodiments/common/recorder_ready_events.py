# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""An event mode that runs once the env's final recorder manager exists."""

from __future__ import annotations

RECORDER_READY_MODE = "recorder_ready"
"""Event mode applied once, after the env's managers (Arena's recorder manager included) exist."""


class AppliesRecorderReadyEvents:
    """Env mixin: apply the :data:`RECORDER_READY_MODE` events at the end of ``load_managers``."""

    def load_managers(self):
        super().load_managers()
        if RECORDER_READY_MODE in self.event_manager.available_modes:
            self.event_manager.apply(mode=RECORDER_READY_MODE)


def recorder_ready_entry_point(entry_point):
    """The gym entry point with :class:`AppliesRecorderReadyEvents` in front of ``entry_point``'s class."""
    from gymnasium.envs.registration import load_env_creator

    base = load_env_creator(entry_point) if isinstance(entry_point, str) else entry_point
    if issubclass(base, AppliesRecorderReadyEvents):
        return base
    return type(base.__name__, (AppliesRecorderReadyEvents, base), {"__module__": __name__, "__doc__": base.__doc__})
