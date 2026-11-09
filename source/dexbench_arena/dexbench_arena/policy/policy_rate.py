# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""How often a policy is asked for an action, and what the control steps between two answers command."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from isaaclab_arena.policy.action_scheduling.action_scheduler import ActionScheduler
from isaaclab_arena.policy.policy_base import PolicyCfg

ACTION_INTERPOLATIONS = ("none", "linear")


@dataclass(kw_only=True)
class PolicyRateCfg(PolicyCfg):
    """A policy cfg's rate fields; keyword-only, so a cfg built on it may declare fields without a default."""

    policy_hz: int | None = None
    """How often the policy is asked [Hz], below the control rate; None asks it every control step."""

    policy_action_interpolation: str = "none"
    """What the steps between two answers command (:data:`ACTION_INTERPOLATIONS`)."""


class PolicyRateScheduler(ActionScheduler):
    """Asks ``chunks`` (Arena's chunk scheduler) for the policy's next action every ``steps_per_query`` control
    steps and interpolates between answers. One phase for every env, so a slower policy runs one env."""

    def __init__(self, chunks: ActionScheduler, control_hz: float | None, policy_hz: int | None,
                 action_interpolation: str | None, num_envs: int) -> None:
        interpolation = action_interpolation or "none"
        if interpolation not in ACTION_INTERPOLATIONS:
            raise ValueError(f"policy_action_interpolation must be one of {ACTION_INTERPOLATIONS}, got {interpolation!r}")
        ratio = 1.0 if policy_hz is None else control_hz / float(policy_hz)
        if round(ratio) < 1 or abs(ratio - round(ratio)) > 1e-6:
            raise ValueError(f"policy_hz {policy_hz} does not divide the {control_hz:g} Hz control rate")
        self.steps_per_query = round(ratio)
        if interpolation != "none" and self.steps_per_query == 1:
            raise ValueError(f"policy_action_interpolation={interpolation!r} needs a policy_hz below the control rate")
        if self.steps_per_query > 1 and num_envs != 1:
            raise ValueError(f"A policy asked every {self.steps_per_query} control steps runs one env, got {num_envs}")
        self.chunks, self.interpolation = chunks, interpolation
        self._restart()

    @property
    def asks_now(self) -> bool:
        """Whether the next :meth:`get_action` asks the policy (a policy step)."""
        return self._step % self.steps_per_query == 0

    def get_action(self, fetch_action_tensor_fn: Callable[[], torch.Tensor],
                   hold_action: torch.Tensor | None = None) -> torch.Tensor:
        phase = self._step % self.steps_per_query
        if phase == 0:
            answer = self.chunks.get_action(fetch_action_tensor_fn, hold_action=hold_action)
            self._previous = answer if self._current is None else self._current
            self._current = answer
        self._step += 1
        if self.interpolation == "none" or phase == self.steps_per_query - 1:
            return self._current
        return self._previous + (phase + 1) / self.steps_per_query * (self._current - self._previous)

    def reset(self, env_ids=None) -> None:
        self.chunks.reset(slice(None) if env_ids is None else env_ids)
        self._restart()

    def _restart(self) -> None:
        self._step, self._previous, self._current = 0, None, None
