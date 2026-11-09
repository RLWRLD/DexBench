# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""A condition held for a span of sim time, per environment: the strict success tiers' hold."""

from __future__ import annotations

import math

import torch
from dexbench_arena.evaluation.replay_guards import deterministic_reset

_HOLDS_ATTR = "_dexbench_sustained_holds"


def _holds(env) -> dict:
    """The holds stored on the unwrapped env (gym wrappers resolve ``unwrapped`` to the base env)."""
    base = getattr(env, "unwrapped", env)
    if not hasattr(base, _HOLDS_ATTR):
        setattr(base, _HOLDS_ATTR, {})
    return getattr(base, _HOLDS_ATTR)


class SustainedHold:
    """Count consecutive qualifying control steps independently for each environment."""

    def __init__(self, like: torch.Tensor):
        self.count = torch.zeros_like(like, dtype=torch.long)
        self.last_step = torch.full_like(self.count, -1)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            self.count = torch.zeros_like(self.count)
            self.last_step = torch.full_like(self.last_step, -1)
            return
        selected = torch.zeros(self.count.shape, dtype=torch.bool, device=self.count.device)
        selected[env_ids] = True
        self.count = torch.where(selected, 0, self.count)
        self.last_step = torch.where(selected, -1, self.last_step)

    def update(
        self, ok: torch.Tensor, step: int, step_dt: float, hold_time_s: float, episode_steps: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Count step ``step`` once and return where ``ok`` has now held for ``hold_time_s``."""
        if step_dt <= 0 or hold_time_s < 0:
            raise ValueError("step_dt must be positive and hold_time_s nonnegative")
        required = max(1, math.ceil(hold_time_s / step_dt - 1e-9))
        fresh = self.last_step != step
        consecutive = self.last_step == step - 1
        count = torch.where(ok, torch.where(consecutive, self.count + 1, 1), 0)
        count = torch.where(fresh, count, self.count)
        if episode_steps is not None:
            episode_steps = episode_steps.to(count.device, count.dtype).reshape(
                episode_steps.shape[0], *([1] * (count.ndim - 1)))
            count = torch.minimum(count, episode_steps)
        self.count = count
        self.last_step = torch.full_like(self.last_step, step)
        return self.count >= required


class SustainedShareHold:
    """Count, per environment, the qualifying steps among the last ``ceil(hold_time_s / step_dt)`` control steps."""

    def __init__(self, like: torch.Tensor):
        self.count = torch.zeros(like.shape[0], dtype=torch.long, device=like.device)
        self.span = torch.zeros_like(self.count)
        self.last_step = torch.full_like(self.count, -1)
        self.window = None

    def reset(self, env_ids=None) -> None:
        selected = torch.ones(self.count.shape, dtype=torch.bool, device=self.count.device)
        if env_ids is not None:
            selected = torch.zeros_like(selected)
            selected[env_ids] = True
        self.count = torch.where(selected, 0, self.count)
        self.span = torch.where(selected, 0, self.span)
        self.last_step = torch.where(selected, -1, self.last_step)
        if self.window is not None:
            self.window = self.window & ~selected.unsqueeze(1)

    def update(
        self,
        ok: torch.Tensor,
        step: int,
        step_dt: float,
        hold_time_s: float,
        share: float,
        episode_steps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Count step ``step`` once and return where ``share`` of the window's steps qualified."""
        if step_dt <= 0 or hold_time_s < 0 or not 0 < share <= 1:
            raise ValueError("step_dt must be positive, hold_time_s nonnegative and share in (0, 1]")
        length = max(1, math.ceil(hold_time_s / step_dt - 1e-9))
        required = max(1, math.ceil(share * length - 1e-9))
        if self.window is None or self.window.shape[1] != length:
            self.window = torch.zeros((ok.shape[0], length), dtype=torch.bool, device=ok.device)
        fresh = (self.last_step != step).unsqueeze(1)
        consecutive = (self.last_step == step - 1).unsqueeze(1)
        kept = torch.where(consecutive, self.window[:, :-1], False)
        window = torch.where(fresh, torch.cat((ok.unsqueeze(1), kept), dim=1), self.window)
        span = torch.where(fresh.squeeze(1), torch.where(consecutive.squeeze(1), self.span + 1, 1), self.span)
        span = span.clamp(max=length)
        if episode_steps is not None:
            episode_steps = episode_steps.to(span.device, span.dtype)
            window = window & (torch.arange(length, device=window.device).unsqueeze(0) < episode_steps.unsqueeze(1))
            span = torch.minimum(span, episode_steps)
        self.window, self.span = window, span
        self.count = window.sum(dim=1)
        self.last_step = torch.full_like(self.last_step, step)
        return (self.span >= length) & (self.count >= required)


def sustained(env, ok: torch.Tensor, hold_time_s: float, key: str, share: float = 1.0) -> torch.Tensor:
    """True where ``ok`` has held for ``hold_time_s`` seconds of control steps (state kept on the env under ``key``)."""
    if hold_time_s <= 0:
        return ok
    base = getattr(env, "unwrapped", env)
    holds = _holds(base)
    hold = holds.get(key)
    kind = SustainedHold if share >= 1.0 else SustainedShareHold
    if kind is SustainedShareHold and ok.ndim > 1:
        raise ValueError(f"hold {key!r}: a share hold counts one window per env; got ok of shape {tuple(ok.shape)}")
    if not isinstance(hold, kind) or hold.count.shape != ok.shape:
        hold = holds[key] = kind(ok)
    episode_steps = base.episode_length_buf
    ok = ok & (episode_steps > 0).reshape(episode_steps.shape[0], *([1] * (ok.ndim - 1)))
    if kind is SustainedHold:
        return hold.update(ok, base.common_step_counter, base.step_dt, hold_time_s, episode_steps)
    return hold.update(ok, base.common_step_counter, base.step_dt, hold_time_s, share, episode_steps)


@deterministic_reset
def reset_sustained(env, env_ids, key: str) -> None:
    """Reset-event body: clear the hold ``key`` for ``env_ids`` (all when None)."""
    hold = _holds(env).get(key)
    if hold is not None:
        hold.reset(env_ids)


def reset_all_sustained(env, env_ids=None) -> None:
    """Clear every sustained hold for ``env_ids`` (all when None)."""
    for hold in _holds(env).values():
        hold.reset(env_ids)


def sustained_count(env, key: str) -> torch.Tensor | None:
    """Consecutive qualifying steps of hold ``key`` per env (in its window, for a share hold), or None before its
    first evaluation."""
    hold = _holds(env).get(key)
    return None if hold is None else hold.count
