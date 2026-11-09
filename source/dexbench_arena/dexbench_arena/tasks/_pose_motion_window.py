# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Measure motion from consecutive poses, with per-environment reset handling."""

import math

import torch


class PoseMotionWindow:
    """Peak translation speed over a full window of consecutive control steps."""

    def __init__(self, window_seconds: float, step_dt: float):
        if not math.isfinite(window_seconds) or window_seconds <= 0:
            raise ValueError("Motion window must be finite and positive")
        if not math.isfinite(step_dt) or step_dt <= 0:
            raise ValueError("Control step duration must be finite and positive")
        self.step_dt = step_dt
        self.steps = max(1, math.ceil(window_seconds / step_dt))
        self.history = None
        self.head = None
        self.previous = None
        self.episode_steps = None
        self.count = None
        self.result = None

    def update(self, positions: torch.Tensor, episode_steps: torch.Tensor) -> torch.Tensor:
        if self.previous is None or self.previous.shape != positions.shape:
            self.previous = positions.clone()
            self.episode_steps = episode_steps.clone()
            self.count = torch.zeros_like(episode_steps)
            self.result = torch.full(positions.shape[:2], float("inf"), device=positions.device)
            self.history = self.result.unsqueeze(0).repeat(self.steps, 1, 1)
            self.head = torch.zeros_like(episode_steps)
            return self.result

        unchanged = (episode_steps == self.episode_steps) & (positions == self.previous).all(dim=(1, 2))
        if bool(unchanged.all()):
            return self.result

        consecutive = episode_steps == self.episode_steps + 1
        speed = torch.linalg.vector_norm(positions - self.previous, dim=-1) / self.step_dt
        speed[~consecutive] = float("inf")
        changed_ids = torch.nonzero(~unchanged, as_tuple=True)[0]
        self.history[self.head[changed_ids], changed_ids] = speed[changed_ids]
        self.head[changed_ids] = (self.head[changed_ids] + 1) % self.steps
        self.count = torch.where(unchanged, self.count, torch.where(consecutive, self.count + 1, 0))
        result = self.history.amax(dim=0)
        result[self.count < self.steps] = float("inf")
        result[unchanged] = self.result[unchanged]
        self.result = result
        self.previous = positions.clone()
        self.episode_steps = episode_steps.clone()
        return result
