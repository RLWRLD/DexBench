# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Shared configuration and six-slot state bookkeeping for measured M3 screw joints."""

from dataclasses import dataclass, fields
import math

import torch


@dataclass(frozen=True)
class FasteningCfg:
    active_bolt_indices: tuple[int, ...] = (0, 1)
    travel_m: float = 0.005
    turns_to_seat: float = 4.0
    motor_rps: float = 4.0
    lateral_tolerance_m: float = 0.002
    axial_tolerance_m: float = 0.004
    alignment_tolerance_deg: float = 10.0
    contact_threshold_n: float = 0.2

    def __post_init__(self):
        if (not isinstance(self.active_bolt_indices, (tuple, list)) or not self.active_bolt_indices
                or any(type(i) is not int or not 0 <= i < 6 for i in self.active_bolt_indices)
                or tuple(sorted(set(self.active_bolt_indices))) != tuple(self.active_bolt_indices)):
            raise ValueError("active_bolt_indices must be unique increasing CAD indices in [0, 5]")
        for field in fields(self):
            if field.name == "active_bolt_indices":
                continue
            value = getattr(self, field.name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field.name} must be positive and finite")
        if self.alignment_tolerance_deg >= 90:
            raise ValueError("alignment_tolerance_deg must be less than 90")


class FasteningModel:
    """Six stable CAD slots, with only configured task bolts eligible for engagement."""

    def __init__(self, seated_heads, num_envs: int, cfg: FasteningCfg = FasteningCfg(), device="cpu"):
        self.cfg = cfg
        self.seated_heads = torch.as_tensor(seated_heads, dtype=torch.float32, device=device)
        if self.seated_heads.shape != (6, 3) or not torch.isfinite(self.seated_heads).all():
            raise ValueError("seated_heads must contain six finite xyz positions (B1 through B6)")
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.turns = torch.zeros((num_envs, 6), device=device)
        self.engaged = torch.zeros_like(self.turns, dtype=torch.bool)
        self.active = torch.zeros(6, dtype=torch.bool, device=device)
        self.active[list(cfg.active_bolt_indices)] = True

    @property
    def remaining_m(self):
        return self.cfg.travel_m * (1 - self.turns / self.cfg.turns_to_seat) * self.active

    @property
    def head_positions(self):
        positions = self.seated_heads.expand(self.turns.shape[0], -1, -1).clone()
        positions[..., 1] -= self.remaining_m
        return positions

    @property
    def seated(self):
        return (self.turns >= self.cfg.turns_to_seat) & (self.remaining_m <= 1e-7)

    @property
    def success(self):
        return self.seated[:, self.active].all(dim=1)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.turns[ids] = 0
        self.engaged[ids] = False
