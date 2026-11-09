# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""DB-L4's parcel relay: which parcel is out, when it is judged, when the next one comes."""

from __future__ import annotations

import math

import torch

from dexbench_arena.layouts.e2_logistics_l4_layout import (
    CARRY_FRACTION,
    FALL_CLEARANCE_M,
    JUDGE_DWELL_S,
    L4_BELT_TOP_Z_M,
    L4_BELT_Y_RANGE_M,
    L4_JUDGE_LINE_X_M,
    MAX_BARCODE_ANGLE_DEG,
    PARCEL_TIMEOUT_S,
    SEQUENCE_INTERVAL_S,
    SETTLE_EPS_M,
)

PENDING, FAILED, DELIVERED = -1, 0, 1


def at_judging_line(max_x, bottom_z, centre_y):
    """Whether a parcel whose bounds end at ``max_x``, whose lowest point is ``bottom_z`` and whose centre is at
    ``centre_y`` lies on the belt at the judging line: past the line, its lowest point within SETTLE of the belt top
    (above or below), its centre over FB1. Floats or tensors of the same shape (the relay's)."""

    return ((max_x >= L4_JUDGE_LINE_X_M)
            & (bottom_z <= L4_BELT_TOP_Z_M + SETTLE_EPS_M) & (bottom_z >= L4_BELT_TOP_Z_M - SETTLE_EPS_M)
            & (centre_y >= L4_BELT_Y_RANGE_M[0]) & (centre_y <= L4_BELT_Y_RANGE_M[1]))


def carried(advance_m, elapsed_s, belt_speed_m_s):
    """Whether a parcel whose leading edge went ``advance_m`` in the ``elapsed_s`` since it reached the line moved
    at CARRY_FRACTION of ``belt_speed_m_s`` or more on average. Floats or tensors of the same shape."""

    return advance_m >= CARRY_FRACTION * belt_speed_m_s * elapsed_s


def fallen(bottom_z):
    """Whether a parcel's lowest point is FALL_CLEARANCE below the belt top. A float or a tensor."""

    return bottom_z < L4_BELT_TOP_Z_M - FALL_CLEARANCE_M


class ParcelRelay:
    """Per-environment relay state of ``num_parcels`` parcels; every time is the episode's, in seconds."""

    def __init__(self, num_envs: int, num_parcels: int, belt_speed_m_s: float, device: torch.device | str = "cpu"):
        shape = (num_envs, num_parcels)
        self.num_parcels = num_parcels
        self.belt_speed_m_s = float(belt_speed_m_s)
        """The belt's speed [m/s], which ``carried`` measures a parcel at the line against."""
        self.staged_at = torch.full(shape, math.nan, device=device)
        """When each parcel was put down; NaN while it waits."""
        self.judged_at = torch.full(shape, math.nan, device=device)
        """When each parcel was judged; NaN until then."""
        self.verdict = torch.full(shape, PENDING, dtype=torch.long, device=device)
        self.at_line_since = torch.full(shape, math.nan, device=device)
        """When each parcel last came to lie on the belt at the judging line; NaN while it does not."""
        self.at_line_x = torch.full(shape, math.nan, device=device)
        """Each parcel's leading edge [m] when it last came to lie at the line; NaN while it does not."""

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        rows = slice(None) if env_ids is None else env_ids
        self.staged_at[rows] = math.nan
        self.judged_at[rows] = math.nan
        self.verdict[rows] = PENDING
        self.at_line_since[rows] = math.nan
        self.at_line_x[rows] = math.nan

    @property
    def active(self) -> torch.Tensor:
        """(envs, parcels) the parcel that is out: put down and not judged yet (at most one per environment)."""
        return ~torch.isnan(self.staged_at) & (self.verdict == PENDING)

    @property
    def delivered(self) -> torch.Tensor:
        return self.verdict == DELIVERED

    @property
    def failed(self) -> torch.Tensor:
        return self.verdict == FAILED

    @property
    def finished(self) -> torch.Tensor:
        """(envs,) every parcel judged."""
        return (self.verdict != PENDING).all(dim=1)

    def advance(
        self, now: torch.Tensor, max_x: torch.Tensor, bottom_z: torch.Tensor, centre_y: torch.Tensor,
        barcode_up_cos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Judge the parcel that is out and say which parcel to put down."""
        now = now[:, None].expand_as(self.staged_at)
        active = self.active
        lying = active & at_judging_line(max_x, bottom_z, centre_y)
        arrived = lying & torch.isnan(self.at_line_since)
        nan = torch.full_like(self.at_line_since, math.nan)
        self.at_line_since = torch.where(lying, torch.where(arrived, now, self.at_line_since), nan)
        self.at_line_x = torch.where(lying, torch.where(arrived, max_x, self.at_line_x), nan)
        elapsed = now - self.at_line_since
        settled = (lying & (elapsed >= JUDGE_DWELL_S - 1e-6)
                   & carried(max_x - self.at_line_x, elapsed, self.belt_speed_m_s))
        upright = barcode_up_cos >= math.cos(math.radians(MAX_BARCODE_ANGLE_DEG))
        fell = active & fallen(bottom_z)
        timed_out = active & (now - self.staged_at >= PARCEL_TIMEOUT_S)
        judged = settled | fell | timed_out
        self.verdict = torch.where(judged, torch.where(settled & upright & ~fell, DELIVERED, FAILED), self.verdict)
        self.judged_at = torch.where(judged, now, self.judged_at)

        waiting = torch.isnan(self.staged_at)
        first = waiting & (waiting.cumsum(dim=1) == 1)
        previous_judged_at = torch.cat((torch.full_like(now[:, :1], -math.inf), self.judged_at[:, :-1]), dim=1)
        stage = first & (now - previous_judged_at >= SEQUENCE_INTERVAL_S - 1e-6)
        self.staged_at = torch.where(stage, now, self.staged_at)
        return judged, stage
