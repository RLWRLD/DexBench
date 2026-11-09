# SPDX-License-Identifier: Apache-2.0
"""Pink IK task costs and gains for the RBY1-M arms, as named profiles."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Rby1mIkProfile:
    """Costs of the two wrist frame tasks, the damping task and the null-space posture task."""

    position_cost: float
    orientation_cost: float
    lm_damping: float
    gain: float
    damping_cost: float
    posture_cost: float


RBY1M_IK_TELEOP = Rby1mIkProfile(
    position_cost=8.0,
    orientation_cost=1.0,
    lm_damping=12.0,
    gain=0.5,
    damping_cost=0.5,
    posture_cost=0.5,
)
"""Headset teleoperation: smooth over the operator's hand jitter."""

RBY1M_IK_GENERATION = replace(RBY1M_IK_TELEOP, gain=1.0, lm_damping=1.0)
"""Mimic data generation: track the transformed recorded trajectories faithfully."""

RBY1M_IK_PROFILES: dict[str, Rby1mIkProfile] = {
    "teleop": RBY1M_IK_TELEOP,
    "generation": RBY1M_IK_GENERATION,
}


def resolve_ik_profile(name: str | Rby1mIkProfile) -> Rby1mIkProfile:
    """The profile for ``name``; a profile instance passes through."""
    if isinstance(name, Rby1mIkProfile):
        return name
    try:
        return RBY1M_IK_PROFILES[name]
    except KeyError:
        raise ValueError(f"unknown IK profile {name!r}; choose one of {sorted(RBY1M_IK_PROFILES)}") from None
