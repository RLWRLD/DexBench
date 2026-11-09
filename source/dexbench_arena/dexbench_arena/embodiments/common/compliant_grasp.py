"""Bounded closing preload for a yielding support hand."""

from dataclasses import dataclass
import math

import torch


@dataclass
class CompliantGraspCfg:
    """Support-hand profile; the embodiment applies actuator and material values."""

    anchor_to_reset: bool = False
    rolling_friction_m: float | None = None
    friction: float = 0.9
    stiffness: float = 8.0
    damping: float = 0.3
    effort_limit_nm: float = 0.25
    preload_rad: float = 0.005
    travel_rad: float = 0.75

    def __post_init__(self):
        if self.rolling_friction_m is not None and (
            not isinstance(self.rolling_friction_m, (int, float))
            or not math.isfinite(self.rolling_friction_m)
            or self.rolling_friction_m <= 0
        ):
            raise ValueError("rolling_friction_m must be finite and positive")
        if not isinstance(self.anchor_to_reset, bool):
            raise ValueError("anchor_to_reset must be a bool")
        for name in (
            "friction",
            "stiffness",
            "damping",
            "effort_limit_nm",
            "preload_rad",
            "travel_rad",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"Compliant grasp {name} must be finite and positive")


def compliant_grasp_targets(
    positions,
    nominal,
    limits,
    closing_direction,
    preload_rad,
    travel_rad=0.45,
    *,
    grasp_reference=None,
):
    """Bias each flexion joint toward closure within a bounded grasp envelope."""
    lower = torch.maximum(limits[..., 0], nominal - travel_rad)
    upper = torch.minimum(limits[..., 1], nominal + travel_rad)
    if grasp_reference is not None:
        positions = torch.where(
            closing_direction > 0,
            torch.maximum(positions, grasp_reference),
            torch.where(
                closing_direction < 0,
                torch.minimum(positions, grasp_reference),
                positions,
            ),
        )
    biased = positions + closing_direction * preload_rad
    target = torch.where(closing_direction == 0, nominal, biased)
    return torch.minimum(torch.maximum(target, lower), upper)


@dataclass
class AdaptiveGraspCfg:
    """Opt-in pose-feedback assistance, separate from the qualified grip profile."""

    hold_preload_rad: float = 0.001
    slip_speed_m_s: float = 0.010
    full_slip_speed_m_s: float = 0.060
    rearrangement_margin_m: float = 0.008
    escape_margin_m: float = 0.09
    max_extra_preload_rad: float = 0.020
    tighten_rate_rad_s: float = 0.060
    relax_rate_rad_s: float = 0.015
    filter_seconds: float = 0.06
    proximity_m: float = 0.025
    target_protection_m: float = 0.010

    def __post_init__(self):
        for name, value in vars(self).items():
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Adaptive grasp {name} must be finite and positive")
        if self.full_slip_speed_m_s <= self.slip_speed_m_s:
            raise ValueError("Full slip speed must exceed the slip deadband")
        if self.escape_margin_m <= self.rearrangement_margin_m:
            raise ValueError("Escape margin must exceed the rearrangement margin")


class AdaptiveGraspController:
    """Batched per-finger preload with rate limits and target-aware protection."""

    @torch.inference_mode(False)
    def __init__(self, relative_centers, finger_distances, cfg=None):
        self.cfg = cfg or AdaptiveGraspCfg()
        shape = finger_distances.shape[:2]
        self.extra = relative_centers.new_zeros(shape)
        self.hold_bias = relative_centers.new_full(shape, self.cfg.hold_preload_rad)
        self.risk = relative_centers.new_zeros(shape)
        self.initial_centers = relative_centers.clone()
        self.previous_displacement = relative_centers.new_zeros(relative_centers.shape[:2])
        self.initial_radius = torch.linalg.vector_norm(relative_centers, dim=-1).clone()
        self.previous_radius = self.initial_radius.clone()
        self.ready = torch.ones(relative_centers.shape[0], dtype=torch.bool, device=relative_centers.device)

    def invalidate(self, env_ids=None):
        if env_ids is None:
            self.ready[:] = False
        else:
            self.ready[env_ids] = False

    def update(self, relative_centers, target_mask, finger_distances, dt):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Adaptive feedback needs a positive timestep")
        if target_mask.shape != relative_centers.shape[:2] or not bool(target_mask.any(dim=1).all()):
            raise ValueError("Adaptive feedback needs an explicit target in every environment")
        if not all(bool(torch.isfinite(x).all()) for x in (relative_centers, finger_distances)):
            raise ValueError("Adaptive grasp feedback contains non-finite geometry")
        cfg = self.cfg
        radius = torch.linalg.vector_norm(relative_centers, dim=-1)
        reset = ~self.ready
        self.initial_centers[reset] = relative_centers[reset]
        self.previous_displacement[reset] = 0
        self.initial_radius[reset] = radius[reset]
        self.previous_radius[reset] = radius[reset]
        self.extra[reset] = 0
        self.risk[reset] = 0
        self.ready[:] = True
        outward = (radius - self.previous_radius) / dt
        self.previous_radius.copy_(radius)
        speed_risk = ((outward - cfg.slip_speed_m_s) / (cfg.full_slip_speed_m_s - cfg.slip_speed_m_s)).clamp(0, 1)
        displacement = ((radius - self.initial_radius - cfg.rearrangement_margin_m) / cfg.rearrangement_margin_m).clamp(0, 1)
        slow_escape = displacement * ((outward - 0.002) / 0.008).clamp(0, 1)
        drift = torch.linalg.vector_norm(relative_centers - self.initial_centers, dim=-1)
        drift_speed = (drift - self.previous_displacement) / dt
        self.previous_displacement.copy_(drift)
        drift_risk = ((drift - cfg.rearrangement_margin_m) / cfg.rearrangement_margin_m).clamp(0, 1) * ((drift_speed - 0.002) / 0.008).clamp(0, 1)
        retained = ~target_mask & (radius < self.initial_radius + cfg.escape_margin_m) & (radius < 0.22)
        object_risk = torch.maximum(torch.maximum(speed_risk, slow_escape), drift_risk) * retained
        near = torch.exp(-finger_distances / cfg.proximity_m)
        finger_risk = (near * object_risk[:, None, :]).amax(dim=-1)
        target_distance = finger_distances.masked_fill(~target_mask[:, None, :], float('inf')).amin(dim=-1)
        other_distance = finger_distances.masked_fill(~retained[:, None, :], float('inf')).amin(dim=-1)
        protection = ((target_distance - other_distance + cfg.target_protection_m) / (2 * cfg.target_protection_m)).clamp(0, 1)
        protection = torch.where(torch.isfinite(other_distance), protection, 0)
        self.hold_bias.copy_(cfg.hold_preload_rad * protection)
        finger_risk *= protection
        self.risk.lerp_(finger_risk, 1 - math.exp(-dt / cfg.filter_seconds))
        desired = torch.where(self.risk > 0.05, self.risk * cfg.max_extra_preload_rad, 0)
        increment = (desired - self.extra).clamp(-cfg.relax_rate_rad_s * dt, cfg.tighten_rate_rad_s * dt)
        self.extra.add_(increment).clamp_(0, cfg.max_extra_preload_rad)
