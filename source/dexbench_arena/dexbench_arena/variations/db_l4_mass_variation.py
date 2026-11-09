# SPDX-License-Identifier: Apache-2.0
"""Per-reset draw of what one parcel of a DB-L4 sequence weighs (GA-P)."""

from __future__ import annotations

import torch
from dataclasses import field
from typing import TYPE_CHECKING

import warp as wp
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab_newton.physics import NewtonManager

from isaaclab_arena.variations.choice_sampler import ChoiceSampler, ChoiceSamplerCfg
from isaaclab_arena.variations.object_mass_variation import ApplyObjectMassFromSampler
from isaaclab_arena.variations.variation_base import RunTimeVariationBase, VariationBaseCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

_CANNOT_DISABLE = (
    "l4_park_slab.mass_slot_<k> cannot be disabled: --l4_mass_draw attached it. Leave --l4_mass_draw unset for fixed "
    "masses (--l4_masses, or a training sequence's own)."
)


@configclass
class L4MassVariationCfg(VariationBaseCfg):
    """Configuration for L4MassVariation."""

    enabled: bool = True
    """On by construction, unlike Arena's ``VariationBaseCfg`` (off); see :class:`L4MassVariation`."""

    sampler_cfg: ChoiceSamplerCfg = field(default_factory=ChoiceSamplerCfg)
    """Uniform over the slot's multipliers."""


class L4MassSampler:
    """What a slot's parcel weighs at a multiplier drawn from ``choice`` [kg], as Arena's mass terms read a sampler:
    ``sample(num_samples, env_ids)`` -> (num_samples, 1). The draw itself is ``choice``'s, whose listeners record
    the multiplier and whose replay sampler a recorded replay binds (``replays_through``), so a replayed episode
    weighs what the recorded one did."""

    shape_per_sample = (1,)

    def __init__(self, choice: ChoiceSampler, nominal_mass_kg: float, multipliers: tuple[float, ...]):
        self.choice = choice
        self.nominal_mass_kg = float(nominal_mass_kg)
        self.multipliers = tuple(multipliers)

    @property
    def replays_through(self) -> ChoiceSampler:
        """The variation's sampler this one draws through, which the replay guard checks is bound."""
        return self.choice

    def set_replay_sampler(self, replay_sampler) -> None:
        """Replay recorded multipliers: bind ``replay_sampler`` to ``choice``, which draws them."""
        self.choice.set_replay_sampler(replay_sampler)

    def sample(self, num_samples: int, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        drawn = self.choice.sample(num_samples=num_samples, choices=self.multipliers, env_ids=env_ids)
        return torch.tensor([[self.nominal_mass_kg * m] for m in drawn], dtype=torch.float32).reshape(-1, 1)


class L4MassVariation(RunTimeVariationBase):
    """Weigh one slot's parcel at a multiplier drawn at every reset."""

    cfg: L4MassVariationCfg

    def __init__(
        self,
        slot: int,
        entity_name: str,
        nominal_mass_kg: float,
        is_polybag: bool,
        multipliers: tuple[float, ...],
        cfg: L4MassVariationCfg | None = None,
    ):
        if not nominal_mass_kg > 0.0 or not multipliers or not all(m > 0.0 for m in multipliers):
            raise ValueError(f"slot {slot}: a positive nominal mass and positive multipliers, got {nominal_mass_kg!r} "
                             f"and {multipliers!r}")
        self.slot = int(slot)
        self.entity_name = entity_name
        self.nominal_mass_kg = float(nominal_mass_kg)
        self.is_polybag = bool(is_polybag)
        self.multipliers = tuple(float(m) for m in multipliers)
        super().__init__(cfg=cfg if cfg is not None else L4MassVariationCfg(), name=f"mass_slot_{self.slot}")

    def apply_cfg(self, cfg: L4MassVariationCfg) -> None:
        if not cfg.enabled:
            raise ValueError(_CANNOT_DISABLE)
        super().apply_cfg(cfg)

    def disable(self) -> None:
        raise ValueError(_CANNOT_DISABLE)

    def build_event_cfg(self) -> tuple[str, EventTermCfg]:
        assert self._sampler is not None, f"L4MassVariation slot {self.slot} is enabled but has no sampler."
        sampler = L4MassSampler(self._sampler, self.nominal_mass_kg, self.multipliers)
        if self.is_polybag:
            term = EventTermCfg(func=ApplyL4MassFromSampler, mode="reset",
                                params={"sampler": sampler, "entity_name": self.entity_name})
        else:
            term = EventTermCfg(func=ApplyObjectMassFromSampler, mode="reset",
                                params={"asset_cfg": SceneEntityCfg(self.entity_name), "sampler": sampler,
                                        "recompute_inertia": True})
        return f"l4_mass_slot_{self.slot}_variation", term


class ApplyL4MassFromSampler(ManagerTermBase):
    """Event term: set a bag's particles to the drawn mass in the resetting environments, each scaled by one factor
    from the masses the build gave them (taken on the first call), so every draw scales from those, never from the
    previous episode's."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset = env.scene[cfg.params["entity_name"]]
        self._default_mass: torch.Tensor | None = None

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        sampler: L4MassSampler,
        entity_name: str,  # noqa: ARG002  (resolved in __init__)
    ):
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int32)
        else:
            env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.int32).reshape(-1)
        if env_ids.numel() == 0:
            return
        masses = sampler.sample(num_samples=len(env_ids), env_ids=env_ids)[:, 0]
        model = NewtonManager.get_model()
        mass = wp.to_torch(model.particle_mass)
        inverse = wp.to_torch(model.particle_inv_mass)
        count = int(self.asset.particles_per_object)
        offsets = torch.as_tensor(self.asset.particle_offsets, dtype=torch.long, device=mass.device)
        if self._default_mass is None:
            self._default_mass = torch.stack([mass[int(o) : int(o) + count].clone() for o in offsets])
        for row, env_id in enumerate(env_ids.long().tolist()):
            start = int(offsets[env_id])
            default = self._default_mass[env_id]
            scaled = default * (masses[row].to(mass.device) / default.sum())
            mass[start : start + count] = scaled
            inverse[start : start + count] = torch.where(scaled > 0.0, 1.0 / scaled, torch.zeros_like(scaled))
