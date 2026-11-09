# SPDX-License-Identifier: Apache-2.0
"""Per-reset choice of which kit-box compartment DB-M5 leaves empty."""

from __future__ import annotations

import torch
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from isaaclab_arena.variations.choice_sampler import ChoiceSampler

from dexbench_arena.variations.pose_layout_variation import ApplyPoseLayout, Pose7, PoseLayoutVariation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class ApplyKitBoxEmptyCompartment(ApplyPoseLayout):
    """Event term: load the kit box with the drawn compartment empty, and keep which it is."""

    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.empty_compartment = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        """Each env's compartment drawn at its last reset; -1 before its first."""

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_names: Sequence[str],
        layouts: Mapping[int, Sequence[Pose7]],
        choices: Sequence[int],
        sampler: ChoiceSampler,
    ) -> list[int]:
        drawn = super().__call__(env, env_ids, asset_names, layouts, choices, sampler)
        if drawn:
            if env_ids is None:
                env_ids = torch.arange(env.scene.num_envs)
            index = torch.as_tensor(env_ids, dtype=torch.long, device=self.empty_compartment.device).reshape(-1)
            updated = self.empty_compartment.clone()
            updated[index] = torch.as_tensor(drawn, dtype=torch.long, device=updated.device)
            self.empty_compartment = updated
        return drawn


class KitBoxEmptyCompartmentVariation(PoseLayoutVariation):
    """Draw the empty compartment at every reset; ``layouts[c]`` loads the parts with ``c`` empty."""

    event_term = ApplyKitBoxEmptyCompartment

    def __init__(self, part_names: Sequence[str], layouts, cfg=None, name: str = "empty_compartment"):
        super().__init__(part_names, layouts, cfg=cfg, name=name)


def empty_compartment_event(env) -> tuple[ApplyKitBoxEmptyCompartment, ChoiceSampler] | None:
    """The env's ``ApplyKitBoxEmptyCompartment`` reset term and the sampler it draws with, or None without one
    (the kit box's loading is not varied)."""

    base = env.unwrapped
    events = getattr(base, "event_manager", None)
    for name in (events.active_terms.get("reset", ()) if events is not None else ()):
        term_cfg = events.get_term_cfg(name)
        if isinstance(term_cfg.func, ApplyKitBoxEmptyCompartment):
            return term_cfg.func, term_cfg.params["sampler"]
    return None


def replay_empty_compartment(env, compartment: int | None) -> bool:
    """Make every later draw of the env's empty compartment ``compartment`` (Arena's replay sampler), or live again
    with None; False when the env has no such variation. The draw must be one of the variation's choices."""

    event = empty_compartment_event(env)
    if event is None:
        return False
    _, sampler = event
    sampler.set_replay_sampler(None if compartment is None else (lambda count, env_ids: [int(compartment)] * count))
    return True
