# SPDX-License-Identifier: Apache-2.0
"""Per-reset draw of how DB-L4 puts one parcel of the sequence down (GA-S)."""

from __future__ import annotations

import torch
from dataclasses import field
from typing import TYPE_CHECKING

from isaaclab.managers import EventTermCfg, ManagerTermBase
from isaaclab.utils.configclass import configclass

from isaaclab_arena.variations.choice_sampler import ChoiceSampler, ChoiceSamplerCfg
from isaaclab_arena.variations.variation_base import RunTimeVariationBase, VariationBaseCfg

from dexbench_arena.layouts.e2_logistics_l4_layout import ROTATION_LABELS
from dexbench_arena.tasks.deliver_parcels_barcode_up import relay_term

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

_CANNOT_DISABLE = (
    "l4_park_slab.rotation_slot_<k> cannot be disabled: --l4_rotation_draw attached it and the scene put the parcels "
    "down for it. Leave --l4_rotation_draw unset for a sequence's fixed rotations (--l4_rotations, or a training "
    "sequence's own)."
)


@configclass
class L4RotationVariationCfg(VariationBaseCfg):
    """Configuration for L4RotationVariation."""

    enabled: bool = True
    """On by construction, unlike Arena's ``VariationBaseCfg`` (off); see :class:`L4RotationVariation`."""

    sampler_cfg: ChoiceSamplerCfg = field(default_factory=ChoiceSamplerCfg)
    """Uniform over the slot's labels."""

    dataset_values: dict[str, int] = field(default_factory=dict)
    """Each label's index in ``ROTATION_LABELS``, which the demo HDF5 records in place of the label. Set from the
    labels, whatever a run config says."""


class L4RotationVariation(RunTimeVariationBase):
    """Put one slot's parcel down at a rotation drawn at every reset."""

    cfg: L4RotationVariationCfg

    def __init__(
        self,
        slot: int,
        labels: tuple[str, ...],
        stage_poses: tuple[tuple[float, ...], ...],
        cfg: L4RotationVariationCfg | None = None,
    ):
        if len(labels) != len(stage_poses) or not labels:
            raise ValueError(f"slot {slot}: one stage pose per label, at least one ({len(labels)} labels, "
                             f"{len(stage_poses)} poses)")
        unknown = sorted(set(labels) - set(ROTATION_LABELS))
        if unknown:
            raise ValueError(f"slot {slot}: {unknown} are not rotation labels")
        self.slot = int(slot)
        self.labels = tuple(labels)
        self.stage_poses = tuple(tuple(float(v) for v in pose) for pose in stage_poses)
        super().__init__(cfg=cfg if cfg is not None else L4RotationVariationCfg(), name=f"rotation_slot_{self.slot}")

    def apply_cfg(self, cfg: L4RotationVariationCfg) -> None:
        if not cfg.enabled:
            raise ValueError(_CANNOT_DISABLE)
        super().apply_cfg(cfg)
        self.cfg.dataset_values = {label: ROTATION_LABELS.index(label) for label in self.labels}

    def disable(self) -> None:
        raise ValueError(_CANNOT_DISABLE)

    def build_event_cfg(self) -> tuple[str, EventTermCfg]:
        assert self._sampler is not None, f"L4RotationVariation slot {self.slot} is enabled but has no sampler."
        return (
            f"l4_rotation_slot_{self.slot}_variation",
            EventTermCfg(
                func=ApplyL4RotationFromSampler,
                mode="reset",
                params={"sampler": self._sampler, "slot": self.slot, "labels": self.labels,
                        "stage_poses": self.stage_poses},
            ),
        )


class ApplyL4RotationFromSampler(ManagerTermBase):
    """Event term: write one slot's drawn stage pose into the relay's table for the resetting environments."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.labels: tuple[str, ...] = tuple(cfg.params["labels"])
        self.poses = torch.tensor(cfg.params["stage_poses"], dtype=torch.float32, device=env.device)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        sampler: ChoiceSampler,
        slot: int,
        labels: tuple[str, ...],  # noqa: ARG002  (resolved in __init__)
        stage_poses: tuple,  # noqa: ARG002
    ):
        table = relay_term(env).stage_poses
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=self.poses.device, dtype=torch.int32)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.poses.device, dtype=torch.int32).reshape(-1)
        if env_ids.numel() == 0:
            return
        drawn = sampler.sample(num_samples=len(env_ids), choices=self.labels, env_ids=env_ids)
        index = torch.tensor([self.labels.index(label) for label in drawn], device=self.poses.device)
        table[env_ids.long(), slot] = self.poses[index]
