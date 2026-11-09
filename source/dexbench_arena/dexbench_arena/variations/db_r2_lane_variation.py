# SPDX-License-Identifier: Apache-2.0
"""Per-reset draw of the Shelf2 lane that holds the DB-R2 target."""

from __future__ import annotations

import torch
from dataclasses import field
from typing import TYPE_CHECKING

import warp as wp
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.variations.choice_sampler import ChoiceSampler, ChoiceSamplerCfg
from isaaclab_arena.variations.variation_base import RunTimeVariationBase, VariationBaseCfg

from dexbench_arena.layouts.db_r2_layout import (
    FC_260821_R2_OPENING_FRAME_INSTANCE_NAME,
    RetailR2Stocking,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

_CANNOT_DISABLE = (
    "r2_target_drink.lane cannot be disabled: the scene was built for the draw (one identity per drink across "
    "lanes, cleared backdrop). Leave --r2_lane_draw unset for the set in --r2_target_lane, or give one lane "
    "(--r2_lane_draw 4) for a fixed lane of the draw scene."
)


@configclass
class R2LaneVariationCfg(VariationBaseCfg):
    """Configuration for R2LaneVariation."""

    enabled: bool = True
    """On by construction, unlike Arena's ``VariationBaseCfg`` (off): the variation exists only
    when ``--r2_lane_draw`` names the lanes, and naming them is the request for the draw.
    It cannot be turned off; see :class:`R2LaneVariation`."""

    sampler_cfg: ChoiceSamplerCfg = field(default_factory=ChoiceSamplerCfg)
    """Uniform over the plan's lanes."""

    dataset_values: dict[str, int] = field(default_factory=dict)
    """Each lane label's Shelf2 lane number, which the demo HDF5 records in place of the label
    (``VariationSamplesRecorder``). Set from the plan, whatever a run config says."""


class R2LaneVariation(RunTimeVariationBase):
    """Move the DB-R2 dynamic set to a lane drawn at every reset."""

    cfg: R2LaneVariationCfg

    def __init__(
        self,
        plan: RetailR2Stocking,
        object_name: str,
        cfg: R2LaneVariationCfg | None = None,
        name: str = "lane",
    ):
        self.plan = plan
        self.object_name = object_name
        super().__init__(cfg=cfg if cfg is not None else R2LaneVariationCfg(), name=name)

    def apply_cfg(self, cfg: R2LaneVariationCfg) -> None:
        if not cfg.enabled:
            raise ValueError(_CANNOT_DISABLE)
        super().apply_cfg(cfg)
        self.cfg.dataset_values = dict(zip(self.plan.labels, self.plan.lanes))

    def disable(self) -> None:
        raise ValueError(_CANNOT_DISABLE)

    def build_event_cfg(self) -> tuple[str, EventTermCfg]:
        assert self._sampler is not None, f"R2LaneVariation on '{self.object_name}' is enabled but has no sampler."
        seats = tuple(
            tuple((p.instance_name, p.position_xyz) for p in lane_seats)
            + ((FC_260821_R2_OPENING_FRAME_INSTANCE_NAME, frame_xyz),)
            for lane_seats, frame_xyz in zip(self.plan.seats, self.plan.opening_frames_xyz)
        )
        return (
            f"{self.object_name}_lane_variation",
            EventTermCfg(
                func=ApplyR2LaneFromSampler,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg(self.object_name),
                    "sampler": self._sampler,
                    "labels": self.plan.labels,
                    "seats": seats,
                },
            ),
        )


class ApplyR2LaneFromSampler(ManagerTermBase):
    """Event term: seat the DB-R2 drinks and the opening frame in one drawn lane."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        seats = cfg.params["seats"]
        self.labels: tuple[str, ...] = tuple(cfg.params["labels"])
        assert len(seats) == len(self.labels), "R2LaneVariation: one seat table per lane"
        names = tuple(name for name, _ in seats[0])
        assert all(tuple(n for n, _ in lane) == names for lane in seats), "every lane seats the same entities"
        missing = [name for name in names if name not in env.scene.keys()]
        assert not missing, f"R2LaneVariation: the scene has no {missing}"
        self.entities = [env.scene[name] for name in names]
        device = self.entities[0].device
        self.seats = torch.tensor(
            [[position for _, position in lane] for lane in seats], dtype=torch.float32, device=device
        )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,  # noqa: ARG002  (resolved in __init__)
        sampler: ChoiceSampler,
        labels: tuple[str, ...],  # noqa: ARG002
        seats: tuple,  # noqa: ARG002
    ):
        device = self.seats.device
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=device, dtype=torch.int32)
        else:
            env_ids = torch.as_tensor(env_ids, device=device, dtype=torch.int32).reshape(-1)
        if env_ids.numel() == 0:
            return
        drawn = sampler.sample(num_samples=len(env_ids), choices=self.labels, env_ids=env_ids)
        lane_index = torch.tensor([self.labels.index(label) for label in drawn], device=device)
        env_index = env_ids.long()
        origins = env.scene.env_origins[env_index]
        for index, entity in enumerate(self.entities):
            pose = wp.to_torch(entity.data.default_root_pose)[env_index].clone()
            pose[:, :3] = self.seats[lane_index, index] + origins
            entity.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
            entity.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros(len(env_ids), 6, device=pose.device), env_ids=env_ids
            )
