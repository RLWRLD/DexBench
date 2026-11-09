# SPDX-License-Identifier: Apache-2.0
"""Per-reset choice among precomputed pose layouts for a set of rigid objects."""

from __future__ import annotations

import torch
from collections.abc import Mapping, Sequence
from dataclasses import field
from typing import TYPE_CHECKING

from isaaclab.managers import EventTermCfg, ManagerTermBase
from isaaclab.utils.configclass import configclass

from isaaclab_arena.variations.choice_sampler import ChoiceSampler, ChoiceSamplerCfg
from isaaclab_arena.variations.variation_base import RunTimeVariationBase, VariationBaseCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

Pose7 = tuple[float, float, float, float, float, float, float]
"""``(px, py, pz, qx, qy, qz, qw)`` in the env-local frame."""


class ApplyPoseLayout(ManagerTermBase):
    """Event term: write one drawn layout per resetting env to the objects, at rest."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.assets = [env.scene[name] for name in cfg.params["asset_names"]]
        layouts: Mapping[int, Sequence[Pose7]] = cfg.params["layouts"]
        self._row = {key: row for row, key in enumerate(layouts)}
        self._poses = torch.tensor(list(layouts.values()), dtype=torch.float32, device=self.assets[0].device)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_names: Sequence[str],  # noqa: ARG002
        layouts: Mapping[int, Sequence[Pose7]],  # noqa: ARG002
        choices: Sequence[int],
        sampler: ChoiceSampler,
    ) -> list[int]:
        """Draw and write the layouts; returns the key drawn for each of ``env_ids``."""

        device = self.assets[0].device
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=device, dtype=torch.int32)
        else:
            env_ids = torch.as_tensor(env_ids, device=device, dtype=torch.int32).reshape(-1)
        if env_ids.numel() == 0:
            return []
        drawn = [int(key) for key in sampler.sample(num_samples=len(env_ids), choices=choices, env_ids=env_ids)]
        poses = self._poses[torch.tensor([self._row[key] for key in drawn], device=device)]
        origins = env.scene.env_origins[env_ids.long()].to(device)
        for index, asset in enumerate(self.assets):
            pose = poses[:, index].clone()
            pose[:, :3] += origins
            asset.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
            asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros_like(pose[:, :6]), env_ids=env_ids)
        return drawn


@configclass
class PoseLayoutVariationCfg(VariationBaseCfg):
    """Configuration for PoseLayoutVariation."""

    choices: list[int] = field(default_factory=list)
    """Keys of the layouts to draw from, uniformly; empty draws from every layout."""

    sampler_cfg: ChoiceSamplerCfg = field(default_factory=ChoiceSamplerCfg)


class PoseLayoutVariation(RunTimeVariationBase):
    """Put ``asset_names`` at one of ``layouts`` at every reset."""

    cfg: PoseLayoutVariationCfg

    event_term: type[ManagerTermBase] = ApplyPoseLayout
    """The event term class that writes the drawn layout; subclasses swap in their own."""

    def __init__(
        self,
        asset_names: Sequence[str],
        layouts: Mapping[int, Sequence[Pose7]],
        cfg: PoseLayoutVariationCfg | None = None,
        name: str = "pose_layout",
    ):
        super().__init__(cfg=cfg if cfg is not None else PoseLayoutVariationCfg(), name=name)
        self.asset_names = tuple(asset_names)
        self.layouts = {int(key): tuple(tuple(float(v) for v in pose) for pose in poses) for key, poses in layouts.items()}
        if not self.layouts:
            raise ValueError(f"{type(self).__name__} on {self.asset_names}: no layouts")
        for key, poses in self.layouts.items():
            if len(poses) != len(self.asset_names) or any(len(pose) != 7 for pose in poses):
                raise ValueError(
                    f"{type(self).__name__}: layout {key} has {len(poses)} poses for {len(self.asset_names)} objects; "
                    "each is (px, py, pz, qx, qy, qz, qw)"
                )

    def resolved_choices(self) -> list[int]:
        """The keys a reset draws from: ``cfg.choices``, or every layout; ValueError for a key with no layout."""

        choices = [int(key) for key in self.cfg.choices] or sorted(self.layouts)
        unknown = sorted(set(choices) - set(self.layouts))
        if unknown:
            raise ValueError(f"{type(self).__name__} '{self.name}': no layout {unknown}; layouts {sorted(self.layouts)}")
        return choices

    def build_event_cfg(self) -> tuple[str, EventTermCfg]:
        event_cfg = EventTermCfg(
            func=self.event_term,
            mode="reset",
            params={
                "asset_names": self.asset_names,
                "layouts": self.layouts,
                "choices": self.resolved_choices(),
                "sampler": self.sampler,
            },
        )
        return f"{self.asset_names[0]}_{self.name}_variation", event_cfg
