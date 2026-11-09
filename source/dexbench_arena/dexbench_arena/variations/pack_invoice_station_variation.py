# SPDX-License-Identifier: Apache-2.0
"""Per-reset variations of the pack-invoice bench station that are not one rigid object's planar pose."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import field
from typing import TYPE_CHECKING

import warp as wp
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.variations.continuous_sampler import ContinuousSampler
from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg
from isaaclab_arena.variations.variation_base import RunTimeVariationBase, VariationBaseCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _env_ids(env, env_ids, device) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.scene.num_envs, device=device, dtype=torch.int32)
    return torch.as_tensor(env_ids, device=device, dtype=torch.int32).reshape(-1)


@configclass
class BenchHeightVariationCfg(VariationBaseCfg):
    """Configuration for BenchHeightVariation."""

    sampler_cfg: UniformSamplerCfg = field(default_factory=lambda: UniformSamplerCfg(low=[0.0], high=[0.0]))
    """Uniform ``[dz]`` [m] added to the bench's spawn height."""


class BenchHeightVariation(RunTimeVariationBase):
    """Raise or lower a kinematic bench at reset, and what stands on it."""

    cfg: BenchHeightVariationCfg

    def __init__(
        self,
        bench_name: str,
        carried: Sequence[str] = (),
        cfg: BenchHeightVariationCfg | None = None,
        name: str = "height",
    ):
        super().__init__(cfg=cfg if cfg is not None else BenchHeightVariationCfg(), name=name)
        self.bench_name = bench_name
        self.carried = tuple(carried)

    def build_event_cfg(self) -> tuple[str, EventTermCfg]:
        assert self._sampler is not None, f"BenchHeightVariation on '{self.bench_name}' is enabled but has no sampler."
        return f"{self.bench_name}_{self.name}_variation", EventTermCfg(
            func=ApplyBenchHeightFromSampler,
            mode="reset",
            params={"asset_cfg": SceneEntityCfg(self.bench_name), "carried": list(self.carried), "sampler": self._sampler},
        )


class ApplyBenchHeightFromSampler(ManagerTermBase):
    """Event term: put the bench and what it carries at their spawn poses plus one ``dz`` draw."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        sampler: ContinuousSampler = cfg.params["sampler"]
        assert tuple(sampler.shape_per_sample) == (1,), (
            f"ApplyBenchHeightFromSampler expects a sampler over [dz]; got {tuple(sampler.shape_per_sample)}."
        )
        self.bench = env.scene[cfg.params["asset_cfg"].name]
        self.carried = [env.scene[name] for name in cfg.params["carried"]]
        self.spawn = [wp.to_torch(a.data.default_root_pose).clone() for a in (self.bench, *self.carried)]

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,  # noqa: ARG002
        carried: list[str],  # noqa: ARG002
        sampler: ContinuousSampler,
    ):
        device = self.bench.device
        env_ids = _env_ids(env, env_ids, device)
        if env_ids.numel() == 0:
            return
        index = env_ids.long()
        dz = sampler.sample(num_samples=len(env_ids), env_ids=env_ids).to(device=device, dtype=torch.float32)[:, 0]
        origins = env.scene.env_origins[index]
        for asset, spawn in zip((self.bench, *self.carried), self.spawn):
            pose = spawn[index].to(device).clone()
            pose[:, 2] += dz
            if asset is not self.bench:
                default = wp.to_torch(asset.data.default_root_pose)
                default[index] = pose.to(default.device, default.dtype)
            world = pose.clone()
            world[:, :3] += origins
            asset.write_root_pose_to_sim_index(root_pose=world, env_ids=env_ids)


class PaperPileListener:
    """The last ``[dx, dy, dyaw]`` the paper-pile variation drew, per environment, for the paper sidecar."""

    def __init__(self):
        self._draw = torch.zeros(0, 3)

    def __call__(self, sample, env_ids) -> None:
        if env_ids is None:
            return
        index = torch.as_tensor(env_ids, dtype=torch.long).cpu()
        if len(index) and int(index.max()) >= len(self._draw):
            grown = torch.zeros(int(index.max()) + 1, 3)
            grown[: len(self._draw)] = self._draw
            self._draw = grown
        self._draw[index] = torch.as_tensor(sample, dtype=torch.float32).cpu()[:, :3]

    def __repr__(self) -> str:
        return "PaperPileListener()"

    def __deepcopy__(self, memo):
        return self

    def draw(self, env_id: int = 0) -> tuple[float, float, float]:
        """The pile offset of environment ``env_id``: x, y [m], yaw [rad]."""
        if env_id >= len(self._draw):
            return (0.0, 0.0, 0.0)
        return tuple(float(v) for v in self._draw[env_id])


@configclass
class PaperPileVariationCfg(VariationBaseCfg):
    """Configuration for PaperPileVariation."""

    sampler_cfg: UniformSamplerCfg = field(
        default_factory=lambda: UniformSamplerCfg(low=[0.0, 0.0, 0.0], high=[0.0, 0.0, 0.0])
    )
    """Uniform ``[dx, dy, dyaw]``: world metres, and radians about world +Z through the invoice pile's centre."""


class PaperPileVariation(RunTimeVariationBase):
    """Draw where the paper piles start at every reset; the paper sidecar puts them there."""

    cfg: PaperPileVariationCfg

    def __init__(
        self,
        host_name: str,
        listener: PaperPileListener,
        cfg: PaperPileVariationCfg | None = None,
        name: str = "paper_pile",
    ):
        super().__init__(cfg=cfg if cfg is not None else PaperPileVariationCfg(), name=name)
        self.host_name = host_name
        self.listener = listener
        self.add_sample_listener(listener)

    def build_event_cfg(self) -> tuple[str, EventTermCfg]:
        assert self._sampler is not None, f"PaperPileVariation on '{self.host_name}' is enabled but has no sampler."
        return f"{self.host_name}_{self.name}_variation", EventTermCfg(
            func=draw_paper_pile, mode="reset", params={"sampler": self._sampler}
        )


def draw_paper_pile(env: ManagerBasedEnv, env_ids: torch.Tensor | None, sampler: ContinuousSampler) -> None:
    """Event term: draw the piles' offset; the sampler's listeners (the sidecar's) take it from there."""

    env_ids = _env_ids(env, env_ids, env.device)
    if env_ids.numel():
        sampler.sample(num_samples=len(env_ids), env_ids=env_ids)
