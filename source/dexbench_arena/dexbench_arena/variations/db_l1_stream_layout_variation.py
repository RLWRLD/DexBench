# SPDX-License-Identifier: Apache-2.0
"""Per-reset draw of a DB-L1 episode's layout: which bags and boxes go down, in which order, where and how turned."""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING

import torch
from isaaclab.managers import EventTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.variations.continuous_sampler import ContinuousSampler
from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg
from isaaclab_arena.variations.variation_base import RunTimeVariationBase, VariationBaseCfg

from dexbench_arena.tasks.polybag_stream import LAYOUT_VARIATION_EVENT, l1_stream

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

L1_STREAM_LAYOUT_VARIATION = "stream_layout"
"""The variation's name on the park slab: its draws are recorded as ``l1_park_slab.stream_layout``."""


@configclass
class L1StreamLayoutCfg(VariationBaseCfg):
    """Configuration for :class:`L1StreamLayoutVariation`: uniform over [0, 1] in every coordinate (the constructor sets
    the size); a narrower box is refused, since the run's ranges, not the draw, bound the layout."""

    enabled: bool = True
    """On by default, unlike Arena's ``VariationBaseCfg``: every episode-mode run records its layout."""

    sampler_cfg: UniformSamplerCfg = field(default_factory=UniformSamplerCfg)


class L1StreamLayoutVariation(RunTimeVariationBase):
    """Draw each episode's layout and hand it to the stream (module docstring)."""

    cfg: L1StreamLayoutCfg

    def __init__(self, size: int, cfg: L1StreamLayoutCfg | None = None):
        if int(size) < 1:
            raise ValueError(f"a layout draw of {size} coordinates")
        self.size = int(size)
        if cfg is None:
            cfg = L1StreamLayoutCfg(sampler_cfg=UniformSamplerCfg(low=[0.0] * self.size, high=[1.0] * self.size))
        super().__init__(cfg=cfg, name=L1_STREAM_LAYOUT_VARIATION)

    def apply_cfg(self, cfg: L1StreamLayoutCfg) -> None:
        low, high = list(cfg.sampler_cfg.low), list(cfg.sampler_cfg.high)
        if low != [0.0] * self.size or high != [1.0] * self.size:
            raise ValueError(f"l1_park_slab.{L1_STREAM_LAYOUT_VARIATION} draws {self.size} values uniform over [0, 1]; "
                             "the run's ranges (l1_spacing, l1_item_y, l1_bag_yaw_off_belt_deg) bound the layout, not "
                             f"the sampler (got low {low}, high {high})")
        super().apply_cfg(cfg)

    def build_event_cfg(self) -> tuple[str, EventTermCfg]:
        return (LAYOUT_VARIATION_EVENT,
                EventTermCfg(func=apply_l1_stream_layout, mode="reset", params={"sampler": self.sampler}))


def apply_l1_stream_layout(env: ManagerBasedEnv, env_ids: torch.Tensor | None, sampler: ContinuousSampler) -> None:
    """Reset event: draw the resetting envs' layouts and hand each to the stream. A reset event runs before the event
    manager resets its terms (Isaac Lab's ``_reset_idx``), so the stream's reset lays the episode out from it."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    ids = torch.as_tensor(env_ids, device=env.device).reshape(-1)
    if ids.numel() == 0:
        return
    drawn = sampler.sample(num_samples=len(ids), env_ids=ids)
    for row, env_id in zip(drawn, ids.tolist()):
        l1_stream(env).set_layout_draw(env_id, row)
