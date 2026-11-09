# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Build-time variation of an interior light's position."""

from __future__ import annotations

from dataclasses import field

from isaaclab.utils.configclass import configclass

from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg
from isaaclab_arena.variations.variation_base import BuildTimeVariationBase, VariationBaseCfg


@configclass
class LightPositionVariationCfg(VariationBaseCfg):
    """Configuration for LightPositionVariation."""

    sampler_cfg: UniformSamplerCfg = field(
        default_factory=lambda: UniformSamplerCfg(
            low=[-0.8, -0.6, -0.25],
            high=[0.8, 0.6, 0.15],
        )
    )
    """Uniform distribution over the position offset (dx, dy, dz) [m]."""


class LightPositionVariation(BuildTimeVariationBase):
    """Sample a position offset and shift the light's spawn pose by it."""

    cfg: LightPositionVariationCfg

    def __init__(
        self,
        light,
        cfg: LightPositionVariationCfg | None = None,
        name: str = "position",
    ):
        super().__init__(cfg=cfg if cfg is not None else LightPositionVariationCfg(), name=name)
        self._light = light
        self._nominal_pose: Pose | None = None

    def _realize_at_build_time(self) -> None:
        assert self.sampler is not None, "LightPositionVariation: sampler not set."
        base = self._nominal_pose or self._light.get_initial_pose()
        if not isinstance(base, Pose):
            return
        self._nominal_pose = base
        offset = self.sampler.sample(num_samples=1)[0].tolist()
        self._light.set_initial_pose(
            Pose(
                position_xyz=(
                    base.position_xyz[0] + float(offset[0]),
                    base.position_xyz[1] + float(offset[1]),
                    base.position_xyz[2] + float(offset[2]),
                ),
                rotation_xyzw=base.rotation_xyzw,
            )
        )
