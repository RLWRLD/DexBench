# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Build-time XY placement of the M3 cabinet and its grounded screw guides."""

import math
from dataclasses import field

from isaaclab.utils.configclass import configclass
from isaaclab_arena.relations.placement_asset import PlaceableAsset
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg
from isaaclab_arena.variations.variation_base import BuildTimeVariationBase, VariationBaseCfg


@configclass
class M3StationPositionVariationCfg(VariationBaseCfg):
    sampler_cfg: UniformSamplerCfg = field(
        default_factory=lambda: UniformSamplerCfg(low=[-.01, -.01], high=[.01, .01])
    )


class M3StationPositionVariation(BuildTimeVariationBase):
    """Translate a complete station and its task sites by one recorded [dx, dy]."""

    def __init__(self, members, task, cfg=None):
        super().__init__(cfg=cfg if cfg is not None else M3StationPositionVariationCfg(), name="position")
        self.members = tuple(members)
        self.task = task
        self._nominal_poses = tuple(Pose(position_xyz=member.get_object_cfg()[1].init_state.pos,
                                         rotation_xyzw=member.get_object_cfg()[1].init_state.rot)
                                    for member in self.members)
        if not self.members:
            raise ValueError("M3 station position requires station members")
        self._nominal_heads = tuple(task.get_events_cfg().power_driver_fastening.params["seated_heads"])

    def _realize_at_build_time(self):
        for bounds in (self.cfg.sampler_cfg.low, self.cfg.sampler_cfg.high):
            self._validate_offset(bounds)
        offset = self.sampler.sample(num_samples=1)[0].tolist()
        self._validate_offset(offset)
        dx, dy = offset
        for member, pose in zip(self.members, self._nominal_poses):
            x, y, z = pose.position_xyz
            position = (x + dx, y + dy, z)
            if isinstance(member, PlaceableAsset):
                member.set_initial_pose(Pose(position_xyz=position, rotation_xyzw=pose.rotation_xyzw))
            else:
                member.get_object_cfg()[1].init_state.pos = position
        self.task.get_events_cfg().power_driver_fastening.params["seated_heads"] = tuple(
            (x + dx, y + dy, z) for x, y, z in self._nominal_heads
        )

    @staticmethod
    def _validate_offset(offset):
        if len(offset) != 2 or any(not math.isfinite(v) or abs(v) > .010000001 for v in offset):
            raise ValueError("M3 station position requires finite [dx, dy] within +/-0.01 m")
