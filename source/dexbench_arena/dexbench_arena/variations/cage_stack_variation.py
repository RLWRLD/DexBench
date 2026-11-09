# SPDX-License-Identifier: Apache-2.0
"""Per-reset variation of the E2 cage's box stack: which size set stands in the cage, and where."""

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from dataclasses import field, replace
from typing import TYPE_CHECKING

import warp as wp
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.utils.yaw import rotate_points_by_yaw
from isaaclab_arena.variations.continuous_sampler import ContinuousSampler
from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg
from isaaclab_arena.variations.variation_base import RunTimeVariationBase, VariationBaseCfg

from dexbench_arena.assets.e2_logistics_cell import E2_PARCEL_CAGE_SLOT_YAW_DEG
from dexbench_arena.layouts.e2_logistics_layout import (
    E2CageLayoutParams,
    e2_cage_box_entity_name,
    e2_cage_box_park_position,
    e2_cage_layout,
    e2_cage_stack_params_for_draw,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


@configclass
class CageStackVariationCfg(VariationBaseCfg):
    """Configuration for CageStackVariation."""

    sampler_cfg: UniformSamplerCfg = field(
        default_factory=lambda: UniformSamplerCfg(low=[0.0, 0.0, -0.05], high=[1.0, 0.10, 0.05])
    )
    """Uniform over ``[set_pick, ddepth, dlateral]``."""


class CageStackVariation(RunTimeVariationBase):
    """Redraw which box set stands in the cage, and where, at every reset."""

    cfg: CageStackVariationCfg

    def __init__(
        self,
        object_name: str,
        nominal: E2CageLayoutParams,
        scales: Sequence[float],
        object_seats: Sequence[tuple[float, float, float]],
        cfg: CageStackVariationCfg | None = None,
        name: str = "cage_stack",
    ):
        super().__init__(cfg=cfg if cfg is not None else CageStackVariationCfg(), name=name)
        self.object_name = object_name
        self.nominal = nominal
        self.scales = tuple(float(scale) for scale in scales)
        self.object_seats = tuple((float(long), float(depth), float(clearance)) for long, depth, clearance in object_seats)
        assert self.scales, "CageStackVariation needs at least one box set."
        assert self.object_seats, "CageStackVariation needs at least one object seat."

    def validate_range(self) -> None:
        """Raise ValueError if any corner of the draw range puts a set, or the object on it, where it does not fit."""

        low, high = self.cfg.sampler_cfg.low, self.cfg.sampler_cfg.high
        assert len(low) == 3 and len(high) == 3, "cage_stack sampler must draw [set_pick, ddepth, dlateral]."
        for index in range(len(self.scales)):
            for ddepth in (low[1], high[1]):
                for dlateral in (low[2], high[2]):
                    set_pick = (index + 0.5) / len(self.scales)
                    _, params = e2_cage_stack_params_for_draw(self.nominal, self.scales, (set_pick, ddepth, dlateral))
                    try:
                        layout = e2_cage_layout(params)
                        for long_side, depth, clearance in self.object_seats:
                            layout.object_pose((long_side, depth), clearance)
                    except ValueError as exc:
                        raise ValueError(
                            f"cage_stack draw range low={list(low)} high={list(high)} does not fit box set "
                            f"{index} (scale {self.scales[index]}): {exc} Narrow "
                            f"{self.object_name}.{self.name}.sampler_cfg.low/high or drop the set."
                        ) from exc

    def build_event_cfg(self) -> tuple[str, EventTermCfg]:
        assert self._sampler is not None, (
            f"CageStackVariation on '{self.object_name}' is enabled but no sampler is set; "
            "call apply_cfg with a cfg that sets sampler_cfg before building the env."
        )
        self.validate_range()
        event_name = f"{self.object_name}_cage_stack_variation"
        nominal = (self.nominal.box_scale, self.nominal.stack_offset_depth_m, self.nominal.stack_offset_lateral_m)
        event_cfg = EventTermCfg(
            func=ApplyCageStackFromSampler,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(self.object_name),
                "sampler": self._sampler,
                "nominal": nominal,
                "scales": self.scales,
                "object_seats": self.object_seats,
            },
        )
        return event_name, event_cfg


class ApplyCageStackFromSampler(ManagerTermBase):
    """Event term: place one box set in the cage per drawn layout, park the rest, seat the object."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        sampler: ContinuousSampler = cfg.params["sampler"]
        assert tuple(sampler.shape_per_sample) == (3,), (
            "ApplyCageStackFromSampler expects a sampler with shape_per_sample (3,) over "
            f"[set_pick, ddepth, dlateral]; got {tuple(sampler.shape_per_sample)}."
        )
        box_scale, depth, lateral = cfg.params["nominal"]
        nominal = E2CageLayoutParams(
            box_scale=float(box_scale), stack_offset_depth_m=float(depth), stack_offset_lateral_m=float(lateral)
        )
        self.scales: tuple[float, ...] = tuple(float(scale) for scale in cfg.params["scales"])
        seats = tuple(cfg.params["object_seats"])
        self.lifted = env.scene[cfg.params["asset_cfg"].name]
        assert hasattr(self.lifted, "write_root_pose_to_sim_index"), "the lifted object must be a rigid object"
        device = self.lifted.device

        self.cage_yaw_rad = math.radians(E2_PARCEL_CAGE_SLOT_YAW_DEG)

        self.boxes: list[tuple[int, object]] = []
        in_cage: list[torch.Tensor] = []
        parked: list[torch.Tensor] = []
        object_in_cage: list[torch.Tensor] = []
        for index, scale in enumerate(self.scales):
            layout = e2_cage_layout(replace(nominal, box_scale=scale))
            for placement in layout.placements:
                entity = e2_cage_box_entity_name(placement.name, index)
                assert entity in env.scene.keys(), f"cage_stack: box set {index} is missing scene entity {entity!r}"
                box = env.scene[entity]
                assert hasattr(box, "write_root_pose_to_sim_index"), f"{entity} must be a rigid (kinematic) object"
                self.boxes.append((index, box))
                in_cage.append(torch.tensor([*placement.position_xyz, *placement.rotation_xyzw]))
                parked.append(
                    torch.tensor([*e2_cage_box_park_position(placement.position_xyz, index), *placement.rotation_xyzw])
                )
            object_in_cage.append(
                torch.stack(
                    [
                        torch.tensor([*position, *rotation])
                        for position, rotation in (
                            layout.object_pose((long_side, depth), clearance) for long_side, depth, clearance in seats
                        )
                    ]
                )
            )
        self.in_cage = torch.stack(in_cage).to(device)
        self.parked = torch.stack(parked).to(device)
        self.object_in_cage = torch.stack(object_in_cage).to(device)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,  # noqa: ARG002
        sampler: ContinuousSampler,
        nominal: tuple[float, float, float],  # noqa: ARG002  (read once in __init__)
        scales: tuple[float, ...],  # noqa: ARG002
        object_seats: tuple[tuple[float, float, float], ...],  # noqa: ARG002
    ):
        device = self.lifted.device
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=device, dtype=torch.int32)
        else:
            env_ids = torch.as_tensor(env_ids, device=device, dtype=torch.int32).reshape(-1)
        if env_ids.numel() == 0:
            return
        env_index = env_ids.long()

        sample = sampler.sample(num_samples=len(env_ids), env_ids=env_ids).to(device=device, dtype=torch.float32)
        n_sets = len(self.scales)
        active = (sample[:, 0] * n_sets).floor().clamp(0, n_sets - 1).long()
        shift = torch.zeros(len(env_ids), 7, device=device)
        local = torch.cat([sample[:, [2, 1]], torch.zeros_like(sample[:, :1])], dim=1)
        shift[:, :2] = rotate_points_by_yaw(local, self.cage_yaw_rad)[:, :2]
        origins = torch.zeros(len(env_ids), 7, device=device)
        origins[:, :3] = env.scene.env_origins[env_index]

        for (index, box), in_cage, parked in zip(self.boxes, self.in_cage, self.parked):
            selected = (active == index).unsqueeze(1)
            pose = torch.where(selected, in_cage + shift, parked.expand_as(shift)) + origins
            box.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)

        member = env_index % self.object_in_cage.shape[1]
        object_local = self.object_in_cage[active, member] + shift
        self.lifted.write_root_pose_to_sim_index(root_pose=object_local + origins, env_ids=env_ids)
        self.lifted.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros(len(env_ids), 6, device=device), env_ids=env_ids
        )
        default_root_pose = wp.to_torch(self.lifted.data.default_root_pose)
        default_root_pose[env_index] = object_local.to(default_root_pose.device, default_root_pose.dtype)
