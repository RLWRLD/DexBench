# SPDX-License-Identifier: Apache-2.0
"""Per-reset draw of a DB-R1 start layout from a layout set: one running scene, many settled piles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import field
from typing import TYPE_CHECKING

import torch
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.variations.choice_sampler import ChoiceSampler, ChoiceSamplerCfg
from isaaclab_arena.variations.variation_base import RunTimeVariationBase, VariationBaseCfg

from dexbench_arena.tasks.r1_clutter import R1_LAYOUT_VARIATION

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

_CANNOT_DISABLE = (
    "r1_fruit_000.layout cannot be disabled: the scene was built for the layout set (--r1_layouts). "
    "One fixed layout is --r1_clutter_file <layout.json> without --r1_layouts."
)


@configclass
class R1LayoutVariationCfg(VariationBaseCfg):
    """Configuration for R1LayoutVariation."""

    enabled: bool = True
    """On by construction: the variation exists only when ``--r1_layouts`` names a set, and naming it is the
    request for the draw. It cannot be turned off; see :class:`R1LayoutVariation`."""

    sampler_cfg: ChoiceSamplerCfg = field(default_factory=ChoiceSamplerCfg)
    """Uniform over the set's layouts; replayed rows replace the draw."""

    dataset_values: dict[str, int] = field(default_factory=dict)
    """Each layout label's seed, which the demo HDF5 records in place of the label
    (``VariationSamplesRecorder``). Set from the set, whatever a run config says."""


class R1LayoutVariation(RunTimeVariationBase):
    """Write one drawn layout of the set over every fruit at every reset."""

    cfg: R1LayoutVariationCfg

    def __init__(self, layouts: Mapping[str, Mapping], object_name: str, cfg: R1LayoutVariationCfg | None = None,
                 name: str = R1_LAYOUT_VARIATION):
        self.labels = tuple(layouts)
        self.seeds = tuple(int(recipe["seed"]) for recipe in layouts.values())
        self.poses = tuple(
            tuple((o["name"], (*map(float, o["position_xyz"]), *map(float, o["rotation_xyzw"]))) for o in recipe["objects"])
            for recipe in layouts.values()
        )
        if not self.poses:
            raise ValueError("R1LayoutVariation needs at least one layout")
        self.object_name = object_name
        super().__init__(cfg=cfg if cfg is not None else R1LayoutVariationCfg(), name=name)

    def apply_cfg(self, cfg: R1LayoutVariationCfg) -> None:
        if not cfg.enabled:
            raise ValueError(_CANNOT_DISABLE)
        super().apply_cfg(cfg)
        self.cfg.dataset_values = dict(zip(self.labels, self.seeds))

    def disable(self) -> None:
        raise ValueError(_CANNOT_DISABLE)

    def build_event_cfg(self) -> tuple[str, EventTermCfg]:
        return (
            f"{self.object_name}_layout_variation",
            EventTermCfg(
                func=ApplyR1LayoutFromSampler,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg(self.object_name),
                    "sampler": self._sampler,
                    "labels": self.labels,
                    "poses": self.poses,
                },
            ),
        )


class ApplyR1LayoutFromSampler(ManagerTermBase):
    """Event term: write the drawn layout's root pose (env-local position, xyzw) and a zero velocity to every fruit."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        poses: Sequence = cfg.params["poses"]
        self.labels: tuple[str, ...] = tuple(cfg.params["labels"])
        assert len(poses) == len(self.labels), "R1LayoutVariation: one pose table per layout"
        names = tuple(name for name, _ in poses[0])
        assert all(tuple(n for n, _ in layout) == names for layout in poses), "every layout poses the same fruit"
        missing = [name for name in names if name not in env.scene.keys()]
        assert not missing, f"R1LayoutVariation: the scene has no {missing}"
        self.entities = [env.scene[name] for name in names]
        self.poses = torch.tensor([[pose for _, pose in layout] for layout in poses], dtype=torch.float32,
                                  device=env.device)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,  # noqa: ARG002  (the host fruit; every fruit is resolved in __init__)
        sampler: ChoiceSampler,
        labels: tuple[str, ...],  # noqa: ARG002
        poses: tuple,  # noqa: ARG002
    ):
        device = self.poses.device
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=device, dtype=torch.int32)
        else:
            env_ids = torch.as_tensor(env_ids, device=device, dtype=torch.int32).reshape(-1)
        if env_ids.numel() == 0:
            return
        drawn = sampler.sample(num_samples=len(env_ids), choices=self.labels, env_ids=env_ids)
        layout = torch.tensor([self.labels.index(label) for label in drawn], device=device)
        origins = env.scene.env_origins[env_ids.long()]
        zero = torch.zeros(len(env_ids), 6, device=device)
        for index, entity in enumerate(self.entities):
            pose = self.poses[layout, index].clone()
            pose[:, :3] += origins
            entity.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
            entity.write_root_velocity_to_sim_index(root_velocity=zero, env_ids=env_ids)
