# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Restore a recorded H3 grasp through Arena's categorical variation interface."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import field
import math

import torch
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab_arena.variations.choice_sampler import ChoiceSamplerCfg
from isaaclab_arena.variations.variation_base import RunTimeVariationBase, VariationBaseCfg

from dexbench_arena.tasks.h3_start_state import validate_grasp_start


@configclass
class H3StartVariationCfg(VariationBaseCfg):
    enabled: bool = True
    sampler_cfg: ChoiceSamplerCfg = field(default_factory=ChoiceSamplerCfg)
    dataset_values: dict[str, int] = field(default_factory=dict)
    head_joint_positions: dict[str, float] = field(default_factory=dict)


class H3StartVariation(RunTimeVariationBase):
    """Draw a saved grasp for each selected world in a fixed trio/target run."""

    def __init__(self, starts, object_name, *, support_action="joint_pos", cfg=None, name="start_state"):
        if not starts:
            raise ValueError("H3 start variation requires at least one saved grasp")
        self.labels = tuple(starts)
        self.starts = tuple(deepcopy(start) for start in starts.values())
        self.object_name = object_name
        self.support_action = support_action
        names = list(self.starts[0]["state"]["objects"])
        if object_name not in names:
            raise ValueError("H3 variation must be attached to a saved utensil")
        for start in self.starts:
            validate_grasp_start(start, names)
        super().__init__(cfg=cfg if cfg is not None else H3StartVariationCfg(), name=name)

    def apply_cfg(self, cfg):
        if not cfg.enabled:
            raise ValueError("A saved H3 start variation cannot be disabled")
        if (not set(cfg.head_joint_positions) <= {'head_0', 'head_1'}
                or not all(math.isfinite(v) for v in cfg.head_joint_positions.values())):
            raise ValueError('H3 gaze overrides require finite head_0/head_1 positions')
        super().apply_cfg(cfg)
        self.cfg.dataset_values = dict(zip(self.labels, range(len(self.labels))))

    def disable(self):
        raise ValueError("A saved H3 start variation cannot be disabled")

    def build_event_cfg(self):
        return f"{self.object_name}_start_state_variation", EventTermCfg(
            func=ApplyH3StartFromSampler, mode="reset",
            params=dict(asset_cfg=SceneEntityCfg(self.object_name), sampler=self._sampler,
                        labels=self.labels, starts=self.starts, support_action=self.support_action,
                        head_joint_positions=dict(self.cfg.head_joint_positions)),
        )


class ApplyH3StartFromSampler(ManagerTermBase):
    """Restore selected worlds and refresh their support-hand reference together."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene["robot"]
        starts = cfg.params["starts"]
        self.object_names = tuple(starts[0]["state"]["objects"])
        if (set(env.scene.articulations) != {"robot"}
                or set(env.scene.rigid_objects) != set(self.object_names)
                or env.scene.deformable_objects or env.scene.cable_objects or env.scene.surface_grippers):
            raise ValueError("H3 saved state must cover every dynamic scene entity")
        joints = self.robot.joint_names
        for start in starts:
            if set(joints) != set(start["state"]["robot"]["joint_positions"]):
                raise ValueError("H3 saved joints do not match the embodiment")
        self.robot_state = {
            key: torch.tensor([start["state"]["robot"][key] for start in starts],
                              dtype=torch.float32, device=env.device)
            for key in ("root_pose", "root_velocity")
        }
        for destination, source in (("joint_position", "joint_positions"), ("joint_velocity", "joint_velocities")):
            self.robot_state[destination] = torch.tensor(
                [[start["state"]["robot"][source][joint] for joint in joints] for start in starts],
                dtype=torch.float32, device=env.device,
            )
        self.objects = {
            name: {key: torch.tensor([start["state"]["objects"][name][key] for start in starts],
                                     dtype=torch.float32, device=env.device)
                   for key in ("root_pose", "root_velocity")}
            for name in self.object_names
        }
        self.commands = torch.tensor([[start["joint_targets"][joint] for joint in joints] for start in starts],
                                     dtype=torch.float32, device=env.device)
        for name, value in cfg.params['head_joint_positions'].items():
            index = joints.index(name)
            self.robot_state['joint_position'][:, index] = value
            self.robot_state['joint_velocity'][:, index] = 0.
            self.commands[:, index] = value

    def __call__(self, env, env_ids, asset_cfg, sampler, labels, starts, support_action, head_joint_positions):
        ids = (torch.arange(env.scene.num_envs, device=env.device, dtype=torch.long) if env_ids is None
               else torch.as_tensor(env_ids, device=env.device, dtype=torch.long).reshape(-1))
        if not len(ids):
            return
        action = env.action_manager.get_term(support_action)
        reset_reference = action.reset_grasp_reference
        drawn = sampler.sample(num_samples=len(ids), choices=labels, env_ids=ids)
        selected = torch.tensor([labels.index(label) for label in drawn], device=env.device)
        robot_state = {key: value[selected].clone() for key, value in self.robot_state.items()}
        state = dict(articulation=dict(robot=robot_state), rigid_object={
            name: {key: value[selected].clone() for key, value in fields.items()}
            for name, fields in self.objects.items()
        })
        commands = self.commands[selected].clone()
        env.scene.reset_to(state, env_ids=ids, is_relative=True)
        self.robot.set_joint_position_target_index(target=commands, env_ids=ids)
        reset_reference(joint_positions=robot_state["joint_position"], joint_targets=commands, env_ids=ids)
