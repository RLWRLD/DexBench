# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""A policy that does nothing on a DexBench embodiment: it holds the robot's reset pose."""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import torch
from gymnasium.spaces.dict import Dict as GymSpacesDict

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg


@dataclass
class HoldPosePolicyCfg(PolicyCfg):
    """Configure the hold of a joint-position embodiment's reset pose."""

    joint_action_term: str = "joint_pos"
    """The embodiment's joint-position action term."""

    hold: str = "default"
    """What is held: ``default``, the robot's default joint positions, or ``targets``, the joint position
    targets the robot is commanded to when the step starts. ``targets`` is for a task whose reset forms the
    robot's start itself (DB-M1's grasp reset leaves the fingers commanded into the drawer): holding the
    default pose there would open the grasp."""

    def __post_init__(self):
        if self.hold not in ("default", "targets"):
            raise ValueError(f"HoldPosePolicyCfg.hold must be 'default' or 'targets', got {self.hold!r}")


def joint_position_action(term, joint_pos: torch.Tensor) -> torch.Tensor:
    """The action that commands ``joint_pos`` (all the robot's joints) through joint-position action ``term``."""
    return (joint_pos[:, term._joint_ids] - term._offset) / term._scale


@register_policy
class HoldPosePolicy(PolicyBase[HoldPosePolicyCfg]):
    """Command the reset joint positions on every step (a condition recording's do-nothing policy)."""

    name = "dexbench_hold_pose"

    def get_action(self, env: gym.Env, observation: GymSpacesDict) -> torch.Tensor:
        live = env.unwrapped
        manager = live.action_manager
        if self.config.joint_action_term in manager.active_terms:
            term = manager.get_term(self.config.joint_action_term)
            return joint_position_action(term, self._held(live, term))
        terms = [manager.get_term(name) for name in manager.active_terms]
        if not terms or not all(callable(getattr(term, "hold_action", None)) for term in terms):
            raise ValueError(
                f"{self.name} holds a joint-position embodiment, or one whose every action term has a hold_action; "
                f"this environment's terms {list(manager.active_terms)} have neither a "
                f"{self.config.joint_action_term!r} term nor a hold for each."
            )
        return torch.cat([term.hold_action(self._held(live, term)) for term in terms], dim=1)

    def _held(self, live, term) -> torch.Tensor:
        robot = live.scene[term.cfg.asset_name]
        held = robot.data.default_joint_pos if self.config.hold == "default" else robot.data.joint_pos_target
        return held.torch
