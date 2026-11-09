# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Absolute joint actions with a saved, yielding support-hand reference."""

from dataclasses import field

import torch
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass

from dexbench_arena.embodiments.common.compliant_grasp import (
    AdaptiveGraspCfg,
    AdaptiveGraspController,
    CompliantGraspCfg,
    compliant_grasp_targets,
)
from dexbench_arena.embodiments.joint_state import as_torch, resolve_joint_order


class SavedGraspJointPositionAction(JointPositionAction):
    """Keep selected fingers anchored to saved q while other joints follow actions."""

    def __init__(self, cfg, env):
        if cfg.scale != 1.0 or cfg.offset != 0.0 or cfg.use_default_offset or cfg.clip is not None:
            raise ValueError("Saved grasp actions require unclipped absolute joint targets")
        if not cfg.grasp_profile.anchor_to_reset:
            raise ValueError("Saved grasp actions require anchor_to_reset")
        names = cfg.grasp_joint_names
        if not names or len(set(names)) != len(names):
            raise ValueError("Saved grasp joint names must be nonempty and unique")
        if len(cfg.closing_directions) != len(names) or any(d not in (-1, 0, 1) for d in cfg.closing_directions):
            raise ValueError("Each grasp joint needs a closing direction of -1, 0 or 1")
        if len(cfg.finger_columns) != len(names) or any(type(i) is not int or i < 0 for i in cfg.finger_columns):
            raise ValueError("Each grasp joint needs a nonnegative finger column")
        super().__init__(cfg, env)
        ids, _ = resolve_joint_order(self._asset, names)
        if not set(names).issubset(self._joint_names):
            raise ValueError("All grasp joints must be controlled by this action")
        self._grasp_ids = ids
        self._grasp_columns = [self._joint_names.index(name) for name in names]
        self._finger_columns = cfg.finger_columns
        self._adaptive_grasp = None
        with torch.inference_mode(False):
            self._directions = torch.tensor(cfg.closing_directions, device=self.device)[None, :]
            self._grasp_reference = torch.zeros((self.num_envs, len(names)), device=self.device)
            self._grasp_nominal = torch.zeros_like(self._grasp_reference)
            self._reset_commands = torch.zeros_like(self._processed_actions)
            self._grasp_ready = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def validate_scene_state_restore(self, state, *, env_ids=None, is_relative=False):
        """Refuse scene-only restores that omit the saved controller reference."""
        raise ValueError("Saved support grasps cannot use scene-only env.reset_to; "
                         "replay a saved-start variation and call env.reset()")

    def reset_grasp_reference(self, *, joint_positions, joint_targets, env_ids=None):
        """Install selected rows in the articulation's full joint order, not action order."""
        ids = slice(None) if env_ids is None else env_ids
        expected = (self._grasp_reference[ids].shape[0], self._asset.num_joints)
        if joint_positions.shape != expected or joint_targets.shape != expected:
            raise ValueError(f"Saved grasp positions and commands must have shape {expected}")
        if not bool(torch.isfinite(joint_positions).all() & torch.isfinite(joint_targets).all()):
            raise ValueError("Saved grasp positions and commands must be finite")
        self._grasp_reference[ids] = joint_positions[:, self._grasp_ids]
        self._grasp_nominal[ids] = joint_targets[:, self._grasp_ids]
        self._reset_commands[ids] = joint_targets[:, self._joint_ids]
        self._grasp_ready[ids] = True
        self.reset(env_ids)

    def reset(self, env_ids=None):
        """Clear stale actions and assist history, retaining the event's new reference."""
        ids = slice(None) if env_ids is None else env_ids
        with torch.inference_mode(False):
            if self._raw_actions.is_inference():
                self._raw_actions = self._raw_actions.clone()
            if self._processed_actions.is_inference():
                self._processed_actions = self._processed_actions.clone()
        super().reset(env_ids)
        self._processed_actions[ids] = self._reset_commands[ids]
        if self._adaptive_grasp is not None:
            adaptive = self._adaptive_grasp
            adaptive.invalidate(env_ids)
            adaptive.extra[ids] = 0
            adaptive.risk[ids] = 0
            adaptive.hold_bias[ids] = adaptive.cfg.hold_preload_rad

    def update_support_grasp(self, relative_centers, target_mask, finger_distances, dt):
        """Receive wrist-frame geometry from the task without changing public actions."""
        if self.cfg.adaptive_profile is None:
            return
        if finger_distances.shape[1] <= max(self._finger_columns):
            raise ValueError("Grasp feedback does not cover every configured finger")
        if self._adaptive_grasp is None:
            self._adaptive_grasp = AdaptiveGraspController(
                relative_centers, finger_distances, self.cfg.adaptive_profile,
            )
        self._adaptive_grasp.update(relative_centers, target_mask, finger_distances, dt)

    def apply_actions(self):
        if not bool(self._grasp_ready.all()):
            raise RuntimeError("Install a saved grasp in every environment before applying actions")
        profile = self.cfg.grasp_profile
        preload = profile.preload_rad
        if self._adaptive_grasp is not None:
            adaptive = self._adaptive_grasp
            preload = preload + adaptive.hold_bias[:, self._finger_columns] + adaptive.extra[:, self._finger_columns]
        elif self.cfg.adaptive_profile is not None:
            preload += self.cfg.adaptive_profile.hold_preload_rad
        targets = self.processed_actions.clone()
        targets[:, self._grasp_columns] = compliant_grasp_targets(
            as_torch(self._asset.data.joint_pos)[:, self._grasp_ids], self._grasp_nominal,
            as_torch(self._asset.data.soft_joint_pos_limits)[:, self._grasp_ids],
            self._directions, preload, profile.travel_rad, grasp_reference=self._grasp_reference,
        )
        self._asset.set_joint_position_target_index(target=targets, joint_ids=self._joint_ids)


@configclass
class SavedGraspJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = SavedGraspJointPositionAction
    use_default_offset: bool = False
    grasp_joint_names: list[str] = field(default_factory=list)
    closing_directions: list[int] = field(default_factory=list)
    finger_columns: list[int] = field(default_factory=list)
    grasp_profile: CompliantGraspCfg = field(default_factory=lambda: CompliantGraspCfg(anchor_to_reset=True))
    adaptive_profile: AdaptiveGraspCfg | None = None
