# SPDX-License-Identifier: Apache-2.0
"""Waist IK for the RBY1-M body: the torso joins the Pink wrist solve."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
from pink.tasks import DampingTask
from pink.utils import get_root_joint_dim

from isaaclab.controllers.pink_ik import DampingTaskCfg, NullSpacePostureTaskCfg
from isaaclab.controllers.pink_ik.null_space_posture_task import NullSpacePostureTask
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.pink_task_space_actions import PinkInverseKinematicsAction
from isaaclab.utils.configclass import configclass

from dexbench_arena.embodiments.rby1m.constants import RBY1M_TORSO_JOINT_NAMES, RBY1M_TORSO_YAW_JOINT_NAME

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

_WAIST_IK_MAX_SPEED_RAD_S = 0.8


def _joint_costs(configuration, arm_cost: float, torso_cost_multiplier: float) -> np.ndarray:
    """Resolve Pink cost weights in reduced-model velocity order by joint name."""
    if not np.isfinite(arm_cost) or arm_cost < 0:
        raise ValueError("RBY1-M arm cost must be finite and non-negative")
    if not np.isfinite(torso_cost_multiplier) or torso_cost_multiplier < 1:
        raise ValueError("RBY1-M torso cost multiplier must be finite and at least one")
    torso_cost = arm_cost * torso_cost_multiplier
    model = configuration.model
    _, root_nv = get_root_joint_dim(model)
    if root_nv != 0 or model.nq != model.nv:
        raise ValueError("RBY1-M waist IK requires a fixed-base model with matching configuration/velocity dimensions")
    costs = np.full(model.nv - root_nv, float(arm_cost))
    for name in RBY1M_TORSO_JOINT_NAMES:
        if not model.existJointName(name):
            raise ValueError(f"RBY1-M waist IK requires controlled joint {name!r}")
        joint = model.joints[model.getJointId(name)]
        start = joint.idx_v - root_nv
        costs[start : start + joint.nv] = arm_cost if name == RBY1M_TORSO_YAW_JOINT_NAME else torso_cost
    return costs


class Rby1mWaistDampingTask(DampingTask):
    """Prefer arm motion using Pink's existing per-DOF damping costs."""

    def __init__(self, cfg):
        self._arm_cost = float(cfg.cost)
        self._torso_cost_multiplier = float(cfg.torso_cost_multiplier)
        super().__init__(cost=self._arm_cost)

    def set_target_from_configuration(self, configuration):
        super().set_target_from_configuration(configuration)
        self.cost = _joint_costs(configuration, self._arm_cost, self._torso_cost_multiplier)


class Rby1mWaistPostureTask(NullSpacePostureTask):
    """Prefer retaining torso posture while the arms reach toward the wrists."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self._arm_cost = float(cfg.cost)
        self._torso_cost_multiplier = float(cfg.torso_cost_multiplier)

    def _build_joint_mapping(self, configuration):
        super()._build_joint_mapping(configuration)
        self.cost = _joint_costs(configuration, self._arm_cost, self._torso_cost_multiplier)


@configclass
class Rby1mWaistDampingTaskCfg(DampingTaskCfg):
    class_type: type = Rby1mWaistDampingTask
    torso_cost_multiplier: float = 16.0


@configclass
class Rby1mWaistPostureTaskCfg(NullSpacePostureTaskCfg):
    class_type: type = Rby1mWaistPostureTask
    torso_cost_multiplier: float = 16.0


class Rby1mPinkInverseKinematicsAction(PinkInverseKinematicsAction):
    """Pink action for the RBY1-M body: upstream's solve, then the waist command limits."""

    def __init__(self, cfg: Rby1mPinkInverseKinematicsActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._waist_ik_previous_target = None
        self._waist_torso_columns = None
        self._waist_torso_ids = None
        if cfg.waist_ik_enabled:
            joint_names = self._isaaclab_controlled_joint_names
            columns = [joint_names.index(name) for name in RBY1M_TORSO_JOINT_NAMES]
            joint_ids = [self._isaaclab_controlled_joint_ids[column] for column in columns]
            self._waist_torso_columns = torch.tensor(columns, device=self.device)
            self._waist_torso_ids = torch.tensor(joint_ids, device=self.device)

    def _compute_ik_solutions(self) -> torch.Tensor:
        solutions = super()._compute_ik_solutions()
        if self._waist_torso_columns is None:
            return solutions

        torso_columns = self._waist_torso_columns
        current = self._asset.data.joint_pos.torch[:, self._waist_torso_ids]
        max_step = _WAIST_IK_MAX_SPEED_RAD_S * self._sim_dt
        ik_target = solutions[:, torso_columns]
        previous_target = self._waist_ik_previous_target
        if previous_target is None:
            self._waist_ik_reset_mask = torch.zeros(current.shape[0], dtype=torch.bool, device=current.device)
            rate_base = current
        else:
            rate_base = torch.where(self._waist_ik_reset_mask[:, None], current, previous_target)
            self._waist_ik_reset_mask = torch.zeros_like(self._waist_ik_reset_mask)
        rate_limited = torch.clamp(ik_target, min=rate_base - max_step, max=rate_base + max_step)
        limited_target = torch.clamp(
            rate_limited,
            min=current - self.cfg.waist_ik_max_lead_rad,
            max=current + self.cfg.waist_ik_max_lead_rad,
        )
        solutions[:, torso_columns] = limited_target
        self._waist_ik_previous_target = limited_target.clone()
        return solutions

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            self._waist_ik_previous_target = None
        elif self._waist_ik_previous_target is not None:
            mask = self._waist_ik_reset_mask.clone()
            mask[env_ids] = True
            self._waist_ik_reset_mask = mask


@configclass
class Rby1mPinkInverseKinematicsActionCfg(PinkInverseKinematicsActionCfg):
    """Pink action configuration for the RBY1-M body, with the waist IK settings."""

    class_type: type = Rby1mPinkInverseKinematicsAction
    waist_ik_enabled: bool = False
    waist_ik_max_lead_rad: float = 0.05


def resolve_waist_ik_args(waist_ik: bool | None, waist_ik_max_lead_rad: float | None) -> tuple[bool, float]:
    """The embodiment constructor's ``waist_ik`` and ``waist_ik_max_lead_rad``, checked."""
    if waist_ik is None:
        waist_ik = False
    if waist_ik_max_lead_rad is None:
        waist_ik_max_lead_rad = 0.05
    if not isinstance(waist_ik, (bool, np.bool_)):
        raise ValueError(f"waist_ik must be a bool, got {waist_ik!r}")
    if isinstance(waist_ik_max_lead_rad, (bool, np.bool_)) or not isinstance(
        waist_ik_max_lead_rad, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"waist_ik_max_lead_rad must be a number, got {waist_ik_max_lead_rad!r}")
    if not np.isfinite(waist_ik_max_lead_rad) or waist_ik_max_lead_rad <= 0:
        raise ValueError("waist_ik_max_lead_rad must be finite and positive")
    return bool(waist_ik), float(waist_ik_max_lead_rad)


def enable_rby1m_waist_ik(ik: Rby1mPinkInverseKinematicsActionCfg) -> None:
    """Let the wrist tasks use the torso while preferring arm motion."""

    if ik.waist_ik_enabled:
        return
    ik.pink_controlled_joint_names.extend(RBY1M_TORSO_JOINT_NAMES)
    for index, task in enumerate(ik.controller.variable_input_tasks):
        if isinstance(task, DampingTaskCfg):
            ik.controller.variable_input_tasks[index] = Rby1mWaistDampingTaskCfg(cost=task.cost)
        elif isinstance(task, NullSpacePostureTaskCfg):
            ik.controller.variable_input_tasks[index] = Rby1mWaistPostureTaskCfg(
                cost=task.cost,
                gain=task.gain,
                lm_damping=task.lm_damping,
                controlled_frames=list(task.controlled_frames),
                controlled_joints=[*task.controlled_joints, *RBY1M_TORSO_JOINT_NAMES],
            )
    ik.waist_ik_enabled = True
