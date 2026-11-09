# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Wrist-pose targets to a joint-position action: an arms-only Pink IK solve every env step, the rest commanded or held."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from dexbench_arena.policy.hold_pose_policy import joint_position_action

SOLVE_RATE = "once per env step, from the measured joint positions, dt = env.step_dt"
HELD_REFERENCE = "the joint position targets at the first step after each reset (the reset's command), if any are held"
POSTURE_REFERENCE = "the env's robot init_state (its start pose), as upstream's Pink action term sets it"
ON_SOLVER_FAILURE = ("upstream PinkIKController returns the measured arm positions (the arms hold); counted per "
                     "episode (an exact zero step is counted as one)")
ON_JOINT_LIMIT = "arm targets clamped to the articulation's soft joint limits; counted per episode"


@dataclasses.dataclass(frozen=True)
class ArmIkLayout:
    """What the embodiment declares: the solved, held and passed-through joints and the controlled wrist frames."""

    arm_joint_names: tuple[str, ...]
    held_joint_names: tuple[str, ...]
    passthrough_joint_names: tuple[str, ...]
    wrist_frames: Mapping[str, str]
    """``{hand: frame name}`` of the Pink frame tasks, in the order the policy names its hands."""

    def __post_init__(self):
        groups = (self.arm_joint_names, self.held_joint_names, self.passthrough_joint_names)
        names = [name for group in groups for name in group]
        if len(set(names)) != len(names):
            raise ValueError("a joint is solved, held or passed through, never two of them")


@dataclasses.dataclass(frozen=True)
class ArmIkSettings:
    """What the controller runs with, all of it stamped (``as_dict``)."""

    layout: ArmIkLayout
    ik_profile: str
    profile: Mapping[str, float]
    """The Pink task costs and gains of the profile (``embodiments/rby1m/ik.py``)."""
    step_dt: float
    solver: str = "upstream isaaclab.controllers.pink_ik.PinkIKController (pink, daqp)"

    def as_dict(self) -> dict[str, Any]:
        return {
            "solver": self.solver,
            "rate": SOLVE_RATE,
            "dt": self.step_dt,
            "ik_profile": self.ik_profile,
            "profile": dict(self.profile),
            "solved_joints": list(self.layout.arm_joint_names),
            "held_joints": list(self.layout.held_joint_names),
            "held_reference": HELD_REFERENCE,
            "passthrough_joints": list(self.layout.passthrough_joint_names),
            "wrist_frames": dict(self.layout.wrist_frames),
            "posture_reference": POSTURE_REFERENCE,
            "on_solver_failure": ON_SOLVER_FAILURE,
            "on_joint_limit": ON_JOINT_LIMIT,
        }


class EefArmIkController:
    """Per-step wrist targets (base frame) and hand targets -> the joint action of every env."""

    def __init__(self, settings: ArmIkSettings, solvers: Sequence[Any], joint_names: Sequence[str], term,
                 soft_joint_limits: torch.Tensor):
        """``solvers``: one upstream ``PinkIKController`` per env, built for ``settings.layout.arm_joint_names``;
        ``joint_names``: the articulation's, in its order; ``term``: the joint-position action term the action is
        for; ``soft_joint_limits``: ``(num_envs, num_joints, 2)``."""
        layout = settings.layout
        self.settings = settings
        self._solvers = list(solvers)
        self._term = term
        names = list(joint_names)
        action_names = [names[index] for index in _ids(term, len(names))]
        covered = set(layout.arm_joint_names) | set(layout.held_joint_names) | set(layout.passthrough_joint_names)
        if set(action_names) != covered or len(action_names) != len(covered):
            raise ValueError(f"the joint action's columns {sorted(set(action_names) ^ covered)} are not exactly the "
                             "solved, held and passed-through joints")
        self._arm_ids = [names.index(name) for name in layout.arm_joint_names]
        self._held_ids = [names.index(name) for name in layout.held_joint_names]
        self._hand_ids = [names.index(name) for name in layout.passthrough_joint_names]
        for solver in self._solvers:
            if list(solver.cfg.joint_names) != list(layout.arm_joint_names):
                raise ValueError("each solver must solve the layout's arm joints, in its order")
        self._fk_ids = [[names.index(name) for name in solver.pink_configuration.all_joint_names_pinocchio_order]
                        for solver in self._solvers]
        self._posture_ids = [[names.index(name) for name in solver.pink_configuration.model.names.tolist()[1:]]
                             for solver in self._solvers]
        self._limits = soft_joint_limits[:, self._arm_ids].detach().cpu().float()
        num_envs = len(self._solvers)
        self._held: torch.Tensor | None = None
        self._capture = torch.ones(num_envs, dtype=torch.bool)
        self._stats = [_EpisodeStats() for _ in range(num_envs)]
        self._using_posture_targets = False

    @property
    def num_envs(self) -> int:
        return len(self._solvers)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Capture the held joints' reference again at the next step, and start new episode counters."""
        ids = range(self.num_envs) if env_ids is None else [int(i) for i in env_ids]
        for env_id in ids:
            self._capture[env_id] = True
            self._stats[env_id] = _EpisodeStats()

    def episode_stats(self, env_id: int) -> dict[str, Any]:
        """The current episode's solves, solver failures and joint-limit clamps of ``env_id``."""
        return dataclasses.asdict(self._stats[env_id])

    def wrist_targets_from_joint_positions(self, joint_pos: torch.Tensor):
        """Forward kinematics of planned joints in the same frames the Pink tasks use."""
        poses = {hand: ([], []) for hand in self.settings.layout.wrist_frames}
        for env_id, solver in enumerate(self._solvers):
            configuration = solver.pink_configuration
            configuration.update(joint_pos[env_id, self._fk_ids[env_id]].detach().cpu().numpy().astype(np.float64))
            for hand, frame in self.settings.layout.wrist_frames.items():
                pose = configuration.get_transform_frame_to_world(frame)
                poses[hand][0].append(pose.translation.copy())
                poses[hand][1].append(pose.rotation.copy())
        return {hand: (torch.as_tensor(np.stack(p)), torch.as_tensor(np.stack(r)))
                for hand, (p, r) in poses.items()}

    def action(self, joint_pos: torch.Tensor, joint_pos_target: torch.Tensor,
               wrist_targets: Mapping[str, tuple[torch.Tensor, torch.Tensor]], hand_targets: torch.Tensor,
               *, posture_targets: torch.Tensor | None = None) -> torch.Tensor:
        """The joint action for this step."""
        if set(wrist_targets) != set(self.settings.layout.wrist_frames):
            raise ValueError(f"wrist targets for {sorted(wrist_targets)}, the layout controls "
                             f"{sorted(self.settings.layout.wrist_frames)}")
        measured = joint_pos.detach().cpu().float()
        if self._held is None:
            self._held = measured[:, self._held_ids].clone()
        for env_id in torch.nonzero(self._capture).flatten().tolist():
            self._held[env_id] = joint_pos_target[env_id, self._held_ids].detach().cpu().float()
            self._capture[env_id] = False
        full = measured.clone()
        for env_id, solver in enumerate(self._solvers):
            if posture_targets is not None:
                solver.update_null_space_joint_targets(
                    posture_targets[env_id, self._posture_ids[env_id]].detach().cpu().numpy().astype(np.float64)
                )
            elif self._using_posture_targets:
                solver.update_null_space_joint_targets(solver.init_joint_positions)
            for hand, frame in self.settings.layout.wrist_frames.items():
                position, rotation = wrist_targets[hand]
                _set_frame_target(solver, frame, position[env_id], rotation[env_id])
            current = measured[env_id, self._arm_ids]
            solved = solver.compute(measured[env_id].numpy().astype(np.float64), self.settings.step_dt)
            solved = torch.as_tensor(solved, dtype=torch.float32).cpu()
            stats = self._stats[env_id]
            stats.solves += 1
            if torch.equal(solved, current):
                stats.solver_failures += 1
            clamped = torch.clamp(solved, self._limits[env_id, :, 0], self._limits[env_id, :, 1])
            excess = float((solved - clamped).abs().max())
            if excess > 0.0:
                stats.clamped_steps += 1
                stats.max_clamp_rad = max(stats.max_clamp_rad, excess)
            full[env_id, self._arm_ids] = clamped
        self._using_posture_targets = posture_targets is not None
        full[:, self._held_ids] = self._held
        full[:, self._hand_ids] = hand_targets.detach().cpu().float()
        return joint_position_action(self._term, full.to(_device(self._term, joint_pos)))


@dataclasses.dataclass
class _EpisodeStats:
    solves: int = 0
    solver_failures: int = 0
    clamped_steps: int = 0
    max_clamp_rad: float = 0.0


def _ids(term, num_joints: int) -> list[int]:
    ids = term._joint_ids
    return list(range(num_joints)) if isinstance(ids, slice) else [int(i) for i in ids]


def _device(term, fallback: torch.Tensor):
    offset = getattr(term, "_offset", None)
    return offset.device if isinstance(offset, torch.Tensor) else fallback.device


def _set_frame_target(solver, frame: str, position: torch.Tensor, rotation: torch.Tensor) -> None:
    """Set the base-frame target of ``solver``'s frame task for ``frame``, the way upstream's Pink action term does
    (``PinkInverseKinematicsAction._set_task_targets``): the task's own target transform, updated in place."""
    for task in solver.cfg.variable_input_tasks:
        if getattr(task, "frame", None) != frame:
            continue
        target = task.transform_target_to_world
        target.translation = np.asarray(position.detach().cpu(), dtype=np.float64)
        target.rotation = np.asarray(rotation.detach().cpu(), dtype=np.float64)
        task.set_target(target)
        return
    raise ValueError(f"the solver has no frame task for {frame!r}")


__all__ = [
    "ArmIkLayout",
    "ArmIkSettings",
    "EefArmIkController",
    "HELD_REFERENCE",
    "ON_JOINT_LIMIT",
    "ON_SOLVER_FAILURE",
    "POSTURE_REFERENCE",
    "SOLVE_RATE",
]
