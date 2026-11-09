# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Record each robot joint's actuator effort as Isaac Lab reports it."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.managers import RecorderTerm, RecorderTermCfg
from isaaclab.utils.configclass import configclass

from dexbench_arena.embodiments.episode_metadata import install_episode_metadata
from dexbench_arena.embodiments.recorder_terms import add_declared_recorder_term

JOINT_TORQUE_DATASET_KEY = "joint_torque"
JOINT_TORQUE_SCHEMA_VERSION = 1


def joint_torque_metadata(joint_names: Sequence[str]) -> dict:
    """What the recorded array means, for the dataset's episode metadata."""
    return {
        "schema_version": JOINT_TORQUE_SCHEMA_VERSION,
        "joint_names": list(joint_names),
        "fields": ["applied_effort"],
        "shape": "(steps, joints)",
        "unit": "N·m (revolute), N (prismatic)",
        "sampled": "computed from the joint state at the start of the last physics step of each env step",
        "source": "Isaac Lab articulation.actuators.applied_effort: for implicit actuators, the PD drive "
        "recomputed by Isaac Lab and clipped at the effort limit (an estimate of the solver's force)",
    }


def applied_joint_effort(robot) -> torch.Tensor:
    """Every joint's actuator effort, (num_envs, joints) in the articulation's joint order."""
    actuators = robot.actuators
    finite = torch.isfinite(actuators.computed_effort.torch)
    return torch.where(finite, actuators.applied_effort.torch, torch.nan)


class JointTorqueRecorder(RecorderTerm):
    """Per-step actuator effort of every robot joint, in the articulation's joint order."""

    def __init__(self, cfg: JointTorqueRecorderCfg, env):
        super().__init__(cfg, env)
        self._metadata_installed = False

    def record_post_step(self):
        robot = self._env.scene[self.cfg.asset_name]
        if not self._metadata_installed:
            install_episode_metadata(self._env, JOINT_TORQUE_DATASET_KEY, joint_torque_metadata(robot.joint_names))
            self._metadata_installed = True
        return JOINT_TORQUE_DATASET_KEY, {"applied_effort": applied_joint_effort(robot)}


@configclass
class JointTorqueRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = JointTorqueRecorder
    asset_name: str = "robot"
    """Scene name of the articulation whose joints are recorded."""


class JointTorqueMixin:
    """Embodiment capability: an opt-out recorder of the robot's joint actuator effort."""

    record_joint_torque: bool = False

    def enable_joint_torque_recording(self) -> None:
        """Record the joint actuator effort in this embodiment's datasets."""
        self.record_joint_torque = True

    def configure_joint_torque_recorders(self, env_cfg) -> None:
        """Put the recorder term next to the embodiment's other per-step terms (``add_declared_recorder_term``)."""
        if self.record_joint_torque:
            add_declared_recorder_term(env_cfg, "record_post_step_joint_torque", JointTorqueRecorderCfg())
