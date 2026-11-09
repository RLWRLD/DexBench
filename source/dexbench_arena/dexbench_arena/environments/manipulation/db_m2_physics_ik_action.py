# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""M2's optional physics-rate servo on the 30 Hz joint-recording embodiment."""

from __future__ import annotations

import torch
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.managers.manager_term_cfg import RecorderTermCfg
from isaaclab.managers.recorder_manager import RecorderTerm
from isaaclab.utils.configclass import configclass

from dexbench_arena.embodiments.recorder_terms import add_declared_recorder_term


class M2PhysicsIKAction(JointPositionAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.servo = None
        self.physics_targets = []

    def bind_servo(self, servo):
        if self._env.cfg.decimation > 1 and self._env.sim.physics_manager.handles_decimation():
            raise ValueError("M2 physics-rate IK requires an external decimation loop; this Newton runtime "
                             "folds it into sim.step and cannot refresh Python IK between physics steps")
        self.servo = servo

    def process_actions(self, actions):
        self.physics_targets = []
        super().process_actions(actions)

    def apply_actions(self):
        if self.physics_targets and self.servo is not None:
            refreshed = self.servo()
            if refreshed is not None:
                super().process_actions(refreshed)
        self.physics_targets.append(self.processed_actions.clone())
        super().apply_actions()

    def reset(self, env_ids=None):
        super().reset(env_ids)
        self.servo = None
        self.physics_targets = []


class M2PhysicsTargetsRecorder(RecorderTerm):
    def record_post_step(self):
        term = self._env.action_manager.get_term("joint_pos")
        if not term.physics_targets:
            return None, None
        return "physics_joint_targets", torch.stack(term.physics_targets, dim=1)


@configclass
class M2PhysicsTargetsRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = M2PhysicsTargetsRecorder


def with_physics_ik_action(previous):
    """Use Arena's cfg callback to install the opt-in action term and 30 Hz substep-command audit rows."""
    def callback(env_cfg):
        if previous is not None:
            env_cfg = previous(env_cfg) or env_cfg
        term = env_cfg.actions.joint_pos
        if term.class_type not in (JointPositionAction, M2PhysicsIKAction, JointPositionActionCfg().class_type):
            raise ValueError(f"M2 physics IK requires the native joint-position action term, got {term.class_type}")
        term.class_type = M2PhysicsIKAction
        add_declared_recorder_term(env_cfg, "record_m2_physics_targets", M2PhysicsTargetsRecorderCfg())
        return env_cfg
    return callback
