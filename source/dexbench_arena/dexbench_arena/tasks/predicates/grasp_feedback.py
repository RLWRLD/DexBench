# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Wrist-relative object geometry for a target-aware support-grasp action."""

import math

import torch
from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import quat_apply_inverse

from dexbench_arena.embodiments.joint_state import as_torch
from dexbench_arena.evaluation.replay_guards import deterministic_reset
from dexbench_arena.tasks.predicates.geometry import object_center_env, root_quat_w
from dexbench_arena.tasks.predicates.object_state import get_root_pos_w


@deterministic_reset
class SupportGraspFeedback(ManagerTermBase):
    """Update one support action at a fixed, shared simulation-time cadence."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        params = cfg.params
        self.robot = env.scene[params['robot_name']]
        self.wrist_id = self.robot.body_names.index(params['reference_body'])
        self.finger_ids = [self.robot.find_bodies(pattern)[0] for pattern in params['finger_body_patterns']]
        if not self.finger_ids or any(not ids for ids in self.finger_ids):
            raise ValueError('Support grasp needs a nonempty body group for every finger')
        if params['target_object'] not in params['object_names']:
            raise ValueError('Support grasp needs an explicit target object')
        if not math.isfinite(params['feedback_dt']) or params['feedback_dt'] <= 0:
            raise ValueError('Support grasp feedback_dt must be finite and positive')
        steps = params['feedback_dt'] / env.step_dt
        if steps < 1 or not math.isclose(steps, round(steps), abs_tol=1e-6):
            raise ValueError('Support grasp feedback_dt must be an integer number of control steps')
        if (cfg.mode != 'interval' or not cfg.is_global_time
                or tuple(cfg.interval_range_s) != (params['feedback_dt'], params['feedback_dt'])):
            raise ValueError('Support grasp needs a fixed global interval matching feedback_dt')
        if len(set(params['object_names'])) != len(params['object_names']):
            raise ValueError('Support grasp object names must be unique')
        if len(params['object_bounds']) != len(params['object_names']):
            raise ValueError('Support grasp needs bounds for every object')
        bounds = torch.tensor(params['object_bounds'])
        if bounds.shape != (len(params['object_names']), 2, 3) or not bool(torch.isfinite(bounds).all()):
            raise ValueError('Support grasp bounds must be finite min/max XYZ')
        if not bool((bounds[:, 0] < bounds[:, 1]).all()):
            raise ValueError('Support grasp min bounds must be below max bounds')

    def __call__(self, env, env_ids, object_names, object_bounds, target_object,
                 reference_body, finger_body_patterns, robot_name='robot',
                 support_action='joint_pos', feedback_dt=1 / 30):
        if env_ids is not None:
            raise ValueError('Support grasp feedback requires a global interval event')
        positions = as_torch(self.robot.data.body_pos_w)
        rotations = as_torch(self.robot.data.body_quat_w)
        centers = torch.stack([
            object_center_env(env, name, tuple((lo + hi) / 2 for lo, hi in zip(*bounds, strict=True)))
            + env.scene.env_origins
            for name, bounds in zip(object_names, object_bounds, strict=True)
        ], dim=1)
        relative = quat_apply_inverse(
            rotations[:, self.wrist_id, None].expand(-1, len(object_names), -1),
            centers - positions[:, self.wrist_id, None],
        )
        distances = []
        for name, bounds in zip(object_names, object_bounds, strict=True):
            points = quat_apply_inverse(
                root_quat_w(env, name)[:, None].expand(-1, positions.shape[1], -1),
                positions - get_root_pos_w(env, name)[:, None],
            )
            lower, upper = points.new_tensor(bounds)
            outside = torch.maximum(lower - points, points - upper).clamp_min(0)
            body_distances = torch.linalg.vector_norm(outside, dim=-1)
            distances.append(torch.stack([body_distances[:, ids].amin(dim=1) for ids in self.finger_ids], dim=1))
        target_mask = torch.tensor([name == target_object for name in object_names],
                                   dtype=torch.bool, device=env.device).expand(env.num_envs, -1)
        env.action_manager.get_term(support_action).update_support_grasp(
            relative, target_mask, torch.stack(distances, dim=-1), feedback_dt,
        )
