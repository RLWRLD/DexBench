# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Extract one named utensil while retaining the rest in the support hand."""

from dataclasses import MISSING

import torch
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg, TerminationTermCfg
from isaaclab.utils import configclass
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

from dexbench_arena.embodiments.joint_state import as_torch
from dexbench_arena.tasks.predicates.geometry import object_center_env, root_quat_w
from dexbench_arena.tasks.predicates.grasp_feedback import SupportGraspFeedback
from dexbench_arena.tasks.predicates.hold import sustained, reset_sustained
from dexbench_arena.tasks.predicates.object_state import get_root_pos_w
from dexbench_arena.tasks.task_termination import task_termination_cfg


BIN_XY = ((3.346, .130), (3.454, .410))
BIN_FLOOR = .757
BIN_RIM = .862
TABLE_TOP = .750


def utensil_status(env, object_names, object_bounds, reference_body):
    """Return retained, released-in-bin and dropped masks, without a speed gate."""
    robot = env.scene['robot']
    positions = as_torch(robot.data.body_pos_w)
    wrist = positions[:, robot.body_names.index(reference_body)]
    body_env = positions - env.scene.env_origins[:, None]
    retained, placed, dropped = [], [], []
    for name, bounds in zip(object_names, object_bounds, strict=True):
        box = AxisAlignedBoundingBox(tuple(bounds[0]), tuple(bounds[1])).to(env.device)
        center = object_center_env(env, name, box.center[0])
        root = get_root_pos_w(env, name) - env.scene.env_origins
        rotated = box.rotated_by_quat(root_quat_w(env, name))
        lower, upper = rotated.min_point + root, rotated.max_point + root
        distance = torch.linalg.vector_norm(center + env.scene.env_origins - wrist, dim=-1)
        retained.append((distance < .18) & (lower[:, 2] > TABLE_TOP + .02))
        lo, hi = center.new_tensor(BIN_XY)
        inside = ((center[:, :2] >= lo) & (center[:, :2] <= hi)).all(dim=-1)
        gap = lower[:, 2] - BIN_FLOOR
        seated = (gap > -.020) & (gap < .008) & (upper[:, 2] < BIN_RIM)
        near = ((body_env >= lower[:, None] - .035) & (body_env <= upper[:, None] + .035)).all(dim=-1)
        placed.append(inside & seated & ~near.any(dim=1))
        dropped.append(lower[:, 2] < TABLE_TOP - .15)
    return tuple(torch.stack(values, dim=1) for values in (retained, placed, dropped))


def extracted_utensil_placed(env, object_names, object_bounds, reference_body, target_object, hold_time_s=.5):
    held, placed, dropped = utensil_status(env, object_names, object_bounds, reference_body)
    target = object_names.index(target_object)
    others = [i for i in range(len(object_names)) if i != target]
    ok = placed[:, target] & held[:, others].all(dim=1) & ~placed[:, others].any(dim=1) & ~dropped.any(dim=1)
    return sustained(env, ok, hold_time_s=hold_time_s, key='h3_extract')


def utensil_dropped(env, object_names, object_bounds, reference_body):
    return utensil_status(env, object_names, object_bounds, reference_body)[2].any(dim=1)


@configclass
class ExtractionTerminations:
    success: TerminationTermCfg = MISSING
    dropped: TerminationTermCfg = MISSING


@configclass
class ExtractionEvents:
    support_feedback: EventTermCfg = MISSING
    reset_hold: EventTermCfg = EventTermCfg(func=reset_sustained, mode='reset', params={'key': 'h3_extract'})


class ExtractUtensilTask(TaskBase):
    """Fixed-target task with an automatically controlled, yielding support hand."""

    def __init__(self, assets, target, joint_targets, episode_length_s=25.):
        noun = 'table knife' if target == 'knife' else target
        super().__init__(episode_length_s, f'Take the {noun} from your left hand and place it in the bin. '
                         'Keep the other utensils in your left hand.')
        self.assets = assets
        self.object_name = next(a['object_name'] for a in assets if a['type'] == target)
        self.robot_joint_pos_overrides = dict(joint_targets)
        self.reference_body = None
        self.events = None

    def configure_for_embodiment(self, embodiment):
        if embodiment.support_hand_mode != 'init_q':
            raise ValueError('H3 requires the saved support-grasp action')
        self.reference_body = embodiment.support_grasp_reference_body
        params = dict(robot_name='robot', reference_body=self.reference_body,
                      finger_body_patterns=embodiment.support_grasp_finger_body_patterns,
                      object_names=[a['object_name'] for a in self.assets],
                      object_bounds=[a['body_bounds'] for a in self.assets], target_object=self.object_name,
                      support_action='joint_pos', feedback_dt=1 / 30)
        self.events = ExtractionEvents(support_feedback=EventTermCfg(
            func=SupportGraspFeedback, mode='interval', is_global_time=True,
            interval_range_s=(1 / 30, 1 / 30), params=params))

    def get_termination_cfg(self):
        if self.reference_body is None:
            raise RuntimeError('Configure the support embodiment before building H3 terms')
        params = dict(object_names=[a['object_name'] for a in self.assets],
                      object_bounds=[a['body_bounds'] for a in self.assets], reference_body=self.reference_body)
        return task_termination_cfg(self.episode_length_s, ExtractionTerminations(
            success=TerminationTermCfg(func=extracted_utensil_placed, params=dict(params, target_object=self.object_name)),
            dropped=TerminationTermCfg(func=utensil_dropped, params=params)))

    def get_scene_cfg(self):
        return None

    def get_events_cfg(self):
        return self.events

    def get_mimic_env_cfg(self, arm_mode):
        return None

    def get_metrics(self):
        return [SuccessRateMetric()]

    def get_viewer_cfg(self):
        return ViewerCfg(eye=(3.05, .85, 1.65), lookat=(3.5, .4, 1.05))
