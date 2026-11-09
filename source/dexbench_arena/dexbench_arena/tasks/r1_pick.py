# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""DB-R1: pick any one fruit of the instructed kind from a pile; keep all other fruit on the tray."""

from dataclasses import MISSING

import torch
import warp as wp
from isaaclab.managers import TerminationTermCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import matrix_from_quat, quat_apply
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_base import TaskBase

from dexbench_arena.contact_materials import WUJI_GRASP_CONTACT_MATERIAL
from dexbench_arena.tasks import r1_rules
from dexbench_arena.tasks.r1_clutter import R1_TASK_DESCRIPTION
from dexbench_arena.tasks.predicates.geometry import root_quat_w
from dexbench_arena.tasks.predicates.hold import sustained, sustained_count
from dexbench_arena.tasks.predicates.object_state import get_env, get_root_pos_w, get_root_lin_vel_w, get_root_ang_vel_w
from dexbench_arena.tasks.predicates.status import publish
from dexbench_arena.tasks.task_termination import task_termination_cfg


def state(env, objects, fruit_id, rules, hand_sensor_names=()):
    """:func:`r1_rules.evaluate` on the pile's current state; ``rules`` are its keyword arguments."""
    base = get_env(env)
    key = (base.common_step_counter, id(rules))
    cached = getattr(base, '_r1_state', None)
    if cached is not None and cached[0] == key:
        return cached[1]
    masks = _evaluate(base, objects, fruit_id, rules, hand_sensor_names)
    base._r1_state = (key, masks)
    return masks


def _evaluate(base, objects, fruit_id, rules, hand_sensor_names):
    roots = torch.stack([get_root_pos_w(base, o['name']) for o in objects], dim=1)
    quats = torch.stack([root_quat_w(base, o['name']) for o in objects], dim=1)
    offsets = roots.new_tensor([o['center_local'] for o in objects]).unsqueeze(0).expand_as(roots)
    centers = roots + quat_apply(quats, offsets) - base.scene.env_origins[:, None, :]
    half = torch.einsum('nfij,fj->nfi', matrix_from_quat(quats).square(),
                        roots.new_tensor([o['half_extents'] for o in objects]).square()).sqrt()
    lin = torch.stack([get_root_lin_vel_w(base, o['name']) for o in objects], dim=1)
    ang = torch.stack([get_root_ang_vel_w(base, o['name']) for o in objects], dim=1)
    forces = centers.new_zeros(centers.shape[:2])
    for sensor_name in hand_sensor_names:
        data = base.scene.sensors[sensor_name].data.force_matrix_w
        if data is None:
            raise ValueError('R1 hand sensor has no filtered fruit forces')
        forces += wp.to_torch(data).norm(dim=-1).sum(dim=1)
    eligible = torch.tensor([o['fruit_id'] == fruit_id for o in objects], device=centers.device)
    masses = centers.new_tensor([o['mass_kg'] for o in objects])
    return r1_rules.evaluate(centers, half, lin, ang, forces, eligible, masses, **rules)


def success(env, objects, fruit_id, rules, hand_sensor_names=(), hold_time_s=1.0):
    masks = state(env, objects, fruit_id, rules, hand_sensor_names)
    held = sustained(env, masks['candidates'], hold_time_s, 'r1:candidates').any(-1)
    publish(env, 'db_r1', r1_rules.operator_status(masks, held, sustained_count(env, 'r1:candidates'),
                                                   get_env(env).step_dt, hold_time_s))
    return held


def failure(env, objects, fruit_id, rules, hand_sensor_names=()):
    return state(env, objects, fruit_id, rules, hand_sensor_names)['failed']


@configclass
class R1SceneCfg:
    r1_left_contact: ContactSensorCfg | None = None
    r1_right_contact: ContactSensorCfg | None = None


@configclass
class R1TerminationsCfg:
    success: TerminationTermCfg = MISSING
    fruit_lost: TerminationTermCfg = MISSING


class R1PickTask(TaskBase):
    """Pick any fruit of the pile's kind into the basket, every other fruit left on the tray."""

    def __init__(self, recipe, objects, *, source_bounds, basket_bounds, workspace_xy_bounds, dropped_min_z_m,
                 basket_margin_m, released_max_force_n, released_max_weight_fraction, still_gates, hold_time_s,
                 episode_length_s,
                 robot_joint_pos_overrides=None):
        if not 0 < hold_time_s <= 10:
            raise ValueError('R1 hold_time_s must be in (0, 10]')
        super().__init__(episode_length_s=episode_length_s, task_description=R1_TASK_DESCRIPTION)
        self.objects = objects
        self.robot_joint_pos_overrides = robot_joint_pos_overrides
        self.hand_contact_material = WUJI_GRASP_CONTACT_MATERIAL
        self.scene_cfg = R1SceneCfg()
        rules = {"source_bounds": source_bounds, "basket_bounds": basket_bounds,
                 "workspace_xy_bounds": workspace_xy_bounds, "dropped_min_z_m": dropped_min_z_m,
                 "basket_margin_m": basket_margin_m, "released_max_force_n": released_max_force_n,
                 "released_max_weight_fraction": released_max_weight_fraction,
                 "still_gates": tuple(still_gates)}
        params = {'objects': objects, 'fruit_id': recipe['fruit_id'], 'rules': rules}
        self.termination_cfg = R1TerminationsCfg(
            success=TerminationTermCfg(func=success, params={**params, 'hold_time_s': hold_time_s}),
            fruit_lost=TerminationTermCfg(func=failure, params=dict(params)))

    def configure_for_embodiment(self, embodiment):
        robot = getattr(getattr(embodiment, 'scene_config', None), 'robot', None)
        if robot is None:
            return
        exprs = getattr(embodiment, 'held_object_contact_body_exprs', None)
        if not exprs or set(exprs) != {'left', 'right'}:
            raise ValueError('R1 requires both held_object_contact_body_exprs to judge release')
        sensor_names = []
        for side, expr in exprs.items():
            name = f'r1_{side}_contact'
            setattr(self.scene_cfg, name, ContactSensorCfg(
                prim_path=f'{robot.prim_path}/({expr})', update_period=0.0,
                filter_prim_paths_expr=[f'{{ENV_REGEX_NS}}/{o["name"]}(/.*)?' for o in self.objects]))
            sensor_names.append(name)
        for term in (self.termination_cfg.success, self.termination_cfg.fruit_lost):
            term.params['hand_sensor_names'] = tuple(sensor_names)

    def get_scene_cfg(self):
        return self.scene_cfg

    def get_termination_cfg(self):
        return task_termination_cfg(self.episode_length_s, self.termination_cfg)

    def get_events_cfg(self):
        return None

    def get_mimic_env_cfg(self, arm_mode):
        return None

    def get_metrics(self):
        return [SuccessRateMetric()]
