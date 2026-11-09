# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""DB-M4 task: pick the leveling leg from its holder and hand-start it into the base cabinet's bracket hole."""

from __future__ import annotations

from dataclasses import MISSING
from typing import Any

import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.utils.configclass import configclass
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg

from dexbench_arena.contact_materials import HandContactMaterialCfg
from dexbench_arena.tasks.predicates.threaded_fastening import stud_engagement, stud_hand_started
from dexbench_arena.tasks.task_termination import task_termination_cfg


@configclass
class HandStartLegTerminationsCfg:
    """Success and failure terms; the episode limit is the task's ``episode_length_s``."""

    success: TerminationTermCfg = MISSING
    leg_dropped: TerminationTermCfg = MISSING


class DexBenchHandStartLegTask(TaskBase):
    """Pick a threaded leg from its holder and turn it into its hole until the thread has taken it."""

    def __init__(
        self,
        object_name: str,
        receiver_name: str,
        hole_in_receiver_xyz: tuple[float, float, float],
        hole_axis_in_receiver: tuple[float, float, float],
        stud_axis_in_object: tuple[float, float, float],
        min_depth_m: float,
        max_offset_m: float,
        max_tilt_deg: float,
        hold_time_s: float,
        dropped_below_z_m: float,
        episode_length_s: float | None = None,
        task_description: str | None = None,
        robot_joint_pos_overrides: dict[str, float] | None = None,
        hand_contact_material: HandContactMaterialCfg | None = None,
    ):
        super().__init__(episode_length_s=episode_length_s, task_description=task_description)
        self.object_name = object_name
        self.hand_contact_material = hand_contact_material
        self.robot_joint_pos_overrides = dict(robot_joint_pos_overrides) if robot_joint_pos_overrides else None
        hole = {
            "object_name": object_name,
            "receiver_name": receiver_name,
            "hole_in_receiver_xyz": tuple(hole_in_receiver_xyz),
            "hole_axis_in_receiver": tuple(hole_axis_in_receiver),
            "stud_axis_in_object": tuple(stud_axis_in_object),
        }
        self.termination_cfg = HandStartLegTerminationsCfg(
            success=TerminationTermCfg(
                func=stud_hand_started,
                params={
                    **hole,
                    "min_depth_m": float(min_depth_m),
                    "max_offset_m": float(max_offset_m),
                    "max_tilt_deg": float(max_tilt_deg),
                    "hold_time_s": float(hold_time_s),
                },
            ),
            leg_dropped=TerminationTermCfg(
                func=base_mdp.root_height_below_minimum,
                params={"minimum_height": float(dropped_below_z_m), "asset_cfg": SceneEntityCfg(object_name)},
            ),
        )
        self.observation_cfg = HandStartLegObservationsCfg()
        for term in ("leg_pos", "leg_quat", "leg_lin_vel", "leg_ang_vel"):
            getattr(self.observation_cfg.task_obs, term).params["asset_cfg"] = SceneEntityCfg(object_name)
        self.observation_cfg.task_obs.thread_engagement.params.update(hole)

    def get_scene_cfg(self) -> Any:
        return None

    def get_observation_cfg(self) -> Any:
        return self.observation_cfg

    def get_termination_cfg(self) -> TaskTerminationCfg:
        return task_termination_cfg(self.episode_length_s, self.termination_cfg)

    def get_events_cfg(self) -> Any:
        return None

    def get_mimic_env_cfg(self, arm_mode: Any) -> Any:
        return None

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]


@configclass
class HandStartLegObservationsCfg:
    """The leg's state and its engagement in the hole, for datasets and evaluation records."""

    @configclass
    class TaskObsCfg(ObsGroup):
        leg_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("leg")})
        leg_quat = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("leg")})
        leg_lin_vel = ObsTerm(func=base_mdp.root_lin_vel_w, params={"asset_cfg": SceneEntityCfg("leg")})
        leg_ang_vel = ObsTerm(func=base_mdp.root_ang_vel_w, params={"asset_cfg": SceneEntityCfg("leg")})
        thread_engagement = ObsTerm(func=stud_engagement, params={})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    task_obs: TaskObsCfg = TaskObsCfg()
