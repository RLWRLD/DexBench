# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""DB-M1 insertion success and observations, shared by in-hand and tabletop starts."""

from __future__ import annotations

from dataclasses import MISSING
from functools import partial
from typing import Any

import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.configclass import configclass
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg

from dexbench_arena.tasks.predicates.drawer_insertion import (
    SlotContactForce,
    drawer_in_slot_window,
    drawer_seated,
)
from dexbench_arena.tasks.task_termination import task_termination_cfg

SLOT_CONTACT_SENSOR_NAME = "slot_contact"
"""The task's contact sensor: the washer's colliders against the drawer's, per collider pair."""


@configclass
class InsertDrawerTerminationsCfg:
    """Success and failure terms; the episode limit is the task's ``episode_length_s``."""

    success: TerminationTermCfg = MISSING
    drawer_dropped: TerminationTermCfg = MISSING


class DexBenchInsertDrawerTask(TaskBase):
    """Seat the drawer in its washer; the environment supplies the episode's start."""

    def __init__(
        self,
        object_name: str,
        washer_name: str,
        seat_in_washer_xyz: tuple[float, float, float],
        depth_window_m: tuple[float, float],
        max_lateral_error_m: float,
        max_tilt_deg: float,
        hold_time_s: float,
        dropped_below_z_m: float,
        drawer_sensing_colliders: tuple[str, ...],
        washer_counterpart_colliders: tuple[str, ...],
        slot_contact_regions: dict[str, tuple[str, ...]],
        progress_stage_depths_m: tuple[float, float, float] | None = None,
        progress_align_max_lateral_error_m: float | None = None,
        progress_align_max_tilt_deg: float | None = None,
        episode_length_s: float | None = None,
        task_description: str | None = None,
        robot_joint_pos_overrides: dict[str, float] | None = None,
        max_linear_speed_m_s: float | None = None,
    ):
        super().__init__(episode_length_s=episode_length_s, task_description=task_description)
        self.object_name = object_name
        self._washer_name = washer_name
        self._seat_in_washer_xyz = tuple(seat_in_washer_xyz)
        self._progress_stage_depths_m = progress_stage_depths_m
        self._progress_align = (progress_align_max_lateral_error_m, progress_align_max_tilt_deg)
        self.robot_joint_pos_overrides = dict(robot_joint_pos_overrides) if robot_joint_pos_overrides else None
        self.termination_cfg = InsertDrawerTerminationsCfg(
            success=TerminationTermCfg(
                func=drawer_seated,
                params={
                    "object_name": object_name,
                    "receiver_name": washer_name,
                    "seat_in_receiver_xyz": tuple(seat_in_washer_xyz),
                    "depth_window_m": tuple(depth_window_m),
                    "max_lateral_error_m": float(max_lateral_error_m),
                    "max_tilt_deg": float(max_tilt_deg),
                    "hold_time_s": float(hold_time_s),
                    **({"max_linear_speed_m_s": float(max_linear_speed_m_s)} if max_linear_speed_m_s is not None else {}),
                },
            ),
            drawer_dropped=TerminationTermCfg(
                func=base_mdp.root_height_below_minimum,
                params={"minimum_height": float(dropped_below_z_m), "asset_cfg": SceneEntityCfg(object_name)},
            ),
        )
        self.scene_cfg = InsertDrawerSceneCfg(
            slot_contact=ContactSensorCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{object_name}/Body",
                sensor_shape_prim_expr=[f"{{ENV_REGEX_NS}}/{object_name}/{expr}" for expr in drawer_sensing_colliders],
                filter_shape_prim_expr=[
                    f"{{ENV_REGEX_NS}}/{washer_name}/{expr}" for expr in washer_counterpart_colliders
                ],
                update_period=0.0,
            )
        )
        self.observation_cfg = InsertDrawerObservationsCfg()
        for term in ("drawer_pos", "drawer_quat", "drawer_lin_vel", "drawer_ang_vel"):
            getattr(self.observation_cfg.task_obs, term).params["asset_cfg"] = SceneEntityCfg(object_name)
        self.observation_cfg.task_obs.slot_contact.params["regions"] = {
            k: tuple(v) for k, v in slot_contact_regions.items()
        }

    def configure_for_embodiment(self, embodiment: Any) -> None:
        """Sense each hand's contact with the drawer on the bodies the embodiment declares."""
        robot = getattr(getattr(embodiment, "scene_config", None), "robot", None)
        if robot is None:
            return
        exprs = getattr(embodiment, "held_object_contact_body_exprs", None)
        if not exprs:
            raise ValueError(
                f"DB-M1 insert_drawer needs the embodiment to declare held_object_contact_body_exprs; "
                f"{getattr(embodiment, 'name', type(embodiment).__name__)} declares none"
            )
        for side in ("left", "right"):
            setattr(
                self.scene_cfg,
                f"{side}_cs",
                ContactSensorCfg(
                    prim_path=f"{robot.prim_path}/({exprs[side]})",
                    filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/{self.object_name}/Body"],
                    update_period=0.0,
                ),
            )

    def get_scene_cfg(self) -> Any:
        return self.scene_cfg

    def get_observation_cfg(self) -> Any:
        return self.observation_cfg

    def get_termination_cfg(self) -> TaskTerminationCfg:
        return task_termination_cfg(
            self.episode_length_s,
            self.termination_cfg,
            partial_credit=self._partial_credit_chain(),
            partial_credit_name="insert_drawer",
            partial_credit_description=(
                "Align the drawer with the slot opening, engage the rails, push it past half depth, seat it."
            ),
        )

    def _partial_credit_chain(self) -> list | None:
        """Partial credit: aligned at the slot opening -> rails engaged -> past half depth -> seated."""
        if self._progress_stage_depths_m is None:
            return None
        max_lateral, max_tilt = self._progress_align
        return [
            partial(
                drawer_in_slot_window,
                object_name=self.object_name,
                receiver_name=self._washer_name,
                seat_in_receiver_xyz=self._seat_in_washer_xyz,
                min_depth_m=float(depth),
                max_lateral_error_m=max_lateral,
                max_tilt_deg=max_tilt,
            )
            for depth in self._progress_stage_depths_m
        ] + [self.termination_cfg.success]

    def get_events_cfg(self) -> Any:
        return None

    def get_mimic_env_cfg(self, arm_mode: Any) -> Any:
        return None

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]


@configclass
class InsertDrawerSceneCfg:
    """The task's sensors."""

    slot_contact: ContactSensorCfg = MISSING
    left_cs: ContactSensorCfg | None = None
    right_cs: ContactSensorCfg | None = None


@configclass
class InsertDrawerObservationsCfg:
    """The drawer's state and the washer's push on it, for datasets and evaluation records."""

    @configclass
    class TaskObsCfg(ObsGroup):
        drawer_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("drawer")})
        drawer_quat = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("drawer")})
        drawer_lin_vel = ObsTerm(func=base_mdp.root_lin_vel_w, params={"asset_cfg": SceneEntityCfg("drawer")})
        drawer_ang_vel = ObsTerm(func=base_mdp.root_ang_vel_w, params={"asset_cfg": SceneEntityCfg("drawer")})
        slot_contact = ObsTerm(
            func=SlotContactForce,
            params={"sensor_name": SLOT_CONTACT_SENSOR_NAME, "regions": {}},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    task_obs: TaskObsCfg = TaskObsCfg()
