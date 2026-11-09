# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""DB-M2 task: push a loose harness plug into its mate on the refrigerator liner and leave the other one seated."""

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

from dexbench_arena.tasks.predicates.connector_insertion import (
    hand_plug_force,
    plug_above,
    plug_along_axis,
    plug_below,
    plug_displaced,
    plug_insertion_observation,
    plug_seated,
)
from dexbench_arena.tasks.task_termination import task_termination_cfg


HAND_SENSORS = ("left_cs", "right_cs")
"""The task's hand-plug contact sensors (``configure_for_embodiment``); observed (``hand_plug_force``), never scored."""


@configclass
class InsertConnectorTerminationsCfg:
    """Success and failure terms; the episode limit is the task's ``episode_length_s``."""

    success: TerminationTermCfg = MISSING
    plug_dropped: TerminationTermCfg = MISSING
    neighbor_unseated: TerminationTermCfg = MISSING


class DexBenchInsertConnectorTask(TaskBase):
    """Insert the task plug into its mate and seat it, without knocking the other plug out (DB-M2)."""

    def __init__(
        self,
        plug_name: str,
        plug_prim_path: str,
        neighbor_plug_name: str,
        seat_xyz: tuple[float, float, float],
        pull_out_axis: tuple[float, float, float],
        seated_quat_xyzw: tuple[float, float, float, float],
        stroke_m: float,
        neighbor_seat_xyz: tuple[float, float, float],
        max_depth_m: float,
        max_lateral_m: float,
        max_tilt_deg: float,
        hold_time_s: float,
        dropped_below_z_m: float,
        neighbor_max_displacement_m: float,
        progress_align_max_lateral_error_m: float | None = None,
        progress_align_max_tilt_deg: float | None = None,
        progress_lift_z_m: float | None = None,
        progress_preinsert: tuple[float, float, float] | None = None,
        episode_length_s: float | None = None,
        task_description: str | None = None,
        robot_joint_pos_overrides: dict[str, float] | None = None,
    ):
        super().__init__(episode_length_s=episode_length_s, task_description=task_description)
        self.object_name = plug_name
        self._plug_prim_path = plug_prim_path
        self._geometry = {
            "plug_name": plug_name,
            "seat_xyz": tuple(seat_xyz),
            "pull_out_axis": tuple(pull_out_axis),
            "seated_quat_xyzw": tuple(seated_quat_xyzw),
        }
        self._stroke_m = float(stroke_m)
        self._progress_align = (progress_align_max_lateral_error_m, progress_align_max_tilt_deg)
        self._progress_lift_z_m = progress_lift_z_m
        self._progress_preinsert = progress_preinsert
        self.robot_joint_pos_overrides = dict(robot_joint_pos_overrides) if robot_joint_pos_overrides else None
        self.termination_cfg = InsertConnectorTerminationsCfg(
            success=TerminationTermCfg(
                func=plug_seated,
                params={
                    **self._geometry,
                    "max_depth_m": float(max_depth_m),
                    "max_lateral_m": float(max_lateral_m),
                    "max_tilt_deg": float(max_tilt_deg),
                    "hold_time_s": float(hold_time_s),
                },
            ),
            plug_dropped=TerminationTermCfg(
                func=plug_below, params={"plug_name": plug_name, "min_z_m": float(dropped_below_z_m)}
            ),
            neighbor_unseated=TerminationTermCfg(
                func=plug_displaced,
                params={
                    "plug_name": neighbor_plug_name,
                    "seat_xyz": tuple(neighbor_seat_xyz),
                    "max_displacement_m": float(neighbor_max_displacement_m),
                },
            ),
        )
        self.scene_cfg = InsertConnectorSceneCfg()
        self.observation_cfg = InsertConnectorObservationsCfg()
        for term in ("plug_pos", "plug_quat", "plug_lin_vel", "plug_ang_vel"):
            getattr(self.observation_cfg.task_obs, term).params["asset_cfg"] = SceneEntityCfg(plug_name)
        self.observation_cfg.task_obs.neighbor_plug_pos.params["asset_cfg"] = SceneEntityCfg(neighbor_plug_name)
        self.observation_cfg.task_obs.plug_insertion.params.update(self._geometry)

    def configure_for_embodiment(self, embodiment: Any) -> None:
        """Sense each hand's contact with the task plug on the bodies the embodiment declares."""
        robot = getattr(getattr(embodiment, "scene_config", None), "robot", None)
        if robot is None:
            return
        exprs = getattr(embodiment, "held_object_contact_body_exprs", None)
        if not exprs:
            raise ValueError(
                f"DB-M2 insert_connector needs the embodiment to declare held_object_contact_body_exprs; "
                f"{getattr(embodiment, 'name', type(embodiment).__name__)} declares none"
            )
        for side in ("left", "right"):
            setattr(
                self.scene_cfg,
                f"{side}_cs",
                ContactSensorCfg(
                    prim_path=f"{robot.prim_path}/({exprs[side]})",
                    filter_prim_paths_expr=[self._plug_prim_path],
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
            partial_credit_name="insert_connector",
            partial_credit_description=("Lift the plug, bring it onto its axis at the mate, half way in, then seat it."
                                        if self._progress_lift_z_m is not None
                                        else "Bring the plug half way into its mate on the axis, then seat it."),
        )

    def _partial_credit_chain(self) -> list | None:
        """Partial credit: [lifted -> at the pre-insertion point ->] half way along the stroke -> seated."""
        max_lateral, max_tilt = self._progress_align
        if max_lateral is None or max_tilt is None:
            return None
        pick = []
        if self._progress_lift_z_m is not None:
            pick.append(partial(plug_above, plug_name=self._geometry["plug_name"], min_z_m=float(self._progress_lift_z_m)))
        if self._progress_preinsert is not None:
            depth, lateral, tilt = self._progress_preinsert
            pick.append(partial(plug_along_axis, **self._geometry, max_depth_m=float(depth),
                                max_lateral_error_m=float(lateral), max_tilt_deg=float(tilt)))
        return pick + [
            partial(
                plug_along_axis,
                **self._geometry,
                max_depth_m=self._stroke_m / 2,
                max_lateral_error_m=float(max_lateral),
                max_tilt_deg=float(max_tilt),
            ),
            self.termination_cfg.success,
        ]

    def get_events_cfg(self) -> Any:
        return None

    def get_mimic_env_cfg(self, arm_mode: Any) -> Any:
        return None

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]


@configclass
class InsertConnectorSceneCfg:
    """The task's sensors."""

    left_cs: ContactSensorCfg | None = None
    right_cs: ContactSensorCfg | None = None


@configclass
class InsertConnectorObservationsCfg:
    """The task plug's state and its insertion along the axis, where the other plug is, and the hands' force on the
    plug, for datasets and records."""

    @configclass
    class TaskObsCfg(ObsGroup):
        plug_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("plug")})
        plug_quat = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("plug")})
        plug_lin_vel = ObsTerm(func=base_mdp.root_lin_vel_w, params={"asset_cfg": SceneEntityCfg("plug")})
        plug_ang_vel = ObsTerm(func=base_mdp.root_ang_vel_w, params={"asset_cfg": SceneEntityCfg("plug")})
        plug_insertion = ObsTerm(func=plug_insertion_observation, params={})
        neighbor_plug_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("neighbor_plug")})
        hand_plug_force = ObsTerm(func=hand_plug_force, params={"sensor_names": HAND_SENSORS})
        """Left and right hand's total contact force on the task plug [N]; 0 without the sensors."""

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False
            self.concatenate_terms = False

    task_obs: TaskObsCfg = TaskObsCfg()
