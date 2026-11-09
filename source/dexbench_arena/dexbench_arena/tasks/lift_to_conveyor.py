# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Lift-a-parcel-onto-the-conveyor task for the DexBench E2 logistics centre."""

from __future__ import annotations

from dataclasses import MISSING
from functools import partial
from typing import Any

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.managers import EventTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg

from dexbench_arena.contact_materials import HandContactMaterialCfg
from dexbench_arena.tasks.cage_clear import package_clear_of_cage, reset_cage_clear

from dexbench_arena.tasks.conveyor_placement_config import (
    CONVEYOR_PLACE_ANGULAR_SPEED_RAD_S,
    CONVEYOR_PLACE_BOTTOM_HEIGHT_TOLERANCE_M,
    CONVEYOR_PLACE_HOLD_TIME_S,
    CONVEYOR_PLACE_LINEAR_SPEED_M_S,
    CONVEYOR_PLACE_MAX_FACE_TILT_DEG,
)

from dexbench_arena.tasks.conveyor_placement import (
    box_over_conveyor,
    conveyor_box_outside_workspace,
    object_settled_on_conveyor,
    reset_conveyor_placement,
)
from dexbench_arena.tasks.predicates.rest import GRASP_HEIGHT_GAIN_M, object_descending_faster_than, object_lifted
from dexbench_arena.tasks.task_termination import task_termination_cfg

LIFT_TO_CONVEYOR_INSTRUCTION = "Lift the package off the cage with both hands and place it on the conveyor."


@configclass
class LiftToConveyorTerminationsCfg:
    """Success and failure terms for the lift-to-conveyor MDP; the episode limit is the task's ``timeout_s``."""

    success: TerminationTermCfg = MISSING

    object_dropped: TerminationTermCfg = MISSING

    object_falling: TerminationTermCfg = MISSING

    object_outside_workspace: TerminationTermCfg = MISSING


@configclass
class LiftToConveyorEventsCfg:
    reset_placement_hold: EventTermCfg = MISSING
    reset_mimic_boundary: EventTermCfg = MISSING


class DexBenchLiftToConveyorTask(TaskBase):
    """Set the box down on the conveyor, resting flat on any face but a long side, and keep it there for 0.5 s."""

    def __init__(
        self,
        object_name: str,
        box_sizes_xyz: tuple[tuple[float, float, float], ...],
        conveyor_xy_rects: tuple[tuple[tuple[float, float], tuple[float, float]], ...],
        conveyor_top_z_range_m: tuple[float, float],
        workspace_xy_bounds: tuple[tuple[float, float], tuple[float, float]],
        object_dropped_min_z_m: float,
        max_descent_speed_m_s: float = 1.0,
        max_face_tilt_deg: float = CONVEYOR_PLACE_MAX_FACE_TILT_DEG,
        velocity_threshold_m_s: float = CONVEYOR_PLACE_LINEAR_SPEED_M_S,
        angular_velocity_threshold_rad_s: float = CONVEYOR_PLACE_ANGULAR_SPEED_RAD_S,
        bottom_height_tolerance_m: float = CONVEYOR_PLACE_BOTTOM_HEIGHT_TOLERANCE_M,
        hold_time_s: float = CONVEYOR_PLACE_HOLD_TIME_S,
        episode_length_s: float | None = 60.0,
        task_description: str | None = None,
        robot_joint_pos_overrides: dict[str, float] | None = None,
        hand_contact_material: HandContactMaterialCfg | None = None,
        log_placement_checks: bool = False,
    ):
        super().__init__(
            episode_length_s=episode_length_s,
            task_description=task_description
            or LIFT_TO_CONVEYOR_INSTRUCTION,
        )
        self.hand_contact_material = hand_contact_material
        self.object_name = object_name
        self._box_sizes_xyz = box_sizes_xyz
        self._conveyor_xy_rects = conveyor_xy_rects
        self._conveyor_floor_z_m = conveyor_top_z_range_m[0] - bottom_height_tolerance_m
        self.robot_joint_pos_overrides = dict(robot_joint_pos_overrides) if robot_joint_pos_overrides else None
        self.termination_cfg = LiftToConveyorTerminationsCfg(
            success=TerminationTermCfg(
                func=object_settled_on_conveyor,
                params={
                    "object_cfg": SceneEntityCfg(object_name),
                    "box_sizes_xyz": box_sizes_xyz,
                    "surface_xy_rects": conveyor_xy_rects,
                    "surface_top_z_range_m": conveyor_top_z_range_m,
                    "max_face_tilt_deg": max_face_tilt_deg,
                    "bottom_height_tolerance_m": bottom_height_tolerance_m,
                    "angular_velocity_threshold_rad_s": angular_velocity_threshold_rad_s,
                    "hold_time_s": hold_time_s,
                    "velocity_threshold_m_s": velocity_threshold_m_s,
                    **({"log_checks": True} if log_placement_checks else {}),
                },
            ),
            object_dropped=TerminationTermCfg(
                func=mdp_isaac_lab.root_height_below_minimum,
                params={
                    "minimum_height": object_dropped_min_z_m,
                    "asset_cfg": SceneEntityCfg(object_name),
                },
            ),
            object_falling=TerminationTermCfg(
                func=object_descending_faster_than,
                params={
                    "object_cfg": SceneEntityCfg(object_name),
                    "max_descent_speed_m_s": max_descent_speed_m_s,
                },
            ),
            object_outside_workspace=TerminationTermCfg(
                func=conveyor_box_outside_workspace,
                params={
                    "object_cfg": SceneEntityCfg(object_name),
                    "box_sizes_xyz": box_sizes_xyz,
                    "xy_bounds": workspace_xy_bounds,
                },
            ),
        )

        self.events_cfg = LiftToConveyorEventsCfg(
            reset_placement_hold=EventTermCfg(
                func=reset_conveyor_placement, mode="reset", params={"object_name": object_name}
            ),
            reset_mimic_boundary=EventTermCfg(
                func=reset_cage_clear,
                mode="reset",
                params={
                    "box_sizes_xyz": box_sizes_xyz,
                    "success_params": dict(self.termination_cfg.success.params),
                },
            ),
        )

    def get_scene_cfg(self) -> Any:
        return None

    def get_termination_cfg(self) -> TaskTerminationCfg:
        return task_termination_cfg(
            self.episode_length_s,
            self.termination_cfg,
            partial_credit=self._partial_credit_chain(),
            partial_credit_name="lift_to_conveyor",
            partial_credit_description=(
                "Grasp the parcel, lift it out of the cage, carry it over the conveyor, set it down settled."
            ),
        )

    def _partial_credit_chain(self) -> list:
        """Partial credit: lifted -> out of the cage -> over the conveyor -> placed and settled."""
        object_cfg = SceneEntityCfg(self.object_name)
        return [
            partial(object_lifted, object_cfg=object_cfg, min_height_gain_m=GRASP_HEIGHT_GAIN_M),
            partial(package_clear_of_cage, object_name=self.object_name, box_sizes_xyz=self._box_sizes_xyz, min_z=None),
            partial(
                box_over_conveyor,
                object_cfg=object_cfg,
                box_sizes_xyz=self._box_sizes_xyz,
                surface_xy_rects=self._conveyor_xy_rects,
                min_z_m=self._conveyor_floor_z_m,
            ),
            self.termination_cfg.success,
        ]

    def get_events_cfg(self) -> Any:
        return self.events_cfg

    def get_mimic_env_cfg(self, arm_mode: Any) -> Any:
        return None

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]
