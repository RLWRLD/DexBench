# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""DB-R2 dense-clutter selective retrieval task (retail E1 refrigerator station)."""

from __future__ import annotations

from dataclasses import MISSING
from functools import partial
from typing import Any

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg

from dexbench_arena.contact_materials import HandContactMaterialCfg
from dexbench_arena.tasks.predicates.e1_retail import db_r2_target_placed, db_r2_wrong_pick
from dexbench_arena.tasks.predicates.geometry import (
    any_object_toppled,
    object_center_outside_xy_bounds,
    object_center_past_y_plane,
    object_center_within_xy_bounds,
)
from dexbench_arena.tasks.predicates.rest import object_lifted
from dexbench_arena.tasks.task_termination import task_termination_cfg


@configclass
class RetrieveFromClutterTerminationsCfg:
    """Success and failure terms for the retrieve-from-clutter MDP; the episode limit is the task's ``timeout_s``."""

    success: TerminationTermCfg = MISSING

    target_dropped: TerminationTermCfg = MISSING

    target_outside_workspace: TerminationTermCfg = MISSING

    wrong_pick: TerminationTermCfg = MISSING

    neighbor_toppled: TerminationTermCfg = MISSING


class DexBenchRetrieveFromClutterTask(TaskBase):
    """Succeed when the instructed drink rests inside the basket with its neighbors undisturbed."""

    def __init__(
        self,
        object_name: str,
        object_center_offset_xyz: tuple[float, float, float],
        opening_plane_y_m: float,
        extraction_clearance_m: float,
        basket_center_xy: tuple[float, float],
        basket_center_halfsize_xy: tuple[float, float],
        basket_floor_z_m: float,
        center_height_range_m: tuple[float, float],
        neighbor_names: list[str],
        max_neighbor_displacement_m: float,
        workspace_xy_bounds: tuple[tuple[float, float], tuple[float, float]],
        object_dropped_min_z_m: float,
        wrong_pick_lift_m: float = 0.03,
        neighbor_topple_max_up_alignment: float = 0.70,
        velocity_threshold_m_s: float = 0.10,
        progress_lift_height_gain_m: float = 0.02,
        progress_over_basket_xy_margin_m: float = 0.05,
        episode_length_s: float | None = 30.0,
        task_description: str | None = None,
        robot_joint_pos_overrides: dict[str, float] | None = None,
        success_override: TerminationTermCfg | None = None,
        opening_plane_axis: int = 1,
        opening_toward_negative: bool = True,
        neighbor_topple_is_failure: bool = True,
        ang_vel_threshold_rad_s: float | None = None,
        rest_gates: tuple[float, float] | None = None,
        hand_contact_material: HandContactMaterialCfg | None = None,
        opening_ref_name: str | None = None,
        destination_ref_name: str | None = None,
    ):
        """``opening_plane_axis`` / ``opening_toward_negative`` place the station's opening plane:
        the FC fridge is a y plane crossed toward -y (the defaults), the produce display an x
        plane crossed toward -x. ``neighbor_topple_is_failure`` is off for round produce, which
        rolls without that being a failure of the pick. ``opening_ref_name`` and ``destination_ref_name``
        are the task frames Mimic generation replays the extraction and the transport relative to
        (retrieve_from_clutter_mimic); success and evaluation do not read them."""
        super().__init__(
            episode_length_s=episode_length_s,
            task_description=task_description
            or f"Take the {object_name} out of the refrigerator and put it in the basket.",
        )
        self.object_name = object_name
        self.neighbor_names = list(neighbor_names)
        self._object_center_offset_xyz = object_center_offset_xyz
        self._opening_plane_y_m = opening_plane_y_m
        self._opening_plane_axis = int(opening_plane_axis)
        self._opening_toward_negative = bool(opening_toward_negative)
        self._extraction_clearance_m = extraction_clearance_m
        self._basket_center_xy = basket_center_xy
        self._basket_center_halfsize_xy = basket_center_halfsize_xy
        self._basket_floor_z_m = basket_floor_z_m
        self._center_height_range_m = center_height_range_m
        self._max_neighbor_displacement_m = max_neighbor_displacement_m
        self._velocity_threshold_m_s = velocity_threshold_m_s
        self._rest_kwargs = {k: v for k, v in (("ang_vel_threshold_rad_s", ang_vel_threshold_rad_s), ("rest_gates", rest_gates)) if v is not None}
        self._rest_gates = rest_gates
        self._progress_lift_height_gain_m = progress_lift_height_gain_m
        self._progress_over_basket_xy_margin_m = progress_over_basket_xy_margin_m
        self.robot_joint_pos_overrides = dict(robot_joint_pos_overrides) if robot_joint_pos_overrides else None
        self.hand_contact_material = hand_contact_material
        self.opening_ref_name = opening_ref_name
        self.destination_ref_name = destination_ref_name

        object_cfg = SceneEntityCfg(object_name)
        self.termination_cfg = RetrieveFromClutterTerminationsCfg(
            success=TerminationTermCfg(
                func=db_r2_target_placed,
                params={
                    "object_cfg": object_cfg,
                    "object_center_offset_xyz": object_center_offset_xyz,
                    "basket_center_xy": basket_center_xy,
                    "basket_center_halfsize_xy": basket_center_halfsize_xy,
                    "basket_floor_z_m": basket_floor_z_m,
                    "center_height_range_m": center_height_range_m,
                    "neighbor_names": self.neighbor_names,
                    "max_neighbor_displacement_m": max_neighbor_displacement_m,
                    "velocity_threshold_m_s": velocity_threshold_m_s,
                    **self._rest_kwargs,
                },
            ),
            target_dropped=TerminationTermCfg(
                func=mdp_isaac_lab.root_height_below_minimum,
                params={"minimum_height": object_dropped_min_z_m, "asset_cfg": object_cfg},
            ),
            target_outside_workspace=TerminationTermCfg(
                func=object_center_outside_xy_bounds,
                params={
                    "object_cfg": object_cfg,
                    "object_center_offset_xyz": object_center_offset_xyz,
                    "xy_bounds": workspace_xy_bounds,
                },
            ),
            wrong_pick=TerminationTermCfg(
                func=db_r2_wrong_pick,
                params={
                    "neighbor_names": self.neighbor_names,
                    "plane_y_m": opening_plane_y_m,
                    "min_lift_m": wrong_pick_lift_m,
                    "toward_negative_y": self._opening_toward_negative,
                    "plane_axis": self._opening_plane_axis,
                    **({"rest_gates": rest_gates} if rest_gates is not None else {}),
                },
            ),
            neighbor_toppled=TerminationTermCfg(
                func=any_object_toppled,
                params={
                    "object_names": self.neighbor_names if neighbor_topple_is_failure else [],
                    "max_up_alignment": neighbor_topple_max_up_alignment,
                },
            ),
        )
        if success_override is not None:
            self.termination_cfg.success = success_override

    def get_scene_cfg(self) -> Any:
        return None

    def get_termination_cfg(self) -> TaskTerminationCfg:
        return task_termination_cfg(
            self.episode_length_s,
            self.termination_cfg,
            partial_credit=self._partial_credit_chain(),
            partial_credit_name="retrieve_from_clutter",
            partial_credit_description=(
                "Grasp the instructed drink, extract it through the refrigerator opening, "
                "carry it over the basket, set it down inside."
            ),
        )

    def get_events_cfg(self) -> Any:
        return None

    def get_mimic_env_cfg(self, arm_mode: Any) -> Any:
        return None

    def _partial_credit_chain(self) -> list:
        """Partial credit: grasped -> extracted -> over the basket -> placed."""
        import torch

        object_cfg = SceneEntityCfg(self.object_name)
        center = torch.as_tensor(self._basket_center_xy, dtype=torch.float64)
        half = torch.as_tensor(self._basket_center_halfsize_xy, dtype=torch.float64)
        margin = self._progress_over_basket_xy_margin_m
        over_basket_bounds = torch.stack((center - half - margin, center + half + margin), dim=-2)
        return [
            partial(
                object_lifted,
                object_cfg=object_cfg,
                min_height_gain_m=self._progress_lift_height_gain_m,
                rest_gates=self._rest_gates,
            ),
            partial(
                object_center_past_y_plane,
                object_cfg=object_cfg,
                object_center_offset_xyz=self._object_center_offset_xyz,
                plane_y_m=self._opening_plane_y_m,
                clearance_m=self._extraction_clearance_m,
                toward_negative_y=self._opening_toward_negative,
                plane_axis=self._opening_plane_axis,
            ),
            partial(
                object_center_within_xy_bounds,
                object_cfg=object_cfg,
                object_center_offset_xyz=self._object_center_offset_xyz,
                xy_bounds=over_basket_bounds,
            ),
            partial(
                db_r2_target_placed,
                object_cfg=object_cfg,
                object_center_offset_xyz=self._object_center_offset_xyz,
                basket_center_xy=self._basket_center_xy,
                basket_center_halfsize_xy=self._basket_center_halfsize_xy,
                basket_floor_z_m=self._basket_floor_z_m,
                center_height_range_m=self._center_height_range_m,
                neighbor_names=self.neighbor_names,
                max_neighbor_displacement_m=self._max_neighbor_displacement_m,
                velocity_threshold_m_s=self._velocity_threshold_m_s,
                **self._rest_kwargs,
            ),
        ]

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]
