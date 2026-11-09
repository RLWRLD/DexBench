# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""DB-H1 "Stable Stacking Multiple Plates": lift one plate stack onto another."""

from __future__ import annotations

import math
import warnings
from dataclasses import MISSING
from functools import partial
from typing import Any

import torch
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.predicates.spatial import object_is_above_height
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg

from dexbench_arena.contact_materials import HandContactMaterialCfg
from dexbench_arena.tasks.predicates.geometry import any_object_root_past_plane, object_center_over_region
from dexbench_arena.tasks.predicates.plates import plates_seated_in_stack
from dexbench_arena.tasks.predicates.rest import GRASP_HEIGHT_GAIN_M, object_lifted
from dexbench_arena.tasks.task_termination import task_termination_cfg

SURFACE_Z_TOLERANCE_M = 0.02


def plate_over_destination(
    env, object_cfg: SceneEntityCfg, stack_center_xy, region_radius_m: float, min_z_m: float, stack_center_offset=None
) -> torch.Tensor:
    """The plate's centre over the square of ``region_radius_m`` around the destination axis, and not below
    ``min_z_m``: :func:`object_center_over_region` with the square moved per env by the destination stack's
    draw (``stack_center_offset``), the way ``plates_seated_in_stack`` moves its region."""
    center = torch.as_tensor(stack_center_xy, dtype=torch.float32, device=env.device).expand(env.num_envs, 2)
    if stack_center_offset is not None:
        center = center + stack_center_offset.xy(env.num_envs, center.device, center.dtype)
    bounds = torch.stack((center - region_radius_m, center + region_radius_m), dim=1)
    return object_center_over_region(env, object_cfg, (0.0, 0.0, 0.0), bounds, min_z_m)


@configclass
class StackPlatesTerminationsCfg:
    """Success and failure terms; the episode limit is the task's ``timeout_s``."""

    success: TerminationTermCfg = MISSING
    plate_dropped: TerminationTermCfg = MISSING


class DexBenchStackPlatesTask(TaskBase):
    """Seat every plate into one stack on the destination surface."""

    def __init__(
        self,
        plate_names: list[str],
        stack_center_xy: tuple[float, float],
        surface_top_z_m: float,
        region_radius_m: float,
        expected_count: int,
        max_rim_offset_mm: float,
        max_lean_deg: float,
        nest_pitch_m: float,
        plate_dropped_min_z_m: float,
        max_seat_height_error_mm: float,
        velocity_threshold_m_s: float = 0.10,
        hands_clear_radius_m: float | None = None,
        hands_clear_height_m: float = 0.03,
        robot_asset_name: str = "robot",
        source_plate_names: list[str] | None = None,
        episode_length_s: float | None = 60.0,
        task_description: str | None = None,
        robot_joint_pos_overrides: dict[str, float] | None = None,
        hand_contact_material: HandContactMaterialCfg | None = None,
        settle_window_s: float = 0.25,
        rim_offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
        stack_center_offset=None,
    ):
        super().__init__(episode_length_s=episode_length_s, task_description=task_description)
        self._source_plate_names = list(source_plate_names) if source_plate_names else None
        self._stack_center_xy = stack_center_xy
        self._region_radius_m = region_radius_m
        self._surface_top_z_m = surface_top_z_m
        if len(rim_offset_xyz) != 3 or not all(math.isfinite(x) for x in rim_offset_xyz):
            raise ValueError("Rim offset must contain three finite coordinates")

        self.hand_contact_material = hand_contact_material

        self.robot_joint_pos_overrides = dict(robot_joint_pos_overrides) if robot_joint_pos_overrides else None

        self.stack_center_offset = stack_center_offset
        """What moves the success region with the destination stack (``PlanarOffsetListener``), or None;
        the environment subscribes it to the stack's pose variation."""
        self.termination_cfg = StackPlatesTerminationsCfg(
            success=TerminationTermCfg(
                func=plates_seated_in_stack,
                params={
                    "plate_names": list(plate_names),
                    "stack_center_xy": stack_center_xy,
                    "surface_top_z_m": surface_top_z_m,
                    "region_radius_m": region_radius_m,
                    "expected_count": int(expected_count),
                    "max_rim_offset_m": max_rim_offset_mm / 1000.0,
                    "rim_offset_xyz": tuple(rim_offset_xyz),
                    "min_up_alignment": math.cos(math.radians(max_lean_deg)),
                    "nest_pitch_m": float(nest_pitch_m),
                    "max_seat_height_error_m": max_seat_height_error_mm / 1000.0,
                    "velocity_threshold_m_s": velocity_threshold_m_s,
                    "settle_window_s": settle_window_s,
                    "stack_center_offset": stack_center_offset,
                    "robot_asset_name": robot_asset_name if hands_clear_radius_m else None,
                    "clear_radius_m": float(hands_clear_radius_m or 0.0),
                    "clear_height_m": float(hands_clear_height_m),
                },
            ),
            plate_dropped=TerminationTermCfg(
                func=any_object_root_past_plane,
                params={
                    "object_names": list(plate_names),
                    "plane_y_m": plate_dropped_min_z_m,
                    "plane_axis": 2,
                },
            ),
        )

    def validate_replay_dataset(self, path, metadata, embodiment):
        """Warn before replay labels use a different root/rim alignment feature."""
        source_cfg = (
            metadata.get("dexbench_rerecord", {}).get("runtime", {}).get("environment_cfg")
            or metadata.get("environment_cfg")
            or metadata.get("dexbench_recording", {}).get("environment_cfg", {})
        )
        source = source_cfg.get("stack_alignment_point")
        if source is None:
            params = metadata.get("dexbench_success_evaluation", {}).get("parameters", {})
            offset = params.get("rim_offset_xyz")
            if offset is not None:
                source = "rim" if any(offset) else "root"
        current = "rim" if any(self.termination_cfg.success.params["rim_offset_xyz"]) else "root"
        if source is not None and source != current:
            warnings.warn(
                f"H1 replay alignment differs for {path}: source={source}, replay={current}. "
                f"Labels may change. Use --stack_alignment_point {source} to retain the source metric.",
                RuntimeWarning,
                stacklevel=2,
            )

    def get_scene_cfg(self) -> Any:
        return None

    def get_termination_cfg(self) -> TaskTerminationCfg:
        return task_termination_cfg(
            self.episode_length_s,
            self.termination_cfg,
            partial_credit=self._partial_credit_chain(),
            partial_credit_name="stack_plates",
            partial_credit_description=(
                "Grasp the table stack, lift it, carry it over the cart stack, seat it there and let go."
            ),
        )

    def _partial_credit_chain(self) -> list | None:
        """Partial credit: grasped -> lifted -> over the destination stack -> stacked."""
        if self._source_plate_names is None:
            return None
        plate = SceneEntityCfg(self._source_plate_names[0])
        floor = self._surface_top_z_m - SURFACE_Z_TOLERANCE_M
        return [
            partial(object_lifted, object_cfg=plate, min_height_gain_m=GRASP_HEIGHT_GAIN_M),
            partial(object_is_above_height, object_name=plate.name, surface_height=floor, distance=0.0),
            partial(
                plate_over_destination,
                object_cfg=plate,
                stack_center_xy=self._stack_center_xy,
                region_radius_m=self._region_radius_m,
                min_z_m=floor,
                stack_center_offset=self.stack_center_offset,
            ),
            self.termination_cfg.success,
        ]

    def get_events_cfg(self) -> Any:
        return None

    def get_mimic_env_cfg(self, arm_mode: Any) -> Any:
        return None

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]
