# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""DB-R3 spatially-constrained slot-placement task (retail E1 gondola)."""

from __future__ import annotations

import inspect
from dataclasses import MISSING
from functools import partial
from typing import Any

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg

from dexbench_arena.contact_materials import HandContactMaterialCfg
from dexbench_arena.tasks.task_termination import task_termination_cfg

from dexbench_arena.tasks.predicates.e1_retail import db_r3_neighbor_toppled_or_fallen, db_r3_placed
from dexbench_arena.tasks.predicates.geometry import object_center_outside_xy_bounds, object_center_within_xy_bounds
from dexbench_arena.tasks.predicates.placement import object_standing_in_slot
from dexbench_arena.tasks.predicates.release import HAND_BACK_OUT_M, HAND_RELEASED_MAX_FORCE_N
from dexbench_arena.tasks.predicates.rest import object_lifted

PROGRESS_LIFT_HEIGHT_GAIN_M = 0.04
PLACE_IN_SLOT_EPISODE_LENGTH_S = 60.0
PROGRESS_OVER_SLOT_XY_MARGIN_M = 0.05
PLACE_IN_SLOT_COLLECTION_HOLD_S = 1.0
HAND_ON_OBJECT_SENSOR_NAME = "hand_on_object_contact"


@configclass
class PlaceInSlotSceneCfg:
    """The task's sensors: the placing hand's contact with the carton."""

    hand_on_object_contact: ContactSensorCfg | None = None


@configclass
class PlaceInSlotTerminationsCfg:
    """Success and failure terms for the place-in-slot MDP; the episode limit is the task's ``timeout_s``."""

    success: TerminationTermCfg = MISSING

    object_dropped: TerminationTermCfg = MISSING

    object_outside_workspace: TerminationTermCfg = MISSING

    neighbor_toppled: TerminationTermCfg | None = None


class DexBenchPlaceInSlotTask(TaskBase):
    """Succeed when the box stands upright at rest in the empty shelf facing and the hand has let go of it."""

    def __init__(
        self,
        object_name: str,
        object_center_offset_xyz: tuple[float, float, float],
        slot_center_xy: tuple[float, float],
        slot_center_halfsize_xy: tuple[float, float],
        object_half_extents_xy: tuple[float, float],
        shelf_top_z_m: float,
        center_height_range_m: tuple[float, float],
        workspace_xy_bounds: tuple[tuple[float, float], tuple[float, float]],
        object_dropped_min_z_m: float,
        front_plane_xy: tuple[float, float],
        outward_xy: tuple[float, float],
        neighbor_names: list[str] | None = None,
        neighbor_reference_m: float | None = None,
        object_local_up_xyz: tuple[float, float, float] = (0.0, 0.0, 1.0),
        min_up_alignment: float = 0.90,
        velocity_threshold_m_s: float = 0.10,
        episode_length_s: float | None = PLACE_IN_SLOT_EPISODE_LENGTH_S,
        task_description: str | None = None,
        robot_joint_pos_overrides: dict[str, float] | None = None,
        success_override: TerminationTermCfg | None = None,
        hand_contact_material: HandContactMaterialCfg | None = None,
        hand: ArmMode = ArmMode.RIGHT,
        max_hand_force_n: float = HAND_RELEASED_MAX_FORCE_N,
        min_back_out_m: float = HAND_BACK_OUT_M,
        collection_hold_s: float = PLACE_IN_SLOT_COLLECTION_HOLD_S,
        neighbor_knock: dict | None = None,
        slot_ref_name: str | None = None,
    ):
        super().__init__(
            episode_length_s=episode_length_s,
            task_description=task_description or f"Place the {object_name} into the empty slot on the shelf.",
        )
        self.hand_contact_material = hand_contact_material
        self.object_name = object_name
        self.slot_ref_name = slot_ref_name
        self._object_center_offset_xyz = object_center_offset_xyz
        self._slot_center_xy = slot_center_xy
        self._slot_center_halfsize_xy = slot_center_halfsize_xy
        self._object_half_extents_xy = object_half_extents_xy
        self._shelf_top_z_m = shelf_top_z_m
        self._center_height_range_m = center_height_range_m
        self._object_local_up_xyz = object_local_up_xyz
        self._min_up_alignment = min_up_alignment
        self._velocity_threshold_m_s = velocity_threshold_m_s
        self._hand = hand
        self._release = {
            "front_plane_xy": front_plane_xy,
            "outward_xy": outward_xy,
            "max_hand_force_n": max_hand_force_n,
            "min_back_out_m": min_back_out_m,
        }
        self.scene_cfg = PlaceInSlotSceneCfg()
        self.robot_joint_pos_overrides = dict(robot_joint_pos_overrides) if robot_joint_pos_overrides else None
        self.termination_cfg = PlaceInSlotTerminationsCfg(
            success=TerminationTermCfg(
                func=db_r3_placed,
                params={
                    "object_cfg": SceneEntityCfg(object_name),
                    "object_center_offset_xyz": object_center_offset_xyz,
                    "slot_center_xy": slot_center_xy,
                    "slot_center_halfsize_xy": slot_center_halfsize_xy,
                    "shelf_top_z_m": shelf_top_z_m,
                    "center_height_range_m": center_height_range_m,
                    "object_local_up_xyz": object_local_up_xyz,
                    "min_up_alignment": min_up_alignment,
                    "velocity_threshold_m_s": velocity_threshold_m_s,
                    "neighbor_names": list(neighbor_names or ()),
                    "neighbor_reference_m": neighbor_reference_m,
                    "neighbor_knock": dict(neighbor_knock) if neighbor_knock else None,
                    **self._release,
                    "hold_time_s": collection_hold_s,
                },
            ),
            object_dropped=TerminationTermCfg(
                func=mdp_isaac_lab.root_height_below_minimum,
                params={
                    "minimum_height": object_dropped_min_z_m,
                    "asset_cfg": SceneEntityCfg(object_name),
                },
            ),
            object_outside_workspace=TerminationTermCfg(
                func=object_center_outside_xy_bounds,
                params={
                    "object_cfg": SceneEntityCfg(object_name),
                    "object_center_offset_xyz": object_center_offset_xyz,
                    "xy_bounds": workspace_xy_bounds,
                },
            ),
        )
        if neighbor_knock and neighbor_names:
            self.termination_cfg.neighbor_toppled = TerminationTermCfg(
                func=db_r3_neighbor_toppled_or_fallen,
                params={
                    "neighbor_names": list(neighbor_names),
                    **{key: neighbor_knock[key] for key in ("neighbor_local_up_xyz", "max_up_alignment", "max_drop_m")},
                },
            )
        if success_override is not None:
            self.termination_cfg.success = success_override
            success_override.params.update({key: value for key, value in self._release.items()
                                            if key not in success_override.params})

    def configure_for_embodiment(self, embodiment: Any) -> None:
        """Sense the placing hand on the box, and name its wrist, for the release the success terms judge."""
        robot = getattr(getattr(embodiment, "scene_config", None), "robot", None)
        if robot is None:
            return
        exprs = getattr(embodiment, "held_object_contact_body_exprs", None)
        side = self._hand.value
        if not exprs or side not in exprs:
            raise ValueError(
                f"place_box_in_slot judges the {side} hand's release, so the embodiment must declare "
                f"held_object_contact_body_exprs[{side!r}]; "
                f"{getattr(embodiment, 'name', type(embodiment).__name__)} declares {sorted(exprs or {})}"
            )
        self.scene_cfg.hand_on_object_contact = ContactSensorCfg(
            prim_path=f"{robot.prim_path}/({exprs[side]})",
            filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/{self.object_name}(/.*)?"],
            update_period=0.0,
        )
        self._release.update(
            hand_sensor_name=HAND_ON_OBJECT_SENSOR_NAME,
            wrist_body_name=embodiment.get_ee_frame_name(self._hand),
        )
        self.termination_cfg.success.params.update(
            hand_sensor_name=self._release["hand_sensor_name"], wrist_body_name=self._release["wrist_body_name"]
        )

    def slot_region(self) -> dict:
        """Where the object's footprint must end up for the active success term, for an operator marker."""
        params = self.termination_cfg.success.params
        if "slot_center_xy" not in params:
            raise ValueError(f"success term {self.termination_cfg.success.func!r} checks no slot")
        seated_keys = inspect.signature(object_standing_in_slot).parameters
        return {
            "center_xy": params["slot_center_xy"],
            "center_halfsize_xy": params["slot_center_halfsize_xy"],
            "object_half_extents_xy": self._object_half_extents_xy,
            "top_z_m": params["shelf_top_z_m"],
            "seated": {key: value for key, value in params.items() if key in seated_keys},
        }

    def get_scene_cfg(self) -> Any:
        return self.scene_cfg

    def get_termination_cfg(self) -> TaskTerminationCfg:
        return task_termination_cfg(
            self.episode_length_s,
            self.termination_cfg,
            partial_credit=self._partial_credit_chain(),
            partial_credit_name="place_in_slot",
            partial_credit_description=(
                "Grasp the supply item, carry it over the empty facing, seat it upright, let go and back out."
            ),
        )

    def get_events_cfg(self) -> Any:
        return None

    def get_mimic_env_cfg(self, arm_mode: Any) -> Any:
        return None

    def _partial_credit_chain(self) -> list:
        """Partial credit as one sequential chain: lift -> over the slot -> seated -> released."""
        import torch

        object_cfg = SceneEntityCfg(self.object_name)
        center = torch.as_tensor(self._slot_center_xy, dtype=torch.float64)
        half = torch.as_tensor(self._slot_center_halfsize_xy, dtype=torch.float64)
        over_slot_bounds = torch.stack(
            (center - half - PROGRESS_OVER_SLOT_XY_MARGIN_M, center + half + PROGRESS_OVER_SLOT_XY_MARGIN_M),
            dim=-2,
        )
        return [
            partial(
                object_lifted,
                object_cfg=object_cfg,
                min_height_gain_m=PROGRESS_LIFT_HEIGHT_GAIN_M,
            ),
            partial(
                object_center_within_xy_bounds,
                object_cfg=object_cfg,
                object_center_offset_xyz=self._object_center_offset_xyz,
                xy_bounds=over_slot_bounds,
            ),
            partial(
                object_standing_in_slot,
                object_cfg=object_cfg,
                object_center_offset_xyz=self._object_center_offset_xyz,
                slot_center_xy=self._slot_center_xy,
                slot_center_halfsize_xy=self._slot_center_halfsize_xy,
                shelf_top_z_m=self._shelf_top_z_m,
                center_height_range_m=self._center_height_range_m,
                object_local_up_xyz=self._object_local_up_xyz,
                min_up_alignment=self._min_up_alignment,
                velocity_threshold_m_s=self._velocity_threshold_m_s,
            ),
            partial(
                db_r3_placed,
                object_cfg=object_cfg,
                object_center_offset_xyz=self._object_center_offset_xyz,
                slot_center_xy=self._slot_center_xy,
                slot_center_halfsize_xy=self._slot_center_halfsize_xy,
                shelf_top_z_m=self._shelf_top_z_m,
                center_height_range_m=self._center_height_range_m,
                object_local_up_xyz=self._object_local_up_xyz,
                min_up_alignment=self._min_up_alignment,
                velocity_threshold_m_s=self._velocity_threshold_m_s,
                hold_key="db_r3_partial_credit",
                publish_status=False,
                **self._release,
            ),
        ]

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]
