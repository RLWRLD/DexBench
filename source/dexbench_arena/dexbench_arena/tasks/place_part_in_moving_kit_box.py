# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""DB-M5: place a parts-bin part into the moving kit box, for the DexBench E3 M5 kitting cell."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import MISSING
from typing import Any

from isaaclab.managers import TerminationTermCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg
from isaaclab_arena.utils.configclass import make_configclass

from dexbench_arena.contact_materials import HandContactMaterialCfg
from dexbench_arena.tasks.kit_box_placement import part_placed_in_kit_box_compartment, kit_box_reached_belt_end
from dexbench_arena.tasks.predicates.release import HAND_RELEASED_MAX_FORCE_N
from dexbench_arena.tasks.recording_hooks import DexBenchRecordingTask
from dexbench_arena.tasks.task_termination import task_termination_cfg
from dexbench_arena.variations.kit_box_compartment_variation import empty_compartment_event, replay_empty_compartment

PLACE_PART_RELEASE_HOLD_S = 0.5
PLACE_PART_RELEASE_MAX_WEIGHT_FRACTION = 0.25


def hand_on_part_sensor_name(part_name: str) -> str:
    """The scene's contact sensor between the hands and ``part_name`` (``configure_for_embodiment``)."""

    return f"hands_on_{part_name}"


@configclass
class PlacePartInMovingKitBoxTerminationsCfg:
    """Success and failure terms; the episode limit is the task's ``episode_length_s``, a backstop only:
    the belt end is the real deadline."""

    success: TerminationTermCfg = MISSING

    kit_box_reached_belt_end: TerminationTermCfg = MISSING


class DexBenchPlacePartInMovingKitBoxTask(DexBenchRecordingTask):
    """Put any one parts-bin part into the kit box's empty compartment before the box leaves the belt."""

    def __init__(
        self,
        kit_box_name: str,
        part_names: tuple[str, ...],
        compartments_min_xy_local: tuple[tuple[float, float], ...],
        compartments_max_xy_local: tuple[tuple[float, float], ...],
        default_empty_compartment: int,
        compartment_floor_z_local: float,
        part_center_local: tuple[float, float, float],
        part_extents_local: tuple[float, float, float],
        part_end_points_local: tuple[tuple[float, float, float], ...],
        kit_box_leading_edge_local_x: float,
        belt_end_x_m: float,
        kit_box_half_xy: tuple[float, float],
        belt_y_range: tuple[float, float],
        belt_top_z_m: float,
        max_belt_gap_m: float,
        episode_length_s: float | None = 60.0,
        task_description: str | None = None,
        robot_joint_pos_overrides: dict[str, float] | None = None,
        hand_contact_material: HandContactMaterialCfg | None = None,
        part_label: str = "M24 bolt",
        max_hand_force_n: float = HAND_RELEASED_MAX_FORCE_N,
        max_hand_weight_fraction: float = PLACE_PART_RELEASE_MAX_WEIGHT_FRACTION,
        release_hold_s: float = PLACE_PART_RELEASE_HOLD_S,
    ):
        super().__init__(
            episode_length_s=episode_length_s,
            task_description=task_description
            or (
                f"Pick one {part_label} from the yellow parts bin and place it in the empty compartment "
                "of the kit box before the box reaches the end of the moving conveyor."
            ),
        )
        if not (math.isfinite(max_hand_force_n) and max_hand_force_n > 0.0):
            raise ValueError(f"max_hand_force_n={max_hand_force_n!r}: the release force must be a positive number of newtons")
        if not (math.isfinite(max_hand_weight_fraction) and max_hand_weight_fraction > 0.0):
            raise ValueError(
                f"max_hand_weight_fraction={max_hand_weight_fraction!r}: the share of the part's weight must be positive"
            )
        if not (math.isfinite(release_hold_s) and release_hold_s >= 0.0):
            raise ValueError(f"release_hold_s={release_hold_s!r}: the hold must be a nonnegative number of seconds")
        self.hand_contact_material = hand_contact_material
        self.robot_joint_pos_overrides = dict(robot_joint_pos_overrides) if robot_joint_pos_overrides else None
        self.part_names = tuple(part_names)
        self.kit_box_name = kit_box_name
        self.scene_cfg = None
        self.termination_cfg = PlacePartInMovingKitBoxTerminationsCfg(
            success=TerminationTermCfg(
                func=part_placed_in_kit_box_compartment,
                params={
                    "kit_box_name": kit_box_name,
                    "part_names": tuple(part_names),
                    "compartments_min_xy_local": tuple(map(tuple, compartments_min_xy_local)),
                    "compartments_max_xy_local": tuple(map(tuple, compartments_max_xy_local)),
                    "default_empty_compartment": int(default_empty_compartment),
                    "compartment_floor_z_local": compartment_floor_z_local,
                    "part_center_local": part_center_local,
                    "part_extents_local": part_extents_local,
                    "part_end_points_local": part_end_points_local,
                    "kit_box_half_xy": tuple(kit_box_half_xy),
                    "belt_y_range": tuple(belt_y_range),
                    "belt_top_z": belt_top_z_m,
                    "max_belt_gap_m": max_belt_gap_m,
                    "max_hand_force_n": max_hand_force_n,
                    "max_hand_weight_fraction": max_hand_weight_fraction,
                    "hold_time_s": release_hold_s,
                },
            ),
            kit_box_reached_belt_end=TerminationTermCfg(
                func=kit_box_reached_belt_end,
                params={
                    "kit_box_name": kit_box_name,
                    "leading_edge_local_x": kit_box_leading_edge_local_x,
                    "belt_end_x": belt_end_x_m,
                },
            ),
        )
        self.events_cfg = None

    def configure_for_embodiment(self, embodiment: Any) -> None:
        """Sense the hands on every candidate part, for the release the success term judges."""
        robot = getattr(getattr(embodiment, "scene_config", None), "robot", None)
        if robot is None:
            return
        exprs = getattr(embodiment, "held_object_contact_body_exprs", None)
        if not exprs or not {"left", "right"} <= set(exprs):
            raise ValueError(
                "place_part_in_kit_box judges whether the hands let go of the part, so the embodiment must "
                "declare held_object_contact_body_exprs for 'left' and 'right'; "
                f"{getattr(embodiment, 'name', type(embodiment).__name__)} declares {sorted(exprs or {})}"
            )
        hands = f"{robot.prim_path}/({exprs['left']}|{exprs['right']})"
        names = [hand_on_part_sensor_name(part) for part in self.part_names]
        scene_cfg_class = make_configclass(
            "PlacePartInMovingKitBoxSceneCfg",
            [(name, ContactSensorCfg | None, dataclasses.field(default=None)) for name in names],
        )
        self.scene_cfg = scene_cfg_class()
        for name, part in zip(names, self.part_names):
            setattr(
                self.scene_cfg,
                name,
                ContactSensorCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/{part}(/.*)?",
                    filter_prim_paths_expr=[hands],
                    update_period=0.0,
                ),
            )
        self.termination_cfg.success.params["hand_sensor_names"] = tuple(names)

    def prepare_replay_episode(self, env, demo_attributes: Mapping[str, Any]) -> None:
        """Before a replay resets env 0 to a demo's initial state (``rerecord_demos.py``,
        ``render_demos_from_states.py``): make that reset draw the empty compartment the demo was recorded with, its
        ``variations/<kit box>/empty_compartment`` row, which Arena's recorder wrote at its start. The restored state
        puts the loaded parts back, but the reset before it draws the compartment afresh, and the success term would
        judge the demo against that draw. Nothing for an env whose kit-box loading is not varied; ValueError for a
        demo of such a variation that does not record its draw."""

        if empty_compartment_event(env) is None:
            return
        key = f"variations/{self.kit_box_name}/empty_compartment"
        drawn = demo_attributes.get(key)
        if not drawn:
            raise ValueError(
                f"place_part_in_kit_box: the demo records no {key}, so a replay would judge it against another draw"
            )
        replay_empty_compartment(env, int(round(float(drawn[0]))))

    def get_scene_cfg(self) -> Any:
        return self.scene_cfg

    def get_termination_cfg(self) -> TaskTerminationCfg:
        """The terms as Arena's ``TaskTerminationCfg`` (``task_termination.task_termination_cfg``): success, the
        belt-end failure, and ``timeout_s`` from ``episode_length_s``; no partial-credit chain."""

        return task_termination_cfg(self.episode_length_s, self.termination_cfg)

    def get_events_cfg(self) -> Any:
        return self.events_cfg

    def get_mimic_env_cfg(self, arm_mode: Any) -> Any:
        return None

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]
