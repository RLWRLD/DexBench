# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""E3: the PSL assembly line, staged for DB-M1, DB-M3, DB-M4 or DB-M5, on Newton."""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentFactory

from dexbench_arena.layouts import db_m1_layout, db_m3_layout, db_m4_layout
from dexbench_arena.layouts import db_m5_layout
from dexbench_arena.environments.manipulation import e3_manufacturing_db_m5_environment as m5
from dexbench_arena.layouts.db_m5_layout import M5_PART_AXIS
from dexbench_arena.environments.manipulation.e3_manufacturing_db_m1_environment import (
    INSERT_DRAWER_TASK_NAME,
    PICK_INSERT_DRAWER_TASK_NAME,
    m1_start_pose,
    m1_base_anchors,
    m1_facing_targets,
    M1_ANCHOR,
    M1_NEWTON_SOLVER_OPTIONS,
    M1StationCfg,
    add_m1_cli_args,
    apply_m1_newton_robot_settings,
    build_m1_scene,
    make_insert_drawer_task,
)
from dexbench_arena.environments.manipulation.e3_manufacturing_db_m4_environment import (
    HAND_START_LEG_TASK_NAME,
    M4_ANCHOR,
    M4_NEWTON_SOLVER_OPTIONS,
    M4StationCfg,
    add_m4_cli_args,
    build_m4_scene,
    make_hand_start_leg_task,
)
from dexbench_arena.environments.manipulation.e3_manufacturing_db_m3_environment import (
    POWER_DRIVER_TASK_NAME, M3_ANCHOR, M3StationCfg, M3_NEWTON_SOLVER_OPTIONS,
    add_m3_cli_args, apply_m3_newton_settings, build_m3_scene,
    configure_m3_environment, make_power_driver_task, validate_m3_cfg,
)
from dexbench_arena.environments.manipulation.scene_environment_base import (
    DexBenchEnvironmentCfg,
    DexBenchSceneEnvironmentBase,
)

if TYPE_CHECKING:
    from isaaclab_arena.scene.scene import Scene

TASK_ANCHORS = {PICK_INSERT_DRAWER_TASK_NAME: M1_ANCHOR, POWER_DRIVER_TASK_NAME: M3_ANCHOR, INSERT_DRAWER_TASK_NAME: M1_ANCHOR, HAND_START_LEG_TASK_NAME: M4_ANCHOR,
                m5.PLACE_PART_IN_KIT_BOX_TASK_NAME: db_m5_layout.M5_ANCHOR}
"""The anchor, and so the staging, each task runs at."""
SHELL_ANCHOR = M1_ANCHOR
"""The staging of a build that names neither a task nor an anchor."""
STATION_CFGS = {M3_ANCHOR: M3StationCfg, M1_ANCHOR: M1StationCfg, M4_ANCHOR: M4StationCfg, db_m5_layout.M5_ANCHOR: m5.M5StationCfg}
"""Each staging's settings; the E3 config takes all four, a per-task id's config its own."""
STATION_CLI_ARGS = {M3_ANCHOR: add_m3_cli_args, M1_ANCHOR: add_m1_cli_args, M4_ANCHOR: add_m4_cli_args,
                    db_m5_layout.M5_ANCHOR: m5.add_m5_cli_args}


def staging_anchor(cfg) -> str:
    """The anchor whose staging a build from ``cfg`` stands at: the task's, else ``cfg.anchor``, else ``SHELL_ANCHOR``."""
    return TASK_ANCHORS.get(getattr(cfg, "task", None)) or cfg.anchor or SHELL_ANCHOR


@dataclass
class DexBenchManufacturingE3EnvironmentCfg(M1StationCfg, M3StationCfg, M4StationCfg, m5.M5StationCfg,
                                             DexBenchEnvironmentCfg):
    """Typed config for the E3 assembly line: its flags, names and defaults, DB-M1's, DB-M3's, DB-M4's and DB-M5's."""

    task: str = "none"
    """``insert_drawer``, ``power_driver_fastening``, ``hand_start_leg``, ``place_part_in_kit_box``, or ``none`` for
    the scene shell."""

    def __post_init__(self):
        anchor = TASK_ANCHORS.get(self.task)
        if anchor is not None and self.anchor not in (None, anchor):
            raise ValueError(f"Task {self.task} runs at anchor {anchor}, not {self.anchor}")
        M1StationCfg.__post_init__(self)
        M3StationCfg.__post_init__(self)
        M4StationCfg.__post_init__(self)
        staging = staging_anchor(self)
        for other, station in STATION_CFGS.items():
            for field in dataclasses.fields(station):
                value = getattr(self, field.name)
                if isinstance(field.default, tuple) and isinstance(value, list):
                    value = tuple(value)
                if other != staging and field.name != "task" and value != field.default:
                    raise ValueError(f"{field.name} is read by the {other} staging only; this build stands at {staging}")


@register_environment
class DexBenchManufacturingE3Environment(
    DexBenchSceneEnvironmentBase,
    ArenaEnvironmentFactory[DexBenchManufacturingE3EnvironmentCfg],
):
    """E3: the PSL assembly line, staged for the task it runs (DB-M1, DB-M3, DB-M4 or DB-M5)."""

    name = "dexbench_manufacturing_e3_environment"
    _legacy_argparse_cfg_type = DexBenchManufacturingE3EnvironmentCfg
    robot_base_anchors_xyz = {**db_m1_layout.ROBOT_BASE_ANCHORS_XYZ, **db_m4_layout.ROBOT_BASE_ANCHORS_XYZ, **db_m3_layout.ROBOT_BASE_ANCHORS_XYZ,
                              **db_m5_layout.ROBOT_BASE_ANCHORS_XYZ}
    anchor_facing_targets_xy = {**db_m1_layout.ANCHOR_FACING_TARGETS_XY, **db_m4_layout.ANCHOR_FACING_TARGETS_XY, **db_m3_layout.ANCHOR_FACING_TARGETS_XY,
                                **db_m5_layout.ANCHOR_FACING_TARGETS_XY}
    anchor_help = (
        "Robot base anchor, and with it the staging: A_m1 (DB-M1, the washer slot) A_m3 (DB-M3, the refrigerator), A_m4 (DB-M4, facing the "
        "cabinet on the workbench) or A_m5 (DB-M5, the kitting station: parts bin <-> kit box). Unset takes the task's; the scene shell's is A_m1."
    )
    embodiment_anchor_overrides = m5.M5_EMBODIMENT_ANCHOR_OVERRIDES
    available_tasks = {
        POWER_DRIVER_TASK_NAME: make_power_driver_task,
        INSERT_DRAWER_TASK_NAME: make_insert_drawer_task,
        PICK_INSERT_DRAWER_TASK_NAME: make_insert_drawer_task,
        HAND_START_LEG_TASK_NAME: make_hand_start_leg_task,
        m5.PLACE_PART_IN_KIT_BOX_TASK_NAME: m5.make_place_part_in_kit_box_task,
    }
    object_axes = {
        POWER_DRIVER_TASK_NAME: db_m3_layout.M3_E3_OBJECT_AXIS,
        INSERT_DRAWER_TASK_NAME: db_m1_layout.M1_DRAWER_OBJECT_AXIS,
        PICK_INSERT_DRAWER_TASK_NAME: db_m1_layout.M1_TABLETOP_DRAWER_OBJECT_AXIS,
        HAND_START_LEG_TASK_NAME: db_m4_layout.M4_LEG_OBJECT_AXIS,
        m5.PLACE_PART_IN_KIT_BOX_TASK_NAME: M5_PART_AXIS,
    }
    shared_settings = ("task",)
    staging_newton_solver_options = {M3_ANCHOR: M3_NEWTON_SOLVER_OPTIONS, M1_ANCHOR: M1_NEWTON_SOLVER_OPTIONS, M4_ANCHOR: M4_NEWTON_SOLVER_OPTIONS,
                                     db_m5_layout.M5_ANCHOR: m5.M5_NEWTON_SOLVER_OPTIONS}
    staging_scenes = {M3_ANCHOR: build_m3_scene, M1_ANCHOR: build_m1_scene, M4_ANCHOR: build_m4_scene, db_m5_layout.M5_ANCHOR: m5.build_m5_scene}

    def build(self, cfg: DexBenchManufacturingE3EnvironmentCfg):
        """Build with ``anchor`` set to the staging's, so the robot stands there and the recorded config names it."""
        anchor = staging_anchor(cfg)
        if anchor not in type(self).robot_base_anchors_xyz:
            raise ValueError(f"{self.name} stages nothing at anchor {anchor!r} (anchors: {tuple(type(self).robot_base_anchors_xyz)})")
        if cfg.anchor is None:
            cfg = dataclasses.replace(cfg, anchor=anchor)
        if anchor == M3_ANCHOR:
            validate_m3_cfg(cfg)
        if anchor == db_m5_layout.M5_ANCHOR:
            cfg = m5.prepare_m5_build(cfg)
        environment = super().build(cfg)
        return configure_m3_environment(environment, cfg) if anchor == M3_ANCHOR else environment

    def newton_solver_options_for(self, cfg: DexBenchManufacturingE3EnvironmentCfg) -> dict:
        return type(self).staging_newton_solver_options[staging_anchor(cfg)]

    def apply_newton_scene_settings(self, scene: Scene, embodiment, cfg: DexBenchManufacturingE3EnvironmentCfg) -> None:
        """Each prop's Newton variant (the scene base's part), then at DB-M5's staging its parts' colliders."""
        super().apply_newton_scene_settings(scene, embodiment, cfg)
        if staging_anchor(cfg) == db_m5_layout.M5_ANCHOR:
            m5.apply_m5_newton_scene_settings(scene, cfg)

    def check_replayed_rows(self, cfg: DexBenchManufacturingE3EnvironmentCfg, path) -> None:
        """At DB-M5's staging, rows of another part or pile set are refused (``m5.check_m5_replayed_rows``)."""
        if staging_anchor(cfg) == db_m5_layout.M5_ANCHOR:
            m5.check_m5_replayed_rows(cfg, path)

    def recorded_build_inputs(self, cfg: DexBenchManufacturingE3EnvironmentCfg) -> dict:
        """At DB-M5's staging, the pile set's hash that ``check_replayed_rows`` asks of a row (``m5.m5_recorded_build_inputs``)."""
        if staging_anchor(cfg) == db_m5_layout.M5_ANCHOR:
            return m5.m5_recorded_build_inputs(cfg)
        return super().recorded_build_inputs(cfg)

    def belt_carried_objects(self, cfg: DexBenchManufacturingE3EnvironmentCfg) -> dict[str, tuple[float, float, float]]:
        """At DB-M5's staging, the kit box and its loaded parts, which the belt carries (``m5.m5_belt_carried_objects``)."""
        if staging_anchor(cfg) == db_m5_layout.M5_ANCHOR:
            return m5.m5_belt_carried_objects(cfg)
        return super().belt_carried_objects(cfg)

    def apply_newton_sim_settings(self, env_cfg, cfg: DexBenchManufacturingE3EnvironmentCfg) -> None:
        """The staging's solver options, at DB-M1's its finger effort and the EA rig's robot settings, and at
        DB-M5's its belt and contact."""
        super().apply_newton_sim_settings(env_cfg, cfg)
        if staging_anchor(cfg) == M3_ANCHOR:
            apply_m3_newton_settings(env_cfg, cfg)
        if staging_anchor(cfg) == M1_ANCHOR:
            apply_m1_newton_robot_settings(env_cfg, cfg)
        if staging_anchor(cfg) == db_m5_layout.M5_ANCHOR:
            m5.apply_m5_newton_sim_settings(env_cfg, cfg)

    def object_axis_for(self, cfg):
        axis = super().object_axis_for(cfg)
        if staging_anchor(cfg) == M3_ANCHOR and axis is not None:
            return (db_m3_layout.M3_E3_OBJECT_AXIS if cfg.m3_scene_layout == "e3_assembly_line"
                    else db_m3_layout.M3_OBJECT_AXIS)
        return axis

    def _start_pose_from_cli(self, cfg):
        pose = super()._start_pose_from_cli(cfg)
        return m1_start_pose(cfg, pose) if staging_anchor(cfg) == M1_ANCHOR else pose

    def _robot_base_anchors_for_args(self, cfg):
        anchors = super()._robot_base_anchors_for_args(cfg)
        return m1_base_anchors(cfg, anchors) if staging_anchor(cfg) == M1_ANCHOR else anchors

    def _anchor_facing_targets_for_args(self, cfg):
        if staging_anchor(cfg) == M1_ANCHOR:
            return m1_facing_targets(cfg, super()._anchor_facing_targets_for_args(cfg))
        if staging_anchor(cfg) == M3_ANCHOR and cfg.m3_scene_layout == "e3_assembly_line":
            return {M3_ANCHOR: (cfg.m3_station_offset_x_m, cfg.m3_station_offset_y_m + .10)}
        return super()._anchor_facing_targets_for_args(cfg)

    def build_scene(self, cfg: DexBenchManufacturingE3EnvironmentCfg) -> Scene:
        return type(self).staging_scenes[staging_anchor(cfg)](self, cfg)

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        super().add_cli_args(parser)
        for anchor in cls.robot_base_anchors_xyz:
            STATION_CLI_ARGS[anchor](parser)


@dataclass
class DexBenchManufacturingE3M1EnvironmentCfg(M1StationCfg, DexBenchEnvironmentCfg):
    """The config of DB-M1's own id: DB-M1's fields only, in the id's order and with its defaults."""

    anchor: str | None = M1_ANCHOR


@register_environment
class DexBenchManufacturingE3M1Environment(
    DexBenchManufacturingE3Environment, ArenaEnvironmentFactory[DexBenchManufacturingE3M1EnvironmentCfg]
):
    """``dexbench_manufacturing_e3_m1_environment``, DB-M1's own id: the E3 environment with DB-M1's staging
    only, its task, anchor and flags. For the DB-M1 benchmark configs and the recordings that name it."""

    name = "dexbench_manufacturing_e3_m1_environment"
    _legacy_argparse_cfg_type = DexBenchManufacturingE3M1EnvironmentCfg
    robot_base_anchors_xyz = db_m1_layout.ROBOT_BASE_ANCHORS_XYZ
    anchor_facing_targets_xy = db_m1_layout.ANCHOR_FACING_TARGETS_XY
    default_anchor = M1_ANCHOR
    anchor_help = "Robot base anchor: A_m1, the washer slot."
    available_tasks = {INSERT_DRAWER_TASK_NAME: make_insert_drawer_task, PICK_INSERT_DRAWER_TASK_NAME: make_insert_drawer_task}
    object_axes = {INSERT_DRAWER_TASK_NAME: db_m1_layout.M1_DRAWER_OBJECT_AXIS, PICK_INSERT_DRAWER_TASK_NAME: db_m1_layout.M1_TABLETOP_DRAWER_OBJECT_AXIS}


@dataclass
class DexBenchManufacturingE3M4EnvironmentCfg(M4StationCfg, DexBenchEnvironmentCfg):
    """The config of DB-M4's own id: DB-M4's fields only, in the id's order and with its defaults."""

    anchor: str | None = M4_ANCHOR


@register_environment
class DexBenchManufacturingE3M4Environment(
    DexBenchManufacturingE3Environment, ArenaEnvironmentFactory[DexBenchManufacturingE3M4EnvironmentCfg]
):
    """``dexbench_manufacturing_e3_m4_environment``, DB-M4's own id: the E3 environment with DB-M4's staging
    only, its task, anchor and flags. For the recordings that name it."""

    name = "dexbench_manufacturing_e3_m4_environment"
    _legacy_argparse_cfg_type = DexBenchManufacturingE3M4EnvironmentCfg
    robot_base_anchors_xyz = db_m4_layout.ROBOT_BASE_ANCHORS_XYZ
    anchor_facing_targets_xy = db_m4_layout.ANCHOR_FACING_TARGETS_XY
    default_anchor = M4_ANCHOR
    anchor_help = "Robot base anchor: A_m4, facing the cabinet on the workbench."
    available_tasks = {HAND_START_LEG_TASK_NAME: make_hand_start_leg_task}
    object_axes = {HAND_START_LEG_TASK_NAME: db_m4_layout.M4_LEG_OBJECT_AXIS}


@dataclass
class DexBenchManufacturingE3M3EnvironmentCfg(M3StationCfg, DexBenchEnvironmentCfg):
    """M3's recorded environment id, with the physical pickup defaults."""

    task: str = POWER_DRIVER_TASK_NAME
    anchor: str | None = M3_ANCHOR
    base_offset_x: float = -.1234
    base_offset_y: float = -.29
    actuator_profile: str | None = None
    """Read recorded configs only; E3 builds have no actuator-profile option."""

    def __post_init__(self):
        M3StationCfg.__post_init__(self)
        if self.actuator_profile not in (None, "power_driver"):
            raise ValueError(f"Unknown recorded M3 actuator_profile {self.actuator_profile!r}")


@register_environment
class DexBenchManufacturingE3M3Environment(
    DexBenchManufacturingE3Environment, ArenaEnvironmentFactory[DexBenchManufacturingE3M3EnvironmentCfg]
):
    """Recorded M3 id, restricted to the shared environment's M3 staging."""

    name = "dexbench_manufacturing_e3_m3_environment"
    _legacy_argparse_cfg_type = DexBenchManufacturingE3M3EnvironmentCfg
    robot_base_anchors_xyz = {M3_ANCHOR: (0., -.78, 0.)}
    anchor_facing_targets_xy = {M3_ANCHOR: (0., 0.)}
    default_anchor = M3_ANCHOR
    available_tasks = {POWER_DRIVER_TASK_NAME: make_power_driver_task}
    object_axes = {POWER_DRIVER_TASK_NAME: db_m3_layout.M3_E3_OBJECT_AXIS}

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        super().add_cli_args(parser)
        parser.set_defaults(task=POWER_DRIVER_TASK_NAME, base_offset_x=-.1234, base_offset_y=-.29)
        parser.add_argument(
            "--actuator_profile", choices=("power_driver",), default=None,
            help="Read recorded M3 configs; task-local driver gains apply to physical pickup.",
        )


@dataclass
class DexBenchManufacturingM5MovingConveyorKittingEnvironmentCfg(m5.M5StationCfg, DexBenchEnvironmentCfg):
    """The config of DB-M5's own id: DB-M5's fields only, in the id's order and with its defaults."""

    anchor: str | None = db_m5_layout.M5_ANCHOR


@register_environment
class DexBenchManufacturingM5MovingConveyorKittingEnvironment(
    DexBenchManufacturingE3Environment, ArenaEnvironmentFactory[DexBenchManufacturingM5MovingConveyorKittingEnvironmentCfg]
):
    """``dexbench_manufacturing_m5_moving_conveyor_kitting_environment``, DB-M5's own id: the E3 environment with
    DB-M5's staging only, its task, anchor and flags. For the recordings, the collection config and the GA set
    that name it."""

    name = "dexbench_manufacturing_m5_moving_conveyor_kitting_environment"
    _legacy_argparse_cfg_type = DexBenchManufacturingM5MovingConveyorKittingEnvironmentCfg
    robot_base_anchors_xyz = db_m5_layout.ROBOT_BASE_ANCHORS_XYZ
    anchor_facing_targets_xy = db_m5_layout.ANCHOR_FACING_TARGETS_XY
    default_anchor = db_m5_layout.M5_ANCHOR
    anchor_help = "Robot base anchor: A_m5, the M5 kitting station (bin <-> kit box)."
    available_tasks = {m5.PLACE_PART_IN_KIT_BOX_TASK_NAME: m5.make_place_part_in_kit_box_task}
    object_axes = {m5.PLACE_PART_IN_KIT_BOX_TASK_NAME: M5_PART_AXIS}
