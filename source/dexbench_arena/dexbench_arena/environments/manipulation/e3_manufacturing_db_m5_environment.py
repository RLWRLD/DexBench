# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""DB-M5: the E3 assembly line's M5 moving-conveyor kitting cell, staged for ``place_part_in_kit_box`` (Newton only)."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import argparse
import re
from pathlib import Path
from typing import TYPE_CHECKING

import newton
from packaging.version import Version

from dexbench_arena.assets.assembly_line_e3_m5_station import M5_PARTS_BIN_RELATIVE_PATH
from dexbench_arena.assets.newton_conveyor import BeltDriveCfg
from dexbench_arena.assets.physics_variants import select_newton_colliders
from dexbench_arena.contact_materials import WUJI_GRASP_CONTACT_MATERIAL
from dexbench_arena.embodiments.lookup import embodiment_class, start_poses
from dexbench_arena.embodiments.rby1m.constants import RBY1M_BASE_MODEL
from dexbench_arena.layouts.db_m5_parts import (
    DEFAULT_PART,
    M5_PART_NEWTON_COLLIDERS,
    bin_part_names,
    get_m5_part,
    kit_box_part_names,
)
from dexbench_arena.layouts.db_m5_parts_bin_clutter import (
    PartsBinClutter,
    PartsBinPileSet,
    load_parts_bin_clutter,
    load_parts_bin_pile_set,
    seed_reports,
)
from dexbench_arena.layouts.db_m5_layout import (
    CONVEYOR_BELT_END_X_M,
    CONVEYOR_FRAME_N_BELT_FACE_Y_M,
    CONVEYOR_FRAME_S_BELT_FACE_Y_M,
    CONVEYOR_TOP_Z_M,
    CONVEYOR_TRAVEL_DIRECTION_W,
    DEFAULT_EMPTY_COMPARTMENT,
    KIT_BOX_COMPARTMENT_FLOOR_TOP_Z,
    KIT_BOX_COMPARTMENTS,
    KIT_BOX_COMPARTMENTS_MAX_XY_LOCAL,
    KIT_BOX_COMPARTMENTS_MIN_XY_LOCAL,
    KIT_BOX_LEADING_EDGE_LOCAL_X,
    KIT_BOX_MAX_BELT_GAP_M,
    KIT_BOX_NAME,
    KIT_BOX_ON_BELT_XYZ,
    KIT_BOX_OUTER_HALF_XY,
    M5_ANCHOR,
    M5_PART_AXIS,
    PARTS_BIN_CLUTTER_POSES_BIN_LOCAL,
    PARTS_BIN_FLOOR_TOP_Z_LOCAL,
    PARTS_BIN_INTERIOR_HALF_XY,
    RBY1_A_M5_BASE_XYZ,
    kit_box_part_poses,
    parts_bin_part_poses_world,
    seeds_in_quantiles,
)
from dexbench_arena.environments.manipulation.newton_isotropic_inertia import install_isotropic_inertia
from dexbench_arena.environments.manipulation.newton_rolling_friction import (
    ROLLING_CONDIM,
    install_rolling_friction,
)
from dexbench_arena.environments.manipulation.newton_shape_contact import (
    ShapeContactRequest,
    install_shape_contact,
)
from dexbench_arena.environments.manipulation.newton_solref import solref_to_newton_ke_kd
from dexbench_arena.environments.manipulation.scene_environment_base import TASKS
from dexbench_arena.evaluation.artifacts import read_jsonl
from dexbench_arena.evaluation.evaluation_signature import NEWTON_SHAPE_OVERRIDES_ATTRIBUTE

if TYPE_CHECKING:
    from isaaclab_arena.scene.scene import Scene

PLACE_PART_IN_KIT_BOX_TASK_NAME = "place_part_in_kit_box"
_M5_ONLY = {TASKS: (PLACE_PART_IN_KIT_BOX_TASK_NAME,)}


PART_ISOTROPIC_INERTIA_NAME = "m5_part_isotropic_inertia"
M5_COLLISION_DECIMATION = 1
M5_SOLREF = (0.02, 2.0)
M5_NEWTON_SOLVER_OPTIONS = {"num_substeps": 8, "solver": "newton"}
M5_LS_ITERATIONS = 50
KIT_BOX_SOLREF = (0.01, 1.0)
KIT_BOX_SOLMIX = 100.0
KIT_BOX_CONTACT_NAME = "m5_kit_box_contact"
KIT_BOX_SHAPE_PATTERN = r"/belt_kit_box/.*KbFC_"
HAND_SOLREF = (0.02, 1.0)
HAND_SOLMIX = 1.0
HAND_CONTACT_NAME = "m5_wuji2_hand_contact"
PART_ROLLING_FRICTION = 2.0e-4
PART_ROLLING_FRICTION_NAME = "m5_part_rolling_friction"


def part_rolling_friction_pattern(part_key: str) -> str:
    """The colliders of the bin pile's and the kit box's ``part_key`` parts and of the parts bin, by prim path."""

    return rf"/(?:bin_part|kit_box)_{re.escape(part_key)}_\d+/|/{re.escape(M5_PARTS_BIN_RELATIVE_PATH)}/"


CONVEYOR_GRAPH_PRIM_NAME = "ConveyorBeltGraph"
CONVEYOR_VELOCITY_ATTR = "graph:variable:Velocity"
DEFAULT_CONVEYOR_VELOCITY_MPS = 0.1
CONVEYOR_BELT_SHAPE_PATTERN = r"/Slot_Conveyor/.*ConvFC_Belt"
BELT_DRIVE_FRICTION = 1.0
CONVEYOR_BELT_TEXTURE_PATTERN = r"/MAT_FC_BeltAnim/UVAnim$"
CONVEYOR_UV_PER_METRE = 0.5




def resolve_parts_bin_clutter(args_cli: argparse.Namespace) -> PartsBinClutter | None:
    """The dbclutter report ``--parts_bin_clutter`` names, loaded and checked; None when unset."""

    path = getattr(args_cli, "parts_bin_clutter", None)
    if not path:
        return None
    if getattr(args_cli, "parts_bin_clutter_dir", None):
        raise ValueError("--parts_bin_clutter names one pile and --parts_bin_clutter_dir a set to draw from; give one")
    return load_parts_bin_clutter(
        path, bin_interior_half_xy=PARTS_BIN_INTERIOR_HALF_XY, bin_floor_top_z=PARTS_BIN_FLOOR_TOP_Z_LOCAL
    )


def m5_part(args_cli: argparse.Namespace) -> str:
    """The part ``args_cli`` fills the parts bin and the kit box with."""

    requested = getattr(args_cli, "part", None)
    clutter = resolve_parts_bin_clutter(args_cli)
    if clutter is not None:
        if requested and requested != clutter.part:
            raise ValueError(f"--part {requested}, but the --parts_bin_clutter report is a pile of {clutter.part}")
        part = clutter.part
    else:
        part = requested or DEFAULT_PART
        if part != DEFAULT_PART and not getattr(args_cli, "parts_bin_clutter_dir", None):
            raise ValueError(
                f"--part {part}: the default pile is {DEFAULT_PART}; give --parts_bin_clutter_dir "
                f"(or a --parts_bin_clutter report) with {part} piles"
            )
    M5_PART_AXIS.resolve(argparse.Namespace(**{M5_PART_AXIS.field: part}))
    return part


def resolve_parts_bin_pile_set(args_cli: argparse.Namespace) -> PartsBinPileSet | None:
    """The piles ``--parts_bin_clutter_dir`` offers the part, within ``--parts_bin_seed_quantiles``; None when unset."""

    root = getattr(args_cli, "parts_bin_clutter_dir", None)
    quantiles = getattr(args_cli, "parts_bin_seed_quantiles", None)
    if not root:
        if quantiles is not None:
            raise ValueError(
                f"--parts_bin_seed_quantiles {list(quantiles)} bounds the seeds of a --parts_bin_clutter_dir set; "
                "give the set"
            )
        return None
    part = m5_part(args_cli)
    reports = seed_reports(Path(root).expanduser() / part)
    if not reports:
        raise ValueError(f"--parts_bin_clutter_dir {root}: no dbclutter reports under {part}/")
    seeds = seeds_in_quantiles(reports, quantiles) if quantiles is not None else tuple(sorted(reports))
    if not seeds:
        raise ValueError(
            f"--parts_bin_seed_quantiles {list(quantiles)} keeps none of the {len(reports)} {part} seeds under {root}"
        )
    return load_parts_bin_pile_set(
        {seed: reports[seed] for seed in seeds},
        part=part,
        bin_interior_half_xy=PARTS_BIN_INTERIOR_HALF_XY,
        bin_floor_top_z=PARTS_BIN_FLOOR_TOP_Z_LOCAL,
    )


def parts_bin_pile(args_cli: argparse.Namespace) -> tuple[str, tuple[tuple[float, ...], ...]]:
    """``(part, poses_bin_local)`` of the pile the scene spawns."""

    part = m5_part(args_cli)
    pile_set = resolve_parts_bin_pile_set(args_cli)
    if pile_set is not None:
        return part, pile_set.piles[0][1].poses_bin_local
    clutter = resolve_parts_bin_clutter(args_cli)
    if clutter is None:
        return DEFAULT_PART, PARTS_BIN_CLUTTER_POSES_BIN_LOCAL
    return clutter.part, clutter.poses_bin_local


def kit_box_empty_compartments(args_cli: argparse.Namespace) -> tuple[int, ...] | None:
    """The compartments ``--kit_box_empty_compartments`` lets a reset leave empty; None leaves DEFAULT_EMPTY_COMPARTMENT empty."""

    compartments = getattr(args_cli, "kit_box_empty_compartments", None)
    if compartments is None:
        return None
    indices = tuple(compartments)
    valid = range(len(KIT_BOX_COMPARTMENTS))
    if not indices or len(set(indices)) != len(indices) or not all(type(index) is int and index in valid for index in indices):
        raise ValueError(
            f"--kit_box_empty_compartments {list(compartments)}: give distinct compartments of {list(valid)} "
            "(KIT_BOX_COMPARTMENTS: 0 far right, 1 near right, 2 far left, 3 near left)"
        )
    return indices


def parts_bin_pile_names(args_cli: argparse.Namespace) -> tuple[str, ...]:
    """The bin pile's instance names for ``args_cli``: ``bin_part_<part>_NN``."""

    part, poses = parts_bin_pile(args_cli)
    return bin_part_names(part, len(poses))




M5_START_POSE = "m5_kitting"
"""The embodiment's named pose the task starts the robot from: the replay home leaned over the parts bin."""


def m5_start_pose(cfg) -> dict[str, float] | None:
    """The joint pose DB-M5's task starts the robot from, the embodiment's ``M5_START_POSE``; None without a robot."""

    embodiment = getattr(cfg, "embodiment", None)
    if embodiment is None:
        return None
    poses = start_poses(embodiment)
    if M5_START_POSE not in poses:
        raise ValueError(
            f"DB-M5 starts from the embodiment's {M5_START_POSE!r} pose; {embodiment!r} declares none "
            f"(available: {tuple(poses)})"
        )
    return dict(poses[M5_START_POSE])


def m5_shell_start_pose(args_cli) -> str | None:
    """The ``--start_pose`` a run without the task takes so its robot still leans over the bin: ``M5_START_POSE``
    for an embodiment that declares it, else None."""

    return M5_START_POSE if M5_START_POSE in start_poses(getattr(args_cli, "embodiment", None)) else None


def build_place_part_in_kit_box_task(
    part: str,
    part_names: tuple[str, ...],
    default_empty_compartment: int,
    robot_joint_pos_overrides: dict[str, float] | None = None,
):
    """DB-M5's task over a bin pile of ``part_names``, instances of ``part`` (a ``db_m5_parts`` key), judged against
    ``default_empty_compartment`` unless a reset records another."""

    from dexbench_arena.tasks.place_part_in_moving_kit_box import DexBenchPlacePartInMovingKitBoxTask

    geometry = get_m5_part(part)
    return DexBenchPlacePartInMovingKitBoxTask(
        kit_box_name=KIT_BOX_NAME,
        part_names=tuple(part_names),
        part_label=geometry.label,
        compartments_min_xy_local=KIT_BOX_COMPARTMENTS_MIN_XY_LOCAL,
        compartments_max_xy_local=KIT_BOX_COMPARTMENTS_MAX_XY_LOCAL,
        default_empty_compartment=default_empty_compartment,
        compartment_floor_z_local=KIT_BOX_COMPARTMENT_FLOOR_TOP_Z,
        part_center_local=geometry.center_local,
        part_extents_local=geometry.extents_local,
        part_end_points_local=geometry.end_points_local,
        kit_box_leading_edge_local_x=KIT_BOX_LEADING_EDGE_LOCAL_X,
        belt_end_x_m=CONVEYOR_BELT_END_X_M,
        kit_box_half_xy=KIT_BOX_OUTER_HALF_XY,
        belt_y_range=(CONVEYOR_FRAME_S_BELT_FACE_Y_M, CONVEYOR_FRAME_N_BELT_FACE_Y_M),
        belt_top_z_m=CONVEYOR_TOP_Z_M,
        max_belt_gap_m=KIT_BOX_MAX_BELT_GAP_M,
        robot_joint_pos_overrides=robot_joint_pos_overrides,
        hand_contact_material=WUJI_GRASP_CONTACT_MATERIAL,
    )


def make_place_part_in_kit_box_task(environment, cfg):
    """Build DB-M5's task (``available_tasks``): any parts-bin part into the kit box's empty compartment before
    the belt end."""

    del environment
    return build_place_part_in_kit_box_task(
        parts_bin_pile(cfg)[0],
        parts_bin_pile_names(cfg),
        (kit_box_empty_compartments(cfg) or (DEFAULT_EMPTY_COMPARTMENT,))[0],
        robot_joint_pos_overrides=m5_start_pose(cfg),
    )


def build_manufacturing_m5_moving_conveyor_kitting_scene(
    asset_registry,
    *,
    m5_station_edits: bool = True,
    hide_room_shell: bool = False,
    part: str = DEFAULT_PART,
    parts_bin_poses_bin_local=PARTS_BIN_CLUTTER_POSES_BIN_LOCAL,
    empty_compartments: tuple[int, ...] | None = None,
    parts_bin_piles: dict[int, tuple[tuple[float, ...], ...]] | None = None,
) -> Scene:
    """Build the M5 kitting scene without an embodiment."""

    from dexbench_arena.assets import background_library as _background_library  # noqa: F401
    from dexbench_arena.assets import object_library as _object_library  # noqa: F401
    from dexbench_arena.variations.kit_box_compartment_variation import KitBoxEmptyCompartmentVariation
    from dexbench_arena.variations.pose_layout_variation import PoseLayoutVariation, PoseLayoutVariationCfg
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose

    background = asset_registry.get_asset_by_name("dexbench_assembly_line_e3")(
        m5_station_edits=m5_station_edits,
        hide_room_shell=hide_room_shell,
    )

    interior_light = asset_registry.get_asset_by_name("dexbench_tube_light")(
        initial_pose=Pose(position_xyz=(2.85, 2.8, 3.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
    )

    kit_box = asset_registry.get_asset_by_name("dexbench_assembly_line_e3_kit_box")(
        instance_name=KIT_BOX_NAME,
        initial_pose=Pose(position_xyz=KIT_BOX_ON_BELT_XYZ, rotation_xyzw=(0.0, 0.0, 0.0, 1.0)),
    )
    part_asset = asset_registry.get_asset_by_name(get_m5_part(part).asset_name)
    kit_box_poses = kit_box_part_poses(part, empty_compartments[0] if empty_compartments else DEFAULT_EMPTY_COMPARTMENT)
    kit_box_names = kit_box_part_names(part, len(kit_box_poses))
    kit_box_parts = [
        part_asset(instance_name=name, initial_pose=Pose(position_xyz=position_xyz, rotation_xyzw=rotation_xyzw))
        for name, (position_xyz, rotation_xyzw) in zip(kit_box_names, kit_box_poses)
    ]

    def as_pose7(poses):
        return tuple((*position_xyz, *rotation_xyzw) for position_xyz, rotation_xyzw in poses)

    kit_box.add_variation(
        KitBoxEmptyCompartmentVariation(
            kit_box_names,
            {index: as_pose7(kit_box_part_poses(part, index)) for index in range(len(KIT_BOX_COMPARTMENTS))},
            cfg=PoseLayoutVariationCfg(enabled=empty_compartments is not None, choices=list(empty_compartments or ())),
        )
    )

    bin_poses = parts_bin_part_poses_world(parts_bin_poses_bin_local)
    bin_names = bin_part_names(part, len(bin_poses))
    bin_parts = [
        part_asset(instance_name=name, initial_pose=Pose(position_xyz=position_xyz, rotation_xyzw=rotation_xyzw))
        for name, (position_xyz, rotation_xyzw) in zip(bin_names, bin_poses)
    ]
    if parts_bin_piles:
        background.add_variation(
            PoseLayoutVariation(
                bin_names,
                {seed: as_pose7(parts_bin_part_poses_world(poses)) for seed, poses in parts_bin_piles.items()},
                cfg=PoseLayoutVariationCfg(enabled=True),
                name="parts_bin_clutter_seed",
            )
        )

    return Scene(assets=[interior_light, background, kit_box, *kit_box_parts, *bin_parts])


NEWTON_WITH_CONTACT_FIXES = "1.8.0.dev0"


def require_newton_contact_fixes() -> None:
    """Refuse a Newton older than ``NEWTON_WITH_CONTACT_FIXES``, naming the image that carries the fixes."""

    if Version(newton.__version__) < Version(NEWTON_WITH_CONTACT_FIXES):
        raise RuntimeError(
            f"DB-M5 needs Newton's MPR/GJK contact fixes, which Newton {newton.__version__} lacks: a part pushed into "
            "the parts bin is thrown out. Run in the image docker/build.py builds (Newton main 646cece4, "
            f"{NEWTON_WITH_CONTACT_FIXES}), with nothing older ahead of it on PYTHONPATH."
        )


@dataclass
class M5StationCfg:
    """DB-M5's settings: the fields the E3 config and DB-M5's own id's config take for this staging."""

    task: str = "none"
    """Task variant the scene builds: ``place_part_in_kit_box``, or ``none`` for the scene shell."""

    teleop_session: bool = field(default=False, metadata=_M5_ONLY)
    """An operator drives the run from inside the cell: the room's four walls are not drawn (they still collide)."""

    conveyor_velocity: float = field(default=DEFAULT_CONVEYOR_VELOCITY_MPS, metadata=_M5_ONLY)
    """Belt speed in m/s; 0 stops the belt."""

    parts_bin_clutter: str | None = field(default=None, metadata=_M5_ONLY)
    """A dbclutter clutter report to take the parts-bin pile and part from; unset is the default bolt pile."""

    part: str | None = field(default=None, metadata=_M5_ONLY)
    """The part (``M5_PART_AXIS``, a ``db_m5_parts`` key) the bin and the kit box are filled with; unset is the
    ``parts_bin_clutter`` report's part, else the default bolt. ``build`` sets it to the part it resolves, so the
    recorded config and ``task_object`` name the part that ran."""

    parts_bin_clutter_dir: str | None = field(default=None, metadata=_M5_ONLY)
    """A directory of dbclutter reports, one subdirectory per part, whose ``part`` piles a reset draws among."""

    kit_box_empty_compartments: list[int] | None = field(default=None, metadata=_M5_ONLY)
    """``KIT_BOX_COMPARTMENTS`` indices a reset draws the empty compartment from; unset leaves DEFAULT_EMPTY_COMPARTMENT (3) empty."""

    parts_bin_seed_quantiles: list[float] | None = field(default=None, metadata=_M5_ONLY)
    """``[low, high)`` share, by rank, of the part's ``parts_bin_clutter_dir`` seeds a reset draws the pile from; unset is every seed."""



def _number_list(kind: type):
    """argparse type for a comma-separated list of ``kind`` (``0,2,3``); the values are checked where they are used."""

    def parse(text: str) -> list:
        try:
            return [kind(token) for token in text.split(",") if token.strip()]
        except ValueError:
            raise argparse.ArgumentTypeError(f"takes comma-separated {kind.__name__}s, got {text!r}") from None

    return parse


M5_EMBODIMENT_ANCHOR_OVERRIDES = {RBY1M_BASE_MODEL: {M5_ANCHOR: RBY1_A_M5_BASE_XYZ}}
"""Stand every RBY1-M embodiment on the floor east of the staging shelf (``embodiment_anchor_overrides``)."""


PARTS_BIN_PILE_SET_ROW_KEY = "parts_bin_clutter_set_sha256"
"""A start-state row's hash of the pile set its seed indexes (``write_db_m5_eval_conditions.py``), beside the draws."""


def m5_recorded_build_inputs(cfg: M5StationCfg) -> dict:
    """What a condition recording at DB-M5's staging writes into each row (``recorded_build_inputs``): the pile set's
    hash under ``PARTS_BIN_PILE_SET_ROW_KEY``, as ``write_db_m5_eval_conditions.py`` writes it, so a recorded row
    replays where a written one does and is refused where it is; none for the default pile, whose rows name none."""

    pile_set = resolve_parts_bin_pile_set(cfg)
    return {PARTS_BIN_PILE_SET_ROW_KEY: pile_set.sha256} if pile_set is not None else {}


def check_m5_replayed_rows(cfg: M5StationCfg, path) -> None:
    """Refuse (ValueError) start-state rows ``path`` this build would start from at another part's piles or at
    another pile set's."""

    part = m5_part(cfg)
    pile_set = resolve_parts_bin_pile_set(cfg)
    expected = pile_set.sha256 if pile_set is not None else None
    _, rows = read_jsonl(path)
    for index, row in enumerate(rows):
        row_part = (row.get("task_object") or {}).get("key")
        row_set = row.get(PARTS_BIN_PILE_SET_ROW_KEY)
        if row_part != part:
            raise ValueError(f"{path} row {index} starts from a pile of {row_part!r}; this build fills the bin with {part!r}")
        if row_set != expected:
            raise ValueError(
                f"{path} row {index} was written for the pile set {row_set or 'none'}; this build's "
                f"(--parts_bin_clutter_dir {getattr(cfg, 'parts_bin_clutter_dir', None)}, --parts_bin_seed_quantiles "
                f"{getattr(cfg, 'parts_bin_seed_quantiles', None)}) is {expected or 'none'}"
            )


def build_m5_scene(environment, cfg: M5StationCfg) -> Scene:
    """The cell staged for DB-M5 (``build_scene``): the bin pile and the kit box, with the reset-time draws."""

    pile_set = resolve_parts_bin_pile_set(cfg)
    part, poses = parts_bin_pile(cfg)
    teleop = bool(getattr(cfg, "teleop_session", False))
    return build_manufacturing_m5_moving_conveyor_kitting_scene(
        environment.asset_registry,
        hide_room_shell=teleop,
        part=part,
        parts_bin_poses_bin_local=poses,
        empty_compartments=kit_box_empty_compartments(cfg),
        parts_bin_piles=(
            {seed: pile.poses_bin_local for seed, pile in pile_set.piles} if pile_set is not None else None
        ),
    )


def prepare_m5_build(cfg: M5StationCfg) -> M5StationCfg:
    """What DB-M5's environment does before the scene base's ``build``: the config to build from."""

    require_newton_contact_fixes()
    part = m5_part(cfg)
    if cfg.part != part:
        cfg = replace(cfg, part=part)
    return cfg


def m5_conveyor_belts(velocity_mps: float) -> tuple:
    """The belt ``BeltDriveCfg`` the conveyor manager drives at ``velocity_mps`` along its own +X; none at 0."""

    if not velocity_mps:
        return ()
    return (
        BeltDriveCfg(
            shape_pattern=CONVEYOR_BELT_SHAPE_PATTERN,
            velocity=(float(velocity_mps), 0.0, 0.0),
            frame="belt",
            friction=BELT_DRIVE_FRICTION,
            texture_pattern=CONVEYOR_BELT_TEXTURE_PATTERN,
            texture_rate=-CONVEYOR_UV_PER_METRE * float(velocity_mps),
        ),
    )


def m5_belt_carried_objects(cfg: M5StationCfg) -> dict[str, tuple[float, float, float]]:
    """What the belt carries, for a condition recording's settle (``belt_carried_objects``): the kit box and the
    parts loaded in it, each at the belt surface's world velocity (``conveyor_velocity`` along
    ``CONVEYOR_TRAVEL_DIRECTION_W``); none when the belt stands, as ``m5_conveyor_belts`` drives none. The parts-bin
    pile stands off the belt and is judged in the world."""

    if not cfg.conveyor_velocity:
        return {}
    velocity_w = tuple(float(cfg.conveyor_velocity) * axis for axis in CONVEYOR_TRAVEL_DIRECTION_W)
    part = m5_part(cfg)
    loaded = kit_box_part_names(part, len(kit_box_part_poses(part)))
    return {name: velocity_w for name in (KIT_BOX_NAME, *loaded)}


def apply_m5_newton_scene_settings(scene: Scene, cfg: M5StationCfg) -> None:
    """Put every part the scene spawns, the bin pile's and the kit box's, on ``M5_PART_NEWTON_COLLIDERS``."""

    pattern = re.compile(rf"(?:bin_part|kit_box)_{re.escape(m5_part(cfg))}_\d+")
    parts = [asset for name, asset in scene.assets.items() if pattern.fullmatch(name)]
    if not parts:
        raise RuntimeError(f"no DB-M5 part matches {pattern.pattern} among the scene's assets {sorted(scene.assets)}")
    for asset in parts:
        select_newton_colliders(asset.object_cfg.spawn, M5_PART_NEWTON_COLLIDERS)


def apply_m5_newton_sim_settings(env_cfg, cfg: M5StationCfg) -> None:
    """DB-M5's solver and contact on Newton: the belt (``m5_conveyor_belts`` through the conveyor manager), the
    line-search budget, the collision decimation, the default shape gains and the kit box's and the hands' own."""

    belts = m5_conveyor_belts(cfg.conveyor_velocity)
    if belts:
        from dexbench_arena.assets.newton_conveyor_manager import drive_conveyor_belts

        env_cfg.sim.physics = drive_conveyor_belts(env_cfg.sim.physics, belts)
    physics = env_cfg.sim.physics
    physics.solver_cfg.ls_iterations = M5_LS_ITERATIONS
    physics.collision_decimation = M5_COLLISION_DECIMATION
    physics.default_shape_cfg.ke, physics.default_shape_cfg.kd = solref_to_newton_ke_kd(*M5_SOLREF)
    own = {PART_ISOTROPIC_INERTIA_NAME, PART_ROLLING_FRICTION_NAME, KIT_BOX_CONTACT_NAME, HAND_CONTACT_NAME}
    recorded = {kind: {name: value for name, value in entries.items() if name not in own}
                for kind, entries in (getattr(env_cfg, NEWTON_SHAPE_OVERRIDES_ATTRIBUTE, None) or {}).items()}

    def record(kind: str, name: str, value: dict) -> None:
        recorded[kind] = {**recorded.get(kind, {}), name: value}

    part = m5_part(cfg)
    install_isotropic_inertia(PART_ISOTROPIC_INERTIA_NAME, (part,), "max")
    install_rolling_friction(PART_ROLLING_FRICTION_NAME, part_rolling_friction_pattern(part), PART_ROLLING_FRICTION)
    record("isotropic_inertia", PART_ISOTROPIC_INERTIA_NAME, {"tokens": [part], "mode": "max"})
    record("rolling_friction", PART_ROLLING_FRICTION_NAME, {
        "pattern": part_rolling_friction_pattern(part), "mu_rolling": PART_ROLLING_FRICTION, "condim": ROLLING_CONDIM})
    kit_box_ke, kit_box_kd = solref_to_newton_ke_kd(*KIT_BOX_SOLREF)
    install_shape_contact(
        ShapeContactRequest(KIT_BOX_CONTACT_NAME, KIT_BOX_SHAPE_PATTERN, kit_box_ke, kit_box_kd, KIT_BOX_SOLMIX)
    )
    record("shape_contact", KIT_BOX_CONTACT_NAME, {
        "pattern": KIT_BOX_SHAPE_PATTERN, "match": "shape", "solref": list(KIT_BOX_SOLREF), "ke": kit_box_ke,
        "kd": kit_box_kd, "solmix": KIT_BOX_SOLMIX})
    hand_pattern = getattr(embodiment_class(getattr(cfg, "embodiment", None)), "hand_body_label_pattern", None)
    if hand_pattern is not None:
        hand_ke, hand_kd = solref_to_newton_ke_kd(*HAND_SOLREF)
        install_shape_contact(
            ShapeContactRequest(HAND_CONTACT_NAME, hand_pattern, hand_ke, hand_kd, HAND_SOLMIX, match="body")
        )
        record("shape_contact", HAND_CONTACT_NAME, {
            "pattern": hand_pattern, "match": "body", "solref": list(HAND_SOLREF), "ke": hand_ke, "kd": hand_kd,
            "solmix": HAND_SOLMIX})
    setattr(env_cfg, NEWTON_SHAPE_OVERRIDES_ATTRIBUTE, {kind: entries for kind, entries in recorded.items() if entries})


def add_m5_cli_args(parser: argparse.ArgumentParser) -> None:
    """DB-M5's flags (``M5StationCfg``), each added once."""
    if "--teleop_session" not in parser._option_string_actions:
        parser.add_argument(
            "--teleop_session",
            "--teleop-session",
            action="store_true",
            help=(
                "This run is driven by an operator from inside the cell; teleop.py takes it "
                "here. What it changes is this "
                "environment's business -- see build_scene (the room's four walls are "
                "not drawn; they still collide)."
            ),
        )
    if "--conveyor_velocity" not in parser._option_string_actions:
        parser.add_argument(
            "--conveyor_velocity",
            "--conveyor-velocity",
            dest="conveyor_velocity",
            type=float,
            default=DEFAULT_CONVEYOR_VELOCITY_MPS,
            help=(
                "Belt surface speed in m/s (delivery-recommended band 0.1..0.2, hardware ceiling 0.6). "
                f"Default {DEFAULT_CONVEYOR_VELOCITY_MPS}; "
                "0.15 is the canonical layer's authored speed, 0 stops the belt. This is the "
                "speed the belt surface carries its cargo at and the texture scrolls at."
            ),
        )
    if "--parts_bin_clutter" not in parser._option_string_actions:
        parser.add_argument(
            "--parts_bin_clutter",
            "--parts-bin-clutter",
            dest="parts_bin_clutter",
            default=None,
            metavar="REPORT.json",
            help=(
                "A db-m5-create-stable-clutter (dbclutter) clutter report: the parts bin is filled "
                "with its settled pile (verdict.parts[].pose_exact, in the bin's frame) and the kit "
                "box with its part (request.part). Unset is the default 8-bolt pile."
            ),
        )
    if "--part" not in parser._option_string_actions:
        parser.add_argument(
            "--part",
            dest="part",
            default=None,
            metavar="KEY",
            help=(
                "The part the parts bin and the kit box are filled with (db_m5_layout.M5_PART_AXIS, a db_m5_parts "
                "key; its split says whether it is a training or a held-out part). Unset is the "
                "--parts_bin_clutter report's part, else the default M24 bolt. Any other part needs its piles: "
                "--parts_bin_clutter_dir or a --parts_bin_clutter report of it."
            ),
        )
    if "--parts_bin_clutter_dir" not in parser._option_string_actions:
        parser.add_argument(
            "--parts_bin_clutter_dir",
            "--parts-bin-clutter-dir",
            dest="parts_bin_clutter_dir",
            default=None,
            metavar="DIR",
            help=(
                "A directory of dbclutter reports with one subdirectory per part, as the dbclutter tarballs "
                "unpack (outputs/clutter/all_x40). Every --part report in it is a seed; each reset fills the "
                "bin with one seed's pile, drawn and recorded as dexbench_assembly_line_e3.parts_bin_clutter_seed, "
                "among the seeds --parts_bin_seed_quantiles keeps. Instead of --parts_bin_clutter."
            ),
        )
    if "--parts_bin_seed_quantiles" not in parser._option_string_actions:
        parser.add_argument(
            "--parts_bin_seed_quantiles",
            "--parts-bin-seed-quantiles",
            dest="parts_bin_seed_quantiles",
            type=_number_list(float),
            default=None,
            metavar="LOW,HIGH",
            help=(
                "The [LOW, HIGH) share, by rank, of the part's --parts_bin_clutter_dir seeds a reset draws the "
                "bin's pile from (0,0.5 is the first 20 of 40). db_m5_layout.M5_SPATIAL_RANGES has the training "
                "range and the range beyond it. Unset draws among every seed."
            ),
        )
    if "--kit_box_empty_compartments" not in parser._option_string_actions:
        parser.add_argument(
            "--kit_box_empty_compartments",
            "--kit-box-empty-compartments",
            dest="kit_box_empty_compartments",
            type=_number_list(int),
            default=None,
            metavar="I,J,...",
            help=(
                "The kit-box compartments a reset draws the empty one from, recorded as "
                "belt_kit_box.empty_compartment (KIT_BOX_COMPARTMENTS: 0 far right, 1 near right, 2 far left, "
                "3 near left, seen from the robot). db_m5_layout.M5_SPATIAL_RANGES has the training range and the "
                "range beyond it. Unset leaves the near-left compartment empty every episode."
            ),
        )
