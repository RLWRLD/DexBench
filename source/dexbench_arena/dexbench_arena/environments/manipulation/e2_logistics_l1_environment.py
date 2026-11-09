# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""DexBench DB-L1 Arena scene: polybags picked off the moving HB2 into one tote while boxes pass (E2 cell, 261014)."""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentFactory

from dexbench_arena import ASSETS_DIR
from dexbench_arena.assets.e2_logistics_cell import E2_SORT_TOTE_FLOOR_COLLIDER, E2_SORT_TOTE_WALL_COLLIDERS
from dexbench_arena.assets.polybag.sort_variants import (
    BOX_VARIANTS_BY_KEY,
    SORT_VARIANTS_BY_KEY,
    VARIANT_PACKAGE,
    variant_usd_relpath,
)
from dexbench_arena.contact_materials import WUJI_GRASP_CONTACT_MATERIAL
from dexbench_arena.embodiments.lookup import start_poses
from dexbench_arena.mimic_plugin import mimic_task_class
from dexbench_arena.environments.manipulation.e2_logistics_environment import (
    DexBenchLogisticsCenterE2Environment,
    add_e2_cell_cli_args,
    build_logistics_center_e2_scene,
    with_delivery_render_settings,
)
from dexbench_arena.layouts.e2_logistics_l1_layout import (
    L1_ANCHOR_YAWS_DEG,
    L1_BAG_SET_AXIS,
    L1_BAG_SETS,
    L1_BELT_CLEAR_Y_RANGE_M,
    L1_BELT_EDGE_MARGIN_M,
    L1_BELT_SPEED_M_S,
    L1_BELT_TOP_Z_M,
    L1_BELT_UV_PER_METRE,
    L1_BELT_X_RANGE_M,
    L1_BELTS,
    L1_BOX_DISTURBED_ACROSS_M,
    L1_BOX_DISTURBED_LIFTED_M,
    L1_BOX_DISTURBED_TURNED_DEG,
    L1_BOX_SPAWN_CLEARANCE_M,
    L1_BOX_YAW_OFF_BELT_DEG,
    L1_CONTINUOUS_EPISODE_LENGTH_S,
    L1_DEFAULT_BAG_SET,
    L1_DROPPED_Z_M,
    L1_ENTRY_EDGE_X_M,
    L1_EPISODE_LENGTH_S,
    L1_ITEM_Y_RANGE_M,
    L1_MISS_LINE_X_M,
    L1_MISSED_ABOVE_BELT_M,
    L1_PARK_SLAB_CENTER_M,
    L1_PARK_SLAB_SIZE_M,
    L1_PARK_SLAB_TOP_Z_M,
    L1_PICK_ZONE_X_M,
    L1_PLACED_HOLD_S,
    L1_PLACED_MAX_SPEED_M_S,
    L1_PLACED_MIN_FRACTION,
    L1_POLYBAG_SPAWN_CLEARANCE_M,
    L1_POLYBAG_STEPPINGS,
    L1_RBY1_START_JOINT_POS,
    L1_ROBOT_BASE_ANCHORS_XYZ,
    L1_SPACING_RANGE_M,
    L1_SPAWN_CLEARANCE_M,
    L1_STREAM_BAGS,
    L1_STREAM_BOX_VARIANTS,
    L1_STREAM_BOXES,
    L1_STREAM_LEADS,
    L1_STREAM_MAX_BAGS,
    L1_STREAM_MAX_BOXES,
    L1_STREAM_MODES,
    L1_VIEW_EYE_LOOKAT,
    L1_YAW_OFF_BELT_DEG,
    l1_bag_set_split,
    l1_park_xy,
    l1_tote_catch_box,
    l1_tote_inner_box,
)
from dexbench_arena.environments.manipulation.scene_environment_base import (
    DexBenchEnvironmentCfg,
    DexBenchSceneEnvironmentBase,
    _add_argument_if_missing,
)
from dexbench_arena.tasks.polybag_stream_plan import layout_draw_size, load_stream_plan
from dexbench_arena.teleop.newton_solver import mjwarp_solver_cfg, refusing_newton_solver_before_newton_main

if TYPE_CHECKING:
    from isaaclab_arena.scene.scene import Scene


ENVIRONMENT_NAME = "dexbench_logistics_e2_l1_environment"
TASK_NAME = "pick_polybag_from_conveyor"

L1_BELT_SHAPE_PATTERN = r"/HB2_Belt/TransportBody/TopRun$"
L1_BELT_UV_SHADER_PATTERN = r"/Belt_HB2/UVAnim$"
L1_TOTE_WALL_PATTERNS = tuple(rf"/{re.escape(path)}$" for path in E2_SORT_TOTE_WALL_COLLIDERS)
L1_TOTE_CATCH_ENTITY = "l1_tote_catch"
L1_PARK_SLAB_ENTITY = "l1_park_slab"
L1_PARTICLE_COLLISION_OFF = (
    rf"/{L1_PARK_SLAB_ENTITY}/",
    rf"/{re.escape(E2_SORT_TOTE_FLOOR_COLLIDER)}$",
    r"/Slot_SortTable/Asset/Geom/SortTable_Top/",
)
L1_WRIST_CAMERA_COLLIDERS = r"/wrist_camera/collisions/"
L1_WRIST_LINK_COLLIDERS = r"/link_(left|right)_arm_[56]/"
L1_MJWARP_LS_ITERATIONS = 50
L1_MJWARP_SOLVER = "newton"
COUPLINGS = ("one_way", "two_way")
FRAME_TRUNCATIONS = ("init", "all", "off")
L1_SELF_CONTACT_DETECTION = {"twice": 0, "once": -1}
L1_VBD_INITIAL_GUESSES = ("inertia", "contact")
L1_FILM_HESSIANS = ("newton", "psd")


def _pair(value, what: str) -> tuple[float, float]:
    """Two finite floats from a tuple or list (a typed config) or ``"a,b"`` (the command line); ValueError otherwise."""

    parts = [part for part in value.split(",") if part.strip()] if isinstance(value, str) else list(value)
    try:
        low, high = (float(part) for part in parts)
    except (TypeError, ValueError):
        raise ValueError(f"{what} {value!r}: two numbers, low and high") from None
    if not (math.isfinite(low) and math.isfinite(high) and low <= high):
        raise ValueError(f"{what} {value!r}: two finite numbers, low <= high")
    return low, high


def _pair_type(text: str) -> tuple[float, float]:
    """argparse type for the two-number flags: ``"0.8,0.95"`` -> ``(0.8, 0.95)``."""

    try:
        return _pair(text, "value")
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from None


def _positive(value, what: str) -> float:
    number = float(value)
    if not (math.isfinite(number) and number > 0.0):
        raise ValueError(f"{what} {value!r}: a positive number")
    return number


@dataclass(frozen=True)
class L1Settings:
    """What a DB-L1 config asks for, checked: the stream, the miss line, the bags' physics."""

    belt_speed_m_s: float
    mode: str
    bags: int
    boxes: int
    lead: str
    bag_set: str
    bag_yaw_off_belt_deg: tuple[float, float]
    item_y_m: tuple[float, float]
    spacing_m: tuple[float, float]
    miss_line_x_m: float
    stepping: dict
    material: dict
    seed: int
    plan_path: str | None
    plan_start: int
    headline_bags: int


def l1_polybag_stepping(cfg) -> dict:
    """The bags' stepping and coupling keyword arguments (PolybagObjectCfg's names): the ``polybag_stepping``
    candidate (e2_logistics_l1_layout.L1_POLYBAG_STEPPINGS) with any ``polybag_*_substeps``/``_iterations`` over it.
    ValueError for an unknown value, MJWarp substeps that do not divide the bags', or two-way coupling with MJWarp
    in fewer than every substep (the coupled manager refuses that)."""

    candidate = cfg.polybag_stepping
    if candidate not in L1_POLYBAG_STEPPINGS:
        raise ValueError(f"polybag_stepping {candidate!r}: one of {tuple(L1_POLYBAG_STEPPINGS)}")
    given = (cfg.polybag_vbd_substeps, cfg.polybag_vbd_iterations, cfg.polybag_mjwarp_substeps)
    vbd, iterations, mjwarp = (int(v if v is not None else d) for v, d in zip(given, L1_POLYBAG_STEPPINGS[candidate]))
    if min(vbd, iterations, mjwarp) < 1:
        raise ValueError(f"polybag stepping {vbd} x {iterations}, MJWarp {mjwarp}: positive integers")
    if vbd % mjwarp:
        raise ValueError(f"polybag_mjwarp_substeps {mjwarp} must divide polybag_vbd_substeps {vbd}")
    if cfg.polybag_coupling not in COUPLINGS:
        raise ValueError(f"polybag_coupling {cfg.polybag_coupling!r}: one of {COUPLINGS}")
    if cfg.polybag_coupling == "two_way" and mjwarp != vbd:
        raise ValueError(f"two-way coupling steps MJWarp in every substep: polybag_mjwarp_substeps {mjwarp} must be "
                         f"polybag_vbd_substeps {vbd}")
    if cfg.l1_bag_wrist_camera_collision not in ("on", "off"):
        raise ValueError(f"l1_bag_wrist_camera_collision {cfg.l1_bag_wrist_camera_collision!r}: on or off")
    if cfg.l1_bag_wrist_link_collision not in ("on", "off"):
        raise ValueError(f"l1_bag_wrist_link_collision {cfg.l1_bag_wrist_link_collision!r}: on or off")
    if cfg.polybag_contact_barrier not in ("on", "off"):
        raise ValueError(f"polybag_contact_barrier {cfg.polybag_contact_barrier!r}: on or off")
    if cfg.polybag_full_surface_contact not in ("on", "off"):
        raise ValueError(f"polybag_full_surface_contact {cfg.polybag_full_surface_contact!r}: on or off")
    if cfg.polybag_frame_truncation not in FRAME_TRUNCATIONS:
        raise ValueError(f"polybag_frame_truncation {cfg.polybag_frame_truncation!r}: one of {FRAME_TRUNCATIONS}")
    l1_self_contact_detection_interval(cfg)
    if cfg.polybag_vbd_init not in L1_VBD_INITIAL_GUESSES:
        raise ValueError(f"polybag_vbd_init {cfg.polybag_vbd_init!r}: one of {L1_VBD_INITIAL_GUESSES}")
    if cfg.polybag_film_hessian not in L1_FILM_HESSIANS:
        raise ValueError(f"polybag_film_hessian {cfg.polybag_film_hessian!r}: one of {L1_FILM_HESSIANS}")
    if cfg.polybag_tet_poisson is not None and not 0.0 <= float(cfg.polybag_tet_poisson) < 0.5:
        raise ValueError(f"polybag_tet_poisson {cfg.polybag_tet_poisson!r}: 0 <= nu < 0.5")
    if cfg.polybag_tet_damping is not None and not 0.0 <= float(cfg.polybag_tet_damping) < math.inf:
        raise ValueError(f"polybag_tet_damping {cfg.polybag_tet_damping!r}: a finite number >= 0 [Pa s]")
    return {
        "coupling_mode": cfg.polybag_coupling,
        "vbd_substeps": vbd,
        "vbd_iterations": iterations,
        "mjwarp_substeps": mjwarp,
        "full_surface_contact": cfg.polybag_full_surface_contact == "on",
        "contact_log_barrier": cfg.polybag_contact_barrier == "on",
        "frame_truncation": None if cfg.polybag_frame_truncation == "off" else cfg.polybag_frame_truncation,
    }


def l1_self_contact_detection_interval(cfg) -> int:
    """SolverVBD's ``particle_collision_detection_interval`` for ``polybag_self_contact_detection``
    (L1_SELF_CONTACT_DETECTION): 0 for twice, the default and DB-L4's, -1 for once. ValueError for another value."""
    mode = cfg.polybag_self_contact_detection
    if mode not in L1_SELF_CONTACT_DETECTION:
        raise ValueError(f"polybag_self_contact_detection {mode!r}: one of {tuple(L1_SELF_CONTACT_DETECTION)}")
    return L1_SELF_CONTACT_DETECTION[mode]


def l1_particle_collision_off(cfg) -> tuple[str, ...]:
    """The shape-label patterns the bags' particles do not collide with (PolybagCoupledSolverCfg.particle_collision_off):
    L1_PARTICLE_COLLISION_OFF, with the wrist's links (which carry the palm-camera mounts) when
    ``l1_bag_wrist_link_collision`` is off, else the mounts alone when ``l1_bag_wrist_camera_collision`` is off."""
    if cfg.l1_bag_wrist_link_collision == "off":
        return L1_PARTICLE_COLLISION_OFF + (L1_WRIST_LINK_COLLIDERS,)
    if cfg.l1_bag_wrist_camera_collision == "off":
        return L1_PARTICLE_COLLISION_OFF + (L1_WRIST_CAMERA_COLLIDERS,)
    return L1_PARTICLE_COLLISION_OFF


def l1_polybag_material(cfg) -> dict:
    """The bags' ``tet_*`` keyword arguments (PolybagObjectCfg's): None keeps the pool asset's value."""

    material = {
        "tet_youngs_modulus": None if cfg.polybag_youngs_modulus is None
        else _positive(cfg.polybag_youngs_modulus, "polybag_youngs_modulus"),
        "tet_contact_ke": _positive(cfg.polybag_contact_ke, "polybag_contact_ke"),
        "tet_contact_kd": _positive(cfg.polybag_contact_kd, "polybag_contact_kd"),
        "tet_collider_ke": _positive(cfg.polybag_collider_ke, "polybag_collider_ke"),
        "tet_collider_kd": _positive(cfg.polybag_collider_kd, "polybag_collider_kd"),
    }
    for name in ("polybag_contact_mu", "polybag_collider_mu"):
        value = getattr(cfg, name)
        if value is not None and not (math.isfinite(float(value)) and float(value) >= 0.0):
            raise ValueError(f"{name} {value!r}: a friction coefficient, 0 or more")
        material["tet_" + name.removeprefix("polybag_")] = None if value is None else float(value)
    if cfg.polybag_coupling == "two_way":
        for side in ("ke", "kd"):
            bag, collider = material[f"tet_contact_{side}"], material[f"tet_collider_{side}"]
            if bag != collider:
                raise ValueError(f"two-way coupling needs equal contact {side} on both sides: polybag_contact_{side} "
                                 f"{bag:g}, polybag_collider_{side} {collider:g}")
    return material


def l1_settings(cfg) -> L1Settings:
    """Every DB-L1 setting of ``cfg``, checked; ValueError for one the scene or the stream cannot take, for a belt
    speed other than the one DB-L1 runs at, and, in a teleop session (``teleop_device``), for a stream, bag set or
    orientation band other than the training demos' (``_check_teleop_settings``). ``l1_stream_bags`` and
    ``l1_stream_boxes`` are an episode's (a continuous stream empties the tote every ``l1_stream_bags`` placed bags and
    alternates a box after every bag while it has boxes)."""

    speed = float(cfg.belt_speed)
    if speed != L1_BELT_SPEED_M_S:
        raise ValueError(f"belt_speed {cfg.belt_speed!r}: HB2 runs at {L1_BELT_SPEED_M_S} m/s in every DB-L1 run "
                         "(belt speed is no axis: a sorting line runs its conveyors at one speed)")
    if cfg.l1_stream_mode not in L1_STREAM_MODES:
        raise ValueError(f"l1_stream_mode {cfg.l1_stream_mode!r}: one of {L1_STREAM_MODES}")
    bags, boxes = int(cfg.l1_stream_bags), int(cfg.l1_stream_boxes)
    if not 1 <= bags <= L1_STREAM_MAX_BAGS:
        raise ValueError(f"l1_stream_bags {bags}: 1 to {L1_STREAM_MAX_BAGS} (the park slab's)")
    if not 0 <= boxes <= L1_STREAM_MAX_BOXES:
        raise ValueError(f"l1_stream_boxes {boxes}: 0 to {L1_STREAM_MAX_BOXES} (the park slab's)")
    if cfg.l1_stream_lead not in L1_STREAM_LEADS:
        raise ValueError(f"l1_stream_lead {cfg.l1_stream_lead!r}: one of {L1_STREAM_LEADS}")
    if cfg.l1_stream_lead == "box" and boxes == 0:
        raise ValueError("l1_stream_lead box: the stream has no boxes")
    if cfg.l1_bag_set not in L1_BAG_SETS:
        raise ValueError(f"l1_bag_set {cfg.l1_bag_set!r}: one of {tuple(L1_BAG_SETS)}")
    if bags > len(L1_BAG_SETS[cfg.l1_bag_set]):
        raise ValueError(f"l1_stream_bags {bags}: bag set {cfg.l1_bag_set} holds {len(L1_BAG_SETS[cfg.l1_bag_set])}")
    yaw = _pair(cfg.l1_bag_yaw_off_belt_deg, "l1_bag_yaw_off_belt_deg")
    if not (0.0 <= yaw[0] and yaw[1] <= 90.0):
        raise ValueError(f"l1_bag_yaw_off_belt_deg {yaw}: degrees between a bag's long side and the belt, within 0-90")
    item_y = _pair(cfg.l1_item_y, "l1_item_y")
    if not (L1_BELT_CLEAR_Y_RANGE_M[0] < item_y[0] and item_y[1] < L1_BELT_CLEAR_Y_RANGE_M[1]):
        raise ValueError(f"l1_item_y {item_y}: item centres inside the clear belt, y {L1_BELT_CLEAR_Y_RANGE_M}")
    spacing = _pair(cfg.l1_spacing, "l1_spacing")
    if spacing[0] <= 0.0:
        raise ValueError(f"l1_spacing {spacing}: a positive centre-to-centre gap [m]")
    miss = float(cfg.l1_miss_line_x)
    if not (L1_ENTRY_EDGE_X_M < miss <= L1_BELT_X_RANGE_M[1]):
        raise ValueError(f"l1_miss_line_x {cfg.l1_miss_line_x!r}: on HB2, past the put-down line {L1_ENTRY_EDGE_X_M} "
                         f"and not past its end {L1_BELT_X_RANGE_M[1]}")
    plan_start = int(cfg.l1_stream_plan_start or 0)
    if plan_start and not cfg.l1_stream_plan:
        raise ValueError("l1_stream_plan_start goes with l1_stream_plan")
    if plan_start < 0:
        raise ValueError(f"l1_stream_plan_start {plan_start}: a plan entry, from 0")
    if cfg.l1_stream_plan and cfg.l1_stream_mode != "episode":
        raise ValueError("l1_stream_plan lays episodes out; a continuous stream (l1_stream_mode) draws as it goes")
    headline = int(cfg.l1_headline_bags or 0)
    if headline and (cfg.l1_stream_mode != "episode" or not 1 <= headline <= bags):
        raise ValueError(f"l1_headline_bags {cfg.l1_headline_bags}: the success of an episode of l1_stream_bags {bags} "
                         f"(1 to {bags}, l1_stream_mode episode), not of a {cfg.l1_stream_mode} stream")
    if cfg.teleop_device is not None:
        _check_teleop_settings(cfg, bags, boxes, yaw)
    return L1Settings(
        belt_speed_m_s=speed,
        mode=cfg.l1_stream_mode,
        bags=bags,
        boxes=boxes,
        lead=cfg.l1_stream_lead,
        bag_set=cfg.l1_bag_set,
        bag_yaw_off_belt_deg=yaw,
        item_y_m=item_y,
        spacing_m=spacing,
        miss_line_x_m=miss,
        stepping=l1_polybag_stepping(cfg),
        material=l1_polybag_material(cfg),
        seed=int(cfg.l1_stream_seed),
        plan_path=cfg.l1_stream_plan or None,
        plan_start=plan_start,
        headline_bags=headline,
    )


def _check_teleop_settings(cfg, bags: int, boxes: int, yaw: tuple[float, float]) -> None:
    """A DB-L1 recording records demos of the training conditions only: the decided
    stream (an episode of three bags and two boxes, a bag first and alternating, laid out from a collection row, or
    the continuous stream with the tote emptied every three placed bags), a training bag set and the training
    orientation band. ValueError for anything else, so that a flag typed after ``--run`` cannot record demos on
    held-out bags or orientations or on another stream. Evaluations run without a teleop device and take every set,
    band and mode."""

    decided = {"l1_stream_bags": (bags, L1_STREAM_BAGS), "l1_stream_boxes": (boxes, L1_STREAM_BOXES),
               "l1_stream_lead": (cfg.l1_stream_lead, "bag")}
    off = {name: given for name, (given, want) in decided.items() if given != want}
    if off:
        raise ValueError(f"a teleop session records DB-L1's stream ({L1_STREAM_BAGS} bags and "
                         f"{L1_STREAM_BOXES} boxes, a bag first, alternating: l1_stream_bags {L1_STREAM_BAGS}, "
                         f"l1_stream_boxes {L1_STREAM_BOXES}, l1_stream_lead bag), not {off}")
    if l1_bag_set_split(cfg.l1_bag_set) != "train":
        raise ValueError(f"l1_bag_set {cfg.l1_bag_set}: a recording takes a training set "
                         f"({', '.join(k for k in L1_BAG_SETS if l1_bag_set_split(k) == 'train')})")
    band = L1_YAW_OFF_BELT_DEG["train"]
    if not (band[0] <= yaw[0] and yaw[1] <= band[1]):
        raise ValueError(f"l1_bag_yaw_off_belt_deg {yaw}: a recording keeps to the training "
                         f"band {band}")


def l1_stream_params(settings: L1Settings) -> dict:
    """``PolybagStream``'s parameters but its items, from the run's settings and the layout."""

    return {
        "mode": settings.mode,
        "episode_bags": settings.bags,
        "episode_boxes": settings.boxes,
        "tote_boxes": (l1_tote_inner_box(),),
        "belt_speed": settings.belt_speed_m_s,
        "belt_top_z": L1_BELT_TOP_Z_M,
        "entry_x": L1_ENTRY_EDGE_X_M,
        "clear_y": L1_BELT_CLEAR_Y_RANGE_M,
        "edge_margin": L1_BELT_EDGE_MARGIN_M,
        "lateral_y": settings.item_y_m,
        "spacing_m": settings.spacing_m,
        "bag_yaw_off_belt_deg": settings.bag_yaw_off_belt_deg,
        "box_yaw_off_belt_deg": L1_BOX_YAW_OFF_BELT_DEG,
        "lead": settings.lead,
        "miss_x": settings.miss_line_x_m,
        "bag_clearance": L1_POLYBAG_SPAWN_CLEARANCE_M,
        "box_clearance": L1_BOX_SPAWN_CLEARANCE_M,
        "spawn_clearance": L1_SPAWN_CLEARANCE_M,
        "min_fraction": L1_PLACED_MIN_FRACTION,
        "max_speed": L1_PLACED_MAX_SPEED_M_S,
        "hold_s": L1_PLACED_HOLD_S,
        "drop_z": L1_DROPPED_Z_M,
        "missed_above_belt": L1_MISSED_ABOVE_BELT_M,
        "pick_zone_x": L1_PICK_ZONE_X_M,
        "disturbed_lateral": L1_BOX_DISTURBED_ACROSS_M,
        "disturbed_yaw_deg": L1_BOX_DISTURBED_TURNED_DEG,
        "disturbed_lift": L1_BOX_DISTURBED_LIFTED_M,
        "seed": settings.seed,
        "headline_bags": settings.headline_bags,
    }


def l1_stream_layout_draw_size(settings: L1Settings) -> int:
    """How many values an episode's layout draw holds (``l1_park_slab.stream_layout``): an ordering key per bag of the
    set and per box, then five per release of the episode's bags and boxes (``polybag_stream_plan.layout_draw_size``)."""

    return layout_draw_size(len(l1_bag_variants(settings)), len(L1_STREAM_BOX_VARIANTS), settings.bags + settings.boxes)


def l1_bag_variants(settings: L1Settings) -> tuple[str, ...]:
    """The scene's bags: every bag of the run's set, entities ``l1_bag_<i>`` in this order (an episode draws
    ``l1_stream_bags`` of them)."""

    return L1_BAG_SETS[settings.bag_set]


def l1_box_items() -> list:
    """The scene's boxes as stream items, every one of L1_STREAM_BOX_VARIANTS (an episode draws ``l1_stream_boxes`` of
    them), entities ``l1_box_<j>``, each waiting in park slot 4 + j at rest on the slab."""

    from dexbench_arena.tasks.polybag_stream import StreamItem

    items = []
    for j, key in enumerate(L1_STREAM_BOX_VARIANTS):
        sx, sy, sz = BOX_VARIANTS_BY_KEY[key].size_m
        park = (*l1_park_xy(L1_STREAM_MAX_BAGS + j), L1_PARK_SLAB_TOP_Z_M + L1_BOX_SPAWN_CLEARANCE_M)
        items.append(StreamItem("box", f"l1_box_{j}", (sx / 2.0, sy / 2.0), (*park, 0.0, 0.0, 0.0, 1.0), -1, sz, key))
    return items


def _load_plan(settings: L1Settings, bags, boxes) -> dict:
    """The collection plan's stream parameters (``tasks.polybag_stream_plan``), checked against the run: its bags,
    boxes and belt speed, the clear belt, the lead and the bags' orientation band; empty without a plan."""

    if settings.plan_path is None:
        return {}
    rest_long_axis = [90.0 if SORT_VARIANTS_BY_KEY[b.variant].body.footprint_m[1] >= SORT_VARIANTS_BY_KEY[
        b.variant].body.footprint_m[0] else 0.0 for b in bags]
    clear = (L1_BELT_CLEAR_Y_RANGE_M[0] + L1_BELT_EDGE_MARGIN_M, L1_BELT_CLEAR_Y_RANGE_M[1] - L1_BELT_EDGE_MARGIN_M)
    plan, digest = load_stream_plan(
        settings.plan_path, bag_variants=[b.variant for b in bags],
        half_extents=[(b.footprint_m[0] / 2.0, b.footprint_m[1] / 2.0) for b in bags] + [
            box.half_extent_xy for box in boxes],
        boxes=len(boxes), belt_speed=settings.belt_speed_m_s, deck_y=clear,
        spacing_m=sum(settings.spacing_m) / 2.0, lead=settings.lead,
        bag_band_deg=settings.bag_yaw_off_belt_deg, rest_long_axis_deg=rest_long_axis, episode_bags=settings.bags,
        episode_boxes=settings.boxes)
    if not settings.plan_start < len(plan):
        raise ValueError(f"l1_stream_plan_start {settings.plan_start}: {settings.plan_path} has entries "
                         f"0-{len(plan) - 1}")
    print(f"[dexbench] L1 collection plan {settings.plan_path}: {len(plan)} episodes, from entry "
          f"{settings.plan_start + 1} ({len(plan) - settings.plan_start} to go)", flush=True)
    return {"plan": plan, "plan_start": settings.plan_start, "plan_path": str(settings.plan_path),
            "plan_sha256": digest}


def _make_pick_polybag_from_conveyor_task(env, cfg: DexBenchLogisticsE2L1EnvironmentCfg):
    """Factory for DB-L1."""

    from dexbench_arena.tasks.pick_polybag_from_conveyor import DexBenchPickPolybagFromConveyorTask, L1Bag

    settings = l1_settings(cfg)
    keys = l1_bag_variants(settings)
    paths = {key: ASSETS_DIR / "object_library" / variant_usd_relpath(key) for key in keys}
    bags = [L1Bag(f"l1_bag_{i}", key, str(paths[key]), SORT_VARIANTS_BY_KEY[key].body.footprint_m, l1_park_xy(i))
            for i, key in enumerate(keys)]
    boxes = l1_box_items()
    stream = {**l1_stream_params(settings), "export_segments": cfg.teleop_device is not None,
              "put_down": not cfg.record_settle, "require_layout_draw": cfg.benchmark_replay,
              **_load_plan(settings, bags, boxes)}
    return mimic_task_class(DexBenchPickPolybagFromConveyorTask)(
        bags=bags,
        boxes=boxes,
        stream=stream,
        park_surface_z=L1_PARK_SLAB_TOP_Z_M,
        park_clearance=L1_POLYBAG_SPAWN_CLEARANCE_M,
        **settings.stepping,
        material=settings.material,
        episode_length_s=L1_CONTINUOUS_EPISODE_LENGTH_S if settings.mode == "continuous" else L1_EPISODE_LENGTH_S,
        robot_joint_pos_overrides=l1_start_pose(cfg),
        hand_contact_material=WUJI_GRASP_CONTACT_MATERIAL,
        viewer_eye_lookat=L1_VIEW_EYE_LOOKAT,
        box_colliders=L1_TOTE_WALL_PATTERNS,
    )


def l1_start_pose(cfg) -> dict[str, float] | None:
    """DB-L1's start pose: the embodiment's replay_home with L1_RBY1_START_JOINT_POS's torso, arms and head."""

    home = start_poses(getattr(cfg, "embodiment", None)).get("replay_home")
    if not home or not set(L1_RBY1_START_JOINT_POS) <= home.keys():
        return None
    return {**home, **L1_RBY1_START_JOINT_POS}


@dataclass
class DexBenchLogisticsE2L1EnvironmentCfg(DexBenchEnvironmentCfg):
    """Typed config for DexBenchLogisticsE2L1Environment: its flags, names and defaults."""

    task: str = "none"
    """Task variant the scene builds: ``pick_polybag_from_conveyor``, or ``none`` for the scene shell."""

    anchor: str = "A_L1"
    """Robot base anchor name (A_L1)."""

    teleop_session: bool = False
    """The run is driven by an operator from inside the cell: the room shell is hidden (still solid)."""

    delivery_render_settings: bool = True
    """False renders at Arena's defaults instead of the delivery's tonemap and exposure."""

    belt_speed: float = L1_BELT_SPEED_M_S
    """HB2's surface speed [m/s], positive = +X: 0.2, the one speed of every DB-L1 run (no speed axis); any other
    value is refused."""

    l1_stream_mode: str = "episode"
    """episode: an episode of ``l1_stream_bags`` bags and ``l1_stream_boxes`` boxes that succeeds once every bag is
    placed and no box disturbed and ends on the first failure (an evaluation); continuous: an endless stream, each
    placed bag a demo of its own and a failure discarding its segment (teleop collection, ``tasks.polybag_stream``)."""

    l1_stream_bags: int = L1_STREAM_BAGS
    """Polybags in an episode, drawn from the set's four (1-4); a continuous stream empties the tote every that many."""

    l1_stream_boxes: int = L1_STREAM_BOXES
    """Boxes in an episode, the distractors that pass, drawn from the four (0-4); a continuous stream alternates a box
    after every bag unless it is 0."""

    l1_stream_lead: str = "bag"
    """What comes first, the stream alternating after it (bag, box)."""

    l1_headline_bags: int | None = None
    """An episode's success is its first that many bags placed in stream order before any failure (DB-L1's
    benchmark: 2, which a model's run sets: ``spec.scoring_environment``), recorded with the
    all-bags rule beside it as ``challenge_success``; the episode still ends on every bag placed or the first failure.
    None: every bag placed and no box disturbed, the success a recording and a replay keep a demo by."""

    l1_bag_set: str = L1_DEFAULT_BAG_SET
    """GA-O: the run's bag set (e2_logistics_l1_layout.L1_BAG_SETS): train_a..train_d, training bags; eval_a and
    eval_b, held-out sizes; eval_c and eval_d, held-out colours."""

    l1_bag_wrist_camera_collision: str = "on"
    """Whether the bags' particles collide with the palm-camera mounts (L1_WRIST_CAMERA_COLLIDERS): on, as the
    delivered asset; off lets a hand get down onto a bag without the mount meeting it first (a feel-session trial)."""

    l1_bag_wrist_link_collision: str = "off"
    """Whether the bags' particles collide with the wrist's links above the palm (L1_WRIST_LINK_COLLIDERS, the
    palm-camera mounts included): off (T19D, the collection's) lets a palm-down hand squeeze a bag with only the palm
    and the fingers meeting it; on, as the delivered asset."""

    l1_bag_yaw_off_belt_deg: tuple[float, float] = L1_YAW_OFF_BELT_DEG["train"]
    """GA-S: the band [deg] of the angle between a bag's long side and the belt's direction it is put down at;
    (0, 30) trains, (60, 90) is held out."""

    l1_item_y: tuple[float, float] = L1_ITEM_Y_RANGE_M
    """Where across HB2 an item's centre is put down [m, world y]."""

    l1_spacing: tuple[float, float] = L1_SPACING_RANGE_M
    """How far the belt runs between two items' releases [m], drawn per item: the gap between their upstream edges."""

    l1_stream_seed: int = 0
    """Seeds each episode's (each continuous stream's) draw, with the episode's count and the environment's index."""

    l1_stream_plan: str | None = None
    """A collection plan (tasks.polybag_stream_plan, JSON) laying the episodes out in place of the draw (episode mode);
    left out of a replay (each demo records its layout)."""

    l1_stream_plan_start: int = 0
    """The plan entry the session starts at (0-based), to go on with a plan."""

    l1_miss_line_x: float = L1_MISS_LINE_X_M
    """An item on the belt whose leading edge reaches this x [m] is missed and taken off."""

    polybag_stepping: str = "a"
    """The bags' stepping candidate: a (8 VBD substeps x 5 iterations, MJWarp 8) or b (16 x 5, MJWarp 16)."""

    polybag_vbd_substeps: int | None = None
    """Physics substeps of the bags (VBD), over the candidate's; None: the candidate's."""

    polybag_vbd_iterations: int | None = 8
    """VBD iterations per substep, over the candidate's: 8 (T19D, the collection's, over candidate a's 5); None: the
    candidate's."""

    polybag_mjwarp_substeps: int | None = None
    """MJWarp substeps per physics step (the robot), over the candidate's; None: the candidate's."""

    polybag_full_surface_contact: str = "on"
    """Full-surface rigid-soft contact (on, off)."""

    polybag_contact_barrier: str = "off"
    """The hands' and colliders' contact with the bags as a log barrier (on) or Newton 1.6.1's linear penalty (off):
    PolybagObjectCfg.contact_log_barrier."""

    polybag_frame_truncation: str = "init"
    """Bag-frame self-contact truncation (init, all, off)."""

    polybag_self_contact_detection: str = "twice"
    """How often VBD finds the bags' self-contacts in a substep (L1_SELF_CONTACT_DETECTION): twice, Newton's default;
    once, faster (a feel-session trial)."""

    polybag_vbd_init: str = "contact"
    """Where VBD starts each substep's iterate (L1_VBD_INITIAL_GUESSES): contact (T19D, the collection's), a supported
    bag without the inertial sink; inertia, Newton's own."""

    polybag_film_hessian: str = "psd"
    """The film springs' Hessian in VBD's per-vertex step (L1_FILM_HESSIANS): psd (T19D, the collection's), its
    compressed transverse term clamped at >= 0, the same force; newton, Newton's own."""

    polybag_tet_poisson: float | None = 0.45
    """Every bag tet's Poisson ratio at its own shear modulus: 0.45 (T19D, the collection's); None: the asset's."""

    polybag_tet_damping: float | None = 10.0
    """Every bag tet's damping [Pa s]: 10 (T19D, the collection's); None: the asset's."""

    polybag_coupling: str = "two_way"
    """How the bags and the robot push on each other (one_way, two_way)."""

    polybag_youngs_modulus: float | None = None
    """The bags' Young's modulus [Pa]; None: the pool asset's own (per tet: 2.5 kPa, 0.5 kPa skin)."""

    polybag_contact_ke: float = 3000.0
    """The bags' contact stiffness [N/m]: their self-contact, and their side of a collider contact."""

    polybag_contact_kd: float = 1.0
    """The bags' contact damping."""

    polybag_contact_mu: float | None = 1.0
    """The bags' friction; None: the asset's (Newton's 0.5)."""

    polybag_collider_ke: float = 3000.0
    """Every collider's contact stiffness against the bags [N/m], the hands' included."""

    polybag_collider_kd: float = 1.0
    """Every collider's contact damping against the bags."""

    polybag_collider_mu: float | None = 3.0
    """Every collider's friction against the bags: 3 (T19D, the collection's); None: the asset's (0.8)."""


@register_environment
class DexBenchLogisticsE2L1Environment(
    DexBenchSceneEnvironmentBase, ArenaEnvironmentFactory[DexBenchLogisticsE2L1EnvironmentCfg]
):
    """DB-L1: polybags and boxes come down HB2; pick every polybag into the tote and let the boxes pass."""

    name = ENVIRONMENT_NAME
    _legacy_argparse_cfg_type = DexBenchLogisticsE2L1EnvironmentCfg
    robot_base_anchors_xyz = L1_ROBOT_BASE_ANCHORS_XYZ
    anchor_yaws_deg = L1_ANCHOR_YAWS_DEG
    default_anchor = "A_L1"
    anchor_help = "Robot base anchor: A_L1, north of HB2's downstream half, facing upstream (-X), the tote behind-right."
    available_tasks = {TASK_NAME: _make_pick_polybag_from_conveyor_task}
    object_axes = {TASK_NAME: L1_BAG_SET_AXIS}
    newton_solver_options = {**DexBenchLogisticsCenterE2Environment.newton_solver_options, "solver": L1_MJWARP_SOLVER}

    def task_object_usd_path(self, task_object) -> str:
        """A bag set's record carries the pool package's version (its main file's sidecar): the bags are built by the
        task, not registered assets."""

        return str(ASSETS_DIR / "object_library" / VARIANT_PACKAGE / "usd" / f"{VARIANT_PACKAGE}.usd")

    def _get_embodiment_joint_pos_overrides(self, cfg: DexBenchLogisticsE2L1EnvironmentCfg):
        """The robot starts in DB-L1's start pose (``l1_start_pose``), the scene shell included."""

        return l1_start_pose(cfg) or super()._get_embodiment_joint_pos_overrides(cfg)

    def apply_newton_sim_settings(self, env_cfg, cfg: DexBenchLogisticsE2L1EnvironmentCfg) -> None:
        """The cell's MJWarp options, then, with a bag in the scene, MJWarp + VBD at the bags' stepping and HB2 moving."""

        from dexbench_arena.assets.polybag.physics import apply_polybag_physics

        super().apply_newton_sim_settings(env_cfg, cfg)
        if not apply_polybag_physics(env_cfg):
            return
        solver = env_cfg.sim.physics.solver_cfg
        mjwarp_solver_cfg(env_cfg.sim.physics).ls_iterations = L1_MJWARP_LS_ITERATIONS
        speed = float(cfg.belt_speed)
        if speed:
            solver.surface_velocities = {L1_BELT_SHAPE_PATTERN: (speed, 0.0, 0.0)}
            solver.texture_scrolls = {L1_BELT_UV_SHADER_PATTERN: -L1_BELT_UV_PER_METRE * speed}
            solver.rigid_belt_velocities = {L1_BELT_SHAPE_PATTERN: (speed, 0.0, 0.0)}
        solver.particle_collision_off = l1_particle_collision_off(cfg)
        interval = l1_self_contact_detection_interval(cfg)
        solver.soft_solver_cfg.particle_collision_detection_interval = interval
        print(f"[dexbench] DB-L1: the bags' self-contact detection {cfg.polybag_self_contact_detection} a substep "
              f"(SolverVBD particle_collision_detection_interval {interval})", flush=True)
        solver.soft_solver_cfg.initial_guess = cfg.polybag_vbd_init
        print(f"[dexbench] DB-L1: the bags' VBD initial guess {cfg.polybag_vbd_init}", flush=True)
        soft = solver.soft_solver_cfg
        soft.film_hessian = cfg.polybag_film_hessian
        soft.tet_poisson = None if cfg.polybag_tet_poisson is None else float(cfg.polybag_tet_poisson)
        soft.tet_damping = None if cfg.polybag_tet_damping is None else float(cfg.polybag_tet_damping)
        print(f"[dexbench] DB-L1: the bags' film Hessian {soft.film_hessian}, tet Poisson ratio "
              f"{'as authored' if soft.tet_poisson is None else f'{soft.tet_poisson:g}'}, tet damping "
              f"{'as authored' if soft.tet_damping is None else f'{soft.tet_damping:g} Pa s'}", flush=True)

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        super().add_cli_args(parser)
        add_e2_cell_cli_args(parser)

        def add(flag: str, **kwargs) -> None:
            _add_argument_if_missing(parser, f"--{flag}", f"--{flag.replace('_', '-')}", dest=flag, **kwargs)

        add("belt_speed", type=float, default=L1_BELT_SPEED_M_S,
            help=f"HB2's surface speed in m/s, positive = +X: {L1_BELT_SPEED_M_S}, the one speed of every DB-L1 run "
                 "(no speed axis); any other value is refused.")
        add("l1_stream_mode", choices=L1_STREAM_MODES, default="episode",
            help="episode: an episode of l1_stream_bags bags and l1_stream_boxes boxes, ended on the first failure (an "
                 "evaluation); continuous: an endless stream, one demo per placed bag (teleop collection).")
        add("l1_stream_bags", type=int, default=L1_STREAM_BAGS,
            help=f"Polybags in an episode, drawn from the set's four (1-4, default {L1_STREAM_BAGS}); a continuous "
                 "stream empties the tote every that many.")
        add("l1_stream_boxes", type=int, default=L1_STREAM_BOXES,
            help=f"Boxes in an episode, drawn from the four (0-4, default {L1_STREAM_BOXES}); a continuous stream "
                 "alternates a box after every bag unless it is 0.")
        add("l1_stream_lead", choices=L1_STREAM_LEADS, default="bag",
            help="What comes first; the stream alternates after it.")
        add("l1_headline_bags", type=int, default=None,
            help="An episode succeeds once its first that many bags are placed in stream order before any failure "
                 "(DB-L1's benchmark: 2); the all-bags rule is recorded beside it as challenge_success. Default: "
                 "every bag placed and no box disturbed.")
        add("l1_bag_set", choices=tuple(L1_BAG_SETS), default=L1_DEFAULT_BAG_SET,
            help="GA-O: the run's bag set; train_a..train_d hold training bags, eval_a and eval_b held-out sizes, "
                 "eval_c and eval_d held-out colours (e2_logistics_l1_layout.L1_BAG_SETS).")
        add("l1_bag_wrist_camera_collision", choices=("on", "off"), default="on",
            help="Whether the bags collide with the palm-camera mounts on link_*_arm_6 (off: they pass through).")
        defaults = DexBenchLogisticsE2L1EnvironmentCfg
        add("l1_bag_wrist_link_collision", choices=("on", "off"), default=defaults.l1_bag_wrist_link_collision,
            help="Whether the bags collide with link_*_arm_5 and link_*_arm_6, the palm-camera mounts included "
                 "(off: they pass through; the palm and the fingers still collide).")
        train, held_out = L1_YAW_OFF_BELT_DEG["train"], L1_YAW_OFF_BELT_DEG["ga_s"]
        add("l1_bag_yaw_off_belt_deg", type=_pair_type, default=train, metavar="LOW,HIGH",
            help="GA-S: the band of angles [deg] between a bag's long side and the belt's direction; "
                 f"{train[0]:g},{train[1]:g} trains, {held_out[0]:g},{held_out[1]:g} is held out.")
        add("l1_item_y", type=_pair_type, default=L1_ITEM_Y_RANGE_M, metavar="LOW,HIGH",
            help="Where across HB2 an item's centre is put down [m, world y].")
        add("l1_spacing", type=_pair_type, default=L1_SPACING_RANGE_M, metavar="LOW,HIGH",
            help="How far the belt runs between two items' releases [m], drawn per item.")
        add("l1_stream_seed", type=int, default=0, help="Seeds each episode's (continuous stream's) draw.")
        add("l1_stream_plan", type=str, default=None,
            help="A collection plan (JSON, tasks.polybag_stream_plan) laying the episodes out in place of the draw "
                 "(episode mode); an entry runs until an episode on it succeeds.")
        add("l1_stream_plan_start", type=int, default=0, help="The plan entry to start at (0-based).")
        add("l1_miss_line_x", type=float, default=L1_MISS_LINE_X_M,
            help="An item on the belt whose leading edge reaches this x [m] is missed and taken off.")
        add("polybag_stepping", choices=tuple(L1_POLYBAG_STEPPINGS), default="a",
            help="The bags' stepping: a (8 VBD substeps x 5 iterations, MJWarp 8) or b (16 x 5, MJWarp 16).")
        add("polybag_vbd_substeps", type=int, default=None, help="Physics substeps of the bags (VBD), over the "
            "stepping candidate's.")
        add("polybag_vbd_iterations", type=int, default=defaults.polybag_vbd_iterations,
            help=f"VBD iterations per substep, over the candidate's (default {defaults.polybag_vbd_iterations}).")
        add("polybag_mjwarp_substeps", type=int, default=None,
            help="MJWarp substeps per physics step, over the candidate's; two-way coupling needs the bags' count.")
        add("polybag_full_surface_contact", choices=("on", "off"), default="on",
            help="Newton's full-surface rigid-soft contact (the bags' surface edges and faces against the colliders).")
        add("polybag_contact_barrier", choices=("on", "off"), default="off",
            help="The colliders' contact with the bags as a log barrier (on) or a linear penalty (off).")
        add("polybag_frame_truncation", choices=FRAME_TRUNCATIONS, default="init",
            help="Self-contact truncation in each bag's own frame (init, all) or SolverVBD's own clamp (off).")
        add("polybag_self_contact_detection", choices=tuple(L1_SELF_CONTACT_DETECTION), default="twice",
            help="How often VBD finds the bags' self-contacts in a substep: twice (default, Newton's) or once "
                 "(about 3 ms a step faster).")
        add("polybag_vbd_init", choices=L1_VBD_INITIAL_GUESSES, default=defaults.polybag_vbd_init,
            help="Where VBD starts each substep's iterate: contact (default, x + v dt for a bag pressed by a collider "
                 "or lying on another bag: no fake support on a resting, lifted or held bag) or inertia (Newton's "
                 "x + v dt + g dt^2).")
        add("polybag_film_hessian", choices=L1_FILM_HESSIANS, default=defaults.polybag_film_hessian,
            help="The film springs' Hessian in VBD's per-vertex step: psd (default, the compressed transverse term "
                 "clamped at >= 0, the same force) or newton (Newton's own, indefinite for a compressed spring).")
        add("polybag_tet_poisson", type=float, default=defaults.polybag_tet_poisson,
            help="Every bag tet's Poisson ratio (0 <= nu < 0.5) at its own shear modulus "
                 f"(default {defaults.polybag_tet_poisson:g}).")
        add("polybag_tet_damping", type=float, default=defaults.polybag_tet_damping,
            help=f"Every bag tet's damping in Pa s (default {defaults.polybag_tet_damping:g}).")
        add("polybag_coupling", choices=COUPLINGS, default="two_way",
            help="two_way (default): the bags' contact reactions push on the robot and the boxes; one_way: they "
                 "move the bags and feel nothing back.")
        add("polybag_youngs_modulus", type=float, default=None,
            help="The bags' Young's modulus in Pa; default the pool asset's own (per tet: 2.5 kPa, 0.5 kPa skin).")
        for side, what in (("contact", "The bags' own"), ("collider", "Every collider's, the hands' included,")):
            add(f"polybag_{side}_ke", type=float, default=3000.0, help=f"{what} contact stiffness against the bags "
                "in N/m (3000; two-way needs the bag's and the colliders' equal).")
            add(f"polybag_{side}_kd", type=float, default=1.0, help=f"{what} contact damping (1; two-way needs "
                "the two sides equal).")
        add("polybag_contact_mu", type=float, default=1.0, help="The bags' friction (1.0).")
        add("polybag_collider_mu", type=float, default=defaults.polybag_collider_mu,
            help=f"Every collider's friction against the bags (default {defaults.polybag_collider_mu:g}; the asset's "
                 "is 0.8).")

    def build(self, cfg: DexBenchLogisticsE2L1EnvironmentCfg):
        from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTermCfg

        from dexbench_arena.tasks.polybag_stream import record_stream_tracks

        l1_settings(cfg)
        environment = with_delivery_render_settings(super().build(cfg), cfg.delivery_render_settings)
        if cfg.task == TASK_NAME:
            environment.episode_recorder_terms["l1_stream_tracks"] = EpisodeRecorderTermCfg(func=record_stream_tracks)
        environment.env_cfg_callback = refusing_newton_solver_before_newton_main(
            environment.env_cfg_callback, f"{ENVIRONMENT_NAME} (DB-L1)")
        return environment

    def build_scene(self, cfg: DexBenchLogisticsE2L1EnvironmentCfg) -> Scene:
        import isaaclab.sim as sim_utils
        from isaaclab_arena.assets.object import Object
        from isaaclab_arena.assets.object_base import ObjectType
        from isaaclab_arena.utils.pose import Pose

        identity = (0.0, 0.0, 0.0, 1.0)
        scene = build_logistics_center_e2_scene(
            self.asset_registry,
            hide_room_shell=bool(cfg.teleop_session),
            spawn_l2_task_items=False,
            spawn_belt_items=False,
            moving_belts=L1_BELTS,
        )
        centre, size = l1_tote_catch_box()

        def hidden_box(name: str, size_m, centre_m) -> Object:
            return Object(
                name=name,
                object_type=ObjectType.BASE,
                spawner_cfg=sim_utils.CuboidCfg(size=size_m, collision_props=sim_utils.CollisionPropertiesCfg(),
                                                visible=False),
                initial_pose=Pose(position_xyz=centre_m, rotation_xyzw=identity),
                tags=["background"],
            )

        park_slab = hidden_box(L1_PARK_SLAB_ENTITY, L1_PARK_SLAB_SIZE_M, L1_PARK_SLAB_CENTER_M)
        assets = [hidden_box(L1_TOTE_CATCH_ENTITY, size, centre), park_slab]
        if cfg.task == TASK_NAME and cfg.l1_stream_mode == "episode":
            from dexbench_arena.variations.db_l1_stream_layout_variation import L1StreamLayoutVariation

            park_slab.add_variation(L1StreamLayoutVariation(l1_stream_layout_draw_size(l1_settings(cfg))))
        if cfg.task == TASK_NAME:
            for item in l1_box_items():
                assets.append(self.asset_registry.get_asset_by_name(BOX_VARIANTS_BY_KEY[item.variant].asset)(
                    instance_name=item.entity,
                    initial_pose=Pose(position_xyz=item.park_pose[:3], rotation_xyzw=item.park_pose[3:]),
                ))
        scene.add_assets(assets)
        return scene
