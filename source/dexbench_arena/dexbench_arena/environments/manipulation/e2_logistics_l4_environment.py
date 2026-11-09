# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""DexBench DB-L4 Arena scene: parcels on the E2 cell's conveyor, turned barcode-up."""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentFactory

from dexbench_arena import ASSETS_DIR
from dexbench_arena.assets.e2_logistics_cell import E2_L4_PARCELS_BY_NAME, polybag_size_scale
from dexbench_arena.contact_materials import WUJI_GRASP_CONTACT_MATERIAL
from dexbench_arena.embodiments.lookup import start_poses
from dexbench_arena.environments.manipulation.e2_logistics_environment import (
    DexBenchLogisticsCenterE2Environment,
    add_e2_cell_cli_args,
    build_logistics_center_e2_scene,
    with_delivery_render_settings,
)
from dexbench_arena.layouts.e2_logistics_l4_layout import (
    IDENTITY,
    L4_ANCHOR_YAWS_DEG,
    L4_BELT_UV_PER_METRE,
    L4_BOX_SPAWN_CLEARANCE_M,
    L4_PARCELS_PER_EPISODE,
    L4_PARK_SLAB_CENTER_M,
    L4_PARK_SLAB_SIZE_M,
    L4_PARK_SLAB_TOP_Z_M,
    L4_POLYBAG_SPAWN_CLEARANCE_M,
    L4_RBY1_START_JOINT_POS,
    L4_ROBOT_BASE_ANCHORS_XYZ,
    L4_SPAWN_SURFACE_Z_M,
    L4_SPAWN_XY_M,
    MASS_MULTIPLIERS,
    ROTATION_DOMAINS,
    box_bounds,
    l4_parcel_entity_name,
    l4_park_xy,
    resting_position,
    rotation_domain,
    rotation_labels,
    world_rotation,
)
from dexbench_arena.layouts.e2_logistics_l4_sequences import L4_SEQUENCE_AXIS, load_l4_sequences
from dexbench_arena.environments.manipulation.scene_environment_base import (
    DexBenchEnvironmentCfg,
    DexBenchSceneEnvironmentBase,
    _add_argument_if_missing,
)
from dexbench_arena.teleop.newton_solver import mjwarp_solver_cfg, refusing_newton_solver_before_newton_main

if TYPE_CHECKING:
    from isaaclab_arena.scene.scene import Scene

    from dexbench_arena.tasks.deliver_parcels_barcode_up import L4Parcel

TASK_NAME = "deliver_parcels_barcode_up"

L4_POLYBAG_USD = ASSETS_DIR / "object_library" / "polybag_cotton_tet_deformable" / "usd" / "polybag_cotton_tet_deformable.usd"
L4_POLYBAG_VBD_SUBSTEPS = 8
L4_POLYBAG_VBD_ITERATIONS = 5
L4_POLYBAG_MJWARP_SUBSTEPS = 8
L4_POLYBAG_FULL_SURFACE_CONTACT = True

L4_BELT_SHAPE_PATTERNS = (r"/HB2_Belt/TransportBody/TopRun$", r"/FB1_Belt/TransportBody/TopRun$")
L4_BELTS = ("HB2", "FB1")
L4_BELT_UV_SHADER_PATTERNS = (r"/Belt_HB2/UVAnim$", r"/Belt_FB1/UVAnim$")
L4_DEFAULT_BELT_SPEED_M_S = 0.45
L4_MJWARP_LS_ITERATIONS = 50
L4_PARK_SLAB_SHAPE_PATTERN = r"/l4_park_slab/"
L4_MJWARP_SOLVER = "newton"


L4_POLYBAG_MATERIAL_FLAGS = {
    "polybag_youngs_modulus": "The bags' Young's modulus in Pa (the delivery's 15000).",
    "polybag_poisson_ratio": "The bags' Poisson ratio, in [0, 0.5) (the delivery's 0.15).",
    "polybag_k_damp": "The damping of the bags' tets (the delivery's 30).",
    "polybag_contact_ke": "The bags' contact stiffness in N/m: their self-contact, and their half of a contact with a "
                          "collider, whose stiffness is the mean of the two sides' (default 3000; Newton's 1e3).",
    "polybag_contact_kd": "The bags' contact damping (default 1; Newton's 10).",
    "polybag_contact_mu": "The bags' friction, their self-contact's and their side of a collider contact (default 0.4; "
                          "Newton's 0.5); a collider contact's is the root of the product of the two sides'.",
    "polybag_collider_ke": "Every collider's contact stiffness against the bags in N/m, the hands' included "
                           "(default 3000; the delivery's 1e4).",
    "polybag_collider_kd": "Every collider's contact damping against the bags (default 1; the delivery's 10).",
    "polybag_collider_mu": "The friction against the bags of every collider without a --polybag_*_mu of its own: the "
                           "belts, the rollers, the frames (the delivery's 0.8).",
}

L4_POLYBAG_COLLIDER_GROUPS = {
    "polybag_robot_mu": ("the robot's colliders, the hands' included", WUJI_GRASP_CONTACT_MATERIAL.dynamic_friction),
    "polybag_plate_mu": ("the work plate's", 0.25),
    "polybag_box_mu": ("the box parcels'", 0.4),
}
L4_ROBOT_SHAPE_PATTERN = r"/Robot/"
L4_PLATE_SHAPE_PATTERN = r"/Plate/WorkPlate/"

L4_POLYBAG_MATERIAL_DEFAULTS = {
    "polybag_contact_ke": 3000.0,
    "polybag_contact_kd": 1.0,
    "polybag_contact_mu": 0.4,
    "polybag_collider_ke": 3000.0,
    "polybag_collider_kd": 1.0,
}


def l4_polybag_material(cfg) -> dict[str, float | None]:
    """The task's ``tet_*`` keyword arguments from the material fields; an unset one takes
    L4_POLYBAG_MATERIAL_DEFAULTS, else None (the delivery's value)."""

    values = {}
    for flag in L4_POLYBAG_MATERIAL_FLAGS:
        value = getattr(cfg, flag, None)
        values["tet_" + flag.removeprefix("polybag_")] = L4_POLYBAG_MATERIAL_DEFAULTS.get(flag) if value is None else value
    return values


def l4_polybag_collider_mu_by_path(cfg, parcels) -> tuple[tuple[str, float], ...]:
    """The task's ``tet_collider_mu_by_path``: L4_POLYBAG_COLLIDER_GROUPS' friction (the field's value, else its
    default) on the robot, the plate and `parcels`' boxes."""

    def mu(flag: str) -> float:
        value = getattr(cfg, flag, None)
        return float(L4_POLYBAG_COLLIDER_GROUPS[flag][1] if value is None else value)

    pairs = [(L4_ROBOT_SHAPE_PATTERN, mu("polybag_robot_mu")), (L4_PLATE_SHAPE_PATTERN, mu("polybag_plate_mu"))]
    boxes = [parcel.entity_name for parcel in parcels if not parcel.is_polybag]
    if boxes:
        pairs.append((r"/(" + "|".join(re.escape(name) for name in boxes) + r")/", mu("polybag_box_mu")))
    return tuple(pairs)


def l4_polybag_stepping(cfg) -> dict:
    """The bags' stepping keyword arguments: what the config asks for, the delivery's validated stepping otherwise."""

    def given(name: str, default):
        value = getattr(cfg, name, None)
        return default if value is None else value

    full_surface = given("polybag_full_surface_contact", None)
    truncation = given("polybag_frame_truncation", "init")
    return {
        "coupling_mode": given("polybag_coupling", "two_way"),
        "vbd_substeps": int(given("polybag_vbd_substeps", L4_POLYBAG_VBD_SUBSTEPS)),
        "vbd_iterations": int(given("polybag_vbd_iterations", L4_POLYBAG_VBD_ITERATIONS)),
        "mjwarp_substeps": int(given("polybag_mjwarp_substeps", L4_POLYBAG_MJWARP_SUBSTEPS)),
        "full_surface_contact": L4_POLYBAG_FULL_SURFACE_CONTACT if full_surface is None else full_surface == "on",
        "frame_truncation": None if truncation == "off" else truncation,
    }


def _rotations_text(value) -> tuple[str, ...] | None:
    """``--l4_rotations``: five labels, comma-separated on the command line or a list in a typed config."""

    if value is None:
        return None
    labels = tuple(part.strip() for part in value.split(",")) if isinstance(value, str) else tuple(value)
    return tuple(label for label in labels if label)


def l4_sequence_rotations(cfg) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    """The run's sequence and the rotation label each of its parcels is put down at; None without a sequence."""

    sequence_id = getattr(cfg, "l4_sequence", None)
    rotations = _rotations_text(getattr(cfg, "l4_rotations", None))
    draw = getattr(cfg, "l4_rotation_draw", None)
    if sequence_id is None:
        if rotations is not None or draw is not None:
            raise ValueError("--l4_rotations and --l4_rotation_draw put a sequence's parcels down; give --l4_sequence")
        return None
    sequences = load_l4_sequences()
    if sequence_id not in sequences:
        raise ValueError(f"{sequence_id!r} is not a DB-L4 sequence (eval_00..eval_11, EVAL_SEQUENCES; train_000.., "
                         "db_l4_training_sequences.json)")
    sequence = sequences[sequence_id]
    if draw is not None:
        if rotations is not None:
            raise ValueError("--l4_rotations fixes the rotations --l4_rotation_draw draws; give one")
        if draw not in ROTATION_DOMAINS:
            raise ValueError(f"--l4_rotation_draw {draw!r}: one of {ROTATION_DOMAINS}")
        return sequence.objects, tuple(rotation_labels(name, draw)[0] for name in sequence.objects)
    if rotations is not None:
        if len(rotations) != len(sequence.objects):
            raise ValueError(f"--l4_rotations gives {len(rotations)} labels for {len(sequence.objects)} parcels")
        for name, label in zip(sequence.objects, rotations):
            rotation_domain(name, label)
        return sequence.objects, rotations
    if sequence.rotations is None:
        raise ValueError(f"{sequence_id} is an evaluation sequence: draw its rotations (--l4_rotation_draw id or ood) "
                         "or give them (--l4_rotations)")
    return sequence.objects, sequence.rotations


def l4_mass_multipliers(cfg) -> tuple[float, ...]:
    """Each slot's mass multiplier over its parcel's nominal mass: ``--l4_masses`` (five, comma-separated on the
    command line or a list in a typed config), else a training sequence's own, else x1. With ``--l4_mass_draw`` x1,
    which the draw replaces at every reset. ValueError for both flags, a domain the draw does not know, a count
    other than the sequence's or a multiplier that is not a positive number."""

    value = getattr(cfg, "l4_masses", None)
    draw = getattr(cfg, "l4_mass_draw", None)
    if draw is not None:
        if value is not None:
            raise ValueError("--l4_masses fixes the masses --l4_mass_draw draws; give one")
        if draw not in MASS_MULTIPLIERS:
            raise ValueError(f"--l4_mass_draw {draw!r}: one of {tuple(MASS_MULTIPLIERS)}")
        return (1.0,) * L4_PARCELS_PER_EPISODE
    if value is None:
        sequence = load_l4_sequences().get(getattr(cfg, "l4_sequence", None) or "")
        if sequence is not None and sequence.masses is not None:
            return sequence.masses
        return (1.0,) * L4_PARCELS_PER_EPISODE
    parts = [part.strip() for part in value.split(",")] if isinstance(value, str) else list(value)
    try:
        multipliers = tuple(float(part) for part in parts if part != "")
    except (TypeError, ValueError):
        raise ValueError(f"--l4_masses {value!r}: five numbers, comma-separated") from None
    if len(multipliers) != L4_PARCELS_PER_EPISODE:
        raise ValueError(f"--l4_masses gives {len(multipliers)} multipliers for {L4_PARCELS_PER_EPISODE} parcels")
    if not all(math.isfinite(m) and m > 0.0 for m in multipliers):
        raise ValueError(f"--l4_masses {value!r}: every multiplier must be a positive number")
    return multipliers


def l4_parcel_mass_kg(name: str, multiplier: float) -> float:
    """What parcel ``name`` weighs at ``multiplier`` over its nominal mass [kg]."""

    return E2_L4_PARCELS_BY_NAME[name].nominal_mass_kg * multiplier


def l4_put_downs(cfg) -> list[tuple[str, str, tuple]]:
    """(entity, parcel name, world rotation) of every parcel of the run's sequence, in order. ValueError without
    a sequence."""

    sequence = l4_sequence_rotations(cfg)
    if sequence is None:
        raise ValueError("DB-L4 needs --l4_sequence (eval_00..eval_11, train_000..train_239)")
    return [(l4_parcel_entity_name(slot), name, world_rotation(name, label))
            for slot, (name, label) in enumerate(zip(*sequence))]


@lru_cache(maxsize=4)
def _polybag_bounds(size_scale: tuple[float, float, float] | None = None) -> tuple[tuple[float, ...], ...]:
    """The tet bag's rest bounds in its own frame at ``size_scale`` (None: the asset's). Loads the bag template,
    which needs Newton."""

    from dexbench_arena.assets.polybag.template import load_template

    rest = load_template(str(L4_POLYBAG_USD), cotton_youngs_modulus=None, cotton_poisson_ratio=None,
                         film_bend_scale=None, film_rest_bend_angles=None, tet_size_scale=size_scale).rest
    return tuple(float(v) for v in rest.min(axis=0)), tuple(float(v) for v in rest.max(axis=0))


def _bounds_and_clearance(name: str):
    row = E2_L4_PARCELS_BY_NAME[name]
    if row.is_polybag:
        return _polybag_bounds(polybag_size_scale(name)), L4_POLYBAG_SPAWN_CLEARANCE_M
    return box_bounds(row.collider_size_xyz_m), L4_BOX_SPAWN_CLEARANCE_M


def l4_stage_pose(name: str, rotation) -> tuple[float, ...]:
    """Where parcel ``name`` turned by ``rotation`` (world) is put down on HB2: (x, y, z, qx, qy, qz, qw),
    environment frame. A box's from its collider, a bag's from its rest particles."""

    import torch
    from isaaclab.utils.math import quat_from_matrix

    bounds, clearance = _bounds_and_clearance(name)
    position = resting_position(rotation, *bounds, L4_SPAWN_XY_M, L4_SPAWN_SURFACE_Z_M, clearance)
    quat = quat_from_matrix(torch.tensor(rotation, dtype=torch.float64))
    return (*(float(v) for v in position), *(float(v) for v in quat))


def l4_parcels(cfg) -> list[L4Parcel]:
    """The run's parcels, each with where it waits and where it is put down on HB2 (environment frame)."""

    from dexbench_arena.tasks.deliver_parcels_barcode_up import L4Parcel

    parcels = []
    for slot, ((entity, name, rotation), multiplier) in enumerate(zip(l4_put_downs(cfg), l4_mass_multipliers(cfg))):
        bounds, clearance = _bounds_and_clearance(name)
        park = resting_position(IDENTITY, *bounds, l4_park_xy(slot), L4_PARK_SLAB_TOP_Z_M, clearance)
        parcels.append(L4Parcel(entity, name, (*park, 0.0, 0.0, 0.0, 1.0), l4_stage_pose(name, rotation),
                                l4_parcel_mass_kg(name, multiplier)))
    return parcels


def l4_mass_variations(cfg) -> list:
    """With ``--l4_mass_draw``, one ``L4MassVariation`` per slot of the sequence; none otherwise."""

    draw = getattr(cfg, "l4_mass_draw", None)
    sequence = l4_sequence_rotations(cfg)
    if draw is None or sequence is None:
        return []
    l4_mass_multipliers(cfg)
    from dexbench_arena.variations.db_l4_mass_variation import L4MassVariation

    return [L4MassVariation(slot, l4_parcel_entity_name(slot), E2_L4_PARCELS_BY_NAME[name].nominal_mass_kg,
                            E2_L4_PARCELS_BY_NAME[name].is_polybag, MASS_MULTIPLIERS[draw])
            for slot, name in enumerate(sequence[0])]


def l4_rotation_variations(cfg) -> list:
    """With ``--l4_rotation_draw``, one ``L4RotationVariation`` per slot of the sequence, each with the stage pose
    of every label of the domain; none otherwise."""

    draw = getattr(cfg, "l4_rotation_draw", None)
    sequence = l4_sequence_rotations(cfg)
    if draw is None or sequence is None:
        return []
    from dexbench_arena.variations.db_l4_rotation_variation import L4RotationVariation

    variations = []
    for slot, name in enumerate(sequence[0]):
        labels = rotation_labels(name, draw)
        variations.append(L4RotationVariation(
            slot, labels, tuple(l4_stage_pose(name, world_rotation(name, label)) for label in labels)))
    return variations


def l4_start_pose(cfg) -> dict[str, float] | None:
    """DB-L4's start pose: the embodiment's replay_home with L4_RBY1_START_JOINT_POS's torso and arms."""

    home = start_poses(getattr(cfg, "embodiment", None)).get("replay_home")
    if not home or not set(L4_RBY1_START_JOINT_POS) <= home.keys():
        return None
    return {**home, **L4_RBY1_START_JOINT_POS}


def _make_deliver_parcels_barcode_up_task(env, cfg: DexBenchLogisticsE2L4EnvironmentCfg):
    """Factory for DB-L4. Newton only (the polybags are VBD deformables): ``build`` refuses another backend."""

    from dexbench_arena.tasks.deliver_parcels_barcode_up import DexBenchDeliverParcelsBarcodeUpTask

    parcels = l4_parcels(cfg)
    return DexBenchDeliverParcelsBarcodeUpTask(
        parcels=parcels,
        belt_speed_m_s=float(cfg.belt_speed or 0.0),
        polybag_usd=str(L4_POLYBAG_USD),
        **l4_polybag_stepping(cfg),
        robot_joint_pos_overrides=l4_start_pose(cfg),
        hand_contact_material=WUJI_GRASP_CONTACT_MATERIAL,
        end_on_parcel_failure=cfg.end_on_parcel_failure,
        **l4_polybag_material(cfg),
        tet_collider_mu_by_path=l4_polybag_collider_mu_by_path(cfg, parcels),
    )


def refuse_mesh_feature_contacts(has_mesh_feature_contacts: bool) -> None:
    """ValueError when the Newton in use has newton#4227's mesh feature contacts (the module's comment above)."""

    if has_mesh_feature_contacts:
        raise ValueError(
            f"{TASK_NAME} needs a Newton before newton#4227 (the image's 646cece4): from it on, the hands' mesh "
            "contacts with the bags overflow and stick. Run it in the image docker/build.py builds."
        )


def _requiring_newton(previous):
    """An env_cfg_callback that refuses a composed cfg on another backend than Newton, or on a Newton with
    newton#4227's mesh feature contacts (``refuse_mesh_feature_contacts``), then runs ``previous``."""

    def callback(env_cfg):
        from isaaclab_newton.physics.newton_manager_cfg import NewtonCfg

        from dexbench_arena.assets.polybag.two_way_reaction import HAS_CONTACT_FILTER

        if not isinstance(getattr(env_cfg.sim, "physics", None), NewtonCfg):
            raise ValueError(f"{TASK_NAME} simulates the polybags with Newton VBD; run it with --presets newton")
        refuse_mesh_feature_contacts(HAS_CONTACT_FILTER)
        return previous(env_cfg) if previous is not None else env_cfg

    return callback


def _sequence_id(text: str) -> str:
    """argparse type for ``--l4_sequence``: one of the sequence file's ids."""

    if text not in load_l4_sequences():
        raise argparse.ArgumentTypeError(f"{text!r} is not a DB-L4 sequence (eval_00..eval_11, train_000..train_239)")
    return text


@dataclass
class DexBenchLogisticsE2L4EnvironmentCfg(DexBenchEnvironmentCfg):
    """Typed config for DexBenchLogisticsE2L4Environment: its flags, names and defaults."""

    task: str = "none"
    """Task variant the scene builds: ``deliver_parcels_barcode_up``, or ``none`` for the scene shell."""

    anchor: str = "A_L4"
    """Robot base anchor name (A_L4)."""

    teleop_session: bool = False
    """The run is driven by an operator from inside the cell: the room shell is hidden (still solid)."""

    delivery_render_settings: bool = True
    """False renders at Arena's defaults instead of the delivery's tonemap and exposure."""

    l4_sequence: str | None = None
    """The sequence (``eval_00``.., ``train_000``..): five parcels by kind and order (e2_logistics_l4_sequences)."""

    l4_rotations: str | None = None
    """The sequence's five rotation labels, comma-separated; None: a training sequence's own."""

    l4_rotation_draw: str | None = None
    """Draw each slot's rotation at every reset from ``id`` or ``ood`` (L4RotationVariation); None: fixed."""

    l4_masses: str | None = None
    """Each slot's mass multiplier over its parcel's nominal mass, five, comma-separated; None: a training sequence's
    own, else x1."""

    l4_mass_draw: str | None = None
    """Draw each slot's mass multiplier at every reset from ``id`` or ``ood`` (L4MassVariation); None: fixed."""

    end_on_parcel_failure: bool = True
    """The first failed parcel ends the episode; False tries every parcel and ends it once the last is judged."""

    polybag_vbd_substeps: int | None = None
    """Physics substeps of the bags (VBD); None: 8."""

    polybag_full_surface_contact: str | None = None
    """Full-surface rigid-soft contact (on, off); None: on."""

    polybag_vbd_iterations: int | None = None
    """VBD iterations per substep; None: 5."""

    polybag_mjwarp_substeps: int | None = None
    """MJWarp substeps per physics step (the robot); None: 8."""

    polybag_frame_truncation: str = "init"
    """Bag-frame self-contact truncation (init, all, off)."""

    polybag_coupling: str | None = None
    """How the bags and the robot push on each other (one_way, two_way); None: two_way."""

    belt_speed: float = L4_DEFAULT_BELT_SPEED_M_S
    """HB2's and FB1's surface speed [m/s], positive = +X."""


    polybag_youngs_modulus: float | None = None
    """The bags' Young's modulus [Pa]; None: the delivery's 15 kPa."""

    polybag_poisson_ratio: float | None = None
    """The bags' Poisson ratio, in [0, 0.5); None: 0.15."""

    polybag_k_damp: float | None = None
    """The damping of the bags' tets; None: 30."""

    polybag_contact_ke: float | None = None
    """The bags' contact stiffness [N/m] (self-contact, and their half of a collider contact); None: 3000 (Newton's 1e3)."""

    polybag_contact_kd: float | None = None
    """The bags' contact damping; None: 1 (Newton's 10)."""

    polybag_contact_mu: float | None = None
    """The bags' friction; None: 0.4 (Newton's 0.5)."""

    polybag_collider_ke: float | None = None
    """Every collider's contact stiffness against the bags [N/m], the hands' included; None: 3000 (the delivery's 1e4)."""

    polybag_collider_kd: float | None = None
    """Every collider's contact damping against the bags; None: 1 (the delivery's 10)."""

    polybag_collider_mu: float | None = None
    """The friction against the bags of every collider without a group of its own below; None: 0.8."""

    polybag_robot_mu: float | None = None
    """The robot's colliders' friction against the bags, the hands' included; None: the grasp material's dynamic
    friction (5.0), the hands' against a box."""

    polybag_plate_mu: float | None = None
    """The work plate's friction against the bags; None: 0.25."""

    polybag_box_mu: float | None = None
    """The box parcels' friction against the bags; None: 0.4."""


@register_environment
class DexBenchLogisticsE2L4Environment(
    DexBenchSceneEnvironmentBase, ArenaEnvironmentFactory[DexBenchLogisticsE2L4EnvironmentCfg]
):
    """DB-L4: parcels come one at a time; turn each barcode-up and put it back on the belt."""

    name = "dexbench_logistics_e2_l4_environment"
    _legacy_argparse_cfg_type = DexBenchLogisticsE2L4EnvironmentCfg
    robot_base_anchors_xyz = L4_ROBOT_BASE_ANCHORS_XYZ
    anchor_yaws_deg = L4_ANCHOR_YAWS_DEG
    default_anchor = "A_L4"
    anchor_help = "Robot base anchor: A_L4, on the work plate's north side, facing it (provisional)."
    available_tasks = {TASK_NAME: _make_deliver_parcels_barcode_up_task}
    object_axes = {TASK_NAME: L4_SEQUENCE_AXIS}
    newton_solver_options = {**DexBenchLogisticsCenterE2Environment.newton_solver_options, "solver": L4_MJWARP_SOLVER}

    def apply_newton_sim_settings(self, env_cfg, cfg: DexBenchLogisticsE2L4EnvironmentCfg) -> None:
        """The cell's MJWarp options, then, with a bag in the scene, MJWarp + VBD at the bags' stepping and the belts moving."""

        from dexbench_arena.assets.polybag.physics import apply_polybag_physics

        super().apply_newton_sim_settings(env_cfg, cfg)
        if not apply_polybag_physics(env_cfg):
            return
        solver = env_cfg.sim.physics.solver_cfg
        mjwarp_solver_cfg(env_cfg.sim.physics).ls_iterations = L4_MJWARP_LS_ITERATIONS
        belt_speed = float(cfg.belt_speed or 0.0)
        if belt_speed:
            solver.surface_velocities = {pattern: (belt_speed, 0.0, 0.0) for pattern in L4_BELT_SHAPE_PATTERNS}
            solver.texture_scrolls = {
                pattern: -L4_BELT_UV_PER_METRE * belt_speed for pattern in L4_BELT_UV_SHADER_PATTERNS}
            solver.rigid_belt_velocities = {pattern: (belt_speed, 0.0, 0.0) for pattern in L4_BELT_SHAPE_PATTERNS}
        solver.particle_collision_off = (L4_PARK_SLAB_SHAPE_PATTERN,)

    def belt_carried_objects(self, cfg: DexBenchLogisticsE2L4EnvironmentCfg) -> dict[str, tuple[float, float, float]]:
        """The sequence's first parcel, which the relay puts down on HB2 at reset and HB2 carries toward the plate at
        ``belt_speed`` along +X (the belts' surface velocity above): a condition recording judges it at rest relative
        to HB2, so a start counts once the parcel has landed and rides. Only while it rides: at the plate's edge it
        stops in the world, so the recording ends before (``DB_L4.condition_episode_s``). A bag is no rigid object and
        is never judged; a box waiting on the park slab is not on a belt and is judged in the world. None without a
        sequence or with the belts standing."""

        belt_speed = float(cfg.belt_speed or 0.0)
        if not belt_speed or l4_sequence_rotations(cfg) is None:
            return {}
        entity, name, _ = l4_put_downs(cfg)[0]
        if E2_L4_PARCELS_BY_NAME[name].is_polybag:
            return {}
        return {entity: (belt_speed, 0.0, 0.0)}

    def object_axis_for(self, cfg):
        """The evaluation sequences' axis for a build of one; none for a training sequence, which is the training set's
        start condition, not a benchmark object (as E2's ``--object_set`` has none)."""
        axis = super().object_axis_for(cfg)
        return axis if axis is not None and getattr(cfg, "l4_sequence", None) in axis.keys() else None

    def task_object_usd_path(self, task_object) -> None:
        """No package version: a sequence is five parcels from several packages, each registered on its own."""
        del task_object
        return None

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        super().add_cli_args(parser)
        add_e2_cell_cli_args(parser)
        _add_argument_if_missing(
            parser,
            "--l4_sequence",
            "--l4-sequence",
            dest="l4_sequence",
            type=_sequence_id,
            default=None,
            help="The sequence: eval_00..eval_11 (the benchmark's) or train_000..train_239 (the training set's), "
                 "five parcels each (e2_logistics_l4_sequences.EVAL_SEQUENCES, "
                 "scripts/evaluation/experiment_configs/db_l4_training_sequences.json).",
        )
        _add_argument_if_missing(
            parser,
            "--l4_rotations",
            "--l4-rotations",
            dest="l4_rotations",
            default=None,
            help="The sequence's five rotation labels, comma-separated, e.g. up/far,near/up,... "
                 "(e2_logistics_l4_layout.rotation_labels); default a training sequence's own.",
        )
        _add_argument_if_missing(
            parser,
            "--l4_rotation_draw",
            "--l4-rotation-draw",
            dest="l4_rotation_draw",
            choices=ROTATION_DOMAINS,
            default=None,
            help="Draw each slot's rotation at every reset, from the in-domain ones (id) or the held-out ones turned "
                 "45 deg (ood), recorded as l4_park_slab.rotation_slot_<k>.",
        )
        _add_argument_if_missing(
            parser,
            "--l4_masses",
            "--l4-masses",
            dest="l4_masses",
            default=None,
            help="Each slot's mass multiplier over its parcel's nominal mass (boxes 1.0 kg, polybags 80-300 g), five, "
                 "comma-separated, e.g. 1,2,0.5,1,4; default a training sequence's own, else x1. Set when the scene "
                 "is built.",
        )
        _add_argument_if_missing(
            parser,
            "--l4_mass_draw",
            "--l4-mass-draw",
            dest="l4_mass_draw",
            choices=tuple(MASS_MULTIPLIERS),
            default=None,
            help="Draw each slot's mass multiplier at every reset, from the in-domain ones (id: x0.5-x2) or the held-out "
                 "ones (ood: x0.25, x0.35, x2.83, x4), recorded as l4_park_slab.mass_slot_<k>.",
        )
        _add_argument_if_missing(
            parser,
            "--end_on_parcel_failure",
            "--end-on-parcel-failure",
            dest="end_on_parcel_failure",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="End the episode, and reset the environment, on the first parcel that fails (barcode not up, fallen, "
                 "or not judged within 20 s); on by default. --no-end_on_parcel_failure tries all five parcels and "
                 "ends the episode once the last is judged.",
        )
        _add_argument_if_missing(
            parser, "--polybag_vbd_substeps", "--polybag-vbd-substeps", dest="polybag_vbd_substeps", type=int,
            default=None, help="Physics substeps of the bags (VBD) per 1/60 s step; default 8.",
        )
        _add_argument_if_missing(
            parser, "--polybag_full_surface_contact", "--polybag-full-surface-contact",
            dest="polybag_full_surface_contact", choices=("on", "off"), default=None,
            help="Newton's full-surface rigid-soft contact (the bags' surface edges and faces against the colliders' "
                 "SDFs, not only their particles); default on.",
        )
        _add_argument_if_missing(
            parser, "--polybag_vbd_iterations", "--polybag-vbd-iterations", dest="polybag_vbd_iterations", type=int,
            default=None, help="VBD iterations per substep; default 5.",
        )
        _add_argument_if_missing(
            parser, "--polybag_mjwarp_substeps", "--polybag-mjwarp-substeps", dest="polybag_mjwarp_substeps",
            type=int, default=None,
            help="MJWarp substeps per physics step, dividing the bags' substeps; default 8 (L2's 1/480 s). Fewer "
                 "steps the robot in every few of the bags' substeps, the bags reading its bodies interpolated between.",
        )
        _add_argument_if_missing(
            parser, "--polybag_frame_truncation", "--polybag-frame-truncation", dest="polybag_frame_truncation",
            choices=("init", "all", "off"), default="init",
            help="Self-contact truncation in each bag's own frame instead of SolverVBD's isotropic DAT clamp, on "
                 "each substep's initial guess (init), on every call (all), or off (the clamp).",
        )
        _add_argument_if_missing(
            parser, "--polybag_coupling", "--polybag-coupling", dest="polybag_coupling",
            choices=("one_way", "two_way"), default=None,
            help="two_way (default): the bags' contact reactions push on the robot and the rigid parcels; one_way: they "
                 "move the bags and feel nothing back.",
        )
        for flag, help_text in L4_POLYBAG_MATERIAL_FLAGS.items():
            _add_argument_if_missing(
                parser, f"--{flag}", f"--{flag.replace('_', '-')}", dest=flag, type=float, default=None, help=help_text
            )
        for flag, (covers, default) in L4_POLYBAG_COLLIDER_GROUPS.items():
            _add_argument_if_missing(
                parser, f"--{flag}", f"--{flag.replace('_', '-')}", dest=flag, type=float, default=None,
                help=f"The friction against the bags of {covers} colliders, in place of --polybag_collider_mu "
                     f"(default {default}).",
            )
        _add_argument_if_missing(
            parser, "--belt_speed", "--belt-speed", dest="belt_speed", type=float, default=L4_DEFAULT_BELT_SPEED_M_S,
            help="HB2's and FB1's surface speed in m/s, positive = +X (the source station's 0.45 by default).",
        )

    def build(self, cfg: DexBenchLogisticsE2L4EnvironmentCfg):
        environment = with_delivery_render_settings(super().build(cfg), cfg.delivery_render_settings)
        if cfg.task == TASK_NAME:
            environment.env_cfg_callback = refusing_newton_solver_before_newton_main(
                _requiring_newton(environment.env_cfg_callback), TASK_NAME)
        return environment

    def build_scene(self, cfg: DexBenchLogisticsE2L4EnvironmentCfg) -> Scene:
        import isaaclab.sim as sim_utils
        from isaaclab_arena.assets.object import Object
        from isaaclab_arena.assets.object_base import ObjectType
        from isaaclab_arena.utils.pose import Pose

        scene = build_logistics_center_e2_scene(
            self.asset_registry,
            hide_room_shell=bool(cfg.teleop_session),
            spawn_l2_task_items=False,
            spawn_belt_items=False,
            moving_belts=L4_BELTS,
        )
        park_slab = Object(
            name="l4_park_slab",
            object_type=ObjectType.BASE,
            spawner_cfg=sim_utils.CuboidCfg(
                size=L4_PARK_SLAB_SIZE_M, collision_props=sim_utils.CollisionPropertiesCfg(), visible=False
            ),
            initial_pose=Pose(position_xyz=L4_PARK_SLAB_CENTER_M, rotation_xyzw=(0.0, 0.0, 0.0, 1.0)),
            tags=["background"],
        )
        assets = [park_slab]
        if cfg.l4_sequence is not None:
            for slot, ((entity, name, _rotation), multiplier) in enumerate(
                    zip(l4_put_downs(cfg), l4_mass_multipliers(cfg))):
                row = E2_L4_PARCELS_BY_NAME[name]
                if row.is_polybag:
                    continue
                park = resting_position(IDENTITY, *box_bounds(row.collider_size_xyz_m), l4_park_xy(slot),
                                        L4_PARK_SLAB_TOP_Z_M, L4_BOX_SPAWN_CLEARANCE_M)
                box = self.asset_registry.get_asset_by_name(row.asset)(
                    instance_name=entity,
                    initial_pose=Pose(position_xyz=park, rotation_xyzw=(0.0, 0.0, 0.0, 1.0)),
                )
                box.object_cfg.spawn.mass_props = sim_utils.MassPropertiesCfg(mass=l4_parcel_mass_kg(name, multiplier))
                assets.append(box)
        if cfg.task == TASK_NAME:
            for variation in (*l4_rotation_variations(cfg), *l4_mass_variations(cfg)):
                park_slab.add_variation(variation)
        scene.add_assets(assets)
        return scene
