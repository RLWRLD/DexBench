# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""DB-M2: insert a harness connector into its mate on the LG M-Next3 refrigerator liner, in the E3 manufacturing cell, on Newton."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import (
    ArenaEnvironmentFactory,
)

from dexbench_arena.embodiments.lookup import embodiment_class, start_poses
from dexbench_arena.layouts.db_m2_layout import (
    ANCHOR_FACING_TARGETS_XY,
    CONNECTOR_SITES,
    DEFAULT_CONNECTOR,
    DEFAULT_START,
    EPISODE_TIMEOUT_S,
    INTERIOR_LIGHT_XYZ,
    M2_CONNECTOR_OBJECT_AXIS,
    LINE_CENTER_XYZ,
    LINE_SIZE_M,
    LINER_NAME,
    LINER_ROOT_XYZ,
    GUIDE_COLLIDERS,
    MATE_COLLIDERS,
    PLUG_COLLIDERS,
    PLUG_COLLISION_VARIANTS,
    MATE_FRICTION,
    NEIGHBOR_MAX_DISPLACEMENT_M,
    NEIGHBOR_PLUG_NAME,
    PLUG_DROP_M,
    PLUG_NAME,
    PROGRESS_ALIGN_MAX_LATERAL_ERROR_M,
    PROGRESS_ALIGN_MAX_TILT_DEG,
    PROGRESS_LIFT_M,
    PROGRESS_PREINSERT_DEPTH_M,
    PROGRESS_PREINSERT_LATERAL_M,
    PROGRESS_PREINSERT_TILT_DEG,
    ROBOT_BASE_ANCHORS_XYZ,
    SEAT_HOLD_S,
    SEAT_MAX_DEPTH_M,
    SEAT_MAX_LATERAL_M,
    SEAT_MAX_TILT_DEG,
    START_MODES,
    TAPE_RELEASE,
    TAPE_REST_FROM_SEAT_M,
    TAPE_SPOT_LIMIT,
    TAPE_SPOT_RANGE,
    TAPE_SPOT_VARIATION,
    BoxPoseDomain,
    other_connector,
    site_in_env,
    start_offset,
)
from dexbench_arena.environments.manipulation.newton_rolling_friction import (
    ROLLING_CONDIM,
    install_rolling_friction,
)
from dexbench_arena.environments.manipulation.newton_shape_contact import (
    ShapeContactRequest,
    install_shape_contact,
)
from dexbench_arena.environments.manipulation.scene_environment_base import (
    DexBenchEnvironmentCfg,
    DexBenchSceneEnvironmentBase,
)
from dexbench_arena.evaluation.evaluation_signature import NEWTON_SHAPE_OVERRIDES_ATTRIBUTE

if TYPE_CHECKING:
    from isaaclab_arena.scene.scene import Scene


NEWTON_PROFILES = ("common_s16", "package")
"""``m2_newton_profile``. ``common_s16`` (the default) is the common Newton settings at 16 substeps, which DB-M2
records and evaluates with: :data:`COMMON_S16_SOLVER`,
the default shape contact :data:`COMMON_S16_DEFAULT_KE_KD`, the Wuji2 hands' own contact and rolling friction on
S2's colliders. ``package`` keeps :attr:`DexBenchManufacturingE3M2Environment.newton_solver_options` alone (8
substeps, the package's contact); give it ``--m2_mate_friction`` ``PACKAGE_MATE_FRICTION``
and ``--m2_plug_collision_variant none`` for the scene that goes with that step."""
DEFAULT_NEWTON_PROFILE = "common_s16"
COMMON_S16_SOLVER = {"num_substeps": 16, "solver": "newton", "iterations": 100, "ls_iterations": 50}
"""MJWarp's step: 16 substeps (1.04 ms at 60 Hz), the newton solver at 100 iterations and 50 line-search ones."""
COMMON_S16_COLLISION_DECIMATION = 1
COMMON_S16_DEFAULT_KE_KD = (625.0, 100.0)
"""Newton shape (ke, kd) of every shape without its own contact: MuJoCo solref (0.02, 2), as DB-M5's ``M5_SOLREF``
(ke = 1 / (timeconst * dampratio)^2, kd = 2 / timeconst)."""
M2_GA_INSTRUCTION = ("Pick up the connector, insert it into its matching socket, press it fully into place, and release "
                     "it.")
"""DB-M2's instruction from the tape start: the task the GA set records (its ``tasks.jsonl``)."""
HAND_CONTACT_NAME = "db_m2_wuji2_hand_contact"
HAND_CONTACT_KE_KD = (2500.0, 100.0)
"""The hands keep Newton's own contact, MuJoCo's (0.02, 1), against the softer default (solmix 1: a pair averages).
Every colliding shape of a hand gets it, by its body's label: the embodiment's ``hand_body_label_pattern``."""
HAND_CONTACT_SOLMIX = 1.0
PLUG_ROLLING_FRICTION_NAME = "db_m2_s2_rolling_friction"
PLUG_ROLLING_FRICTION_PATTERN = r"/Task/Connector_S2/Collision/"
PLUG_ROLLING_FRICTION_M = 2e-4
"""Rolling friction [m] with condim 6 on S2's colliders (``newton_rolling_friction``); torsional friction stays."""


def apply_m2_newton_profile(env_cfg, cfg) -> str:
    """Set ``cfg.m2_newton_profile`` on a Newton env cfg; returns the profile."""

    profile = getattr(cfg, "m2_newton_profile", DEFAULT_NEWTON_PROFILE)
    if profile not in NEWTON_PROFILES:
        raise ValueError(f"m2_newton_profile {profile!r}: expected one of {NEWTON_PROFILES}")
    if profile == "package":
        if getattr(env_cfg, NEWTON_SHAPE_OVERRIDES_ATTRIBUTE, None) is not None:
            delattr(env_cfg, NEWTON_SHAPE_OVERRIDES_ATTRIBUTE)
        return profile
    physics = env_cfg.sim.physics
    physics.num_substeps = COMMON_S16_SOLVER["num_substeps"]
    for key in ("solver", "iterations", "ls_iterations"):
        setattr(physics.solver_cfg, key, COMMON_S16_SOLVER[key])
    physics.collision_decimation = COMMON_S16_COLLISION_DECIMATION
    physics.default_shape_cfg.ke, physics.default_shape_cfg.kd = COMMON_S16_DEFAULT_KE_KD
    recorded = {"shape_contact": {}, "rolling_friction": {}}
    hand_pattern = getattr(embodiment_class(getattr(cfg, "embodiment", None)), "hand_body_label_pattern", None)
    if hand_pattern is not None:
        ke, kd = HAND_CONTACT_KE_KD
        install_shape_contact(ShapeContactRequest(HAND_CONTACT_NAME, hand_pattern, ke, kd, HAND_CONTACT_SOLMIX,
                                                  match="body"))
        recorded["shape_contact"][HAND_CONTACT_NAME] = {"pattern": hand_pattern, "match": "body", "ke": ke, "kd": kd,
                                                        "solmix": HAND_CONTACT_SOLMIX}
    install_rolling_friction(PLUG_ROLLING_FRICTION_NAME, PLUG_ROLLING_FRICTION_PATTERN, PLUG_ROLLING_FRICTION_M)
    recorded["rolling_friction"][PLUG_ROLLING_FRICTION_NAME] = {
        "pattern": PLUG_ROLLING_FRICTION_PATTERN, "mu_rolling": PLUG_ROLLING_FRICTION_M, "condim": ROLLING_CONDIM}
    setattr(env_cfg, NEWTON_SHAPE_OVERRIDES_ATTRIBUTE, recorded)
    print(
        f"[dexbench] DB-M2 Newton profile {profile}: {COMMON_S16_SOLVER}, collision decimation "
        f"{COMMON_S16_COLLISION_DECIMATION}, default ke/kd {COMMON_S16_DEFAULT_KE_KD}, S2 rolling friction "
        f"{PLUG_ROLLING_FRICTION_M} m (condim 6)",
        flush=True,
    )
    return profile


def plug_collision_variant(cfg, key: str) -> str | None:
    """The collision variant the task plug spawns with: ``default`` is the connector's
    (``PLUG_COLLISION_VARIANTS``), ``none`` or None the package's hulls, anything else a variant name or ``.json``
    path (``connector_collision_variant.collision_variant_path``)."""

    variant = getattr(cfg, "m2_plug_collision_variant", "default")
    if variant is None or str(variant).lower() == "none":
        return None
    if variant == "default":
        return PLUG_COLLISION_VARIANTS.get(key)
    return str(variant)


@dataclass
class DexBenchManufacturingE3M2EnvironmentCfg(DexBenchEnvironmentCfg):
    """Typed config for DB-M2 in the E3 cell: its flags, names and defaults."""

    task: str = "none"
    """Task variant the scene builds: ``insert_connector``, or ``none`` for the scene with both plugs mated."""
    rby1_control_hz: int | None = None
    """Explicit execution/recording rate: 30 Hz, physics-IK recordings included."""
    m2_scripted_physics_ik: bool = False
    """Install the optional arm IK refresh between physics steps, retaining 30 Hz action/recording."""
    anchor: str = "A_m2"
    """Robot base anchor name."""
    m2_connector: str = DEFAULT_CONNECTOR
    """The connector the task inserts (``S1`` or ``S2``); the other stays mated."""
    m2_start: str = DEFAULT_START
    """``tape`` (the plug standing on the liner face beside its mate: pick, carry onto the axis, insert) or ``entry``
    (the plug in its mate's entry: push it home). See ``db_m2_layout.START_MODES``."""
    m2_mate_friction: float | None = MATE_FRICTION
    """Friction bound at spawn on both connectors' mates and S2's guide (``MATE_COLLIDERS``, ``GUIDE_COLLIDERS``,
    ``db_m2_layout.MATE_FRICTION``); None or 0 keeps the package's (and the gravity seating that comes with it)."""
    m2_guide_friction: float | None = None
    """Optional guide-only friction; None retains the mate friction on the guide."""
    m2_plug_collision_variant: str | None = "default"
    """The task plug's collision hulls (``plug_collision_variant``): ``default`` (S2 chamfered,
    ``PLUG_COLLISION_VARIANTS``; S1 the package's), ``none``, or an S2 variant name or JSON path. The USD file is
    never changed; the live stage is."""
    m2_newton_profile: str = DEFAULT_NEWTON_PROFILE
    """Newton step and contact (``NEWTON_PROFILES``): ``common_s16`` (the training data's) or ``package``."""
    m2_plug_condim: int | None = 4
    """``mjc:condim`` authored at spawn on both plugs' colliders: 4 (the default) gives every plug contact torsional
    friction, so a fingertip pinch holds the plug's orientation (with 3, point contacts, a lifted plug swings about the
    pad-to-pad line like a pendulum); the mating is unchanged. None keeps the package's 3."""
    m2_plug_torsional: float | None = None
    """Torsional friction [m] on both plugs' colliders (a copy of their material; needs ``m2_plug_condim`` 4): a pinch's
    torque capacity about its pad-to-pad line is this x the grip, per pad. None keeps Newton's default (0.005)."""
    m2_fingertip_condim: int | None = None
    """Opt-in pad contact dimension through the embodiment capability. Use 4 with plug condim 3 to give only
    the soft fingertip pads torsional contact; None preserves the embodiment's authored pad contacts."""
    m2_wire_stiffness: float | None = None
    """Stiffness [N m/rad] of an implicit drive holding both plugs' wires in their authored shape (damping a tenth
    of it); None leaves the package's near-free chain."""
    m2_wire_length_m: float | None = 0.18
    """Length of the selected connector's anchored cable [m], 0.16-0.4; None or zero keeps the package's free tail.
    The rest chain is built at the configured start pose."""
    m2_wire_collision: str = "none"
    """Cable contact: 'none' preserves the source model; 'panel' enables only cable/panel-and-walls contact."""
    m2_wire_joint_armature: float = 1e-6
    """Additional cable joint inertia [kg m²], applied only with cable collision to stabilize the light links."""
    m2_wire_rest_spot: tuple[float, float, float] | None = None
    """The [dx m, dy m, yaw rad] tape spot the cable's rest chain is authored at; None: the tape release itself."""
    m2_wire_reset_shape: bool = True
    """At each reset, set the anchored cable's joints and their drive targets to the shape that reaches its anchor
    from the reset's tape spot (``_CableStartShape``), as a build at that spot would author it. False keeps the authored
    shape, which a spot other than ``m2_wire_rest_spot`` starts pulled off its anchor and which the chain then
    relaxes back to."""
    m2_start_pull_out_mm: float | None = None
    """How far along its axis from the seat the plug is released at reset [mm]; None for the connector's
    (``START_PULL_OUT_M``: S2 its Unplugged frame, 11.5). It settles in the mate's entry. The ``entry``
    start only."""


class _CableStartShape:
    """Sets the anchored cable, at each reset, to the shape that reaches its anchor from the tape spot."""

    def __init__(self, key: str, plug_pose: tuple[tuple[float, ...], tuple[float, ...]], spot):
        self.key = key
        self.plug_pose = (tuple(float(v) for v in plug_pose[0]), tuple(float(v) for v in plug_pose[1]))
        self.spot = tuple(float(value) for value in spot)
        self.shape = None
        self._joint_ids = None

    def __repr__(self) -> str:
        return f"_CableStartShape({self.key!r}, spot={list(self.spot)})"

    def __call__(self, env, env_ids, sample) -> None:
        import torch
        from isaaclab.sim import get_current_stage

        from dexbench_arena.assets.connector_harness_shape import HarnessStartShape

        plug = env.scene[PLUG_NAME]
        if self.shape is None:
            chain = f"{env.scene.env_prim_paths[0]}/{LINER_NAME}/Task/Wires/{self.key}_0"
            self.shape = HarnessStartShape.from_stage(get_current_stage(), chain, self.plug_pose, self.spot)
            names = [f"J_{joint:02d}:{axis}" for joint in range(self.shape.joints) for axis in range(3)]
            self._joint_ids, _ = plug.find_joints(names, preserve_order=True)
        position = torch.stack([self.shape.joint_coordinates(spot).reshape(-1)
                                for spot in sample.detach().cpu().double()]).to(device=plug.device, dtype=torch.float32)
        plug.write_joint_state_to_sim_index(position=position, velocity=torch.zeros_like(position),
                                            joint_ids=self._joint_ids, env_ids=env_ids)
        plug.actuators.target_command.set_position_index(value=position, joint_ids=self._joint_ids, env_ids=env_ids)


@register_environment
class DexBenchManufacturingE3M2Environment(
    DexBenchSceneEnvironmentBase,
    ArenaEnvironmentFactory[DexBenchManufacturingE3M2EnvironmentCfg],
):
    """DB-M2: the E3 cell staged for the harness connector insertion on the refrigerator liner."""

    name = "dexbench_manufacturing_e3_m2_environment"
    _legacy_argparse_cfg_type = DexBenchManufacturingE3M2EnvironmentCfg
    robot_base_anchors_xyz = ROBOT_BASE_ANCHORS_XYZ
    anchor_facing_targets_xy = ANCHOR_FACING_TARGETS_XY
    default_anchor = "A_m2"
    anchor_help = "Robot base anchor: A_m2, the liner's robot side."
    newton_solver_options: ClassVar[dict] = {"num_substeps": 8, "nconmax": 6000, "njmax": 6000}

    def apply_newton_sim_settings(self, env_cfg, cfg: DexBenchManufacturingE3M2EnvironmentCfg) -> None:
        """:attr:`newton_solver_options`, then the ``m2_newton_profile`` (``apply_m2_newton_profile``)."""
        super().apply_newton_sim_settings(env_cfg, cfg)
        apply_m2_newton_profile(env_cfg, cfg)

    @staticmethod
    def _start_mode(cfg: DexBenchManufacturingE3M2EnvironmentCfg, key: str) -> str:
        mode = getattr(cfg, "m2_start", DEFAULT_START)
        if mode not in START_MODES:
            raise ValueError(f"--m2_start {mode!r}: expected one of {START_MODES}")
        if mode == "tape" and key not in TAPE_RELEASE:
            raise ValueError(f"--m2_start tape has no tape spot for {key} yet (only {tuple(TAPE_RELEASE)}); use --m2_start entry")
        return mode

    @staticmethod
    def _connector_key(cfg: DexBenchManufacturingE3M2EnvironmentCfg) -> str:
        """The task connector (``M2_CONNECTOR_OBJECT_AXIS``; refuses one the package does not model)."""
        return M2_CONNECTOR_OBJECT_AXIS.selected(cfg).key

    def _make_insert_connector_task(self, cfg: DexBenchManufacturingE3M2EnvironmentCfg):
        from dexbench_arena.tasks.insert_connector import DexBenchInsertConnectorTask

        site = CONNECTOR_SITES[self._connector_key(cfg)]
        neighbor = CONNECTOR_SITES[other_connector(site.key)]
        seat = site_in_env(site.seated_xyz)
        start = self._start_mode(cfg, site.key)
        lift_z = seat[2] + TAPE_REST_FROM_SEAT_M[site.key][2] + PROGRESS_LIFT_M if start == "tape" else None
        preinsert = ((PROGRESS_PREINSERT_DEPTH_M, PROGRESS_PREINSERT_LATERAL_M, PROGRESS_PREINSERT_TILT_DEG)
                     if start == "tape" else None)
        return DexBenchInsertConnectorTask(
            plug_name=PLUG_NAME,
            plug_prim_path=f"{{ENV_REGEX_NS}}/{LINER_NAME}/{site.plug_prim}",
            neighbor_plug_name=NEIGHBOR_PLUG_NAME,
            seat_xyz=seat,
            pull_out_axis=site.pull_out_axis,
            seated_quat_xyzw=site.seated_quat_xyzw,
            stroke_m=site.stroke_m,
            neighbor_seat_xyz=site_in_env(neighbor.seated_xyz),
            max_depth_m=SEAT_MAX_DEPTH_M,
            max_lateral_m=SEAT_MAX_LATERAL_M,
            max_tilt_deg=SEAT_MAX_TILT_DEG,
            hold_time_s=SEAT_HOLD_S,
            dropped_below_z_m=seat[2] - PLUG_DROP_M,
            neighbor_max_displacement_m=NEIGHBOR_MAX_DISPLACEMENT_M,
            progress_align_max_lateral_error_m=PROGRESS_ALIGN_MAX_LATERAL_ERROR_M,
            progress_align_max_tilt_deg=PROGRESS_ALIGN_MAX_TILT_DEG,
            progress_lift_z_m=lift_z,
            progress_preinsert=preinsert,
            episode_length_s=EPISODE_TIMEOUT_S,
            task_description=(M2_GA_INSTRUCTION if start == "tape"
                              else "Push the harness connector into its mate on the liner until it is seated."),
            robot_joint_pos_overrides=self._m2_start_pose(cfg),
        )

    def build(self, cfg: DexBenchManufacturingE3M2EnvironmentCfg):
        """The base build, plus the task's observations in every recording (``hand_plug_force``, ``plug_insertion``):
        the teleop dataset's QA (``scripts/imitation_learning/check_db_m2_demos.py``) reads them. The variation draws are declared
        on the recorders too (``with_declared_variation_samples``): the collection checks each capture's
        ``variations/plug/tape_spot`` against the row it ran."""
        from dexbench_arena.tasks.task_obs_recorder import with_task_obs_recording
        from dexbench_arena.variations.variation_samples_recorder import with_declared_variation_samples

        environment = super().build(cfg)
        if cfg.m2_fingertip_condim is not None:
            configure = getattr(environment.embodiment, "configure_fingertip_contact_dim", None)
            if configure is None:
                raise ValueError("This embodiment does not support m2_fingertip_condim")
            configure(cfg.m2_fingertip_condim)
        environment.env_cfg_callback = with_task_obs_recording(environment.env_cfg_callback)
        environment.env_cfg_callback = with_declared_variation_samples(environment.env_cfg_callback)
        if cfg.m2_scripted_physics_ik:
            from dexbench_arena.environments.manipulation.db_m2_physics_ik_action import with_physics_ik_action
            environment.env_cfg_callback = with_physics_ik_action(environment.env_cfg_callback)
        return environment

    @staticmethod
    def _m2_start_pose(cfg: DexBenchManufacturingE3M2EnvironmentCfg):
        """The embodiment's ``teleop_ready``: hands raised and retracted to chest height, head pitched 0.9 rad."""
        return start_poses(cfg.embodiment).get("teleop_ready")

    available_tasks: ClassVar[dict] = {"insert_connector": _make_insert_connector_task}
    object_axes: ClassVar[dict] = {"insert_connector": M2_CONNECTOR_OBJECT_AXIS}

    def build_scene(self, cfg: DexBenchManufacturingE3M2EnvironmentCfg) -> Scene:
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.utils.pose import Pose

        from dexbench_arena.assets import background_library as _background_library  # noqa: F401
        from dexbench_arena.assets import object_library as _object_library  # noqa: F401
        from dexbench_arena.assets.connector_plug import ConnectorPlugReference

        background = self.asset_registry.get_asset_by_name("dexbench_assembly_line_e3_m1")()
        light = self.asset_registry.get_asset_by_name("dexbench_tube_light")(
            initial_pose=Pose(position_xyz=INTERIOR_LIGHT_XYZ)
        )
        line = self.asset_registry.get_asset_by_name("dexbench_table")(
            prim_path="LinerLine",
            size=LINE_SIZE_M,
            pos=LINE_CENTER_XYZ,
            color=(0.25, 0.25, 0.27),
        )
        liner = self.asset_registry.get_asset_by_name("lg_mnext3_liner_connector_task")(
            instance_name=LINER_NAME, initial_pose=Pose(position_xyz=LINER_ROOT_XYZ)
        )
        mate_friction = getattr(cfg, "m2_mate_friction", MATE_FRICTION)
        guide_friction = cfg.m2_guide_friction
        if guide_friction is not None and (not math.isfinite(guide_friction) or guide_friction <= 0):
            raise ValueError("m2_guide_friction must be finite and positive, or None")
        collision_variant = plug_collision_variant(cfg, self._connector_key(cfg))
        plug_condim = getattr(cfg, "m2_plug_condim", None)
        plug_torsional = getattr(cfg, "m2_plug_torsional", None)
        harness_length = cfg.m2_wire_length_m
        harness_collision = cfg.m2_wire_collision
        if harness_collision not in ("none", "panel"):
            raise ValueError("m2_wire_collision must be 'none' or 'panel'")
        if not math.isfinite(cfg.m2_wire_joint_armature) or cfg.m2_wire_joint_armature < 0.0:
            raise ValueError("m2_wire_joint_armature must be finite and nonnegative")
        if harness_length and not 0.16 <= harness_length <= 0.4:
            raise ValueError("m2_wire_length_m must be 0.16-0.4 m, or zero/None for the original free tail")
        needs_spawner = (collision_variant is not None or mate_friction or guide_friction is not None or plug_condim is not None or plug_torsional is not None
                         or harness_length or harness_collision != "none" or cfg.m2_wire_rest_spot is not None)
        if needs_spawner:
            from dexbench_arena.assets.connector_mate_friction import mate_friction_spawner

            colliders = [path for key in sorted(MATE_COLLIDERS) for path in MATE_COLLIDERS[key]]
            colliders += [path for key in sorted(GUIDE_COLLIDERS) for path in GUIDE_COLLIDERS[key]]
            plugs = [path for key in sorted(PLUG_COLLIDERS) for path in PLUG_COLLIDERS[key]]
        site = CONNECTOR_SITES[self._connector_key(cfg)]
        neighbor = CONNECTOR_SITES[other_connector(site.key)]
        pull_out_m = None if cfg.m2_start_pull_out_mm is None else float(cfg.m2_start_pull_out_mm) * 1e-3
        if pull_out_m is not None and not 0.0 <= pull_out_m < 0.1:
            raise ValueError(f"--m2_start_pull_out_mm {cfg.m2_start_pull_out_mm}: expected 0 to 100 mm")
        released = getattr(cfg, "task", None) == "insert_connector"
        offset, rotation = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)
        tape = released and self._start_mode(cfg, site.key) == "tape"
        reset_shape = tape and bool(harness_length) and cfg.m2_wire_reset_shape
        if tape:
            offset, rotation = TAPE_RELEASE[site.key]
        elif released:
            offset = start_offset(site, pull_out_m)
        plug = ConnectorPlugReference(
            name=PLUG_NAME,
            prim_path=f"{{ENV_REGEX_NS}}/{LINER_NAME}/{site.plug_prim}",
            parent_asset=liner,
            start_offset_xyz=offset,
            start_rotation_xyzw=rotation,
            wire_drive=self._wire_drive(cfg),
            wire_actuator=reset_shape,
        )
        start_shape = None
        if needs_spawner:
            initial = plug.get_initial_pose()
            if cfg.m2_wire_rest_spot is not None:
                spot = cfg.m2_wire_rest_spot
                if (not tape or not harness_length or len(spot) != 3
                        or any(not math.isfinite(v) or not low <= v <= high
                               for v, low, high in zip(spot, *TAPE_SPOT_LIMIT[site.key]))):
                    raise ValueError("m2_wire_rest_spot requires an extended tape-start cable and an admissible dx/dy/yaw")
                from isaaclab_arena.utils.yaw import rotate_quat_by_yaw

                initial = Pose(position_xyz=(initial.position_xyz[0] + spot[0],
                                             initial.position_xyz[1] + spot[1], initial.position_xyz[2]),
                               rotation_xyzw=rotate_quat_by_yaw(initial.rotation_xyzw, spot[2]))
            if reset_shape:
                start_shape = _CableStartShape(site.key, (initial.position_xyz, initial.rotation_xyzw),
                                               cfg.m2_wire_rest_spot or (0.0, 0.0, 0.0))
            liner.object_cfg.spawn.func = mate_friction_spawner(
                colliders, float(mate_friction) if mate_friction else None, plugs,
                None if plug_condim is None else int(plug_condim), None if plug_torsional is None else float(plug_torsional),
                harness_length_m=float(harness_length) if harness_length else None, harness_key=site.key,
                harness_pose=(initial.position_xyz, initial.rotation_xyzw), harness_collision=harness_collision,
                harness_joint_armature=cfg.m2_wire_joint_armature,
                guide_colliders=[path for key in sorted(GUIDE_COLLIDERS) for path in GUIDE_COLLIDERS[key]],
                guide_friction=guide_friction, collision_variant=collision_variant)
        if tape:
            self._attach_start_state(plug, site.key, start_shape)
        neighbor_plug = ConnectorPlugReference(
            name=NEIGHBOR_PLUG_NAME,
            prim_path=f"{{ENV_REGEX_NS}}/{LINER_NAME}/{neighbor.plug_prim}",
            parent_asset=liner,
            wire_drive=self._wire_drive(cfg),
        )
        return Scene(assets=[background, light, line, liner, plug, neighbor_plug])

    @staticmethod
    def _wire_drive(cfg) -> tuple[float, float] | None:
        stiffness = getattr(cfg, "m2_wire_stiffness", None)
        return None if not stiffness else (float(stiffness), 0.1 * float(stiffness))

    @staticmethod
    def _attach_start_state(plug, key: str, cable: _CableStartShape | None = None) -> None:
        """Draw the tape spot per episode; Arena records and replays the ``[dx, dy, dyaw]`` draw."""
        from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg

        from dexbench_arena.variations.object_pose_variation import ObjectPoseVariation, ObjectPoseVariationCfg

        low, high = TAPE_SPOT_RANGE[key]
        plug.add_variation(
            ObjectPoseVariation(
                PLUG_NAME,
                name=TAPE_SPOT_VARIATION,
                cfg=ObjectPoseVariationCfg(enabled=True, sampler_cfg=UniformSamplerCfg(low=list(low), high=list(high))),
                domain=BoxPoseDomain(*TAPE_SPOT_LIMIT[key], reason=f"{key} settles upright on the bracket ledge only there"),
                on_applied=cable,
            )
        )

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        super().add_cli_args(parser)
        if "--m2_scripted_physics_ik" not in parser._option_string_actions:
            parser.add_argument("--m2_scripted_physics_ik", action=argparse.BooleanOptionalAction, default=False,
                                help="Refresh the arm IK between physics steps while recording at 30 Hz.")
        if "--m2_connector" not in parser._option_string_actions:
            parser.add_argument(
                "--m2_connector",
                choices=M2_CONNECTOR_OBJECT_AXIS.keys(),
                default=M2_CONNECTOR_OBJECT_AXIS.default,
                help="The connector the task inserts (S1 or S2); the other stays mated.",
            )
        if "--m2_start" not in parser._option_string_actions:
            parser.add_argument(
                "--m2_start",
                choices=START_MODES,
                default=DEFAULT_START,
                help="tape: the plug standing on the liner face beside its mate (pick, carry onto the axis, insert); "
                "entry: the plug in its mate's entry (S2 on its Unplugged frame; push it home).",
            )
        if "--m2_mate_friction" not in parser._option_string_actions:
            parser.add_argument(
                "--m2_mate_friction",
                type=float,
                default=MATE_FRICTION,
                help="Friction bound on the connector mates at spawn (a proxy for terminal spring and latch); 0 keeps PBT.",
            )
        if "--m2_guide_friction" not in parser._option_string_actions:
            parser.add_argument("--m2_guide_friction", type=float, default=None,
                                help="Guide-only friction override; None retains mate friction on the guide.")
        if "--m2_plug_collision_variant" not in parser._option_string_actions:
            parser.add_argument("--m2_plug_collision_variant", default="default",
                                help="The task plug's collision hulls: default (S2 chamfered, S1 the package's), none, "
                                "or an S2 variant name or JSON path.")
        if "--m2_newton_profile" not in parser._option_string_actions:
            parser.add_argument("--m2_newton_profile", choices=NEWTON_PROFILES, default=DEFAULT_NEWTON_PROFILE,
                                help="Newton step and contact: common_s16 (16 substeps, the common contact; the "
                                "training data's) or package (8 substeps, the package's contact).")
        if "--m2_plug_condim" not in parser._option_string_actions:
            parser.add_argument("--m2_plug_condim", type=int, choices=(3, 4, 6), default=4,
                                help="mjc:condim on both plugs' colliders at spawn (4: torsional friction on every plug contact).")
        if "--m2_fingertip_condim" not in parser._option_string_actions:
            parser.add_argument("--m2_fingertip_condim", type=int, choices=(1, 3, 4, 6), default=None,
                                help="Optional sensor-pad contact dimension; use 4 with m2_plug_condim 3 for soft-pad grasps.")
        if "--m2_plug_torsional" not in parser._option_string_actions:
            parser.add_argument("--m2_plug_torsional", type=float, default=None,
                                help="Torsional friction [m] on both plugs' colliders (with --m2_plug_condim 4).")
        if "--m2_wire_stiffness" not in parser._option_string_actions:
            parser.add_argument("--m2_wire_stiffness", type=float, default=None,
                                help="Drive both plugs' wire joints to their authored shape with this stiffness [N m/rad].")
        if "--m2_wire_length_m" not in parser._option_string_actions:
            parser.add_argument("--m2_wire_length_m", type=float, default=0.18,
                                help="Anchored cable length in metres (0.16-0.4); zero keeps the original free tail.")
        if "--m2_wire_collision" not in parser._option_string_actions:
            parser.add_argument("--m2_wire_collision", choices=("none", "panel"), default="none",
                                help="Cable collision partners: none, or only the panel surface and walls.")
        if "--m2_wire_joint_armature" not in parser._option_string_actions:
            parser.add_argument("--m2_wire_joint_armature", type=float, default=1e-6,
                                help="Additional cable joint inertia [kg m^2] when cable collision is enabled.")
        if "--m2_wire_rest_spot" not in parser._option_string_actions:
            parser.add_argument("--m2_wire_rest_spot", type=float, nargs=3, default=None, metavar=("DX", "DY", "YAW"),
                                help="Author the cable at this tape spot [m, m, rad] (default: the tape release).")
        if "--m2_wire_reset_shape" not in parser._option_string_actions:
            parser.add_argument("--m2_wire_reset_shape", action=argparse.BooleanOptionalAction, default=True,
                                help="Set the anchored cable to reach its anchor from each reset's tape spot.")
        if "--m2_start_pull_out_mm" not in parser._option_string_actions:
            parser.add_argument(
                "--m2_start_pull_out_mm",
                type=lambda value: None if value.lower() == "none" else float(value),
                default=None,
                help="How far along its axis from the seat the plug is released at reset [mm] (default: S2 its "
                "Unplugged frame, 11.5); it settles in the mate's entry.",
            )
