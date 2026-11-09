# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Shared base for DexBench scene environments with optional embodiment mounting."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.assets.registries import EnvironmentRegistry
from isaaclab_arena.cli.dataclass_cli import dataclass_from_cli
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg
from isaaclab_arena.utils.physics_backend import PhysicsBackend
from isaaclab_arena.utils.yaw import rotate_quat_by_yaw
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

from dexbench_arena.environments.manipulation.newton_build_scope import begin_newton_build
from dexbench_arena.assets.local_payload import require_local_assets
from dexbench_arena.assets.physics_variants import asset_package_version
from dexbench_arena.cli import (
    NEWTON_NCONMAX,
    NEWTON_NJMAX,
    apply_dexbench_env_settings,
    apply_dexbench_physics_settings,
    apply_embodiment_recorders,
)
from dexbench_arena.environments.arena_builder_hooks import install_builder_hooks
from dexbench_arena.environments.scene_visibility import hide_replay_prims_callback
from dexbench_arena.teleop.newton_solver import apply_solver_options
from dexbench_arena.benchmark.frozen import check_frozen_row_file
from dexbench_arena.evaluation.evaluation_signature import EVALUATION_SETTINGS_KEY, record_evaluation_settings
from dexbench_arena.evaluation.replay_guards import (
    ObjectIdentity,
    ReplayGuardError,
    bound_variation_events,
    check_benchmark_replay,
    check_replay_source,
    check_replayed_reset_events,
    enabled_variation_keys,
    variation_ranges,
)
from dexbench_arena.embodiments.lookup import ik_profile_for_source_origin, register_dexbench_embodiments, start_poses
from dexbench_arena.embodiments.common.recorder_ready_events import RECORDER_READY_MODE
from dexbench_arena.tasks.object_axis import ObjectAxis, TaskObject, task_object_record
from dexbench_arena.environments.fingertip_contact import (
    add_fingertip_contact_cli_args,
    configure_environment_fingertip_contact,
)
from dexbench_arena.environments.hand_contact import (
    add_hand_contact_material_cli_args,
    configure_environment_hand_contact_material,
)
from dexbench_arena.environments.joint_torque import add_joint_torque_cli_args, configure_environment_joint_torque

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

EMBODIMENT_ROOT_Z_M = 0.0
RBY1_CONTROL_HZ_CHOICES = (30, 60)


def _add_argument_if_missing(parser: argparse.ArgumentParser, *option_strings: str, **kwargs) -> None:
    """Add an argparse option unless another launcher layer already owns it."""

    if any(option in parser._option_string_actions for option in option_strings):
        return
    parser.add_argument(*option_strings, **kwargs)


def _with_variation_samples(previous):
    """Chain an env cfg callback that records every variation draw in demos and Mimic data."""

    def callback(env_cfg):
        if previous is not None:
            env_cfg = previous(env_cfg) or env_cfg
        from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg

        from dexbench_arena.variations.variation_samples_recorder import VariationSamplesRecorderCfg

        for name in ("demo_recorder_config", "mimic_recorder_config"):
            if not hasattr(env_cfg, name):
                continue
            if getattr(env_cfg, name) is None:
                setattr(env_cfg, name, ActionStateRecorderManagerCfg())
            getattr(env_cfg, name).record_variation_samples = VariationSamplesRecorderCfg()
        return env_cfg

    return callback


def _with_dexbench_settings(previous, settings: _EnvironmentSettings):
    """Chain an env cfg callback that applies DexBench's settings, the solver's included."""

    def callback(env_cfg):
        begin_newton_build()
        if previous is not None:
            env_cfg = previous(env_cfg) or env_cfg
        apply_dexbench_env_settings(
            env_cfg, settings.cfg, embodiment=settings.embodiment, output_base_dir=settings.output_base_dir
        )
        apply_dexbench_physics_settings(env_cfg, settings.cfg, environment=settings.factory)
        apply_embodiment_recorders(env_cfg, settings.embodiment)
        if settings.replayed_variation_samples is not None and not settings.cfg.benchmark_replay:
            check_replayed_reset_events(
                getattr(env_cfg, "events", None),
                what=f"Replaying recorded variation samples from {settings.replayed_variation_samples}",
            )
        return env_cfg

    return callback


def mimic_action_noise_callback(previous, cfg: DexBenchEnvironmentCfg):
    """``previous`` with every Mimic subtask's ``action_noise`` set to ``cfg.mimic_action_noise``; ``previous``
    when that is None. Arena builds the Mimic env cfg's ``subtask_configs`` from the task before the callback
    runs, and the clean-label recorders read the noise off the composed cfg. Refuses a set
    ``DEXBENCH_MIMIC_ACTION_NOISE``, which nothing reads and which would otherwise be ignored without a word."""
    if os.environ.get("DEXBENCH_MIMIC_ACTION_NOISE") is not None:
        raise ValueError("DEXBENCH_MIMIC_ACTION_NOISE is not read; pass --mimic_action_noise")
    if cfg.mimic_action_noise is None:
        return previous
    from dexbench_arena.embodiments.common.clean_action_labels import mimic_action_noise

    noise = mimic_action_noise(cfg.mimic_action_noise)

    def with_action_noise(env_cfg):
        if previous is not None:
            env_cfg = previous(env_cfg)
        for subtasks in (getattr(env_cfg, "subtask_configs", None) or {}).values():
            for subtask in subtasks:
                subtask.action_noise = noise
        return env_cfg

    return with_action_noise


def mimic_keep_failed_callback(previous, cfg: DexBenchEnvironmentCfg):
    """``previous`` with the Mimic env cfg's ``datagen_config.generation_keep_failed`` set to
    ``cfg.mimic_keep_failed``; ``previous`` when that is None (the task's Mimic cfg
    decides). Arena copies the task's ``datagen_config`` into the composed cfg before the callback runs, and the
    generator reads it from there. Refuses a set ``DEXBENCH_MIMIC_KEEP_FAILED``, which nothing reads and which
    would otherwise be ignored without a word."""
    if os.environ.get("DEXBENCH_MIMIC_KEEP_FAILED") is not None:
        raise ValueError("DEXBENCH_MIMIC_KEEP_FAILED is not read; pass --mimic_keep_failed")
    if cfg.mimic_keep_failed is None:
        return previous
    keep = bool(cfg.mimic_keep_failed)

    def with_keep_failed(env_cfg):
        if previous is not None:
            env_cfg = previous(env_cfg)
        datagen = getattr(env_cfg, "datagen_config", None)
        if datagen is not None:
            datagen.generation_keep_failed = keep
        return env_cfg

    return with_keep_failed


def teleop_target_marker_callback(previous, cfg: DexBenchEnvironmentCfg, task):
    """``previous`` with the operator's target marker added when ``cfg.teleop_target_marker`` is on."""
    if not cfg.teleop_target_marker:
        return previous
    object_name = getattr(task, "object_name", None)
    if object_name is None:
        raise ValueError(
            f"--teleop_target_marker: task {getattr(cfg, 'task', None)!r} names no target object (object_name)"
        )
    slot_region = task.slot_region() if hasattr(task, "slot_region") else None

    def callback(env_cfg):
        if previous is not None:
            env_cfg = previous(env_cfg) or env_cfg
        from dexbench_arena.teleop.target_marker import teleop_slot_marker_event, teleop_target_marker_event

        env_cfg.events.teleop_target_marker = teleop_target_marker_event(object_name)
        if slot_region is not None:
            env_cfg.events.teleop_slot_marker = teleop_slot_marker_event(slot_region)
        return env_cfg

    return callback


def record_environment_cfg(env, env_id: int, environment_cfg: dict) -> dict:
    """Episode recorder term: the typed config the episode's environment was built from."""
    del env, env_id
    return {"environment_cfg": environment_cfg}


def terminating_terms(env, env_id: int) -> list[str]:
    """The termination terms that ended ``env_id``'s episode, read at its reset (``time_out`` alone when it ran to its
    cap). Empty when no term fired at the reset: the run's end cut the episode short, or the script reset it itself
    (``record_demos`` judges success outside the termination manager and resets with ``env.reset()``, so its exported
    and discarded takes are empty too). A row without this field: not recorded, not "no term"."""
    manager = env.termination_manager
    return [name for name in manager.active_terms if bool(manager.get_term(name)[env_id])]


def record_terminated_by(env, env_id: int) -> dict:
    """Episode recorder term: the terms that ended the episode (:func:`terminating_terms`), so a report tells a task's
    failure term (DB-L3's ``sheet_dropped``) from the cap without inferring it from the episode's length."""
    return {"terminated_by": terminating_terms(env, env_id)}


def record_task_object(env, env_id: int, task_object: dict) -> dict:
    """Episode recorder term: the object the episode's task was run with (``task_object_record``)."""
    del env, env_id
    return {"task_object": task_object}


def record_build_inputs(env, env_id: int, build_inputs: dict) -> dict:
    """Episode recorder term: what the build read beside its cfg that a replay of the row must find again
    (``recorded_build_inputs``), at the row's top level as a written row carries it."""
    del env, env_id
    return dict(build_inputs)


TASK_OBJECT_METADATA_KEY = "task_object"
"""The ``env_args`` key a demo HDF5 carries the same record under."""


class _TaskObjectEpisodeMetadata:
    """Add the run's ``task_object`` to what the recorder writes into ``env_args`` on export."""

    def __init__(self, get_ep_meta, task_object: dict):
        self._get_ep_meta = get_ep_meta
        self._task_object = task_object

    def __call__(self) -> dict:
        result = dict(self._get_ep_meta())
        result[TASK_OBJECT_METADATA_KEY] = self._task_object
        return result


def install_task_object_recording_metadata(env, env_ids, task_object: dict) -> None:
    """Recorder-ready event: every HDF5 this env exports (demos, Mimic data, an evaluation's
    ``--record_dataset``) names the object, as the episode JSONL does."""
    del env_ids
    recorder = env.recorder_manager
    if not isinstance(recorder.get_ep_meta, _TaskObjectEpisodeMetadata):
        recorder.get_ep_meta = _TaskObjectEpisodeMetadata(recorder.get_ep_meta, task_object)


def _with_task_object_metadata(previous, task_object: dict):
    """Chain an env cfg callback that installs :func:`install_task_object_recording_metadata`."""

    def callback(env_cfg):
        if previous is not None:
            env_cfg = previous(env_cfg) or env_cfg
        if getattr(env_cfg, "events", None) is not None:
            from isaaclab.managers import EventTermCfg

            env_cfg.events.dexbench_task_object = EventTermCfg(
                func=install_task_object_recording_metadata, mode=RECORDER_READY_MODE, params={"task_object": task_object}
            )
        return env_cfg

    return callback


def exported_demos(env) -> dict[int, str]:
    """The HDF5 demo each env's running episode was exported as, for :func:`record_exported_demo`."""
    if getattr(env, "dexbench_exported_demos", None) is None:
        env.dexbench_exported_demos = {}
    return env.dexbench_exported_demos


def record_exported_demo(env, env_id: int) -> dict:
    """Episode recorder term: the demo the episode was exported as, null for a take not exported."""
    exported = getattr(env, "dexbench_exported_demos", None)
    if exported is None:
        return {}
    return {"exported_demo": exported.pop(env_id, None)}


TASKS = "tasks"
"""Config field metadata key: the tasks of a multi-task environment that read the field
(:meth:`DexBenchSceneEnvironmentBase.task_settings`); a field without it is every task's."""


def environment_cfg_snapshot(cfg: DexBenchEnvironmentCfg) -> dict:
    """The config as plain JSON values (paths, enums and the like as their strings)."""
    return json.loads(json.dumps(dataclasses.asdict(cfg), default=str))


@dataclass
class DexBenchEnvironmentCfg(ArenaEnvironmentCfg):
    """Settings every DexBench environment factory takes, with the command line's defaults."""

    embodiment: str | None = None
    """Embodiment asset name, e.g. ``dexbench_rby1_wuji2_joint``; None builds the scene-only shell."""

    teleop_device: str | None = None
    """Registered teleop device the environment hands to Arena; None for none."""

    arm_mode: str = "dual_arm"
    """Which arms the embodiment exposes: ``left``, ``right`` or ``dual_arm``."""

    camera_scale: float | None = None
    """Raster scale at fixed FOV; None uses the embodiment default (Wuji2: 0.5, 640x360)."""

    wrist_cameras: bool | None = None
    """GA wrist RGB opt-in (default off); the embodiment always keeps the physical mounts."""

    """Show the authored wrist colliders in RGB/viewport; for inspection, not policy input."""

    ik_profile: str | None = None
    """Pink IK profile for an IK embodiment (``teleop`` or ``generation``); None keeps the embodiment's."""

    source_demo_origin: tuple[str, str] | None = None
    """How the demos a Mimic generation run reads were made, and what says so (``generate_dataset.py``
    sets it from ``--input_file``): with no ``ik_profile``, an IK embodiment picks its profile from it."""

    show_ik_warnings: bool = False
    """Print Pink IK solver warnings (IK embodiments only)."""

    mimic_boundaries: str | None = None
    """JSON table of stored subtask boundaries for ``annotate_demos.py --auto``."""

    mimic_action_noise: float | None = None
    """Per-step Mimic action noise for every subtask of a generation run; None keeps the task's (0).
    With noise, the generated files carry the clean labels beside the executed ones (clean_action_labels)."""

    mimic_keep_failed: bool | None = None
    """Keep a Mimic generation run's failed attempts (``<output>_failed.hdf5``) to replay and measure them;
    None keeps the task's choice."""

    waist_ik: bool | None = None
    """Enable torso IK on an embodiment whose constructor supports it; None leaves it to the embodiment."""

    waist_ik_max_lead_rad: float | None = None
    """Torso command tracking bound in radians; None uses the embodiment default."""

    head_action: bool | None = None
    """Command the head directly after an IK embodiment's columns (``dexbench_rby1_wuji2``: 56-D); None leaves
    the head to the embodiment."""

    anchor: str | None = None
    """Robot base anchor name; None takes the environment's ``default_anchor``."""

    hand_static_friction: float | None = None
    """Run override of the task's static hand contact coefficient; None inherits."""

    hand_dynamic_friction: float | None = None
    """Run override of the task's dynamic hand contact coefficient; None inherits."""

    hand_friction_combine_mode: str | None = None
    """Run override of the hand material's PhysX combine mode; None inherits."""

    hand_contact_material_enabled: bool | None = None
    """False puts the robot model's own contact material on the hand (Wuji v2: its MJCF's 1.0 / 1.0, max)
    instead of the task's; None inherits the task's setting."""

    record_fingertip_contact: bool | None = None
    """Record each fingertip's contact force (net, normal, tangential) in datasets. None records where
    supported (Wuji v2 on Newton), True requires it, False turns it off."""

    record_joint_torque: bool | None = None
    """Record each robot joint's actuator effort (Isaac Lab's applied_effort) in datasets. None records
    where the embodiment declares it (Wuji v2), True requires it, False turns it off."""

    start_pose: str | None = None
    """Named embodiment reset pose, e.g. ``rrc_ready`` for RBY1/Wuji; None keeps the task's or the embodiment's."""

    base_offset_x: float = 0.0
    """Deterministic embodiment-base X offset from the selected anchor [m]."""

    base_offset_y: float = 0.0
    """Deterministic embodiment-base Y offset from the selected anchor [m]."""

    base_jitter_radius: float = 0.0
    """Recorded per-reset embodiment-base XY jitter half-extent [m]."""

    base_jitter_low: tuple[float, float, float] | None = None
    """Lower bounds of the recorded per-reset base draw ``[dx, dy]`` [m] in the world and ``dyaw`` [deg],
    with ``base_jitter_high``, for a base that goes further to one side than the other; in place of
    ``base_jitter_radius``."""

    base_jitter_high: tuple[float, float, float] | None = None
    """Upper bounds of that draw, ``[dx, dy, dyaw]`` [m, m, deg]."""

    rby1_control_hz: int | None = None
    """Control/recording rate, 30 or 60; 30 matches the trained checkpoints. None follows
    ``DEXBENCH_CONTROL_HZ``, else ``cli.DEXBENCH_DEFAULT_CONTROL_HZ`` (30 Hz)."""

    camera_hz: int | None = None
    """How often the cameras take a new frame [Hz], below the control rate: a policy asked at that rate reads only
    those frames (``cli.set_camera_update_period``). None takes one every control step."""

    episode_length_s: float | None = None
    """Episode timeout in simulated seconds for whichever task the run selects; None keeps the task's
    own (``TaskBase.episode_length_s``, which Arena's builder reads through ``get_episode_length_s``).
    Upstream's environment configs carry the same field (``droid_table_multi_object_placement`` at the pin)."""

    physics_variants: bool = True
    """False spawns props from their neutral base instead of the backend's physics runtime variant (A/B run)."""

    newton_gravity_compensation: bool = True
    """False leaves ``disable_gravity`` bodies uncompensated on Newton (A/B run)."""

    newton_nconmax: int = NEWTON_NCONMAX
    """Lower bound on MJWarp's contact budget on Newton."""

    newton_njmax: int = NEWTON_NJMAX
    """Lower bound on MJWarp's constraint budget on Newton."""

    newton_cuda_graph: bool = True
    """False steps Newton without CUDA graph capture (slower, survives many resets)."""

    newton_solver: str | None = None
    """MJWarp's constraint solver on Newton (``cg`` or ``newton``), over the environment's Newton profile; None
    keeps the profile's (E2: cg, DB-L1: newton). Newton 1.6.1's ``newton`` solver goes non-finite on recorded
    hand contacts."""

    newton_num_substeps: int | None = None
    """MJWarp substeps per physics step on Newton, over the environment's Newton profile; None keeps the profile's."""

    newton_solver_tolerance: float | None = None
    """MJWarp's constraint-solver convergence tolerance on Newton; None keeps the profile's (Isaac Lab's 1e-6)."""

    mimic: bool = False
    """Build the scene for Isaac Lab Mimic (the task's Mimic embodiment setup). Arena's shared
    ``--mimic`` flag on the command line; a typed run sets it here as well as on its builder."""

    num_envs: int = 1
    """Parallel environments, for the cameras' tiled representation. Arena's shared ``--num_envs``
    on the command line; a typed run sets it here as well as on its builder."""

    record_dataset: bool = False
    """Export every episode's per-step actions, observations and states to an HDF5 dataset in the run
    directory. The evaluation runners' ``--record_dataset`` on the command line."""

    benchmark_replay: bool = False
    """A benchmark cell or sweep: the run must replay rows the task's frozen set lists unchanged
    (``benchmark.frozen``), which record every enabled variation and the build's object, and its
    reset events must replay; refused otherwise whatever
    ``replay_guards.REPLAY_GUARD_MODE`` says (``replay_guards.check_benchmark_replay``). The
    generated eval and sweep files set it."""

    record_settle: bool = False
    """Add a ``settle`` field to every episode row: whether the scene's rigid objects came to rest by the
    episode's end and which terms ended it (``tasks.condition_recording``). A condition recording sets it."""

    record_dataset_images: bool = False
    """Also store the raw camera frames (``camera_obs``); implies ``record_dataset`` and needs
    ``enable_cameras``. The evaluation runners' ``--record_dataset_images`` on the command line."""

    hide_replay_prims: bool = False
    """Hide the environment's ``replay_hidden_prim_patterns`` from every render, as the demo re-record
    (``rerecord_demos.py``) does: an evaluation of a policy trained on re-rendered demos then sees what
    its training frames saw (DB-H1: the vision-only tablecloth and the ground-plane grid). Render only."""

    teleop_target_marker: bool = False
    """Hover a marker over the task's target object for the operator (``teleop.target_marker``),
    and draw where it must end up when the task has a slot (DB-R3, green once it is seated there):
    visual only, no collider, not observed or recorded. Teleoperation only; a camera recording
    would see it."""


class _EnvironmentSettings:
    """What DexBench applies to one built environment once a builder composes it."""

    def __init__(self, factory: DexBenchSceneEnvironmentBase, scene: Scene, embodiment, cfg: DexBenchEnvironmentCfg):
        self.factory = factory
        self.scene = scene
        self.embodiment = embodiment
        self.cfg = cfg
        self.output_base_dir: str | None = None
        """The runner's ``--output_base_dir`` (``get_env``): a recording run writes its dataset in the run directory under it."""
        self.replayed_variation_samples: str | None = None
        """The builder's ``recorded_variation_samples_path``, set before it composes (``before_compose``)."""
        axis = factory.object_axis_for(cfg)
        task_object = axis.selected(cfg) if axis is not None else None
        asset_version = factory.task_object_asset_version(task_object) if task_object is not None else None
        self.object_identity = ObjectIdentity.of(axis, cfg, asset_version=asset_version) if axis is not None else None
        """The object and fixture profile the build is run with, what a replay source must match."""

    def check_replay_source(self, path, *, what: str) -> None:
        """Warn or refuse (``replay_guards.REPLAY_GUARD_MODE``) when ``path`` (an episode JSONL or a demo
        HDF5) was made with another object or fixture profile than this build."""
        check_replay_source(self.object_identity, path, what=what)

    def before_compose(self, builder_cfg, backend) -> None:
        """Check ``builder_cfg`` (``ArenaEnvBuilderCfg``) against the config before the builder composes
        the scene. ``backend`` is the builder's resolved one (its ``presets``, else the environment's
        ``default_physics_backend``, Newton); any other is refused (:func:`check_backend_supported`)."""
        check_backend_supported(self.factory, backend)
        check_builder_agrees(self.cfg, builder_cfg)
        replayed = self.replayed_variation_samples = getattr(builder_cfg, "recorded_variation_samples_path", None)
        if self.cfg.benchmark_replay and replayed is None:
            raise ReplayGuardError("[DexBench benchmark replay] a benchmark run replays frozen rows; this one names none "
                                   "(environment_builder.recorded_variation_samples_path)")
        if self.cfg.benchmark_replay:
            try:
                check_frozen_row_file(replayed)
            except ValueError as error:
                raise ReplayGuardError(f"[DexBench benchmark replay] {error}") from None
        if replayed is not None:
            self.check_replay_source(replayed, what="Replaying recorded variation samples")
            self.factory.check_replayed_rows(self.cfg, replayed)

    def after_compose(self, variations, env_cfg=None) -> None:
        """With ``benchmark_replay``, refuse rows that leave an enabled variation live or do not record the
        build's object (``replay_guards.check_benchmark_replay``), then refuse any reset, interval or
        startup event of ``env_cfg`` that may draw: one holding a sampler none of these (bound) variations
        owns included. ``variations`` is the builder's ``get_all_variations()`` once it has applied the
        run's overrides and bound the replay."""
        if self.cfg.benchmark_replay:
            check_benchmark_replay(self.object_identity, self.replayed_variation_samples,
                                   enabled_variation_keys(variations), what="Benchmark replay",
                                   ranges=variation_ranges(variations))
            check_replayed_reset_events(getattr(env_cfg, "events", None), what="Benchmark replay", strict=True,
                                        bound_events=bound_variation_events(variations))


def check_builder_agrees(cfg: DexBenchEnvironmentCfg, builder_cfg) -> None:
    """Refuse a builder whose ``num_envs`` or ``mimic`` differs from the environment config's."""
    pairs = {
        "num_envs": (int(cfg.num_envs or 1), int(builder_cfg.num_envs)),
        "mimic": (bool(cfg.mimic), bool(builder_cfg.mimic)),
    }
    differ = {name: values for name, values in pairs.items() if values[0] != values[1]}
    if differ:
        raise ValueError(
            "The environment config and its builder disagree on "
            + ", ".join(f"{name} (environment {env}, environment_builder {builder})" for name, (env, builder) in differ.items())
            + "; a typed Run sets these under both `environment:` and `environment_builder:`."
        )


def check_backend_supported(factory, backend) -> None:
    """Refuse ``backend`` (``"physx"`` / ``"newton"`` or Arena's ``PhysicsBackend``) unless it is Newton."""
    if str(backend) != PhysicsBackend.NEWTON.value:
        raise ValueError(
            f"{factory.name} runs on Newton only, and this build resolved {backend}: drop --presets "
            "(environment_builder.presets in a run config) or set it to newton (NEWTON)."
        )


def _set_camera_parallelism(embodiment, cfg: DexBenchEnvironmentCfg) -> None:
    """Select Arena's camera representation from the requested environment count."""
    num_envs = int(cfg.num_envs or 1)
    embodiment.camera_config.set_use_tiled_camera(num_envs > 1)


def _base_draw_bound(text: str) -> tuple[float, float, float]:
    """argparse type for ``--base_jitter_low/high``: ``"dx,dy,dyaw"``, as an experiment's list field is passed."""
    values = tuple(float(token) for token in text.split(","))
    if len(values) != 3:
        raise argparse.ArgumentTypeError(f"needs dx,dy,dyaw; got {text!r}")
    return values


class DexBenchSceneEnvironmentBase(ExampleEnvironmentBase):
    """Scene environment that can optionally mount an embodiment at a base anchor."""

    _legacy_argparse_cfg_type: type[DexBenchEnvironmentCfg] = DexBenchEnvironmentCfg

    robot_base_anchors_xyz: dict[str, tuple[float, float, float]] = {}
    anchor_facing_targets_xy: dict[str, tuple[float, float]] = {}
    anchor_yaws_deg: dict[str, float] = {}
    embodiment_forward_axis_xy: tuple[float, float] = (1.0, 0.0)
    default_anchor: str | None = None
    anchor_help: str = "Robot base anchor."
    legacy_values: dict[str, dict[str, str]] = {}
    embodiment_root_z_m: float = EMBODIMENT_ROOT_Z_M
    embodiment_root_xyz_overrides: dict[str, tuple[float, float, float]] = {}
    embodiment_anchor_overrides: dict[str, dict[str, tuple[float, float, float]]] = {}
    embodiment_root_z_overrides: dict[str, float] = {}

    def build_scene(self, cfg: DexBenchEnvironmentCfg) -> Scene:
        """Build the embodiment-free scene."""

        raise NotImplementedError

    available_tasks: dict[str, object] = {}
    object_axes: dict[str, ObjectAxis] = {}
    shared_settings: tuple[str, ...] = ()

    @classmethod
    def task_settings(cls, task: str) -> frozenset[str]:
        """The config fields that are settings of ``task``: every field no task claims, and those marked for it."""
        if task not in cls.available_tasks:
            raise ValueError(f"{cls.name} has no task {task!r} (available: {tuple(cls.available_tasks)})")
        fields = dataclasses.fields(EnvironmentRegistry().get_environment_cfg_type(cls))
        for field in fields:
            marked = field.metadata.get(TASKS, ())
            if TASKS in field.metadata and not (isinstance(marked, tuple) and marked
                                                and all(isinstance(name, str) and name in cls.available_tasks for name in marked)):
                raise ValueError(f"{cls.name}: config field {field.name} is marked for {marked!r}, "
                                 f"not a tuple of its tasks {tuple(cls.available_tasks)}")
        if len(cls.available_tasks) > 1:
            common = {field.name for field in dataclasses.fields(DexBenchEnvironmentCfg)}
            own = {field.name: TASKS in field.metadata for field in fields if field.name not in common}
            shared = set(cls.shared_settings)
            unclassified = sorted(name for name, is_marked in own.items() if is_marked == (name in shared))
            if unclassified or shared - set(own):
                raise ValueError(f"{cls.name}: mark each field of its config for its tasks (TASKS metadata) or list it "
                                 f"in shared_settings, not both: {unclassified}; listed but no such field: "
                                 f"{sorted(shared - set(own))}")
        return frozenset(field.name for field in fields if TASKS not in field.metadata or task in field.metadata[TASKS])


    teleop_hidden_prim_patterns: tuple[str, ...] = ()
    replay_hidden_prim_patterns: tuple[str, ...] = ()

    def get_env(self, args_cli: argparse.Namespace) -> IsaacLabArenaEnvironment:
        """Construct the environment from a parsed command line."""
        presets = getattr(args_cli, "presets", None)
        if presets is not None:
            check_backend_supported(self, presets)
        environment = self.build(dataclass_from_cli(type(self)._legacy_argparse_cfg_type, args_cli))
        environment.dexbench_settings.output_base_dir = getattr(args_cli, "output_base_dir", None)
        return environment

    def build(self, cfg: DexBenchEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Construct the environment from its typed config (Arena's typed path, and the command line's through :meth:`get_env`)."""
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTermCfg

        renamed = {field: values[getattr(cfg, field)] for field, values in type(self).legacy_values.items()
                   if getattr(cfg, field, None) in values}
        cfg = dataclasses.replace(cfg, **renamed) if renamed else cfg
        embodiment, teleop_device = self._build_embodiment(cfg)
        scene = self.build_scene(cfg)
        require_local_assets(scene, embodiment, environment_name=self.name)
        settings = _EnvironmentSettings(self, scene, embodiment, cfg)
        self.apply_newton_scene_settings(scene, embodiment, cfg)
        task = self._build_task(cfg)
        if cfg.episode_length_s is not None:
            if not (math.isfinite(cfg.episode_length_s) and cfg.episode_length_s > 0.0):
                raise ValueError(f"episode_length_s must be a positive number of seconds, got {cfg.episode_length_s}")
            task.episode_length_s = float(cfg.episode_length_s)
        configure = getattr(task, "configure_mimic_embodiment", None)
        if cfg.mimic and configure is not None:
            configure(embodiment)

        joint_pos_overrides = self._start_pose_from_cli(cfg)
        if joint_pos_overrides is None:
            joint_pos_overrides = getattr(task, "robot_joint_pos_overrides", None)
        if joint_pos_overrides is None:
            joint_pos_overrides = self._get_embodiment_joint_pos_overrides(cfg)
        if joint_pos_overrides and embodiment is not None:
            embodiment.scene_config.robot.init_state.joint_pos = dict(joint_pos_overrides)

        environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=teleop_device,
            default_physics_backend=PhysicsBackend.NEWTON,
        )
        configure_environment_hand_contact_material(environment, cfg)
        configure_environment_fingertip_contact(environment, cfg)
        configure_environment_joint_torque(environment, cfg)
        environment.episode_recorder_terms["environment_cfg"] = EpisodeRecorderTermCfg(
            func=record_environment_cfg, params={"environment_cfg": environment_cfg_snapshot(cfg)}
        )
        environment.episode_recorder_terms[EVALUATION_SETTINGS_KEY] = EpisodeRecorderTermCfg(func=record_evaluation_settings)
        environment.episode_recorder_terms["exported_demo"] = EpisodeRecorderTermCfg(func=record_exported_demo)
        environment.episode_recorder_terms["terminated_by"] = EpisodeRecorderTermCfg(func=record_terminated_by)
        if cfg.record_settle:
            from dexbench_arena.tasks.condition_recording import record_settle

            belt_carried = self.belt_carried_objects(cfg)
            environment.episode_recorder_terms["settle"] = EpisodeRecorderTermCfg(
                func=record_settle, params={"belt_carried": belt_carried} if belt_carried else {}
            )
        build_inputs = self.recorded_build_inputs(cfg)
        if build_inputs:
            environment.episode_recorder_terms["build_inputs"] = EpisodeRecorderTermCfg(
                func=record_build_inputs, params={"build_inputs": build_inputs}
            )
        task_object = self.task_object(cfg)
        if task_object is not None:
            environment.episode_recorder_terms["task_object"] = EpisodeRecorderTermCfg(
                func=record_task_object, params={"task_object": task_object}
            )
        environment.env_cfg_callback = mimic_action_noise_callback(environment.env_cfg_callback, cfg)
        environment.env_cfg_callback = mimic_keep_failed_callback(environment.env_cfg_callback, cfg)
        boundaries = cfg.mimic_boundaries
        if boundaries:
            previous = environment.env_cfg_callback

            def _with_subtask_boundaries(env_cfg):
                if previous is not None:
                    env_cfg = previous(env_cfg)
                env_cfg.subtask_boundaries_path = str(boundaries)
                return env_cfg

            environment.env_cfg_callback = _with_subtask_boundaries
        environment.env_cfg_callback = _with_variation_samples(environment.env_cfg_callback)
        environment.env_cfg_callback = teleop_target_marker_callback(environment.env_cfg_callback, cfg, task)
        environment.env_cfg_callback = hide_replay_prims_callback(
            environment.env_cfg_callback,
            cfg,
            type(self).replay_hidden_prim_patterns,
        )
        if task_object is not None:
            environment.env_cfg_callback = _with_task_object_metadata(environment.env_cfg_callback, task_object)
        environment.env_cfg_callback = _with_dexbench_settings(environment.env_cfg_callback, settings)
        environment.dexbench_settings = settings
        install_builder_hooks()
        return environment

    def object_axis_for(self, cfg: DexBenchEnvironmentCfg) -> ObjectAxis | None:
        """The object axis ``cfg``'s task is run on (``object_axes``); None for a task without one."""
        return type(self).object_axes.get(getattr(cfg, "task", None))

    def task_object_fit(self, cfg: DexBenchEnvironmentCfg, task_object: TaskObject) -> dict:
        """What this environment derives from the object beyond the axis' own fit rule; an environment
        whose derivation needs its scene (E2's seat on the cage) adds it here."""
        del cfg, task_object
        return {}

    def task_object(self, cfg: DexBenchEnvironmentCfg) -> dict | None:
        """The ``task_object`` record of a build from ``cfg`` (``task_object_record``); None without an axis."""
        axis = self.object_axis_for(cfg)
        if axis is None:
            return None
        task_object, fit = axis.resolve(cfg)
        fit.update(self.task_object_fit(cfg, task_object))
        usd_path = self.task_object_usd_path(task_object)
        version = asset_package_version(usd_path) if isinstance(usd_path, str) else None
        return task_object_record(axis, task_object, fit, version)

    def task_object_usd_path(self, task_object: TaskObject) -> str | None:
        """The USD whose package version a ``task_object`` record carries: the registered asset's; an environment
        whose objects are not registered assets (DB-L1's bag sets, VBD bags the task builds) names its own."""
        return getattr(self.asset_registry.get_asset_by_name(task_object.asset), "usd_path", None)

    def task_object_asset_version(self, task_object: TaskObject) -> str | None:
        """Current package-sidecar stamp for benchmark compatibility, also usable before a scene build."""
        path = self.task_object_usd_path(task_object)
        return asset_package_version(path) if isinstance(path, str) else None

    def _build_task(self, cfg: DexBenchEnvironmentCfg):
        """Build the task selected via ``--task`` (``NoTask`` when unset/none, or for an environment without tasks)."""

        from isaaclab_arena.tasks.no_task import NoTask

        task_name = getattr(cfg, "task", None)
        if not task_name or task_name == "none":
            return NoTask()
        factory = type(self).available_tasks.get(task_name)
        assert factory is not None, (
            f"{self.name} has no task {task_name!r} (available: {tuple(type(self).available_tasks)})"
        )
        return factory(self, cfg)

    @staticmethod
    def _start_pose_from_cli(cfg: DexBenchEnvironmentCfg):
        """Resolve ``--start_pose <name>`` against the embodiment's named-pose registry."""

        name = cfg.start_pose
        if not name:
            return None
        embodiment_name = cfg.embodiment
        poses = start_poses(embodiment_name)
        if not poses:
            raise ValueError(
                f"--start_pose has no named poses for embodiment {embodiment_name or '(none)'} yet"
            )
        if name not in poses:
            raise ValueError(f"Unknown --start_pose {name!r}; available: {tuple(poses)}")
        return poses[name]

    def _get_embodiment_joint_pos_overrides(self, cfg: DexBenchEnvironmentCfg):
        """Return scene-shell joint defaults for embodiments that need them."""

        return start_poses(cfg.embodiment).get("replay_home")

    def _robot_base_anchors_for_args(
        self, cfg: DexBenchEnvironmentCfg
    ) -> dict[str, tuple[float, float, float]]:
        """Resolve robot anchors for a runtime-selected scene layout."""

        return type(self).robot_base_anchors_xyz

    def _anchor_facing_targets_for_args(
        self, cfg: DexBenchEnvironmentCfg
    ) -> dict[str, tuple[float, float]]:
        """Resolve anchor-facing targets for a runtime-selected scene layout."""

        return type(self).anchor_facing_targets_xy

    def _embodiment_anchor_overrides_for_args(
        self, cfg: DexBenchEnvironmentCfg, base: str | None
    ) -> dict[str, tuple[float, float, float]]:
        """Resolve per-station positions of the ``base`` model for the selected layout."""

        return type(self).embodiment_anchor_overrides.get(base, {})

    newton_solver_options: dict = {}
    """The scene's MuJoCo-Warp solver and contact options on Newton (keys of
    ``teleop.newton_solver.SOLVER_OPTIONS``); empty keeps Arena's preset with DexBench's budgets."""

    def newton_solver_options_for(self, cfg: DexBenchEnvironmentCfg) -> dict:
        """The solver options a build from ``cfg`` runs with: :attr:`newton_solver_options`, unless the
        environment's stagings or tasks need different ones (E3's DB-M1 and DB-M4, E1's DB-R2 and DB-R3)."""

        del cfg
        return type(self).newton_solver_options

    def check_replayed_rows(self, cfg: DexBenchEnvironmentCfg, path) -> None:
        """Refuse (ValueError) start-state rows ``path`` (the builder's ``recorded_variation_samples_path``) that a
        build from ``cfg`` cannot start from, beyond what every environment's object check covers
        (``check_replay_source``, which only warns outside a benchmark replay). The default accepts them; DB-M5's
        staging overrides it (``e3_assembly_line_environment``)."""

    def recorded_build_inputs(self, cfg: DexBenchEnvironmentCfg) -> dict:
        """Fields every row recorded from a build of ``cfg`` carries at its top level, so that
        :meth:`check_replayed_rows` refuses the row on a build that read other inputs, as it refuses a written row
        that carries them. The default is none; DB-M5's staging overrides it (``e3_assembly_line_environment``)."""

        del cfg
        return {}

    def belt_carried_objects(self, cfg: DexBenchEnvironmentCfg) -> dict[str, tuple[float, float, float]]:
        """The rigid objects a moving belt carries in a build from ``cfg``, each with that belt surface's world
        velocity (m/s): a condition recording judges them at rest relative to their belt
        (``tasks.condition_recording.record_settle``). The default is none, every object judged in the world;
        DB-M5's staging (``e3_assembly_line_environment``) and DB-L4 (``e2_logistics_l4_environment``) override it."""

        del cfg
        return {}

    def apply_newton_sim_settings(self, env_cfg, cfg: DexBenchEnvironmentCfg) -> None:
        """Set :meth:`newton_solver_options_for` on a Newton env cfg (``cli.apply_dexbench_physics_settings`` calls it last)."""

        options = self.newton_solver_options_for(cfg)
        if options:
            apply_solver_options(env_cfg.sim.physics, options)

    def apply_newton_scene_settings(self, scene: Scene, embodiment, cfg: DexBenchEnvironmentCfg) -> None:
        """Fit the scene to Newton: each prop's Newton physics variant and gravity compensation."""

        self._apply_physics_variants(scene, cfg)
        self._apply_newton_gravity_compensation(scene, embodiment, cfg)

    def _apply_physics_variants(self, scene: Scene, cfg: DexBenchEnvironmentCfg) -> None:
        """Select the Newton runtime variant on every rigid object that carries it."""

        from dexbench_arena.assets.physics_variants import apply_physics_variants, physics_variants_enabled

        if not physics_variants_enabled(cfg):
            print("[dexbench] physics variants off (--no-physics_variants); props spawn from their neutral base.")
            return
        selected = apply_physics_variants(scene)
        if selected:
            print(f"[dexbench] {selected[0][1]} variant selected on {len(selected)} props: " + ", ".join(n for n, _ in selected))

    def _apply_newton_gravity_compensation(self, scene: Scene, embodiment, cfg: DexBenchEnvironmentCfg) -> None:
        """Re-express ``disable_gravity`` as MuJoCo gravity compensation: Newton does not read it."""

        from dexbench_arena.assets.newton_gravity import (
            apply_newton_gravity_compensation,
            newton_gravity_compensation_enabled,
        )

        if not newton_gravity_compensation_enabled(cfg):
            print("[dexbench] Newton gravity compensation off (--no-newton_gravity_compensation); disable_gravity bodies fall at g.")
            return
        compensated = apply_newton_gravity_compensation(scene, embodiment)
        if compensated:
            print(f"[dexbench] Newton gravity compensation (mjc:gravcomp=1.0) for {len(compensated)}: " + ", ".join(compensated))

    def embodiment_constructor_kwargs(self, cfg):
        """Constructor parameters for the selected embodiment; tasks extend this mapping."""
        from isaaclab_arena.embodiments.common.arm_mode import ArmMode

        embodiment_kwargs = {
            "enable_cameras": cfg.enable_cameras,
            "arm_mode": ArmMode(cfg.arm_mode),
        }
        flag_args = ("show_ik_warnings", "waist_ik", "head_action")
        value_args = ("ik_profile", "waist_ik_max_lead_rad", "camera_scale", "wrist_cameras")
        optional_args = (*flag_args, *value_args)
        for option in optional_args:
            value = getattr(cfg, option)
            if (value if option in flag_args else value is not None):
                embodiment_kwargs[option] = value
        return embodiment_kwargs

    def _build_embodiment(self, cfg: DexBenchEnvironmentCfg):
        embodiment_name = cfg.embodiment
        base_jitter_xy_m = float(cfg.base_jitter_radius or 0.0)
        if base_jitter_xy_m < 0.0:
            raise ValueError("--base_jitter_radius must be non-negative")
        if not embodiment_name:
            if base_jitter_xy_m or getattr(cfg, "base_jitter_low", None) is not None:
                raise ValueError("--base_jitter_radius and --base_jitter_low/high require --embodiment")
            return None, None

        from isaaclab_arena.embodiments.common.arm_mode import ArmMode
        from isaaclab_arena.utils.pose import Pose

        if cfg.mimic:
            mimic_task = self._build_task(cfg)
            arm_mode_value = cfg.arm_mode or ArmMode.DUAL_ARM.value
            if mimic_task is None or mimic_task.get_mimic_env_cfg(ArmMode(arm_mode_value)) is None:
                task_name = getattr(cfg, "task", None)
                raise ValueError(
                    f"{self.name} --task {task_name!r} does not define a Mimic env cfg "
                    f"(available tasks: {tuple(type(self).available_tasks)})"
                )

        anchors = self._robot_base_anchors_for_args(cfg)
        assert anchors, f"{self.name} does not define robot base anchors; cannot mount an embodiment"
        anchor = cfg.anchor or self.default_anchor
        if anchor not in anchors:
            raise ValueError(f"{self.name} has no robot base anchor {anchor!r} (available: {tuple(anchors)})")

        register_dexbench_embodiments()
        embodiment_cls = self.asset_registry.get_asset_by_name(embodiment_name)
        embodiment_kwargs = self.embodiment_constructor_kwargs(cfg)
        optional_args = tuple(embodiment_kwargs)
        if cfg.ik_profile is None and cfg.source_demo_origin is not None:
            profile = ik_profile_for_source_origin(embodiment_cls, *cfg.source_demo_origin)
            if profile is not None:
                embodiment_kwargs["ik_profile"] = profile
        try:
            embodiment = embodiment_cls(**embodiment_kwargs)
        except TypeError as exc:
            rejected = [name for name in optional_args if f"'{name}'" in str(exc)]
            if "unexpected keyword argument" not in str(exc) or not rejected:
                raise
            raise ValueError(f"{embodiment_name} does not take --{rejected[0]}") from exc
        _set_camera_parallelism(embodiment, cfg)
        anchor_x, anchor_y, _anchor_z = anchors[anchor]
        forward_x, forward_y = type(self).embodiment_forward_axis_xy
        forward_yaw_rad = math.atan2(forward_y, forward_x)
        heading_deg = type(self).anchor_yaws_deg.get(anchor)
        if heading_deg is None:
            target_x, target_y = self._anchor_facing_targets_for_args(cfg)[anchor]
            target_yaw_rad = math.atan2(target_y - anchor_y, target_x - anchor_x)
        else:
            target_yaw_rad = math.radians(heading_deg)
        yaw_rad = target_yaw_rad - forward_yaw_rad
        base = getattr(embodiment_cls, "base_model", None)
        xyz_overrides = type(self).embodiment_root_xyz_overrides
        anchor_overrides = self._embodiment_anchor_overrides_for_args(cfg, base)
        if anchor in anchor_overrides:
            root_x, root_y, root_z = anchor_overrides[anchor]
        elif base in xyz_overrides:
            root_x, root_y, root_z = xyz_overrides[base]
        else:
            root_x, root_y = anchor_x, anchor_y
            root_z = type(self).embodiment_root_z_overrides.get(base, type(self).embodiment_root_z_m)
        base_offset_x = float(cfg.base_offset_x or 0.0)
        base_offset_y = float(cfg.base_offset_y or 0.0)
        if base_offset_x or base_offset_y:
            root_x += base_offset_x
            root_y += base_offset_y
            if heading_deg is None:
                yaw_rad = math.atan2(target_y - root_y, target_x - root_x) - forward_yaw_rad
        embodiment.set_initial_pose(
            Pose(
                position_xyz=(root_x, root_y, root_z),
                rotation_xyzw=rotate_quat_by_yaw((0.0, 0.0, 0.0, 1.0), yaw_rad),
            )
        )
        from dexbench_arena.variations.object_pose_variation import make_planar_pose_variation

        low, high = getattr(cfg, "base_jitter_low", None), getattr(cfg, "base_jitter_high", None)
        bounds = None
        if (low is None) != (high is None):
            raise ValueError("--base_jitter_low and --base_jitter_high go together")
        if low is not None:
            if base_jitter_xy_m:
                raise ValueError("--base_jitter_low/high replace --base_jitter_radius")
            (x0, y0, w0), (x1, y1, w1) = (tuple(float(v) for v in low), tuple(float(v) for v in high))
            bounds = ([x0, y0, math.radians(w0)], [x1, y1, math.radians(w1)])
        embodiment.add_variation(
            make_planar_pose_variation(
                "robot",
                name="base_pose",
                xy_half_extent_m=base_jitter_xy_m,
                **({"low": bounds[0], "high": bounds[1]} if bounds is not None else {}),
            )
        )

        if cfg.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(cfg.teleop_device)()
        else:
            teleop_device = None
        return embodiment, teleop_device

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """Declare the environment's flags: the fields of its config, under the same names and defaults."""
        add_hand_contact_material_cli_args(parser)
        add_fingertip_contact_cli_args(parser)
        add_joint_torque_cli_args(parser)
        _add_argument_if_missing(
            parser,
            "--embodiment",
            type=str,
            default=None,
            help="Optional embodiment asset name (e.g. dexbench_rby1_wuji2); omit for the scene-only shell.",
        )
        _add_argument_if_missing(
            parser, "--enable_cameras", action="store_true", default=argparse.SUPPRESS
        )
        _add_argument_if_missing(parser, "--teleop_device", type=str, default=None)
        _add_argument_if_missing(
            parser,
            "--start_pose",
            type=str,
            default=None,
            help="Named start pose overriding the task's default, from the embodiment's start_poses "
            "(RBY1-M Wuji2: replay_home | teleop_home | teleop_ready | rrc_home | rrc_ready | retail_ready | desk_ready | r3_carry).",
        )
        _add_argument_if_missing(
            parser,
            "--base_offset_x",
            "--base-offset-x",
            dest="base_offset_x",
            type=float,
            default=0.0,
            help="GA-S: offset the embodiment base X from its anchor [m]; yaw re-aims at the "
            "station's facing target. 0 keeps the anchor pose bit-identical.",
        )
        _add_argument_if_missing(
            parser,
            "--base_offset_y",
            "--base-offset-y",
            dest="base_offset_y",
            type=float,
            default=0.0,
            help="GA-S: offset the embodiment base Y from its anchor [m]; see --base_offset_x.",
        )
        _add_argument_if_missing(
            parser,
            "--base_jitter_radius",
            "--base-jitter-radius",
            dest="base_jitter_radius",
            type=float,
            default=0.0,
            help="GA-S: per-reset uniform XY half-extent [m] around the resolved embodiment "
            "base pose. The draw is recorded as <embodiment>.base_pose; unlike "
            "--base_offset_x/y, it is resampled on every reset.",
        )
        for bound in ("low", "high"):
            _add_argument_if_missing(
                parser,
                f"--base_jitter_{bound}",
                f"--base-jitter-{bound}",
                dest=f"base_jitter_{bound}",
                type=_base_draw_bound,
                metavar="DX,DY,DYAW",
                default=None,
                help=f"The {bound} end of a per-reset base draw that goes further to one side than the other: "
                "[dx, dy] [m] in the world, dyaw [deg]; with the other end, in place of --base_jitter_radius. "
                "Recorded as <embodiment>.base_pose.",
            )
        _add_argument_if_missing(
            parser,
            "--episode_length_s",
            "--episode-length-s",
            dest="episode_length_s",
            type=float,
            default=None,
            help="Episode timeout in simulated seconds for the selected task; unset keeps the task's own.",
        )
        _add_argument_if_missing(
            parser,
            "--record_settle",
            "--record-settle",
            dest="record_settle",
            action="store_true",
            help="Add a `settle` field to every episode row (did the scene come to rest, what ended the episode); "
            "what a condition recording sets.",
        )
        _add_argument_if_missing(
            parser,
            "--benchmark_replay",
            "--benchmark-replay",
            dest="benchmark_replay",
            action="store_true",
            help="A benchmark cell or sweep: refuse rows that leave an enabled variation live or do not record the "
            "build's object, whatever REPLAY_GUARD_MODE says.",
        )
        _add_argument_if_missing(
            parser,
            "--hide_replay_prims",
            "--hide-replay-prims",
            dest="hide_replay_prims",
            action="store_true",
            help="Hide the environment's replay_hidden_prim_patterns from every render, as rerecord_demos.py "
            "does, so an evaluation sees what the re-rendered training frames saw.",
        )
        _add_argument_if_missing(
            parser,
            "--teleop_target_marker",
            "--teleop-target-marker",
            dest="teleop_target_marker",
            action="store_true",
            help="Hover a marker over the task's target object, and draw the region a slot task (DB-R3) "
            "wants it in, green once it is seated there, for the operator in the headset or the Kit "
            "window. Visual only (no collider, not observed or recorded); not for camera recordings.",
        )
        _add_argument_if_missing(
            parser,
            "--physics_variants",
            "--physics-variants",
            dest="physics_variants",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Select the Newton physics runtime variant on every prop that carries it. "
            "--no-physics_variants spawns them from their neutral base, for an A/B run.",
        )
        _add_argument_if_missing(
            parser,
            "--newton_gravity_compensation",
            "--newton-gravity-compensation",
            dest="newton_gravity_compensation",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="On Newton, give every disable_gravity body mjc:gravcomp=1.0 (Newton does not read "
            "disable_gravity). --no-newton_gravity_compensation lets them fall, for an A/B run.",
        )
        _add_argument_if_missing(
            parser,
            "--newton_nconmax",
            "--newton-nconmax",
            dest="newton_nconmax",
            type=int,
            default=NEWTON_NCONMAX,
            help="Lower bound on MJWarp's contact budget on Newton (Arena's preset is 400; a smaller value is ignored). "
            "An environment's own Newton profile is applied afterwards and wins (E2 sets 1024).",
        )
        _add_argument_if_missing(
            parser,
            "--newton_njmax",
            "--newton-njmax",
            dest="newton_njmax",
            type=int,
            default=NEWTON_NJMAX,
            help="Lower bound on MJWarp's constraint budget on Newton (Arena's preset is 300; a smaller value is ignored). "
            "An environment's own Newton profile is applied afterwards and wins (E2 sets 1024).",
        )
        _add_argument_if_missing(
            parser,
            "--newton_cuda_graph",
            "--newton-cuda-graph",
            dest="newton_cuda_graph",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Capture Newton's step as a CUDA graph. --no-newton_cuda_graph steps eagerly: slower, but a "
            "session with many resets does not segfault freeing the previous model during capture.",
        )
        _add_argument_if_missing(
            parser,
            "--newton_solver",
            "--newton-solver",
            dest="newton_solver",
            choices=("cg", "newton"),
            default=None,
            help="MJWarp's constraint solver on Newton, in place of the environment's Newton profile "
            "(E2: cg, DB-L1: newton). On Newton 1.6.1 the newton solver went non-finite on recorded hand contacts"
            ".",
        )
        _add_argument_if_missing(
            parser,
            "--newton_num_substeps",
            "--newton-num-substeps",
            dest="newton_num_substeps",
            type=int,
            default=None,
            help="MJWarp substeps per 60 Hz physics step on Newton, in place of the environment's Newton profile "
            "(E2: 2, retail_v1: 8). Fewer is faster.",
        )
        _add_argument_if_missing(
            parser,
            "--newton_solver_tolerance",
            "--newton-solver-tolerance",
            dest="newton_solver_tolerance",
            type=float,
            default=None,
            help="MJWarp's constraint-solver convergence tolerance on Newton (E2: 1e-4, else Isaac Lab's 1e-6). The solver "
            "stops iterating once the residual is below it, so a larger value ends contact-heavy steps sooner.",
        )
        _add_argument_if_missing(
            parser,
            "--arm_mode",
            type=str,
            default="dual_arm",
            choices=("left", "right", "dual_arm"),
            help="Arm mode for the mounted embodiment.",
        )
        _add_argument_if_missing(
            parser,
            "--camera_scale",
            "--camera-scale",
            type=float,
            default=None,
            help=(
                "Embodiment camera raster scale with fixed FOV (Wuji2 default: 0.5, "
                "640x360; 0.25 selects 320x180; 1.0 restores 1280x720)."
            ),
        )
        _add_argument_if_missing(
            parser, "--rby1_control_hz", "--rby1-control-hz", dest="rby1_control_hz", type=int,
            default=None, choices=RBY1_CONTROL_HZ_CHOICES,
            help="RBY1 control/recording rate. Pass 30 to match the 30 Hz-trained checkpoints; "
            "unset follows DEXBENCH_CONTROL_HZ, else 30 Hz.",
        )
        _add_argument_if_missing(
            parser, "--camera_hz", "--camera-hz", dest="camera_hz", type=int, default=None,
            help="Take a new camera frame only at this rate, below the control rate (the steps a policy asked at "
            "it reads); unset takes one every control step.",
        )
        _add_argument_if_missing(
            parser, "--wrist-cameras", "--wrist_cameras",
            action=argparse.BooleanOptionalAction, default=None,
            help="GA: render bilateral palm GMSL2 RGB (requires --enable_cameras; on by default in Wuji2 rerecord); mounts remain when off.",
        )
        _add_argument_if_missing(
            parser,
            "--ik_profile",
            type=str,
            default=None,
            help="Pink IK task costs and gains for an IK embodiment, by name ('teleop', the default, smooths "
            "the operator's hand; 'generation' tracks recorded trajectories faithfully). An embodiment that solves no IK rejects the flag.",
        )
        _add_argument_if_missing(
            parser,
            "--show_ik_warnings",
            action="store_true",
            help="Print Pink IK solver warnings (IK embodiments only).",
        )
        _add_argument_if_missing(
            parser,
            "--mimic_boundaries",
            type=str,
            default=None,
            help="JSON table of stored subtask boundaries for annotate_demos.py --auto, indexed by source "
            "episode (see dexbench_arena/embodiments/common/mimic_boundaries.py).",
        )
        _add_argument_if_missing(
            parser,
            "--mimic_action_noise",
            type=float,
            default=None,
            help="Per-step Mimic action noise for every subtask (Mimic generation only); unset keeps the task's 0. "
            "With noise the generated files carry actions_clean beside the executed actions.",
        )
        _add_argument_if_missing(
            parser,
            "--mimic_keep_failed",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Keep a Mimic generation run's failed attempts in <output>_failed.hdf5 (--no-mimic_keep_failed "
            "drops them); unset keeps the task's choice.",
        )
        _add_argument_if_missing(
            parser, "--waist_ik", "--waist-ik", action="store_true", default=None,
            help="Enable torso participation in Pink IK for embodiments supporting waist_ik.",
        )
        _add_argument_if_missing(
            parser, "--waist_ik_max_lead_rad", type=float, default=None,
            help="Maximum torso command distance from measured joint angles [rad] (default: 0.05).",
        )
        _add_argument_if_missing(
            parser, "--head_action", "--head-action", action="store_true", default=None,
            help="Take the head out of Pink IK and command it directly after the IK columns (dexbench_rby1_wuji2: "
            "56-D), to replay a recording with a gaze of its own.",
        )
        if cls.robot_base_anchors_xyz:
            _add_argument_if_missing(
                parser,
                "--anchor",
                type=str,
                default=cls.default_anchor,
                choices=(*cls.robot_base_anchors_xyz, *cls.legacy_values.get("anchor", {})),
                help=cls.anchor_help,
            )
        if cls.available_tasks:
            _add_argument_if_missing(
                parser,
                "--task",
                type=str,
                default="none",
                choices=("none", *cls.available_tasks),
                help="Optional task defining success/termination conditions (default: scene shell, no task).",
            )
