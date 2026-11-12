# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Helpers for launching upstream Arena scripts with DexBench environments."""

from __future__ import annotations

import contextvars
import faulthandler
import importlib
import json
import os
import re
import runpy
import signal
import sys
from pathlib import Path

from dexbench_arena.app_launch import DEXBENCH_KIT_ARGS, with_dexbench_kit_args
from dexbench_arena.assets.physics_variants import apply_newton_usd_parse_guard
from dexbench_arena.benchmark.protocol import launcher_hook_variables
from dexbench_arena.evaluation.experiment_backend import experiment_run_collection_rows, experiment_run_command_line
from dexbench_arena.variations.collection_rows import CollectionRows, install_collection_recording


DEXBENCH_ROOT = Path(__file__).resolve().parents[2]
ARENA_ROOT = Path(
    os.environ.get("ISAACLAB_ARENA_PATH", DEXBENCH_ROOT / "submodule" / "isaaclab_arena")
).expanduser()
ARENA_PACKAGE_DIR = ARENA_ROOT / "isaaclab_arena"
ARENA_IM_SCRIPT_DIR = ARENA_PACKAGE_DIR / "scripts" / "imitation_learning"
ARENA_EVAL_SCRIPT_DIR = ARENA_PACKAGE_DIR / "evaluation"
ISAACLAB_ROOT = ARENA_ROOT / "submodules" / "IsaacLab"
EXPERIMENT_CONFIG_DIR = DEXBENCH_ROOT / "scripts" / "evaluation" / "experiment_configs"
BENCHMARK_MANIFEST = EXPERIMENT_CONFIG_DIR / "benchmark" / "benchmark_manifest.json"
COLLECTION_SCRIPTS = {"record_demos.py", "teleop.py"}
"""The scripts that take ``--collection_rows`` (``_take_collection_rows``)."""

ISAACLAB_SCRIPT_RELPATHS = {
    "annotate_demos.py": Path("scripts/imitation_learning/isaaclab_mimic/annotate_demos.py"),
    "generate_dataset.py": Path("scripts/imitation_learning/isaaclab_mimic/generate_dataset.py"),
    "record_demos.py": Path("scripts/tools/record_demos.py"),
    "replay_demos.py": Path("scripts/tools/replay_demos.py"),
    "teleop.py": Path("scripts/environments/teleoperation/teleop_se3_agent.py"),
}

TELEOP_CAPABLE_LAB_SCRIPTS = {"teleop.py", "record_demos.py"}
TASK_STATUS_PERIOD_S = 0.5
"""How often, in sim seconds, a teleop or recording session prints its task's status line (``[DexBench status]``):
the parts of the success term at its last evaluation (``dexbench_arena.tasks.predicates.status``), which the
collection station shows the pilot."""

OPENXR_KIT_ARGS = (
    "--/rtx-transient/resourcemanager/enableTextureStreaming=true",
    "--/rtx-transient/resourcemanager/texturestreaming/memoryBudget=0.25",
)

LAB_SCRIPTS_LOADING_NUMPY_BEFORE_LAUNCH = {"record_demos.py", "replay_demos.py", "teleop.py"}


def _register_dexbench_environments_with_arena(*, evaluation: bool = False, script_name: str | None = None) -> None:
    """Let Arena's own environment CLI find DexBench's environments."""

    import isaaclab_arena_environments.cli as environments_cli

    from dexbench_arena.cli import ensure_dexbench_environments_registered
    from dexbench_arena.evaluation.dataset_recording import (
        add_dataset_recording_cli_args,
        install_dataset_recording_patches,
    )

    upstream_ensure = environments_cli.ensure_environments_registered
    if not getattr(upstream_ensure, "_dexbench_registers", False):

        def ensure_environments_registered() -> None:
            upstream_ensure()
            ensure_dexbench_environments_registered()

        ensure_environments_registered._dexbench_registers = True
        ensure_environments_registered.__wrapped__ = upstream_ensure
        environments_cli.ensure_environments_registered = ensure_environments_registered

    if not evaluation:
        return

    upstream_parser = environments_cli.get_isaaclab_arena_environments_cli_parser
    if not getattr(upstream_parser, "_dexbench_runner_flags", False):

        def get_isaaclab_arena_environments_cli_parser(args_parser=None):
            if args_parser is None:
                from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser

                args_parser = get_isaaclab_arena_cli_parser()
            add_dataset_recording_cli_args(args_parser)
            return upstream_parser(args_parser)

        get_isaaclab_arena_environments_cli_parser._dexbench_runner_flags = True
        get_isaaclab_arena_environments_cli_parser.__wrapped__ = upstream_parser
        environments_cli.get_isaaclab_arena_environments_cli_parser = get_isaaclab_arena_environments_cli_parser

    install_dataset_recording_patches(script_name, sys.argv[1:], after_launch=_after_app_launch)


_AFTER_APP_LAUNCH: list = []


def _after_app_launch(hook) -> None:
    """Run ``hook`` once Kit is up: right after the next ``AppLauncher`` has started the app."""

    import isaaclab.app

    launcher_type = isaaclab.app.AppLauncher
    _AFTER_APP_LAUNCH.append(hook)
    upstream_init = launcher_type.__init__
    if getattr(upstream_init, "_dexbench_after_launch", False):
        return
    _refuse_launch_without_hooks(isaaclab.app)

    def __init__(self, *args, **kwargs):
        upstream_init(self, *args, **kwargs)
        while _AFTER_APP_LAUNCH:
            _AFTER_APP_LAUNCH.pop(0)()

    __init__._dexbench_after_launch = True
    __init__.__wrapped__ = upstream_init
    launcher_type.__init__ = __init__


def _refuse_launch_without_hooks(app_module) -> None:
    """Shadow ``app_module.launch_simulation`` so a launch with ``_after_app_launch`` hooks queued fails."""

    def launch_simulation(*args, **kwargs):
        if _AFTER_APP_LAUNCH:
            raise RuntimeError(
                f"{len(_AFTER_APP_LAUNCH)} DexBench post-launch hook(s) are queued, but this script starts Kit "
                "through isaaclab.app.launch_simulation (IsaacLab#8090), not AppLauncher, so they would "
                "never run: no recording provenance, no generation attempt log, no Newton annotate reset. "
                "Fix _after_app_launch in scripts/imitation_learning/_dexbench_arena_script.py to run them once "
                "launch_simulation has started its launchers (wrap it, or isaaclab_physx.app.KitLauncher.__init__) "
                "before taking an Isaac Lab with IsaacLab#8090."
            )
        from isaaclab.app.sim_launcher import launch_simulation as upstream

        return upstream(*args, **kwargs)

    app_module.launch_simulation = launch_simulation


def _install_teleop_trace_step_patch(script_name: str) -> None:
    """Patch gym.make so env.step writes robot-side telemetry and optional EEF markers."""

    from dexbench_arena.diagnostics import teleop_trace

    trace_enabled = teleop_trace.is_enabled()
    visualize_eef_obs = _env_flag("DEXBENCH_VISUALIZE_EEF_OBS")
    joint_stream_target = os.environ.get("DEXBENCH_PLOTJUGGLER_UDP")
    joint_log_path = os.environ.get("DEXBENCH_JOINT_LOG_CSV")
    if not trace_enabled and not visualize_eef_obs and not joint_stream_target and not joint_log_path:
        return

    if joint_stream_target:
        print(f"[DexBench PlotJuggler] Streaming per-step joint state as UDP JSON to {joint_stream_target}")
    if joint_log_path:
        print(f"[DexBench Joint Log] Writing per-step joint telemetry CSV to {joint_log_path}")

    if trace_enabled:
        truncate = os.environ.get("DEXBENCH_TELEOP_TRACE_TRUNCATE", "1").lower() not in {"0", "false", "no", "off"}
        path = teleop_trace.start_session({"script": script_name, "cwd": str(Path.cwd())}, truncate=truncate)
        if path is not None:
            print(f"[DexBench Teleop Trace] Writing trace to {path}")
    if visualize_eef_obs:
        print("[DexBench EEF Obs] Visualizing policy EEF observation frames.")

    def trace(env) -> None:
        try:
            _patch_env_step_for_trace(env)
        except Exception as exc:
            print(f"[DexBench Teleop Trace] Failed to patch env.step: {exc}")

    _on_env_made(trace)


def _refuse_launcher_hooks(script_name: str) -> None:
    """An evaluation runs what its experiment file declares, so it refuses the teleop and recording hooks
    (``protocol.LAUNCHER_HOOK_VARIABLES`` and every ``DEXBENCH_WATER_*``) rather than installing them: a variable
    left in the shell would otherwise add a second water to DB-H2's, hide prims or change the render of a cell,
    recorded nowhere."""
    names = launcher_hook_variables(os.environ)
    if names:
        raise RuntimeError(f"{script_name} is an evaluation: unset {', '.join(names)} (teleop and recording hooks of "
                           "the launcher, which no cell declares)")


def _install_record_on_reset_patch() -> None:
    """Opt-in: write the episode when the operator resets, instead of dropping it."""

    if os.environ.get("DEXBENCH_RECORD_EXPORT_ON_RESET") != "1":
        return
    _on_env_made(_patch_recorder_reset_for_export)


def _patch_recorder_reset_for_export(env) -> None:
    recorder_manager = getattr(env, "recorder_manager", None)
    if recorder_manager is None or getattr(recorder_manager, "_dexbench_record_on_reset", False):
        return
    original_reset = recorder_manager.reset

    def reset_with_export(env_ids=None):
        try:
            ids = (
                list(range(env.num_envs))
                if env_ids is None
                else (env_ids.tolist() if hasattr(env_ids, "tolist") else list(env_ids))
            )
            episodes = getattr(recorder_manager, "_episodes", {})
            pending = [i for i in ids if i in episodes and not episodes[i].is_empty()]
            if pending:
                from isaaclab.managers.recorder_manager import DatasetExportMode

                recorder_manager.record_pre_reset(pending, force_export_or_skip=False)
                succeeded = [bool(episodes[i].success) for i in pending]
                mode = recorder_manager.cfg.dataset_export_mode
                recorder_manager.cfg.dataset_export_mode = DatasetExportMode.EXPORT_ALL
                try:
                    recorder_manager.export_episodes(pending)
                finally:
                    recorder_manager.cfg.dataset_export_mode = mode
                handler = recorder_manager._dataset_file_handler
                n = handler.get_num_episodes()
                names = [f"demo_{n - len(pending) + k}" for k in range(len(pending))]
                for name in names:
                    handler._hdf5_data_group[name].attrs["operator_reset"] = True
                handler.flush()
                tracked = getattr(env, "dexbench_exported_demos", None)
                if tracked is not None:
                    tracked.update(zip(pending, names))
                where = getattr(getattr(handler, "_hdf5_file_stream", None), "filename", "dataset")
                print(f"[DexBench Record] saved {', '.join(names)} at an operator reset (success "
                      f"{', '.join(map(str, succeeded))}; {n} in {where}); recording demo_{n} next", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[DexBench Record] export on reset failed: {exc}", flush=True)
        return original_reset(env_ids)

    recorder_manager.reset = reset_with_export
    recorder_manager._dexbench_record_on_reset = True
    print("[DexBench Record] resets will export the episode first", flush=True)


def _install_render_cfg_patch() -> None:
    """Turn on RTX translucency/reflections when the scene has glassware."""

    keys = [
        key
        for key, flag in (
            ("/rtx/translucency/enabled", "DEXBENCH_RENDER_TRANSLUCENCY"),
            ("/rtx/reflections/enabled", "DEXBENCH_RENDER_REFLECTIONS"),
        )
        if _env_flag(flag)
    ]
    if not keys:
        return

    def apply_render_settings(env) -> None:
        try:
            import carb

            settings = carb.settings.get_settings()
            for key in keys:
                settings.set(key, True)
            print("[DexBench Render] carb: " + ", ".join(f"{key}={settings.get(key)}" for key in keys))
        except Exception as exc:  # noqa: BLE001
            print(f"[DexBench Render] carb settings not applied: {exc}")

    _on_env_made(apply_render_settings)


def _install_water_pour_sidecar_patch() -> None:
    """Attach the Newton MPM water sidecar to a teleop or recording launcher's env."""

    if not _env_flag("DEXBENCH_WATER"):
        return

    def attach(env) -> None:
        if env.num_envs != 1:
            raise RuntimeError(f"DEXBENCH_WATER=1 seeds env 0's water only and this env has {env.num_envs}: "
                               "run the teleop water with --num_envs 1")
        try:
            _patch_env_for_water_sidecar(env)
        except Exception as exc:  # noqa: BLE001
            print(f"[DexBench Water] Failed to attach water sidecar: {exc}")

    _on_env_made(attach)
    print("[DexBench Water] MPM water sidecar will attach on env creation (DEXBENCH_WATER=1)")


def _patch_env_for_water_sidecar(env) -> None:
    if getattr(env, "_dexbench_water_patched", False):
        return

    holder: dict[str, object] = {"sidecar": None}
    original_step = env.step
    original_reset = env.reset
    original_reset_to = getattr(env, "reset_to", None)
    report_every = _read_int_env("DEXBENCH_WATER_REPORT_EVERY", 0)
    state = {"n": 0}

    def reseed() -> None:
        """Seed the water where the reset put the jug (built at the first reset)."""
        _hide = [n.strip() for n in os.environ.get("DEXBENCH_HIDE_PRIMS", "").split(",") if n.strip()]
        if _hide:
            try:
                import omni.usd

                _stage = omni.usd.get_context().get_stage()
                for _pr in _stage.Traverse():
                    if _pr.GetName() in _hide:
                        _pr.SetActive(False)
                        print(f"[DexBench Water] deactivated prim {_pr.GetPath()}", flush=True)
            except Exception as _exc:  # noqa: BLE001
                print(f"[DexBench Water] hide prims failed: {_exc}", flush=True)
        try:
            if holder["sidecar"] is None:
                from dexbench_arena.fluids.water_pour_setup import build_water_pour_sidecar

                holder["sidecar"] = build_water_pour_sidecar(env)
            else:
                holder["sidecar"].reset()
        except Exception as exc:  # noqa: BLE001
            print(f"[DexBench Water] sidecar reset/build failed: {exc}")

    def water_reset(*args, **kwargs):
        result = original_reset(*args, **kwargs)
        reseed()
        return result

    def water_reset_to(*args, **kwargs):
        result = original_reset_to(*args, **kwargs)
        reseed()
        return result

    def water_step(action, *args, **kwargs):
        result = original_step(action, *args, **kwargs)
        sidecar = holder["sidecar"]
        if sidecar is not None and isinstance(result, tuple) and len(result) == 5 and bool((result[2] | result[3])[0]):
            reseed()
        elif sidecar is not None:
            try:
                sidecar.step(getattr(env, "step_dt", None))
                if report_every:
                    state["n"] += 1
                    if state["n"] >= report_every:
                        state["n"] = 0
                        m = sidecar.metrics()
                        print(
                            f"[DexBench Water] caught={m['caught']}/{m['total']} "
                            f"({m['caught_frac'] * 100:.1f}%) spilled={m['spilled']}",
                            flush=True,
                        )
            except Exception as exc:  # noqa: BLE001
                print(f"[DexBench Water] sidecar step failed: {exc}")
                holder["sidecar"] = None
        return result

    env.step = water_step
    env.reset = water_reset
    if original_reset_to is not None:
        env.reset_to = water_reset_to
    env._dexbench_water_patched = True
    env.dexbench_water = holder


_ON_ENV_MADE: list = []


def _print_task_status(env) -> None:
    """Print the task's status line every :data:`TASK_STATUS_PERIOD_S` of sim time while the session steps."""
    import json as _json

    from dexbench_arena.tasks.predicates import status

    original_step = env.step
    period_steps = max(1, round(TASK_STATUS_PERIOD_S / env.step_dt))
    counter = {"steps": 0}

    def step_and_print(*args, **kwargs):
        result = original_step(*args, **kwargs)
        counter["steps"] += 1
        if counter["steps"] % period_steps == 0:
            latest = status.latest(env)
            if latest is not None:
                print(f"[DexBench status] {_json.dumps(latest, separators=(',', ':'))}", flush=True)
        return result

    env.step = step_and_print


def _on_env_made(hook) -> None:
    """Run ``hook(env)`` on the unwrapped env of every ``gym.make`` the upstream script calls."""

    if not _ON_ENV_MADE:
        _after_app_launch(_patch_gym_make)
    _ON_ENV_MADE.append(hook)


def _patch_gym_make() -> None:
    import gymnasium as gym

    original_make = gym.make

    def make(*args, **kwargs):
        env = original_make(*args, **kwargs)
        for hook in _ON_ENV_MADE:
            hook(env.unwrapped)
        return env

    gym.make = make


def _write_episode_records_of_recording(env) -> None:
    """Write Arena's episode JSONL beside the dataset Isaac Lab's ``record_demos.py`` records."""

    from dexbench_arena.recording.rerecord_metadata import episode_records_path, finish_episode_records, write_episode_records

    from dexbench_arena.environments.manipulation.scene_environment_base import exported_demos

    recorders = env.cfg.recorders
    dataset = Path(recorders.dataset_export_dir_path) / f"{recorders.dataset_filename}.hdf5"
    path = write_episode_records(env, episode_records_path(dataset), job_name="record_demos")
    exported = exported_demos(env)
    print(f"[DexBench Recording] episode records -> {path} (every take, discarded ones included)")
    recorder_manager = env.recorder_manager
    export = recorder_manager.export_episodes

    def export_and_mark(*args, **kwargs) -> None:
        count = recorder_manager.exported_successful_episode_count
        export(*args, **kwargs)
        count_after = recorder_manager.exported_successful_episode_count
        if count_after > count:
            exported[0] = f"demo_{count_after - 1}"

    recorder_manager.export_episodes = export_and_mark
    close = env.close

    def close_after_the_last_row() -> None:
        env.close = close
        finish_episode_records(env)
        close()

    env.close = close_after_the_last_row


def _patch_env_step_for_trace(env) -> None:
    from dexbench_arena.diagnostics import teleop_trace

    if getattr(env, "_dexbench_trace_step_patched", False):
        return

    original_step = env.step
    step_counter = {"value": 0}
    eef_visualizer = _EefObservationVisualizer(env) if _env_flag("DEXBENCH_VISUALIZE_EEF_OBS") else None
    joint_stream_target = os.environ.get("DEXBENCH_PLOTJUGGLER_UDP")
    joint_streamer = _JointStreamPublisher(env, joint_stream_target) if joint_stream_target else None
    joint_log_path = os.environ.get("DEXBENCH_JOINT_LOG_CSV")
    joint_logger = _JointCsvLogger(env, joint_log_path) if joint_log_path else None

    def traced_step(action, *args, **kwargs):
        result = original_step(action, *args, **kwargs)
        step = step_counter["value"]
        step_counter["value"] += 1
        if joint_streamer is not None:
            joint_streamer.publish()
        if joint_logger is not None:
            joint_logger.publish()

        obs = result[0] if isinstance(result, tuple) and result else getattr(env, "obs_buf", None)
        policy_obs = _extract_policy_observation(obs)
        if eef_visualizer is not None:
            eef_visualizer.visualize(policy_obs)
        if teleop_trace.is_enabled():
            robot_metadata = _extract_robot_metadata(env)
            teleop_trace.record_event(
                "robot_step",
                {
                    "action": _build_action_payload(action),
                    "action_terms": _extract_action_terms(env),
                    "policy_obs": policy_obs,
                    "robot": robot_metadata,
                },
                step=step,
            )
        return result

    env.step = traced_step
    env._dexbench_trace_step_patched = True


def _build_joint_payload(
    env,
    joint_names: list[str] | None,
    previous_vectors: dict[str, list[float]] | None,
) -> tuple[dict[str, float], list[str], dict[str, list[float]]]:
    robot = env.scene["robot"]
    if joint_names is None:
        joint_names = list(robot.joint_names)
    step_dt = float(getattr(env, "step_dt", 0.02))
    step_dt = step_dt if step_dt > 0.0 else 0.02
    vectors = {
        "pos": _first_env_row(robot.data.joint_pos),
        "vel": _first_env_row(robot.data.joint_vel),
        "target": _first_env_row(robot.data.joint_pos_target),
    }
    payload = {
        "t": float(getattr(env, "common_step_counter", 0)) * step_dt,
        "step": float(getattr(env, "common_step_counter", 0)),
        "episode_step": int(env.episode_length_buf[0]),
    }
    for prefix, values in (
        ("joint_pos", vectors["pos"]),
        ("joint_vel", vectors["vel"]),
        ("joint_target", vectors["target"]),
    ):
        for name, value in zip(joint_names, values):
            payload[f"{prefix}/{name}"] = float(value)
    _add_joint_jitter_payload(payload, joint_names, vectors, previous_vectors, step_dt)
    return payload, joint_names, vectors


def _add_joint_jitter_payload(
    payload: dict[str, float],
    joint_names: list[str],
    vectors: dict[str, list[float]],
    previous_vectors: dict[str, list[float]] | None,
    step_dt: float,
) -> None:
    pos = vectors["pos"]
    vel = vectors["vel"]
    target = vectors["target"]
    previous_vel = previous_vectors.get("vel", []) if previous_vectors else []
    previous_target = previous_vectors.get("target", []) if previous_vectors else []
    previous_target_rate = previous_vectors.get("target_rate", []) if previous_vectors else []

    tracking_error_abs_values: list[float] = []
    target_step_abs_values: list[float] = []
    target_rate_abs_values: list[float] = []
    target_accel_abs_values: list[float] = []
    velocity_step_abs_values: list[float] = []
    velocity_accel_abs_values: list[float] = []
    current_target_rate: list[float] = []

    for index, name in enumerate(joint_names):
        current_pos = _at(pos, index)
        current_vel = _at(vel, index)
        current_target = _at(target, index)

        tracking_error_abs = abs(current_target - current_pos)
        target_step = current_target - _at(previous_target, index, current_target)
        target_step_abs = abs(target_step)
        target_rate = target_step / step_dt
        target_rate_abs = abs(target_rate)
        target_accel_abs = abs(target_rate - _at(previous_target_rate, index, target_rate)) / step_dt
        velocity_step = current_vel - _at(previous_vel, index, current_vel)
        velocity_step_abs = abs(velocity_step)
        velocity_accel_abs = velocity_step_abs / step_dt

        current_target_rate.append(target_rate)
        tracking_error_abs_values.append(tracking_error_abs)
        target_step_abs_values.append(target_step_abs)
        target_rate_abs_values.append(target_rate_abs)
        target_accel_abs_values.append(target_accel_abs)
        velocity_step_abs_values.append(velocity_step_abs)
        velocity_accel_abs_values.append(velocity_accel_abs)

        payload[f"jitter/tracking_error_abs/{name}"] = tracking_error_abs
        payload[f"jitter/target_step_abs/{name}"] = target_step_abs
        payload[f"jitter/target_rate_abs/{name}"] = target_rate_abs
        payload[f"jitter/target_accel_abs/{name}"] = target_accel_abs
        payload[f"jitter/velocity_step_abs/{name}"] = velocity_step_abs
        payload[f"jitter/velocity_accel_abs/{name}"] = velocity_accel_abs

    vectors["target_rate"] = current_target_rate
    for name, values in (
        ("tracking_error_abs", tracking_error_abs_values),
        ("target_step_abs", target_step_abs_values),
        ("target_rate_abs", target_rate_abs_values),
        ("target_accel_abs", target_accel_abs_values),
        ("velocity_step_abs", velocity_step_abs_values),
        ("velocity_accel_abs", velocity_accel_abs_values),
    ):
        payload[f"jitter_summary/{name}_mean"] = _mean(values)
        payload[f"jitter_summary/{name}_rms"] = _rms(values)
        payload[f"jitter_summary/{name}_max"] = max(values) if values else 0.0


def _at(values: list[float], index: int, default: float = 0.0) -> float:
    if index >= len(values):
        return default
    return float(values[index])


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _rms(values: list[float]) -> float:
    if not values:
        return 0.0
    return (sum(value * value for value in values) / len(values)) ** 0.5


def _first_env_row(value) -> list[float]:
    import torch

    if not isinstance(value, torch.Tensor):
        import warp as wp

        value = wp.to_torch(value)
    return value[0].detach().cpu().tolist()


class _JointCsvLogger:
    """Write per-step joint telemetry to CSV for offline plotting."""

    def __init__(self, env, path: str) -> None:
        import atexit
        import csv
        from pathlib import Path

        self._env = env
        self._path = Path(path).expanduser()
        self._csv = csv
        self._file = None
        self._writer = None
        self._joint_names: list[str] | None = None
        self._previous_vectors: dict[str, list[float]] | None = None
        self._failed = False
        self._rows_written = 0
        self._flush_every = max(1, _read_int_env("DEXBENCH_JOINT_LOG_FLUSH_EVERY", 50))
        atexit.register(self.close)

    def publish(self) -> None:
        if self._failed:
            return
        try:
            payload, self._joint_names, self._previous_vectors = _build_joint_payload(
                self._env, self._joint_names, self._previous_vectors
            )
            if self._writer is None:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                self._file = self._path.open("w", encoding="utf-8", newline="")
                self._writer = self._csv.DictWriter(self._file, fieldnames=list(payload.keys()))
                self._writer.writeheader()
            self._writer.writerow(payload)
            self._rows_written += 1
            if self._rows_written == 1:
                print(
                    "[DexBench Joint Log] Wrote first joint sample "
                    f"({len(self._joint_names or [])} joints) to {self._path}"
                )
            if self._rows_written % self._flush_every == 0 and self._file is not None:
                self._file.flush()
        except Exception as exc:
            self._failed = True
            print(f"[DexBench Joint Log] Joint CSV logging disabled after error: {exc}")
            self.close()

    def close(self) -> None:
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None


class _JointStreamPublisher:
    """Stream per-step joint state as UDP JSON for PlotJuggler."""

    def __init__(self, env, target: str) -> None:
        import socket

        host, _, port = str(target).rpartition(":")
        self._address = (host or "127.0.0.1", int(port))
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._env = env
        self._joint_names: list[str] | None = None
        self._previous_vectors: dict[str, list[float]] | None = None
        self._failed = False
        self._sent_first_sample = False

    def publish(self) -> None:
        if self._failed:
            return
        import json

        try:
            payload, self._joint_names, self._previous_vectors = _build_joint_payload(
                self._env, self._joint_names, self._previous_vectors
            )
            self._socket.sendto(json.dumps(payload).encode("utf-8"), self._address)
            if not self._sent_first_sample:
                self._sent_first_sample = True
                print(
                    "[DexBench PlotJuggler] Sent first joint sample "
                    f"({len(self._joint_names)} joints, flat JSON fields) to {self._address[0]}:{self._address[1]}"
                )
        except Exception as exc:
            self._failed = True
            print(f"[DexBench PlotJuggler] Joint stream disabled after error: {exc}")


def _resolve_openxr_action_order(action_dim: int) -> tuple[str, ...]:
    """Name the OpenXR action columns when the action has the Wuji2 layout's width."""

    from dexbench_arena.embodiments.rby1_wuji2.constants import RBY1_WUJI2_OPENXR_ACTION_ORDER

    return RBY1_WUJI2_OPENXR_ACTION_ORDER if len(RBY1_WUJI2_OPENXR_ACTION_ORDER) == action_dim else ()


def _build_action_payload(action) -> dict:
    from dexbench_arena.diagnostics import teleop_trace

    values = teleop_trace.to_jsonable(_first_env_value(action))
    if not isinstance(values, list):
        values = [values]

    action_order = _resolve_openxr_action_order(len(values))

    named = {}
    if action_order:
        named = {name: values[index] for index, name in enumerate(action_order)}

    hand_named = {name: named[name] for name in action_order[14:]}
    left_hand = {
        name: value
        for name, value in hand_named.items()
        if name.startswith("left_")
    }
    right_hand = {
        name: value
        for name, value in hand_named.items()
        if name.startswith("right_")
    }
    return {
        "values": values,
        "order": list(action_order),
        "named": named,
        "left_wrist_pose": values[0:7],
        "right_wrist_pose": values[7:14],
        "left_hand": left_hand,
        "right_hand": right_hand,
    }


def _extract_policy_observation(obs) -> dict:
    policy = None
    if isinstance(obs, dict):
        policy = obs.get("policy", obs)
    if not isinstance(policy, dict):
        return {}

    keys = (
        "actions",
        "robot_joint_pos",
        "robot_joint_vel",
        "robot_links_state",
        "left_eef_pos",
        "left_eef_quat",
        "right_eef_pos",
        "right_eef_quat",
        "hand_joint_state",
        "head_joint_state",
    )
    return {key: _first_env_value(policy[key]) for key in keys if key in policy}


def _extract_robot_metadata(env) -> dict:
    try:
        robot = env.scene["robot"]
    except Exception:
        return {}

    return {
        "joint_names": list(getattr(robot, "joint_names", [])),
        "body_names": list(getattr(robot, "body_names", [])),
    }


def _extract_action_terms(env) -> dict:
    manager = getattr(env, "action_manager", None)
    if manager is None:
        return {}

    terms = getattr(manager, "_terms", None)
    if terms is None:
        terms = getattr(manager, "terms", None)
    if not isinstance(terms, dict):
        return {}

    payload = {}
    for term_name, term in terms.items():
        term_payload = {}
        for attr_name in ("raw_actions", "processed_actions"):
            value = getattr(term, attr_name, None)
            if value is not None:
                term_payload[attr_name] = _first_env_value(value)
        for attr_name in ("_isaaclab_controlled_joint_names", "_hand_joint_names", "_controlled_joint_names"):
            value = getattr(term, attr_name, None)
            if value is not None:
                term_payload[attr_name.removeprefix("_")] = list(value)
        if term_payload:
            payload[str(term_name)] = term_payload
    return payload


def _first_env_value(value):
    shape = getattr(value, "shape", None)
    ndim = getattr(value, "ndim", None)
    if shape is not None and ndim is not None:
        try:
            if ndim > 1 and int(shape[0]) > 0:
                return value[0]
        except Exception:
            return value
    return value


class _EefObservationVisualizer:
    """Draw frame markers at observed EEF poses from policy observations."""

    def __init__(self, env) -> None:
        self._env = env
        self._enabled = True
        self._left_marker = None
        self._right_marker = None

    def visualize(self, policy_obs: dict) -> None:
        if not self._enabled:
            return

        try:
            import numpy as np

            if not self._ensure_markers():
                return
            origin = self._env_origin()
            for side, marker in (("left", self._left_marker), ("right", self._right_marker)):
                pos = _array3(policy_obs.get(f"{side}_eef_pos"))
                quat = _array4(policy_obs.get(f"{side}_eef_quat"))
                if pos is None or quat is None:
                    continue
                marker.visualize(
                    translations=np.asarray([pos + origin], dtype=np.float32),
                    orientations=np.asarray([quat], dtype=np.float32),
                )
        except Exception as exc:
            self._enabled = False
            print(f"[DexBench EEF Obs] Disabling EEF observation markers after error: {exc}")

    def _ensure_markers(self) -> bool:
        if self._left_marker is not None and self._right_marker is not None:
            return True

        from isaaclab.markers import VisualizationMarkers
        from isaaclab.markers.config import FRAME_MARKER_CFG

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.075, 0.075, 0.075)
        self._left_marker = VisualizationMarkers(
            marker_cfg.replace(prim_path="/Visuals/dexbench_obs_left_eef_frame")
        )
        self._right_marker = VisualizationMarkers(
            marker_cfg.replace(prim_path="/Visuals/dexbench_obs_right_eef_frame")
        )
        return True

    def _env_origin(self):
        import numpy as np

        try:
            origin = self._env.scene.env_origins[0]
            detach = getattr(origin, "detach", None)
            if callable(detach):
                origin = origin.detach().cpu().numpy()
            return np.asarray(origin, dtype=np.float32)
        except Exception:
            return np.zeros(3, dtype=np.float32)


def _array3(value):
    arr = _numpy_array(value)
    if arr is None or arr.size < 3:
        return None
    return arr.reshape(-1)[:3]


def _array4(value):
    arr = _numpy_array(value)
    if arr is None or arr.size < 4:
        return None
    return arr.reshape(-1)[:4]


def _numpy_array(value):
    if value is None:
        return None
    import numpy as np

    value = _first_env_value(value)
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


def _read_int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _install_stack_dump_signal() -> None:
    signal_name = os.environ.get("DEXBENCH_STACK_DUMP_SIGNAL")
    if not signal_name:
        return

    normalized = signal_name.upper()
    if not normalized.startswith("SIG"):
        normalized = f"SIG{normalized}"
    dump_signal = getattr(signal, normalized, None)
    if dump_signal is None:
        print(f"[DexBench Stack Dump] Unknown signal: {signal_name}")
        return

    faulthandler.register(dump_signal, file=sys.stderr, all_threads=True, chain=False)
    print(f"[DexBench Stack Dump] Registered {normalized} for Python stack dumps.")


def _install_upstream_script_patches(script_name: str, *, patch_environment_cli: bool, evaluation: bool) -> None:
    """The patches every upstream entrypoint gets before it runs, in launch order."""

    from dexbench_arena.contact_materials import validate_legacy_hand_friction_environment
    from dexbench_arena.app_launch import stop_unneeded_viewport_rendering

    validate_legacy_hand_friction_environment()
    apply_newton_usd_parse_guard()
    _after_app_launch(stop_unneeded_viewport_rendering)
    _install_stack_dump_signal()
    if patch_environment_cli:
        _register_dexbench_environments_with_arena(evaluation=evaluation, script_name=script_name)
    _install_teleop_trace_step_patch(script_name)
    if script_name in TELEOP_CAPABLE_LAB_SCRIPTS:
        _on_env_made(_print_task_status)
    if evaluation:
        _refuse_launcher_hooks(script_name)
    else:
        _install_water_pour_sidecar_patch()
        _install_render_cfg_patch()
        _install_record_on_reset_patch()
    if script_name == "record_demos.py":
        _on_env_made(_write_episode_records_of_recording)


def _run_upstream_script(
    script_path: Path, script_name: str, *, patch_environment_cli: bool, evaluation: bool = False
) -> None:
    """Install DexBench patches, then run an upstream Arena script as ``__main__``."""

    _install_upstream_script_patches(script_name, patch_environment_cli=patch_environment_cli, evaluation=evaluation)

    if not script_path.exists():
        raise FileNotFoundError(f"Missing upstream Arena script: {script_path}")

    sys.argv[1:] = with_kit_args_first(sys.argv[1:])
    runpy.run_path(str(script_path), run_name="__main__")


def _flag_groups(tokens: list[str]) -> list[list[str]]:
    """``tokens`` as ``[flag, value?]`` groups: a token not starting with ``--`` is the value of the flag before it."""
    groups: list[list[str]] = []
    for token in tokens:
        if token.startswith("--") or not groups:
            groups.append([token])
        else:
            groups[-1].append(token)
    return groups


def expand_experiment_run(argv: list[str]) -> list[str]:
    """Replace ``--experiment <yaml> --run <name>`` on ``argv`` with that Run's environment command line."""
    argv = list(argv)
    index = next((i for i, token in enumerate(argv) if token == "--experiment" or token.startswith("--experiment=")), None)
    if index is None:
        return argv
    if argv[index] != "--experiment":
        argv[index : index + 1] = ["--experiment", argv[index].split("=", 1)[1]]
    if argv[index + 2 : index + 3] != ["--run"] or len(argv) < index + 4:
        raise SystemExit("--experiment <yaml> must be followed by --run <name>, then any environment overrides")
    path = Path(argv[index + 1])
    if not path.exists() and (EXPERIMENT_CONFIG_DIR / path).exists():
        path = EXPERIMENT_CONFIG_DIR / path
    head, run, overrides = argv[:index], argv[index + 3], argv[index + 4 :]
    run_argv = experiment_run_command_line(path, run, embodiment=_effective_flag_value(head, overrides, "--embodiment"))
    rows = experiment_run_collection_rows(path, run)
    if rows is not None and not _flag_present([*head, *overrides], [], "--collection_rows"):
        head = ["--collection_rows", rows, *head]
    env_index = 2 if run_argv[0] == "--presets" else 0
    presets, env_name, fields = run_argv[:env_index], run_argv[env_index], run_argv[env_index + 1 :]
    if env_name in head:
        named = head.index(env_name)
        head, overrides = head[:named], [*head[named + 1 :], *overrides]
    other = next((token for token in head if _ENVIRONMENT_NAME.fullmatch(token)), None)
    if other is not None:
        raise SystemExit(
            f"{other} is named before --experiment, but Run {run!r} of {path} builds {env_name}; "
            "drop the environment name, the Run gives it"
        )
    if _flag_present(overrides, [], "--presets"):
        raise SystemExit(
            "--presets is a launcher flag: Arena reads it before the environment name and the environment "
            "parser drops it after. Put it before --experiment."
        )
    if _flag_present(head, [], "--presets"):
        presets = []
    overridden = {token.split("=", 1)[0] for token in overrides}
    split = next((i for i, token in enumerate(fields) if "=" in token and not token.startswith("-")), len(fields))
    kept = [token for group in _flag_groups(fields[:split]) if group[0].split("=", 1)[0] not in overridden for token in group]
    kept += [token for token in fields[split:] if token.split("=", 1)[0] not in overridden]
    return [*head, *presets, env_name, *kept, *overrides]


def run_upstream_imitation_script(script_name: str, *, patch_environment_cli: bool = True) -> None:
    """Run an upstream IsaacLab-Arena imitation-learning script from DexBench."""

    sys.argv[1:] = expand_experiment_run(sys.argv[1:])
    sys.argv[1:], collection = _take_collection_rows(script_name, sys.argv[1:])

    if script_name == "record_demos.py" and "--enable_cameras" not in sys.argv:
        print(
            "\n[DexBench Recording WARNING] --enable_cameras is absent. "
            "The HDF5 file will contain state/action data only and no camera_obs frames.\n",
            file=sys.stderr,
            flush=True,
        )

    if script_name in ISAACLAB_SCRIPT_RELPATHS:
        _run_isaaclab_interop_script(
            ISAACLAB_ROOT / ISAACLAB_SCRIPT_RELPATHS[script_name], script_name, collection
        )
        return
    if collection is not None:
        raise SystemExit(f"{script_name} does not take --collection_rows")

    _run_upstream_script(
        ARENA_IM_SCRIPT_DIR / script_name, script_name, patch_environment_cli=patch_environment_cli
    )


def strip_collection_flags(argv: list[str]) -> tuple[list[str], list[str | None], list[str | None]]:
    """``argv`` without ``--collection_rows``, ``--collection_first_row`` and ``--collection_requeue_file``, and the
    values the first two were given (``_take_requeue_file`` reads the third)."""
    values = _flag_values(argv, "--collection_rows")
    firsts = _flag_values(argv, "--collection_first_row")
    argv, _ = _take_flag(argv, "--collection_rows", has_value=True)
    argv, _ = _take_flag(argv, "--collection_first_row", has_value=True)
    argv, _ = _take_flag(argv, "--collection_requeue_file", has_value=True)
    return argv, values, firsts


def _take_requeue_file(argv: list[str]) -> tuple[list[str], str | None]:
    """``argv`` without ``--collection_requeue_file <path>``, and the path (``CollectionRows.requeue_path``)."""
    values = _flag_values(argv, "--collection_requeue_file")
    argv, _ = _take_flag(argv, "--collection_requeue_file", has_value=True)
    if None in values or len(set(values)) > 1:
        raise SystemExit(f"--collection_requeue_file names one file; got {values}")
    return argv, (values[-1] if values else None)


def _take_collection_rows(script_name: str, argv: list[str]) -> tuple[list[str], CollectionRows | None]:
    """Take ``--collection_rows <file>`` and ``--collection_first_row <i>`` off ``argv``: the session collects from
    a recorded start-state dump (``dexbench_arena.variations.collection_rows``)."""
    argv, requeue = _take_requeue_file(argv)
    argv, values, firsts = strip_collection_flags(argv)
    if not values:
        if firsts or requeue:
            raise SystemExit("--collection_first_row and --collection_requeue_file need --collection_rows")
        return argv, None
    if script_name not in COLLECTION_SCRIPTS:
        raise SystemExit(f"{script_name} does not take --collection_rows (only {sorted(COLLECTION_SCRIPTS)})")
    if None in values or len(set(values)) > 1 or None in firsts:
        raise SystemExit(f"--collection_rows names one file and --collection_first_row one row; got {values} and {firsts}")
    manifest = json.loads(BENCHMARK_MANIFEST.read_text()) if BENCHMARK_MANIFEST.is_file() else None
    try:
        first_row = int(firsts[-1]) if firsts else 0
        rows = CollectionRows.read(values[0], first_row, manifest, requeue_path=requeue)
    except ValueError as error:
        raise SystemExit(f"--collection_rows: {error}") from None
    remaining = rows.rows - rows.first_row + len(rows.requeued())
    if script_name == "record_demos.py":
        asked = [value for value in _flag_values(argv, "--num_demos") if value is not None]
        if not asked:
            argv = ["--num_demos", str(remaining), *argv]
        elif not 0 < int(asked[-1]) <= remaining:
            raise SystemExit(f"--num_demos {asked[-1]}: a collection from {rows.path} keeps one demo per row and stops "
                             f"after its last, so 1..{remaining} from row {rows.first_row}")
    print(f"[DexBench collection] collecting from {rows.path} ({rows.rows} rows, sha256 {rows.sha256[:12]}"
          f"{', manifest Run ' + rows.manifest_run if rows.manifest_run else ''}) from row {rows.first_row}", flush=True)
    return argv, rows


def _take_flag(tokens: list[str], flag: str, *, has_value: bool) -> tuple[list[str], list[str]]:
    """Split ``tokens`` into (without-flag, extracted-flag-tokens)."""
    kept: list[str] = []
    taken: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == flag:
            taken.append(token)
            if has_value and index + 1 < len(tokens):
                taken.append(tokens[index + 1])
                index += 1
        elif has_value and token.startswith(flag + "="):
            taken.append(token)
        else:
            kept.append(token)
        index += 1
    return kept, taken


def with_kit_args_first(argv: list[str], leading: tuple[str, ...] = DEXBENCH_KIT_ARGS) -> list[str]:
    """``argv`` with its ``--kit_args`` (either form, any number of them) replaced by one ``--kit_args=`` token in
    front: ``leading`` (``DEXBENCH_KIT_ARGS``), then the values given, in order, so a setting the user names wins."""
    rest, given = _take_flag(argv, "--kit_args", has_value=True)
    values = [token.partition("=")[2] if token.startswith("--kit_args=") else token
              for token in given if token != "--kit_args"]
    return ["--kit_args=" + with_dexbench_kit_args(" ".join(values), leading), *rest]


def _head_runs_headless(head: list[str]) -> bool:
    """Whether the launcher flags ask for no visualizer -- the headless form."""

    chosen: str | None = None
    for index, token in enumerate(head):
        for flag in ("--visualizer", "--viz"):
            if token == flag:
                following = head[index + 1] if index + 1 < len(head) else None
                if following is not None and not following.startswith("-"):
                    chosen = following
            elif token.startswith(flag + "="):
                chosen = token.split("=", 1)[1]
    return chosen is not None and {value.strip() for value in chosen.split(",")} == {"none"}


def _install_headless_replay_keyboard_patch() -> None:
    """Let Isaac Lab's replay_demos.py run headless."""

    import builtins

    class _NoKeyboard:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def add_callback(self, *args, **kwargs) -> None:
            pass

        def reset(self) -> None:
            pass

    real_import = builtins.__import__

    def patched_import(name, globals=None, locals=None, fromlist=(), level=0):
        module = real_import(name, globals, locals, fromlist, level)
        if name == "isaaclab.devices" and fromlist and "Se3Keyboard" in fromlist:
            module.Se3Keyboard = _NoKeyboard
        return module

    builtins.__import__ = patched_import


def _install_annotation_episode_reset(head: list[str], *, replay: bool = False) -> None:
    """Guard replay joint order, and start annotation episodes the way rerecord_demos.py does."""

    input_flag = "--dataset_file" if replay else "--input_file"
    input_file = _effective_flag_value(head, [], input_flag) or "./datasets/dataset.hdf5"

    def on_env_made(annotated) -> None:
        from dexbench_arena.recording.physics_runtime import physics_runtime_metadata
        from dexbench_arena.recording.rerecord_metadata import check_replay_joint_names, read_source_metadata

        try:
            check_replay_joint_names(
                annotated, read_source_metadata(input_file), physics_runtime_metadata(annotated.cfg)["physics_backend"],
                records_joint_targets=False)
        except ValueError as error:
            raise ValueError(f"{input_file}: {error}") from error
        if replay:
            return
        annotated.sim.reset = lambda *args, **kwargs: None
        print("[annotate_demos] Newton: no hard sim.reset() between episodes.", flush=True)

    _on_env_made(on_env_made)


def _flag_values(tokens: list[str], flag: str) -> list[str | None]:
    """Every value ``flag`` is given in ``tokens``, in order."""
    values: list[str | None] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == flag:
            following = tokens[index + 1] if index + 1 < len(tokens) else None
            if following is not None and not following.startswith("-"):
                values.append(following)
                index += 1
            else:
                values.append(None)
        elif token.startswith(flag + "="):
            values.append(token.split("=", 1)[1])
        index += 1
    return values


def _effective_flag_value(head: list[str], tail: list[str], flag: str) -> str | None:
    """Resolve ``flag`` the way the callback resolves it, or ``None`` if unset."""
    values = [value for value in (*_flag_values(head, flag), *_flag_values(tail, flag)) if value is not None]
    return values[-1] if values else None


_ENVIRONMENT_NAME = re.compile(r"dexbench_\w+_environment")


def _environment_index(argv: list[str]) -> int | None:
    """Index of the DexBench environment name on ``argv`` (its first ``dexbench_*_environment`` token), or ``None``."""
    return next((i for i, token in enumerate(argv) if _ENVIRONMENT_NAME.fullmatch(token)), None)


def _flag_present(head: list[str], tail: list[str], flag: str) -> bool:
    """Whether ``flag`` appears anywhere on the command line, in either form."""
    return any(token == flag or token.startswith(flag + "=") for token in (*head, *tail))


def _compute_lab_argv(script_name: str, argv: list[str]) -> tuple[list[str], list[str], str, list[str]]:
    """Rewrite a DexBench command line for an Isaac Lab script."""

    env_index = _environment_index(argv)
    if env_index is None:
        raise SystemExit(
            f"{script_name}: no DexBench environment named on the command line. "
            "Name it as a positional argument, e.g. dexbench_retail_v1_environment."
        )
    env_name = argv[env_index]
    head, tail = argv[:env_index], argv[env_index + 1 :]

    lab_argv, _ = _take_flag(head, "--enable_cameras", has_value=False)
    lab_argv, _ = _take_flag(lab_argv, "--teleop_device", has_value=True)

    openxr = False
    if script_name in TELEOP_CAPABLE_LAB_SCRIPTS:
        device = _effective_flag_value(head, tail, "--teleop_device")
        if device is None:
            raise SystemExit(
                f"{script_name}: no --teleop_device on the command line. Name one after the "
                "environment, e.g. --teleop_device openxr. Without it the builder composes "
                "neither isaac_teleop nor teleop_devices, and the Lab script falls back to an "
                "SE(3) keyboard whose 7-dim command cannot drive a DexBench embodiment -- it "
                "fails one step into the run, inside a handler that only logs."
            )
        if "openxr" in device.lower():
            if "--xr" not in lab_argv:
                lab_argv.append("--xr")
            lab_argv, _ = _take_flag(lab_argv, "--cloudxr_env", has_value=True)
            lab_argv += ["--cloudxr_env", _effective_flag_value(head, tail, "--cloudxr_env") or "none"]
            for flag in ("--auto_launch_cloudxr", "--no-auto_launch_cloudxr"):
                if flag in (*head, *tail) and flag not in lab_argv:
                    lab_argv.append(flag)
            openxr = True
        else:
            lab_argv += ["--teleop_device", device]
        if not _flag_present(head, tail, "--enable_cameras"):
            lab_argv.append("--disable_external_cameras")

    lab_argv = with_kit_args_first(lab_argv, (*DEXBENCH_KIT_ARGS, *(OPENXR_KIT_ARGS if openxr else ())))

    if script_name == "record_demos.py":
        reset_flags = ("--reset_sim_buffer_each_episode", "--no-reset_sim_buffer_each_episode")
        chosen = [token for token in (*head, *tail) if token in reset_flags]
        lab_argv = [token for token in lab_argv if token not in reset_flags]
        tail = [token for token in tail if token not in reset_flags]
        lab_argv.append(chosen[-1] if chosen else "--no-reset_sim_buffer_each_episode")

    lab_argv += [
        "--task", env_name,
        "--external_callback",
        "dexbench_arena.isaaclab_interop.environment_registration_callback",
    ]
    return lab_argv, head, env_name, tail


def _install_recording_provenance(head: list[str], env_name: str, tail: list[str], *,
                                  episode_video_dir: str | None = None) -> None:
    """Make record_demos.py's recordings ones the teleop preprocessing pipeline replays."""

    def install() -> None:
        from dexbench_arena.recording.rerecord_metadata import (
            CAPTURE_SOURCE_KEY,
            JOINT_NAMES_KEY,
            as_built_metadata,
            replay_arguments,
            take_capture_source,
            warn_if_not_replayable,
            write_replay_sidecar,
        )

        source, error = take_capture_source(DEXBENCH_ROOT)
        warn_if_not_replayable(DEXBENCH_ROOT, source, error, recorder="record_demos")
        dataset_file = _effective_flag_value(head, [], "--dataset_file") or "./datasets/dataset.hdf5"
        replay_args = replay_arguments([*head, env_name, *tail], env_name)

        import gymnasium as gym

        original_make = gym.make

        def recording_make(*args, **kwargs):
            cfg = kwargs.get("cfg")
            if recording_make.installed or getattr(cfg, "recorders", None) is None:
                return original_make(*args, **kwargs)
            recording_make.installed = True

            import dexbench_arena.isaaclab_interop as interop
            from dexbench_arena.recording.dataset_contract import make_dataset_file_handler
            from dexbench_arena.recording.physics_runtime import physics_runtime_metadata
            from dexbench_arena.tasks.recording_hooks import recording_hooks, with_episode_attributes

            arena_env = interop.BUILDER.arena_env
            task = getattr(arena_env, "task", None)
            embodiment = arena_env.embodiment
            hooks = recording_hooks(task)
            if hooks is not None:
                hooks.validate_recording_dataset(dataset_file)
            sidecar = write_replay_sidecar(
                dataset_file,
                env=env_name,
                args=replay_args,
                task=task.get_task_description() if task is not None else None,
                source=source,
            )
            if source is None:
                print(f"[record_demos] replay sidecar -> {sidecar} (NO SOURCE: {error})")
            else:
                print(f"[record_demos] replay sidecar -> {sidecar} (commit {source['git_sha'][:7]}"
                      f"{', DIRTY' if source['dirty'] else ''})")
            physics = physics_runtime_metadata(cfg)

            def env_args():
                return {CAPTURE_SOURCE_KEY: source, "dexbench_recording": physics}

            live = {}
            handler = make_dataset_file_handler(env_args)
            if hooks is not None:
                handler = with_episode_attributes(
                    handler, lambda episode: hooks.episode_attributes(live["env"], 0))
            cfg.recorders.dataset_file_handler_class_type = handler
            if episode_video_dir is not None:
                from dexbench_arena.video.episode_video import install_episode_video

                install_episode_video(cfg, episode_video_dir)
            env = original_make(*args, **kwargs)
            live["env"] = env.unwrapped
            physics.update(as_built_metadata(env.unwrapped))
            if JOINT_NAMES_KEY not in env.unwrapped.recorder_manager.get_ep_meta():
                reason = f"the {embodiment.name!r} recorder names no joints in the recording"
                print(f"\n{'!' * 78}\n[record_demos] These recordings carry no joint contract, so the teleop "
                      f"preprocessing pipeline cannot convert them:\n  - {reason}\n{'!' * 78}\n", flush=True)
            return env

        recording_make.installed = False
        gym.make = recording_make

    _after_app_launch(install)


def _install_generation_provenance(head: list[str]) -> None:
    """Make generate_dataset.py record what it ran on and which source each attempt used."""
    attempts_log: list[Path] = []
    input_file = _effective_flag_value(head, [], "--input_file")

    def install() -> None:
        import gymnasium as gym
        import torch
        from isaaclab_mimic.datagen.data_generator import DataGenerator

        import dexbench_arena.isaaclab_interop as interop
        from dexbench_arena.recording.rerecord_metadata import as_built_metadata, file_sha256, take_capture_source

        source, error = take_capture_source(DEXBENCH_ROOT)
        if source is None:
            print(f"[generate_dataset] no code provenance for the generated data: {error}")
        source_sha256 = file_sha256(input_file) if input_file and Path(input_file).is_file() else None
        original_make = gym.make

        def generation_make(*args, **kwargs):
            cfg = kwargs.get("cfg")
            if generation_make.installed or getattr(cfg, "recorders", None) is None:
                return original_make(*args, **kwargs)
            generation_make.installed = True
            from dexbench_arena.embodiments.episode_metadata import install_episode_metadata
            from dexbench_arena.recording.physics_runtime import physics_runtime_metadata

            physics = physics_runtime_metadata(cfg)
            recorders = cfg.recorders
            attempts_log.append(Path(recorders.dataset_export_dir_path) / f"{recorders.dataset_filename}.attempts.jsonl")
            choice = getattr(interop.GENERATION, "ik_profile_choice", None) or {}
            run = {"code": source, "ik_profile": choice.get("ik_profile"), "ik_profile_reason": choice.get("reason"),
                   "ik_profile_values": choice.get("ik_profile_values"),
                   "source_file": input_file, "source_sha256": source_sha256,
                   **_generation_action_labels(cfg)}
            env = original_make(*args, **kwargs)
            physics.update(as_built_metadata(env.unwrapped))
            install_episode_metadata(env.unwrapped, "dexbench_recording", physics)
            install_episode_metadata(env.unwrapped, "dexbench_mimic_generation_run", run)
            return env

        generation_make.installed = False
        gym.make = generation_make

        attempts_started = [False]
        selected: contextvars.ContextVar[tuple[int, list] | None] = contextvars.ContextVar("dexbench_selected", default=None)
        upstream_select = DataGenerator.select_source_demo
        upstream_generate = DataGenerator.generate

        def select_source_demo(self, *args, **kwargs):
            index = upstream_select(self, *args, **kwargs)
            attempt = selected.get()
            if attempt is not None:
                env_id, picks = attempt
                eef_name = kwargs.get("eef_name", args[0] if args else None)
                picks.append((eef_name, int(index)))
                self.env.recorder_manager.add_to_episodes(
                    f"dexbench_src_demo_inds/{eef_name}",
                    torch.tensor([[int(index)]], dtype=torch.int64, device=self.env.device),
                    env_ids=[env_id],
                )
            return index

        async def generate(self, env_id, *args, **kwargs):
            picks: list[tuple[str, int]] = []
            selected.set((int(env_id), picks))
            results = await upstream_generate(self, env_id, *args, **kwargs)
            if not attempts_started[0]:
                attempts_log[0].parent.mkdir(parents=True, exist_ok=True)
            with attempts_log[0].open("a" if attempts_started[0] else "w") as log:
                attempts_started[0] = True
                log.write(json.dumps({"success": bool(results.get("success")),
                                      "selected_src_demo_inds": [index for _, index in picks],
                                      "selected_src_demo_eefs": [eef for eef, _ in picks]}) + "\n")
            return results

        select_source_demo.__wrapped__ = upstream_select
        generate.__wrapped__ = upstream_generate
        DataGenerator.select_source_demo = select_source_demo
        DataGenerator.generate = generate

    _after_app_launch(install)


def _generation_action_labels(cfg) -> dict:
    """What a generation run's actions carry: Mimic's per-step noise and which labels are recorded."""
    from dexbench_arena.embodiments.common.clean_action_labels import mimic_draws_noise

    noise = {eef: [float(subtask.action_noise) for subtask in subtasks]
             for eef, subtasks in (getattr(cfg, "subtask_configs", None) or {}).items()}
    if not mimic_draws_noise(cfg):
        labels = "noise_free"
    else:
        clean = getattr(cfg.recorders, "record_post_step_processed_actions_clean", None) is not None
        labels = "clean" if clean else "noisy"
    return {"action_noise": noise, "action_labels": labels}


def _install_annotation_source_origin(head: list[str]) -> None:
    """Carry where the demos came from into annotate_demos.py's output."""
    input_file = _effective_flag_value(head, [], "--input_file") or "./datasets/dataset.hdf5"

    def install() -> None:
        import gymnasium as gym
        import h5py
        from isaaclab.utils.datasets import HDF5DatasetFileHandler

        from dexbench_arena.recording.provenance_keys import CAPTURE_SOURCE_KEY, GENERATION_KEYS

        if not Path(input_file).is_file():
            return
        with h5py.File(input_file, "r") as handle:
            source_args = json.loads(handle["data"].attrs.get("env_args", "{}"))
        origin = {key: source_args[key]
                  for key in (CAPTURE_SOURCE_KEY, "dexbench_recording", "dexbench_refinement", "dexbench_rrc_dataset",
                              *GENERATION_KEYS)
                  if key in source_args}
        reader = HDF5DatasetFileHandler()
        reader.open(input_file)
        try:
            replay_order = list(reader.get_episode_names())
        finally:
            reader.close()
        original_make = gym.make

        def annotation_make(*args, **kwargs):
            cfg = kwargs.get("cfg")
            if annotation_make.installed or getattr(cfg, "recorders", None) is None:
                return original_make(*args, **kwargs)
            annotation_make.installed = True
            from dexbench_arena.embodiments.episode_metadata import install_episode_metadata

            if getattr(cfg, "subtask_boundaries_path", None):
                cfg.subtask_boundaries_episode_names = replay_order
            env = original_make(*args, **kwargs)
            for key, value in origin.items():
                install_episode_metadata(env.unwrapped, key, value)
            return env

        annotation_make.installed = False
        gym.make = annotation_make

    _after_app_launch(install)


def _take_episode_video_dir(script_name: str, argv: list[str]) -> tuple[list[str], str | None]:
    """``argv`` without record_demos.py's ``--episode_video_dir`` and the directory it names (None without it)."""
    if script_name != "record_demos.py":
        return list(argv), None
    episode_video_dir = _effective_flag_value(argv, [], "--episode_video_dir")
    if _flag_present(argv, [], "--episode_video_dir") and not episode_video_dir:
        raise SystemExit("record_demos.py: --episode_video_dir needs a directory")
    argv, _ = _take_flag(argv, "--episode_video_dir", has_value=True)
    return argv, episode_video_dir


def _install_collection_rows_recording(env) -> None:
    """Advance the held row on every kept demo and stamp the recording (``install_collection_recording``)."""
    if install_collection_recording(env, env.variation_replay_scheduler):
        print("[DexBench collection] a row moves on only when a demo of it is kept; discarded, reset and timed-out "
              "takes start from it again", flush=True)
    else:
        print("[DexBench collection] this session records no demos, so it holds its first row", flush=True)


def _run_isaaclab_interop_script(
    script_path: Path, script_name: str, collection: CollectionRows | None = None
) -> None:
    """Run an Isaac Lab IL script against a DexBench environment."""

    argv, episode_video_dir = _take_episode_video_dir(script_name, sys.argv[1:])
    lab_argv, head, env_name, tail = _compute_lab_argv(script_name, argv)

    import dexbench_arena.isaaclab_interop as interop

    interop.HEAD_ARGV = list(head)
    interop.ENVIRONMENT_NAME = env_name
    interop.TAIL_ARGV = list(tail)
    interop.IM_SCRIPT_NAME = script_name
    interop.COLLECTION = collection
    _install_upstream_script_patches(script_name, patch_environment_cli=False, evaluation=False)
    if collection is not None:
        _on_env_made(_install_collection_rows_recording)
    if script_name == "record_demos.py":
        _install_recording_provenance(head, env_name, tail, episode_video_dir=episode_video_dir)
    if script_name == "generate_dataset.py":
        _install_generation_provenance(head)
    if script_name == "annotate_demos.py":
        _install_annotation_source_origin(head)
    if script_name == "replay_demos.py" and _head_runs_headless(head):
        _install_headless_replay_keyboard_patch()
    if script_name in ("annotate_demos.py", "replay_demos.py"):
        _install_annotation_episode_reset(head, replay=script_name == "replay_demos.py")

    if not script_path.exists():
        raise FileNotFoundError(
            f"Missing Isaac Lab script: {script_path}. The nested IsaacLab submodule may be "
            "on the wrong revision -- run: git -C submodule/isaaclab_arena submodule update"
        )

    if script_name in LAB_SCRIPTS_LOADING_NUMPY_BEFORE_LAUNCH:
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    sys.argv = [sys.argv[0], *lab_argv]
    runpy.run_path(str(script_path), run_name="__main__")


def run_upstream_evaluation_script(script_name: str, *, patch_environment_cli: bool = True) -> None:
    """Run an upstream IsaacLab-Arena evaluation script from DexBench."""

    _run_upstream_script(
        ARENA_EVAL_SCRIPT_DIR / script_name,
        script_name,
        patch_environment_cli=patch_environment_cli,
        evaluation=True,
    )


def run_upstream_evaluation_module(
    module_name: str,
    *,
    wrapper_path: str,
    entrypoint: str = "main",
    patch_environment_cli: bool = True,
) -> None:
    """Run an upstream evaluation entrypoint by importing it, preserving ``sys.argv[0]``."""

    _install_upstream_script_patches(module_name, patch_environment_cli=patch_environment_cli, evaluation=True)

    sys.argv[0] = wrapper_path
    os.environ["DEXBENCH_TELEOP_TRACE_TRUNCATE"] = "0"
    sys.argv[1:] = with_kit_args_first(sys.argv[1:])

    module = importlib.import_module(module_name)
    module_file = Path(getattr(module, "__file__", "") or "").resolve()
    arena_root = ARENA_ROOT.resolve()
    if not module_file.is_relative_to(arena_root):
        raise FileNotFoundError(
            f"{module_name} resolved to {module_file}, outside the configured Arena root {arena_root} "
            "(ISAACLAB_ARENA_PATH); the wrapper refuses to run a different Arena tree than the one "
            "the other wrappers use."
        )
    getattr(module, entrypoint)()
