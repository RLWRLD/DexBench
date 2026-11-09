# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Optional per-step HDF5 recording for policy evaluations."""

from __future__ import annotations

import argparse
import contextvars
import dataclasses
import functools
import hashlib
import inspect
import os
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import yaml

from dexbench_arena.evaluation.artifacts import redact_api_tokens

CAMERA_RECORDER_TERM = "record_pre_step_flat_camera_observations"
"""The ``ArenaEnvRecorderManagerCfg`` term that stores raw camera frames."""


def add_dataset_recording_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the dataset-recording flags to an evaluation parser."""

    parser.add_argument(
        "--record_dataset",
        action="store_true",
        default=False,
        help=(
            "Export every episode's per-step actions, policy observations and scene states to an"
            " HDF5 dataset in the run directory (data/demo_N/{actions,obs,states,...}). Camera"
            " frames are left out unless --record_dataset_images is given; about 5 KiB per step."
        ),
    )
    parser.add_argument(
        "--record_dataset_images",
        action="store_true",
        default=False,
        help=(
            "Also store the raw camera frames (camera_obs) in the dataset. Implies --record_dataset"
            " and needs --enable_cameras. About 220 KiB per step for two 180x320 cameras, roughly"
            " 40x the file; --record_camera_video keeps the same frames as mp4 far more cheaply."
            " Frames stay on the simulation device until each episode's reset: at native 720x1280"
            " a 45 s episode needs 3.5 GiB of GPU memory per camera with --device cuda."
        ),
    )
    return parser


_TYPED_RUNNERS = ("isaaclab_arena.evaluation.experiment_runner",)


def install_dataset_recording_patches(
    script_name: str | None, argv: Sequence[str], *, after_launch: Callable[[Callable[[], None]], None]
) -> None:
    """Install the patches a recording run of ``script_name`` needs, and nothing otherwise."""

    name = os.path.basename(script_name) if script_name and script_name.endswith(".py") else script_name
    if name == "policy_runner.py":
        tokens = set(argv)
        if not any(token.startswith("--record_dataset") for token in tokens):
            return
        if "--record_dataset_images" in tokens and not tokens & {"--enable_cameras", "--record_camera_video"}:
            raise SystemExit(
                "--record_dataset_images needs --enable_cameras (or --record_camera_video, which turns the"
                " cameras on): the camera_obs frames only exist when the cameras are enabled."
            )
        install_shared_run_dir()
        after_launch(install_recorder_rows_once_per_call)
    elif name in _TYPED_RUNNERS:
        after_launch(install_run_output_dir_capture)
        after_launch(install_experiment_yaml_snapshot)
        after_launch(install_recorder_rows_once_per_call)


def dataset_recording_request(environment_cfg: Any) -> tuple[bool, bool]:
    """``(record, include_images)`` an environment's config asks for."""

    include_images = bool(getattr(environment_cfg, "record_dataset_images", False))
    record = bool(getattr(environment_cfg, "record_dataset", False)) or include_images
    return record, include_images



def _tensor_leaves(key: str, value: Any) -> Iterator[tuple[str, Any]]:
    """``(key, tensor)`` for every leaf of a nested recorder value, keys joined by "/" as Isaac Lab's
    ``RecorderManager.add_to_episodes`` joins them, warp arrays as torch tensors."""
    import warp as wp

    if isinstance(value, dict):
        for sub_key, sub_value in value.items():
            yield from _tensor_leaves(f"{key}/{sub_key}", sub_value)
    elif isinstance(value, wp.array):
        yield key, wp.to_torch(value)
    else:
        yield key, value


_PINNED_ADD_TO_EPISODES_SHA256 = "0f32a57aa89a978fc5aa4e62c313633ca720823209a66ec1686eac2c8619ab16"
"""SHA-256 of the source of ``ArenaRecorderManager.add_to_episodes`` at the Arena pin (59d95cc9), the method
:func:`install_recorder_rows_once_per_call` re-implements."""


def _env_ids_to_tensor(env_ids: Any, num_envs: int, device: Any):
    """Env ids as a long tensor on ``device``: None for every env, a slice into them, or a sequence or tensor.
    What upstream Arena's ``env_ids_to_tensor`` does (main e87874de); the pin's private ``_normalize_env_ids``
    takes no slice and is gone upstream, so the workaround does not call either."""
    import torch

    if env_ids is None:
        env_ids = slice(None)
    if isinstance(env_ids, slice):
        return torch.arange(num_envs, dtype=torch.long, device=device)[env_ids]
    return torch.as_tensor(env_ids, dtype=torch.long, device=device)


def install_recorder_rows_once_per_call() -> None:
    """Make ``ArenaRecorderManager.add_to_episodes`` pick the active episodes' rows once per call."""
    from isaaclab.managers.recorder_manager import RecorderManager
    from isaaclab_arena.recording.arena_recorder_manager import ArenaRecorderManager

    if getattr(ArenaRecorderManager.add_to_episodes, "_dexbench_rows_once", False):
        return
    source_hash = hashlib.sha256(inspect.getsource(ArenaRecorderManager.add_to_episodes).encode()).hexdigest()
    if source_hash != _PINNED_ADD_TO_EPISODES_SHA256:
        raise RuntimeError(
            "ArenaRecorderManager.add_to_episodes is not the method the recorder workaround re-implements "
            f"(source sha256 {source_hash[:12]}, expected {_PINNED_ADD_TO_EPISODES_SHA256[:12]}): the Arena pin "
            "moved. If Arena now filters a nested value once per call, delete install_recorder_rows_once_per_call; "
            "otherwise port it to the new method and update _PINNED_ADD_TO_EPISODES_SHA256 "
            "(dexbench_arena/evaluation/dataset_recording.py)."
        )

    def add_to_episodes(self, key, value, env_ids=None) -> None:
        if not self.active_terms or key is None:
            return
        requested_env_ids = _env_ids_to_tensor(env_ids, self._env.num_envs, self._env.device)
        active_rows = self._env.active_episode_mask[requested_env_ids].nonzero(as_tuple=False).flatten()
        if active_rows.numel() == 0:
            return
        active_env_ids = requested_env_ids[active_rows].tolist()
        for leaf_key, leaf in _tensor_leaves(key, value):
            RecorderManager.add_to_episodes(self, leaf_key, leaf[active_rows.to(device=leaf.device)], active_env_ids)

    add_to_episodes._dexbench_rows_once = True  # type: ignore[attr-defined]
    add_to_episodes.__doc__ = ArenaRecorderManager.add_to_episodes.__doc__
    ArenaRecorderManager.add_to_episodes = add_to_episodes


_RUN_DIRS: dict[str, str] = {}


def run_output_dir(base_dir: str) -> str:
    """The run directory under ``base_dir``, chosen once per process."""

    if base_dir not in _RUN_DIRS:
        from isaaclab_arena.video import video_recording

        stamp = getattr(video_recording.timestamped_run_dir, "_dexbench_original", video_recording.timestamped_run_dir)
        _RUN_DIRS[base_dir] = stamp(base_dir)
    return _RUN_DIRS[base_dir]


def install_shared_run_dir() -> None:
    """Make upstream's ``timestamped_run_dir`` return :func:`run_output_dir`."""

    from isaaclab_arena.video import video_recording

    if getattr(video_recording.timestamped_run_dir, "_dexbench_shared", False):
        return
    original = video_recording.timestamped_run_dir

    def shared_run_dir(base_dir: str) -> str:
        return run_output_dir(base_dir)

    shared_run_dir._dexbench_original = original  # type: ignore[attr-defined]
    shared_run_dir._dexbench_shared = True  # type: ignore[attr-defined]
    shared_run_dir.__doc__ = original.__doc__
    video_recording.timestamped_run_dir = shared_run_dir


@dataclasses.dataclass
class _TypedRun:
    """The typed run ``build_and_run`` is executing: its directory."""

    output_dir: str


_CURRENT_RUN: contextvars.ContextVar[_TypedRun | None] = contextvars.ContextVar("dexbench_typed_run", default=None)


def current_run_output_dir() -> str | None:
    """The typed run's output directory while upstream's ``build_and_run`` executes, else None."""

    run = _CURRENT_RUN.get()
    return None if run is None else run.output_dir


def install_run_output_dir_capture() -> None:
    """Make the run directory of a typed run visible while its environment is built."""

    import importlib

    run_execution = importlib.import_module("isaaclab_arena.evaluation.run_execution")

    if getattr(run_execution.build_and_run, "_dexbench_captures_output_dir", False):
        return
    original = run_execution.build_and_run

    @functools.wraps(original)
    def build_and_run(cfg, output_dir, *args, **kwargs):
        token = _CURRENT_RUN.set(_TypedRun(str(output_dir)))
        try:
            return original(cfg, output_dir, *args, **kwargs)
        finally:
            _CURRENT_RUN.reset(token)

    build_and_run._dexbench_captures_output_dir = True  # type: ignore[attr-defined]
    build_and_run._dexbench_original = original  # type: ignore[attr-defined]
    run_execution.build_and_run = build_and_run


EXPERIMENT_YAML_FILENAME = "experiment.yaml"
"""The effective experiment, written into the experiment's output directory before its Runs start."""


def experiment_yaml(experiment_cfg: Any) -> str:
    """``experiment_cfg`` as Arena serializes it, with each policy server's token written as the
    environment variable it comes from (``artifacts.redact_api_tokens``): Arena's serializer
    resolves the interpolation, and this file is kept with the run's outputs."""
    from isaaclab_arena.hydra.typed_experiment_serializer import serialize_arena_experiment_to_yaml

    experiment = redact_api_tokens(yaml.safe_load(serialize_arena_experiment_to_yaml(experiment_cfg)))
    return yaml.safe_dump(experiment, sort_keys=False)


def install_experiment_yaml_snapshot() -> None:
    """Write the experiment the runner is about to execute into its output directory."""

    import importlib

    run_execution = importlib.import_module("isaaclab_arena.evaluation.run_execution")
    if getattr(run_execution.execute_experiment, "_dexbench_writes_experiment_yaml", False):
        return
    original = run_execution.execute_experiment

    @functools.wraps(original)
    def execute_experiment(experiment_cfg, output_dir, *args, **kwargs):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, EXPERIMENT_YAML_FILENAME), "w", encoding="utf-8") as stream:
            stream.write(experiment_yaml(experiment_cfg))
        return original(experiment_cfg, output_dir, *args, **kwargs)

    execute_experiment._dexbench_writes_experiment_yaml = True  # type: ignore[attr-defined]
    execute_experiment._dexbench_original = original  # type: ignore[attr-defined]
    run_execution.execute_experiment = execute_experiment


def make_dataset_recorder_terms_cfg(*, include_images: bool) -> Any:
    """Arena's action/state recorder terms, with the camera term only on request."""

    from isaaclab_arena.utils.configclass import transform_configclass_instance
    from isaaclab_arena.terms.recorders import ArenaEnvRecorderManagerCfg

    terms_cfg = ArenaEnvRecorderManagerCfg()
    if include_images:
        return terms_cfg
    return transform_configclass_instance(
        terms_cfg, lambda fields: [field for field in fields if field[0] != CAMERA_RECORDER_TERM]
    )


def apply_dataset_recording(
    env_cfg: Any,
    environment_cfg: Any,
    *,
    enable_cameras: bool,
    export_dir: str | None = None,
    output_base_dir: str | None = None,
) -> None:
    """Extend ``env_cfg.recorders`` with per-step terms when the CLI asked for them."""

    record, include_images = dataset_recording_request(environment_cfg)
    if not record:
        return
    if include_images and not enable_cameras:
        raise ValueError(
            "--record_dataset_images needs --enable_cameras: the camera_obs group the frames come"
            " from only exists when the cameras are enabled."
        )

    from isaaclab.managers import DatasetExportMode
    from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg

    from isaaclab_arena.utils.configclass import combine_configclass_instances, transform_configclass_instance

    base_cfg = env_cfg.recorders
    existing = {field.name for field in dataclasses.fields(base_cfg)} if base_cfg is not None else set()
    additions = transform_configclass_instance(
        make_dataset_recorder_terms_cfg(include_images=include_images),
        lambda fields: [field for field in fields if field[0] not in existing],
    )
    recorder_cfg = combine_configclass_instances(
        "RecorderManagerCfg",
        base_cfg,
        additions,
        bases=(RecorderManagerBaseCfg,),
    )
    if export_dir is None:
        export_dir = current_run_output_dir()
    if export_dir is None:
        if output_base_dir is not None:
            export_dir = run_output_dir(output_base_dir)
        else:
            export_dir = base_cfg.dataset_export_dir_path
    recorder_cfg.dataset_export_dir_path = export_dir
    recorder_cfg.dataset_filename = getattr(base_cfg, "dataset_filename", "dataset")
    recorder_cfg.dataset_export_mode = DatasetExportMode.EXPORT_ALL
    recorder_cfg.export_in_record_pre_reset = True
    recorder_cfg.export_in_close = False
    recorder_cfg.dataset_compression = True
    env_cfg.recorders = recorder_cfg
    print(
        "[dexbench] recording per-step dataset"
        f" ({'with' if include_images else 'without'} camera frames) ->"
        f" {recorder_cfg.dataset_export_dir_path}/{recorder_cfg.dataset_filename}.hdf5"
    )
