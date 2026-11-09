# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Preserve source semantics after native recorder export has closed the file."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import h5py
import numpy as np

from dexbench_arena.evaluation.artifacts import CREDENTIAL_ENVIRONMENT as _CREDENTIAL_ENVIRONMENT
from dexbench_arena.recording.git_status import git_dirty_paths
from dexbench_arena.recording.physics_runtime import robot_drive_metadata
from dexbench_arena.recording.provenance_keys import CAPTURE_SOURCE_KEY, GENERATION_KEYS

JOINT_NAMES_KEY = "dexbench_rrc_dataset"
STATE_RENDER_KEY = "dexbench_state_render"


def copy_state_render_demo(source: h5py.Group, data: h5py.Group, name: str) -> list[str]:
    """Copy trajectory data and attributes, leaving camera storage to the new video bundle."""
    target = data.create_group(name)
    target.attrs.update(source.attrs)
    dropped = []
    for key, item in source.items():
        if key in ("camera_obs", "camera_videos"):
            dropped.append(key)
        else:
            source.copy(item, target, name=key)
    return dropped


def append_state_render_record(metadata: dict, record: dict) -> None:
    """Retain the source's complete render/gaze provenance when adding another camera pass."""
    current = copy.deepcopy(record)
    current["schema_version"] = 2
    if STATE_RENDER_KEY in metadata:
        current["previous"] = copy.deepcopy(metadata[STATE_RENDER_KEY])
    metadata[STATE_RENDER_KEY] = current

CAPTURE_SOURCE_SCHEMA_VERSION = 1
_DIRTY_PATHS_KEPT = 20
_HOST_ONLY_ENVIRONMENT = re.compile(
    r"DEXBENCH_("
    r"IMAGE_.*|CONTAINER_.*|SOURCE.*|ARENA_.*|ISAACLAB_REV|HF_ASSETS_REV|.*_ROOT|ROOT|WORKDIR|DATA_.*|"
    r"REPOSITORY|REPRODUCIBLE|MEM_LIMIT|MEMORY_SWAP|FASTDDS_PROFILE|CLOUDXR_PROFILE_SHA256|BUILD_.*|BASE_.*|"
    r"ALLOW_DIRTY_CONTEXT|SKIP_ASSETS_CHECK|TELEOP_TRACE.*|RRC_JOINT_TRACE|JOINT_LOG_.*|TELEOP_TIMING|"
    r"STACK_DUMP_SIGNAL|PLOTJUGGLER_UDP|SHOT_.*|.*_DEBUG|.*_DEBUG_.*"
    r")\Z"
)
_SESSION_RENDER_ENVIRONMENT = re.compile(
    r"DEXBENCH_(RENDER_INTERVAL|RENDER_MIN_INTERVAL|ADAPTIVE_RENDER|XR_OVERSAMPLE|DLSS_MODE|CAMERA_SCALE|"
    r"VISUALIZE_EEF_OBS)\Z"
)
_NO_EFFECT_ENVIRONMENT = re.compile(r"DEXBENCH_(SOLVER_POSITION_ITER|RRC_MAX_STEP_DEG)\Z")
_LEFT_BEHIND = (_HOST_ONLY_ENVIRONMENT, _SESSION_RENDER_ENVIRONMENT, _CREDENTIAL_ENVIRONMENT, _NO_EFFECT_ENVIRONMENT)
_ENVIRONMENT_NAME = re.compile(r"DEXBENCH_[A-Z0-9_]+\Z")


def replay_environment(environ=None):
    """The DEXBENCH_* environment a recording's simulation ran under, to set again
    for its replay (see _LEFT_BEHIND for what is left out). A value with a newline
    or NUL cannot be set again faithfully, so it is refused here rather than
    recorded."""
    environ = os.environ if environ is None else environ
    kept = {}
    for name, value in sorted(environ.items()):
        if not _ENVIRONMENT_NAME.match(name) or any(pattern.match(name) for pattern in _LEFT_BEHIND):
            continue
        if any(character in value for character in "\n\r\0"):
            raise ValueError(f"{name} holds a newline or NUL; its replay could not set it again")
        kept[name] = value
    return kept


def dexbench_environment(environ=None):
    """Every DEXBENCH_* variable the process ran with, host-only and render ones included,
    for the record rather than a replay. A credential keeps its name and loses its value:
    the recording ships with the dataset."""
    environ = os.environ if environ is None else environ
    return {name: "<redacted>" if _CREDENTIAL_ENVIRONMENT.match(name) else value
            for name, value in sorted(environ.items()) if _ENVIRONMENT_NAME.match(name)}


def as_built_metadata(env, environ=None, *, hide_replay_prims: bool | None = None):
    """What the recording's simulation was built with beyond its config, for env_args next
    to ``physics_config``: the robot's joint drive settings (``robot_drive``,
    :func:`dexbench_arena.recording.physics_runtime.robot_drive_metadata`) and the DEXBENCH_* environment
    (``dexbench_environment``). The hand contact material has its own env_args key
    (``hand_contact_material``, the embodiment's episode metadata). Additive: no reader
    requires it, so older recordings without it need no migration. Reuse the
    environment recorder's typed snapshot; geometry generation has no episode
    JSONL writer. A replay can state the visibility it applied after building.
    """
    metadata = {"robot_drive": robot_drive_metadata(env), "dexbench_environment": dexbench_environment(environ)}
    recorders = getattr(getattr(env, "cfg", None), "episode_recorders", None)
    term = getattr(recorders, "environment_cfg", None)
    if term is not None:
        snapshot = copy.deepcopy(term.params["environment_cfg"])
        if hide_replay_prims is not None:
            snapshot["hide_replay_prims"] = hide_replay_prims
        metadata["environment_cfg"] = snapshot
    return metadata


def source_revision(repo_root, environ=None):
    """The DexBench commit this run's code is, for the provenance it records."""
    if environ is None:
        environ, git_env = os.environ, None
    else:
        git_env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")} | dict(environ)
    root = Path(repo_root)
    if (root / ".git").exists() or environ.get("GIT_DIR"):
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True,
                                       env=git_env).strip()
    revision = environ.get("DEXBENCH_IMAGE_REVISION", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError(f"{root} is not a git checkout and DEXBENCH_IMAGE_REVISION is {revision!r}, "
                           "not a commit: nothing says which source this run is")
    return revision


def image_paths(repo_root):
    """The pathspecs the replay image is built from (docker/arena/image-paths.sh)."""
    listed = subprocess.check_output(["bash", "docker/arena/image-paths.sh"], cwd=repo_root, text=True)
    paths = [line for line in listed.splitlines() if line]
    if not paths:
        raise RuntimeError("docker/arena/image-paths.sh listed nothing")
    return paths


def capture_source(repo_root):
    """Which code a recording is being made with, and whether that code is a commit."""
    root = Path(repo_root)
    git_sha = source_revision(root)
    if (root / ".git").exists() or os.environ.get("GIT_DIR"):
        dirty = git_dirty_paths(root, image_paths(root))
        outside = [path for path in git_dirty_paths(root) if path not in set(dirty)]
        branch = subprocess.check_output(["git", "-C", str(root), "rev-parse", "--abbrev-ref", "HEAD"],
                                         text=True).strip()
    else:
        dirty, outside, branch = [], [], None
    return {"schema_version": CAPTURE_SOURCE_SCHEMA_VERSION, "git_sha": git_sha, "dirty": bool(dirty),
            "dirty_paths": dirty[:_DIRTY_PATHS_KEPT], "dirty_paths_outside_image": outside[:_DIRTY_PATHS_KEPT],
            "branch": branch, "environment": replay_environment()}


def take_capture_source(repo_root):
    """capture_source as ``(source, error)``, for a recorder to take once at
    start-up. When git cannot say (a worktree's .git mounted into a container
    points at a host path), or a DEXBENCH_* value could not be set again for the
    replay (a newline in it), the session still records, with no source, and
    the pipeline refuses it."""
    try:
        return capture_source(repo_root), None
    except (OSError, subprocess.CalledProcessError, RuntimeError, ValueError) as error:
        return None, error


def commit_on(repo_root, commit, ref="origin/main"):
    """Whether ``commit`` is in ``ref``'s history as this checkout last fetched
    it, or None when git cannot say (no such ref, not a checkout)."""
    result = subprocess.run(["git", "-C", str(repo_root), "merge-base", "--is-ancestor", commit, ref],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return {0: True, 1: False}.get(result.returncode)


def replay_warnings(source, error, on_main):
    """What to tell the operator before the first take: ``(refused, unchecked)``."""
    refused, unchecked = [], []
    if source is None:
        refused.append(f"which code this is cannot be determined ({error})")
        return refused, unchecked
    if source["dirty"]:
        refused.append(f"the checkout has uncommitted changes to what the replay image is built from "
                       f"({len(source['dirty_paths'])}+ paths, e.g. {source['dirty_paths'][0]})")
    if source["branch"] is None:
        unchecked.append(f"whether commit {source['git_sha'][:7]} is on main: this image workspace has no "
                         "checkout to ask; recordings from a commit main did not publish are refused")
    else:
        if on_main is False:
            refused.append(f"commit {source['git_sha'][:7]} (on {source['branch']}) is not on origin/main "
                           "as this checkout last fetched it")
        elif on_main is None:
            unchecked.append(f"whether commit {source['git_sha'][:7]} (on {source['branch']}) is on main: this "
                             "checkout has no origin/main to check against; recordings from a commit main "
                             "did not publish are refused")
    return refused, unchecked


def warn_if_not_replayable(repo_root, source, error, *, recorder):
    """Say so before the first take, not after the session, when the pipeline
    will refuse these recordings: it replays each on the image built from its
    commit, so the code has to be a commit main published. Recording goes on --
    a session for local use is still a session. ``recorder`` prefixes the
    lines."""
    on_main = commit_on(repo_root, source["git_sha"]) if source is not None and source["branch"] else None
    refused, unchecked = replay_warnings(source, error, on_main)
    banner = "!" * 78
    if refused:
        print(f"\n{banner}\n[{recorder}] The teleop preprocessing pipeline will REFUSE these recordings:\n"
              + "\n".join(f"  - {reason}" for reason in refused)
              + "\n  Record from a clean checkout of a commit on main to have them replayed.\n"
              f"{banner}\n", flush=True)
    if unchecked:
        print(f"\n[{recorder}] Could not check before recording:\n"
              + "\n".join(f"  - {reason}" for reason in unchecked) + "\n", flush=True)


_OPERATOR_FLAGS_WITH_VALUE = ("--teleop_device", "--cloudxr_env", "--l1_stream_plan", "--l1-stream-plan",
                              "--l1_stream_plan_start", "--l1-stream-plan-start")
_OPERATOR_SWITCHES = (
    "--teleop_session",
    "--teleop-session",
    "--auto_launch_cloudxr",
    "--no-auto_launch_cloudxr",
    "--log_placement_checks",
    "--log-placement-checks",
    "--teleop_target_marker",
    "--teleop-target-marker",
)


def without_operator_switches(argv) -> tuple[list[str], list[str]]:
    """``argv`` less the operator's switches (_OPERATOR_SWITCHES, both spellings), and the switches it held."""
    tokens = [str(token) for token in argv]
    return ([token for token in tokens if token not in _OPERATOR_SWITCHES],
            [token for token in tokens if token in _OPERATOR_SWITCHES])


def replay_arguments(argv, env, embodiment=None):
    """The backend preset and environment arguments from a recorder's argv."""
    tokens = list(argv)
    if env not in tokens:
        raise ValueError(f"environment {env!r} is not on the command line {tokens!r}")
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--presets")
    launcher, _ = parser.parse_known_args(tokens[:tokens.index(env)])
    presets = ["--presets", launcher.presets] if launcher.presets is not None else []
    dropped_with_value = _OPERATOR_FLAGS_WITH_VALUE + (("--embodiment",) if embodiment is not None else ())
    kept, skip_value = [], False
    for token in tokens[tokens.index(env) + 1:]:
        if skip_value:
            skip_value = False
        elif token in dropped_with_value:
            skip_value = True
        elif not (token in _OPERATOR_SWITCHES or token.startswith(tuple(f"{flag}=" for flag in dropped_with_value))):
            kept.append(token)
    return [*presets, "--embodiment", embodiment, *kept] if embodiment is not None else [*presets, *kept]


def write_replay_sidecar(dataset_file, *, env, args, task, source):
    """``<stem>.job.json`` beside the recording: how the pipeline replays it."""
    path = Path(dataset_file).with_suffix(".job.json")
    sidecar = {"env": env, "args": list(args), "source": source}
    if task:
        sidecar["task"] = task
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(sidecar, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    return path


def episode_records_path(dataset_file):
    """``<stem>.episodes.jsonl`` beside the recording: Arena's per-episode records of it."""
    return Path(dataset_file).with_suffix(".episodes.jsonl")


def write_episode_records(env, path, *, job_name):
    """Have Arena's episode recorder write its JSONL to ``path`` from now on."""
    env.episode_recorder.set_job_name(job_name)
    env.episode_recorder.set_output_path(path)
    return Path(path)


def finish_episode_records(env):
    """Write the row of the episode still running, which no reset will end."""
    env.episode_recorder.record_pre_reset(None)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def episode_sort_key(name):
    """A recording's demos in the order the replay, render and conversion scripts take them: ``demo_N`` by N
    (``demo_2`` before ``demo_10``), then any other name by name."""
    match = re.fullmatch(r"demo_([0-9]+)", name)
    return (0, int(match.group(1))) if match else (1, name)


RECORDED_DRAW_PREFIX = "variations/"
"""``demo_attributes``' keys for the demo's recorded variation draws, ``variations/<asset>/<variation>``: a path no
HDF5 attribute name takes."""


def demo_attributes(path, name):
    """What demo ``name`` of the HDF5 at ``path`` records about its episode, for ``prepare_replay_episode``: its
    attributes, Isaac Lab's (``num_samples``, ``success``, ...) and a task's per-demo record (``episode_attributes``,
    e.g. DB-L1's ``dexbench_l1_stream``), and under ``variations/<asset>/<variation>`` the first row of each draw Arena's
    recorder wrote to its ``variations`` group (a list, one value per column; DB-M5's empty compartment). That row is
    the episode's start: a replay's ``reset_to`` draws afresh what the restored state does not carry, so a task reads
    the recording's draw here and sets it before that reset."""
    with h5py.File(path, "r") as source:
        demo = source["data"][name]
        record = dict(demo.attrs)
        variations = demo.get("variations")
        if variations is not None:
            def add(path, item):
                if isinstance(item, h5py.Dataset) and item.shape and item.shape[0] > 0:
                    record[f"{RECORDED_DRAW_PREFIX}{path}"] = np.asarray(item[0]).reshape(-1).tolist()
            variations.visititems(add)
        return record


DEMO_ATTRIBUTE_PREFIX = "dexbench_"
"""A source demo's attributes a re-recording carries over (preserve_rerecord_metadata): DexBench's own, what a task
recorded per demo; Isaac Lab's (``num_samples``, ``success``) are the re-recording's."""


def read_source_metadata(path):
    with h5py.File(path, "r") as source:
        metadata = json.loads(source["data"].attrs.get("env_args", "{}"))
    contract = metadata.get("dexbench_rrc_dataset")
    if contract is not None:
        for key in ("state_joint_names", "action_joint_names"):
            names = contract.get(key)
            if (not isinstance(names, list) or not names
                    or not all(isinstance(n, str) and n for n in names)
                    or len(set(names)) != len(names)):
                raise ValueError(f"Invalid source contract: {key}")
    return metadata


def recorded_physics_backend(source_metadata):
    """The physics backend a recording names, latest execution first; None if it names none."""
    rerecord = source_metadata.get("dexbench_rerecord") or {}
    for section in (rerecord.get("runtime"), source_metadata.get("dexbench_recording"),
                    source_metadata.get("hand_contact_material")):
        if isinstance(section, dict) and section.get("physics_backend"):
            return section["physics_backend"]
    return None


def check_replay_joint_names(env, source_metadata, physics_backend, *, records_joint_targets=True):
    """Refuse, before replay, a re-record whose output would misname its joints."""
    live = env.recorder_manager.get_ep_meta().get(JOINT_NAMES_KEY) if records_joint_targets else None
    contract = source_metadata.get(JOINT_NAMES_KEY)
    if contract is None:
        source_backend = recorded_physics_backend(source_metadata)
        if source_backend is None and not records_joint_targets:
            return None
        if (live is None and records_joint_targets) or source_backend == physics_backend:
            return None
        if source_backend is None:
            return "the source names no joints and records no physics backend"
        raise ValueError(
            f"Source names no joints and was recorded on {source_backend}; the replay runs "
            f"{physics_backend}. Its joint state restores by index in the articulation order of "
            "the backend it was recorded on: re-record it on that backend"
        )
    robot = env.scene["robot"]
    if list(robot.joint_names) != contract["state_joint_names"]:
        raise ValueError("Source state joint order does not match rerecord robot")
    terms = [env.action_manager.get_term(name) for name in env.action_manager.active_terms]
    if len(terms) == 1 and hasattr(terms[0], "_joint_ids"):
        ids = terms[0]._joint_ids
        ids = list(range(len(robot.joint_names)))[ids] if isinstance(ids, slice) else ids
        if [robot.joint_names[int(i)] for i in ids] != contract["action_joint_names"]:
            raise ValueError("Source action joint order does not match rerecord action term")
    elif records_joint_targets and (live is None or live["action_joint_names"] != contract["action_joint_names"]):
        raise ValueError(
            "Source action joint names match neither a direct joint action term "
            "nor the joint targets the rerecord env records"
        )
    return None


def named_target_columns(source_names, action_names, actions=None, *, constant_tol=1e-6):
    """Columns of a recording's joint targets, named by ``source_names``, in ``action_names`` order."""
    source_names, action_names = list(source_names), list(action_names)
    if len(set(source_names)) != len(source_names) or len(set(action_names)) != len(action_names):
        raise ValueError("Joint names must be unique")
    missing = [name for name in action_names if name not in source_names]
    if missing:
        raise ValueError(f"The source names no target for action joints {missing}")
    columns = [source_names.index(name) for name in action_names]
    dropped = [name for name in source_names if name not in set(action_names)]
    if actions is not None:
        if actions.shape[-1] != len(source_names):
            raise ValueError(f"{actions.shape[-1]} action columns against {len(source_names)} source joint names")
        for name in dropped:
            column = actions[:, source_names.index(name)]
            if float(column.max() - column.min()) > constant_tol:
                raise ValueError(f"Source joint {name} moves, but the action term does not drive it")
    return columns, dropped


def recording_problems(path, required, hw=None, format_version=1):
    """Why the recording at ``path`` may not be kept, as a list (empty: keep it)."""
    problems = []
    with h5py.File(path, "r") as handle:
        version = handle.attrs.get("format_version")
        if version is None or int(version) != format_version:
            problems.append(f"root format_version is {version!r}, not {format_version}")
        demos = list(handle["data"])
        if not demos:
            problems.append("no demo under data/")
        for name in demos:
            demo = handle["data"][name]
            problems += [f"{name}: no {dataset}" for dataset in required if not isinstance(demo.get(dataset), h5py.Dataset)]
            if hw is None:
                continue
            sizes = {f"camera_videos/{camera}": (int(group.attrs["height"]), int(group.attrs["width"]))
                     for camera, group in demo.get("camera_videos", {}).items()}
            sizes |= {f"camera_obs/{camera}": tuple(int(v) for v in frames.shape[1:3])
                      for camera, frames in demo.get("camera_obs", {}).items()}
            if not sizes:
                problems.append(f"{name}: no camera frames")
            problems += [f"{name}: {key} is {size[1]}x{size[0]}, not {hw[1]}x{hw[0]}"
                         for key, size in sizes.items() if size != hw]
    return problems


CLEAN_LABEL_DATASETS = ("actions_clean", "processed_actions_clean")


def check_clean_labels(source_demo, output_demo, name):
    """Refuse a re-recorded demo the source's clean labels would not line up with, step for step."""
    for dataset in CLEAN_LABEL_DATASETS:
        if dataset not in source_demo:
            continue
        if dataset in output_demo:
            raise ValueError(f"Refusing to overwrite exported {dataset}: {name}")
        steps = output_demo["actions"].shape[0]
        if source_demo[dataset].shape[0] != steps:
            raise ValueError(f"{name}: source {dataset} has {source_demo[dataset].shape[0]} steps, "
                             f"the re-recording {steps}")


def copy_clean_labels(source_demo, output_demo):
    """Copy a source demo's clean labels into its re-recorded demo (check_clean_labels first)."""
    for dataset in CLEAN_LABEL_DATASETS:
        if dataset in source_demo:
            source_demo.copy(source_demo[dataset], output_demo, name=dataset)


def preserve_rerecord_metadata(source_path, output_path, *, source_metadata,
                               calibration, episodes, git_sha, runtime=None,
                               expected_source_sha256=None):
    """Copy variations, each demo's own DexBench attributes (DEMO_ATTRIBUTE_PREFIX) and existing source,
    refinement and generation metadata; never copy replay states."""
    if Path(source_path).resolve() == Path(output_path).resolve():
        raise ValueError("Source and output must differ")
    digest = file_sha256(source_path)
    if expected_source_sha256 is not None and digest != expected_source_sha256:
        raise ValueError("Source SHA-256 differs from the original recording input")
    with h5py.File(source_path, "r") as source, h5py.File(output_path, "r+") as output:
        data = output["data"]
        for name in episodes:
            if name not in source["data"] or name not in data:
                raise ValueError(f"Missing source/exported episode: {name}")
            if "variations" in data[name]:
                raise ValueError(f"Refusing to overwrite exported variations: {name}")
            clash = [key for key, value in source["data"][name].attrs.items()
                     if key.startswith(DEMO_ATTRIBUTE_PREFIX) and key in data[name].attrs
                     and not np.array_equal(value, data[name].attrs[key])]
            if clash:
                raise ValueError(f"Refusing to overwrite exported demo attributes {clash}: {name}")
            check_clean_labels(source["data"][name], data[name], name)
        metadata = json.loads(data.attrs.get("env_args", "{}"))
        for key in (
            "dexbench_rrc_dataset",
            "dexbench_recording",
            "dexbench_action_layout",
            "dexbench_success_evaluation",
            "dexbench_refinement",
            CAPTURE_SOURCE_KEY,
            *GENERATION_KEYS,
        ):
            if key in source_metadata:
                metadata[key] = copy.deepcopy(source_metadata[key])
        if (runtime or {}).get("unnamed_joints") and JOINT_NAMES_KEY not in source_metadata:
            metadata.pop(JOINT_NAMES_KEY, None)
        contract = metadata.get("dexbench_rrc_dataset")
        if contract is not None:
            contract.pop("camera_calibration", None)
            if calibration:
                contract["camera_calibration"] = copy.deepcopy(calibration)
        metadata["dexbench_rerecord"] = {
            "schema_version": 1, "source_sha256": digest,
            "git_sha": git_sha, "runtime": runtime, "camera_calibration": calibration,
            "episodes": episodes,
            "state_error_metric": "maximum absolute component error in native HDF5 state units",
        }
        for name in episodes:
            if "variations" in source["data"][name]:
                source.copy(source["data"][name]["variations"], data[name], name="variations")
            copy_clean_labels(source["data"][name], data[name])
            data[name].attrs.update({key: value for key, value in source["data"][name].attrs.items()
                                     if key.startswith(DEMO_ATTRIBUTE_PREFIX)})
        data.attrs["env_args"] = json.dumps(metadata, allow_nan=False)


def checkpoint_export(bundle, *, source_path, output_path, source_metadata, calibration,
                      episode_reports, git_sha, source_sha256, camera_storage, runtime=None):
    """Atomic metadata journal after the native recorder has flushed an episode."""
    payload = dict(source_path=str(source_path), source_sha256=source_sha256,
                   camera_storage=camera_storage, output_path=str(output_path),
                   work_path=str(bundle.hdf5_path), source_metadata=source_metadata,
                   calibration=calibration, episode_reports=episode_reports, git_sha=git_sha, runtime=runtime,
                   references=bundle.references, video_relative=str(bundle.video_relative))
    path = bundle.stage / 'recovery.json'
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(payload, allow_nan=False))
    temporary.replace(path)
