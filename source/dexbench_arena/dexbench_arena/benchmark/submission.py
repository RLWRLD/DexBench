"""Offline submission evidence checks for reported policy evaluations."""

from __future__ import annotations

import json
import math
from pathlib import Path

import h5py
import numpy as np
from jsonschema import Draft202012Validator

from dexbench_arena.benchmark import Refusal
from dexbench_arena.benchmark.leaderboard_models import submission_model
from dexbench_arena.benchmark.reference_settings import evaluation_contract_sha256
from dexbench_arena.benchmark.run_manifest import COMPOSED, read_run_manifest
from dexbench_arena.embodiments.rby1_wuji2.camera_contract import EGO_CAMERAS
from dexbench_arena.evaluation.artifacts import file_sha256, recorded

SCHEMA = Path(__file__).with_name("submission.schema.json")
REQUIRED_VIDEO_CAMERA = EGO_CAMERAS["camera_ego_left"]
"""The designated head view required for every submitted success and failure."""


def load_submission(path: Path) -> dict:
    """Read a versioned submission and validate its public metadata contract."""
    submission = json.loads(path.read_text())
    validator = Draft202012Validator(json.loads(SCHEMA.read_text()))
    errors = sorted(validator.iter_errors(submission), key=lambda error: str(error.json_path))
    if errors:
        raise Refusal("; ".join(f"{error.json_path}: {error.message}" for error in errors[:10]))
    model_entry = json.loads(checked_file(path.resolve().parent, submission["model_entry"]).read_text())
    model_schema = json.loads(SCHEMA.with_name("model_entry.schema.json").read_text())
    model_errors = list(Draft202012Validator(model_schema).iter_errors(model_entry))
    if model_errors:
        raise Refusal(f"model_entry: {model_errors[0].message}")
    if submission["training"]["additional_data"]:
        raise Refusal("standard-data cannot declare additional DexBench training data or interaction")
    return submission


def bundle_path(root: Path, relative: str) -> Path:
    """Resolve a bundle-local artifact, refusing absolute paths and escaping symlinks."""
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise Refusal(f"artifact escapes bundle: {relative}")
    if not path.exists():
        raise Refusal(f"missing artifact: {relative}")
    return path


def checked_file(root: Path, artifact: dict) -> Path:
    path = bundle_path(root, artifact["path"])
    if not path.is_file() or file_sha256(path) != artifact["sha256"]:
        raise Refusal(f"artifact SHA-256 mismatch: {artifact['path']}")
    return path


def _datasets(node, ancestors=()):
    """Walk only self-contained HDF5 datasets; do not follow links or external storage."""
    if isinstance(node, h5py.Dataset):
        if node.is_virtual or node.external:
            raise Refusal(f"{node.name}: external/virtual dataset is not self-contained")
        yield node
        return
    address = h5py.h5o.get_info(node.id).addr
    if address in ancestors:
        raise Refusal(f"{node.name}: cyclic HDF5 group")
    for key in node:
        if not isinstance(node.get(key, getlink=True), h5py.HardLink):
            raise Refusal(f"{node.name}/{key}: HDF5 links are not allowed")
        yield from _datasets(node[key], (*ancestors, address))


def _series(node, length: int) -> None:
    leaves = list(_datasets(node))
    if not leaves:
        raise Refusal(f"{node.name}: empty trajectory field")
    for dataset in leaves:
        if (not dataset.shape or dataset.shape[0] != length or dataset.size == 0
                or dataset.dtype.kind not in "biuf"):
            raise Refusal(f"{dataset.name}: expected {length} finite numeric samples, got {dataset.shape}")
        for start in range(0, length, 256):
            if not np.isfinite(dataset[start:start + 256]).all():
                raise Refusal(f"{dataset.name}: non-finite trajectory value")


def _identity(demo, name: str) -> int:
    path = f"episode_id/{name}"
    if path not in demo or not isinstance(demo[path], h5py.Dataset):
        raise Refusal(f"{demo.name}: missing {path}; record with --submission")
    value = demo[path]
    if value.size != 1 or value.dtype.kind not in "iu" or int(value[()].item()) < 0:
        raise Refusal(f"{demo.name}: invalid {path}")
    return int(value[()].item())


def validate_trajectories(root: Path, files: list[dict], runs: dict) -> list[dict]:
    """Join every demo to (Run, rebuild, env_id, episode_in_env), including failures."""
    expected = {}
    for run_name, run in runs.items():
        for rebuild in run["rebuilds"]:
            for episode in rebuild["episodes"]:
                key = (run_name, rebuild["index"], episode["env_id"], episode["episode_in_env"])
                if key in expected:
                    raise Refusal(f"duplicate result episode identity: {key}")
                expected[key] = episode
    seen, paths, joined = set(), set(), []
    for artifact in files:
        path = checked_file(root, artifact)
        if path in paths:
            raise Refusal(f"duplicate trajectory file: {path}")
        paths.add(path)
        with h5py.File(path, "r") as handle:
            list(_datasets(handle))
            if "data" not in handle or not isinstance(handle["data"], h5py.Group) or not len(handle["data"]):
                raise Refusal(f"{path}: no recorded demos")
            if not isinstance(json.loads(handle["data"].attrs.get("env_args", "null")), dict):
                raise Refusal(f"{path}: missing original env_args metadata")
            for name, demo in handle["data"].items():
                if not isinstance(demo, h5py.Group):
                    raise Refusal(f"{path}: {name} is not a demo group")
                key = (artifact["run"], artifact["rebuild"], _identity(demo, "env_id"),
                       _identity(demo, "episode_in_env"))
                if key not in expected or key in seen:
                    raise Refusal(f"unknown or duplicate trajectory episode: {key}")
                episode = expected[key]
                steps = episode["episode_length"]
                if type(steps) is not int or steps <= 0 or demo.attrs.get("num_samples") != steps:
                    raise Refusal(f"{key}: trajectory length differs from episode_length")
                for field in ("actions", "processed_actions", "obs", "states", "initial_state"):
                    if field not in demo:
                        raise Refusal(f"{key}: missing trajectory field {field}")
                    _series(demo[field], 1 if field == "initial_state" else steps)
                if "articulation" not in demo["states"] or not len(demo["states/articulation"]):
                    raise Refusal(f"{key}: missing articulated robot state")
                for articulation in demo["states/articulation"].values():
                    if not all(field in articulation for field in ("root_pose", "root_velocity", "joint_position", "joint_velocity")):
                        raise Refusal(f"{key}: incomplete articulation state")
                for state in _datasets(demo["states"]):
                    relative = state.name.removeprefix(demo["states"].name + "/")
                    if relative not in demo["initial_state"] or demo["initial_state"][relative].shape[1:] != state.shape[1:]:
                        raise Refusal(f"{key}: initial and post-step state layouts differ")
                if not isinstance(demo["actions"], h5py.Dataset) or demo["actions"].ndim != 2:
                    raise Refusal(f"{key}: actions must be a T x action_dim array")
                if "success" not in demo or not isinstance(demo["success"], h5py.Dataset):
                    raise Refusal(f"{key}: missing recorded success verdict")
                verdict = demo["success"][()]
                if (verdict.size != 1 or verdict.item() not in (0, 1)
                        or type(episode["success"]) is not bool or bool(verdict.item()) != episode["success"]):
                    raise Refusal(f"{key}: trajectory success differs from episode result")
                control = recorded(episode, "evaluation_settings")["control"]
                dt = control["sim_dt"] * control["decimation"]
                if not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0:
                    raise Refusal(f"{key}: invalid control step period")
                seen.add(key)
                joined.append({"run": key[0], "rebuild": key[1], "env_id": key[2], "episode_in_env": key[3],
                               "replay_source_record_index": episode["replay_source_record_index"],
                               "path": artifact["path"], "demo": name, "steps": steps,
                               "control_dt_s": dt, "success": episode["success"]})
    if seen != set(expected):
        raise Refusal(f"missing trajectories for {len(set(expected) - seen)} episodes (failures are required too)")
    return joined


def validate_run(root: Path, run: dict, submission: dict, manifest_path: Path) -> tuple[Path, list[dict]]:
    """Check immutable run evidence before passing raw episodes to the official scorer."""
    directory = bundle_path(root, run["directory"])
    checked_file(directory, {"path": "run_manifest.json", "sha256": run["manifest_sha256"]})
    metadata = read_run_manifest(directory)
    benchmark = submission["benchmark"]
    checkpoint = submission["checkpoints"][run["task"]]
    expected = {"task": run["task"], "cell": run["cell"], "section": "headline",
                "model": submission["model"]["key"], "checkpoint": checkpoint["label"],
                "arena_output": "arena",
                "record_dataset": True, "record_camera_video": True,
                "benchmark_manifest_sha256": benchmark["manifest_sha256"],
                "evaluation_contract_sha256": evaluation_contract_sha256(json.loads(manifest_path.read_text()), run["task"])}
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise Refusal(f"{directory}: run manifest {key} does not match submission")
    if metadata.get("code", {}).get("dirty") is not False:
        raise Refusal("submission requires a recorded clean evaluator checkout")
    entry = submission_model(json.loads(checked_file(root, submission["model_entry"]).read_text()))
    if entry["key"] != submission["model"]["key"] or metadata.get("model_entry") != entry:
        raise Refusal("run model_entry differs from the submitted model entry")
    if entry["registration"] == "config" and metadata.get("checkpoint_sha256") != checkpoint["sha256"]:
        raise Refusal("config model checkpoint hash differs from the run record")
    if metadata.get("code", {}).get("revision") != benchmark["code_commit"]:
        raise Refusal(f"{directory}: benchmark code revision differs")
    if metadata.get("eval_version") not in ({"v1", "v1-plus"} if benchmark["eval_version"] == "v1" else {"v1-plus"}):
        raise Refusal(f"{directory}: incompatible evaluation version")
    checked_file(directory, {"path": COMPOSED, "sha256": metadata["composed_sha256"]})
    result_path = checked_file(directory, {"path": "arena/arena_experiment_result.json", "sha256": run["result_sha256"]})
    result = json.loads(result_path.read_text())
    for result_run in result["runs"].values():
        for rebuild in result_run["rebuilds"]:
            for episode in rebuild["episodes"]:
                if recorded(episode, "evaluation_settings")["code"].get("dirty") is not False:
                    raise Refusal("submission episode requires a recorded clean evaluator checkout")
                if recorded(episode, "evaluation_settings")["code"]["revision"] != benchmark["code_commit"]:
                    raise Refusal(f"{directory}: episode code revision differs from submission")
    trajectories = validate_trajectories(directory, run["trajectories"], result["runs"])
    video_paths = set()
    for video in run["videos"]:
        if checked_file(directory, video).stat().st_size == 0:
            raise Refusal("empty rollout video")
        video_paths.add(Path(video["path"]))
    for episode in trajectories:
        filename = (f"robot-cam-rebuild{episode['rebuild']}-env{episode['env_id']}-"
                    f"{REQUIRED_VIDEO_CAMERA}-episode-{episode['episode_in_env']}.mp4")
        expected_video = Path("arena") / episode["run"] / filename
        if expected_video not in video_paths:
            raise Refusal(f"missing required {REQUIRED_VIDEO_CAMERA} video: {expected_video}; "
                          "every successful and failed episode must be listed")
    return directory, trajectories
