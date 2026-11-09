"""Absolute-joint client for a Cosmos3 DexBench policy server, on :class:`RemoteJointPolicy`."""

from __future__ import annotations

import base64
import hashlib
import json
import urllib.error
import urllib.request
from dataclasses import dataclass

import cv2
import numpy as np

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase

from dexbench_arena.policy.image_preprocessing import (
    image_preprocessing_record,
    resize_rgb_frames,
)
from dexbench_arena.policy.policy_rate import PolicyRateCfg
from dexbench_arena.policy.remote_joint_policy import Codec, RemoteJointPolicy, Transport
from dexbench_arena.embodiments.rby1_wuji2.camera_contract import camera_mapping


@dataclass
class Cosmos3JointPolicyCfg(PolicyRateCfg):
    remote_url: str = "http://127.0.0.1:8000"
    remote_api_token: str | None = None
    action_chunk_length: int = 32
    request_timeout_s: float = 600.0
    policy_device: str = "cpu"
    left_camera: str = "robot_pov_left_cam_rgb"
    right_camera: str = "robot_pov_right_cam_rgb"
    robot_asset_name: str = "robot"
    joint_action_term: str = "joint_pos"
    policy_variant: str | None = None
    """The model this Run scores, as ``arena_experiment_result.json`` records it (Arena reads a
    cfg's ``policy_variant`` before the class name). The benchmark sets the registry key
    (``benchmark.models``), so two checkpoints of one policy class are two models."""
    relative_joint_slices: tuple = ()
    """The action columns the checkpoint was trained on relative to the state, as ``[start, stop)`` pairs
    (``[[0, 14]]``: both arms); empty for an absolute checkpoint. Checked against the server's ``/info``;
    the actions the client receives are absolute either way."""
    scheduler: str = "chunk"
    """``chunk`` or ``synced_batch`` (:data:`remote_joint_policy.SCHEDULERS`)."""


def _slices(slices) -> list[list[int]]:
    return [[int(start), int(stop)] for start, stop in (slices or ())]


def representation_mismatch(server_info: dict, relative_joint_slices) -> str | None:
    """Why a server stating ``server_info`` cannot serve a Run expecting ``relative_joint_slices``, or None."""
    served = server_info.get("action_representation") or {"type": "absolute"}
    expected = _slices(relative_joint_slices)
    served_slices = _slices(served.get("relative_joint_slices")) if served.get("type") == "relative_joints" else []
    if served.get("type") not in ("absolute", "relative_joints"):
        return f"the server states an unknown action representation {served!r}"
    if served_slices != expected:
        def name(slices):
            return f"relative_joints {slices}" if slices else "absolute"
        return (f"the server's checkpoint was trained with {name(served_slices)} actions, "
                f"this Run expects {name(expected)} (relative_joint_slices)")
    return None


class Cosmos3HttpTransport(Transport):
    """``GET /info`` and ``POST /predict`` with a bearer token; the server keeps no episode state."""

    def __init__(self, config: Cosmos3JointPolicyCfg):
        self.config = config

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.config.remote_api_token:
            headers["Authorization"] = f"Bearer {self.config.remote_api_token}"
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(
            self.config.remote_url.rstrip("/") + path, data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(req, timeout=self.config.request_timeout_s) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Cosmos3 server {path}: HTTP {e.code} {e.read().decode(errors='replace')}") from e

    def info(self):
        return self._request("GET", "/info")

    def infer(self, request, env_ids):
        response = self._request("POST", "/predict", request)
        if "error" in response:
            raise RuntimeError(f"Cosmos3 server error: {response['error']}")
        return response


class Cosmos3Codec(Codec):
    """The joint names, frame size and horizon the server states at ``/info``; one request per env, its head
    stereo pair area-downscaled to the training size and PNG-encoded, with the measured joint positions."""

    requests = "each"

    def __init__(self, config: Cosmos3JointPolicyCfg):
        self.config = config
        self._image_record = None

    def check(self, server_info):
        self.server_info = server_info
        self.joint_names = list(server_info["joint_names"])
        self.frame_hw = tuple(int(v) for v in server_info["frame_hw"])
        self.action_horizon = int(server_info["action_horizon"])
        mismatch = representation_mismatch(server_info, self.config.relative_joint_slices)
        if mismatch is not None:
            raise ValueError(f"Cosmos3 server at {self.config.remote_url}: {mismatch}")
        camera_set = server_info["camera_set"]
        if isinstance(camera_set, list):
            self.cameras = camera_mapping(camera_set)
        elif camera_set in ("left", "stereo_stack"):
            self.cameras = {"left": self.config.left_camera}
            if camera_set == "stereo_stack":
                self.cameras["right"] = self.config.right_camera
        else:
            raise ValueError(f"Unsupported Cosmos3 camera_set: {camera_set}")
        print(f"[Cosmos3JointPolicy] server {json.dumps(server_info)}", flush=True)

    def bind(self, robot_joint_names):
        self._state_ids = [robot_joint_names.index(n) for n in self.joint_names]

    def _png(self, frames: np.ndarray, name: str) -> str:
        """Base64 PNG of one ``(H, W, C)`` RGB render, area-downscaled to the training size."""
        frame = resize_rgb_frames(frames[None], *self.frame_hw)[0]
        ok, png = cv2.imencode(".png", np.ascontiguousarray(frame[..., ::-1]), [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if not ok:
            raise RuntimeError(f"PNG encoding failed for {name}")
        return base64.b64encode(png.tobytes()).decode("ascii")

    def encode(self, observation, env_ids, task_description):
        (env_id,) = env_ids
        cameras = observation["camera_obs"]
        missing = set(self.cameras.values()) - cameras.keys()
        if missing:
            raise ValueError(f"Missing required camera observations: {sorted(missing)}")
        frames = {k: cameras[cam][env_id].detach().cpu().numpy() for k, cam in self.cameras.items()}
        record = {k: image_preprocessing_record(f.shape[:2], self.frame_hw) for k, f in frames.items()}
        if record != self._image_record:
            self._image_record = record
            print(f"[Cosmos3JointPolicy] image_preprocessing={json.dumps(record)}", flush=True)
        return {
            "images": {k: self._png(f, k) for k, f in frames.items()},
            "state": observation["policy"]["robot_joint_pos"][env_id, self._state_ids].detach().cpu().double().tolist(),
            "prompt": task_description,
        }

    def decode(self, response, num_envs):
        actions = np.asarray(response["action"], dtype=np.float32)
        if actions.shape != (self.action_horizon, len(self.joint_names)) or not np.isfinite(actions).all():
            raise ValueError(f"Invalid Cosmos3 action chunk {actions.shape}")
        return actions[None]

    def stamp(self):
        return {"relative_joint_slices": _slices(self.config.relative_joint_slices), "camera_mapping": self.cameras}

    def provenance(self):
        return {
            "server_info": self.server_info,
            "camera_mapping": self.cameras,
            "contract_sha256": hashlib.sha256(json.dumps(self.server_info, sort_keys=True).encode()).hexdigest(),
        }


@register_policy
class Cosmos3JointPolicy(RemoteJointPolicy, PolicyBase[Cosmos3JointPolicyCfg]):
    name = "dexbench_cosmos3_joint"

    def __init__(self, config: Cosmos3JointPolicyCfg):
        super().__init__(config, Cosmos3HttpTransport(config), Cosmos3Codec(config), config.action_chunk_length)
