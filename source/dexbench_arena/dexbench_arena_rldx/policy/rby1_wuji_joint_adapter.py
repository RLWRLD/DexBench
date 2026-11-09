# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""RLDX-1 codec for RBY1-M Wuji-hand joint-space inference."""

from __future__ import annotations

import json
from collections import deque
from typing import Any

import numpy as np
import torch

from dexbench_arena.policy.image_preprocessing import resize_rgb_frames
from dexbench_arena.policy.remote_joint_policy import Codec
from dexbench_arena.embodiments.rby1_wuji2.camera_contract import EGO_CAMERAS, camera_mapping
from dexbench_arena.embodiments.rby1_wuji2.joint_groups import WUJI2_JOINT_GROUPS, WUJI2_JOINT_LAYOUTS

RLDX_RBY1_WUJI2_STATE_GROUPS = WUJI2_JOINT_GROUPS


class RldxRby1WujiJointCodec(Codec):
    """Requests and action chunks of an RLDX-1 server for the DexBench RBY1-Wuji envs."""

    left_video_key = "camera_ego_left"
    right_video_key = "camera_ego_right"
    language_key = "annotation.human.task_description"
    arena_left_camera_key = "robot_pov_left_cam_rgb"
    arena_right_camera_key = "robot_pov_right_cam_rgb"

    requests = "needed"

    def __init__(self, joint_layout: str = "wuji2_v1", target_image_size: tuple[int, int] = (360, 640),
                 denoising_timesteps: list[float] | None = None) -> None:
        if joint_layout not in WUJI2_JOINT_LAYOUTS:
            raise ValueError(f"Unknown joint layout: {joint_layout}")
        self.joint_layout = joint_layout
        self.denoising_timesteps = None if denoising_timesteps is None else [float(t) for t in denoising_timesteps]
        self.state_groups = WUJI2_JOINT_LAYOUTS[joint_layout]
        self.joint_names = [name for names in self.state_groups.values() for name in names]
        self.target_image_size = tuple(int(size) for size in target_image_size)
        self._history: dict[str, deque[np.ndarray]] | None = None
        self._warned_monocular = False
        self.cameras = dict(EGO_CAMERAS)

    @property
    def video_keys(self) -> tuple[str, ...]:
        return tuple(self.cameras)

    def check(self, configs) -> None:
        if not isinstance(configs, dict) or not configs:
            raise ValueError("RLDX server did not provide a modality contract")
        for kind in ("video", "state", "action"):
            cfg = configs.get(kind)
            if not getattr(cfg, "modality_keys", None) or not getattr(cfg, "delta_indices", None):
                raise ValueError(f"Server {kind} contract is missing keys or delta indices")
        self.cameras = camera_mapping(configs["video"].modality_keys)
        if self.left_video_key not in self.cameras:
            raise ValueError("RLDX video contract must include the left ego camera")
        for kind in ("state", "action"):
            if set(configs[kind].modality_keys) != set(self.state_groups):
                raise ValueError(f"Served checkpoint expects {kind} keys {sorted(configs[kind].modality_keys)}, "
                                 f"joint layout {self.joint_layout} provides {sorted(self.state_groups)}: this "
                                 "checkpoint was trained with a different group convention")
        video = [int(d) for d in configs["video"].delta_indices]
        if video != sorted(video) or video[-1] != 0:
            raise ValueError(f"Unexpected video delta_indices from server: {video}")
        action = [int(d) for d in configs["action"].delta_indices]
        if action != list(range(len(action))):
            raise ValueError("Server action deltas must be contiguous from zero")
        self.video_delta_indices = tuple(video)
        self.action_horizon = len(action)
        self.server_info = json.loads(json.dumps(configs, default=str))

    def bind(self, robot_joint_names):
        self._state_ids = {group: torch.tensor([robot_joint_names.index(name) for name in names], dtype=torch.long)
                           for group, names in self.state_groups.items()}

    def _current_frames(self, observation: dict[str, Any]) -> dict[str, np.ndarray]:
        """The served camera frames at the target size: {video_key: (num_envs, H, W, 3) uint8}."""
        cameras = observation["camera_obs"]
        left = cameras[self.arena_left_camera_key]
        right = cameras.get(self.arena_right_camera_key)
        if self.right_video_key in self.cameras and right is None:
            if not self._warned_monocular:
                print("[RldxRby1WujiJointCodec] No right ego camera; sending the left frame for both eyes.")
                self._warned_monocular = True
            right = left
        frames_by_source = dict(cameras)
        if right is not None:
            frames_by_source[self.arena_right_camera_key] = right
        missing = set(self.cameras.values()) - frames_by_source.keys()
        if missing:
            raise ValueError(f"Missing required camera observations: {sorted(missing)}")
        return {key: resize_rgb_frames(frames_by_source[source].detach().cpu().numpy(), *self.target_image_size)
                for key, source in self.cameras.items()}


    def observe(self, observation):
        frames = self._current_frames(observation)
        if self._history is None:
            window = 1 - min(self.video_delta_indices)
            self._history = {key: deque(maxlen=window) for key in self.video_keys}
            self._appended = 0
            self._episode_start = np.zeros(len(frames[self.left_video_key]), dtype=np.int64)
        for key, value in frames.items():
            self._history[key].append(value)
        self._appended += 1

    def reset(self, env_ids=None):
        if self._history is None:
            return
        ids = slice(None) if env_ids is None else torch.as_tensor(env_ids).reshape(-1).cpu().numpy()
        self._episode_start[ids] = self._appended

    def _video_windows(self, env_ids: list[int]) -> dict[str, np.ndarray]:
        """Per-camera frame windows at the delta offsets (oldest first), clamped to each env's episode start
        (repeat-padding its earliest frame) and to the oldest buffered frame."""
        latest = self._appended - 1
        oldest = self._appended - len(self._history[self.left_video_key])
        windows = {}
        for key, buffer in self._history.items():
            per_env = []
            for env_id in env_ids:
                floor = max(int(self._episode_start[env_id]), oldest)
                positions = [max(latest + delta, floor) - oldest for delta in self.video_delta_indices]
                per_env.append(np.stack([buffer[p][env_id] for p in positions]))
            windows[key] = np.ascontiguousarray(np.stack(per_env))
        return windows

    def encode(self, observation, env_ids, task_description):
        joint_pos = observation["policy"]["robot_joint_pos"][env_ids]
        state = {group: joint_pos[:, ids.to(joint_pos.device)].unsqueeze(1).detach().cpu().numpy().astype(np.float32)
                 for group, ids in self._state_ids.items()}
        return {
            "video": self._video_windows(env_ids),
            "state": state,
            "language": {self.language_key: [[task_description]] * len(env_ids)},
        }

    def decode(self, action, num_envs):
        parts = []
        for group, names in self.state_groups.items():
            if group not in action:
                raise ValueError(f"RLDX server response is missing action group {group!r}; got {sorted(action)}")
            values = np.asarray(action[group], dtype=np.float32)
            if values.shape != (num_envs, self.action_horizon, len(names)):
                raise ValueError(f"Invalid RLDX action shape for {group}: {values.shape}")
            parts.append(values)
        return np.concatenate(parts, axis=-1)

    def stamp(self):
        return {"joint_layout": self.joint_layout, "video_delta_indices": list(self.video_delta_indices),
                "image_size": list(self.target_image_size), "camera_mapping": self.cameras,
                "denoising_timesteps": self.denoising_timesteps}

    def provenance(self):
        return {"server_info": self.server_info, "camera_mapping": self.cameras}
