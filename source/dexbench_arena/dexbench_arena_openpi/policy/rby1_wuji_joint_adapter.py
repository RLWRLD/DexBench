# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""openpi codec for RBY1-M Wuji-hand joint-space checkpoints."""

from __future__ import annotations

import json

import numpy as np
from openpi_client import image_tools

from dexbench_arena.embodiments.rby1_wuji2.constants import RBY1_WUJI2_JOINT_ACTION_NAMES
from dexbench_arena.policy.remote_joint_policy import Codec
from dexbench_arena.embodiments.rby1_wuji2.camera_contract import camera_mapping

OPENPI_RBY1_WUJI2_JOINT_ORDER: tuple[str, ...] = tuple(RBY1_WUJI2_JOINT_ACTION_NAMES)

OPENPI_RBY1_WUJI_STATE_DIM = len(OPENPI_RBY1_WUJI2_JOINT_ORDER)
assert OPENPI_RBY1_WUJI_STATE_DIM == 62, OPENPI_RBY1_WUJI_STATE_DIM

IMAGE_SIZE = (224, 224)


class Pi0Rby1WujiJointCodec(Codec):
    """Requests and action chunks of an openpi server serving an RBY1-Wuji joint-space checkpoint."""

    left_camera = "robot_pov_left_cam_rgb"
    right_camera = "robot_pov_right_cam_rgb"
    requests = "each"

    def __init__(self, action_chunk_length: int) -> None:
        self.joint_names = list(OPENPI_RBY1_WUJI2_JOINT_ORDER)
        self.action_horizon = action_chunk_length
        self._warned_monocular = False

    def check(self, metadata) -> None:
        self.server_metadata = json.loads(json.dumps(metadata, default=str))
        keys = metadata.get("camera_keys")
        self.cameras = (camera_mapping(keys) if keys is not None else
                        {"robot_pov_left_cam": self.left_camera, "robot_pov_right_cam": self.right_camera})

    def bind(self, robot_joint_names):
        self._state_ids = [robot_joint_names.index(name) for name in self.joint_names]

    def _wire_image(self, frame: np.ndarray) -> np.ndarray:
        """(H, W, 3) uint8 env frame -> (3, 224, 224) uint8, letterboxed."""
        if frame.shape[:2] != IMAGE_SIZE:
            frame = image_tools.resize_with_pad(frame, *IMAGE_SIZE)
        return np.ascontiguousarray(frame.transpose(2, 0, 1))

    def encode(self, observation, env_ids, task_description):
        (env_id,) = env_ids
        cameras = observation["camera_obs"]
        frames = dict(cameras)
        if self.right_camera in self.cameras.values() and self.right_camera not in frames:
            if not self._warned_monocular:
                print("[Pi0Rby1WujiJointCodec] No right ego camera; sending the left frame for both slots.")
                self._warned_monocular = True
            frames[self.right_camera] = frames[self.left_camera]
        missing = set(self.cameras.values()) - frames.keys()
        if missing:
            raise ValueError(f"Missing required camera observations: {sorted(missing)}")
        images = {key: self._wire_image(frames[source][env_id].detach().cpu().numpy())
                  for key, source in self.cameras.items()}
        state = observation["policy"]["robot_joint_pos"][env_id, self._state_ids]
        return {
            "images": images,
            "state": state.detach().cpu().numpy().astype(np.float32),
            "prompt": task_description,
        }

    def decode(self, response, num_envs):
        chunk = np.asarray(response["actions"])
        if chunk.ndim != 2 or chunk.shape[1] != OPENPI_RBY1_WUJI_STATE_DIM:
            raise ValueError(f"Expected openpi actions of shape (H, {OPENPI_RBY1_WUJI_STATE_DIM}); got {chunk.shape}")
        if chunk.shape[0] < self.action_horizon:
            raise ValueError(f"openpi server returned horizon {chunk.shape[0]} < action_chunk_length {self.action_horizon}")
        return chunk[None, : self.action_horizon].astype(np.float32)

    def stamp(self):
        return {"image": {"size": list(IMAGE_SIZE), "resize": "openpi resize_with_pad"},
                "camera_mapping": self.cameras}

    def provenance(self):
        return {"server_metadata": self.server_metadata, "camera_mapping": self.cameras}
