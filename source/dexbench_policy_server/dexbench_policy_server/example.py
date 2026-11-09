"""An example :class:`~dexbench_policy_server.ServedPolicy`: it holds the robot still (it answers the state it was
given), which is enough to check the plumbing of a launch before a model is in it::"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from dexbench_policy_server.policy import ServedPolicy

JOINT_NAMES = Path(__file__).with_name("rby1_wuji2_joint_names.json")
"""The robot's 62 action joints in the benchmark's canonical order; a test of the benchmark holds it to the
embodiment's constants. A model states the order it was trained in, and the client maps it to the robot by name."""


class HoldPose(ServedPolicy):
    """Answers the state it was given, ``action_horizon`` times: the robot holds still."""

    def load(self, checkpoint: str | None, options: dict[str, str]) -> None:
        self.joint_names = json.loads(Path(options.get("joint_names") or JOINT_NAMES).read_text())
        self.action_horizon = int(options.get("action_horizon", 16))
        height, width = (int(value) for value in options.get("frame_hw", "360x640").split("x"))
        self.frame_hw = [height, width]
        self.cameras = options.get("cameras", "camera_ego_left,camera_ego_right").split(",")

    def info(self) -> dict[str, Any]:
        return {"joint_names": self.joint_names, "action_horizon": self.action_horizon, "frame_hw": self.frame_hw,
                "camera_set": self.cameras, "action_representation": {"type": "absolute"}}

    def predict(self, images: dict[str, np.ndarray], state: np.ndarray, prompt: str) -> np.ndarray:
        return np.tile(state, (self.action_horizon, 1))
