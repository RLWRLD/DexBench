# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 joint client against a stub of the cosmos-framework DexBench policy server."""

import base64
import importlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

pytest.importorskip("isaaclab_arena.assets.register", exc_type=ImportError)

from dexbench_arena.embodiments.rby1_wuji2.constants import (
    RBY1_WUJI2_JOINT_ACTION_NAMES,
    RBY1_WUJI2_WHEEL_JOINT_NAMES,
)
from dexbench_arena_cosmos.policy.cosmos3_joint_policy import (
    Cosmos3JointPolicy,
    Cosmos3JointPolicyCfg,
    representation_mismatch,
)

HORIZON = 4
TOKEN = "secret"


class StubServer(ThreadingHTTPServer):
    """Echo policy: every action row is the received state plus the row index."""

    def __init__(self):
        super().__init__(("127.0.0.1", 0), _Handler)
        self.requests = []
        self.info = {
            "joint_names": list(RBY1_WUJI2_JOINT_ACTION_NAMES),
            "frame_hw": [360, 640],
            "camera_set": "stereo_stack",
            "action_horizon": HORIZON,
        }


class _Handler(BaseHTTPRequestHandler):
    def _send(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self):
        if self.headers.get("Authorization") == f"Bearer {TOKEN}":
            return True
        self._send(401, {"error": "token"})
        return False

    def do_GET(self):
        if self._authorized():
            self._send(200, self.server.info)

    def do_POST(self):
        if not self._authorized():
            return
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append(req)
        state = np.asarray(req["state"], dtype=np.float32)
        self._send(200, {"action": (state[None] + np.arange(HORIZON)[:, None]).tolist()})

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    srv = StubServer()
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()


def environment():
    names = list(RBY1_WUJI2_JOINT_ACTION_NAMES)
    robot_names = list(RBY1_WUJI2_WHEEL_JOINT_NAMES) + names[::-1]
    action_names = names[16:] + names[:16]
    live = SimpleNamespace(
        num_envs=1,
        scene={"robot": SimpleNamespace(joint_names=robot_names)},
        action_manager=SimpleNamespace(
            total_action_dim=62, get_term=lambda _: SimpleNamespace(_joint_names=action_names)
        ),
    )
    rng = np.random.default_rng(0)
    frames = {
        f"robot_pov_{side}_cam_rgb": torch.from_numpy(rng.integers(0, 255, (1, 720, 1280, 3), dtype=np.uint8))
        for side in ("left", "right")
    }
    obs = {"policy": {"robot_joint_pos": torch.arange(66, dtype=torch.float32)[None]}, "camera_obs": frames}
    return SimpleNamespace(unwrapped=live), obs, robot_names, action_names


def test_names_bind_by_name_and_frames_follow_the_training_resize(server):
    env, obs, robot_names, action_names = environment()
    policy = Cosmos3JointPolicy(
        Cosmos3JointPolicyCfg(
            remote_url=f"http://127.0.0.1:{server.server_port}", remote_api_token=TOKEN, action_chunk_length=3
        )
    )
    policy.set_task_description("Lift the package from the cage onto the conveyor.")
    actions = [policy.get_action(env, obs) for _ in range(4)]

    assert len(server.requests) == 2
    joint = obs["policy"]["robot_joint_pos"][0]
    state = [float(joint[robot_names.index(n)]) for n in RBY1_WUJI2_JOINT_ACTION_NAMES]
    assert server.requests[0]["state"] == state
    expected = torch.tensor([float(joint[robot_names.index(n)]) for n in action_names])
    for step, action in zip((0, 1, 2, 0), actions):
        assert torch.equal(action[0], expected + step)

    for side in ("left", "right"):
        png = base64.b64decode(server.requests[0]["images"][side])
        sent = cv2.cvtColor(cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        rendered = obs["camera_obs"][f"robot_pov_{side}_cam_rgb"][0].numpy()
        assert np.array_equal(sent, cv2.resize(rendered, (640, 360), interpolation=cv2.INTER_AREA))


def test_rejects_a_chunk_longer_than_the_server_horizon_and_a_wrong_token(server):
    url = f"http://127.0.0.1:{server.server_port}"
    with pytest.raises(ValueError, match="action_chunk_length"):
        Cosmos3JointPolicy(Cosmos3JointPolicyCfg(remote_url=url, remote_api_token=TOKEN, action_chunk_length=5))
    with pytest.raises(RuntimeError, match="HTTP 401"):
        Cosmos3JointPolicy(Cosmos3JointPolicyCfg(remote_url=url, remote_api_token="wrong"))


def test_the_client_refuses_a_server_of_the_other_action_representation():
    """A server that states nothing is absolute."""
    unstated = {"joint_names": [], "action_horizon": 4}
    relative = {**unstated, "action_representation": {"type": "relative_joints", "relative_joint_slices": [[0, 14]]}}
    assert representation_mismatch(unstated, ()) is None and representation_mismatch(relative, ((0, 14),)) is None
    assert "expects relative_joints [[0, 14]]" in representation_mismatch(unstated, [[0, 14]])
    assert "expects absolute" in representation_mismatch(relative, ())
    assert "expects relative_joints [[0, 7]]" in representation_mismatch(relative, [[0, 7]])


def test_cosmos_server_declares_four_camera_inputs(server):
    from dexbench_arena.embodiments.rby1_wuji2.camera_contract import CAMERA_OBSERVATIONS

    server.info["camera_set"] = list(CAMERA_OBSERVATIONS)
    policy = Cosmos3JointPolicy(Cosmos3JointPolicyCfg(
        remote_url=f"http://127.0.0.1:{server.server_port}", remote_api_token=TOKEN, action_chunk_length=3))
    policy.set_task_description("Lift")
    env, obs, _, _ = environment()
    for side, pixel in (("left", 31), ("right", 79)):
        obs["camera_obs"][f"wrist_{side}_cam_rgb"] = torch.full((1, 360, 640, 3), pixel, dtype=torch.uint8)
    policy.get_action(env, obs)
    assert set(server.requests[0]["images"]) == set(CAMERA_OBSERVATIONS)
    png = base64.b64decode(server.requests[0]["images"]["camera_wrist_right"])
    assert np.all(cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR) == 79)
    assert policy.codec.stamp()["camera_mapping"] == policy.codec.provenance()["camera_mapping"]
    policy.reset()
    del obs["camera_obs"]["wrist_right_cam_rgb"]
    with pytest.raises(ValueError, match="wrist_right_cam_rgb"):
        policy.get_action(env, obs)
    policy.close()


def test_old_dotted_path_names_the_same_policy():
    module_path, class_name = "dexbench_arena.policy.cosmos3_joint_policy.Cosmos3JointPolicy".rsplit(".", 1)
    module = importlib.import_module(module_path)
    assert getattr(module, class_name) is Cosmos3JointPolicy
    assert module.Cosmos3JointPolicyCfg is Cosmos3JointPolicyCfg
