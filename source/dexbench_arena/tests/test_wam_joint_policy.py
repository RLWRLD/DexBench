# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""WAM joint client against a stub of the DexBench-FastWAM / DexBench-OpenWAM policy servers."""

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

from dexbench_arena.benchmark.models import MODELS_BY_KEY
from dexbench_arena.embodiments.rby1_wuji2.constants import (
    RBY1_WUJI2_JOINT_ACTION_NAMES,
    RBY1_WUJI2_WHEEL_JOINT_NAMES,
)
from dexbench_arena_wam.policy.wam_joint_policy import WamJointPolicy, WamJointPolicyCfg

HORIZON = 32
TOKEN = "secret"
GA_CAMERAS = ["camera_ego_left", "camera_wrist_left", "camera_wrist_right"]


class StubServer(ThreadingHTTPServer):
    """Echo policy stating the GA wrist-camera contract: every action row is the state plus the row index."""

    def __init__(self, family):
        super().__init__(("127.0.0.1", 0), _Handler)
        self.requests = []
        self.info = {
            "model_family": family,
            "joint_names": list(RBY1_WUJI2_JOINT_ACTION_NAMES),
            "camera_set": GA_CAMERAS,
            "frame_hw": [180, 320],
            "action_horizon": HORIZON,
            "action_representation": {"type": "absolute"},
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


@pytest.fixture(params=["fastwam", "openwam"])
def server(request):
    srv = StubServer(request.param)
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
        name: torch.from_numpy(rng.integers(0, 255, (1, 360, 640, 3), dtype=np.uint8))
        for name in ("robot_pov_left_cam_rgb", "robot_pov_right_cam_rgb", "wrist_left_cam_rgb", "wrist_right_cam_rgb")
    }
    obs = {"policy": {"robot_joint_pos": torch.arange(66, dtype=torch.float32)[None]}, "camera_obs": frames}
    return SimpleNamespace(unwrapped=live), obs, robot_names, action_names


def cfg(server, **kwargs):
    return WamJointPolicyCfg(remote_url=f"http://127.0.0.1:{server.server_port}", remote_api_token=TOKEN,
                             model_family=server.info["model_family"], action_chunk_length=16, **kwargs)


def test_sends_the_ga_cameras_at_the_stated_frame_and_binds_joints_by_name(server):
    env, obs, robot_names, action_names = environment()
    policy = WamJointPolicy(cfg(server))
    policy.set_task_description("Take the can out of the fridge.")
    actions = [policy.get_action(env, obs) for _ in range(17)]

    assert len(server.requests) == 2
    request = server.requests[0]
    assert sorted(request["images"]) == sorted(GA_CAMERAS) and request["prompt"] == "Take the can out of the fridge."
    joint = obs["policy"]["robot_joint_pos"][0]
    assert request["state"] == [float(joint[robot_names.index(n)]) for n in RBY1_WUJI2_JOINT_ACTION_NAMES]
    expected = torch.tensor([float(joint[robot_names.index(n)]) for n in action_names])
    assert torch.equal(actions[3][0], expected + 3) and torch.equal(actions[16][0], expected)
    for key, observation in (("camera_ego_left", "robot_pov_left_cam_rgb"), ("camera_wrist_right", "wrist_right_cam_rgb")):
        png = base64.b64decode(request["images"][key])
        sent = cv2.cvtColor(cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        rendered = obs["camera_obs"][observation][0].numpy()
        assert np.array_equal(sent, cv2.resize(rendered, (320, 180), interpolation=cv2.INTER_AREA))
    policy.close()


def test_refuses_a_server_of_another_family(server):
    other = "openwam" if server.info["model_family"] == "fastwam" else "fastwam"
    with pytest.raises(ValueError, match=f"expects '{other}'"):
        WamJointPolicy(WamJointPolicyCfg(remote_url=f"http://127.0.0.1:{server.server_port}",
                                         remote_api_token=TOKEN, model_family=other, action_chunk_length=16))
    server.info.pop("model_family")
    with pytest.raises(ValueError, match="states model_family None"):
        WamJointPolicy(WamJointPolicyCfg(remote_url=f"http://127.0.0.1:{server.server_port}",
                                         remote_api_token=TOKEN, action_chunk_length=16))


def test_registry_blocks_name_their_family():
    for key, port in (("fastwam", 5815), ("openwam", 5816)):
        policy = MODELS_BY_KEY[key].policy
        assert policy["type"] == "dexbench_arena_wam.policy.wam_joint_policy.WamJointPolicy"
        assert policy["model_family"] == policy["policy_variant"] == key
        assert policy["remote_url"] == f"http://127.0.0.1:{port}" and policy["action_chunk_length"] == 16


def test_old_dotted_path_names_the_same_policy():
    module_path, class_name = "dexbench_arena.policy.wam_joint_policy.WamJointPolicy".rsplit(".", 1)
    module = importlib.import_module(module_path)
    assert getattr(module, class_name) is WamJointPolicy
    assert module.WamJointPolicyCfg is WamJointPolicyCfg
