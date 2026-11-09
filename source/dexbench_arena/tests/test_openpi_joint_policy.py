# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the RBY1-M Wuji-hand joint-space openpi remote policy and its codec."""

from __future__ import annotations

import socket
import time
import types

import numpy as np

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("isaaclab.managers", reason="needs the DexBench container runtime")
pytest.importorskip("isaaclab_arena.policy.action_scheduling", reason="needs the pinned Arena runtime")

from dexbench_arena.embodiments.rby1_wuji2.constants import (
    RBY1_WUJI2_WHEEL_JOINT_NAMES,
)
from dexbench_arena_openpi.policy import rby1_wuji_joint_policy
from dexbench_arena_openpi.policy.rby1_wuji_joint_adapter import OPENPI_RBY1_WUJI2_JOINT_ORDER
from dexbench_arena_openpi.policy.rby1_wuji_joint_policy import (
    OpenPiWebsocketTransport,
    Pi0Rby1WujiJointPolicyCfg,
    Pi0Rby1WujiJointRemotePolicy,
)

_EXTRA_JOINTS = tuple(RBY1_WUJI2_WHEEL_JOINT_NAMES)


def _articulation_joint_names(order=OPENPI_RBY1_WUJI2_JOINT_ORDER) -> list[str]:
    """A scrambled articulation ordering that differs from the openpi order."""
    names = list(order)[::-1]
    for offset, extra in enumerate(_EXTRA_JOINTS):
        names.insert(3 + 5 * offset, extra)
    return names


def _fake_env(num_envs: int, articulation_joint_names: list[str], action_term_joint_names: list[str]):
    robot = types.SimpleNamespace(joint_names=list(articulation_joint_names))
    action_term = types.SimpleNamespace(_joint_names=list(action_term_joint_names))
    action_manager = types.SimpleNamespace(get_term=lambda name: {"joint_pos": action_term}[name],
                                           total_action_dim=len(action_term_joint_names))
    unwrapped = types.SimpleNamespace(num_envs=num_envs, scene={"robot": robot}, action_manager=action_manager)
    return types.SimpleNamespace(unwrapped=unwrapped)


def _fake_observation(num_envs: int, articulation_joint_names: list[str], with_right_camera: bool = False) -> dict:
    joint_pos = torch.arange(len(articulation_joint_names), dtype=torch.float32).repeat(num_envs, 1)
    camera_obs = {"robot_pov_left_cam_rgb": torch.zeros((num_envs, 480, 640, 3), dtype=torch.uint8)}
    if with_right_camera:
        camera_obs["robot_pov_right_cam_rgb"] = torch.ones((num_envs, 480, 640, 3), dtype=torch.uint8)
    return {"camera_obs": camera_obs, "policy": {"robot_joint_pos": joint_pos}}


class _FakeWebsocket:
    """Stands in for the websocket client: answers a horizon-20 chunk whose dims carry their openpi index."""

    def __init__(self, **kwargs):
        self.requests = []
        self._ws = None

    def get_server_metadata(self):
        return {}

    def infer(self, request):
        self.requests.append(request)
        return {"actions": np.tile(np.arange(62, dtype=np.float32), (20, 1))}


@pytest.fixture
def server(monkeypatch):
    fake = _FakeWebsocket()
    monkeypatch.setattr(rby1_wuji_joint_policy, "server_listening", lambda host, port, timeout: None)
    monkeypatch.setattr(rby1_wuji_joint_policy, "WebsocketClientPolicy", lambda **kwargs: fake)
    return fake


def _policy(**overrides) -> Pi0Rby1WujiJointRemotePolicy:
    policy = Pi0Rby1WujiJointRemotePolicy(Pi0Rby1WujiJointPolicyCfg(**{"policy_device": "cpu", **overrides}))
    policy.set_task_description("pick and place object")
    return policy


def test_openpi_joint_order_layout():
    order = OPENPI_RBY1_WUJI2_JOINT_ORDER
    assert len(order) == 62
    assert order[0] == "left_arm_0"
    assert order[7] == "right_arm_0"
    assert order[14] == "head_0"
    assert order[56:] == ("torso_0", "torso_1", "torso_2", "torso_3", "torso_4", "torso_5")
    assert order[16] == "left_thumb_cmc_flex"
    assert order[21] == "right_thumb_cmc_flex"
    assert order[55] == "right_pinky_dip"


def test_request_reorders_state_and_letterboxes_frames(server):
    order = OPENPI_RBY1_WUJI2_JOINT_ORDER
    articulation = _articulation_joint_names(order)
    policy = _policy()
    obs = _fake_observation(num_envs=2, articulation_joint_names=articulation)
    obs["camera_obs"]["robot_pov_left_cam_rgb"][:] = 255
    policy.get_action(_fake_env(2, articulation, list(order)), obs)

    request = server.requests[1]
    assert set(request) == {"images", "state", "prompt"}
    np.testing.assert_array_equal(request["state"], [articulation.index(name) for name in order])
    assert request["state"].dtype == np.float32
    assert request["prompt"] == "pick and place object"
    image = request["images"]["robot_pov_left_cam"]
    assert image.shape == (3, 224, 224) and image.dtype == np.uint8
    assert image[:, 0, :].max() == 0 and image[:, -1, :].max() == 0
    assert image[:, 112, :].min() == 255


def test_request_duplicates_left_camera_when_monocular(server):
    articulation = _articulation_joint_names()
    env = _fake_env(1, articulation, list(OPENPI_RBY1_WUJI2_JOINT_ORDER))
    _policy().get_action(env, _fake_observation(1, articulation, with_right_camera=False))
    images = server.requests[0]["images"]
    np.testing.assert_array_equal(images["robot_pov_left_cam"], images["robot_pov_right_cam"])

    _policy().get_action(env, _fake_observation(1, articulation, with_right_camera=True))
    images = server.requests[1]["images"]
    assert images["robot_pov_right_cam"].max() == 1 and images["robot_pov_left_cam"].max() == 0


def test_policy_permutes_actions_into_env_action_term_order(server):
    order = OPENPI_RBY1_WUJI2_JOINT_ORDER
    articulation = _articulation_joint_names(order)

    for term_joint_names in (list(order), [name for name in articulation if name in order]):
        actions = _policy().get_action(_fake_env(2, articulation, term_joint_names),
                                       _fake_observation(num_envs=2, articulation_joint_names=articulation))
        assert actions.shape == (2, 62)
        expected = torch.tensor([order.index(name) for name in term_joint_names], dtype=torch.float32).repeat(2, 1)
        torch.testing.assert_close(actions, expected)


def test_policy_replays_chunk_for_action_chunk_length(server):
    articulation = _articulation_joint_names()
    policy = _policy()
    env = _fake_env(1, articulation, list(OPENPI_RBY1_WUJI2_JOINT_ORDER))
    obs = _fake_observation(num_envs=1, articulation_joint_names=articulation)
    for _ in range(16):
        policy.get_action(env, obs)
    assert len(server.requests) == 1
    policy.get_action(env, obs)
    assert len(server.requests) == 2


def test_policy_refuses_a_chunk_shorter_than_its_execution_horizon(server):
    articulation = _articulation_joint_names()
    policy = _policy(action_chunk_length=24)
    with pytest.raises(ValueError, match="horizon 20"):
        policy.get_action(_fake_env(1, articulation, list(OPENPI_RBY1_WUJI2_JOINT_ORDER)),
                          _fake_observation(num_envs=1, articulation_joint_names=articulation))


def test_openpi_server_declares_four_canonical_image_keys(server):
    from dexbench_arena.embodiments.rby1_wuji2.camera_contract import CAMERA_OBSERVATIONS

    server.get_server_metadata = lambda: {"camera_keys": list(CAMERA_OBSERVATIONS)}
    policy = _policy()
    articulation = _articulation_joint_names()
    env = _fake_env(1, articulation, list(OPENPI_RBY1_WUJI2_JOINT_ORDER))
    obs = _fake_observation(1, articulation, with_right_camera=True)
    for side, pixel in (("left", 31), ("right", 79)):
        obs["camera_obs"][f"wrist_{side}_cam_rgb"] = torch.full((1, 360, 640, 3), pixel, dtype=torch.uint8)
    policy.get_action(env, obs)
    assert set(server.requests[0]["images"]) == set(CAMERA_OBSERVATIONS)
    assert server.requests[0]["images"]["camera_wrist_right"][:, 112, 112].tolist() == [79] * 3
    assert policy.codec.stamp()["camera_mapping"] == policy.codec.provenance()["camera_mapping"]
    policy.reset()
    del obs["camera_obs"]["wrist_left_cam_rgb"]
    with pytest.raises(ValueError, match="wrist_left_cam_rgb"):
        policy.get_action(env, obs)
    policy.close()


def test_a_server_that_is_not_listening_is_reported_within_the_timeout():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    start = time.monotonic()
    with pytest.raises(ConnectionError, match="Cannot reach the openpi server"):
        OpenPiWebsocketTransport(Pi0Rby1WujiJointPolicyCfg(remote_host="127.0.0.1", remote_port=port, connect_timeout=1.0))
    assert time.monotonic() - start < 5


def test_a_reconnect_to_a_server_that_died_is_reported(monkeypatch):
    calls = []

    def listening_once(host, port, timeout):
        calls.append(port)
        if len(calls) > 1:
            raise ConnectionError("Cannot reach the openpi server")

    class Dropped(_FakeWebsocket):
        def infer(self, request):
            raise OSError("connection lost")

    monkeypatch.setattr(rby1_wuji_joint_policy, "server_listening", listening_once)
    monkeypatch.setattr(rby1_wuji_joint_policy, "WebsocketClientPolicy", lambda **kwargs: Dropped())
    transport = OpenPiWebsocketTransport(Pi0Rby1WujiJointPolicyCfg())
    with pytest.raises(ConnectionError, match="Cannot reach the openpi server"):
        transport.infer({}, [0])
    assert len(calls) == 2
