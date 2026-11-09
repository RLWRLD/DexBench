# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the RBY1-M Wuji-hand RLDX-1 remote policy and its codec."""

from __future__ import annotations

import numpy as np
import types
import argparse
import json
import runpy
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("isaaclab.managers", reason="needs the DexBench container runtime")
pytest.importorskip("isaaclab_arena.policy.action_scheduling", reason="needs the pinned Arena runtime")

from dexbench_arena.embodiments.rby1_wuji2.constants import (
    RBY1_WUJI2_JOINT_ACTION_NAMES,
    RBY1_WUJI2_WHEEL_JOINT_NAMES,
)
from dexbench_arena_rldx.policy.rby1_wuji_joint_adapter import RLDX_RBY1_WUJI2_STATE_GROUPS
from dexbench_arena_rldx.policy.rby1_wuji_joint_policy import (
    RldxRby1WujiJointRemotePolicy,
    RldxRemotePolicyCfg,
)
from dexbench_arena_rldx.policy.dataset_contract import DATASET_JOINT_GROUPS

_EXTRA_JOINTS = tuple(RBY1_WUJI2_WHEEL_JOINT_NAMES)

_FLAT_GROUP_JOINTS = [name for names in RLDX_RBY1_WUJI2_STATE_GROUPS.values() for name in names]
_JOINT_VALUE = {name: float(index) for index, name in enumerate(_FLAT_GROUP_JOINTS)}

_TERM_JOINT_NAMES = list(RBY1_WUJI2_JOINT_ACTION_NAMES)

_TEST_DELTAS = [-4, -2, 0]


def test_upstream_registry_resolves_rldx_config():
    from isaaclab_arena.assets.registries import PolicyRegistry
    from isaaclab_arena.policy.policy_base import PolicyCfg

    registry = PolicyRegistry()
    assert issubclass(RldxRemotePolicyCfg, PolicyCfg)
    assert registry.get_policy_cfg_type(RldxRby1WujiJointRemotePolicy) is RldxRemotePolicyCfg
    assert registry.get_policy_type_for_cfg(RldxRemotePolicyCfg()) is RldxRby1WujiJointRemotePolicy


@pytest.fixture
def runner_parser(monkeypatch):
    """Enter the real launcher, stopping before simulator startup."""
    from isaaclab_arena.evaluation import policy_runner_cli

    from dexbench_arena.cli import get_dexbench_arena_cli_parser

    parsers = []

    def run_upstream_evaluation_script(script_name):
        assert script_name == "policy_runner.py"
        parser = get_dexbench_arena_cli_parser(argparse.ArgumentParser())
        parsers.append(policy_runner_cli.add_policy_cli_args(parser, RldxRby1WujiJointRemotePolicy))

    launcher = types.ModuleType("_dexbench_arena_script")
    launcher.run_upstream_evaluation_script = run_upstream_evaluation_script
    monkeypatch.setitem(sys.modules, "_dexbench_arena_script", launcher)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.setattr(policy_runner_cli, "add_policy_cli_args", policy_runner_cli.add_policy_cli_args)
    script = Path(__file__).resolve().parents[3] / "scripts" / "evaluation" / "policy_runner.py"
    runpy.run_path(str(script), run_name="__main__")
    return parsers[0]


@pytest.mark.parametrize("image_size", [None, [180, 320, 3]])
@pytest.mark.parametrize("image_size_last", [False, True], ids=["before_other_options", "before_environment"])
def test_runner_cli_builds_rldx_without_model(monkeypatch, runner_parser, image_size, image_size_last):
    from isaaclab_arena.evaluation.policy_runner_cli import build_policy_from_cli

    policy_args = ["--policy_device", "cpu", "--action_chunk_length", "4", "--scheduler", "chunk"]
    image_args = []
    if image_size is not None:
        image_args = ["--target_image_size", *map(str, image_size)]
    argv = policy_args + image_args if image_size_last else image_args + policy_args
    argv += ["dexbench_retail_v1_environment", "--embodiment", "dexbench_rby1_wuji2_joint", "--arm_mode", "dual_arm"]
    args = runner_parser.parse_args(argv)
    assert args.example_environment == "dexbench_retail_v1_environment"
    assert args.embodiment == "dexbench_rby1_wuji2_joint"
    assert args.arm_mode == "dual_arm"
    client = _FakeRldxClient(video_deltas=_TEST_DELTAS)
    client.ping = lambda: True
    client.timeout_ms = 15000
    client.socket = types.SimpleNamespace(setsockopt=lambda option, value: None, close=lambda **kwargs: None)
    module = types.ModuleType("rldx.policy.server_client")
    module.PolicyClient = lambda **kwargs: client
    monkeypatch.setitem(sys.modules, "rldx.policy.server_client", module)
    policy = build_policy_from_cli(RldxRby1WujiJointRemotePolicy, args)
    expected_size = image_size or [360, 640, 3]
    assert policy.config == RldxRemotePolicyCfg(
        policy_device="cpu", action_chunk_length=4, scheduler="chunk", target_image_size=expected_size
    )
    policy.set_task_description("Retrieve the can into the basket.")
    env = _fake_env(1, _articulation_joint_names())
    observation = _fake_observation(1, _articulation_joint_names())
    assert policy.get_action(env, observation).shape == (1, 62)
    policy.reset()
    assert policy.get_action(env, observation).shape == (1, 62)
    assert len(client.calls) == 2
    for request, _ in client.calls:
        for camera, pixel in (("camera_ego_left", 0), ("camera_ego_right", 100)):
            frames = request["video"][camera]
            assert frames.shape == (1, len(_TEST_DELTAS), *expected_size)
            assert frames.dtype == np.uint8
            np.testing.assert_array_equal(frames, np.full(frames.shape, pixel, dtype=np.uint8))
    policy.close()


@pytest.mark.parametrize("sizes", [[], [180], [180, 320], [180, 320, 3, 3], [180, "wide", 3]])
@pytest.mark.parametrize("parse_method", ["parse_args", "parse_known_args"])
def test_runner_cli_rejects_invalid_image_arguments(runner_parser, sizes, parse_method):
    argv = ["--target_image_size", *map(str, sizes), "dexbench_retail_v1_environment"]
    with pytest.raises(SystemExit) as error:
        getattr(runner_parser, parse_method)(argv)
    assert error.value.code == 2


def test_runner_cli_preserves_hydra_overrides(runner_parser):
    from isaaclab_arena.utils.hydra_overrides import assert_hydra_overrides

    argv = [
        "--target_image_size", "180", "320", "3",
        "dexbench_retail_v1_environment", "--arm_mode", "dual_arm",
        "light.hdr_image.enabled=true",
    ]
    args, overrides = runner_parser.parse_known_args(argv)
    assert args.target_image_size == [180, 320, 3]
    assert args.example_environment == "dexbench_retail_v1_environment"
    assert overrides == ["light.hdr_image.enabled=true"]
    assert_hydra_overrides(overrides, runner_parser)
    assert "--target_image_size HEIGHT WIDTH CHANNELS" in runner_parser.format_help()


def test_runner_cli_keeps_the_environment_after_denoising_timesteps(runner_parser):
    argv = ["--denoising_timesteps", "0,0.25,0.5,0.75", "dexbench_retail_v1_environment", "--arm_mode", "dual_arm"]
    args, _ = runner_parser.parse_known_args(argv)
    assert args.denoising_timesteps == [0.0, 0.25, 0.5, 0.75]
    assert args.example_environment == "dexbench_retail_v1_environment"
    assert runner_parser.parse_args(["dexbench_retail_v1_environment"]).denoising_timesteps is None
    with pytest.raises(SystemExit) as error:
        runner_parser.parse_args(["--denoising_timesteps", "0,fast", "dexbench_retail_v1_environment"])
    assert error.value.code == 2


def test_cli_compatibility_preserves_openpi_options():
    from isaaclab_arena.evaluation.policy_runner_cli import (
        add_policy_cli_args as add_upstream_policy_args,
        policy_cfg_from_cli,
    )

    from dexbench_arena.cli import get_dexbench_arena_cli_parser
    from dexbench_arena.evaluation.policy_runner_cli import add_policy_cli_args
    from dexbench_arena_openpi.policy.rby1_wuji_joint_policy import Pi0Rby1WujiJointRemotePolicy

    argv = [
        "--policy_variant", "openpi_pi05", "--policy_device", "cpu", "--action_chunk_length", "8",
        "dexbench_retail_v1_environment", "--arm_mode", "dual_arm",
    ]
    configs = []
    for add_args in (add_upstream_policy_args, add_policy_cli_args):
        parser = get_dexbench_arena_cli_parser(argparse.ArgumentParser())
        parser = add_args(parser, Pi0Rby1WujiJointRemotePolicy)
        configs.append(policy_cfg_from_cli(Pi0Rby1WujiJointRemotePolicy, parser.parse_args(argv)))
    assert configs[0] == configs[1]
    assert configs[1].action_chunk_length == 8 and configs[1].policy_variant == "openpi_pi05"


@pytest.mark.parametrize("kwargs", [{"action_chunk_length": 0}, {"action_chunk_length": -1}, {"scheduler": "invalid"}])
def test_config_rejects_invalid_scheduling_before_connection(kwargs):
    with pytest.raises(ValueError):
        RldxRemotePolicyCfg(**kwargs)


@pytest.mark.parametrize("steps", [[], [0.5, 0.25], [0.0, 1.0], [-0.1, 0.5], [True], "0 0.5"])
def test_config_rejects_invalid_denoising_timesteps_before_connection(steps):
    with pytest.raises(ValueError, match="denoising_timesteps"):
        RldxRemotePolicyCfg(denoising_timesteps=steps)


@pytest.mark.parametrize(
    "size",
    [[], [180, 320], [0, 320, 3], [180, -1, 3], [180, 320, 4], [180.5, 320, 3], [True, 320, 3]],
)
def test_config_rejects_invalid_image_contract_before_connection(size):
    with pytest.raises(ValueError, match="target_image_size"):
        RldxRemotePolicyCfg(target_image_size=size)


def _articulation_joint_names() -> list[str]:
    """A scrambled articulation ordering that differs from every wire ordering."""
    names = list(_FLAT_GROUP_JOINTS)[::-1]
    for offset, extra in enumerate(_EXTRA_JOINTS):
        names.insert(3 + 5 * offset, extra)
    return names


def _fake_env(num_envs: int, articulation_joint_names: list[str]):
    robot = types.SimpleNamespace(joint_names=list(articulation_joint_names))
    action_term = types.SimpleNamespace(_joint_names=list(_TERM_JOINT_NAMES))
    action_manager = types.SimpleNamespace(get_term=lambda name: {"joint_pos": action_term}[name],
                                           total_action_dim=len(_TERM_JOINT_NAMES))
    unwrapped = types.SimpleNamespace(
        num_envs=num_envs,
        scene={"robot": robot},
        action_manager=action_manager,
    )
    return types.SimpleNamespace(unwrapped=unwrapped)


def _fake_observation(
    num_envs: int,
    articulation_joint_names: list[str],
    frame_label: int = 0,
    with_right_camera: bool = True,
) -> dict:
    joint_pos = torch.arange(len(articulation_joint_names), dtype=torch.float32).repeat(num_envs, 1)
    camera_obs = {
        "robot_pov_left_cam_rgb": torch.full((num_envs, 360, 640, 3), frame_label % 256, dtype=torch.uint8)
    }
    if with_right_camera:
        camera_obs["robot_pov_right_cam_rgb"] = torch.full(
            (num_envs, 360, 640, 3), (frame_label + 100) % 256, dtype=torch.uint8
        )
    return {
        "camera_obs": camera_obs,
        "policy": {"robot_joint_pos": joint_pos},
    }


class _FakeRldxClient:
    """Stands in for rldx.policy.server_client.PolicyClient."""

    def __init__(self, horizon: int = 16, video_deltas: list[int] | None = None):
        self.horizon = horizon
        self.video_deltas = video_deltas
        self.calls: list[tuple[dict, dict]] = []

    def get_modality_config(self):
        if self.video_deltas is None:
            return {}
        return {
            "video": types.SimpleNamespace(
                modality_keys=["camera_ego_left", "camera_ego_right"],
                delta_indices=list(self.video_deltas),
            ),
            "state": types.SimpleNamespace(
                modality_keys=list(RLDX_RBY1_WUJI2_STATE_GROUPS), delta_indices=[0]
            ),
            "action": types.SimpleNamespace(
                modality_keys=list(RLDX_RBY1_WUJI2_STATE_GROUPS),
                delta_indices=list(range(self.horizon)),
            ),
        }

    def get_action(self, observation, options=None):
        self.calls.append((observation, options))
        batch = observation["video"]["camera_ego_left"].shape[0]
        action = {}
        for group, names in RLDX_RBY1_WUJI2_STATE_GROUPS.items():
            values = np.array([_JOINT_VALUE[name] for name in names], dtype=np.float32)
            action[group] = np.tile(values, (batch, self.horizon, 1))
        return action, {}


def _make_policy(
    client: _FakeRldxClient, scheduler: str = "synced_batch", action_chunk_length: int = 16, **cfg
) -> RldxRby1WujiJointRemotePolicy:
    policy = RldxRby1WujiJointRemotePolicy(
        RldxRemotePolicyCfg(policy_device="cpu", scheduler=scheduler, action_chunk_length=action_chunk_length, **cfg),
        client=client,
    )
    policy.set_task_description("Open the microwave.")
    return policy


def _window_labels(request: dict, camera: str, env_row: int) -> list[int]:
    """The frame-fill labels of one env's sampled window, oldest first."""
    return [int(frame[0, 0, 0]) for frame in request["video"][camera][env_row]]


def test_state_group_layout():
    assert list(RLDX_RBY1_WUJI2_STATE_GROUPS) == [
        "left_arm_joints",
        "right_arm_joints",
        "head_joints",
        "hands_joints",
        "torso_joints",
    ]
    sizes = {group: len(names) for group, names in RLDX_RBY1_WUJI2_STATE_GROUPS.items()}
    assert sizes == {
        "left_arm_joints": 7,
        "right_arm_joints": 7,
        "head_joints": 2,
        "hands_joints": 40,
        "torso_joints": 6,
    }
    assert tuple(_FLAT_GROUP_JOINTS) == tuple(RBY1_WUJI2_JOINT_ACTION_NAMES)


def test_config_rejects_unknown_hand_variant():
    with pytest.raises(ValueError, match="hand variant"):
        RldxRemotePolicyCfg(hand_variant="unknown")


def test_policy_refuses_a_server_without_a_modality_contract():
    with pytest.raises(ValueError, match="modality contract"):
        _make_policy(_FakeRldxClient(video_deltas=None))


def test_policy_adopts_server_video_deltas_and_pads_first_window():
    client = _FakeRldxClient(video_deltas=_TEST_DELTAS)
    policy = _make_policy(client)
    articulation = _articulation_joint_names()
    env = _fake_env(1, articulation)

    policy.get_action(env, _fake_observation(1, articulation, frame_label=0))
    request, _ = client.calls[0]
    assert request["video"]["camera_ego_left"].shape == (1, len(_TEST_DELTAS), 360, 640, 3)
    assert _window_labels(request, "camera_ego_left", 0) == [0, 0, 0]
    assert request["language"]["annotation.human.task_description"] == [["Open the microwave."]]


def test_policy_sends_the_left_frame_for_both_eyes_when_monocular():
    client = _FakeRldxClient(video_deltas=[0])
    policy = _make_policy(client)
    articulation = _articulation_joint_names()
    policy.get_action(_fake_env(1, articulation), _fake_observation(1, articulation, 9, with_right_camera=False))
    request, _ = client.calls[0]
    np.testing.assert_array_equal(request["video"]["camera_ego_left"], request["video"]["camera_ego_right"])


@pytest.mark.parametrize("image_size", [[360, 640, 3], [180, 320, 3]])
def test_policy_video_window_sampling_at_offsets(image_size):
    from isaaclab_arena.evaluation.legacy_eval_config import run_cfgs_from_legacy_eval_config

    config = {"jobs": [{
        "name": "rldx_image_contract",
        "arena_env_args": {"environment": "test_scene.yaml"},
        "policy_type": "dexbench_arena_rldx.policy.rby1_wuji_joint_policy.RldxRby1WujiJointRemotePolicy",
        "policy_config_dict": {"policy_device": "cpu", "target_image_size": image_size, "action_chunk_length": 8},
        "num_steps": 9,
    }]}
    run_cfg, = run_cfgs_from_legacy_eval_config(config, device="cpu")
    client = _FakeRldxClient(video_deltas=_TEST_DELTAS)
    policy = RldxRby1WujiJointRemotePolicy(run_cfg.policy, client=client)
    policy.set_task_description("Retrieve the can.")
    articulation = _articulation_joint_names()
    env = _fake_env(1, articulation)

    for step in range(9):
        policy.get_action(env, _fake_observation(1, articulation, frame_label=step))
    assert len(client.calls) == 2
    request, _ = client.calls[1]
    for video in request["video"].values():
        assert video.shape == (1, len(_TEST_DELTAS), *image_size)
    assert _window_labels(request, "camera_ego_left", 0) == [4, 6, 8]
    assert _window_labels(request, "camera_ego_right", 0) == [104, 106, 108]
    policy.close()


def test_policy_reset_clamps_video_window_to_episode_start():
    client = _FakeRldxClient(video_deltas=_TEST_DELTAS)
    policy = _make_policy(client, scheduler="chunk", action_chunk_length=8)
    articulation = _articulation_joint_names()
    env = _fake_env(2, articulation)

    for step in range(6):
        policy.get_action(env, _fake_observation(2, articulation, frame_label=step))
    assert len(client.calls) == 1

    policy.reset(env_ids=torch.tensor([0]))
    policy.get_action(env, _fake_observation(2, articulation, frame_label=6))
    assert len(client.calls) == 2
    request, options = client.calls[1]
    assert request["video"]["camera_ego_left"].shape[0] == 1
    assert _window_labels(request, "camera_ego_left", 0) == [6, 6, 6]
    assert options["reset_memory"] == [True]


def test_policy_sessions_memory_resets_and_actions():
    client = _FakeRldxClient(video_deltas=_TEST_DELTAS)
    policy = _make_policy(client, action_chunk_length=8)
    articulation = _articulation_joint_names()
    env = _fake_env(2, articulation)
    obs = _fake_observation(num_envs=2, articulation_joint_names=articulation)

    actions = policy.get_action(env, obs)
    assert actions.shape == (2, 62)
    expected = torch.tensor([_JOINT_VALUE[name] for name in _TERM_JOINT_NAMES], dtype=torch.float32).repeat(2, 1)
    torch.testing.assert_close(actions, expected)
    request, options = client.calls[0]
    for group, names in RLDX_RBY1_WUJI2_STATE_GROUPS.items():
        state = request["state"][group]
        assert state.shape == (2, 1, len(names)) and state.dtype == np.float32
        np.testing.assert_array_equal(state[0, 0], [articulation.index(name) for name in names])

    assert len(set(options["session_ids"])) == 2
    assert options["reset_memory"] == [True, True]

    for _ in range(7):
        policy.get_action(env, obs)
    assert len(client.calls) == 1
    policy.get_action(env, obs)
    assert len(client.calls) == 2
    _, options = client.calls[1]
    assert options["session_ids"] == client.calls[0][1]["session_ids"]
    assert options["reset_memory"] == [False, False]



def test_a_stateless_server_gets_no_session_options():
    """RLDX 7a518b5a's stateless policy refuses session_ids and reset_memory; the GA rldx11 entry sends none."""
    client = _FakeRldxClient(video_deltas=_TEST_DELTAS)
    policy = _make_policy(client, action_chunk_length=8, memory_sessions=False)
    articulation = _articulation_joint_names()
    env = _fake_env(2, articulation)
    obs = _fake_observation(num_envs=2, articulation_joint_names=articulation)
    for _ in range(8):
        policy.get_action(env, obs)
    policy.reset(env_ids=torch.tensor([0]))
    policy.get_action(env, obs)
    assert len(client.calls) == 2
    assert [options for _, options in client.calls] == [None, None]

def test_policy_flags_memory_reset_only_for_reset_envs():
    client = _FakeRldxClient(video_deltas=_TEST_DELTAS)
    policy = _make_policy(client, action_chunk_length=8)
    articulation = _articulation_joint_names()
    env = _fake_env(2, articulation)
    obs = _fake_observation(num_envs=2, articulation_joint_names=articulation)

    policy.get_action(env, obs)
    policy.reset(env_ids=torch.tensor([0]))

    hold_expected = torch.tensor(
        [articulation.index(name) for name in _TERM_JOINT_NAMES], dtype=torch.float32
    )
    for _ in range(7):
        actions = policy.get_action(env, obs)
        torch.testing.assert_close(actions[0], hold_expected)
    assert len(client.calls) == 1

    policy.get_action(env, obs)
    assert len(client.calls) == 2
    _, options = client.calls[1]
    assert options["reset_memory"] == [True, False]


@pytest.mark.parametrize("server_horizon", [16, 8])
def test_raw_dataset_six_group_contract_scatter_and_horizon(server_horizon):
    groups = DATASET_JOINT_GROUPS

    class DatasetClient(_FakeRldxClient):
        def get_modality_config(self):
            cfgs = super().get_modality_config()
            for kind in ("state", "action"):
                cfgs[kind].modality_keys = list(groups)
            return cfgs

        def get_action(self, observation, options=None):
            self.calls.append((observation, options))
            return {key: np.repeat(value, self.horizon, axis=1)
                    for key, value in observation["state"].items()}, {}

    client = DatasetClient(horizon=server_horizon, video_deltas=[0])
    steps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    cfg = RldxRemotePolicyCfg(policy_device="cpu", joint_layout="wuji2_rrc_v1", denoising_timesteps=steps)
    if server_horizon < 16:
        with pytest.raises(ValueError, match="action_chunk_length"):
            RldxRby1WujiJointRemotePolicy(cfg, client=client)
        assert client.calls == []
        return
    policy = RldxRby1WujiJointRemotePolicy(cfg, client=client)
    policy.set_task_description("Stack the dishes.")
    articulation = _articulation_joint_names()
    env = _fake_env(1, articulation)
    obs = _fake_observation(1, articulation, frame_label=17)
    action = policy.get_action(env, obs)
    expected = obs["policy"]["robot_joint_pos"][:, [articulation.index(n) for n in _TERM_JOINT_NAMES]]
    torch.testing.assert_close(action, expected, rtol=0, atol=0)
    request, _ = client.calls[0]
    assert list(request["state"]) == list(DATASET_JOINT_GROUPS)
    assert request["video"]["camera_ego_left"].shape == (1, 1, 360, 640, 3)
    np.testing.assert_array_equal(request["video"]["camera_ego_left"][:, 0],
                                  obs["camera_obs"]["robot_pov_left_cam_rgb"].numpy())
    assert env.unwrapped.dexbench_policy_settings["denoising_timesteps"] == steps


@pytest.mark.parametrize("include_right_ego", [False, True], ids=["three-cameras", "four-cameras"])
def test_rldx_wrist_cameras_keep_the_served_temporal_window(include_right_ego):
    from dexbench_arena.embodiments.rby1_wuji2.camera_contract import CAMERA_OBSERVATIONS

    client = _FakeRldxClient(video_deltas=_TEST_DELTAS)
    configs = client.get_modality_config()
    camera_keys = [key for key in CAMERA_OBSERVATIONS if include_right_ego or key != "camera_ego_right"]
    configs["video"].modality_keys = camera_keys
    client.get_modality_config = lambda: configs
    policy = _make_policy(client, action_chunk_length=8)
    articulation = _articulation_joint_names()
    env = _fake_env(1, articulation)
    for step in range(9):
        obs = _fake_observation(1, articulation, frame_label=step)
        if not include_right_ego:
            del obs["camera_obs"]["robot_pov_right_cam_rgb"]
        for side, offset in (("left", 31), ("right", 79)):
            obs["camera_obs"][f"wrist_{side}_cam_rgb"] = torch.full((1, 360, 640, 3), step + offset, dtype=torch.uint8)
        policy.get_action(env, obs)
    request, _ = client.calls[1]
    assert list(request["video"]) == camera_keys
    assert not policy.codec._warned_monocular
    assert _window_labels(request, "camera_wrist_left", 0) == [35, 37, 39]
    assert _window_labels(request, "camera_wrist_right", 0) == [83, 85, 87]
    assert policy.codec.stamp()["camera_mapping"] == policy.codec.provenance()["camera_mapping"]
    policy.reset()
    del obs["camera_obs"]["wrist_right_cam_rgb"]
    with pytest.raises(ValueError, match="wrist_right_cam_rgb"):
        policy.get_action(env, obs)
    policy.close()
