# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""The GR00T joint-policy client and its config writer, the env server's frame resizing and one openpi client
case: a client binds the Run's joint names, replays and caches action chunks across partial resets under real,
unmodified Arena schedulers, sends the frames the env server would, reports a server that does not answer, and
refuses a server, config or chunk that breaks the contract."""

import importlib.util
import json
import logging
import socket
import time
from copy import deepcopy
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
import yaml
import zmq
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig
from gr00t.policy.server_client import PolicyClient

from dexbench_arena.embodiments.rby1_wuji2.camera_contract import CAMERA_OBSERVATIONS, EGO_CAMERAS, WRIST_CAMERAS
from dexbench_arena.embodiments.rby1_wuji2.constants import (
    RBY1_WUJI2_JOINT_ACTION_NAMES,
    RBY1_WUJI2_WHEEL_JOINT_NAMES,
)
from dexbench_arena.evaluation.env_server import (
    DexBenchEnvServer,
    DexBenchEnvServerConfig,
)
from dexbench_arena_gr00t.policy.gr00t_joint_policy import (
    Gr00tJointPolicy,
    Gr00tJointPolicyCfg,
    served_relative_groups,
)
from dexbench_arena.policy.image_preprocessing import image_preprocessing_record
from dexbench_arena.policy.remote_joint_policy import server_reachable
from dexbench_arena_gr00t.policy.wuji2_gr00t_config import write_config
from dexbench_arena_gr00t.policy.wuji2_gr00t_contract import ACTION_HORIZON
from dexbench_arena.embodiments.rby1_wuji2.joint_groups import WUJI2_JOINT_GROUPS
from dexbench_arena_openpi.policy.rby1_wuji_joint_policy import (
    Pi0Rby1WujiJointPolicyCfg,
    Pi0Rby1WujiJointRemotePolicy,
)


def environment(num_envs=2):
    names = list(RBY1_WUJI2_JOINT_ACTION_NAMES)
    robot_names = names[::-1] + list(RBY1_WUJI2_WHEEL_JOINT_NAMES)
    action_names = names[7:] + names[:7]
    live = SimpleNamespace(
        num_envs=num_envs,
        scene={"robot": SimpleNamespace(joint_names=robot_names)},
        action_manager=SimpleNamespace(
            total_action_dim=62,
            get_term=lambda _: SimpleNamespace(_joint_names=action_names),
        ),
    )
    obs = {
        "policy": {
            "robot_joint_pos": torch.arange(num_envs * 66).reshape(num_envs, 66).float()
        },
        "camera_obs": {
            f"robot_pov_{side}_cam_rgb": torch.ones(
                num_envs, 180, 320, 3, dtype=torch.uint8
            )
            for side in ("left", "right")
        },
    }
    expected = obs["policy"]["robot_joint_pos"][
        :, [robot_names.index(n) for n in action_names]
    ]
    return SimpleNamespace(unwrapped=live), obs, expected


class Echo:
    def __init__(self, **kwargs):
        self.calls = self.resets = self.closed = 0
        self.timeout_ms = 15000
        self.socket = SimpleNamespace(
            close=lambda **kw: setattr(self, "closed", self.closed + 1),
            setsockopt=lambda option, value: None,
        )
        self.context = SimpleNamespace(term=lambda: None)
        self._ws = SimpleNamespace(
            close=lambda: setattr(self, "closed", self.closed + 1)
        )
        self.bad = False
        self.modalities = None

    def ping(self):
        return True

    def reset(self):
        self.resets += 1

    def get_modality_config(self):
        return self.modalities

    def get_server_metadata(self):
        return {}

    def infer(self, request):
        self.calls += 1
        actions = np.repeat(request["state"][None], 8, axis=0)
        if self.bad:
            actions[:] = np.nan
        return {"actions": actions}

    def get_action(self, request):
        self.calls += 1
        self.last_request = request
        actions = {
            g: np.repeat(v, ACTION_HORIZON, axis=1)
            for g, v in request["state"].items()
        }
        if self.bad:
            next(iter(actions.values()))[:] = np.nan
        return actions, {}


@pytest.mark.parametrize(
    "policy_cls,argv,cfg_cls",
    [
        (Pi0Rby1WujiJointRemotePolicy, ["--policy_device", "cpu"], Pi0Rby1WujiJointPolicyCfg),
        (
            Gr00tJointPolicy,
            ["--policy_config_yaml_path", "contract.yaml"],
            Gr00tJointPolicyCfg,
        ),
    ],
)
def test_upstream_typed_cli(policy_cls, argv, cfg_cls):
    import argparse
    from isaaclab_arena.evaluation.policy_runner_cli import (
        add_policy_cli_args,
        policy_cfg_from_cli,
    )

    parser = add_policy_cli_args(argparse.ArgumentParser(), policy_cls)
    cfg = policy_cfg_from_cli(policy_cls, parser.parse_args(argv))
    assert isinstance(cfg, cfg_cls)


CAMERA_SETS = {
    "ego": tuple(EGO_CAMERAS),
    "ego-left": ("camera_ego_left",),
    "ego-left-wrists": ("camera_ego_left", *WRIST_CAMERAS),
    "four-cameras": tuple(CAMERA_OBSERVATIONS),
}


@pytest.mark.parametrize("camera_keys", CAMERA_SETS.values(), ids=CAMERA_SETS)
def test_gr00t_config_uses_local_contract_and_rejects_overwrite(tmp_path, monkeypatch, camera_keys):
    from gr00t.configs.data import embodiment_configs
    from dexbench_arena_gr00t.policy.gr00t_joint_policy import JointContract
    from isaaclab_arena_gr00t.utils.io_utils import create_config_from_yaml

    path = write_config(tmp_path, camera_keys=camera_keys)
    config = create_config_from_yaml(path, JointContract)
    assert not hasattr(config, "task_mode_name")
    assert config.embodiment_tag == "NEW_EMBODIMENT"
    assert config.action_horizon == config.action_chunk_length == 40
    assert tuple(config.target_image_size) == (180, 320, 3)
    monkeypatch.setattr(embodiment_configs, "MODALITY_CONFIGS", {})
    spec = importlib.util.spec_from_file_location("contract_modality", config.modality_config_path)
    spec.loader.exec_module(importlib.util.module_from_spec(spec))
    assert embodiment_configs.MODALITY_CONFIGS["new_embodiment"]["video"].modality_keys == list(camera_keys)
    assert config.pov_cam_name_sim == [CAMERA_OBSERVATIONS[key] for key in camera_keys]
    assert write_config(tmp_path, camera_keys=camera_keys) == path
    from pathlib import Path

    from dexbench_arena_gr00t.policy.wuji2_gr00t_config import LEGACY_MODALITY_DIR

    current = path.read_text()
    legacy = str(LEGACY_MODALITY_DIR / Path(config.modality_config_path).name)
    assert Path(legacy).is_file()
    path.write_text(current.replace(str(config.modality_config_path), legacy))
    assert write_config(tmp_path, camera_keys=camera_keys) == path
    assert path.read_text() == current
    assert current.count("action_horizon: 40\n") == 1
    other_checkout = str(Path("/elsewhere/dexbench_arena/policy") / Path(legacy).name)
    for refused in (current.replace(str(config.modality_config_path), legacy).replace("action_horizon: 40\n", "action_horizon: 16\n"),
                    current.replace(str(config.modality_config_path), other_checkout)):
        path.write_text(refused)
        with pytest.raises(FileExistsError, match="differs"):
            write_config(tmp_path, camera_keys=camera_keys)
        assert path.read_text() == refused
    path.write_text(current)
    original = {file.name: file.read_bytes() for file in tmp_path.iterdir()}
    other_cameras = tuple(CAMERA_OBSERVATIONS) if camera_keys == tuple(EGO_CAMERAS) else tuple(EGO_CAMERAS)
    with pytest.raises(FileExistsError, match="differs"):
        write_config(tmp_path, camera_keys=other_cameras)
    assert {file.name: file.read_bytes() for file in tmp_path.iterdir()} == original
    (tmp_path / "policy_joints.yaml").write_text("edited by hand\n")
    with pytest.raises(FileExistsError, match="differs"):
        write_config(tmp_path, camera_keys=camera_keys)


def test_old_dotted_path_names_the_same_policy():
    module_path, class_name = "dexbench_arena.policy.gr00t_joint_policy.Gr00tJointPolicy".rsplit(".", 1)
    module = importlib.import_module(module_path)
    assert getattr(module, class_name) is Gr00tJointPolicy
    assert module.Gr00tJointPolicyCfg is Gr00tJointPolicyCfg


@pytest.fixture
def gr00t_factory(monkeypatch, tmp_path, request):
    import dexbench_arena_gr00t.policy.gr00t_joint_policy as module

    echo = Echo()
    monkeypatch.setattr(module, "PolicyClient", lambda **kw: echo)
    modalities = {
        k: SimpleNamespace(
            modality_keys=list(WUJI2_JOINT_GROUPS),
            delta_indices=list(range(ACTION_HORIZON)),
        )
        for k in ("state", "action")
    }
    modalities["action"].action_configs = [SimpleNamespace(rep="ABSOLUTE") for _ in WUJI2_JOINT_GROUPS]
    camera_keys = getattr(request, "param", tuple(EGO_CAMERAS))
    modalities.update(
        video=SimpleNamespace(modality_keys=list(camera_keys), delta_indices=[0]),
        language=SimpleNamespace(modality_keys=["annotation.human.task_description"]),
    )
    echo.modalities = deepcopy(modalities)
    monkeypatch.setattr(
        module, "load_gr00t_modality_config_from_file", lambda *args: modalities
    )
    path = write_config(tmp_path, camera_keys=camera_keys)
    return (
        lambda scheduler: Gr00tJointPolicy(
            Gr00tJointPolicyCfg(str(path), policy_device="cpu", scheduler=scheduler)
        ),
        echo,
    )


@pytest.mark.parametrize("scheduler", ["chunk", "synced_batch"])
def test_gr00t_reordered_actions_cache_partial_reset_and_close(
    gr00t_factory, scheduler
):
    make, echo = gr00t_factory
    policy = make(scheduler)
    env, obs, expected = environment()
    policy.set_task_description("Retrieve the drink")
    policy.reset()
    for _ in range(3):
        torch.testing.assert_close(policy.get_action(env, obs), expected)
    assert echo.calls == 1
    policy.reset(torch.tensor([0]))
    for _ in range(ACTION_HORIZON + 1):
        torch.testing.assert_close(policy.get_action(env, obs), expected)
    assert echo.calls >= 2 and echo.resets == 2
    policy.reset()
    echo.bad = True
    with pytest.raises(ValueError, match="Non-finite"):
        policy.get_action(env, obs)
    policy.close()
    policy.close()
    assert echo.closed == 1


def test_gr00t_run_sets_the_execution_horizon_over_the_contract(gr00t_factory, tmp_path):
    path = write_config(tmp_path / "horizon")
    policy = Gr00tJointPolicy(Gr00tJointPolicyCfg(str(path), policy_device="cpu", action_chunk_length=16))
    assert policy.contract.action_chunk_length == 16
    with pytest.raises(ValueError, match="action_chunk_length"):
        Gr00tJointPolicy(Gr00tJointPolicyCfg(str(path), policy_device="cpu", action_chunk_length=ACTION_HORIZON + 1))


def test_gr00t_rejects_additional_action_terms(gr00t_factory):
    make, _ = gr00t_factory
    policy = make("chunk")
    env, obs, _ = environment()
    env.unwrapped.action_manager.total_action_dim = 63
    policy.set_task_description("Retrieve")
    with pytest.raises(ValueError, match="joint-only"):
        policy.get_action(env, obs)
    policy.close()


@pytest.mark.parametrize("failure", ["missing_group", "wrong_horizon"])
def test_gr00t_rejects_malformed_chunk(gr00t_factory, failure):
    make, echo = gr00t_factory
    policy = make("chunk")
    env, obs, _ = environment()
    policy.set_task_description("Retrieve")
    original = echo.get_action

    def malformed(request):
        result, options = original(request)
        group = next(iter(result))
        if failure == "missing_group":
            del result[group]
        else:
            result[group] = result[group][:, :1]
        return result, options

    echo.get_action = malformed
    with pytest.raises(ValueError, match="groups disagree|Invalid GR00T action shape"):
        policy.get_action(env, obs)
    policy.close()


def test_gr00t_failed_connection_releases_client(gr00t_factory):
    make, echo = gr00t_factory
    echo.ping = lambda: False
    with pytest.raises(ConnectionError):
        make("chunk")
    assert echo.closed == 1


@pytest.mark.parametrize("mismatch", ["action_horizon", "state_keys", "action_keys"])
def test_gr00t_rejects_server_modality_mismatch(gr00t_factory, mismatch):
    make, echo = gr00t_factory
    if mismatch == "action_horizon":
        echo.modalities["action"].delta_indices = list(range(50))
        message = "server action delta_indices disagree"
    else:
        key = mismatch.removesuffix("_keys")
        echo.modalities[key].modality_keys = ["other_group"]
        message = f"server {key} modality keys disagree"
    with pytest.raises(ValueError, match=message):
        make("chunk")
    assert echo.closed == 1


def test_gr00t_rejects_server_without_modality_contract(gr00t_factory):
    make, echo = gr00t_factory
    echo.modalities = {}
    with pytest.raises(ValueError, match="did not provide state/action/video modalities"):
        make("chunk")
    assert echo.closed == 1


GROUPS = list(WUJI2_JOINT_GROUPS)
ARMS = ("left_arm_joints", "right_arm_joints")


def _served(relative=(), stated=True):
    configs = [ActionConfig(rep=ActionRepresentation.RELATIVE if group in relative else ActionRepresentation.ABSOLUTE,
                            type=ActionType.NON_EEF, format=ActionFormat.DEFAULT) for group in GROUPS]
    return ModalityConfig(delta_indices=list(range(40)), modality_keys=GROUPS, action_configs=configs if stated else None)


def test_the_client_refuses_a_server_of_the_other_action_representation():
    assert served_relative_groups(_served(), ()) == []
    assert served_relative_groups(_served(ARMS), ARMS) == sorted(ARMS)
    for served, expected in ((_served(), ARMS), (_served(ARMS), ())):
        with pytest.raises(ValueError, match="this Run expects"):
            served_relative_groups(served, expected)
    with pytest.raises(ValueError, match="states no action_configs"):
        served_relative_groups(_served(stated=False), ())
    short = _served()
    short.action_configs = short.action_configs[:-1]
    with pytest.raises(ValueError, match="action_configs for"):
        served_relative_groups(short, ())


def test_a_server_that_does_not_answer_is_reported_within_the_timeout():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    client = PolicyClient(host="127.0.0.1", port=port)
    start = time.monotonic()
    assert not server_reachable(client, 300)
    assert time.monotonic() - start < 5
    assert client.socket.getsockopt(zmq.RCVTIMEO) == -1


def test_gr00t_rejects_yaml_modality_horizon_mismatch(gr00t_factory, tmp_path):
    make, _ = gr00t_factory
    path = tmp_path / "policy.yaml"
    config = yaml.safe_load(path.read_text())
    config["action_horizon"] = 8
    config["action_chunk_length"] = 8
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="disagrees with modality action horizon"):
        make("chunk")


def test_gr00t_is_not_bound_to_wuji_names_or_dof(gr00t_factory, monkeypatch):
    import dexbench_arena_gr00t.policy.gr00t_joint_policy as module

    make, _ = gr00t_factory
    groups = {"grip": ["axis_b", "left_thumb_custom"], "body": ["torso_a"]}
    robot_names = ["torso_a", "axis_b", "left_thumb_custom"]
    action_names = ["left_thumb_custom", "torso_a", "axis_b"]
    config = {name: i for i, name in enumerate(robot_names)}
    monkeypatch.setattr(
        module, "load_gr00t_joint_configs", lambda _: (groups, config, config.copy())
    )
    modalities = {
        k: SimpleNamespace(
            modality_keys=list(groups), delta_indices=list(range(ACTION_HORIZON))
        )
        for k in ("state", "action")
    }
    modalities["action"].action_configs = [SimpleNamespace(rep="ABSOLUTE") for _ in groups]
    modalities.update(
        video=SimpleNamespace(modality_keys=["left", "right"], delta_indices=[0]),
        language=SimpleNamespace(modality_keys=["task"]),
    )
    monkeypatch.setattr(
        module, "load_gr00t_modality_config_from_file", lambda *args: modalities
    )
    gr00t_factory[1].modalities = deepcopy(modalities)
    env, obs, _ = environment(1)
    env.unwrapped.scene["robot"].joint_names = robot_names
    env.unwrapped.action_manager = SimpleNamespace(
        total_action_dim=3,
        get_term=lambda _: SimpleNamespace(_joint_names=action_names),
    )
    obs["policy"]["robot_joint_pos"] = torch.tensor([[0.1, 0.2, 0.3]])
    policy = make("chunk")
    policy.set_task_description("Move")
    torch.testing.assert_close(
        policy.get_action(env, obs), torch.tensor([[0.3, 0.1, 0.2]])
    )
    policy.close()


CAMERAS = ("robot_pov_left_cam_rgb", "robot_pov_right_cam_rgb")


def textured_frames(height, width, num_envs=2):
    """Per-pixel noise with alpha, so padding, skipped source pixels or a kept alpha show."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (num_envs, height, width, 4), dtype=np.uint8)


def area_downscaled(frames, height=180, width=320):
    """The relayout's ``ffmpeg scale=320:180:flags=area``, which INTER_AREA matches."""
    return np.stack(
        [cv2.resize(f[..., :3], (width, height), interpolation=cv2.INTER_AREA) for f in frames]
    )


def test_gr00t_area_downscales_off_size_frames_without_padding(gr00t_factory, capsys):
    make, echo = gr00t_factory
    policy = make("chunk")
    env, obs, _ = environment()
    frames = textured_frames(360, 640)
    obs["camera_obs"] = {name: torch.from_numpy(frames) for name in CAMERAS}
    policy.set_task_description("Retrieve")
    policy.get_action(env, obs)
    record = capsys.readouterr().out.split("image_preprocessing=", 1)[1]
    assert json.loads(record)[CAMERAS[0]] == {
        "name": "rgb_area_v1",
        "source_hw": [360, 640],
        "target_hw": [180, 320],
        "resize": "cv2.INTER_AREA",
        "channels": "RGB",
    }
    for key in ("camera_ego_left", "camera_ego_right"):
        np.testing.assert_array_equal(
            echo.last_request["video"][key][:, 0], area_downscaled(frames)
        )
    obs["camera_obs"] = {name: torch.from_numpy(textured_frames(720, 1280)) for name in CAMERAS}
    policy.reset()
    policy.get_action(env, obs)
    record = capsys.readouterr().out.split("image_preprocessing=", 1)[1]
    assert json.loads(record)[CAMERAS[0]]["source_hw"] == [720, 1280]
    obs["camera_obs"] = {name: torch.from_numpy(textured_frames(480, 640)) for name in CAMERAS}
    policy.reset()
    with pytest.raises(ValueError, match="different aspect ratio"):
        policy.get_action(env, obs)
    policy.close()


def wire_server():
    """An env server asked for 180x320 wire frames; packaging needs no sim or socket."""
    server = DexBenchEnvServer.__new__(DexBenchEnvServer)
    server.config = DexBenchEnvServerConfig(
        wire_image_resize_height=180, wire_image_resize_width=320
    )
    server.logger = logging.getLogger("test_env_server")
    server._wire_image_record = None
    return server


def env_server_wire_frames(frames):
    """What the env server sends a client when asked for 180x320 wire frames, and its record."""
    server = wire_server()
    observation = {"camera_obs": {name: torch.from_numpy(frames) for name in CAMERAS}}
    wire = server._package_observation(observation)
    frames = {name: wire["camera_obs"][name].numpy() for name in CAMERAS}
    return frames, server._wire_image_record


def test_env_server_wire_resize_leaves_non_rgb_camera_terms_alone():
    normals = torch.rand(2, 360, 640, 3)
    server = wire_server()
    wire = server._package_observation(
        {"camera_obs": {CAMERAS[0]: torch.from_numpy(textured_frames(360, 640)), "normals": normals}}
    )
    assert wire["camera_obs"][CAMERAS[0]].shape == (2, 180, 320, 3)
    torch.testing.assert_close(wire["camera_obs"]["normals"], normals)
    assert set(server._wire_image_record) == {CAMERAS[0]}


@pytest.mark.parametrize("height,width", [(360, 640), (720, 1280)])
def test_gr00t_native_and_env_server_paths_send_the_same_frames(
    gr00t_factory, height, width
):
    """One checkpoint contract, one set of pixels, whichever path serves it."""
    make, echo = gr00t_factory
    policy = make("chunk")
    env, obs, _ = environment()
    frames = textured_frames(height, width)
    obs["camera_obs"] = {name: torch.from_numpy(frames) for name in CAMERAS}
    policy.set_task_description("Retrieve")
    policy.get_action(env, obs)
    native = echo.last_request["video"]["camera_ego_left"][:, 0]
    policy.close()

    wire, record = env_server_wire_frames(frames)
    assert record == {name: image_preprocessing_record((height, width), (180, 320)) for name in CAMERAS}
    np.testing.assert_array_equal(wire[CAMERAS[0]], native)
    np.testing.assert_array_equal(native, area_downscaled(frames))
    assert native.reshape(native.shape[0], 180, -1).max(axis=-1).min() > 0, "padding rows"


def test_env_server_paths_refuse_frames_of_another_aspect_ratio():
    frames = textured_frames(480, 640)
    with pytest.raises(ValueError, match="different aspect ratio"):
        env_server_wire_frames(frames)


def test_openpi_native_cache_reset_and_named_actions(monkeypatch):
    import dexbench_arena_openpi.policy.rby1_wuji_joint_policy as module

    echo = Echo()
    monkeypatch.setattr(module, "server_listening", lambda host, port, timeout: None)
    monkeypatch.setattr(module, "WebsocketClientPolicy", lambda **kw: echo)
    policy = Pi0Rby1WujiJointRemotePolicy(Pi0Rby1WujiJointPolicyCfg(policy_device="cpu", action_chunk_length=8))
    env, obs, expected = environment()
    policy.set_task_description("Retrieve")
    for _ in range(3):
        torch.testing.assert_close(policy.get_action(env, obs), expected)
    assert echo.calls == 2
    policy.reset(torch.tensor([0]))
    torch.testing.assert_close(policy.get_action(env, obs), expected)
    assert echo.calls == 3
    policy.reset()
    echo.bad = True
    with pytest.raises(ValueError, match="Non-finite"):
        policy.get_action(env, obs)
    policy.close()
    assert echo.closed == 1


@pytest.mark.parametrize("gr00t_factory", [tuple(CAMERA_OBSERVATIONS)], indirect=True)
def test_gr00t_four_camera_contract_and_missing_wrist(gr00t_factory):
    make, echo = gr00t_factory
    policy = make("chunk")
    env, obs, expected = environment()
    for side, pixel in (("left", 31), ("right", 79)):
        obs["camera_obs"][f"wrist_{side}_cam_rgb"] = torch.full((2, 180, 320, 3), pixel, dtype=torch.uint8)
    policy.set_task_description("Retrieve")
    torch.testing.assert_close(policy.get_action(env, obs), expected)
    for side, pixel in (("left", 31), ("right", 79)):
        assert np.all(echo.last_request["video"][f"camera_wrist_{side}"] == pixel)
    assert policy.codec.stamp()["camera_mapping"] == policy.codec.provenance()["camera_mapping"]
    policy.reset()
    del obs["camera_obs"]["wrist_left_cam_rgb"]
    with pytest.raises(ValueError, match="wrist_left_cam_rgb"):
        policy.get_action(env, obs)
    policy.close()


@pytest.mark.parametrize("gr00t_factory", [("camera_ego_left", *WRIST_CAMERAS)], indirect=True)
def test_gr00t_ego_left_and_wrists_without_ego_right(gr00t_factory):
    make, echo = gr00t_factory
    policy = make("chunk")
    env, obs, expected = environment()
    served = {"camera_ego_left": CAMERAS[0], **WRIST_CAMERAS}
    frames = {name: textured_frames(180, 320) for name in served.values()}
    for name, pixel in (("wrist_left_cam_rgb", 31), ("wrist_right_cam_rgb", 79)):
        frames[name][..., 0] = pixel
    obs["camera_obs"] = {name: torch.from_numpy(f) for name, f in frames.items()}
    policy.set_task_description("Retrieve")
    torch.testing.assert_close(policy.get_action(env, obs), expected)
    assert list(echo.last_request["video"]) == list(served)
    for key, name in served.items():
        np.testing.assert_array_equal(echo.last_request["video"][key][:, 0], frames[name][..., :3])
    policy.close()


def test_gr00t_rejects_temporal_video_contract(gr00t_factory):
    make, echo = gr00t_factory
    echo.modalities["video"].delta_indices = [-1, 0]
    with pytest.raises(ValueError, match="video delta_indices"):
        make("chunk")
