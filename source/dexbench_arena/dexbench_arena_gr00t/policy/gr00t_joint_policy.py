"""Absolute-joint GR00T on :class:`RemoteJointPolicy`, with Arena's GR00T observation helpers."""

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from gr00t.policy.server_client import PolicyClient

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase
from isaaclab_arena_gr00t.policy.gr00t_core import (
    build_gr00t_policy_observations,
    extract_obs_numpy_from_torch,
    load_gr00t_joint_configs,
)
from isaaclab_arena_gr00t.utils.io_utils import (
    create_config_from_yaml,
    load_gr00t_modality_config_from_file,
)

from dexbench_arena.policy.image_preprocessing import (
    image_preprocessing_record,
    resize_rgb_frames,
)
from dexbench_arena.policy.policy_rate import PolicyRateCfg
from dexbench_arena.policy.remote_joint_policy import Codec, RemoteJointPolicy, Transport, server_reachable


@dataclass
class JointContract:
    policy_joints_config_path: Path
    action_joints_config_path: Path
    state_joints_config_path: Path
    modality_config_path: Path
    pov_cam_name_sim: list[str]
    target_image_size: tuple[int, int, int]
    action_horizon: int
    action_chunk_length: int
    embodiment_tag: str = "NEW_EMBODIMENT"

    def __post_init__(self):
        if not 0 < self.action_chunk_length <= self.action_horizon:
            raise ValueError("Require 0 < action_chunk_length <= action_horizon")
        if (
            not self.pov_cam_name_sim
            or len(self.target_image_size) != 3
            or self.target_image_size[2] != 3
        ):
            raise ValueError("Require camera names and an RGB target image size")


@dataclass
class Gr00tJointPolicyCfg(PolicyRateCfg):
    policy_config_yaml_path: str
    policy_device: str = "cuda"
    remote_host: str = "localhost"
    remote_port: int = 5555
    remote_api_token: str | None = None
    action_chunk_length: int | None = None
    """Steps executed from each chunk before re-planning (the execution horizon); None keeps the
    contract YAML's. Lets an experiment Run state its horizon without a contract file per horizon."""
    scheduler: str = "chunk"
    """``chunk`` or ``synced_batch``. A plain ``str`` (checked in the constructor): Arena composes
    typed experiment YAMLs through OmegaConf, which refuses a ``Literal`` annotation."""
    robot_asset_name: str = "robot"
    joint_action_term: str = "joint_pos"
    policy_variant: str | None = None
    """The model this Run scores, as ``arena_experiment_result.json`` records it (Arena reads a
    cfg's ``policy_variant`` before the class name). The benchmark sets the registry key
    (``benchmark.models``), so two checkpoints of one policy class are two models."""
    relative_action_groups: tuple = ()
    """The action groups the checkpoint was trained on relative to the state (``left_arm_joints``,
    ``right_arm_joints`` for arm-relative); every other group absolute. Checked against each action
    key's ``rep`` in the server's modality config, so an absolute model never runs a relative checkpoint
    unnoticed, nor the other way round. The server answers absolute targets either way."""


def served_relative_groups(served_action, relative_action_groups) -> list[str]:
    """The action groups ``served_action`` (the server's action ``ModalityConfig``) serves relative, sorted;
    raises ValueError unless every group is absolute or relative as ``relative_action_groups`` expects."""
    configs = served_action.action_configs
    if configs is None:
        raise ValueError("GR00T server states no action_configs, so its action representation is unknown")
    keys = list(served_action.modality_keys)
    if len(configs) != len(keys):
        raise ValueError(f"GR00T server states {len(configs)} action_configs for {len(keys)} action keys {keys}")
    served = {key: getattr(config.rep, "name", config.rep) for key, config in zip(keys, configs)}
    expected = {key: "RELATIVE" if key in relative_action_groups else "ABSOLUTE" for key in served}
    if served != expected or not set(relative_action_groups) <= set(served):
        raise ValueError(f"GR00T server serves action representations {served}, this Run expects {expected} "
                         f"(relative_action_groups={list(relative_action_groups)})")
    return sorted(key for key, rep in served.items() if rep == "RELATIVE")


class Gr00tTransport(Transport):
    """gr00t's ZMQ ``PolicyClient``; the server keeps the episode's state until :meth:`reset`."""

    def __init__(self, config: Gr00tJointPolicyCfg):
        self.address = f"{config.remote_host}:{config.remote_port}"
        self.client = PolicyClient(
            host=config.remote_host,
            port=config.remote_port,
            api_token=config.remote_api_token,
            strict=False,
        )

    def info(self):
        if not server_reachable(self.client, self.client.timeout_ms):
            raise ConnectionError(
                f"Cannot reach the GR00T server at {self.address} within {self.client.timeout_ms / 1000:g} s: "
                "start it before the run"
            )
        return self.client.get_modality_config()

    def infer(self, request, env_ids):
        response, _ = self.client.get_action(request)
        return response

    def reset(self, env_ids=None):
        self.client.reset()

    def close(self):
        try:
            self.client.socket.close(linger=0)
        finally:
            self.client.context.term()


class Gr00tCodec(Codec):
    """The contract YAML's joint groups and modalities, checked against the server's; requests built by
    Arena's ``build_gr00t_policy_observations`` from frames resized as the training data was."""

    def __init__(self, config: Gr00tJointPolicyCfg):
        self.config = config
        self.contract = create_config_from_yaml(
            config.policy_config_yaml_path, JointContract
        )
        if config.action_chunk_length is not None:
            self.contract = replace(self.contract, action_chunk_length=config.action_chunk_length)
        self.groups, action_config, self.state_config = load_gr00t_joint_configs(
            self.contract
        )
        self.joint_names = [name for names in self.groups.values() for name in names]
        if (
            not self.joint_names
            or len(set(self.joint_names)) != len(self.joint_names)
            or set(self.joint_names) != set(action_config)
        ):
            raise ValueError("Policy groups must name every action joint exactly once")
        if not set(self.joint_names) <= set(self.state_config):
            raise ValueError("All action joints require observed state")
        self.modalities = load_gr00t_modality_config_from_file(
            self.contract.modality_config_path,
            self.contract.embodiment_tag,
        )
        for key in ("state", "action"):
            if set(self.modalities[key].modality_keys) != set(self.groups):
                raise ValueError(f"{key} modalities disagree with joint groups")
        modality_horizon = len(self.modalities["action"].delta_indices)
        if modality_horizon != self.contract.action_horizon:
            raise ValueError(
                f"YAML action_horizon={self.contract.action_horizon} disagrees with "
                f"modality action horizon={modality_horizon}"
            )
        self.action_horizon = self.contract.action_horizon
        self._image_record = None

    def check(self, server_modalities):
        self._provenance = {
            "server_info": json.loads(json.dumps(server_modalities, default=str)),
            "contract_sha256": hashlib.sha256(b"".join(
                p.name.encode() + p.read_bytes()
                for p in sorted(Path(self.config.policy_config_yaml_path).parent.iterdir()) if p.is_file()
            )).hexdigest(),
        }
        required_modalities = {"state", "action", "video"}
        if (
            not isinstance(server_modalities, dict)
            or not required_modalities <= server_modalities.keys()
        ):
            raise ValueError("GR00T server did not provide state/action/video modalities")
        for key in ("state", "action", "video"):
            local_keys = list(self.modalities[key].modality_keys)
            server_keys = list(server_modalities[key].modality_keys)
            if sorted(server_keys) != sorted(local_keys):
                raise ValueError(
                    f"GR00T server {key} modality keys disagree with local config: "
                    f"{server_keys} != {local_keys}"
                )
        local_indices = list(self.modalities["action"].delta_indices)
        server_indices = list(server_modalities["action"].delta_indices)
        if server_indices != local_indices:
            raise ValueError(
                "GR00T server action delta_indices disagree with local config: "
                f"{server_indices} != {local_indices}"
            )
        local_video = self.modalities["video"]
        if list(local_video.delta_indices) != [0] or list(server_modalities["video"].delta_indices) != [0]:
            raise ValueError("GR00T video delta_indices must be [0]; temporal video is not supported")
        keys, sources = list(local_video.modality_keys), self.contract.pov_cam_name_sim
        if len(keys) != len(sources) or len(set(keys)) != len(keys) or len(set(sources)) != len(sources):
            raise ValueError("GR00T camera mapping must pair unique video keys and simulator observations")
        self.cameras = dict(zip(keys, sources))
        self._provenance["camera_mapping"] = self.cameras
        self.relative_action_groups = served_relative_groups(
            server_modalities["action"], self.config.relative_action_groups
        )

    def bind(self, robot_joint_names):
        self.state_config = {n: robot_joint_names.index(n) for n in self.state_config}

    def encode(self, observation, env_ids, task_description):
        missing = set(self.cameras.values()) - observation["camera_obs"].keys()
        if missing:
            raise ValueError(f"Missing required camera observations: {sorted(missing)}")
        rgb, joints = extract_obs_numpy_from_torch(
            observation, self.contract.pov_cam_name_sim
        )
        height, width = self.contract.target_image_size[:2]
        sources = [frames.shape[1:3] for frames in rgb]
        rgb = [resize_rgb_frames(frames, height, width) for frames in rgb]
        record = {
            name: image_preprocessing_record(source, (height, width))
            for name, source in zip(self.contract.pov_cam_name_sim, sources)
        }
        if record != self._image_record:
            self._image_record = record
            print(
                f"[Gr00tJointPolicy] image_preprocessing={json.dumps(record)}", flush=True
            )
        return build_gr00t_policy_observations(
            rgb,
            joints,
            task_description,
            self.contract,
            self.state_config,
            self.groups,
            self.modalities,
        )

    def decode(self, response, num_envs):
        if set(response) != set(self.groups):
            raise ValueError("GR00T response groups disagree with the joint contract")
        parts = []
        for group, names in self.groups.items():
            values = np.asarray(response[group])
            if values.shape != (num_envs, self.action_horizon, len(names)):
                raise ValueError(
                    f"Invalid GR00T action shape for {group}: {values.shape}"
                )
            parts.append(values)
        return np.concatenate(parts, axis=-1)

    def stamp(self):
        return {"relative_action_groups": self.relative_action_groups, "camera_mapping": self.cameras}

    def provenance(self):
        return self._provenance


@register_policy
class Gr00tJointPolicy(RemoteJointPolicy, PolicyBase[Gr00tJointPolicyCfg]):
    name = "dexbench_gr00t_joint"

    def __init__(self, config: Gr00tJointPolicyCfg):
        codec = Gr00tCodec(config)
        self.contract = codec.contract
        super().__init__(config, Gr00tTransport(config), codec, codec.contract.action_chunk_length)
