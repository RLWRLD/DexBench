"""Absolute-joint client for the DexBench servers of the world action models FastWAM and OpenWAM."""

from __future__ import annotations

from dataclasses import dataclass

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase

from dexbench_arena.policy.remote_joint_policy import RemoteJointPolicy
from dexbench_arena_cosmos.policy.cosmos3_joint_policy import Cosmos3Codec, Cosmos3HttpTransport, Cosmos3JointPolicyCfg

MODEL_FAMILIES = ("fastwam", "openwam")


@dataclass
class WamJointPolicyCfg(Cosmos3JointPolicyCfg):
    model_family: str = "fastwam"
    """The ``model_family`` the server must state at ``/info``: ``fastwam`` or ``openwam``."""


class WamCodec(Cosmos3Codec):
    """The Cosmos3 codec, for a server that states the model family the Run expects."""

    def check(self, server_info):
        family = server_info.get("model_family")
        if self.config.model_family not in MODEL_FAMILIES:
            raise ValueError(f"model_family must be one of {MODEL_FAMILIES}, got {self.config.model_family!r}")
        if family != self.config.model_family:
            raise ValueError(f"The server at {self.config.remote_url} states model_family {family!r}, "
                             f"this Run expects {self.config.model_family!r}")
        super().check(server_info)


@register_policy
class WamJointPolicy(RemoteJointPolicy, PolicyBase[WamJointPolicyCfg]):
    name = "dexbench_wam_joint"

    def __init__(self, config: WamJointPolicyCfg):
        super().__init__(config, Cosmos3HttpTransport(config), WamCodec(config), config.action_chunk_length)
