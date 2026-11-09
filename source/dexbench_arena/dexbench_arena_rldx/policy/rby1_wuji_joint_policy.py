# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Remote RLDX-1 policy for DexBench RBY1-M Wuji-hand joint-space control, on :class:`RemoteJointPolicy`."""

from __future__ import annotations

import argparse
import uuid
from dataclasses import dataclass, field
from typing import Any

import torch

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase

from dexbench_arena.policy.policy_rate import PolicyRateCfg
from dexbench_arena.policy.remote_joint_policy import SCHEDULERS, RemoteJointPolicy, Transport, server_reachable
from dexbench_arena_rldx.policy.rby1_wuji_joint_adapter import RldxRby1WujiJointCodec


def _float_list(value: str) -> list[float]:
    """One comma-separated argument as a list of floats: ``0,0.25,0.5``."""
    try:
        return [float(item) for item in value.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected comma-separated numbers, got {value!r}") from None


@dataclass
class RldxRemotePolicyCfg(PolicyRateCfg):
    """Connection + runtime config for ``RldxRby1WujiJointRemotePolicy``."""

    policy_device: str = "cuda"
    hand_variant: str = "wuji2"
    remote_host: str = "localhost"
    remote_port: int = 5555
    remote_api_token: str | None = None
    action_chunk_length: int = 16
    """Steps executed from each chunk before re-planning (the execution horizon); at most the server's
    action horizon."""
    scheduler: str = "synced_batch"
    """``synced_batch`` or ``chunk``. Plain ``str`` fields, checked in ``__post_init__``: Arena composes typed
    experiment YAMLs through OmegaConf, which refuses a ``Literal`` annotation."""
    target_image_size: list[int] = field(default_factory=lambda: [360, 640, 3])
    """(height, width, channels) of the frames sent: the size the checkpoint's training frames were stored at."""
    joint_layout: str = "wuji2_v1"
    """``wuji2_v1`` or ``wuji2_rrc_v1``, the layout the checkpoint was trained with."""
    policy_variant: str | None = None
    """The model this Run scores, as ``arena_experiment_result.json`` records it (Arena reads a
    cfg's ``policy_variant`` before the class name). The benchmark sets the registry key
    (``benchmark.models``), so two checkpoints of one policy class are two models."""
    robot_asset_name: str = "robot"
    joint_action_term: str = "joint_pos"
    memory_sessions: bool = True
    """Send each request's memory session options (``session_ids``, ``reset_memory``). False for a stateless
    checkpoint served by RLDX 7a518b5a or later (``rldx.eval.run_rldx_server``), whose policy refuses any option it
    does not know; the first request would fail."""
    denoising_timesteps: list[float] | None = None
    """The timesteps the server was started with (``run_rldx_server.py --denoising-timesteps``), or None when the
    Run does not say. The server does not state them, so the Run declares them, the episode rows carry them, and
    the scorer holds them to the model's entry; whoever starts the server passes the same list."""

    def __post_init__(self):
        if self.joint_layout not in ("wuji2_v1", "wuji2_rrc_v1"):
            raise ValueError(f"Unknown joint layout: {self.joint_layout}")
        if self.hand_variant != "wuji2":
            raise ValueError(f"Unknown hand variant: {self.hand_variant}")
        if self.scheduler not in SCHEDULERS:
            raise ValueError(f"Unknown scheduler: {self.scheduler!r}")
        if self.action_chunk_length < 1:
            raise ValueError("action_chunk_length must be positive")
        if (
            not isinstance(self.target_image_size, (list, tuple))
            or len(self.target_image_size) != 3
            or any(type(size) is not int or size <= 0 for size in self.target_image_size)
            or self.target_image_size[2] != 3
        ):
            raise ValueError("target_image_size must be [positive height, positive width, 3] for RGB")
        steps = self.denoising_timesteps
        if steps is not None and (
            not isinstance(steps, (list, tuple))
            or not steps
            or any(isinstance(t, bool) or not isinstance(t, (int, float)) or not 0 <= t < 1 for t in steps)
            or any(later <= earlier for earlier, later in zip(steps, steps[1:]))
        ):
            raise ValueError("denoising_timesteps must be increasing values in [0, 1)")


class RldxZmqTransport(Transport):
    """RLDX's native ZMQ ``PolicyClient``, with a server memory session per env."""

    def __init__(self, config: RldxRemotePolicyCfg, client: Any | None = None):
        if client is None:
            from rldx.policy.server_client import PolicyClient

            client = PolicyClient(host=config.remote_host, port=config.remote_port,
                                  api_token=config.remote_api_token, strict=False)
            if not server_reachable(client, client.timeout_ms):
                raise ConnectionError(
                    f"Cannot reach the RLDX server at {config.remote_host}:{config.remote_port} within "
                    f"{client.timeout_ms / 1000:g} s: start it before the run"
                )
        self.client = client
        self._memory_sessions = config.memory_sessions
        self._run_id = uuid.uuid4().hex[:8]
        self._started: set[int] = set()

    def info(self):
        return self.client.get_modality_config()

    def infer(self, request, env_ids):
        options = None
        if self._memory_sessions:
            options = {"session_ids": [f"dexbench-{self._run_id}-env{env_id}" for env_id in env_ids],
                       "reset_memory": [env_id not in self._started for env_id in env_ids]}
        action, _info = self.client.get_action(request, options)
        self._started.update(env_ids)
        return action

    def reset(self, env_ids=None):
        if env_ids is None:
            self._started.clear()
        else:
            self._started.difference_update(torch.as_tensor(env_ids).reshape(-1).tolist())

    def close(self):
        """Release the ZMQ connection to the RLDX server (the server outlives this client)."""
        socket = getattr(self.client, "socket", None)
        context = getattr(self.client, "context", None)
        try:
            if socket is not None:
                socket.close(linger=0)
        finally:
            if context is not None:
                context.term()


@register_policy
class RldxRby1WujiJointRemotePolicy(RemoteJointPolicy, PolicyBase[RldxRemotePolicyCfg]):
    """RLDX-1 closed-loop policy for the DexBench RBY1-Wuji joint-control twin."""

    name = "rldx_rby1_wuji_joint_remote"

    @classmethod
    def adjust_policy_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        """Keep the list arguments from consuming the environment subcommand."""
        image_size = parser._option_string_actions["--target_image_size"]
        image_size.nargs = 3
        image_size.metavar = ("HEIGHT", "WIDTH", "CHANNELS")
        timesteps = parser._option_string_actions["--denoising_timesteps"]
        timesteps.nargs = None
        timesteps.type = _float_list
        timesteps.metavar = "T0,T1,..."

    def __init__(self, config: RldxRemotePolicyCfg, client: Any | None = None):
        codec = RldxRby1WujiJointCodec(config.joint_layout, tuple(config.target_image_size[:2]),
                                       config.denoising_timesteps)
        super().__init__(config, RldxZmqTransport(config, client), codec, config.action_chunk_length)
