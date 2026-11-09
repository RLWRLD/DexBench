# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""A joint-space policy served by a remote model: one class for what every such client does, so a model is only
how its requests travel (:class:`Transport`) and what it sees and answers (:class:`Codec`)."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zmq

from isaaclab_arena.policy.action_scheduling import ActionChunkScheduler, SyncedBatchActionScheduler

from dexbench_arena.evaluation.dataset_recording import current_run_output_dir
from dexbench_arena.evaluation.evaluation_signature import stamp_policy_settings
from dexbench_arena.policy.policy_rate import PolicyRateScheduler

SCHEDULERS = {"chunk": ActionChunkScheduler, "synced_batch": SyncedBatchActionScheduler}


def server_reachable(client: Any, timeout_ms: int) -> bool:
    """Whether a ZMQ ``PolicyClient``'s server answers a ping within ``timeout_ms``."""
    client.socket.setsockopt(zmq.LINGER, 0)
    client.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
    try:
        return client.ping()
    finally:
        client.socket.setsockopt(zmq.LINGER, 0)
        client.socket.setsockopt(zmq.RCVTIMEO, -1)


class Transport(ABC):
    """How requests reach a model server and its answers come back."""

    @abstractmethod
    def info(self) -> Any:
        """What the server states about the model it serves (its contract), checked by the codec."""

    @abstractmethod
    def infer(self, request: Any, env_ids: list[int]) -> Any:
        """The server's answer to one request, made for ``env_ids``."""

    def reset(self, env_ids=None) -> None:
        """A new episode for ``env_ids`` (all when None): forget the server's state for them; nothing for a
        stateless server."""

    def close(self) -> None:
        """Release the connection."""


class Codec(ABC):
    """What a model sees and answers: the request built from an observation, the action chunk read from the
    answer, and the checks of the server's contract. Set ``joint_names`` (the model's state and action order)
    and ``action_horizon`` by the end of :meth:`check`."""

    joint_names: list[str]
    action_horizon: int
    requests: str = "all"
    """At a re-plan: ``all``, one request for every env; ``needed``, one for the envs whose chunk ran out;
    ``each``, one per such env."""

    @abstractmethod
    def check(self, info: Any) -> None:
        """Refuse a server whose stated contract is not this Run's model."""

    def bind(self, robot_joint_names: list[str]) -> None:
        """Called once with the live robot's joint order, before the first request."""

    def observe(self, observation: dict) -> None:
        """Called with every policy step's observation, before any request that step."""

    def reset(self, env_ids=None) -> None:
        """A new episode for ``env_ids`` (all when None)."""

    @abstractmethod
    def encode(self, observation: dict, env_ids: list[int], task_description: str) -> Any:
        """The request for ``env_ids``."""

    @abstractmethod
    def decode(self, response: Any, num_envs: int) -> np.ndarray:
        """``(num_envs, action_horizon, len(joint_names))`` absolute joint targets in the model's joint order."""

    def stamp(self) -> dict[str, Any]:
        """The model's own fields of the episode rows' policy stamp (its action representation)."""
        return {}

    def provenance(self) -> dict[str, Any]:
        """What ``policy_provenance.json`` keeps: what the server stated and a hash of the contract."""
        return {}


class RemoteJointPolicy:
    """Mixin for ``(RemoteJointPolicy, PolicyBase[Cfg])``. The cfg (a ``policy_rate.PolicyRateCfg``) names
    ``policy_device``, ``robot_asset_name``, ``joint_action_term``, ``policy_variant`` and ``scheduler``, and
    ``policy_hz`` and ``policy_action_interpolation``."""

    def __init__(self, config, transport: Transport, codec: Codec, action_chunk_length: int) -> None:
        super().__init__(config)
        if config.scheduler not in SCHEDULERS:
            raise ValueError("Unknown action scheduler")
        self.transport, self.codec = transport, codec
        self.task_description = None
        self._bound_env = None
        self._scheduler = None
        try:
            codec.check(transport.info())
            if not 0 < action_chunk_length <= codec.action_horizon:
                raise ValueError(f"action_chunk_length must be in 1..{codec.action_horizon} (the model's horizon), "
                                 f"got {action_chunk_length}")
            if len(set(codec.joint_names)) != len(codec.joint_names):
                raise ValueError("The model's joint names repeat")
            self.action_chunk_length = action_chunk_length
            output_dir = current_run_output_dir()
            if output_dir is not None:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                (Path(output_dir) / "policy_provenance.json").write_text(json.dumps(codec.provenance(), indent=2) + "\n")
        except Exception:
            self.close()
            raise

    def _bind(self, env) -> None:
        live = env.unwrapped
        robot_names = list(live.scene[self.config.robot_asset_name].joint_names)
        action_names = list(live.action_manager.get_term(self.config.joint_action_term)._joint_names)
        if len(action_names) != len(self.codec.joint_names) or set(action_names) != set(self.codec.joint_names):
            raise ValueError("Live joint action contract differs from the model's joint names")
        if live.action_manager.total_action_dim != len(action_names):
            raise ValueError("This policy requires a joint-only action space")
        self.codec.bind(robot_names)
        self._action_ids = [self.codec.joint_names.index(name) for name in action_names]
        self._hold_ids = [robot_names.index(name) for name in action_names]
        chunks = SCHEDULERS[self.config.scheduler](
            num_envs=live.num_envs,
            action_chunk_length=self.action_chunk_length,
            action_horizon=self.codec.action_horizon,
            action_dim=len(action_names),
            device=self.config.policy_device,
            dtype=torch.float,
        )
        self._scheduler = PolicyRateScheduler(chunks, 1.0 / live.step_dt if self.config.policy_hz else None,
                                              self.config.policy_hz, self.config.policy_action_interpolation,
                                              live.num_envs)
        self._bound_env = live
        stamp_policy_settings(live, {"policy": self.name, "policy_variant": self.config.policy_variant,
                                     "action_chunk_length": self.action_chunk_length, "policy_hz": self.config.policy_hz,
                                     "policy_action_interpolation": self._scheduler.interpolation,
                                     "steps_per_query": self._scheduler.steps_per_query, **self.codec.stamp()})

    def _action_chunk(self, observation) -> torch.Tensor:
        num_envs = self._bound_env.num_envs
        if self.codec.requests == "all":
            batches = [list(range(num_envs))]
        else:
            needed = self._scheduler.chunks.env_requires_new_chunk.nonzero().flatten().tolist()
            batches = [needed] if self.codec.requests == "needed" else [[env_id] for env_id in needed]
        chunk = None
        for env_ids in batches:
            request = self.codec.encode(observation, env_ids, self.task_description)
            answer = self.codec.decode(self.transport.infer(request, env_ids), len(env_ids))
            if len(env_ids) == num_envs:
                chunk = answer
            else:
                if chunk is None:
                    chunk = np.zeros((num_envs, *answer.shape[1:]), dtype=np.float32)
                chunk[env_ids] = answer
        if not np.isfinite(chunk).all():
            raise ValueError(f"Non-finite actions from {self.name}")
        return torch.as_tensor(chunk[..., self._action_ids], dtype=torch.float32, device=self.config.policy_device)

    def get_action(self, env, observation) -> torch.Tensor:
        if self.transport is None:
            raise RuntimeError("Policy is closed")
        if not self.task_description:
            raise ValueError("Set a non-empty task description before inference")
        if self._bound_env is None:
            self._bind(env)
        elif self._bound_env is not env.unwrapped:
            raise ValueError("Recreate the policy when switching environments")
        if self._scheduler.asks_now:
            self.codec.observe(observation)
        hold = observation["policy"]["robot_joint_pos"][:, self._hold_ids].to(self.config.policy_device)
        return self._scheduler.get_action(lambda: self._action_chunk(observation), hold_action=hold)

    def reset(self, env_ids=None) -> None:
        if self.transport is None:
            raise RuntimeError("Policy is closed")
        self.transport.reset(env_ids)
        self.codec.reset(env_ids)
        if self._scheduler is not None:
            self._scheduler.reset(env_ids)

    def close(self) -> None:
        transport, self.transport = self.transport, None
        if transport is not None:
            transport.close()
