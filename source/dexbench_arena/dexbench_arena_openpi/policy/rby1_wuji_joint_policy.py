# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Remote openpi policy for DexBench RBY1-M Wuji-hand joint-space control, on :class:`RemoteJointPolicy`."""

from __future__ import annotations

import socket
from dataclasses import dataclass

import websockets.exceptions

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase
from isaaclab_arena.policy.websocket_client import WebsocketClientPolicy

from dexbench_arena.policy.policy_rate import PolicyRateCfg
from dexbench_arena.policy.remote_joint_policy import RemoteJointPolicy, Transport
from dexbench_arena_openpi.policy.rby1_wuji_joint_adapter import Pi0Rby1WujiJointCodec

MAX_RECONNECT_ATTEMPTS = 3


@dataclass
class Pi0Rby1WujiJointPolicyCfg(PolicyRateCfg):
    """Connection + runtime config for ``Pi0Rby1WujiJointRemotePolicy`` (each field becomes a runner CLI flag)."""

    policy_device: str = "cuda"
    remote_host: str = "localhost"
    remote_port: int = 8000
    ping_interval: float | None = 20.0
    """Seconds between websocket keepalive pings, or None to disable pings."""
    ping_timeout: float | None = 300.0
    """Seconds to wait for a keepalive pong before dropping, or None to wait indefinitely."""
    connect_timeout: float = 15.0
    """Seconds to wait for the server's port when connecting (the ZMQ clients' 15 s): a server that is not
    listening is reported instead of waited for."""
    action_chunk_length: int = 16
    """Steps executed from each chunk before re-planning (the execution horizon)."""
    scheduler: str = "chunk"
    """``chunk`` or ``synced_batch`` (:data:`remote_joint_policy.SCHEDULERS`)."""
    policy_variant: str | None = None
    """The model this Run scores, as ``arena_experiment_result.json`` records it (Arena reads a
    cfg's ``policy_variant`` before the class name). The benchmark sets the registry key
    (``benchmark.models``), so two checkpoints of one policy class are two models."""
    robot_asset_name: str = "robot"
    joint_action_term: str = "joint_pos"

    def __post_init__(self):
        if self.action_chunk_length < 1:
            raise ValueError("action_chunk_length must be positive")
        if self.connect_timeout <= 0:
            raise ValueError("connect_timeout must be positive")


def server_listening(host: str, port: int, timeout: float) -> None:
    """Raise ``ConnectionError`` unless something accepts a TCP connection at ``host:port`` within ``timeout``."""
    try:
        socket.create_connection((host, port), timeout=timeout).close()
    except OSError as exc:
        raise ConnectionError(
            f"Cannot reach the openpi server at {host}:{port} within {timeout:g} s ({exc}): start openpi's "
            "scripts/serve_policy.py before the run"
        ) from exc


class OpenPiWebsocketTransport(Transport):
    """openpi's websocket client (Arena's, with keepalive settings), reconnecting up to
    ``MAX_RECONNECT_ATTEMPTS`` times when the connection drops."""

    def __init__(self, config: Pi0Rby1WujiJointPolicyCfg):
        self.config = config
        self.client = self._connect()

    def _connect(self) -> WebsocketClientPolicy:
        config = self.config
        server_listening(config.remote_host, config.remote_port, config.connect_timeout)
        return WebsocketClientPolicy(host=config.remote_host, port=config.remote_port,
                                     ping_interval=config.ping_interval, ping_timeout=config.ping_timeout)

    def info(self):
        return self.client.get_server_metadata()

    def infer(self, request, env_ids):
        for attempt in range(MAX_RECONNECT_ATTEMPTS):
            try:
                return self.client.infer(request)
            except (websockets.exceptions.ConnectionClosedError, websockets.exceptions.ConnectionClosedOK,
                    OSError) as exc:
                if attempt + 1 == MAX_RECONNECT_ATTEMPTS:
                    raise
                print(f"[OpenPiWebsocketTransport] Connection lost ({exc}); reconnecting "
                      f"(attempt {attempt + 1}/{MAX_RECONNECT_ATTEMPTS - 1}) ...")
                self.close()
                self.client = self._connect()
        raise RuntimeError("unreachable")

    def close(self):
        """Release the websocket (the server outlives this client); a peer already gone is fine."""
        try:
            ws = getattr(self.client, "_ws", None)
            if ws is not None:
                ws.close()
        except (websockets.exceptions.ConnectionClosed, OSError):
            pass


@register_policy
class Pi0Rby1WujiJointRemotePolicy(RemoteJointPolicy, PolicyBase[Pi0Rby1WujiJointPolicyCfg]):
    """openpi remote closed-loop policy for the DexBench RBY1-Wuji joint-control twin."""

    name = "pi0_rby1_wuji_joint_remote"

    def __init__(self, config: Pi0Rby1WujiJointPolicyCfg) -> None:
        super().__init__(config, OpenPiWebsocketTransport(config), Pi0Rby1WujiJointCodec(config.action_chunk_length),
                         config.action_chunk_length)
