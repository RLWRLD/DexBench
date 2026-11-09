# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Simulator-free ZeroMQ client for the DexBench env server."""

from __future__ import annotations

from typing import Any

import torch
import zmq

from dexbench_arena.evaluation.serialization import TorchSerializer


class EnvServerError(RuntimeError):
    """The server processed the request but returned status='error'."""


class EnvServerTimeoutError(TimeoutError):
    """The server did not respond within the configured timeout."""


class DexBenchEnvClient:
    """ZMQ REQ client for :class:`~dexbench_arena.evaluation.env_server.DexBenchEnvServer`."""

    def __init__(self, host: str = "localhost", port: int = 5560, timeout_ms: int = 120_000):
        """
        Args:
            host: Server hostname or IP.
            port: Server port.
            timeout_ms: Per-request response timeout. Resets (scene randomization
                plus stabilization steps) can take tens of seconds on first call.
        """
        self._host = host
        self._port = port
        self._timeout_ms = timeout_ms
        self._context = zmq.Context()
        self._socket: zmq.Socket | None = None
        self._connect()

    def _connect(self) -> None:
        """(Re)create the REQ socket; also used to recover from a timed-out request."""
        if self._socket is not None:
            self._socket.close(linger=0)
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(f"tcp://{self._host}:{self._port}")

    def call(self, endpoint: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send one request and return the response dict (status stripped)."""
        assert self._socket is not None, "Client is closed"
        self._socket.send(TorchSerializer.to_bytes({"endpoint": endpoint, "data": data or {}}))
        if self._socket.poll(self._timeout_ms) == 0:
            self._connect()
            raise EnvServerTimeoutError(
                f"No response from tcp://{self._host}:{self._port} for '{endpoint}' within {self._timeout_ms} ms"
            )
        response = TorchSerializer.from_bytes(self._socket.recv())
        if response.get("status") != "ok":
            raise EnvServerError(response.get("message", "Unknown server error"))
        return response

    def ping(self) -> bool:
        """Liveness check; returns True when the server answers."""
        self.call("ping")
        return True

    def info(self) -> dict[str, Any]:
        """Environment metadata: num_envs, action/observation structure, joint names, ..."""
        return self.call("info")

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        """Reset the environment; returns {"observation", "info"}."""
        return self.call("reset", {"seed": seed})

    def step(self, action) -> dict[str, Any]:
        """Step with a (num_envs, action_dim) action."""
        action = torch.as_tensor(action, dtype=torch.float32).cpu()
        return self.call("step", {"action": action})

    def hold_action(self) -> torch.Tensor:
        """Current 'stay in place' action from the server, shape (num_envs, action_dim)."""
        return self.call("hold_action")["action"]

    def close(self, stop_server: bool = False) -> None:
        """Close the client; optionally ask the server process to shut down too."""
        if self._socket is None:
            return
        if stop_server:
            try:
                self.call("close")
            except (EnvServerError, EnvServerTimeoutError):
                pass
        self._socket.close(linger=0)
        self._socket = None
        self._context.term()

    def __enter__(self) -> "DexBenchEnvClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
