"""What a model implements to be served: :class:`ServedPolicy`."""

from __future__ import annotations

from typing import Any

import numpy as np


class ServedPolicy:
    """A model the server answers for. The server makes one with no arguments, calls :meth:`load` once before it
    prints its ready line, then :meth:`info` once and :meth:`predict` for every request, one request at a time, all on
    the server's main thread (thread-local settings made in :meth:`load` hold in ``predict``, and what needs the main
    thread, such as ``signal.signal``, works there)."""

    def load(self, checkpoint: str | None, options: dict[str, str]) -> None:
        """Load the model from ``checkpoint`` (``--checkpoint``), with ``options`` (``--option key=value``, as
        strings). Anything slow (weights, compilation, a warm-up) belongs here: the ready line follows it."""

    def info(self) -> dict[str, Any]:
        raise NotImplementedError

    def predict(self, images: dict[str, np.ndarray], state: np.ndarray, prompt: str) -> np.ndarray:
        """``images``: camera key to ``(height, width, 3)`` uint8 RGB, at ``frame_hw``; ``state``: the measured
        joint positions in ``joint_names`` order (radians); ``prompt``: the task's instruction. Returns
        ``(action_horizon, len(joint_names))`` absolute joint targets in radians. Each request stands alone: the
        client sends no episode id."""
        raise NotImplementedError
