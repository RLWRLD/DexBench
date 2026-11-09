# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Torch-based wire serialization shared by the env server and client."""

from __future__ import annotations

from io import BytesIO
from typing import Any

import torch
from torch.utils import _pytree as pytree


class TorchSerializer:
    """Serialize nested Python/tensor payloads with ``torch.save``/``torch.load``."""

    @staticmethod
    def to_bytes(data: dict[str, Any]) -> bytes:
        buffer = BytesIO()
        torch.save(data, buffer)
        return buffer.getvalue()

    @staticmethod
    def from_bytes(data: bytes) -> dict[str, Any]:
        return torch.load(BytesIO(data), weights_only=False)


def tree_to_cpu(data: Any) -> Any:
    """Detach and move every tensor leaf to CPU so clients never need CUDA."""

    return pytree.tree_map(
        lambda leaf: leaf.detach().cpu() if isinstance(leaf, torch.Tensor) else leaf,
        data,
    )
