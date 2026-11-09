# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""A robot's cuRobo collision-sphere model, pinned by SHA-256."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CollisionSphereModel:
    """The sphere file, its reviewed SHA-256, and the base box the spheres leave out."""

    path: Path
    sha256: str
    base_box_m: dict[str, tuple[float, float, float]] | None = None

    def load(self) -> dict:
        """The model as a dict; ``ValueError`` when the file is not the reviewed one."""
        raw = self.path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != self.sha256:
            raise ValueError(f"{self.path}: sha256 {digest}, the reviewed model is {self.sha256}")
        return json.loads(raw)
