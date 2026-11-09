# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Pinocchio-ready URDF generation for the RBY1-M Wuji Hand 2 embodiment."""

from __future__ import annotations

import functools
import getpass
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from dexbench_arena.embodiments.rby1_wuji2.constants import (
    RBY1_WUJI2_URDF_PATH,
)


_FALLBACK_DIRS: dict[str, Path] = {}


@functools.lru_cache(maxsize=None)
def _writable_scratch_dir(subdir: str) -> Path:
    """Per-user, per-process scratch dir under the system temp root, verified writable."""

    preferred = Path(tempfile.gettempdir()) / f"dexbench_arena_{getpass.getuser()}" / subdir / str(os.getpid())
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        probe = preferred / ".write_probe"
        probe.touch()
        probe.unlink()
        return preferred
    except OSError:
        return Path(tempfile.mkdtemp(prefix=f"dexbench_arena_{subdir}_"))


def get_rby1_wuji2_pinocchio_urdf_path(source_urdf: Path | None = None) -> Path:
    """Write (idempotently) and return the Pinocchio-ready temp URDF path."""

    source_urdf = RBY1_WUJI2_URDF_PATH if source_urdf is None else Path(source_urdf)
    output_dir = _writable_scratch_dir("rby1_wuji2")
    output_path = output_dir / f"{source_urdf.stem}_pinocchio.urdf"

    tree = ET.parse(source_urdf)
    root = tree.getroot()
    for joint in root.iter("joint"):
        if joint.get("type") == "continuous":
            joint.set("type", "revolute")
            limit = joint.find("limit")
            if limit is None:
                limit = ET.SubElement(joint, "limit")
            limit.set("lower", "-6.2832")
            limit.set("upper", "6.2832")
            limit.set("effort", "100")
            limit.set("velocity", "10")
    for link in root.iter("link"):
        for geometry in [child for child in link if child.tag in ("visual", "collision")]:
            link.remove(geometry)

    try:
        tree.write(str(output_path), encoding="utf-8", xml_declaration=True)
    except PermissionError:
        fallback = _FALLBACK_DIRS.setdefault(
            "rby1_wuji2", Path(tempfile.mkdtemp(prefix="dexbench_arena_rby1_wuji2_"))
        )
        output_path = fallback / output_path.name
        tree.write(str(output_path), encoding="utf-8", xml_declaration=True)
    return output_path
