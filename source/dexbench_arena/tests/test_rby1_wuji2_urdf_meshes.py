# Copyright (c) 2026, RLWRLD Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every mesh the RBY1-Wuji2 source URDF names exists in the assets payload."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from dexbench_arena.embodiments.rby1_wuji2.constants import RBY1_WUJI2_MESH_DIR, RBY1_WUJI2_URDF_PATH


def test_every_mesh_reference_resolves_to_an_existing_file():
    if not RBY1_WUJI2_URDF_PATH.is_file():
        pytest.skip(f"{RBY1_WUJI2_URDF_PATH} is absent (assets submodule not checked out?)")
    filenames = [m.get("filename") for m in ET.parse(RBY1_WUJI2_URDF_PATH).getroot().iter("mesh") if m.get("filename")]
    assert filenames, f"no mesh references in {RBY1_WUJI2_URDF_PATH}"

    def resolves(filename: str) -> bool:
        if filename.startswith("package://"):
            return False
        path = Path(filename)
        if path.is_absolute():
            return path.is_file()
        return (RBY1_WUJI2_URDF_PATH.parent / path).is_file() or (RBY1_WUJI2_MESH_DIR / path.name).is_file()

    missing = sorted({f for f in filenames if not resolves(f)})
    assert not missing, f"{len(missing)} mesh references do not resolve: {missing[:5]}"
