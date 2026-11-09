# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Isaac-free RBY1 Wuji2 Pinocchio URDF rewrite helper."""

from __future__ import annotations

import getpass
import os
import stat
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from dexbench_arena.embodiments.rby1_wuji2 import pinocchio_urdf

_URDF_TEMPLATE = """<?xml version="1.0"?>
<robot name="mini_wuji2">
  <link name="base"/>
  <link name="wheel"/>
  <link name="arm"/>
  <joint name="right_wheel_1" type="continuous">
    <parent link="base"/>
    <child link="wheel"/>
    <axis xyz="0 0 1"/>
  </joint>
  <joint name="arm_joint" type="revolute">
    <parent link="base"/>
    <child link="arm"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="10" velocity="1"/>
  </joint>
  <link name="meshy">
    <visual>
      <geometry>
        <mesh filename="parts/meshes/meshy.STL"/>
      </geometry>
    </visual>
  </link>
  <joint name="meshy_fixed" type="fixed">
    <parent link="arm"/>
    <child link="meshy"/>
  </joint>
</robot>
"""


@pytest.fixture()
def mini_urdf(tmp_path: Path) -> Path:
    urdf = tmp_path / "rby1m_wujihand2_beta2.urdf"
    urdf.write_text(_URDF_TEMPLATE)
    mesh = tmp_path / "parts" / "meshes" / "meshy.STL"
    mesh.parent.mkdir(parents=True)
    mesh.write_bytes(b"solid meshy")
    return urdf


@pytest.fixture(autouse=True)
def _fresh_scratch_state():
    pinocchio_urdf._writable_scratch_dir.cache_clear()
    pinocchio_urdf._FALLBACK_DIRS.clear()
    yield
    pinocchio_urdf._writable_scratch_dir.cache_clear()
    pinocchio_urdf._FALLBACK_DIRS.clear()


def test_rewrite_contract(mini_urdf: Path, tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path / "scratch"))
    (tmp_path / "scratch").mkdir()
    import tempfile

    tempfile.tempdir = None
    try:
        out = pinocchio_urdf.get_rby1_wuji2_pinocchio_urdf_path(mini_urdf)
        assert out.name == "rby1m_wujihand2_beta2_pinocchio.urdf"
        assert f"dexbench_arena_{getpass.getuser()}" in str(out)

        root = ET.parse(out).getroot()
        wheel = next(j for j in root.iter("joint") if j.get("name") == "right_wheel_1")
        assert wheel.get("type") == "revolute"
        limit = wheel.find("limit")
        assert limit is not None and limit.get("lower") == "-6.2832" and limit.get("upper") == "6.2832"

        assert not [e.tag for e in root.iter() if e.tag in ("visual", "collision", "mesh")]
        assert [link.get("name") for link in root.iter("link")] == ["base", "wheel", "arm", "meshy"]
    finally:
        tempfile.tempdir = None


def test_independent_ik_processes_do_not_overwrite_each_others_urdf(mini_urdf, tmp_path, monkeypatch):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setenv("TMPDIR", str(scratch))
    code = (
        "import sys; from pathlib import Path; "
        "from dexbench_arena.embodiments.rby1_wuji2.pinocchio_urdf "
        "import get_rby1_wuji2_pinocchio_urdf_path; "
        "print(get_rby1_wuji2_pinocchio_urdf_path(Path(sys.argv[1])))"
    )
    first = Path(subprocess.check_output([sys.executable, "-c", code, str(mini_urdf)], text=True).strip())
    original = first.read_bytes()
    mini_urdf.write_text(mini_urdf.read_text().replace('name="arm_joint"', 'name="changed_joint"'))
    second = Path(subprocess.check_output([sys.executable, "-c", code, str(mini_urdf)], text=True).strip())
    assert first != second
    assert first.read_bytes() == original
    assert b'changed_joint' in second.read_bytes()


def test_unwritable_preferred_dir_falls_back_to_mkdtemp(mini_urdf: Path, monkeypatch):
    real_mkdir = Path.mkdir

    def selective_mkdir(self, *args, **kwargs):
        if "dexbench_arena_" in str(self):
            raise PermissionError(13, "Permission denied", str(self))
        return real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", selective_mkdir)
    out = pinocchio_urdf.get_rby1_wuji2_pinocchio_urdf_path(mini_urdf)
    monkeypatch.undo()
    assert out.exists()
    assert f"dexbench_arena_{getpass.getuser()}/" not in str(out)


def test_root_owned_leftover_file_survived(mini_urdf: Path, monkeypatch, tmp_path: Path):
    """A writable dir with an unwritable leftover output file triggers the retry."""

    monkeypatch.setenv("TMPDIR", str(tmp_path / "scratch2"))
    (tmp_path / "scratch2").mkdir()
    import tempfile

    tempfile.tempdir = None
    try:
        preferred = pinocchio_urdf._writable_scratch_dir("rby1_wuji2")
        leftover = preferred / "rby1m_wujihand2_beta2_pinocchio.urdf"
        leftover.write_text("stale")
        leftover.chmod(stat.S_IRUSR)
        if os.geteuid() == 0:
            pytest.skip("root ignores file permission bits")

        out = pinocchio_urdf.get_rby1_wuji2_pinocchio_urdf_path(mini_urdf)
        assert out != leftover
        assert out.exists()
        assert "rby1_wuji2" in pinocchio_urdf._FALLBACK_DIRS
        again = pinocchio_urdf.get_rby1_wuji2_pinocchio_urdf_path(mini_urdf)
        assert again == out
    finally:
        tempfile.tempdir = None
