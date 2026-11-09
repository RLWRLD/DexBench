"""The source release and a wheel rebuilt from it must carry the code's notices."""
from __future__ import annotations

import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def test_sdist_and_rebuilt_wheel_preserve_license_and_notices(tmp_path):
    pytest.importorskip("build")
    pytest.importorskip("setuptools")
    pytest.importorskip("toml")
    project = tmp_path / "project"
    shutil.copytree(ROOT / "source/dexbench_arena", project,
                    ignore=shutil.ignore_patterns("build", "dist", "*.egg-info", "__pycache__"))
    expected = {name: (ROOT / name).read_bytes() for name in ("LICENSE", "NOTICE")}
    for name, content in expected.items():
        assert (project / name).read_bytes() == content, f"Sync the package's {name} with the root copy"
    subprocess.run([sys.executable, "-m", "build", "--no-isolation", "--sdist", str(project)],
                   check=True, capture_output=True, text=True)
    source = next((project / "dist").glob("*.tar.gz"))
    unpacked = tmp_path / "unpacked"
    with tarfile.open(source) as archive:
        for name, content in expected.items():
            member = next(m for m in archive.getmembers() if m.name.endswith("/" + name))
            assert archive.extractfile(member).read() == content
        archive.extractall(unpacked, filter="data")
    rebuilt = next(unpacked.iterdir())
    subprocess.run([sys.executable, "-m", "build", "--no-isolation", "--wheel", str(rebuilt)],
                   check=True, capture_output=True, text=True)
    with zipfile.ZipFile(next((rebuilt / "dist").glob("*.whl"))) as wheel:
        for name, content in expected.items():
            member = next(p for p in wheel.namelist() if ".dist-info/" in p and p.endswith("/" + name))
            assert wheel.read(member) == content
        metadata = wheel.read(next(p for p in wheel.namelist() if p.endswith(".dist-info/METADATA"))).decode()
        for name in expected:
            assert f"License-File: {name}" in metadata
