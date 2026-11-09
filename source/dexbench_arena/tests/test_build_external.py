# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""``docker/build.py``'s snapshots keep Git LFS pointers, so a build needs none of the LFS objects."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def test_the_snapshot_keeps_an_lfs_pointer_whose_object_this_host_lacks(tmp_path):
    if shutil.which("git-lfs") is None:
        pytest.skip("needs git-lfs")
    spec = importlib.util.spec_from_file_location("build_external", ROOT / "docker/build.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "test")
    git("lfs", "install", "--local")
    (repo / ".gitattributes").write_text("*.png filter=lfs diff=lfs merge=lfs -text\n")
    (repo / "golden.png").write_bytes(b"\x89PNG payload\n")
    git("add", ".")
    git("commit", "-q", "-m", "fixture")
    shutil.rmtree(repo / ".git/lfs/objects")
    builder.archive_into(repo, "HEAD", tmp_path / "snapshot")
    assert (tmp_path / "snapshot/golden.png").read_bytes().startswith(b"version https://git-lfs.github.com/spec/v1")
