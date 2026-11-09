"""stage_tree.sh against a local repository: the tree it builds is the commit's, with the runtime inputs its own
download script installs, .source-commit and SHA256SUMS, renamed into place only at the end; a second build to the
same destination is refused."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from eval_launch import PACKAGE_DIR

pytestmark = [pytest.mark.skipif(shutil.which("git-lfs") is None, reason="stage_tree.sh needs git-lfs"),
              pytest.mark.skipif(os.geteuid() == 0, reason="stage_tree.sh refuses to run as root")]

DOWNLOAD = """#!/bin/bash
# stands in for the tree's download script: records its arguments, installs one row file
mkdir -p outputs/db_conditions && echo '{}' > outputs/db_conditions/rows.jsonl && echo "$*" > outputs/download_args
"""


FILE_SUBMODULES = {"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "protocol.file.allow", "GIT_CONFIG_VALUE_0": "always"}
"""Local submodules: git refuses file:// submodule clones unless asked."""


def _with_arena_submodule(source, tmp_path, environment=None):
    """The tree's submodule/isaaclab_arena, which stage_tree.sh checks out (a one-file stand-in repository)."""
    arena = tmp_path / "arena_source"
    arena.mkdir()
    env = {**(environment or os.environ), **FILE_SUBMODULES}
    run = lambda *args, cwd: subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd,
                                            env=env, check=True, capture_output=True)
    (arena / "README.md").write_text("arena\n")
    run("init", "-q", cwd=arena)
    run("add", ".", cwd=arena)
    run("commit", "-qm", "arena", cwd=arena)
    run("submodule", "add", "-q", str(arena), "submodule/isaaclab_arena", cwd=source)


def _git(*args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def test_a_tree_is_built_from_a_commit_in_one_command_and_never_over_another(tmp_path):
    source = tmp_path / "source"
    (source / "scripts").mkdir(parents=True)
    (source / "scripts/download_datasets.sh").write_text(DOWNLOAD)
    (source / "scripts/download_datasets.sh").chmod(0o755)
    (source / "README.md").write_text("tree\n")
    _git("init", "-q", cwd=source)
    _with_arena_submodule(source, tmp_path)
    _git("-c", "user.name=t", "-c", "user.email=t@t", "add", ".", cwd=source)
    _git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "tree", cwd=source)
    commit = _git("rev-parse", "HEAD", cwd=source)
    tools = tmp_path / "bin"
    tools.mkdir()
    (tools / "hf").write_text("#!/bin/sh\nexit 0\n")
    (tools / "hf").chmod(0o755)
    environment = {**os.environ, "PATH": f"{tools}:{Path(sys.executable).parent}:{os.environ['PATH']}",
                   "HOME": str(tmp_path / "home"), **FILE_SUBMODULES}
    (tmp_path / "home").mkdir()
    dest = tmp_path / "trees" / commit[:12]
    command = [str(PACKAGE_DIR / "stage_tree.sh"), "--repo", str(source), commit[:12], str(dest)]
    built = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=300)
    assert built.returncode == 0, built.stderr
    assert (dest / ".source-commit").read_text().strip() == commit
    assert (dest / "outputs/download_args").read_text().split() == ["--eval-conditions"]
    sums = (dest / "SHA256SUMS").read_text().splitlines()
    assert sorted(line.split()[1] for line in sums) == [".gitmodules", "README.md", "scripts/download_datasets.sh"]
    assert (dest / "submodule/isaaclab_arena/README.md").read_text() == "arena\n"
    assert not Path(f"{dest}.partial").exists()
    again = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=300)
    assert again.returncode != 0 and "exists" in again.stderr
    assert not Path(f"{dest}.partial").exists(), "refused before anything was written"


def test_only_the_assets_lfs_payloads_are_fetched_and_checked_whatever_the_hosts_lfs_setup(tmp_path):
    """A host with git-lfs installed globally (the usual one) would smudge every pointer of the commit at checkout, and
    git lfs fsck would check pointers outside assets/ that the pull never fetched. Here the source's docs/ object is
    missing, as one the user cannot read would be: the tree is built, with assets/ filled and docs/ left a pointer."""
    home = tmp_path / "home"
    home.mkdir()
    environment = {**os.environ, "HOME": str(home), "GIT_CONFIG_NOSYSTEM": "1"}
    subprocess.run(["git", "lfs", "install", "--skip-repo"], cwd=tmp_path, env=environment, check=True,
                   capture_output=True)
    source = tmp_path / "source"
    (source / "scripts").mkdir(parents=True)
    (source / "scripts/download_datasets.sh").write_text(DOWNLOAD)
    (source / "scripts/download_datasets.sh").chmod(0o755)
    (source / "assets").mkdir()
    (source / "assets/part.bin").write_text("payload\n")
    (source / "docs").mkdir()
    (source / "docs/figure.png").write_text("not ours to fetch\n")

    def git(*args):
        return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=source, env=environment,
                              check=True, capture_output=True, text=True).stdout.strip()

    git("init", "-q")
    _with_arena_submodule(source, tmp_path, environment)
    git("lfs", "track", "assets/**", "*.png")
    git("add", ".")
    git("commit", "-qm", "tree")
    figure_oid = git("lfs", "ls-files", "--long", "--include=docs/**").split()[0]
    for stored in (source / ".git/lfs/objects").rglob(figure_oid):
        stored.unlink()
    commit = git("rev-parse", "HEAD")
    tools = tmp_path / "bin"
    tools.mkdir()
    (tools / "hf").write_text("#!/bin/sh\nexit 0\n")
    (tools / "hf").chmod(0o755)
    environment.update(PATH=f"{tools}:{Path(sys.executable).parent}:{os.environ['PATH']}", **FILE_SUBMODULES)
    dest = tmp_path / "tree"
    built = subprocess.run([str(PACKAGE_DIR / "stage_tree.sh"), "--repo", str(source), commit, str(dest)], env=environment,
                           capture_output=True, text=True, timeout=300)
    assert built.returncode == 0, built.stderr
    assert (dest / "assets/part.bin").read_text() == "payload\n"
    assert (dest / "docs/figure.png").read_text().startswith("version https://git-lfs")


def test_without_repo_the_tree_comes_from_the_scripts_own_checkouts_origin(tmp_path):
    """A user of another copy of the benchmark stages from their own repository: stage_tree.sh clones its checkout's
    origin, not ours. A checkout without an origin is refused, asking for --repo."""
    source = tmp_path / "source"
    (source / "scripts").mkdir(parents=True)
    (source / "scripts/download_datasets.sh").write_text(DOWNLOAD)
    (source / "scripts/download_datasets.sh").chmod(0o755)
    _git("init", "-q", cwd=source)
    _with_arena_submodule(source, tmp_path)
    _git("-c", "user.name=t", "-c", "user.email=t@t", "add", ".", cwd=source)
    _git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "tree", cwd=source)
    commit = _git("rev-parse", "HEAD", cwd=source)
    checkout = tmp_path / "checkout"
    _git("clone", "-q", str(source), str(checkout), cwd=tmp_path)
    shutil.copy(PACKAGE_DIR / "stage_tree.sh", checkout / "stage_tree.sh")
    tools = tmp_path / "bin"
    tools.mkdir()
    (tools / "hf").write_text("#!/bin/sh\nexit 0\n")
    (tools / "hf").chmod(0o755)
    (tmp_path / "home").mkdir()
    environment = {**os.environ, "PATH": f"{tools}:{Path(sys.executable).parent}:{os.environ['PATH']}",
                   "HOME": str(tmp_path / "home"), **FILE_SUBMODULES}
    built = subprocess.run([str(checkout / "stage_tree.sh"), commit, str(tmp_path / "tree")], env=environment,
                           capture_output=True, text=True, timeout=300)
    assert built.returncode == 0, built.stderr
    assert (tmp_path / "tree/.source-commit").read_text().strip() == commit
    _git("remote", "remove", "origin", cwd=checkout)
    refused = subprocess.run([str(checkout / "stage_tree.sh"), commit, str(tmp_path / "other")], env=environment,
                             capture_output=True, text=True, timeout=60)
    assert refused.returncode == 2 and "--repo" in refused.stderr and not (tmp_path / "other.partial").exists()
