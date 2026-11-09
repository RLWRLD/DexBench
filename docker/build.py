#!/usr/bin/env python3
"""Build the external image from committed source with verifiable provenance labels."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import verify_reproducibility as vr  # noqa: E402


def archive_into(repo: Path, revision: str, destination: Path) -> None:
    """Expand a Git snapshot, without untracked files or mutable checkout contents."""
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile() as archive:
        subprocess.run(["git", "archive", "--format=tar", revision], cwd=repo, stdout=archive, check=True,
                       env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"})
        archive.seek(0)
        subprocess.run(["tar", "-x", "-C", str(destination)], stdin=archive, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--tag", help="local image name; default: dexbench_arena:<committed recipe tag>")
    args = parser.parse_args()
    root = args.root.resolve()
    pins = vr.derive_pins(root, dockerfile=vr.EXTERNAL_DOCKERFILE)
    missing = vr.require_pins(pins)
    if missing:
        parser.error("unreadable pins: " + "; ".join(missing))
    dirty = vr.git_dirty_paths(root)
    if dirty:
        parser.error("commit or remove changes before building: " + ", ".join(dirty[:20]))
    sources = [(root, pins["source"], Path(".")),
               (root / "submodule/isaaclab_arena", pins["isaaclab_arena"], Path("submodule/isaaclab_arena")),
               (root / "submodule/isaaclab_arena/submodules/IsaacLab", pins["isaaclab"],
                Path("submodule/isaaclab_arena/submodules/IsaacLab"))]
    for repo, revision, _ in sources:
        if not (repo / ".git").exists() or vr.git_head(repo) != revision:
            parser.error(f"initialise {repo} at its pinned commit {revision}")
    build_args = {"DEXBENCH_SOURCE_REV": pins["source"],
                  "DEXBENCH_ARENA_REV": pins["isaaclab_arena"],
                  "DEXBENCH_ISAACLAB_REV": pins["isaaclab"],
                  "DEXBENCH_HF_ASSETS_REV": pins["assets"],
                  "DEXBENCH_CLOUDXR_PROFILE_SHA256": pins["cloudxr_profile_sha256"],
                  "DEXBENCH_REPRODUCIBLE": "true"}
    command = ["docker", "build", "-f", vr.EXTERNAL_DOCKERFILE,
               "-t", args.tag or f"{vr.IMAGE_NAME}:{pins['image_tag']}"]
    for name, value in build_args.items():
        command.extend(["--build-arg", f"{name}={value}"])
    with tempfile.TemporaryDirectory(prefix="dexbench-external-build-") as temporary:
        context = Path(temporary)
        for repo, revision, relative in sources:
            archive_into(repo, revision, context / relative)
        subprocess.run([*command, "."], cwd=context, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
