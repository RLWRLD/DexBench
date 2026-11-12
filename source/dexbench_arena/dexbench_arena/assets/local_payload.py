# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Check local asset files before the simulator reads them. No simulator imports are required."""

from pathlib import Path

import dexbench_arena
from dexbench_arena.assets.scene_bundles import geometry_payload, runtime_usd
from dexbench_arena.assets.snapshot import ASSETS_LOCK, restore_command


def require_local_assets(scene, embodiment, *, environment_name: str) -> None:
    """Fail early with download instructions when local USD payloads are absent."""

    candidates = list(scene.assets.values())
    if embodiment is not None:
        robot = getattr(getattr(embodiment, "scene_config", None), "robot", None)
        candidates.append(getattr(robot, "spawn", None))

    usd_paths = []
    for asset in candidates:
        usd_paths.append(getattr(asset, "usd_path", None))
        usd_paths.append(getattr(getattr(asset, "look", None), "usd_path", None))

    missing = []
    usd_paths = [value for path in usd_paths if isinstance(path, str)
                 for runtime in (runtime_usd(path),)
                 for value in (runtime, str(geometry_payload(runtime)))]
    for usd_path in usd_paths:
        if not isinstance(usd_path, str):
            continue
        path = Path(usd_path)
        if not path.is_absolute() or not path.is_relative_to(dexbench_arena.ASSETS_DIR):
            continue
        try:
            with path.open("rb") as stream:
                if not stream.read(40).startswith(b"version https://git-lfs"):
                    continue
        except OSError:
            pass
        if path not in missing:
            missing.append(path)

    if missing:
        missing_lines = "\n".join(f"  - {path.relative_to(dexbench_arena.REPO_ROOT)}" for path in missing)
        if (dexbench_arena.REPO_ROOT / ASSETS_LOCK).is_file():
            source = f"assets/ is the Hugging Face snapshot that {ASSETS_LOCK} pins"
            restore = "Download it on the host, as your own user:\n"
        else:
            source = "assets/ uses LFS payload from RLWRLD/DexBench-assets, see README \"Assets Reference\""
            restore = ("Restore the payload on the host using README setup commands:\n"
                       "  hf auth login --add-to-git-credential   # once per machine\n")
        raise FileNotFoundError(
            f"{environment_name} needs local assets that are not present ({source}):\n"
            f"{missing_lines}\n{restore}  {restore_command(dexbench_arena.REPO_ROOT)}"
        )
