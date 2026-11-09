"""Resolve explicit USD views of prepared scenes and their logical source paths."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from dexbench_arena import ASSETS_DIR


@lru_cache(maxsize=4)
def _index(assets_root: Path) -> dict:
    path = assets_root / "env/scene_bundle_index.json"
    if not path.is_file():
        raise FileNotFoundError(f"Integrated scene index is missing: {path}")
    data = json.loads(path.read_text())
    if data.get("format_version") != 1:
        raise ValueError(f"Unsupported scene bundle index: {path}")
    return data


def runtime_usd(path: str | Path) -> str:
    """Return the installed reference view for an integrated source, or its unchanged path."""
    original = str(path)
    source = Path(original)
    if not source.is_relative_to(ASSETS_DIR):
        return original
    row = _index(ASSETS_DIR)["sources"].get(source.relative_to(ASSETS_DIR).as_posix())
    if row is None:
        return original
    view = ASSETS_DIR / row["view"]
    if not view.is_file():
        raise FileNotFoundError(f"Integrated runtime view is missing: {view}")
    return str(view)


def source_usd(path: str | Path) -> Path:
    """Find the retained source sidecar's USD path without loading its geometry."""
    usd = Path(path)
    if not usd.is_relative_to(ASSETS_DIR):
        return usd
    source = _index(ASSETS_DIR)["views"].get(usd.relative_to(ASSETS_DIR).as_posix())
    return ASSETS_DIR / source if source is not None else usd


def geometry_payload(path: str | Path) -> Path:
    """Locate the actual LFS geometry behind a tiny reference view for early payload checks."""
    usd = Path(path)
    if not usd.is_relative_to(ASSETS_DIR):
        return usd
    index = _index(ASSETS_DIR)
    relative = usd.relative_to(ASSETS_DIR).as_posix()
    source = index["views"].get(relative)
    return ASSETS_DIR / index["sources"][source]["bundle"] if source else usd
