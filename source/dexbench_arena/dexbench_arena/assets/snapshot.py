# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""The assets of a checkout that downloads them from Hugging Face instead of tracking them in Git."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import re
from pathlib import Path

ASSETS_LOCK = "reproducibility/assets.lock.json"
"""Where a checkout that downloads its assets pins them: ``repo_id`` (the dataset), ``revision`` (its commit) and
``source_assets_tree`` (the ``assets/`` tree that commit was uploaded from)."""

RECEIPT = "asset-upload-manifest.json"
"""The snapshot's receipt, at the dataset's root (``assets/`` after the download)."""

DOWNLOAD_STATE = ".cache"
"""What ``hf download --local-dir`` keeps in ``assets/`` beside the files. It is not part of the snapshot."""

_OBJECT_ID = re.compile(r"[0-9a-f]{40}")


def lock_problems(lock: object) -> list[str]:
    """Why ``lock`` (the parsed ``ASSETS_LOCK``) cannot pin a snapshot."""
    if not isinstance(lock, dict):
        return [f"{ASSETS_LOCK} is not a JSON object"]
    problems = []
    if not isinstance(lock.get("repo_id"), str) or "/" not in lock["repo_id"]:
        problems.append(f"{ASSETS_LOCK}: repo_id must name a Hugging Face dataset, not {lock.get('repo_id')!r}")
    for key, what in (("revision", "the dataset commit"), ("source_assets_tree", "the assets/ tree it was uploaded from")):
        if not _OBJECT_ID.fullmatch(str(lock.get(key, ""))):
            problems.append(f"{ASSETS_LOCK}: {key} must be the full id of {what}, not {lock.get(key)!r}")
    return problems


def parse_lock(text: str) -> dict:
    """``ASSETS_LOCK``'s text as a lock. Raises ValueError when it cannot pin a snapshot."""
    try:
        lock = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"{ASSETS_LOCK} is not JSON: {error}") from error
    problems = lock_problems(lock)
    if problems:
        raise ValueError("; ".join(problems))
    return lock


def read_lock(root: str | Path) -> dict | None:
    """The lock of the checkout at ``root``, or None when the checkout tracks ``assets/`` in Git."""
    path = Path(root) / ASSETS_LOCK
    return parse_lock(path.read_text()) if path.is_file() else None


def locked_assets_tree(text: str | None) -> str | None:
    """The assets pin a lock's text names (``source_assets_tree``), or None for no text or an invalid lock."""
    try:
        return parse_lock(text)["source_assets_tree"] if text else None
    except ValueError:
        return None


def restore_command(root: str | Path, include: str | None = None) -> str:
    """The host command that installs the asset files under ``include`` (a glob relative to ``assets/``; every
    asset when None) in the checkout at ``root``."""
    if (Path(root) / ASSETS_LOCK).is_file():
        return "./scripts/download_assets.sh" + (f' --include "{include}"' if include else "")
    return f'git lfs pull --include="assets/{include or "**"}" --exclude=""'


def read_receipt(assets_dir: str | Path) -> dict | None:
    """The receipt of the snapshot in ``assets_dir``, or None when there is none or it is not JSON."""
    try:
        receipt = json.loads((Path(assets_dir) / RECEIPT).read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return receipt if isinstance(receipt, dict) else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shared_files(receipt: dict) -> list[str]:
    """The files that the snapshot stores in Git, not LFS (``"storage": "git"`` in the receipt): the indexes, reference
    views, sidecars and documents, about 7 MB. A Git checkout of ``assets/`` holds them before any LFS download, and
    asset registration reads some of them for every package (``env/scene_bundle_index.json``, the reference views),
    so a partial download installs them all."""
    return [entry["path"] for entry in receipt.get("files", ()) if entry.get("storage") == "git"]


def selected_files(receipt: dict, include=()) -> list[dict]:
    """The receipt's file records that a download of ``include`` (globs relative to ``assets/``, as ``hf download
    --include`` takes them) installs: the files they match and the ``shared_files``; all of them when it is empty."""
    shared = set(shared_files(receipt)) if include else set()
    return [entry for entry in receipt.get("files", ())
            if not include or entry["path"] in shared
            or any(fnmatch.fnmatchcase(entry["path"], pattern) for pattern in include)]


def file_mismatches(assets_dir: str | Path, receipt: dict, include=()) -> list[str]:
    """Each file of the snapshot (of ``selected_files``) that ``assets_dir`` does not hold as the receipt records it:
    missing, or another size or SHA-256."""
    root = Path(assets_dir)
    found = []
    for entry in selected_files(receipt, include):
        relative = entry["path"]
        path = root / relative
        if not path.is_file():
            found.append(f"{relative}: missing")
        elif path.stat().st_size != entry["size"]:
            found.append(f"{relative}: {path.stat().st_size} bytes, the snapshot has {entry['size']}")
        elif _sha256(path) != entry["sha256"]:
            found.append(f"{relative}: SHA-256 differs from the snapshot's")
    return found


def extra_files(assets_dir: str | Path, receipt: dict) -> list[str]:
    """Files in ``assets_dir`` that the snapshot does not have: left by an earlier snapshot, or added by hand."""
    root = Path(assets_dir)
    known = {entry["path"] for entry in receipt.get("files", ())} | {RECEIPT}
    return sorted(relative for relative in (path.relative_to(root).as_posix() for path in root.rglob("*")
                                            if path.is_file() or path.is_symlink())
                  if relative not in known and relative.split("/", 1)[0] != DOWNLOAD_STATE)


def snapshot_problems(assets_dir: str | Path, lock: dict, include=()) -> list[str]:
    """Why ``assets_dir`` is not the snapshot ``lock`` pins: no receipt, a receipt of another dataset or tree, a file
    that differs (``file_mismatches``), and, for the whole snapshot (no ``include``), a file it does not have."""
    receipt = read_receipt(assets_dir)
    if receipt is None:
        return [f"assets/{RECEIPT} is missing or unreadable: the download did not finish"]
    problems = []
    if receipt.get("repository") != lock["repo_id"]:
        problems.append(f"assets/ is a snapshot of {receipt.get('repository')!r}, not {lock['repo_id']}")
    if receipt.get("source_assets_tree") != lock["source_assets_tree"]:
        problems.append(f"assets/ was uploaded from assets tree {receipt.get('source_assets_tree')}, "
                        f"not {lock['source_assets_tree']}: download {lock['repo_id']}@{lock['revision']}")
    problems += file_mismatches(assets_dir, receipt, include)
    if not include:
        problems += [f"{relative}: not in the snapshot" for relative in extra_files(assets_dir, receipt)]
    return problems
