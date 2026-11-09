#!/usr/bin/env python3
# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Check that what runs is what the commit says.

The pins are the commit itself: the assets subtree OID (or the assets gitlink of a
commit that has one) fixes ``assets/``; submodule gitlinks fix ``submodule/isaaclab_arena`` and, through
it, IsaacLab and Isaac-GR00T. The ``ARG``
lines and ``pip install ...==`` pins of ``docker/Dockerfile`` fix the base
image digest, the image tag and the package versions; ``docker/cloudxr_quest3.env`` is the
CloudXR profile. Nothing is declared twice, and nothing is read from the working tree:
``derive_pins`` reads all of it from a commit's tree (``git ls-tree``, ``git show``).

Everything else is one comparison. A built image's provenance labels and a dataset card
are turned into the same pin record (``pins_from_labels``, ``card["pins"]``) and compared
with ``compare_pins``; the working tree is what gets compared *against* the commit, never
the source of a pin. Two rules keep the comparison honest: an unknown expectation is a
failure, not a skipped check (a pin that cannot be read says "initialise the submodule",
it never passes), and a git command that fails raises instead of reading as "clean".

* ``--check host``: the working tree matches the commit (assets checkout on the pin, clean,
  LFS payload present; the submodule objects available; the mounted CloudXR profile
  identical to the committed one) and the dataset directory exists;
* ``--check image``: the image's labels and recording packages against the commit;
* ``--check cloudxr``: the running CloudXR server's marker, socket and container;
* ``--profile <dexbench_card.json>``: a dataset card
  against this checkout, and the card against itself (its image labels against its pins).

``--print-pins`` prints the commit's pins as shell assignments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source" / "dexbench_arena"))
from dexbench_arena.recording.git_status import GitUnavailable, git_dirty_paths  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = "docker/Dockerfile"
EXTERNAL_DOCKERFILE = "docker/Dockerfile"
IMAGE_NAME = "dexbench_arena"
CLOUDXR_PROFILE = "docker/cloudxr_quest3.env"
CLOUDXR_RUNTIME_JSON = "openxr_cloudxr.json"
CLOUDXR_RUNTIME_ENV = "run/cloudxr.env"
CLOUDXR_RUNTIME_SOCKET = "run/ipc_cloudxr"
CLOUDXR_IMAGE_MARKER = "run/dexbench_repro_image.env"
MARKER_PIN_KEYS = ("source", "cloudxr_profile_sha256")
RECORDING_PACKAGES = ("isaacteleop", "av", "huggingface-hub", "torchcodec")
LOCAL_LABEL_PACKAGES = ("torchcodec",)
PIN_KEYS = (
    "source",
    "isaaclab_arena",
    "isaaclab",
    "isaac_groot",
    "assets",
    "base_image",
    "image_tag",
    "cloudxr_profile",
    "cloudxr_profile_sha256",
)
LABEL_OF_PIN = {
    "source": "org.opencontainers.image.revision",
    "isaaclab_arena": "io.dexbench.isaaclab-arena-revision",
    "isaaclab": "io.dexbench.isaaclab-revision",
    "base_image": "io.dexbench.base-image",
    "assets": "io.dexbench.hf-assets-revision",
    "cloudxr_profile_sha256": "io.dexbench.cloudxr-profile-sha256",
}
PIN_HINTS = {
    "source": "not a git checkout, or the revision does not exist",
    "isaaclab_arena": "the commit carries no submodule/isaaclab_arena gitlink",
    "isaaclab": "initialise IsaacLab-Arena: git submodule update --init submodule/isaaclab_arena",
    "isaac_groot": "initialise IsaacLab-Arena: git submodule update --init submodule/isaaclab_arena",
    "assets": "the commit carries no assets subtree or legacy gitlink",
    "base_image": "selected Dockerfile has no ARG BASE_IMAGE in this commit",
    "image_tag": "selected Dockerfile has no ARG DEXBENCH_IMAGE_TAG in this commit",
    "cloudxr_profile_sha256": f"{CLOUDXR_PROFILE} is not in this commit",
}
PIN_LABELS = {
    "source": "DexBench commit",
    "isaaclab_arena": "IsaacLab-Arena commit",
    "isaaclab": "IsaacLab commit",
    "isaac_groot": "Isaac-GR00T commit",
    "assets": "assets revision",
    "base_image": "base image",
    "image_tag": "image tag",
    "cloudxr_profile": "CloudXR profile",
    "cloudxr_profile_sha256": "CloudXR profile sha256",
}


class Checks:
    def __init__(self) -> None:
        self.ok_count = 0
        self.errors: list[str] = []

    def ok(self, message: str) -> None:
        self.ok_count += 1
        print(f"[OK] {message}")

    def note(self, message: str) -> None:
        print(f"[NOTE] {message}")

    def fail(self, message: str) -> None:
        self.errors.append(message)
        print(f"[FAIL] {message}", file=sys.stderr)

    def equal(self, label: str, actual: str | None, expected: str | None) -> None:
        if expected is None:
            self.note(f"{label}: expected value unknown, not compared")
        elif actual == expected:
            self.ok(f"{label}: {expected}")
        else:
            self.fail(f"{label}: expected {expected}, got {actual or '<missing>'}")

    def finish(self) -> int:
        if self.errors:
            print(
                f"Reproducibility preflight failed: {len(self.errors)} error(s), "
                f"{self.ok_count} check(s) passed.",
                file=sys.stderr,
            )
            return 1
        print(f"Reproducibility preflight passed ({self.ok_count} checks).")
        return 0




def _run(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(args, cwd=cwd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.stdout.strip()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(cwd: Path) -> str | None:
    try:
        return _run("git", "rev-parse", "HEAD", cwd=cwd)
    except (OSError, subprocess.CalledProcessError):
        return None


def git_show(path: str, *, revision: str, cwd: Path) -> bytes | None:
    """The blob at ``path`` in ``revision``'s tree, or None when absent."""
    try:
        return subprocess.run(
            ["git", "show", f"{revision}:{path}"], cwd=cwd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def gitlink(path: str, *, revision: str, cwd: Path) -> str | None:
    """The commit a submodule gitlink names in ``revision``'s tree, or None."""
    try:
        output = _run("git", "ls-tree", revision, path, cwd=cwd)
    except (OSError, subprocess.CalledProcessError):
        return None
    fields = output.split()
    return fields[2] if len(fields) >= 3 and fields[1] == "commit" else None


def asset_revision(*, revision: str, cwd: Path) -> str | None:
    """The assets tree OID, or the assets gitlink OID of a commit that pins assets as a submodule."""
    try:
        fields = _run("git", "ls-tree", revision, "assets", cwd=cwd).split()
    except (OSError, subprocess.CalledProcessError):
        return None
    return fields[2] if len(fields) >= 3 and fields[1] in {"tree", "commit"} else None


def git_has_commit(commit: str, cwd: Path) -> bool:
    try:
        _run("git", "cat-file", "-e", f"{commit}^{{commit}}", cwd=cwd)
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


LFS_POINTER_HEAD = b"version https://git-lfs.github.com/spec/v1"


def parse_lfs_pointer(blob: bytes) -> tuple[str, int] | None:
    """``(sha256 oid, size)`` of an LFS pointer blob, or None for any other content."""
    if not blob.startswith(LFS_POINTER_HEAD):
        return None
    oid, size = None, None
    for line in blob.decode("utf-8", "replace").splitlines():
        key, _, value = line.partition(" ")
        if key == "oid" and value.startswith("sha256:"):
            oid = value[len("sha256:"):]
        elif key == "size" and value.isdigit():
            size = int(value)
    return (oid, size) if oid and size is not None else None


def _git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_mismatches(cwd: Path, revision: str, *, env: dict[str, str] | None = None) -> list[str]:
    """Working-tree files that are not the blobs ``revision``'s tree names, by content."""
    try:
        listing = subprocess.run(
            ["git", "ls-tree", "--full-tree", "-r", "-l", "-z", revision], cwd=cwd, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", "") or str(error)
        raise GitUnavailable(f"git ls-tree {revision} failed in {cwd}: {detail.strip()}") from error
    entries = []
    for entry in listing.split("\0"):
        if not entry:
            continue
        meta, _, relative = entry.partition("\t")
        mode, kind, oid, size = meta.split()
        if kind == "blob":
            entries.append((relative, oid, int(size), mode))
    small = [oid for _, oid, size, _ in entries if size < 1024]
    pointers: dict[str, tuple[str, int]] = {}
    if small:
        batch = subprocess.run(
            ["git", "cat-file", "--batch"], cwd=cwd, check=True, input=("\n".join(small) + "\n").encode(), stdout=subprocess.PIPE, env=env,
        ).stdout
        pos = 0
        while pos < len(batch):
            header_end = batch.index(b"\n", pos)
            header = batch[pos:header_end].split()
            length = int(header[2])
            body = batch[header_end + 1:header_end + 1 + length]
            pos = header_end + 1 + length + 1
            parsed = parse_lfs_pointer(body)
            if parsed:
                pointers[header[0].decode()] = parsed
    mismatches = []
    for relative, oid, size, mode in entries:
        path = cwd / relative
        if mode == "120000":
            if not path.is_symlink():
                mismatches.append(f"{relative}: not a symlink")
            continue
        if not path.is_file():
            mismatches.append(f"{relative}: missing")
            continue
        if oid in pointers:
            payload_oid, payload_size = pointers[oid]
            actual_size = path.stat().st_size
            if actual_size != payload_size:
                what = "LFS pointer, payload not pulled" if actual_size < 1024 and path.read_bytes().startswith(LFS_POINTER_HEAD) else f"size {actual_size} != {payload_size}"
                mismatches.append(f"{relative}: {what}")
            elif _sha256_file(path) != payload_oid:
                mismatches.append(f"{relative}: LFS payload sha256 differs from the pointer's oid")
        elif _git_blob_sha1(path) != oid:
            mismatches.append(f"{relative}: content differs from blob {oid[:12]}")
    return mismatches




def dockerfile_arg(name: str, text: str) -> str | None:
    """The default of ``ARG name=value`` (first declaration; quotes stripped), or None."""
    match = re.search(rf"^ARG\s+{re.escape(name)}=(\"?)(.*?)\1\s*$", text, re.MULTILINE)
    return match.group(2) if match else None


def dockerfile_package_pins(text: str) -> dict[str, str]:
    """Every ``name==version`` (``name[extras]==version``) the Dockerfile installs."""
    pins: dict[str, str] = {}
    pattern = re.compile(r"(?<![\w./-])([A-Za-z0-9][A-Za-z0-9_.-]*)(?:\[[^\]]*\])?==([0-9][A-Za-z0-9_.+-]*)")
    for match in pattern.finditer(text):
        pins[match.group(1)] = match.group(2)
    return pins


def derive_pins(root: Path = REPO_ROOT, revision: str = "HEAD", *, dockerfile: str | None = None) -> dict[str, Any]:
    """What the commit ``revision`` of the checkout at ``root`` pins."""
    try:
        source = _run("git", "rev-parse", f"{revision}^{{commit}}", cwd=root)
    except (OSError, subprocess.CalledProcessError):
        source = None
    selected = dockerfile or DOCKERFILE
    dockerfile_blob = git_show(selected, revision=revision, cwd=root) if source else None
    if source and dockerfile is None and dockerfile_blob is None:
        dockerfile_blob = git_show(EXTERNAL_DOCKERFILE, revision=revision, cwd=root)
    dockerfile = dockerfile_blob.decode("utf-8") if dockerfile_blob else ""
    profile_blob = git_show(CLOUDXR_PROFILE, revision=revision, cwd=root) if source else None
    arena = root / "submodule" / "isaaclab_arena"
    arena_rev = gitlink("submodule/isaaclab_arena", revision=revision, cwd=root) if source else None
    nested: dict[str, str | None] = {"isaaclab": None, "isaac_groot": None}
    if arena_rev and (arena / ".git").exists():
        nested["isaaclab"] = gitlink("submodules/IsaacLab", revision=arena_rev, cwd=arena)
        nested["isaac_groot"] = gitlink("submodules/Isaac-GR00T", revision=arena_rev, cwd=arena)
    packages = dockerfile_package_pins(dockerfile)
    return {
        "source": source,
        "isaaclab_arena": arena_rev,
        **nested,
        "assets": asset_revision(revision=revision, cwd=root) if source else None,
        "base_image": dockerfile_arg("BASE_IMAGE", dockerfile),
        "image_tag": dockerfile_arg("DEXBENCH_IMAGE_TAG", dockerfile),
        "cloudxr_profile": CLOUDXR_PROFILE,
        "cloudxr_profile_sha256": _sha256_bytes(profile_blob) if profile_blob is not None else None,
        "packages": {name: packages.get(name) for name in RECORDING_PACKAGES},
    }


def pins_from_labels(labels: dict[str, str] | None) -> dict[str, str | None]:
    """The pin record an image's provenance labels describe (missing labels -> None)."""
    labels = labels or {}
    return {pin: labels.get(label) for pin, label in LABEL_OF_PIN.items()}


def compare_pins(checks: Checks, expected: dict[str, Any], actual: dict[str, Any], *, label: str, keys=PIN_KEYS) -> None:
    """One pin record against another: for every key in ``keys``, ``actual`` must equal ``expected``."""
    for key in keys:
        expected_value = expected.get(key)
        if expected_value is None:
            checks.fail(f"{label}: {PIN_LABELS[key]} is unknown on the reference side ({PIN_HINTS.get(key, 'cannot be compared')})")
        else:
            checks.equal(f"{label}: {PIN_LABELS[key]}", actual.get(key), expected_value)


def require_pins(pins: dict[str, Any], keys=PIN_KEYS) -> list[str]:
    """The pins in ``keys`` that could not be read, with what to do about each."""
    return [f"{PIN_LABELS[key]}: {PIN_HINTS.get(key, 'unreadable')}" for key in keys if pins.get(key) is None]


def shell_pins(pins: dict[str, Any]) -> str:
    """The scalar pins as ``PIN_<KEY>='value'`` lines (empty when unknown), for ``eval``."""
    return "\n".join(f"PIN_{key.upper()}={shlex.quote(pins.get(key) or '')}" for key in PIN_KEYS) + "\n"




def worktree_state(root: Path, pins: dict[str, Any]) -> dict[str, Any]:
    """How the working tree at ``root`` relates to the commit's pins."""
    assets = root / "assets"
    legacy = bool(pins.get("source") and gitlink("assets", revision=pins["source"], cwd=root))
    profile = root / CLOUDXR_PROFILE
    state: dict[str, Any] = {
        "dirty": git_dirty_paths(root),
        "assets_initialised": (assets / ".git").exists() if legacy else assets.is_dir(),
        "assets_head": None,
        "assets_untracked": [],
        "assets_mismatches": [],
        "cloudxr_profile_sha256": _sha256_file(profile) if profile.is_file() else None,
    }
    if state["assets_initialised"] and pins.get("assets"):
        state["assets_head"] = git_head(assets) if legacy else asset_revision(revision="HEAD", cwd=root)
        state["assets_untracked"] = git_dirty_paths(assets) if legacy else git_dirty_paths(root, ("assets",), all_untracked=True)
        state["assets_mismatches"] = tree_mismatches(assets, pins["assets"])
    return state


def check_host(checks: Checks, pins: dict[str, Any], root: Path, datasets_root: Path) -> dict[str, Any] | None:
    try:
        state = worktree_state(root, pins)
    except GitUnavailable as error:
        checks.fail(f"working tree state unknown: {error}")
        return None

    if pins["assets"] is None:
        checks.fail("the commit carries no assets subtree or legacy gitlink")
    elif not state["assets_initialised"]:
        checks.fail("assets payload directory is missing; restore the assets tree (initialise assets for a legacy gitlink)")
    else:
        checks.equal("working tree: assets checkout", state["assets_head"], pins["assets"])
        if state["assets_mismatches"]:
            shown = "; ".join(state["assets_mismatches"][:5])
            checks.fail(f"assets files differ from the pinned tree ({len(state['assets_mismatches'])}): {shown}; restore the committed assets and pull their LFS payload")
        else:
            checks.ok("assets files are the pinned tree's blobs (LFS payload by sha256)")
        if state["assets_untracked"]:
            checks.fail(f"assets checkout has changes on top of the pinned tree ({len(state['assets_untracked'])}): {state['assets_untracked'][:5]}")
        else:
            checks.ok("assets checkout has nothing on top of the pinned tree")

    checks.equal("working tree: CloudXR profile sha256", state["cloudxr_profile_sha256"], pins["cloudxr_profile_sha256"])

    data_dir = datasets_root / "dexbench_arena"
    if data_dir.is_dir():
        checks.ok(f"dataset output directory: {data_dir} -> /datasets/dexbench_arena")
    else:
        checks.fail(f"dataset output directory missing: {data_dir}; run ./scripts/download_datasets.sh")

    arena = root / "submodule" / "isaaclab_arena"
    if pins["isaaclab_arena"] is None:
        checks.fail("the commit carries no submodule/isaaclab_arena gitlink")
    elif not (arena / ".git").exists():
        checks.fail("IsaacLab-Arena submodule not initialised; run\n  git submodule update --init --recursive submodule/isaaclab_arena")
    else:
        for key, path in (
            ("isaaclab_arena", arena),
            ("isaaclab", arena / "submodules" / "IsaacLab"),
            ("isaac_groot", arena / "submodules" / "Isaac-GR00T"),
        ):
            expected = pins[key]
            if expected is None:
                checks.fail(f"{PIN_LABELS[key]} unreadable: the IsaacLab-Arena checkout lacks commit {pins['isaaclab_arena']}")
            elif not (path / ".git").exists():
                checks.fail(f"{PIN_LABELS[key]} checkout not initialised at {path}")
            elif git_has_commit(expected, path):
                checks.ok(f"{PIN_LABELS[key]} is available: {expected}")
            else:
                checks.fail(f"{PIN_LABELS[key]} {expected} is unavailable locally; run git submodule update --init --recursive")

    if state["dirty"]:
        checks.note(f"working tree differs from the commit in {len(state['dirty'])} paths (e.g. {state['dirty'][0]}); a reproducible build uses the commit, not these edits")
    return state




def _docker_inspect(image: str) -> dict[str, Any] | None:
    try:
        output = _run("docker", "image", "inspect", image)
    except (OSError, subprocess.CalledProcessError):
        return None
    data = json.loads(output)
    return data[0] if data else None


def image_labels(image_info: dict[str, Any] | None) -> dict[str, str]:
    return (image_info or {}).get("Config", {}).get("Labels") or {}


def check_image_labels(checks: Checks, pins: dict[str, Any], labels: dict[str, str] | None, *, label: str) -> None:
    """Provenance labels (of an image, or recorded in a card) against a pin record."""
    labels = labels or {}
    checks.equal(f"{label}: reproducible label", labels.get("io.dexbench.reproducible"), "true")
    compare_pins(checks, pins, pins_from_labels(labels), label=label, keys=tuple(LABEL_OF_PIN))


def check_image(checks: Checks, pins: dict[str, Any], image: str) -> dict[str, Any] | None:
    inspected = _docker_inspect(image)
    if inspected is None:
        checks.fail(f"Docker image is unavailable: {image}")
        return None
    checks.ok(f"Docker image exists: {image} ({inspected.get('Id', '<unknown>')})")
    check_image_labels(checks, pins, image_labels(inspected), label="image")

    package_script = (
        "import importlib.metadata as m, json; "
        f"names={RECORDING_PACKAGES!r}; "
        "print(json.dumps({name:m.version(name) for name in names}))"
    )
    try:
        output = _run("docker", "run", "--rm", "--entrypoint", "/isaac-sim/python.sh", image, "-c", package_script)
        installed = json.loads(output)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError):
        checks.fail(f"could not inspect Python packages in image: {image}")
    else:
        for package, expected in pins["packages"].items():
            actual = installed.get(package)
            if (package in LOCAL_LABEL_PACKAGES and actual and expected
                    and "+" not in expected and actual.split("+", 1)[0] == expected):
                actual = expected
            checks.equal(f"image package {package}", actual, expected)
    return inspected


def _read_marker(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def check_cloudxr(checks: Checks, pins: dict[str, Any], cloudxr_dir: Path, image_info: dict[str, Any] | None) -> None:
    socket_mtime_ns: int | None = None
    for key, relative in (("runtime_json", CLOUDXR_RUNTIME_JSON), ("runtime_env", CLOUDXR_RUNTIME_ENV)):
        path = cloudxr_dir / relative
        if path.is_file():
            checks.ok(f"CloudXR {key}: {path}")
        else:
            checks.fail(f"CloudXR {key} missing: {path}")

    socket_path = cloudxr_dir / CLOUDXR_RUNTIME_SOCKET
    try:
        mode = socket_path.stat().st_mode
    except FileNotFoundError:
        checks.fail(f"CloudXR runtime socket missing: {socket_path}")
    else:
        if stat.S_ISSOCK(mode):
            checks.ok(f"CloudXR runtime socket is live: {socket_path}")
            socket_mtime_ns = socket_path.stat().st_mtime_ns
        else:
            checks.fail(f"CloudXR runtime path is not a socket: {socket_path}")

    marker_path = cloudxr_dir / CLOUDXR_IMAGE_MARKER
    if not marker_path.is_file():
        checks.fail(f"CloudXR image marker missing: {marker_path}; start the CloudXR server as docker/README.md says")
        return

    marker = _read_marker(marker_path)
    checks.ok(f"CloudXR image marker: {marker_path}")
    if socket_mtime_ns is not None:
        if socket_mtime_ns >= marker_path.stat().st_mtime_ns:
            checks.ok("CloudXR socket was created by the marked runtime launch")
        else:
            checks.fail("CloudXR socket predates the marked runtime launch; restart the CloudXR server")

    if image_info is not None:
        checks.equal("CloudXR server image ID", marker.get("DEXBENCH_IMAGE_ID"), image_info.get("Id", ""))
    compare_pins(
        checks, pins,
        {"source": marker.get("DEXBENCH_SOURCE_REV"), "cloudxr_profile_sha256": marker.get("DEXBENCH_CLOUDXR_PROFILE_SHA256")},
        label="CloudXR server", keys=MARKER_PIN_KEYS,
    )

    container_id = marker.get("DEXBENCH_CONTAINER_ID")
    if not container_id:
        checks.fail("CloudXR marker has no DEXBENCH_CONTAINER_ID")
        return
    try:
        inspected = json.loads(_run("docker", "container", "inspect", container_id))[0]
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError, IndexError):
        checks.fail(f"CloudXR container is unavailable: {container_id}")
        return
    if inspected.get("State", {}).get("Running") is True:
        checks.ok(f"CloudXR container is running: {container_id}")
    else:
        checks.fail(f"CloudXR container is not running: {container_id}")
    if image_info is not None:
        checks.equal("CloudXR container image ID", inspected.get("Image"), image_info.get("Id", ""))




CARD_FORMAT_VERSION = 2


def card_structure_errors(card: Any) -> list[str]:
    """What keeps ``card`` from being a dataset card this script can judge."""
    if not isinstance(card, dict):
        return ["not a JSON object"]
    errors = []
    if card.get("format_version") != CARD_FORMAT_VERSION:
        errors.append(f"format_version {card.get('format_version')!r} != {CARD_FORMAT_VERSION}")
    if not isinstance(card.get("dataset"), str) or not card["dataset"]:
        errors.append("no dataset name")
    pins = card.get("pins")
    if not isinstance(pins, dict):
        errors.append("no pins")
    else:
        errors.extend(f"pins.{key} missing" for key in PIN_KEYS if not pins.get(key))
    image = card.get("image")
    if not isinstance(image, dict) or not image.get("id") or not isinstance(image.get("labels"), dict):
        errors.append("no image record (id and labels)")
    return errors


def check_profile(
    checks: Checks, pins: dict[str, Any], card: dict[str, Any], image_info: dict[str, Any] | None,
    *, root: Path = REPO_ROOT,
) -> None:
    """A dataset card against this checkout, and against itself."""
    errors = card_structure_errors(card)
    if errors:
        checks.fail("dataset card is incomplete: " + "; ".join(errors))
        return
    name = card["dataset"]
    card_pins = card["pins"]
    compare_pins(checks, card_pins, pins, label=f"{name}: checkout")
    if card.get("dirty"):
        checks.note(f"{name} was recorded from a dirty checkout ({len(card['dirty'])} paths); the commit does not fully describe it")
    check_image_labels(checks, card_pins, card["image"]["labels"], label=f"{name}: recorded image")
    if image_info is not None:
        checks.equal(f"{name}: image ID", image_info.get("Id"), card["image"]["id"])
        check_image_labels(checks, card_pins, image_labels(image_info), label=f"{name}: local image")
    if checks.errors and card_pins.get("source"):
        source = card_pins["source"]
        checks.note(
            f"to reproduce {name}: preserve the existing checkout and use a separate clone"
            ". With Git LFS installed, run "
            f"GIT_LFS_SKIP_SMUDGE=1 git checkout --detach {shlex.quote(source)} there."
        )
        if gitlink("assets", revision=source, cwd=root):
            checks.note(
                "historical assets gitlink: GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --depth 1 assets "
                "&& git -C assets lfs pull"
            )
        elif asset_revision(revision=source, cwd=root):
            checks.note(
                "assets subtree: git lfs install --local --skip-smudge --skip-repo "
                "&& git config core.hooksPath .githooks "
                '&& git lfs pull --include="assets/**" --exclude=""'
            )
        else:
            checks.note(
                "source assets layout is unavailable locally; fetch that commit in the separate clone"
                ""
            )
        checks.note("initialise code: git submodule update --init --recursive submodule/isaaclab_arena")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", choices=("host", "image", "cloudxr", "all"), default="host")
    parser.add_argument("--profile", type=Path, help="a dataset card to compare the checkout with")
    parser.add_argument("--print-pins", action="store_true", help="print the commit's pins as shell assignments and exit")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="the checkout to read the commit from")
    parser.add_argument("--source-revision", default="HEAD", help="the commit whose pins are the expectation (default HEAD)")
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=Path(os.environ.get("DEXBENCH_DATASETS_ROOT", Path.home() / "datasets")).expanduser(),
    )
    parser.add_argument("--image")
    parser.add_argument("--dockerfile", choices=(DOCKERFILE, EXTERNAL_DOCKERFILE),
                        help="recipe in the committed tree; default: docker/Dockerfile")
    parser.add_argument(
        "--cloudxr-dir",
        type=Path,
        default=Path(os.environ.get("CXR_HOST_VOLUME_PATH", Path.home() / ".cloudxr")),
    )
    args = parser.parse_args()

    root = args.root.resolve()
    pins = derive_pins(root, args.source_revision, dockerfile=args.dockerfile)
    if args.print_pins:
        sys.stdout.write(shell_pins(pins))
        return 0
    if pins["source"] is None:
        print(f"{root} has no commit {args.source_revision}; the pins are the commit, so there is nothing to check against", file=sys.stderr)
        return 1

    image = args.image or (f"{IMAGE_NAME}:{pins['image_tag']}" if pins["image_tag"] else None)
    checks = Checks()
    if args.check in ("host", "all"):
        check_host(checks, pins, root, args.datasets_root)

    image_info = None
    if args.check in ("image", "cloudxr", "all"):
        if image is None:
            checks.fail("no image to check: pass --image or declare DEXBENCH_IMAGE_TAG in the Dockerfile")
        else:
            image_info = check_image(checks, pins, image)

    if args.check in ("cloudxr", "all"):
        check_cloudxr(checks, pins, args.cloudxr_dir, image_info)

    if args.profile is not None:
        card = json.loads(args.profile.read_text(encoding="utf-8"))
        if image_info is None and image is not None and args.check == "host":
            image_info = _docker_inspect(image)
        check_profile(checks, pins, card, image_info, root=args.root)

    return checks.finish()


if __name__ == "__main__":
    raise SystemExit(main())
