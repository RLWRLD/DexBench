"""A checkout that downloads its assets: the lock, the command an environment names, and the snapshot check."""
from __future__ import annotations

import hashlib
import json

import pytest

from dexbench_arena.assets import snapshot

TREE = "1" * 40
LOCK = {"repo_id": "RLWRLD/DexBench-assets", "revision": "2" * 40, "source_assets_tree": TREE}


def write_snapshot(assets, files, *, tree=TREE, repository="RLWRLD/DexBench-assets"):
    """``files`` (path -> bytes) under ``assets``, with the receipt an upload writes for them."""
    entries = []
    for relative, body in files.items():
        (assets / relative).parent.mkdir(parents=True, exist_ok=True)
        (assets / relative).write_bytes(body)
        entries.append({"path": relative, "size": len(body), "sha256": hashlib.sha256(body).hexdigest()})
    receipt = {"repository": repository, "source_assets_tree": tree, "files": entries}
    (assets / snapshot.RECEIPT).write_text(json.dumps(receipt))
    return receipt


def test_the_lock_must_name_a_dataset_its_commit_and_the_tree_it_was_uploaded_from():
    assert snapshot.lock_problems(LOCK) == []
    assert snapshot.parse_lock(json.dumps(LOCK)) == LOCK
    for broken in ({**LOCK, "revision": "main"}, {**LOCK, "source_assets_tree": None}, {**LOCK, "repo_id": "assets"}):
        with pytest.raises(ValueError):
            snapshot.parse_lock(json.dumps(broken))
    assert snapshot.locked_assets_tree(json.dumps(LOCK)) == TREE
    assert snapshot.locked_assets_tree("unknown") is None and snapshot.locked_assets_tree(None) is None


def test_an_environment_names_the_command_of_its_checkouts_layout(tmp_path):
    assert snapshot.restore_command(tmp_path) == 'git lfs pull --include="assets/**" --exclude=""'
    assert snapshot.restore_command(tmp_path, "env/e4/**") == 'git lfs pull --include="assets/env/e4/**" --exclude=""'
    assert snapshot.read_lock(tmp_path) is None
    (tmp_path / "reproducibility").mkdir()
    (tmp_path / snapshot.ASSETS_LOCK).write_text(json.dumps(LOCK))
    assert snapshot.read_lock(tmp_path) == LOCK
    assert snapshot.restore_command(tmp_path) == "./scripts/download_assets.sh"
    assert snapshot.restore_command(tmp_path, "env/e4/**") == './scripts/download_assets.sh --include "env/e4/**"'


def test_the_snapshot_check_names_each_file_that_is_not_the_snapshots(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    assert snapshot.snapshot_problems(assets, LOCK) == [
        f"assets/{snapshot.RECEIPT} is missing or unreadable: the download did not finish"]
    write_snapshot(assets, {"env/scene.usd": b"scene", "object_library/cup.usd": b"cup", "README.md": b"card"})
    (assets / ".cache/huggingface").mkdir(parents=True)
    (assets / ".cache/huggingface/state").write_text("hf download keeps this")
    assert snapshot.snapshot_problems(assets, LOCK) == []
    (assets / "env/scene.usd").write_bytes(b"other")
    (assets / "object_library/cup.usd").unlink()
    (assets / "env/old.usd").write_bytes(b"left by an earlier snapshot")
    assert snapshot.snapshot_problems(assets, LOCK) == [
        "env/scene.usd: SHA-256 differs from the snapshot's",
        "object_library/cup.usd: missing",
        "env/old.usd: not in the snapshot",
    ]
    assert snapshot.snapshot_problems(assets, LOCK, ["env/**"]) == ["env/scene.usd: SHA-256 differs from the snapshot's"]
    assert snapshot.snapshot_problems(assets, LOCK, ["README.md"]) == []


def test_a_partial_download_also_installs_and_checks_the_files_stored_in_git(tmp_path):
    """The files that the snapshot stores in Git (the scene index, the reference views) are read for every asset, so a
    partial download installs them and its check covers them."""
    receipt = write_snapshot(tmp_path, {"env/scene_bundle_index.json": b"{}", "env/views/cup.usda": b"#usda 1.0",
                                        "object_library/cup.usd": b"cup", "env/scene.usdz": b"scene"})
    for entry in receipt["files"]:
        entry["storage"] = "lfs" if entry["path"].endswith((".usd", ".usdz")) else "git"
    (tmp_path / snapshot.RECEIPT).write_text(json.dumps(receipt))
    assert snapshot.shared_files(receipt) == ["env/scene_bundle_index.json", "env/views/cup.usda"]
    assert [e["path"] for e in snapshot.selected_files(receipt, ["object_library/**"])] == [
        "env/scene_bundle_index.json", "env/views/cup.usda", "object_library/cup.usd"]
    assert len(snapshot.selected_files(receipt)) == 4
    (tmp_path / "env/scene_bundle_index.json").unlink()
    assert snapshot.snapshot_problems(tmp_path, LOCK, ["object_library/**"]) == [
        "env/scene_bundle_index.json: missing"]


def test_the_snapshot_must_be_the_one_the_lock_pins(tmp_path):
    write_snapshot(tmp_path, {"a.usd": b"a"}, tree="3" * 40, repository="someone/else")
    problems = snapshot.snapshot_problems(tmp_path, LOCK)
    assert problems[0] == "assets/ is a snapshot of 'someone/else', not RLWRLD/DexBench-assets"
    assert problems[1].startswith(f"assets/ was uploaded from assets tree {'3' * 40}, not {TREE}")


def test_a_submission_finds_an_asset_changed_or_added_after_a_partial_download(tmp_path):
    """Git does not track the downloaded assets, so ``git status`` does not see a change to them: submission
    preparation checks the files that are there instead."""
    assert snapshot.checkout_problems(tmp_path) == []
    (tmp_path / "reproducibility").mkdir()
    (tmp_path / snapshot.ASSETS_LOCK).write_text(json.dumps(LOCK))
    assets = tmp_path / "assets"
    write_snapshot(assets, {"env/scene.usd": b"scene", "object_library/cup.usd": b"cup"})
    (assets / "object_library/cup.usd").unlink()
    assert snapshot.checkout_problems(tmp_path) == []
    (assets / "env/scene.usd").write_bytes(b"other")
    (assets / "env/extra.usd").write_bytes(b"added")
    assert snapshot.checkout_problems(tmp_path) == [
        "env/scene.usd: SHA-256 differs from the snapshot's", "env/extra.usd: not in the snapshot"]
