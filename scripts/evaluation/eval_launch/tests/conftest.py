"""Every test's launcher runs on a stand-in of a staged tree: every row file and runtime input the manifest names,
empty (the planner checks that they are there, ``plan.check_inputs``; nothing here reads them). A test of that check
points ``launch.TREE`` elsewhere."""

from __future__ import annotations

import json

import pytest

from eval_launch import launch, plan


@pytest.fixture(scope="session")
def staged_tree(tmp_path_factory):
    root = tmp_path_factory.mktemp("staged_tree")
    manifest = json.loads(plan.MANIFEST.read_text())
    paths = {row["path"] for entry in manifest["tasks"].values() for cell in entry["cells"].values()
             for row in cell["rows"].values()}
    paths |= {path for entry in manifest["tasks"].values() for path in entry.get("runtime_inputs") or {}}
    for path in paths:
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).touch()
    return root


@pytest.fixture(autouse=True)
def _staged_tree(staged_tree, monkeypatch):
    monkeypatch.setattr(launch, "TREE", staged_tree)
