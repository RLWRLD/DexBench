# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""The paper demo recorder writes physics replay commands only where the checkout has the replay script."""

from __future__ import annotations

import pytest

from dexbench_arena.demos import paper_demo_hdf5


def test_no_replay_commands_without_the_replay_script(monkeypatch, tmp_path):
    monkeypatch.setattr(paper_demo_hdf5, "REPO_ROOT", tmp_path)
    assert paper_demo_hdf5.replay_commands(tmp_path / "outputs/demo/episode.hdf5", {}) is None


def test_replay_commands_run_the_replay_script_where_the_checkout_has_it():
    root = paper_demo_hdf5.REPO_ROOT
    if not (root / paper_demo_hdf5.REPLAY_SCRIPT).is_file():
        pytest.skip("this checkout has no replay script")
    commands = paper_demo_hdf5.replay_commands(root / "outputs/demo/episode.hdf5", {})
    assert paper_demo_hdf5.REPLAY_SCRIPT in commands["container"]
    assert commands["output"] == "outputs/demo/replay/episode"
