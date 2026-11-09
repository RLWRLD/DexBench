# Copyright (c) 2026, RLWRLD Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures, and the split between the tests that need Kit and the rest."""

from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

KIT_MARKER = "kit"
KIT_HINT = "needs a running Kit: run these with `/isaac-sim/python.sh -m pytest --kit source/dexbench_arena/tests`"


@pytest.fixture(scope="session")
def git_checkout():
    """Require this source tree to own its Git metadata, including in linked worktrees."""
    root = Path(__file__).resolve().parents[3]
    try:
        top = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout (e.g. exported tree)")
    if Path(top).resolve() != root:
        pytest.skip("exported tree inside another git checkout")


def pytest_addoption(parser):
    parser.addoption(
        "--kit",
        action="store_true",
        default=False,
        help="Start a headless Kit before collection and run only the tests that need it.",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", f"{KIT_MARKER}: needs a running Kit (uses the session-wide simulation_app fixture)"
    )


def pytest_sessionstart(session):
    """With --kit, bring Kit up before collection imports pxr or omni anywhere."""

    if session.config.getoption("--kit"):
        _start_kit()


def pytest_report_header(config):
    return "kit tests: " + ("running (Kit started at session start)" if config.getoption("--kit") else "deselected, pass --kit")


def pytest_collection_modifyitems(config, items):
    """Mark the Kit users; with --kit keep only them, without it skip them."""

    kit_items = []
    for item in items:
        if "simulation_app" in getattr(item, "fixturenames", ()):
            item.add_marker(pytest.mark.kit)
            kit_items.append(item)
    if config.getoption("--kit"):
        deselected = [item for item in items if item not in set(kit_items)]
        if deselected:
            config.hook.pytest_deselected(items=deselected)
            items[:] = kit_items
    else:
        for item in kit_items:
            item.add_marker(pytest.mark.skip(reason=KIT_HINT))


def pytest_sessionfinish(session, exitstatus):
    """Feed pytest's verdict to Arena's shutdown hook, which otherwise exits 0."""

    if not session.config.getoption("--kit"):
        return
    try:
        from isaaclab_arena.tests.utils import subprocess as arena_subprocess
    except ImportError:
        return
    arena_subprocess._AT_LEAST_ONE_TEST_FAILED = exitstatus != pytest.ExitCode.OK


def _start_kit():
    """One headless Kit for the whole session, through Arena's own test utility."""

    pytest.importorskip("isaaclab.app", reason="--kit needs the Isaac Sim runtime")
    from isaaclab_arena.tests.utils.persistent_simulation_app import get_persistent_simulation_app

    return get_persistent_simulation_app(headless=True)


@pytest.fixture(scope="session")
def simulation_app(request):
    """The session's Kit. Without --kit the test is skipped before getting here."""

    if not request.config.getoption("--kit"):
        pytest.skip(KIT_HINT)
    return _start_kit()
