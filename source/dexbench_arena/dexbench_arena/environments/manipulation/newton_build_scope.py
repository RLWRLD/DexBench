# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Own task-specific Newton builder requests for one composed environment."""

from collections.abc import Callable

_CLEANUPS: list[Callable[[], None]] = []


def begin_newton_build() -> None:
    """Remove the previous build's opted-in requests before composing another."""
    pending = tuple(_CLEANUPS)
    _CLEANUPS.clear()
    for cleanup in pending:
        cleanup()


def register_build_cleanup(cleanup: Callable[[], None]) -> None:
    """Retain an idempotent request cleanup until the next build, not each reset."""
    if cleanup not in _CLEANUPS:
        _CLEANUPS.append(cleanup)
