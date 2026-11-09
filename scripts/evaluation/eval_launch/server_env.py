# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""The policy servers' environment, handed from one process to the next through a file that lives only until it is read.

The servers must start in the environment the launch had before ``/isaac-sim/python.sh`` added Isaac Sim's paths to it
(``eval_worker.PolicyServer``), so it is saved with ``env -0`` first: by ``launch.sh`` on a host, by the Job's script in
a pod. That environment holds whatever the shell or the pod holds, tokens and ``secret_env`` values included, so the
file is created readable by its owner only and removed by the process that reads it: ``launch.sh`` writes none (it
hands the launcher a pipe, ``/dev/fd/3``), and the launcher writes each
local worker a file of its own, which the worker reads and removes as it starts; a driver it execs gets a descriptor
too (:func:`handover`), so a local driver (``--local``) leaves nothing even when killed. A directory-backend worker
killed between the write and the read (SIGKILL) leaves its file until the launcher that started it returns.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path


def read(path: str | Path) -> dict[str, str]:
    """The environment saved in ``path`` (``env -0``), which is removed once read; ``/dev/fd/<n>``, a descriptor (launch.sh's
    pipe, or :func:`handover`'s file that is already gone), is closed instead."""
    path = Path(path)
    try:
        return dict(item.split("=", 1) for item in path.read_text().split("\0") if "=" in item)
    finally:
        if path.parent == Path("/dev/fd"):
            with contextlib.suppress(OSError, ValueError):
                os.close(int(path.name))
        else:
            path.unlink(missing_ok=True)


def write(environment: Mapping[str, str]) -> Path:
    """A new file holding ``environment`` as ``env -0`` writes it, readable by its owner only, for one reader."""
    descriptor, name = tempfile.mkstemp(prefix="eval-server-env.")
    with os.fdopen(descriptor, "w") as handle:
        handle.write("".join(f"{key}={value}\0" for key, value in environment.items()))
    return Path(name)


def handover(environment: Mapping[str, str]) -> str:
    """``environment`` for a process this one execs into: an inheritable descriptor (``/dev/fd/<n>``) of a file that is
    already removed, so a process that fails before it reads it (e.g. a driver whose Ray import fails) leaves nothing
    on disk. :func:`read` closes it."""
    path = write(environment)
    try:
        descriptor = os.open(path, os.O_RDONLY)
    finally:
        path.unlink(missing_ok=True)
    os.set_inheritable(descriptor, True)
    return f"/dev/fd/{descriptor}"
