# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""The eval launcher: plans a benchmark evaluation, runs it on one host, on Ray or as Kubernetes Jobs, resumes a
stopped launch and scores it. ``launch.sh`` runs it (``python -m eval_launch``, with ``scripts/evaluation`` on the
path).
"""

import os
from collections.abc import Callable
from importlib.resources import files
from pathlib import Path
from typing import Any, NoReturn

PACKAGE_DIR = Path(str(files(__name__)))
"""This package's directory: its data (``timings/``, ``examples/``) and scripts (``launch.sh``, ``signal_group.sh``)."""
STATUS_VARIABLE = "EVAL_LAUNCH_STATUS"
"""A file ``launch.sh`` names for the exit status: ``/isaac-sim/python.sh`` turns every non-zero status of the Python it
runs into 1 (``... || error_exit``), so the launcher's 2, 3, 75 and 130 reach ``launch.sh``'s caller only this way."""


_status: str | None = None
"""The status file this process writes (:func:`run_main` takes it out of the environment, so no child inherits it)."""


def status_file() -> str | None:
    """The file :data:`STATUS_VARIABLE` named for this process, for the one hop that hands it on: the launcher's exec
    into its driver."""
    return _status


def _write_status(number: int) -> None:
    if _status:
        try:
            Path(_status).write_text(f"{number}\n")
        except OSError:
            pass


def run_main(main: Callable[[], Any]) -> NoReturn:
    """Exit with what ``main()`` returns (or the ``SystemExit`` it raises), writing that status to the file
    :data:`STATUS_VARIABLE` names, if any. The variable is taken out of this process's environment first, so a child
    (the smoke's driver, a worker) cannot write the file; and 1 is written before ``main()`` runs, so an uncaught
    exception leaves 1 there."""
    global _status
    _status = os.environ.pop(STATUS_VARIABLE, None)
    _write_status(1)
    try:
        code = main()
    except SystemExit as stop:
        code = stop.code
    _write_status(code if isinstance(code, int) else (0 if code is None else 1))
    raise SystemExit(code)
