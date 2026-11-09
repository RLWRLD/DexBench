# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Where a model's run of a cell keeps what ran, and how the scorer reads it (Kit-free, dependency-free)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dexbench_arena.benchmark import Refusal

COMPOSED = "composed_experiment.yaml"
RUN_MANIFEST = "run_manifest.json"
ARENA_OUTPUT = "arena"
"""The experiment runner's ``--experiment_output_directory`` inside the run's directory (it must start empty)."""
RUN_MANIFEST_VERSION = 1


def read_run_manifest(directory: str | Path) -> dict[str, Any]:
    """A run directory's manifest; refuses a directory without one (the baseline, run as it is, is never scored)."""
    path = Path(directory) / RUN_MANIFEST
    if not path.is_file():
        raise Refusal(f"{directory}: no {RUN_MANIFEST}; a scored run is a model's, made by run_benchmark_cell.py "
                      "(the cell file run as it is is the baseline, a pipeline check that is never scored)")
    try:
        run_manifest = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: not valid JSON ({error})") from error
    if run_manifest.get("version") != RUN_MANIFEST_VERSION:
        raise Refusal(f"{path}: run manifest version {run_manifest.get('version')!r}, not {RUN_MANIFEST_VERSION}")
    return run_manifest


__all__ = ["ARENA_OUTPUT", "COMPOSED", "RUN_MANIFEST", "RUN_MANIFEST_VERSION", "read_run_manifest"]
