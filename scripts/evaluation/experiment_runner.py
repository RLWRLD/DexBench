#!/usr/bin/env python3
# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Run an Arena experiment config (multi-run evaluation) with DexBench environments.

Wraps upstream's ``experiment_runner`` the same way ``policy_runner.py`` wraps the
single-run entrypoint, but by importing the module rather than ``runpy``: upstream's
``--chunk_size`` re-spawns ``[sys.executable, *sys.argv]`` per chunk, and only an
``argv[0]`` that still names this wrapper makes each child register DexBench's
environments with Arena's CLI again before it looks the environment up. Use it with a ``{"jobs": [...]}`` JSON
or one of the typed ``*_variations.yaml`` experiments in
``scripts/evaluation/experiment_configs/`` (the YAMLs load but do not chunk)::

    /isaac-sim/python.sh scripts/evaluation/experiment_runner.py --viz none \\
        --experiment_config scripts/evaluation/experiment_configs/benchmark/db_r3_sensitivity_conditions.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "imitation_learning"))

from _dexbench_arena_script import run_upstream_evaluation_module  # noqa: E402
from dexbench_arena.evaluation.experiment_backend import refuse_experiment_mimic_runs  # noqa: E402


if __name__ == "__main__":
    refuse_experiment_mimic_runs(sys.argv[1:])
    run_upstream_evaluation_module(
        "isaaclab_arena.evaluation.experiment_runner",
        wrapper_path=str(Path(__file__).resolve()),
    )
