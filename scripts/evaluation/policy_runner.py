#!/usr/bin/env python3
# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Run Arena policy evaluation with DexBench-native environments."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "imitation_learning"))

from _dexbench_arena_script import run_upstream_evaluation_script  # noqa: E402


if __name__ == "__main__":
    from isaaclab_arena.evaluation import policy_runner_cli

    from dexbench_arena.evaluation.policy_runner_cli import add_policy_cli_args

    policy_runner_cli.add_policy_cli_args = add_policy_cli_args
    run_upstream_evaluation_script("policy_runner.py")
