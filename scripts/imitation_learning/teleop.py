#!/usr/bin/env python3
# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Run teleoperation with DexBench-native environments.

Routed to Isaac Lab's ``teleop_se3_agent.py``, which drives the IsaacTeleop
(OpenXR) stack; Arena has no teleop runner of its own (IsaacLab-Arena#1022, 4a97a0cce).
The wrapper translates the DexBench CLI for it.
``--collection_rows <rows.jsonl>`` starts every take from one row of a recorded start-state
dump (``--collection_first_row``, default 0), as record_demos.py does; teleop keeps no demos,
so the row never moves on.
"""

from __future__ import annotations

from _dexbench_arena_script import run_upstream_imitation_script


if __name__ == "__main__":
    run_upstream_imitation_script("teleop.py")
