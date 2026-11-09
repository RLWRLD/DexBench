# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""The DexBench benchmark protocol: cells, caps, models and the per-task declarations."""

BENCHMARK_CONFIGS = "scripts/evaluation/experiment_configs/benchmark"
"""Where the generated files live, relative to the repository: nothing else is in that directory, so the generator owns
all of it. Here rather than in ``expand`` so that Kit-free readers (the eval launcher) need not import the environments."""
MANIFEST = "benchmark_manifest.json"


class Refusal(ValueError):
    """A benchmark check refused its input; the message names what was refused and why."""

