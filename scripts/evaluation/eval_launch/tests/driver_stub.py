# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Test stand-ins for the driver's tests: a cell runner that does what a test plan says, and the hook that puts it (and a
unit check that reads what it wrote) in place of the real ones in every slot and server process (Ray's
``worker_process_setup_hook``, the local backend's ``backend.local.setup_hook``), optionally with a skewed clock.

The cell runner is started by ``eval_worker.run_unit`` exactly as ``run_benchmark_cell.py`` would be, as its own Kit
stand-in process. ``EVAL_TEST_PLAN`` (a JSON file) maps a unit id to what its attempts do, in order (the attempt is
the count of earlier attempt directories of the unit; past the list, ``ok``):

- ``ok``, ``slow:<s>``: write the run manifest and a completion mark, and what the process saw (its GPU, its
  arguments, whether it got the server's token); exit 0.
- ``fail``: write the run manifest, exit 1.
- ``slow_fail:<s>``: write the run manifest, sleep ``s`` seconds, exit 1: a unit that fails after the others ended.
- ``hang``: write the run manifest and sleep (the unit's deadline stops it).
- ``kill_actor``: SIGKILL the slot process that started it (its parent: a Ray actor, a local slot), then sleep: a
  slot lost mid-unit, with its Kit left.
- ``freeze_actor``: SIGSTOP the slot process, then sleep: a slot that never answers (the driver revokes it).
- ``refuse``: exit 2 without a run manifest (the cell runner refusing the tree).
- ``crash_after:<k>``: write the first ``k`` episodes of its slices as Arena does (``fake_cell``), then exit 1 with the
  next episode's videos cut off: a simulator crash mid-unit.
- ``kill_after:<k>``: the same episodes, then SIGKILL the slot process that started it: a slot lost mid-unit after
  finishing some.
- ``timed:<slowdown>``: as ``ok``, and the step timings and control rate a unit of that slowdown writes (600 steps
  at 60 Hz), which the worker reads into its outcome.

``EVAL_TEST_GPU_MEMORY`` (``used,total`` MiB), when set, is what every slot reads of its GPU's memory
(``gpu_memory.query``), where the test host has no such GPU; ``none`` reads nothing, as without ``nvidia-smi``.

Not a test module: pytest collects ``test_*.py`` only.
"""

from __future__ import annotations

import json
import os
import random
import signal
import sys
import time
from pathlib import Path
from typing import Any

from isaaclab_arena.evaluation.arena_experiment_result import ARENA_EXPERIMENT_TIMINGS_FILENAME

from eval_launch import eval_worker, gpu_memory
from eval_launch.tests import fake_cell

STUB = Path(__file__).resolve()
COMPLETE = "stub_complete"
SEEN = "stub_seen.json"


def check_unit(unit_dir: str | Path, slices: Any) -> dict[str, Any]:
    """The unit check's verdict on what the stand-in wrote."""
    complete = (Path(unit_dir) / COMPLETE).is_file()
    return {"complete": complete, "episodes": 1 if complete else 0, "expected": 1, "cameras": [], "instructions": [],
            "problems": [] if complete else ["no completion mark"]}


def setup() -> None:
    """The setup hook: every slot and server process runs the stand-ins; with ``EVAL_TEST_SKEW``, its wall clock is off
    by up to 30 days either way and jumps on every read."""
    eval_worker.RUN_CELL = STUB
    eval_worker.check_unit = check_unit
    if os.environ.get("EVAL_TEST_GPU_MEMORY") == "none":
        gpu_memory.query = lambda gpu: None
    elif os.environ.get("EVAL_TEST_GPU_MEMORY"):
        used, total = (float(value) for value in os.environ["EVAL_TEST_GPU_MEMORY"].split(","))
        gpu_memory.query = lambda gpu: (used, total)
    if os.environ.get("EVAL_TEST_SKEW"):
        skew_clock(random.Random(os.getpid()).uniform(-30 * 86400, 30 * 86400))


def skew_clock(offset: float) -> None:
    real = time.time
    jumps = random.Random(os.getpid() + 1)
    time.time = lambda: real() + offset + jumps.uniform(-3600, 3600)


def _argument(name: str) -> str:
    return sys.argv[sys.argv.index(name) + 1]


def main() -> int:
    output = Path(_argument("--output"))
    unit_id = output.parent.name
    earlier = [path for path in output.parent.iterdir() if path.is_dir() and path != output] if output.parent.is_dir() else []
    plan = json.loads(Path(os.environ["EVAL_TEST_PLAN"]).read_text())
    steps = plan.get(unit_id, [])
    step = steps[len(earlier)] if len(earlier) < len(steps) else "ok"
    if step == "refuse":
        sys.exit("Refusal: rows not under the tree")
    output.mkdir(parents=True)
    (output / "run_manifest.json").write_text("{}")
    if step.startswith("slow_fail:"):
        time.sleep(float(step.split(":", 1)[1]))
    if step == "fail" or step.startswith("slow_fail:"):
        sys.exit("simulator state is not finite")
    if step == "hang":
        time.sleep(3600)
    if step.startswith(("crash_after:", "kill_after:")):
        fake_cell.write_attempt(output, json.loads(_argument("--slices")), stop=int(step.split(":", 1)[1]))
        (output / SEEN).write_text(json.dumps({"argv": sys.argv[1:], "attempt": output.name}))
        if step.startswith("kill_after:"):
            os.kill(os.getppid(), signal.SIGKILL)
            time.sleep(3600)
        sys.exit("RuntimeError: Warp CUDA error 700: an illegal memory access was encountered")
    if step in ("kill_actor", "freeze_actor"):
        os.kill(os.getppid(), signal.SIGKILL if step == "kill_actor" else signal.SIGSTOP)
        time.sleep(3600)
    if step.startswith("slow:"):
        time.sleep(float(step.split(":", 1)[1]))
    if step.startswith("timed:"):
        (output / "arena").mkdir()
        (output / "arena" / ARENA_EXPERIMENT_TIMINGS_FILENAME).write_text(json.dumps(
            {"totals": [{"name": "step", "total_ms": float(step.split(":", 1)[1]) * 10_000.0, "count": 600}]}))
        (output / "composed_experiment.yaml").write_text(
            "shared: {environment: {rby1_control_hz: 30}}\nruns: {r: {environment: {rby1_control_hz: 60}}}\n")
    seen = {"gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "token": bool(os.environ.get("DEXBENCH_POLICY_API_TOKEN")),
            "argv": sys.argv[1:], "attempt": output.name, "pythonpath": os.environ.get("PYTHONPATH", "")}
    (output / SEEN).write_text(json.dumps(seen))
    stamp = time.time() + random.uniform(-30 * 86400, 30 * 86400)
    (output / COMPLETE).write_text("ok")
    os.utime(output / COMPLETE, (stamp, stamp))
    return 0


if __name__ == "__main__":
    sys.exit(main())
