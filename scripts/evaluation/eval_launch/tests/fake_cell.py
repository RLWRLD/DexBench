# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""A cell runner that writes what Arena writes, episode by episode, and can crash mid-unit: for the salvage tests.

Started by ``eval_worker.run_unit`` in place of ``run_benchmark_cell.py`` (``RUN_CELL``), it writes the run manifest and
the composed file, then replays its ``--slices`` in order (Runs in the slices' order, rows ascending): for each episode
an Arena result line (``env_id``, ``episode_in_env``, ``success``, ``replay_source_record_index`` counted within the
slice, ``episode_length``) and an mp4 per camera of ``episode_length - 1`` frames, as Arena's recorder leaves them. What
each attempt does comes from ``EVAL_TEST_PLAN`` (a JSON file: unit id -> a step per attempt, the attempt being the count
of the unit's earlier attempt directories; past the list, ``ok``):

- ``ok``: every row, the experiment result marking each Run completed; exit 0.
- ``crash_after:<k>``: the first ``k`` rows, then the next row's videos cut off mid-write; exit 1 (Warp's CUDA error).
- ``crash_at:<run>:<row>``: every row before that one (``row`` counted as the unit's slices count), then the same crash.

Also usable in-process (:func:`write_attempt`). Not a test module: pytest collects ``test_*.py`` only.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import av
import numpy as np

CAMERAS = ("head", "wrist")
FRAMES = 5
"""Frames per episode video: the episode lasts ``FRAMES + 1`` steps (Arena records no frame of the step that ends it)."""
SEEN = "fake_seen.json"


def video(path: Path, frames: int = FRAMES, cut: bool = False) -> None:
    """An mp4 of ``frames`` noise frames; ``cut``: only its first half, as an encoder stopped mid-file leaves it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=4)
        stream.width, stream.height, stream.pix_fmt = 32, 32, "yuv420p"
        rng = np.random.default_rng(len(path.name))
        for _ in range(frames):
            for packet in stream.encode(av.VideoFrame.from_ndarray(rng.integers(0, 255, (32, 32, 3), np.uint8), format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    if cut:
        data = path.read_bytes()
        path.write_bytes(data[: len(data) // 2])


def episode(arena: Path, run: str, episode_in_env: int, row: int, success: bool = True, frames: int = FRAMES) -> None:
    """One finished episode of ``run``: its result line and its videos."""
    for camera in CAMERAS:
        video(arena / run / f"robot-cam-rebuild0-env0-{camera}-episode-{episode_in_env}.mp4", frames)
    with (arena / run / "episode_results_rebuild0.jsonl").open("a") as results:
        results.write(json.dumps({"env_id": 0, "episode_in_env": episode_in_env, "success": success,
                                  "replay_source_record_index": row, "episode_length": frames + 1}) + "\n")


def write_attempt(unit_dir: Path, slices: dict[str, list[int]], stop: int | None = None) -> None:
    """What an attempt of ``slices`` leaves: every row (``stop`` None, with the experiment result), or the first ``stop``
    rows and the next one's videos cut off (no experiment result: the runner died)."""
    arena = unit_dir / "arena"
    unit_dir.mkdir(parents=True, exist_ok=True)
    (unit_dir / "run_manifest.json").write_text(json.dumps({"slices": slices}))
    (unit_dir / "composed_experiment.yaml").write_text(json.dumps({"runs": {run: {} for run in slices}}))
    done = 0
    for run, (start, end) in slices.items():
        (arena / run).mkdir(parents=True, exist_ok=True)
        (arena / run / "episode_results_rebuild0.jsonl").touch()
        for index in range(end - start):
            if stop is not None and done == stop:
                for camera in CAMERAS:
                    video(arena / run / f"robot-cam-rebuild0-env0-{camera}-episode-{index}.mp4", cut=True)
                return
            episode(arena, run, index, index, success=(start + index) % 2 == 0)
            done += 1
    if stop is None:
        (arena / "arena_experiment_result.json").write_text(json.dumps(
            {"runs": {run: {"status": "completed", "rebuilds": []} for run in slices}}))


def _rows_before(slices: dict[str, list[int]], run: str, row: int) -> int:
    count = 0
    for name, (start, end) in slices.items():
        for index in range(start, end):
            if (name, index) == (run, row):
                return count
            count += 1
    return count


def main() -> int:
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    slices = json.loads(sys.argv[sys.argv.index("--slices") + 1])
    earlier = [path for path in output.parent.iterdir() if path.is_dir() and path != output] if output.parent.is_dir() else []
    steps: list[str] = json.loads(Path(os.environ["EVAL_TEST_PLAN"]).read_text()).get(output.parent.name, [])
    step = steps[len(earlier)] if len(earlier) < len(steps) else "ok"
    stop: int | None = None
    if step.startswith("crash_after:"):
        stop = int(step.split(":", 1)[1])
    elif step.startswith("crash_at:"):
        _, run, row = step.split(":")
        stop = _rows_before(slices, run, int(row))
    write_attempt(output, slices, stop)
    (output / SEEN).write_text(json.dumps({"slices": slices, "step": step}))
    if stop is not None:
        sys.exit("RuntimeError: Error launching kernel: Warp CUDA error 700: an illegal memory access was encountered")
    return 0


def plan(path: Path, steps: dict[str, Any]) -> Path:
    path.write_text(json.dumps(steps))
    return path


if __name__ == "__main__":
    sys.exit(main())
