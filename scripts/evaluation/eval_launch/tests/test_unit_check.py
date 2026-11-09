# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""A unit is a result only when each of its Runs completed with exactly its slice's episodes, each replaying its own
row of the slice with a success value, as the scorer requires, and on video from every camera the unit recorded,
every camera rendering."""

from __future__ import annotations

import json

import numpy as np
import pytest

av = pytest.importorskip("av", reason="unit_check decodes the unit's videos with PyAV")
pytest.importorskip("isaaclab_arena.visualization.episode_results_files", reason="needs IsaacLab-Arena on the path")

from eval_launch.unit_check import check_unit  # noqa: E402

CAMERAS = ("head", "wrist")


def _video(path, blank=False):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=4)
        stream.width, stream.height, stream.pix_fmt = 32, 32, "yuv420p"
        rng = np.random.default_rng(0)
        for _ in range(8):
            pixels = np.zeros((32, 32, 3), np.uint8) if blank else rng.integers(0, 255, (32, 32, 3), np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _unit(tmp_path, episodes, status="completed", cameras=CAMERAS, blank=(), success=False, instruction=None):
    """A unit directory with ``episodes`` = {run: [replayed row within the slice, ...]}, each instructed
    ``instruction`` (Arena's episode results record what the environment sent)."""
    arena = tmp_path / "arena"
    runs = {}
    for run, rows in episodes.items():
        (arena / run).mkdir(parents=True)
        lines = [json.dumps({"env_id": 0, "episode_in_env": i, "success": success, "replay_source_record_index": row,
                             "language_instruction": instruction})
                 for i, row in enumerate(rows)]
        (arena / run / "episode_results_rebuild0.jsonl").write_text("\n".join(lines) + "\n")
        for i in range(len(rows)):
            for camera in cameras:
                _video(arena / run / f"robot-cam-rebuild0-env0-{camera}-episode-{i}.mp4", blank=(run, i, camera) in blank)
        runs[run] = {"status": status, "rebuilds": []}
    (arena / "arena_experiment_result.json").write_text(json.dumps({"runs": runs}))
    return tmp_path


def test_a_complete_unit_passes(tmp_path):
    result = check_unit(_unit(tmp_path, {"a": [1, 0], "b": [0]}), {"a": [0, 2], "b": [5, 6]})
    assert result["complete"], result["problems"]
    assert result["episodes"] == result["expected"] == 3 and result["cameras"] == list(CAMERAS)


@pytest.mark.parametrize("episodes, status, cameras, blank, success, problem", [
    ({"a": [0, 1, 2], "b": []}, "completed", CAMERAS, (), False, "b: 0 episodes"),
    ({"a": [0, 0], "b": [0]}, "completed", CAMERAS, (), False, "1 distinct rows"),
    ({"a": [0, 1], "b": [5]}, "completed", CAMERAS, (), False, "expected rows 0..0"),
    ({"a": [0, 1], "b": [0]}, "completed", CAMERAS, (), None, "record no success"),
    ({"a": [0, 1], "b": [0]}, "failed", CAMERAS, (), False, "Arena status 'failed'"),
    ({"a": [0, 1], "b": [0]}, "completed", CAMERAS, (("b", 0, "wrist"),), False, "wrist: 2/2 blank"),
], ids=["wrong-run", "duplicate-row", "row-outside-slice", "no-success", "run-failed", "camera-never-renders"])
def test_an_incomplete_unit_fails_with_the_reason(tmp_path, episodes, status, cameras, blank, success, problem):
    result = check_unit(_unit(tmp_path, episodes, status, cameras, blank, success), {"a": [0, 2], "b": [5, 6]})
    assert not result["complete"] and any(problem in line for line in result["problems"]), result["problems"]


def test_an_episode_missing_one_camera_fails(tmp_path):
    unit = _unit(tmp_path, {"a": [0, 1]})
    next((tmp_path / "arena" / "a").glob("*wrist-episode-1.mp4")).unlink()
    result = check_unit(unit, {"a": [0, 2]})
    assert not result["complete"] and any("no video from ['wrist']" in line for line in result["problems"])


def test_the_units_instructions_are_reported_and_never_a_problem(tmp_path):
    result = check_unit(_unit(tmp_path, {"a": [0, 1]}, instruction="Pick the can."), {"a": [0, 2]})
    assert result["complete"] and result["instructions"] == ["Pick the can."]


def test_an_episode_that_sent_no_instruction_reports_none(tmp_path):
    """A missing instruction is left out of the report, not reported as the prompt 'None'."""
    result = check_unit(_unit(tmp_path, {"a": [0, 1]}), {"a": [0, 2]})
    assert result["complete"] and result["instructions"] == []
