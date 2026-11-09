# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Whether a finished unit is a result: every Run it asked for completed, with exactly the episodes its slice names,
each on video from every camera, and every camera rendering.

Per Run (``<unit_dir>/arena/<run>/``, read with Arena's own result and video-name helpers):

- Arena's experiment result (``arena/arena_experiment_result.json``) marks the Run ``completed``;
- its episode results hold exactly ``end - start`` episodes for the slice ``[start, end)``, replaying the slice's rows
  ``0 .. end - start - 1`` once each (``replay_source_record_index`` counts within the slice), each with a ``success``
  value, and no unreadable line: what the scorer refuses otherwise, so such a unit is retried rather than accepted
  and refused only at ``--score``;
- each of those episodes has a video from every camera the unit recorded, and each video renders.

The cameras are the ones the unit's own videos show (the cell's environment decides them, not the launcher), so an
episode missing one camera is caught; a cell that rendered none at all is caught by the episode count.

The unit's distinct instructions (Arena's episode results record what the environment sent) are reported; they are
not checked against anything.

An attempt that does not pass is read again for salvage (:func:`finished_episodes`): the episodes it finished, each
with its result line and a complete video from every camera, which the driver's record accepts row by row.

Frames are sampled at about 1 fps; a frame is blank when near-constant (luma std < 2). A video is invalid when it is
unreadable, has no frames, or never renders (every sampled frame blank). A camera that goes black for a stretch is a
valid episode: an arm-relative policy can drift an arm in front of a head camera or push a wrist camera into the
scene's geometry (2026-10-27, 57 such stretches checked: each fades in behind a hard edge while the other cameras keep
rendering), and rerunning those units would drop exactly the drifting episodes from the score.
"""

from __future__ import annotations

import collections
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import av
import numpy as np
import yaml
from isaaclab_arena.evaluation.arena_experiment_result import ARENA_EXPERIMENT_RESULT_FILENAME
from isaaclab_arena.visualization.episode_results_files import (
    find_episode_results_files,
    find_episode_video_files,
    parse_episode_results_filename,
    parse_episode_video_filename,
    read_episode_results,
)

from dexbench_arena.benchmark.run_manifest import COMPOSED


def _decode(mp4: Path) -> tuple[bool, str, int]:
    """Whether the video renders, a summary, and how many frames it decoded (0 when unreadable)."""
    sampled = blank = frames = 0
    try:
        with av.open(str(mp4)) as container:
            stream = container.streams.video[0]
            step = max(1, int(round(float(stream.average_rate or 30))))
            for index, frame in enumerate(container.decode(stream)):
                frames += 1
                if index % step:
                    continue
                sampled += 1
                blank += int(frame.to_ndarray(format="gray").astype(np.float32).std() < 2.0)
    except Exception as error:
        return False, f"unreadable: {error}", 0
    return sampled > blank, f"{blank}/{sampled} blank", frames


def _renders(mp4: Path) -> tuple[bool, str]:
    ok, summary, _ = _decode(mp4)
    return ok, summary


def check_unit(unit_dir: str | Path, slices: Mapping[str, Any]) -> dict[str, Any]:
    """``{"complete": bool, "episodes": n, "expected": n, "cameras": [...], "instructions": [...], "problems": [...]}``
    for the unit's ``slices`` (``{run: [start, end)}``)."""
    arena = Path(unit_dir) / "arena"
    problems: list[str] = []
    try:
        runs = json.loads((arena / ARENA_EXPERIMENT_RESULT_FILENAME).read_text())["runs"]
    except (OSError, ValueError, KeyError) as error:
        runs = {}
        problems.append(f"no readable {ARENA_EXPERIMENT_RESULT_FILENAME}: {error}")
    expected_total = recorded_total = 0
    seen: collections.Counter[str] = collections.Counter()
    videos: dict[tuple[str, int, int, int], dict[str, Path]] = collections.defaultdict(dict)
    for run, (start, end) in slices.items():
        expected = end - start
        expected_total += expected
        status = (runs.get(run) or {}).get("status")
        if status != "completed":
            problems.append(f"{run}: Arena status {status!r}")
        episodes = []
        for path in find_episode_results_files(arena / run):
            records, issues = read_episode_results(path, arena)
            episodes += records
            problems += [f"{run}: {issue}" for issue in issues]
        recorded_total += len(episodes)
        rows = [episode.get("replay_source_record_index") for episode in episodes]
        if not (all(type(row) is int for row in rows) and sorted(rows) == list(range(expected))):
            problems.append(f"{run}: {len(episodes)} episodes over {len(set(map(str, rows)))} distinct rows "
                            f"{sorted(set(map(str, rows)))[:5]}, expected rows 0..{expected - 1} once each")
        seen.update(str(episode["language_instruction"]) for episode in episodes
                    if episode.get("language_instruction") is not None)
        unscored = sum(1 for episode in episodes if episode.get("success") is None)
        if unscored:
            problems.append(f"{run}: {unscored} episodes record no success")
        for mp4 in find_episode_video_files(arena / run):
            name = parse_episode_video_filename(mp4.name)
            videos[(run, name.rebuild_index, name.env_index, name.episode_index)][name.camera_name] = mp4
        filmed = sum(1 for key in videos if key[0] == run)
        if filmed != expected:
            problems.append(f"{run}: videos of {filmed} episodes, expected {expected}")
    cameras = sorted({camera for by_camera in videos.values() for camera in by_camera})
    for key, by_camera in sorted(videos.items()):
        missing = [camera for camera in cameras if camera not in by_camera]
        if missing:
            problems.append(f"{key}: no video from {missing}")
        for camera, mp4 in sorted(by_camera.items()):
            ok, summary = _renders(mp4)
            if not ok:
                problems.append(f"{key} {camera}: {summary}")
    return {"complete": not problems, "episodes": recorded_total, "expected": expected_total, "cameras": cameras,
            "instructions": sorted(seen), "problems": problems[:20]}


def _run_order(unit_dir: Path, runs: Sequence[str]) -> list[str]:
    """The order the experiment runner ran ``runs`` in: the composed file's (the cell's order, which the runner keeps),
    else the order given."""
    try:
        composed = list((yaml.safe_load((unit_dir / COMPOSED).read_text()) or {}).get("runs") or {})
    except (OSError, yaml.YAMLError, AttributeError):
        composed = []
    return [run for run in composed if run in runs] + [run for run in runs if run not in composed]


def finished_episodes(unit_dir: str | Path, slices: Mapping[str, Any]) -> dict[str, Any]:
    """The episodes an attempt that did not complete finished, for salvage: ``{"finished": {run: [row, ...]},
    "started": [run, ...], "order": [run, ...], "problems": [...]}`` for its ``slices`` (``{run: [start, end)}``), rows
    counted at the version as the slices are."""
    unit_dir = Path(unit_dir)
    arena = unit_dir / "arena"
    problems: list[str] = []
    finished: dict[str, list[int]] = {}
    started: list[str] = []
    lines: dict[str, list[tuple[tuple[int, int, int], dict[str, Any]]]] = {}
    videos: dict[tuple[str, int, int, int], dict[str, Path]] = collections.defaultdict(dict)
    for run in slices:
        files = find_episode_results_files(arena / run)
        if files:
            started.append(run)
        lines[run] = []
        for path in files:
            rebuild = parse_episode_results_filename(path.name).rebuild_index
            records, issues = read_episode_results(path, arena)
            problems += [f"{run}: {issue}" for issue in issues]
            lines[run] += [((rebuild, record.get("env_id"), record.get("episode_in_env")), record) for record in records]
        for mp4 in find_episode_video_files(arena / run):
            name = parse_episode_video_filename(mp4.name)
            videos[(run, name.rebuild_index, name.env_index, name.episode_index)][name.camera_name] = mp4
    cameras = sorted({camera for by_camera in videos.values() for camera in by_camera})
    for run, (start, end) in slices.items():
        counts = collections.Counter(record.get("replay_source_record_index") for _, record in lines[run])
        rows = []
        for (rebuild, env, episode), record in lines[run]:
            row = record.get("replay_source_record_index")
            if not (type(row) is int and 0 <= row < end - start and counts[row] == 1):
                problems.append(f"{run}: a line replaying {row!r}, not one row of the slice once")
                continue
            if record.get("success") is None or type(record.get("episode_length")) is not int:
                problems.append(f"{run}:{row}: no success or episode_length")
                continue
            by_camera = videos.get((run, rebuild, env, episode), {})
            missing = [camera for camera in cameras if camera not in by_camera]
            if missing or not cameras:
                problems.append(f"{run}:{row}: no video from {missing or 'any camera'}")
                continue
            bad = []
            for camera, mp4 in sorted(by_camera.items()):
                ok, summary, frames = _decode(mp4)
                if not ok or frames != record["episode_length"] - 1:
                    bad.append(f"{camera} {summary}, {frames} frames of {record['episode_length'] - 1}")
            if bad:
                problems.append(f"{run}:{row}: {'; '.join(bad)}")
                continue
            rows.append(start + row)
        if rows:
            finished[run] = sorted(rows)
    return {"finished": finished, "started": started, "order": _run_order(unit_dir, list(slices)),
            "problems": problems[:20]}
