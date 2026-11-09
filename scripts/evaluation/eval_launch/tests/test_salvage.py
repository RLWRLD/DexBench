# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Episode-level salvage: when an attempt's simulator crashes mid-unit, every episode it finished (its Arena result line
and a complete video from every camera) is accepted, row by row and once; the unit's next attempt runs only the rows
left; the row whose episode was under way is charged, alone, and fails alone after ``max_attempts``; a revoked
attempt's late result adds nothing; a driver stopped between a salvage and the next attempt resumes where it was; and
the score reads each row from the one attempt it was accepted from, with the salvage counted. The cell runner is a
stand-in that writes Arena's files (``fake_cell.py``), run by the real ``eval_worker.run_unit``."""

from __future__ import annotations

import json
import os
import random
import time

import pytest

av = pytest.importorskip("av", reason="the salvage check decodes the attempt's videos with PyAV")
pytest.importorskip("isaaclab_arena.visualization.episode_results_files", reason="needs IsaacLab-Arena on the path")

from eval_launch import acceptance, eval_worker, launch  # noqa: E402
from eval_launch.acceptance import LaunchState, charged_rows  # noqa: E402
from eval_launch.tests import fake_cell  # noqa: E402
from eval_launch.unit_check import finished_episodes  # noqa: E402

SLICES = {"a": [0, 4], "b": [10, 13]}
"""A unit of two Runs: rows 0-3 of ``a`` and 10-12 of ``b`` (seven episodes)."""


def _units(slices=SLICES):
    return [{"id": "t.train.000", "task": "t", "cell": "train", "slices": slices, "checkpoint": "/a", "expected_s": 60}]


class _Server:
    port, token, started_s, checkpoint = 5810, "t", 1.0, "/a"

    def client_set(self):
        return []


def _launch(tmp_path, monkeypatch, steps, units=None, max_attempts=2):
    """A driver launch's output, its record opened as a driver would, and the stand-in runner with ``steps``."""
    units = units or _units()
    output = tmp_path / "launch"
    output.mkdir()
    (output / "plan.json").write_text(json.dumps({"config": {"plan": {"max_attempts": max_attempts}}, "units": units,
                                                  "tasks": {"t": {}}}))
    monkeypatch.setattr(eval_worker, "RUN_CELL", fake_cell.__file__)
    monkeypatch.setenv("EVAL_TEST_PLAN", str(fake_cell.plan(tmp_path / "steps.json", steps)))
    state = LaunchState(output, units, max_attempts=max_attempts, salvage=finished_episodes)
    state.open_driver()
    return output, state


def _drive(state, output, limit=10):
    """What the driver does with each pending unit, one attempt at a time: begin it, run it, finish it."""
    states = []
    for _ in range(limit):
        pending = state.pending()
        if not pending:
            return states
        unit = pending[0]
        attempt = state.begin(unit["id"], "s0")
        unit_dir = output / "units" / unit["id"] / attempt.token
        outcome = eval_worker.run_unit(unit, unit_dir, _Server(), "gr00t", deadline=120.0)
        ok = outcome.get("rc") == 0 and outcome.get("complete") is True
        states.append(state.finish(attempt, {**outcome, "output": str(unit_dir)}, ok))
    raise AssertionError(f"still pending after {limit} attempts: {states}")


def _seen(output, token):
    return json.loads((output / "units" / "t.train.000" / token / fake_cell.SEEN).read_text())["slices"]


def test_a_crashed_attempt_salvages_exactly_the_episodes_it_finished(tmp_path):
    """Three episodes finished, the fourth's videos cut off when the simulator died: the three, and nothing else, and
    the row under way is the one charged."""
    fake_cell.write_attempt(tmp_path / "u", SLICES, stop=3)
    result = finished_episodes(tmp_path / "u", SLICES)
    assert result["finished"] == {"a": [0, 1, 2]} and result["started"] == ["a"] and result["order"] == ["a", "b"]
    assert charged_rows(SLICES, acceptance.mapping_rows(result["finished"]), result) == {("a", 3)}
    assert charged_rows(SLICES, {("a", i) for i in range(4)}, {"finished": {}, "started": ["a"], "order": ["a", "b"]}) \
        == {("b", 10), ("b", 11), ("b", 12)}


@pytest.mark.parametrize("damage", ["line-without-its-video", "video-closed-early", "torn-line", "camera-missing"])
def test_an_episode_without_its_line_or_a_complete_video_from_every_camera_is_not_salvaged(tmp_path, damage):
    """Arena writes an episode's line before its encoder finalises the videos: a crash between them leaves the line and
    a video still being written. An encoder that saw its input end early finalises a short file. Neither passes."""
    unit = tmp_path / "u"
    fake_cell.write_attempt(unit, SLICES, stop=2)
    results = unit / "arena" / "a" / "episode_results_rebuild0.jsonl"
    if damage == "line-without-its-video":
        fake_cell.episode(unit / "arena", "a", 2, 2)
        for camera in fake_cell.CAMERAS:
            fake_cell.video(unit / "arena" / "a" / f"robot-cam-rebuild0-env0-{camera}-episode-2.mp4", cut=True)
    elif damage == "video-closed-early":
        fake_cell.video(unit / "arena" / "a" / "robot-cam-rebuild0-env0-head-episode-1.mp4", frames=fake_cell.FRAMES - 2)
    elif damage == "torn-line":
        results.write_text(results.read_text()[:-20])
    elif damage == "camera-missing":
        (unit / "arena" / "a" / "robot-cam-rebuild0-env0-wrist-episode-1.mp4").unlink()
    result = finished_episodes(unit, SLICES)
    assert result["finished"] == ({"a": [0, 1]} if damage == "line-without-its-video" else {"a": [0]}), result
    assert result["problems"]


@pytest.mark.parametrize("skew", [False, True], ids=["clocks-agree", "clocks-skewed"])
def test_the_next_attempt_runs_only_the_rows_left_and_the_score_reads_each_row_once(tmp_path, monkeypatch, skew):
    """The first attempt finishes three of seven rows and crashes in the fourth's episode; the second runs the four left
    and completes. Each row is accepted from one attempt; the score reads both, each row once, and counts the three
    salvaged episodes and the one row that ran twice. Wall clocks off by a month either way change none of it."""
    if skew:
        real = time.time
        monkeypatch.setattr(time, "time", lambda: real() + random.uniform(-30, 30) * 86400)
    output, state = _launch(tmp_path, monkeypatch, {"t.train.000": ["crash_after:3"]})
    assert _drive(state, output) == ["todo", "done"]
    first, second = (begin["attempt"] for _, begin in sorted(state.units["t.train.000"].begins.items()))
    assert _seen(output, first) == SLICES and _seen(output, second) == {"a": [3, 4], "b": [10, 13]}
    if skew:
        for path in output.rglob("*.json"):
            stamp = random.uniform(0, 4e9)
            os.utime(path, (stamp, stamp))
    reloaded = LaunchState(output, _units())
    assert reloaded.status() == {"todo": 0, "running": 0, "done": 1, "failed": 0}
    rows = reloaded.units["t.train.000"].rows
    assert sorted(rows) == acceptance.slice_rows(SLICES)
    assert {rows[("a", i)] for i in range(3)} == {first} and {token for row, token in rows.items() if row[1] >= 3} == {second}
    record = reloaded.accepted_records()[0]
    assert record["accepted"] == second and [part["attempt"] for part in record["parts"]] == [first, second]
    assert record["retried_rows"] == {"a": [3]}
    parts = launch.accepted_parts(output)["t"]["train"]
    assert parts == [
        {"output": str(output / "units/t.train.000" / first), "rows": {"a": [0, 1, 2]}, "salvaged": True, "retried": {}},
        {"output": str(output / "units/t.train.000" / second), "rows": None, "salvaged": False, "retried": {"a": [0]}}]
    lines = [json.loads(line) for line in (output / "units/t.train.000" / first / "arena/a/episode_results_rebuild0.jsonl")
             .read_text().splitlines()]
    assert [line["replay_source_record_index"] for line in lines] == [0, 1, 2]


def test_a_row_that_keeps_crashing_fails_alone_and_the_rest_of_its_unit_completes(tmp_path, monkeypatch):
    """Row ``a``:1 crashes its episode every time: after ``max_attempts`` charges it fails, the rows after it run in the
    next attempt, and the unit ends failed with exactly that row failed; --score names it; a retry round runs it alone."""
    units = _units({"a": [0, 4]})
    output, state = _launch(tmp_path, monkeypatch, {"t.train.000": ["crash_at:a:1", "crash_at:a:1", "ok", "crash_at:a:1"]},
                            units=units)
    assert _drive(state, output) == ["todo", "todo", "failed"]
    tokens = [begin["attempt"] for _, begin in sorted(state.units["t.train.000"].begins.items())]
    assert [_seen(output, token) for token in tokens] == [{"a": [0, 4]}, {"a": [1, 4]}, {"a": [2, 4]}]
    assert state.status() == {"todo": 0, "running": 0, "done": 0, "failed": 1}
    (failed,) = state.failed_records()
    assert failed["failed_rows"] == {"a": [1]} and failed["accepted_rows"] == {"a": [0, 2, 3]}
    assert json.loads(next((output / "queue" / "failed").glob("*.json")).read_text())["failed_rows"] == {"a": [1]}
    with pytest.raises(SystemExit, match=r'failed rows \{"t.train.000": \{"a": \[1\]\}\}'):
        launch.score(output, partial=False)
    assert state.retry_failed() == ["t.train.000"]
    assert [unit["slices"] for unit in state.pending()] == [{"a": [1, 2]}]
    assert _drive(state, output) == ["todo", "done"]
    record = state.accepted_records()[0]
    assert [part["rows"] for part in record["parts"]] == [{"a": [0]}, {"a": [2, 3]}, {"a": [1]}]
    assert record["retried_rows"] == {"a": [1]}


def test_a_late_result_from_a_revoked_attempt_accepts_nothing_more(tmp_path, monkeypatch):
    """The driver revokes an attempt that went quiet, taking what it had finished by then; the attempt then finishes the
    unit and its result arrives. Its end was taken: it is superseded, and the rows it finished after the revocation
    are run again by the next attempt, from that attempt's directory only."""
    output, state = _launch(tmp_path, monkeypatch, {})
    unit = state.pending()[0]
    late = state.begin(unit["id"], "s0")
    late_dir = output / "units" / unit["id"] / late.token
    fake_cell.write_attempt(late_dir, unit["slices"], stop=2)
    assert state.end(late, "revoked", {"error": "no result past its deadline"})
    assert state.settle(unit["id"]) == "todo"
    fake_cell.write_attempt(late_dir, unit["slices"])
    assert state.finish(late, {"rc": 0, "complete": True, "output": str(late_dir)}, ok=True) == "superseded"
    assert set(state.units[unit["id"]].rows) == {("a", 0), ("a", 1)}
    assert [u["slices"] for u in state.pending()] == [{"a": [2, 4], "b": [10, 13]}]
    assert _drive(state, output) == ["done"]
    record = state.accepted_records()[0]
    assert [(part["attempt"], part["rows"]) for part in record["parts"]] == [
        (late.token, {"a": [0, 1]}), (record["accepted"], {"a": [2, 3], "b": [10, 11, 12]})]
    assert record["parts"][0]["kind"] == "revoked" and record["retried_rows"] == {"a": [2]}


def test_a_driver_stopped_after_a_salvage_resumes_where_it_was(tmp_path, monkeypatch):
    """The driver recorded a crashed attempt's end (with what it finished) and died before writing the rows; another
    attempt's actor wrote its outcome and the driver died before reading it; a third attempt's driver died with no
    outcome written. The next driver accepts each one's finished rows once and runs only what is left."""
    units = [{**_units()[0], "id": f"t.train.00{i}"} for i in range(3)]
    output, state = _launch(tmp_path, monkeypatch, {}, units=units)
    attempts = [state.begin(unit["id"], "s0") for unit in units]
    directories = [output / "units" / a.unit_id / a.token for a in attempts]
    for directory, stop in zip(directories, (3, 1, 5)):
        fake_cell.write_attempt(directory, SLICES, stop=stop)
    salvage = finished_episodes(directories[0], SLICES)
    assert state.end(attempts[0], "failed", {"rc": 1, "salvage": salvage, "output": str(directories[0])})
    outcome = {"rc": 1, "complete": False, "salvage": finished_episodes(directories[1], SLICES), "output": str(directories[1])}
    state.outcome_path(units[1]["id"], attempts[1].token).write_text(json.dumps(outcome))
    state.close_driver()

    after = LaunchState(output, units, salvage=finished_episodes)
    after.open_driver()
    assert not after.units["t.train.000"].rows
    assert after.recover() == {"accepted": [], "failed": ["t.train.001"], "uncertain": ["t.train.002"], "completed": []}
    assert {unit_id: sorted(record.rows) for unit_id, record in after.units.items()} == {
        "t.train.000": [("a", 0), ("a", 1), ("a", 2)], "t.train.001": [("a", 0)],
        "t.train.002": [("a", 0), ("a", 1), ("a", 2), ("a", 3), ("b", 10)]}
    assert [unit["slices"] for unit in after.pending()] == [
        {"a": [3, 4], "b": [10, 13]}, {"a": [1, 4], "b": [10, 13]}, {"b": [11, 13]}]
    assert after.row_charges("t.train.002") == {("b", 11): 1}
    assert after.recover() == {"accepted": [], "failed": [], "uncertain": [], "completed": []}
    assert _drive(after, output) == ["done", "done", "done"]
    assert all(set(record.rows) == set(acceptance.slice_rows(SLICES)) for record in after.units.values())


def test_records_written_before_rows_existed_read_as_whole_units(tmp_path):
    """A launch recorded by the driver before salvage: begins without slices, ends without rows, an accepted record
    without parts. Its done unit is read from its accepted attempt, its failed unit failed every row, and a unit with
    one failed attempt runs its whole slice again."""
    units = [{**_units()[0], "id": f"t.train.00{i}"} for i in range(3)]
    output = tmp_path / "launch"
    attempts = output / "state" / "attempts"
    (output / "state" / "drivers").mkdir(parents=True)
    (output / "plan.json").write_text(json.dumps({"config": {"plan": {"max_attempts": 2}}, "units": units}))

    def old(unit_id, seq, kind, outcome):
        token = f"a{seq}-s0-e1-000000"
        (attempts / unit_id).mkdir(parents=True, exist_ok=True)
        (attempts / unit_id / f"{seq:03d}.begin.json").write_text(json.dumps(
            {"attempt": token, "seq": seq, "round": 0, "station": "s0", "epoch": 1, "started_at": 0}))
        (attempts / unit_id / f"{seq:03d}.end.json").write_text(json.dumps(
            {"attempt": token, "kind": kind, "charged": True, "outcome": {**outcome, "attempt": token}, "nonce": "n"}))
        return token

    first = old("t.train.000", 1, "failed", {"rc": 1, "output": "/u/one"})
    second = old("t.train.000", 2, "accepted", {"rc": 0, "complete": True, "output": "/u/two"})
    (output / "accepted").mkdir()
    (output / "accepted" / "t.train.000.json").write_text(json.dumps(
        {"unit": units[0], "accepted": second, "attempts": 2, "nonce": "n", "outcomes": [
            {"rc": 1, "output": "/u/one", "attempt": first}, {"rc": 0, "output": "/u/two", "attempt": second}]}))
    old("t.train.001", 1, "failed", {"rc": 1})
    old("t.train.001", 2, "lost", {"error": "lost"})
    old("t.train.002", 1, "failed", {"rc": 1})
    state = launch.ray_state(output)
    assert [state.unit_state(unit["id"]) for unit in units] == ["done", "failed", "todo"]
    assert state.failed_records()[0]["failed_rows"] == acceptance.rows_mapping(acceptance.slice_rows(SLICES))
    assert [unit["slices"] for unit in state.pending()] == [SLICES]
    assert launch.accepted_parts(output) == {"t": {"train": [
        {"output": "/u/two", "rows": None, "salvaged": False, "retried": {"a": [0, 1, 2, 3], "b": [0, 1, 2]}}]}}
