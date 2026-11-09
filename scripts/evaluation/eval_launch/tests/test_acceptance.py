# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""The Ray backend's record of a launch: an attempt ends once, a unit is accepted once, a result that comes after its
attempt was revoked counts for nothing, a stopped driver's attempts are resolved by the next one, and no decision reads
a clock. Ray-free: these are the guarantees the driver relies on whatever Ray does."""

from __future__ import annotations

import json
import multiprocessing
import os
import subprocess
import sys
import time

import pytest

from eval_launch import acceptance
from eval_launch.acceptance import DriverRunning, LaunchState


def _units(n, checkpoint="/a"):
    return [{"id": f"t.train.{i:03d}", "task": "t", "cell": "train", "slices": {"r": [i, i + 1]},
             "checkpoint": checkpoint, "expected_s": 100 - i} for i in range(n)]


def _state(path, n=1, max_attempts=2):
    state = LaunchState(path, _units(n), max_attempts=max_attempts)
    state.open_driver()
    return state


def test_a_unit_is_accepted_once_and_its_record_has_the_directory_backends_shape(tmp_path):
    state = _state(tmp_path)
    first = state.begin("t.train.000", "s0")
    assert state.finish(first, {"rc": 1, "output": "/u/first"}, ok=False) == "todo"
    second = state.begin("t.train.000", "s1")
    assert first.token != second.token and second.token.startswith("a2-s1-e1-")
    assert state.finish(second, {"rc": 0, "complete": True, "output": f"/u/{second.token}"}, ok=True) == "done"
    assert state.finish(second, {"rc": 0, "complete": True}, ok=True) == "superseded"
    record = json.loads((tmp_path / "accepted" / "t.train.000.json").read_text())
    assert record["accepted"] == second.token and [o["attempt"] for o in record["outcomes"]] == [first.token, second.token]
    assert state.status() == {"todo": 0, "running": 0, "done": 1, "failed": 0}
    with pytest.raises(RuntimeError, match="not pending"):
        state.begin("t.train.000", "s0")
    assert json.loads((tmp_path / "queue" / "done" / "000000__t.train.000.json").read_text())["accepted"] == second.token


def test_a_result_after_its_attempt_was_revoked_is_never_accepted(tmp_path):
    """The driver revokes an attempt that went quiet; it then answers. The revocation took the attempt's end first,
    so the late result is superseded and the unit's accepted attempt is a later one."""
    state = _state(tmp_path)
    late = state.begin("t.train.000", "s0")
    assert state.end(late, "revoked", {"error": "no result"})
    assert state.finish(late, {"rc": 0, "complete": True, "output": "/u/late"}, ok=True) == "superseded"
    assert not (tmp_path / "accepted" / "t.train.000.json").exists()
    retry = state.begin("t.train.000", "s0")
    assert state.finish(retry, {"rc": 0, "complete": True}, ok=True) == "done"
    assert state.accepted_records()[0]["accepted"] == retry.token
    ends = [json.loads(p.read_text()) for p in sorted((tmp_path / "state/attempts/t.train.000").glob("*.end.json"))]
    assert [(e["attempt"], e["kind"]) for e in ends] == [(late.token, "revoked"), (retry.token, "accepted")]


def _race(path, units, seq, token, action, ready, results):
    state = LaunchState(path, units)
    attempt = acceptance.Attempt("t.train.000", seq, token, 0, "s0")
    ready.wait()
    if action == "finish":
        results.put((action, state.finish(attempt, {"rc": 0, "complete": True}, ok=True)))
    else:
        results.put((action, "won" if state.end(attempt, "revoked", {"error": "late"}) else "lost"))


def test_racing_processes_end_an_attempt_once(tmp_path):
    """Eight processes at once, half delivering the attempt's result and half revoking it: one decides."""
    units = _units(1)
    attempt = _state(tmp_path).begin("t.train.000", "s0")
    context = multiprocessing.get_context("fork")
    ready, results = context.Event(), context.Queue()
    workers = [context.Process(target=_race, args=(tmp_path, units, attempt.seq, attempt.token,
                                                    "finish" if i % 2 else "revoke", ready, results)) for i in range(8)]
    for worker in workers:
        worker.start()
    ready.set()
    for worker in workers:
        worker.join(60)
    outcomes = [results.get(timeout=10) for _ in workers]
    winners = [o for o in outcomes if o[1] in ("done", "won")]
    assert len(winners) == 1, outcomes
    accepted = tmp_path / "accepted" / "t.train.000.json"
    assert accepted.exists() == (winners[0][0] == "finish")


def test_two_drivers_on_one_unit_accept_one_attempt(tmp_path):
    """The lock keeps a second driver out; if it ever got in (a filesystem without working locks), each would start its
    own attempt of the same unit, and still only one could be the unit's result."""
    units = _units(1)
    one, two = LaunchState(tmp_path, units), LaunchState(tmp_path, units)
    a, b = one.begin("t.train.000", "s0"), two.begin("t.train.000", "s1")
    assert a.seq != b.seq
    results = {one.finish(a, {"rc": 0, "complete": True}, ok=True), two.finish(b, {"rc": 0, "complete": True}, ok=True)}
    assert results == {"done", "superseded"}
    assert len(list((tmp_path / "accepted").iterdir())) == 1


def test_a_second_driver_is_refused_while_one_holds_the_launch(tmp_path):
    held = _state(tmp_path)
    holder = ("import sys, time\nfrom eval_launch.acceptance import LaunchState, DriverRunning\n"
              "try:\n    LaunchState(sys.argv[1], []).open_driver()\nexcept DriverRunning:\n    sys.exit(3)\n")
    assert subprocess.run([sys.executable, "-c", holder, str(tmp_path)]).returncode == 3
    held.close_driver()
    assert subprocess.run([sys.executable, "-c", holder, str(tmp_path)]).returncode == 0
    assert _state(tmp_path).epoch == 3


def test_the_next_driver_resolves_what_a_stopped_one_left(tmp_path):
    """A driver killed with units under way: an attempt whose actor wrote a checked, complete outcome is accepted; one
    that wrote a failed outcome fails; one that wrote nothing is uncertain and charged; an acceptance whose record
    was not yet written is completed. Nothing is lost and nothing is accepted twice."""
    units = _units(4)
    first = LaunchState(tmp_path, units)
    first.open_driver()
    attempts = [first.begin(unit["id"], "s0") for unit in units]
    for attempt, outcome in zip(attempts[:2], ({"rc": 0, "complete": True}, {"rc": 1, "complete": False})):
        path = first.outcome_path(attempt.unit_id, attempt.token)
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(outcome))
    first.end(attempts[3], "accepted", {"rc": 0, "complete": True})
    first.close_driver()
    second = LaunchState(tmp_path, units)
    second.open_driver()
    resolved = second.recover()
    assert resolved == {"accepted": ["t.train.000"], "failed": ["t.train.001"], "uncertain": ["t.train.002"],
                        "completed": ["t.train.003"]}
    assert [second.unit_state(u["id"]) for u in units] == ["done", "todo", "todo", "done"]
    assert second.units["t.train.002"].charged() == 1
    assert {r["unit"]["id"]: r["accepted"] for r in second.accepted_records()} == {
        "t.train.000": attempts[0].token, "t.train.003": attempts[3].token}
    assert second.recover() == {"accepted": [], "failed": [], "uncertain": [], "completed": []}


def test_attempts_are_charged_until_the_budget_and_a_retry_round_gives_a_new_one(tmp_path):
    state = _state(tmp_path, max_attempts=2)
    released = state.begin("t.train.000", "s0")
    state.end(released, "released", {"error": "server would not start"})
    lost = state.begin("t.train.000", "s0")
    state.end(lost, "lost", {"error": "ActorDiedError"})
    assert state.settle("t.train.000") == "todo"
    failed = state.begin("t.train.000", "s0")
    assert state.finish(failed, {"rc": 1}, ok=False) == "failed"
    assert state.pending() == [] and state.failed_records()[0]["attempts"] == 2
    assert (tmp_path / "queue" / "failed" / "000000__t.train.000.json").is_file()
    assert state.retry_failed() == ["t.train.000"]
    assert [unit["id"] for unit in state.pending()] == ["t.train.000"]
    assert not (tmp_path / "queue" / "failed" / "000000__t.train.000.json").exists()
    again = state.begin("t.train.000", "s0")
    assert again.round == 1 and state.finish(again, {"rc": 0, "complete": True}, ok=True) == "done"
    assert len(state.accepted_records()[0]["outcomes"]) == 3


def test_no_decision_reads_a_clock(tmp_path, monkeypatch):
    """Days of skew and jumps on every read change no state and no answer."""
    calls = iter(range(10**6))
    monkeypatch.setattr(time, "time", lambda: 1e9 + (next(calls) % 7 - 3) * 86400.0 * 30)
    state = _state(tmp_path, n=2)
    a = state.begin("t.train.000", "s0")
    state.end(a, "revoked", {"error": "quiet"})
    assert state.finish(a, {"rc": 0, "complete": True}, ok=True) == "superseded"
    b = state.begin("t.train.000", "s0")
    assert state.finish(b, {"rc": 0, "complete": True}, ok=True) == "done"
    os.utime(tmp_path / "accepted" / "t.train.000.json", (0, 0))
    reloaded = LaunchState(tmp_path, _units(2))
    assert reloaded.status() == {"todo": 1, "running": 0, "done": 1, "failed": 0}
