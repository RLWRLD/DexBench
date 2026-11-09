# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""A launch's work queue as a directory on a filesystem every worker sees: no queue service to run.

``todo/<rank>__<unit id>.json`` holds a unit waiting to run, ``<rank>`` its place in the plan (longest first), so a
sorted listing is the plan's order. A finished unit leaves a record in ``done/`` or ``failed/``; a unit that failed
``max_attempts`` times stays in ``failed/`` and is never dropped silently.

Claims are fenced. Every claim is a new attempt with its own id, and its file is ``claimed/<base>.<attempt id>``, a
name no other attempt ever has; the attempt writes its own output directory. Every move out of a state renames the
file to a hidden transition name first (``claimed/.<step>.<base>...``), which exactly one caller can do, and only then
writes the destination:

- ``claim`` renames ``todo/<base>`` to a hidden name, so no stale scan can see a claim before it carries its attempt
  and a fresh heartbeat, then publishes ``claimed/<base>.<attempt>``;
- ``finish``, ``release`` and ``requeue_stale`` take the claim by its exact name, so a worker whose claim was requeued
  under it gets ``superseded`` back and records nothing, and a later attempt is never credited with its result.

A hidden transition file left by a process that died mid-step is recovered by ``requeue_stale`` once it is stale:
removed when its destination was already written, put back in the queue otherwise, so a unit is never lost and an
accepted unit never runs again. A stale claim counts as an attempt. Hidden names are never read as states.

The queue is the launch's state: every unit in ``done/`` stays done, so a launch stopped for any reason resumes by
starting workers on the same directory again (``python -m eval_launch --resume``).

Every age is measured on the queue filesystem's clock (:func:`fs_now`), never the host's: on NFS a file's mtime is
set by the file server, so a pod whose clock ran ahead of it would have seen live claims as stale.
"""

from __future__ import annotations

import errno
import json
import os
import secrets
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

STATES = ("todo", "claimed", "done", "failed")
MIN_RECOVERY_S = 60.0
"""A hidden transition file younger than this belongs to a step still under way, whatever ``stale_s`` says."""


def fs_now(root: str | Path) -> float:
    """Now, on the clock of the filesystem ``root`` is on: the mtime of a file created there this moment. On NFS a new
    file's mtime and a touch (``os.utime`` with no time) take the file server's clock, while ``time.time()`` is this
    host's; comparing the two made a pod whose clock ran about 9.5 min ahead (600 s stale limit less a 30 s
    heartbeat) requeue live claims, each an attempt. The probe file is a temporary name, which
    no listing reads."""
    path = Path(root) / f".clock.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        return os.fstat(descriptor).st_mtime
    finally:
        os.close(descriptor)
        path.unlink(missing_ok=True)


def _since_taken(path: Path, now: float) -> float:
    """How long ago a hidden transition file was taken. A rename keeps the file's mtime, so a unit that waited in
    ``todo/`` for an hour became a transition that looked an hour old to recovery at once and could be taken from a
    step still under way. Every step now touches the file right after its rename, so its
    mtime is when it was taken; what is left is the instant between the two calls. (ctime would close that too, but
    no test can age it, and a mount need not set it on a rename.)"""
    return now - path.stat().st_mtime


def record_name(rank: int, unit_id: str) -> str:
    """A queue record's file name, ``<rank>__<unit id>.json``: listed, the plan's order. The Ray backend writes its
    views under the same names (:mod:`eval_launch.acceptance`)."""
    return f"{rank:06d}__{unit_id}.json"


def _base(name: str) -> str | None:
    """``<rank>__<unit id>.json`` for a queue file's name (a state file, a claim, or a hidden transition file);
    None for a temporary write."""
    if name.endswith(".tmp"):
        return None
    if name.startswith("."):
        name = name[1:].split(".", 1)[1]
    return name.split(".json", 1)[0] + ".json" if ".json" in name else None


def _unit_id(base: str) -> str:
    return base.split("__", 1)[1].rsplit(".json", 1)[0]


def _entries(directory: Path, hidden: bool = False) -> Iterator[Path]:
    """The queue files in ``directory``: the visible ones, or with ``hidden`` the transition files. ``Path.glob``
    matches hidden names too, so every listing goes through here."""
    for path in sorted(directory.iterdir()) if directory.is_dir() else ():
        if path.name.endswith(".tmp") or path.name.startswith(".") != hidden:
            continue
        if _base(path.name):
            yield path


def _write_new(path: Path, data: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=1) + "\n")
    os.replace(temporary, path)


def _ids(root: Path) -> dict[str, set[str]]:
    """Unit ids per state; a unit with a hidden transition file in ``claimed/`` counts as claimed."""
    found = {state: {_unit_id(_base(path.name)) for path in _entries(root / state)} for state in STATES}
    found["claimed"] |= {_unit_id(_base(path.name)) for path in _entries(root / "claimed", hidden=True)}
    return found


def init(root: str | Path, units: Sequence[Mapping[str, Any]]) -> int:
    """Put ``units`` (in plan order) into ``todo/`` behind whatever the queue holds; returns how many were added.
    A unit already in any state is left as it is, so running it again on a launch adds only what is missing."""
    root = Path(root)
    for state in STATES:
        (root / state).mkdir(parents=True, exist_ok=True)
    known = set().union(*_ids(root).values())
    if any(unit["id"] not in known for unit in units):
        known |= set().union(*_ids(root).values())
    ranks = [int(_base(path.name).split("__", 1)[0]) for state in STATES
             for hidden in (False, True) for path in _entries(root / state, hidden)]
    start = max(ranks, default=-1) + 1
    added = 0
    for unit in units:
        if unit["id"] in known:
            continue
        _write_new(root / "todo" / record_name(start + added, unit["id"]), {"unit": dict(unit), "attempts": 0})
        added += 1
    return added


def claim(root: str | Path, worker: str, prefer: Mapping[str, Any] | None = None) -> tuple[Path, dict] | None:
    """Claim the next unit, one matching ``prefer`` (e.g. the checkpoint the worker's server has loaded) first.
    Returns the claim's path and its record, or None when ``todo/`` is empty."""
    root = Path(root)
    waiting = list(_entries(root / "todo"))
    if prefer:
        def matches(path: Path) -> bool:
            try:
                unit = json.loads(path.read_text())["unit"]
            except (OSError, ValueError):
                return False
            return all(unit.get(key) == value for key, value in prefer.items())
        waiting = [path for path in waiting if matches(path)] + [path for path in waiting if not matches(path)]
    for path in waiting:
        token = secrets.token_hex(4)
        hidden = root / "claimed" / f".claiming.{path.name}.{token}"
        try:
            os.utime(path)
            os.rename(path, hidden)
        except FileNotFoundError:
            continue
        try:
            os.utime(hidden)
            record = json.loads(hidden.read_text())
        except FileNotFoundError:
            continue
        except OSError as error:
            if error.errno != errno.ESTALE:
                raise
            continue
        record["attempts"] += 1
        record["worker"] = worker
        record["claimed_at"] = time.time()
        record["attempt"] = f"a{record['attempts']}-{worker}-{token}"
        target = root / "claimed" / f"{path.name}.{record['attempt']}"
        _write_new(target, record)
        hidden.unlink(missing_ok=True)
        return target, record
    return None


def heartbeat(claim_path: str | Path) -> None:
    """Touch the claim; FileNotFoundError once it is no longer this attempt's."""
    os.utime(claim_path)


def lost_race(error: OSError) -> bool:
    """Whether ``error`` means another worker's recovery scan took the file: it is gone, or the file server dropped
    the handle while the scan renamed it (ESTALE)."""
    return isinstance(error, FileNotFoundError) or error.errno == errno.ESTALE


def _touch_taken(taken: Path) -> bool:
    """Mark a hidden transition file as taken now (_since_taken); False when another worker's scan took it first.
    Until the touch the file carries the age it had before its rename, so a scan can take it in between, and a pod
    whose clock runs behind can make it look old after the touch too (claim, a launch on 2026-11-04). The scan then
    puts the unit back in the queue, so the step that lost the file has nothing left to do."""
    try:
        os.utime(taken)
    except OSError as error:
        if not lost_race(error):
            raise
        return False
    return True


def _read_taken(taken: Path) -> dict | None:
    """A hidden transition file's record; None when another worker's scan took the file (:func:`_touch_taken`)."""
    try:
        return json.loads(taken.read_text())
    except OSError as error:
        if not lost_race(error):
            raise
        return None


def _take(claim_path: Path, step: str) -> Path | None:
    """Take this exact claim by renaming it to a hidden name; None when it is no longer there, or when another
    worker's scan took the hidden file before the touch marked it taken (a stale claim's mtime is its last
    heartbeat, so it looks like a dead step's residue until then)."""
    taken = claim_path.with_name(f".{step}.{claim_path.name}")
    try:
        os.rename(claim_path, taken)
    except FileNotFoundError:
        return None
    return taken if _touch_taken(taken) else None


def finish(root: str | Path, claim_path: str | Path, outcome: Mapping[str, Any], *, ok: bool,
           max_attempts: int = 2) -> str:
    """Record a claimed unit's outcome: ``done``, back to ``todo`` for another attempt, or ``failed`` for good.
    Returns ``superseded`` and records nothing when this attempt's claim was requeued meanwhile."""
    root, claim_path = Path(root), Path(claim_path)
    taken = _take(claim_path, "finish")
    record = None if taken is None else _read_taken(taken)
    if record is None:
        return "superseded"
    record = {**record, "finished_at": time.time()}
    record.setdefault("outcomes", []).append({**dict(outcome), "attempt": record.get("attempt")})
    if ok:
        state = "done"
        record["accepted"] = record.get("attempt")
    elif record["attempts"] < max_attempts:
        state = "todo"
    else:
        state = "failed"
    _write_new(root / state / _base(claim_path.name), record)
    taken.unlink(missing_ok=True)
    return state


def release(root: str | Path, claim_path: str | Path) -> bool:
    """Hand a claimed unit back untried (the worker could not run anything, e.g. its policy server would not start):
    back to ``todo/`` with this attempt not counted. False when the claim was no longer this attempt's."""
    root, claim_path = Path(root), Path(claim_path)
    taken = _take(claim_path, "release")
    record = None if taken is None else _read_taken(taken)
    if record is None:
        return False
    record["attempts"] = max(0, record["attempts"] - 1)
    _write_new(root / "todo" / _base(claim_path.name), record)
    taken.unlink(missing_ok=True)
    return True


def _requeue(root: Path, taken: Path, max_attempts: int, why: str) -> str | None:
    """Put a taken unit back in the queue; None when another worker's scan took the file first and requeued it."""
    record = _read_taken(taken)
    if record is None:
        return None
    record.setdefault("outcomes", []).append({"error": why, "attempt": record.get("attempt"),
                                              "worker": record.get("worker")})
    state = "failed" if record["attempts"] >= max_attempts else "todo"
    _write_new(root / state / _base(taken.name), record)
    taken.unlink(missing_ok=True)
    return _unit_id(_base(taken.name))


def requeue_stale(root: str | Path, stale_s: float, now: float | None = None, max_attempts: int = 2) -> list[str]:
    """Take claims whose heartbeat stopped more than ``stale_s`` ago (``now``: the filesystem's, :func:`fs_now`): back
    to ``todo/``, or to ``failed/`` once the unit has used ``max_attempts``; and recover hidden transition files a dead
    process left. Returns the unit ids put back in a state."""
    root = Path(root)
    now = fs_now(root) if now is None else now
    moved = []
    for path in _entries(root / "claimed"):
        try:
            if now - path.stat().st_mtime < stale_s:
                continue
        except FileNotFoundError:
            continue
        taken = _take(path, "stale")
        unit_id = None if taken is None else _requeue(root, taken, max_attempts, f"heartbeat stopped for {stale_s:.0f} s")
        if unit_id is not None:
            moved.append(unit_id)
    states = _ids(root)
    for path in _entries(root / "claimed", hidden=True):
        try:
            if _since_taken(path, now) < max(stale_s, MIN_RECOVERY_S):
                continue
        except FileNotFoundError:
            continue
        unit_id = _unit_id(_base(path.name))
        published = any(unit_id in states[state] for state in ("todo", "done", "failed")) or any(
            _unit_id(_base(claim_path.name)) == unit_id for claim_path in _entries(root / "claimed"))
        if published:
            path.unlink(missing_ok=True)
            continue
        taken = path.with_name(f".recover.{_base(path.name)}.{secrets.token_hex(4)}")
        try:
            os.rename(path, taken)
        except FileNotFoundError:
            continue
        unit_id = _requeue(root, taken, max_attempts, "a queue step's process died") if _touch_taken(taken) else None
        if unit_id is not None:
            moved.append(unit_id)
    return moved


def retry_failed(root: str | Path) -> list[str]:
    """Put every unit in ``failed/`` back into ``todo/`` with its attempts reset (``--resume --retry-failed``);
    its earlier outcomes stay in the record. Returns their unit ids."""
    root = Path(root)
    moved = []
    for path in _entries(root / "failed"):
        record = json.loads(path.read_text())
        record["attempts"] = 0
        _write_new(root / "todo" / path.name, record)
        path.unlink()
        moved.append(_unit_id(path.name))
    return moved


def terminal(root: str | Path) -> bool:
    """Whether every unit is done or failed: nothing waiting and nothing running."""
    counts = status(root)
    return counts["todo"] == 0 and counts["claimed"] == 0


def status(root: str | Path) -> dict[str, int]:
    """Units per state, each counted once: a unit caught between two states counts in the one it reached."""
    ids = _ids(Path(root))
    ended = ids["done"] | ids["failed"]
    return {"todo": len(ids["todo"] - ended), "claimed": len(ids["claimed"] - ended - ids["todo"]),
            "done": len(ids["done"]), "failed": len(ids["failed"] - ids["done"])}


def unit_ids(root: str | Path, state: str) -> set[str]:
    """The unit ids in ``state``."""
    return _ids(Path(root))[state]
