# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""A driver launch's record (``backend.kind: local`` or ``ray``): every attempt its driver started, how each ended,
and the one accepted per unit.

One process writes it, the launch's driver (:mod:`eval_launch.driver`), and it holds ``state/driver.lock`` while it
runs, so a second driver on the same output refuses to start. The executor (local or Ray) decides where an attempt runs and
reports how it went; this record decides what counts. Under ``<output>``:

- ``accepted/<unit id>.json``: the unit's result, written once every row of it is accepted: the attempt that
  completed it and every charged attempt's outcome, in the shape of the directory backend's done record (``unit``,
  ``accepted``, ``outcomes``), so ``--score`` reads either, and ``parts``, which attempt each row is read from.
- ``state/attempts/<unit id>/<seq>.begin.json``: an attempt the driver authorised, with its token
  (``a<seq>-<station>-e<driver epoch>-<random>``, also the name of its output directory ``units/<id>/<token>/``) and
  the rows it runs (``slices``).
- ``state/attempts/<unit id>/<seq>.end.json``: how it ended (:data:`KINDS`), the rows it finished (``rows``) and the
  rows it is charged for (``charged_rows``).
- ``state/rows/<unit id>/<run>/<row>.json``: the attempt a row was accepted from.
- ``state/attempts/<unit id>/retry-<k>.json``: ``--retry-failed`` gave the unit round ``k``.
- ``state/drivers/<epoch>.json``: one per driver start; the epoch goes into every token it makes.
- ``queue/done/`` and ``queue/failed/``: views in the directory backend's names, rewritten from the above, for
  readers of that layout (a results collector). Nothing here reads them.

Every record is created once and never changed: written whole under a temporary name, then hard-linked to its name,
which fails when the name exists (:func:`create_once`). That makes two records the fences:

- an attempt's end: its result, a revocation (the driver gave up on it) and a restarted driver's recovery all try to
  create it, and the first decides. A result that arrives after its attempt was revoked finds the end taken and is
  ``superseded``; it is never accepted, and its output directory is never named by an accepted record.
- a row's record: created once, from the end of the attempt that finished it, so each (unit, Run, row) is read from
  one attempt for the launch's life, also if two drivers ever ran at once; the unit's ``accepted/`` record, created
  once all of them exist, lists them.

Rows, not units, are accepted (episode-level salvage). An attempt that completes accepts every row it ran. One that
does not (its simulator crashed, its deadline stopped it, its actor or node was lost, it was revoked) still accepts
each episode it finished (``unit_check.finished_episodes``: its result line, and a complete video from every camera);
the next attempt of the unit runs only the rows left. Its unfinished rows are charged, not the unit: the row in
progress when it stopped (:func:`charged_rows`), or every row it was given when it never got into an episode (or said
nothing of what it finished). A row charged ``max_attempts`` times in its round and not accepted has failed, alone; a
unit is failed once each row is accepted or failed and one failed. So a crash that strikes one episode in twenty costs
that episode's rerun, not the unit's, and the episodes that ran clean are not rerun until a run of the whole unit
happens to avoid it (which would leave the score with only the episodes no crash interrupted).

No decision reads a clock: an attempt ends by its result, by the driver's revocation (timed on the driver's monotonic
clock) or by recovery, never by comparing a timestamp. Charges are counted per round: an attempt that was authorised
is charged however it ended (it may have run), except one handed back untried (``released``).

Records written before rows existed read as whole units: a begin without ``slices`` ran the unit's slices, an
``accepted`` end without ``rows`` finished all of them, a charged end without ``charged_rows`` charged all of them, and
an ``accepted/`` record without ``parts`` is read from its accepted attempt alone.

Ray-free and Kit-free.
"""

from __future__ import annotations

import errno
import fcntl
import json
import os
import secrets
import socket
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eval_launch.queue_dir import record_name

KINDS = {
    "accepted": True,
    "failed": True,
    "lost": True,
    "revoked": True,
    "uncertain": True,
    "released": False,
}
"""How an attempt can end, and whether it is charged to the unit's ``max_attempts``."""

ACCEPTED, STATE = "accepted", "state"
SALVAGED_KINDS = ("lost", "revoked", "uncertain")
"""Ends with no outcome from the simulator's actor: the driver reads what the attempt finished itself, when it was given
a reader (``LaunchState(salvage=...)``)."""
Row = tuple[str, int]
"""A unit's row: its Run and its index at the version, as the unit's ``slices`` count them."""


def slice_rows(slices: Mapping[str, Sequence[int]]) -> list[Row]:
    """Every row of ``{run: [start, end)}``, in the slices' order."""
    return [(run, index) for run, (start, end) in slices.items() for index in range(start, end)]


def rows_mapping(rows: Iterable[Row]) -> dict[str, list[int]]:
    """``{run: [row, ...]}``, as records hold rows."""
    mapping: dict[str, list[int]] = {}
    for run, index in sorted(set(rows)):
        mapping.setdefault(run, []).append(index)
    return mapping


def mapping_rows(mapping: Mapping[str, Iterable[int]] | None) -> set[Row]:
    return {(run, int(index)) for run, indices in (mapping or {}).items() for index in indices}


def charged_rows(slices: Mapping[str, Sequence[int]], finished: set[Row], salvage: Mapping[str, Any] | None) -> set[Row]:
    """The rows an attempt that did not complete is charged for: the first row it did not finish, in the order it ran its
    Runs (``salvage["order"]``), when that row's Run had started (``salvage["started"]``), so the simulator stopped in
    that row's episode or before its next one; otherwise, or with no ``salvage`` to go by, every row it did not finish.
    An attempt that finished every row is charged nothing."""
    salvage = salvage or {}
    order = [run for run in salvage.get("order") or () if run in slices]
    order += [run for run in slices if run not in order]
    unfinished = [(run, index) for run in order for index in range(*slices[run]) if (run, index) not in finished]
    if not unfinished:
        return set()
    if "finished" in salvage and unfinished[0][0] in (salvage.get("started") or ()):
        return {unfinished[0]}
    return set(unfinished)
OUTCOME_SUFFIX = ".outcome.json"
"""``units/<id>/<token>.outcome.json``: what the simulator's actor wrote once its unit ended and was checked, so a
driver started after a crash can still accept a result the stopped one never received."""


def create_once(path: Path, data: Mapping[str, Any]) -> bool:
    """Publish ``data`` at ``path`` unless something is there; whether this call published it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(dict(data), indent=1) + "\n").encode()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, path)
        created = True
    except FileExistsError:
        created = path.read_bytes() == payload
    finally:
        temporary.unlink(missing_ok=True)
    _sync_directory(path.parent)
    return created


def _sync_directory(directory: Path) -> None:
    """Make a new name in ``directory`` durable where the filesystem supports it (a file's fsync does not)."""
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _replace(path: Path, data: Mapping[str, Any]) -> None:
    """Write a view whole or not at all."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(dict(data), indent=1) + "\n")
    os.replace(temporary, path)


@dataclass(frozen=True)
class Attempt:
    unit_id: str
    seq: int
    token: str
    round: int
    station: str


@dataclass
class UnitRecord:
    unit: dict[str, Any]
    rank: int
    begins: dict[int, dict[str, Any]] = field(default_factory=dict)
    ends: dict[int, dict[str, Any]] = field(default_factory=dict)
    rounds: int = 0
    accepted: dict[str, Any] | None = None
    rows: dict[Row, str] = field(default_factory=dict)
    """Each accepted row's attempt token."""

    def live(self) -> list[int]:
        """Attempts begun and not ended."""
        return sorted(seq for seq in self.begins if seq not in self.ends)

    def charged(self) -> int:
        """Charged attempts in the current round."""
        return sum(1 for seq, end in self.ends.items()
                   if KINDS[end["kind"]] and self.begins.get(seq, {}).get("round", 0) == self.rounds)


class DriverRunning(RuntimeError):
    """Another driver holds the launch."""


EXIT_ENDED_WITH_FAILURES = 3
"""The driver's exit code when every unit has ended and some failed for good (0: every unit accepted). Distinct from 1,
which any uncaught Python error also returns, so a caller can tell a launch that has ended from a driver that stopped
(the k8s head pod lets its workers go only on 0 or this)."""
EXIT_DRIVER_RUNNING = 75
"""The driver's exit code when another driver holds the launch (EX_TEMPFAIL): a caller that restarts the driver, such as
the k8s head pod after its predecessor died, waits for the old holder's lock to go and tries again."""


class LaunchState:
    """A launch's attempts and acceptances, loaded once and kept in step by the one driver that writes them."""

    def __init__(self, output: str | Path, units: Sequence[Mapping[str, Any]], max_attempts: int = 2,
                 salvage: Callable[[Path, Mapping[str, Any]], Mapping[str, Any]] | None = None):
        """``salvage`` (``unit_check.finished_episodes``) reads what an attempt finished when its end comes with no
        outcome from its actor (:data:`SALVAGED_KINDS`); without it, such an attempt is charged every row it ran."""
        self.output = Path(output)
        self.max_attempts = max_attempts
        self.salvage = salvage
        self.epoch = 0
        self._lock: int | None = None
        self.units: dict[str, UnitRecord] = {}
        for rank, unit in enumerate(units):
            self.units[unit["id"]] = UnitRecord(dict(unit), rank)
        self.load()


    @property
    def state_dir(self) -> Path:
        return self.output / STATE

    def _unit_dir(self, unit_id: str) -> Path:
        return self.state_dir / "attempts" / unit_id

    def _begin_path(self, unit_id: str, seq: int) -> Path:
        return self._unit_dir(unit_id) / f"{seq:03d}.begin.json"

    def _end_path(self, unit_id: str, seq: int) -> Path:
        return self._unit_dir(unit_id) / f"{seq:03d}.end.json"

    def accepted_path(self, unit_id: str) -> Path:
        return self.output / ACCEPTED / f"{unit_id}.json"

    def outcome_path(self, unit_id: str, token: str) -> Path:
        return self.output / "units" / unit_id / f"{token}{OUTCOME_SUFFIX}"

    def attempt_dir(self, unit_id: str, token: str) -> Path:
        return self.output / "units" / unit_id / token

    def _row_path(self, unit_id: str, row: Row) -> Path:
        return self.state_dir / "rows" / unit_id / row[0] / f"{row[1]:06d}.json"


    def load(self) -> None:
        """Read every record of the planned units from disk."""
        for unit_id in self.units:
            self._load_unit(unit_id)

    def _load_unit(self, unit_id: str) -> None:
        record = self.units[unit_id]
        record.begins, record.ends, record.rounds, record.accepted = {}, {}, 0, None
        directory = self._unit_dir(unit_id)
        for path in sorted(directory.iterdir()) if directory.is_dir() else ():
            if path.name.startswith("."):
                continue
            if path.name.endswith(".begin.json"):
                record.begins[int(path.name.split(".", 1)[0])] = json.loads(path.read_text())
            elif path.name.endswith(".end.json"):
                record.ends[int(path.name.split(".", 1)[0])] = json.loads(path.read_text())
            elif path.name.startswith("retry-"):
                record.rounds = max(record.rounds, int(path.name[len("retry-"):].split(".", 1)[0]))
        accepted = self.accepted_path(unit_id)
        if accepted.is_file():
            record.accepted = json.loads(accepted.read_text())
        record.rows = self._read_rows(unit_id)

    def _read_rows(self, unit_id: str) -> dict[Row, str]:
        rows = {}
        directory = self.state_dir / "rows" / unit_id
        for path in directory.glob("*/[0-9]*.json") if directory.is_dir() else ():
            rows[(path.parent.name, int(path.name.split(".", 1)[0]))] = json.loads(path.read_text())["attempt"]
        return rows

    def is_ray_launch(self) -> bool:
        return (self.state_dir / "drivers").is_dir()


    def open_driver(self, note: Mapping[str, Any] | None = None) -> int:
        """Hold the launch for this process and take the next driver epoch; DriverRunning when another holds it."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.state_dir / "driver.lock", os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            os.close(descriptor)
            if error.errno in (errno.EAGAIN, errno.EACCES):
                raise DriverRunning(f"another driver holds {self.state_dir / 'driver.lock'}") from None
            raise
        self._lock = descriptor
        drivers = self.state_dir / "drivers"
        drivers.mkdir(exist_ok=True)
        epoch = 1 + max((int(path.name.split(".", 1)[0]) for path in drivers.glob("[0-9]*.json")), default=0)
        record = {"host": socket.gethostname(), "pid": os.getpid(), "nonce": secrets.token_hex(8), **dict(note or {})}
        while not create_once(drivers / f"{epoch:06d}.json", record):
            epoch += 1
        self.epoch = epoch
        self.load()
        return epoch

    def close_driver(self) -> None:
        if self._lock is not None:
            fcntl.flock(self._lock, fcntl.LOCK_UN)
            os.close(self._lock)
            self._lock = None


    def begin(self, unit_id: str, station: str) -> Attempt:
        """Authorise a new attempt of a pending unit; its token is new and never reused."""
        record = self.units[unit_id]
        if self.unit_state(unit_id) != "todo":
            raise RuntimeError(f"{unit_id} is not pending (accepted, failed or running)")
        slices = self.next_slices(unit_id)
        seq = 1 + max(record.begins, default=0)
        while True:
            token = f"a{seq}-{station}-e{self.epoch}-{secrets.token_hex(3)}"
            begin = {"attempt": token, "seq": seq, "round": record.rounds, "station": station, "epoch": self.epoch,
                     "slices": slices, "started_at": time.time()}
            if create_once(self._begin_path(unit_id, seq), begin):
                record.begins[seq] = begin
                return Attempt(unit_id, seq, token, record.rounds, station)
            seq += 1

    def end(self, attempt: Attempt, kind: str, outcome: Mapping[str, Any]) -> bool:
        """End ``attempt`` as ``kind``; False when it had ended already (a revocation, or a result, came first)."""
        if kind not in KINDS:
            raise ValueError(f"an attempt ends as one of {sorted(KINDS)}, not {kind!r}")
        record = self.units[attempt.unit_id]
        if record.ends.get(attempt.seq) is not None:
            return False
        outcome = {**dict(outcome), "attempt": attempt.token}
        slices = self.attempt_slices(attempt.unit_id, attempt.seq)
        if kind == "accepted":
            finished, charged = set(slice_rows(slices)), set()
        elif KINDS[kind]:
            salvage = outcome.get("salvage")
            directory = self.attempt_dir(attempt.unit_id, attempt.token)
            if salvage is None and kind in SALVAGED_KINDS and self.salvage is not None and directory.is_dir():
                try:
                    salvage = outcome["salvage"] = dict(self.salvage(directory, slices))
                except Exception as error:
                    outcome["salvage"] = {"error": repr(error)[:500]}
            finished = mapping_rows((salvage or {}).get("finished")) & set(slice_rows(slices))
            charged = charged_rows(slices, finished, salvage)
            if finished:
                outcome.setdefault("output", str(directory))
        else:
            finished, charged = set(), set()
        end = {"attempt": attempt.token, "kind": kind, "charged": KINDS[kind], "outcome": outcome,
               "rows": rows_mapping(finished), "charged_rows": rows_mapping(charged), "nonce": secrets.token_hex(8)}
        if not create_once(self._end_path(attempt.unit_id, attempt.seq), end):
            path = self._end_path(attempt.unit_id, attempt.seq)
            record.ends[attempt.seq] = json.loads(path.read_text())
            return False
        record.ends[attempt.seq] = end
        return True

    def finish(self, attempt: Attempt, outcome: Mapping[str, Any], ok: bool) -> str:
        """Record an attempt's result: ``done`` (the unit's every row is accepted), ``todo`` (rows are left for another
        attempt), ``failed`` (rows failed and none is left) or ``superseded`` (the attempt had ended, e.g. was revoked,
        or every row it finished was accepted from another attempt, so its result counts for nothing)."""
        if not self.end(attempt, "accepted" if ok else "failed", outcome):
            return "superseded"
        state = self.settle(attempt.unit_id)
        mine = self.ended_rows(attempt.unit_id, attempt.seq)
        if mine and not any(self.units[attempt.unit_id].rows.get(row) == attempt.token for row in mine):
            return "superseded"
        return state

    def settle(self, unit_id: str) -> str:
        """The unit's state after one of its attempts ended: the rows its ended attempts finished are accepted, the unit
        once all of them are, and its view is written when that state is final."""
        self._accept_rows(unit_id)
        state = self.unit_state(unit_id)
        if state in ("done", "failed"):
            self._write_views(unit_id)
        return state

    def _accept_rows(self, unit_id: str) -> None:
        """Create a row record for each row an ended attempt finished that has none (the first attempt to create it is the
        row's, for good), then the unit's accepted record once every row has one."""
        record = self.units[unit_id]
        for seq, end in sorted(record.ends.items()):
            for row in sorted(self.ended_rows(unit_id, seq) - set(record.rows)):
                path = self._row_path(unit_id, row)
                create_once(path, {"attempt": end["attempt"], "seq": seq, "nonce": secrets.token_hex(8)})
                record.rows[row] = json.loads(path.read_text())["attempt"]
        if record.accepted is None and set(record.rows) >= self.planned_rows(unit_id):
            self._publish_acceptance(unit_id)

    def _publish_acceptance(self, unit_id: str) -> None:
        """Create the unit's accepted record once every row has its attempt: ``accepted`` the last of them (the attempt
        that completed the unit), ``parts`` each contributing attempt with its output and rows, ``retried_rows`` the
        rows some attempt was charged for before one finished them."""
        record = self.units[unit_id]
        if not {record.rows[row] for row in self.planned_rows(unit_id)} <= {end["attempt"] for end in record.ends.values()}:
            self._load_unit(unit_id)
        by_token = {end["attempt"]: (seq, end) for seq, end in record.ends.items()}
        parts = []
        for token in sorted({record.rows[row] for row in self.planned_rows(unit_id)}, key=lambda t: by_token[t][0]):
            seq, end = by_token[token]
            parts.append({"attempt": token, "kind": end["kind"], "salvaged": end["kind"] != "accepted",
                          "output": end["outcome"].get("output") or str(self.attempt_dir(unit_id, token)),
                          "slices": self.attempt_slices(unit_id, seq),
                          "rows": rows_mapping(row for row in self.planned_rows(unit_id) if record.rows[row] == token)})
        retried = set().union(*(self.charged_by(unit_id, seq) for seq in record.ends)) & self.planned_rows(unit_id)
        outcomes = [end["outcome"] for number, end in sorted(record.ends.items()) if end["charged"]]
        accepted = {"unit": record.unit, "accepted": parts[-1]["attempt"], "attempts": len(outcomes),
                    "outcomes": outcomes, "parts": parts, "retried_rows": rows_mapping(retried),
                    "epoch": self.epoch, "nonce": secrets.token_hex(8)}
        create_once(self.accepted_path(unit_id), accepted)
        record.accepted = json.loads(self.accepted_path(unit_id).read_text())
        self._write_views(unit_id)

    def recover(self) -> dict[str, list[str]]:
        """At a driver's start, end every attempt a stopped driver left: accepted when its actor had written a checked,
        complete outcome, failed when it had written another, uncertain (charged) when it wrote none. Also completes
        an acceptance whose record a stopped driver did not get to write. Returns the unit ids per resolution."""
        resolved: dict[str, list[str]] = {"accepted": [], "failed": [], "uncertain": [], "completed": []}
        for unit_id, record in self.units.items():
            for seq in record.live():
                begin = record.begins[seq]
                attempt = Attempt(unit_id, seq, begin["attempt"], begin["round"], begin["station"])
                path = self.outcome_path(unit_id, attempt.token)
                if path.is_file():
                    outcome = json.loads(path.read_text())
                    ok = outcome.get("rc") == 0 and outcome.get("complete") is True
                    self.finish(attempt, {**outcome, "recovered": True}, ok=ok)
                    resolved["accepted" if ok else "failed"].append(unit_id)
                else:
                    self.end(attempt, "uncertain", {"error": "the driver that started this attempt stopped before it "
                                                             "ended; no checked outcome was written"})
                    resolved["uncertain"].append(unit_id)
            if record.accepted is None:
                self.settle(unit_id)
                if record.accepted is not None:
                    resolved["completed"].append(unit_id)
        self.write_views()
        return resolved

    def retry_failed(self) -> list[str]:
        """Give every failed unit a new round of ``max_attempts`` (``--retry-failed``); its earlier attempts stay."""
        moved = []
        for unit_id, record in self.units.items():
            if self.failed(unit_id):
                k = record.rounds + 1
                create_once(self._unit_dir(unit_id) / f"retry-{k:03d}.json", {"round": k, "nonce": secrets.token_hex(8)})
                record.rounds = k
                moved.append(unit_id)
        self.write_views()
        return moved


    def planned_rows(self, unit_id: str) -> set[Row]:
        return set(slice_rows(self.units[unit_id].unit["slices"]))

    def attempt_slices(self, unit_id: str, seq: int) -> dict[str, list[int]]:
        """The rows attempt ``seq`` was given (a begin from before rows existed: the unit's)."""
        record = self.units[unit_id]
        return {run: list(bound) for run, bound in
                ((record.begins.get(seq) or {}).get("slices") or record.unit["slices"]).items()}

    def ended_rows(self, unit_id: str, seq: int) -> set[Row]:
        """The rows attempt ``seq``'s end says it finished (an ``accepted`` end from before rows existed: all it ran)."""
        end = self.units[unit_id].ends.get(seq)
        if end is None:
            return set()
        if "rows" in end:
            return mapping_rows(end["rows"])
        return set(slice_rows(self.attempt_slices(unit_id, seq))) if end["kind"] == "accepted" else set()

    def charged_by(self, unit_id: str, seq: int) -> set[Row]:
        """The rows attempt ``seq``'s end is charged for (a charged end from before rows existed: all it ran)."""
        end = self.units[unit_id].ends.get(seq)
        if end is None or not end["charged"] or end["kind"] == "accepted":
            return set()
        if "charged_rows" in end:
            return mapping_rows(end["charged_rows"])
        return set(slice_rows(self.attempt_slices(unit_id, seq)))

    def row_charges(self, unit_id: str) -> Counter[Row]:
        """How often each row was charged in the unit's current round."""
        record = self.units[unit_id]
        charges: Counter[Row] = Counter()
        for seq in record.ends:
            if record.begins.get(seq, {}).get("round", 0) == record.rounds:
                charges.update(self.charged_by(unit_id, seq))
        return charges

    def failed_rows(self, unit_id: str) -> set[Row]:
        """Rows not accepted that were charged ``max_attempts`` times in the current round."""
        record = self.units[unit_id]
        return {row for row, count in self.row_charges(unit_id).items()
                if count >= self.max_attempts and row not in record.rows and row in self.planned_rows(unit_id)}

    def remaining_rows(self, unit_id: str) -> set[Row]:
        """Rows neither accepted nor failed."""
        record = self.units[unit_id]
        return self.planned_rows(unit_id) - set(record.rows) - self.failed_rows(unit_id)

    def next_slices(self, unit_id: str) -> dict[str, list[int]]:
        """The rows the unit's next attempt runs: for each Run of the unit, in its order, the first unbroken stretch of
        rows still to run (``--slices`` takes one stretch per Run; a gap left by a failed row is run by a later
        attempt, charged nothing for the rows it was not given)."""
        remaining = self.remaining_rows(unit_id)
        slices = {}
        for run, (start, end) in self.units[unit_id].unit["slices"].items():
            left = [index for index in range(start, end) if (run, index) in remaining]
            if left:
                stop = left[0]
                while stop + 1 in left:
                    stop += 1
                slices[run] = [left[0], stop + 1]
        return slices

    def failed(self, unit_id: str) -> bool:
        """No attempt runs, no row is left to run, and a row failed."""
        record = self.units[unit_id]
        return (record.accepted is None and not record.live() and not self.remaining_rows(unit_id)
                and bool(self.failed_rows(unit_id)))

    def unit_state(self, unit_id: str) -> str:
        """``done``, ``failed``, ``running`` or ``todo``: done once every row is accepted (its record follows at the
        next :meth:`settle`, also a restarted driver's :meth:`recover`)."""
        record = self.units[unit_id]
        if record.accepted is not None or set(record.rows) >= self.planned_rows(unit_id):
            return "done"
        if record.live():
            return "running"
        if self.failed(unit_id):
            return "failed"
        return "todo"

    def pending(self) -> list[dict[str, Any]]:
        """Units to run, in plan order, each with the slices its next attempt runs (the rows not yet accepted)."""
        return [{**record.unit, "slices": self.next_slices(unit_id)} for unit_id, record in self.units.items()
                if self.unit_state(unit_id) == "todo"]

    def status(self) -> dict[str, int]:
        counts = {"todo": 0, "running": 0, "done": 0, "failed": 0}
        for unit_id in self.units:
            counts[self.unit_state(unit_id)] += 1
        return counts

    def accepted_records(self) -> list[dict[str, Any]]:
        return [record.accepted for record in self.units.values() if record.accepted is not None]

    def failed_records(self) -> list[dict[str, Any]]:
        return [self._failed_view(unit_id) for unit_id in self.units if self.failed(unit_id)]


    def _failed_view(self, unit_id: str) -> dict[str, Any]:
        record = self.units[unit_id]
        outcomes = [end["outcome"] for _, end in sorted(record.ends.items()) if end["charged"]]
        return {"unit": record.unit, "attempts": record.charged(), "outcomes": outcomes,
                "failed_rows": rows_mapping(self.failed_rows(unit_id)), "accepted_rows": rows_mapping(record.rows)}

    def _write_views(self, unit_id: str) -> None:
        record = self.units[unit_id]
        name = record_name(record.rank, unit_id)
        done, failed = self.output / "queue" / "done" / name, self.output / "queue" / "failed" / name
        if record.accepted is not None:
            _replace(done, record.accepted)
            failed.unlink(missing_ok=True)
        elif self.failed(unit_id):
            _replace(failed, self._failed_view(unit_id))
        else:
            failed.unlink(missing_ok=True)

    def write_views(self, unit_ids: Iterable[str] | None = None) -> None:
        for unit_id in unit_ids if unit_ids is not None else self.units:
            self._write_views(unit_id)
