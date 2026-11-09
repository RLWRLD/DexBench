# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""One driver runs a launch's units on stations an executor places: the decisions both the local and the Ray backend make.

A *station* is one policy server and its simulators (:mod:`eval_launch.layout`: ``colocated``, ``server+N``,
``{sim_gpus, sims_per_gpu}``). An :class:`Executor` puts a station's processes on GPUs and runs what the driver asks
of them; the driver decides everything else, and its record (:mod:`eval_launch.acceptance`) decides what counts:

- :class:`eval_launch.local_backend.LocalExecutor` (``backend.kind: local``): this machine's GPUs, one child process
  per server and per running simulator, standard library only;
- :class:`eval_launch.ray_backend.RayExecutor` (``backend.kind: ray``): stations Ray places, on any machine Ray runs on.

How an attempt ends, and what the driver does:

- a result: accepted when it exited 0 and ``unit_check`` passed, else a failed attempt; the unit runs again until it
  has ``plan.max_attempts`` charged attempts.
- its simulator slot died under it (:class:`Lost`): ``lost``, charged; the slot takes no unit until the executor says
  it is back (:meth:`Executor.probe`), and a slot that keeps losing its process gets its station rebuilt.
- no result ``REVOKE_GRACE_S`` past the unit's deadline (on the driver's monotonic clock): ``revoked``, charged; the
  slot is killed (:meth:`Executor.kill_slot`), and a result that still arrives is fenced out (its attempt has ended).
- the station could not run it (:class:`Raised`: its server would not start, the cell runner refused the unit):
  ``released``, not charged, and the station is rebuilt, ``station_restarts`` times at most.
- the driver is stopped (SIGTERM, Ctrl-C): ``released``; killed outright: the next driver ends it from what its slot
  wrote (:meth:`LaunchState.recover`).

Every process a station starts carries the launch's owner tag (:func:`owner_scope`), so what a killed process left
(a Kit, a server: each runs in its own session) is stopped by the executor's sweep at start and stop
(``process_group.stop_orphans``).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from eval_launch import eval_worker, server_env
from eval_launch.acceptance import EXIT_DRIVER_RUNNING, EXIT_ENDED_WITH_FAILURES, Attempt, DriverRunning, LaunchState
from eval_launch.launch import adopt_plan, record_launch, write_plan
from eval_launch.layout import Layout, parse_layout, stations
from eval_launch.plan import MANIFEST, check_recipe, load_config, load_timings, task_list
from eval_launch.unit_check import finished_episodes

REVOKE_GRACE_S = 1800.0
"""How long past its deadline (``eval_worker.deadline_s``) a unit may go without a result before the driver revokes it:
the slot stops its Kit at the deadline itself, then checks the unit (decoding every video), which takes minutes."""
STATUS_EVERY_S = 60.0


def owner_scope(output: Path) -> str:
    """The owner tag every process of this launch's stations carries (``process_group``)."""
    return f"{output.resolve()}:driver"


class Raised(RuntimeError):
    """The slot or server raised: its station cannot run units (its server, environment or filesystem, or a cell the
    runner refused before running it), never the unit itself."""


class Lost(RuntimeError):
    """The process (or its node) died under the call."""


class Gone(Lost):
    """The slot's process died and will not come back (out of restarts)."""


@dataclass(frozen=True)
class ServerAddress:
    """What a simulator needs of its station's server: ``eval_worker.run_unit`` reads these three."""

    port: int
    token: str
    items: tuple[str, ...]

    def client_set(self) -> list[str]:
        return list(self.items)


def run_slot(unit: Mapping[str, Any], token: str, output: str, server: Mapping[str, Any], model: str | Mapping[str, Any],
             deadline: float, environment: Mapping[str, str], worker: str, where: Mapping[str, Any]) -> dict[str, Any]:
    """What a simulator slot does with one unit, on either executor: ``eval_worker.run_unit`` into
    ``units/<id>/<token>/``; the outcome, also written next to that directory so a driver started after this one's
    driver stopped can still accept it."""
    unit_dir = Path(output) / "units" / unit["id"] / token
    address = ServerAddress(int(server["port"]), str(server["token"]), tuple(server["client_set"]))
    outcome = eval_worker.run_unit(unit, unit_dir, address, model, deadline, environment)
    outcome = {**outcome, "output": str(unit_dir), "server_load_s": server.get("started_s"), "worker": worker,
               "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "server_gpu": server.get("gpu"), **where}
    path = unit_dir.parent / f"{token}.outcome.json"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(outcome, indent=1) + "\n")
    os.replace(temporary, path)
    return outcome


def join_pythonpath(pythonpath: list[str]) -> None:
    """Put the model's client packages (``backend.pythonpath``) last on this process's path, for everything it starts:
    the recipe's setup (which checks the simulator can import the client) and the simulator. The directory backend
    does the same for each worker (``launch.run_local``), and a pod's script for the k8s backend."""
    if pythonpath:
        os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, (os.environ.get("PYTHONPATH"), *pythonpath)))


class Executor(Protocol):
    """Where a station's processes run, and how the driver reaches them. Each call that returns a handle starts
    something and returns at once; :meth:`wait` says which handles have finished and :meth:`result` gives what one
    returned, or raises :class:`Raised`, :class:`Lost` or :class:`Gone`."""

    kind: str
    gpus: int
    port_base: int
    slot_restarts: int
    """How many times in a row a slot may lose its process before its station is rebuilt."""
    station_restarts: int

    def start(self, epoch: int) -> list[int]:
        """Bring the executor up; stop what dead processes of this launch left and return their pids."""

    def stop(self) -> None:
        """Release what :meth:`start` brought up (the stations are torn down first)."""

    def place(self, station: Station) -> Any:
        """Reserve the station's GPUs; a handle that finishes once :meth:`open` can start its processes."""

    def open(self, station: Station) -> None:
        """Start the station's server and simulator slots once :meth:`place`'s handle has finished."""

    def ensure_server(self, station: Station, checkpoint: str) -> Any:
        """Serve ``checkpoint`` on the station (start or keep its server); a handle to what its simulators need:
        ``checkpoint``, ``port``, ``token``, ``client_set``, ``started_s``, ``gpu``."""

    def run(self, station: Station, slot: int, unit: Mapping[str, Any], token: str, server: Mapping[str, Any],
            deadline: float) -> Any:
        """Run ``unit`` as attempt ``token`` on the slot (:func:`run_slot`); a handle to its outcome."""

    def probe(self, station: Station, slot: int) -> Any:
        """A handle that finishes once the slot can take a unit again (:class:`Gone` when it never will)."""

    def kill_slot(self, station: Station, slot: int) -> None:
        """Stop the slot's process now (it ran past its deadline and does not answer); its call ends as :class:`Lost`."""

    def tear_down(self, stations: list[Station], graceful: bool = False) -> None:
        """Stop the stations' processes, free their GPUs and sweep what the stopped ones left."""

    def quiet(self, stations: list[Station]) -> set[str]:
        """The names of the ``stations`` whose processes answer, now, that they run nothing (a station not named is
        kept: it did not answer in time, or answered that it is busy)."""

    def release_nodes(self, kept: list[Station]) -> list[str]:
        """Let go of the machines that host none of the ``kept`` stations (the driver released every other station),
        so whatever holds them (a pod) can end and free its GPUs; the names of the ones newly let go. An executor
        whose GPUs are not anyone else's to take back (this machine's) lets go of nothing."""

    def wait(self, handles: list[Any], timeout: float) -> list[Any]:
        """The finished ones of ``handles``, once one has finished or ``timeout`` seconds have passed."""

    def result(self, handle: Any) -> Any:
        """What a finished handle returned; raises :class:`Raised`, :class:`Lost` or :class:`Gone`."""


@dataclass
class Inflight:
    attempt: Attempt
    station: "Station"
    slot: int
    revoke_at: float
    revoked: bool = False


@dataclass
class Station:
    index: int
    port: int
    placing: Any = None
    """The executor's handle for the station's placement, until it has finished."""
    ready: bool = False
    held: Any = None
    """What the executor holds for the station (its placement group and actors, its processes)."""
    loading: Any = None
    server_info: dict[str, Any] | None = None
    verify: bool = False
    busy: dict[int, Any] = field(default_factory=dict)
    probing: dict[int, Any] = field(default_factory=dict)
    """Slots that lost their process: a probe of each, which finishes once the slot is back."""
    probe_at: dict[int, float] = field(default_factory=dict)
    lost_in_a_row: dict[int, int] = field(default_factory=dict)
    failures: int = 0
    retired: bool = False
    retiring: bool = False
    """Marked for release: no unit goes to it while the driver checks it again (:meth:`Driver._release_idle`)."""
    released: bool = False
    """Torn down for good once no unit was left to start (:meth:`Driver._release_idle`): not a failure."""

    @property
    def name(self) -> str:
        return f"s{self.index}"


class Driver:
    """Runs every pending unit of a launch on its stations until each is accepted or failed."""

    def __init__(self, config: Mapping[str, Any], output: Path, plan: Mapping[str, Any], executor: Executor,
                 poll_s: float = 5.0, retry_failed: bool = False, revoke_grace_s: float = REVOKE_GRACE_S,
                 release_every_s: float = 30.0):
        self.config, self.output, self.plan, self.executor = config, Path(output), plan, executor
        self.poll_s, self.retry_failed, self.revoke_grace_s = poll_s, retry_failed, revoke_grace_s
        self.release_every_s = release_every_s
        self.layout: Layout = parse_layout(config.get("servers"))
        self.stations_wanted = stations(self.layout, executor.gpus)
        self.plan_config = config.get("plan") or {}
        self.state = LaunchState(output, plan["units"], max_attempts=int(self.plan_config.get("max_attempts", 2)),
                                 salvage=finished_episodes)
        self.stations: list[Station] = []
        self.inflight: dict[Any, Inflight] = {}
        self.tag = f"[{executor.kind}]"
        self.next_release = 0.0

    def log(self, message: str) -> None:
        print(f"{self.tag} {message}", flush=True)


    def _build(self, station: Station) -> None:
        station.ready, station.held = False, None
        station.loading = station.server_info = None
        station.busy, station.lost_in_a_row, station.verify = {}, {}, False
        station.probing, station.probe_at = {}, {}
        station.placing = self.executor.place(station)

    def _station_failed(self, station: Station, why: str) -> None:
        """The station cannot run units: what it holds goes back untried, then it is rebuilt or retired."""
        for handle, entry in list(self.inflight.items()):
            if entry.station is station:
                del self.inflight[handle]
                self.state.end(entry.attempt, "lost", {"error": f"its station failed: {why}", "worker": station.name})
                self.state.settle(entry.attempt.unit_id)
        self.executor.tear_down([station])
        station.placing, station.ready, station.held = None, False, None
        station.failures += 1
        if station.failures > self.executor.station_restarts:
            station.retired = True
            self.log(f"{station.name} retired after {station.failures} failures; last: {why}")
        else:
            self.log(f"{station.name} rebuilt ({station.failures}): {why}")
            self._build(station)


    def _dispatch(self, station: Station) -> None:
        if (station.retired or station.released or station.retiring or not station.ready
                or station.loading is not None):
            return
        now = time.monotonic()
        for slot, when in list(station.probe_at.items()):
            if now >= when:
                del station.probe_at[slot]
                station.probing[slot] = self.executor.probe(station, slot)
        idle = [slot for slot in range(len(self.layout.sim_slots()))
                if slot not in station.busy and slot not in station.probing and slot not in station.probe_at]
        if not idle:
            return
        pending = self.state.pending()
        if not pending:
            return
        current = station.server_info["checkpoint"] if station.server_info else None
        mine = [unit for unit in pending if unit["checkpoint"] == current]
        if station.verify or not mine:
            if not mine and station.busy:
                return
            station.loading = self.executor.ensure_server(station, current if mine else pending[0]["checkpoint"])
            return
        for slot, unit in zip(idle, mine):
            attempt = self.state.begin(unit["id"], station.name)
            deadline = eval_worker.deadline_s(unit, self.plan_config) * self.layout.sims_per_gpu
            handle = self.executor.run(station, slot, unit, attempt.token, station.server_info, deadline)
            station.busy[slot] = handle
            self.inflight[handle] = Inflight(attempt, station, slot, time.monotonic() + deadline + self.revoke_grace_s)
            self.log(f"{unit['id']} {attempt.token}: started on {station.name}.{slot}")

    def _loaded(self, station: Station) -> None:
        handle, station.loading = station.loading, None
        try:
            station.server_info, station.verify = self.executor.result(handle), False
        except (Raised, Lost) as error:
            self._station_failed(station, f"its server: {error}")

    def _ended(self, handle: Any) -> None:
        entry = self.inflight.pop(handle)
        station, attempt = entry.station, entry.attempt
        station.busy.pop(entry.slot, None)
        try:
            outcome = self.executor.result(handle)
        except Raised as error:
            if self.state.end(attempt, "released", {"error": str(error), "worker": station.name}):
                self.state.settle(attempt.unit_id)
            self._station_failed(station, str(error))
            return
        except Lost as error:
            lost = self.state.end(attempt, "lost", {"error": f"lost: {error}", "worker": station.name})
            state = self.state.settle(attempt.unit_id) if lost else "superseded"
            self.log(f"{attempt.unit_id} {attempt.token}: {state} (lost: {str(error)[:200]})")
            station.verify = True
            station.probe_at[entry.slot] = time.monotonic()
            station.lost_in_a_row[entry.slot] = station.lost_in_a_row.get(entry.slot, 0) + 1
            if station.lost_in_a_row[entry.slot] > self.executor.slot_restarts:
                self._station_failed(station, f"slot {entry.slot} lost its process {station.lost_in_a_row[entry.slot]} "
                                              "times")
            return
        station.lost_in_a_row[entry.slot] = 0
        ok = outcome.get("rc") == 0 and outcome.get("complete") is True
        state = self.state.finish(attempt, outcome, ok)
        if not ok:
            station.verify = True
        self.log(f"{attempt.unit_id} {attempt.token}: {state} {json.dumps(outcome)[:300]}")

    def _probed(self, station: Station, slot: int) -> None:
        handle = station.probing.pop(slot)
        try:
            self.executor.result(handle)
        except Gone as error:
            self._station_failed(station, f"slot {slot}'s process is dead: {error}")
        except (Raised, Lost):
            station.probe_at[slot] = time.monotonic() + 1.0
        else:
            self.log(f"{station.name}.{slot}: its slot is back")

    def _revoke_overdue(self) -> None:
        now = time.monotonic()
        for entry in self.inflight.values():
            if entry.revoked or now < entry.revoke_at:
                continue
            entry.revoked = True
            if self.state.end(entry.attempt, "revoked", {"error": f"no result {self.revoke_grace_s:.0f} s past its "
                                                                  "deadline: revoked", "worker": entry.station.name}):
                self.log(f"{entry.attempt.unit_id} {entry.attempt.token}: revoked "
                         f"({self.state.settle(entry.attempt.unit_id)})")
            self.executor.kill_slot(entry.station, entry.slot)

    def _idle(self, station: Station) -> bool:
        """The station holds no attempt: none started on any of its slots (an attempt is the slot's until its outcome
        has been read here and recorded: the unit check runs in the slot's call), its server is not loading, and no
        slot is being probed back after losing its process."""
        return (not station.busy and station.loading is None and not station.probing and not station.probe_at
                and not any(entry.station is station for entry in self.inflight.values()))

    def _release_idle(self) -> None:
        """Once no unit is left to start, tear down stations with nothing to do, so their GPUs go back (on Kubernetes,
        their pods end) while the last units run. Decided here, by the launch's only writer, from its own record:"""
        if self.state.pending() or time.monotonic() < self.next_release:
            return
        self.next_release = time.monotonic() + self.release_every_s
        live = [station for station in self.stations if not station.retired and not station.released]
        running = len(self.inflight)
        spare = -(-running // len(self.layout.sim_slots()))
        idle = [station for station in live if station.ready and self._idle(station)]
        unplaced = [station for station in live if not station.ready and self._idle(station)]
        if len(idle) < spare:
            return
        spares, candidates = idle[:spare], unplaced + idle[spare:]
        if not candidates and not any(station.released for station in self.stations):
            return
        for station in candidates:
            station.retiring = True
        answered = self.executor.quiet([station for station in spares + candidates if station.ready])
        spares_ok = all(station.name in answered for station in spares)
        if not spares_ok:
            self.log(f"no release: a kept spare did not answer ({', '.join(s.name for s in spares if s.name not in answered)})")
        if candidates:
            going, kept_back = [], []
            for station in candidates:
                safe = (spares_ok and not self.state.pending() and self._idle(station)
                        and (not station.ready or station.name in answered))
                (going if safe else kept_back).append(station)
            for station in kept_back:
                station.retiring = False
                self.log(f"release of {station.name} withdrawn: busy {sorted(station.busy)}, loading "
                         f"{station.loading is not None}, answered {station.name in answered}, spares answered "
                         f"{spares_ok}, pending {len(self.state.pending())}")
            if going:
                self.log(f"releasing {len(going)} stations with nothing left to run ({', '.join(s.name for s in going)}): "
                         f"no unit waiting, {running} units running on "
                         f"{sorted({entry.station.name for entry in self.inflight.values()})}, {spare} idle stations "
                         f"kept for their retries ({', '.join(s.name for s in idle[:spare]) or 'none'})")
                self.executor.tear_down(going)
                for station in going:
                    station.released, station.retiring = True, False
                    station.ready, station.placing, station.held = False, None, None
                self.log(f"released {', '.join(s.name for s in going)}; "
                         f"{sum(1 for s in self.stations if not s.retired and not s.released)} stations stay")
        if not spares_ok or not any(station.released for station in self.stations):
            return
        kept = [station for station in self.stations if not station.retired and not station.released]
        released = self.executor.release_nodes(kept)
        if released:
            self.log(f"let go of {len(released)} nodes that host none of the kept stations "
                     f"({', '.join(s.name for s in kept)}): {', '.join(released)}")

    def _release_mine(self, why: str) -> None:
        """End every attempt this driver started and saw no end of as handed back untried."""
        for unit_id, record in self.state.units.items():
            for seq in record.live():
                begin = record.begins[seq]
                if begin.get("epoch") == self.state.epoch:
                    attempt = Attempt(unit_id, seq, begin["attempt"], begin["round"], begin["station"])
                    self.state.end(attempt, "released", {"error": why})
                    self.state.settle(unit_id)


    def run(self) -> int:
        try:
            self.state.open_driver({"layout": self.layout.name, "gpus": self.executor.gpus,
                                    "executor": self.executor.kind})
        except DriverRunning as error:
            print(f"{self.output}: {error}; one driver per launch", file=sys.stderr, flush=True)
            return EXIT_DRIVER_RUNNING
        try:
            resolved = self.state.recover()
            if any(resolved.values()):
                self.log(f"recovered what the last driver left: {json.dumps(resolved)}")
            if self.retry_failed:
                self.log(f"another round for {len(self.state.retry_failed())} failed units")
            if not self.state.pending():
                self.log(f"nothing to run: {self.state.status()}")
                return 0 if not self.state.status()["failed"] else EXIT_ENDED_WITH_FAILURES
            return self._loop()
        finally:
            self.state.close_driver()

    def _loop(self) -> int:
        previous = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, _interrupt)
        last_status = 0.0
        try:
            swept = self.executor.start(self.state.epoch)
            if swept:
                self.log(f"stopped {len(swept)} processes dead slots of this launch left: {swept}")
            self.stations = [Station(index, self.executor.port_base + 10 * index)
                             for index in range(self.stations_wanted)]
            for station in self.stations:
                self._build(station)
            self.log(f"{len(self.stations)} stations of layout {self.layout.name} ({self.layout.gpus} GPU each, "
                     f"{self.layout.sims} simulators); {self.state.status()}")
            while True:
                for station in self.stations:
                    self._dispatch(station)
                if not self.inflight and not self.state.pending():
                    break
                self._release_idle()
                if all(station.retired or station.released for station in self.stations):
                    self.log(f"every station is retired or released with units left {self.state.status()}: see "
                             f"{self.output / 'logs'}, then --resume")
                    return 1
                waiting = list(self.inflight) + [s.loading for s in self.stations if s.loading is not None] + [
                    s.placing for s in self.stations if s.placing is not None and not s.ready and not s.retired] + [
                    handle for s in self.stations for handle in s.probing.values()]
                done = self.executor.wait(waiting, self.poll_s) if waiting else []
                for handle in done:
                    self._handle(handle)
                self._revoke_overdue()
                if time.monotonic() - last_status > STATUS_EVERY_S:
                    last_status = time.monotonic()
                    self.log(f"{time.strftime('%H:%M:%S')} {self.state.status()}")
        except KeyboardInterrupt:
            self.log("stopping: units under way go back untried (--resume continues)")
            self._release_mine("the driver was stopped")
            return 130
        finally:
            try:
                self.executor.tear_down(self.stations, graceful=True)
            except Exception:
                traceback.print_exc()
            signal.signal(signal.SIGTERM, previous)
            self.executor.stop()
        status = self.state.status()
        self.log(f"every unit has ended: {status}; --score next")
        return 0 if not status["failed"] else EXIT_ENDED_WITH_FAILURES

    def _handle(self, handle: Any) -> None:
        if handle in self.inflight:
            self._ended(handle)
            return
        for station in self.stations:
            for slot, probe in list(station.probing.items()):
                if handle == probe:
                    self._probed(station, slot)
                    return
            if station.loading is not None and handle == station.loading:
                self._loaded(station)
                return
            if station.placing is not None and handle == station.placing:
                station.placing = None
                self.executor.open(station)
                station.ready = True
                return


def _interrupt(signum, frame) -> None:
    raise KeyboardInterrupt


def main(argv: list[str] | None, kind: str, make_executor: Callable[..., Executor],
         check: Callable[[Mapping[str, Any]], Any], description: str,
         options: Callable[[argparse.ArgumentParser], Callable[[argparse.Namespace, dict], None]] | None = None) -> int:
    """A backend module's command line: plan or adopt the launch, then run its driver with ``make_executor(config,
    output, server_environment)``. ``options`` adds the backend's own flags to the parser and returns what applies them
    to the loaded config before it is checked (the Ray backend's ``--ray-address``)."""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    parser.add_argument("--resume", action="store_true", help="Continue a launch that has a plan.json.")
    parser.add_argument("--retry-failed", action="store_true", help="Failed units get another round.")
    parser.add_argument("--tasks", help="Comma-separated tasks: plan only these (a new launch).")
    parser.add_argument("--server-env", type=Path,
                        help="The environment the policy servers start in (env -0), saved before /isaac-sim/python.sh; "
                             "read and removed at once (it holds the shell's tokens).")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--timings", type=Path)
    apply_options = options(parser) if options is not None else None
    early = argparse.ArgumentParser(add_help=False)
    early.add_argument("--server-env", type=Path)
    saved = early.parse_known_args(argv)[0].server_env
    server_environment = server_env.read(saved) if saved else None
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if apply_options is not None:
        apply_options(args, config)
    try:
        check_recipe(config)
        check(config)
    except ValueError as error:
        parser.error(f"{args.config}: {error}")
    if not config.get("output"):
        parser.error("the config's output names the launch's directory")
    output = Path(config["output"])
    if any((output / "queue" / state).is_dir() and any((output / "queue" / state).iterdir())
           for state in ("todo", "claimed")):
        parser.error(f"{output} holds a directory backend's queue: a launch keeps the backend it started with")
    if args.resume:
        plan = adopt_plan(config, output)
    else:
        plan = write_plan(config, output, task_list(args.tasks), args.manifest,
                          load_timings(config, args.timings))
    record_launch(output, config, f"{kind} driver{' --resume' if args.resume else ''}")
    executor = make_executor(config, output, server_environment)
    return Driver(config, output, plan, executor, retry_failed=args.retry_failed).run()
