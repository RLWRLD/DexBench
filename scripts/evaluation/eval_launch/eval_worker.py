#!/usr/bin/env python3
# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""One GPU's worker: claim a unit, run it, check it, record it, and again until every unit has ended.

The same program on every backend; the backend only starts one per GPU with the launch's directory mounted::

    /isaac-sim/python.sh -m eval_launch.eval_worker --launch <output dir> --worker gpu0 --port 5810

``<output dir>`` holds ``plan.json`` (``python -m eval_launch --start`` writes it with the queue) and ``queue/``. The worker
reads the launch's config from ``plan.json`` when it starts, so a resumed launch runs with the recipe and backend
settings it was resumed with.

The policy server is the config's recipe for the model (``servers.recipe``): its setup (run in the worker's
environment, before the server; anything that can fail before a unit runs belongs here), its command, the line it
prints when ready, an optional probe, and ``client_set``, ``--set`` overrides for the simulator's policy block (e.g.
where the setup wrote the client's contract). Fields filled in (``plan.RECIPE_FIELDS``, with
``str.format``, so a literal brace is doubled; each value already shell-quoted): ``{checkpoint}``, ``{checkpoint_id}``
(a short hash of its path), ``{port}``, ``{token_file}``, ``{worker_dir}`` (this worker's own directory) and
``{launcher}``. The server runs on this worker's GPU from its first unit and is kept
for every later unit of the same checkpoint, which the worker prefers when it claims.

A unit runs ``run_benchmark_cell.py --slices`` in this tree, so the strict replay and the frozen pins are checked as
for any run. Each attempt runs into its own ``units/<unit id>/<attempt>/`` and is the unit's result only if the
worker still owns the claim when it finishes (``queue_dir``'s fencing); the done record names the accepted attempt,
the one the scorer reads. A unit is a result only when it ran within its deadline, exited 0, and passes
``unit_check`` (every Run completed with exactly its slice's episodes, all on video, every camera rendering);
otherwise it goes back to the queue (``plan.max_attempts``). Earlier attempts are kept, never deleted. An attempt that
is not a result also reports the episodes it finished (its outcome's ``salvage``), which the driver's record
accepts row by row (``acceptance``); the queue directory ignores it and reruns the whole unit.

A failure of the worker itself (its server will not start, its environment or filesystem raises) hands the unit back
untried and stops the worker; the backend starts it again. A SIGTERM stops the worker's simulator and server with it;
a worker killed outright leaves them, and the worker started in its place stops them first
(``process_group.stop_orphans``).

When nothing is left to claim, one worker (the sentinel) stays until every unit has ended, requeueing claims whose
worker died; the others exit and free their GPUs once they have seen the sentinel renew its lease.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import shlex
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import yaml
from isaaclab_arena.evaluation.arena_experiment_result import ARENA_EXPERIMENT_TIMINGS_FILENAME

from dexbench_arena import REPO_ROOT
from dexbench_arena.benchmark.models import ModelEntry, resolve_model
from dexbench_arena.benchmark.run_manifest import RUN_MANIFEST

from eval_launch import PACKAGE_DIR, gpu_memory, queue_dir, server_env
from eval_launch.plan import deadline_s, model_spec, unit_eval_version
from eval_launch.process_group import OWNER_VARIABLE, claim_ownership, stop_group, stop_orphans
from eval_launch.unit_check import check_unit, finished_episodes

RUN_CELL = REPO_ROOT / "scripts/evaluation/run_benchmark_cell.py"
HEARTBEAT_S = 30.0
STALE_S = 600.0
SERVER_READY_TIMEOUT_S = 1800.0
SENTINEL_POLL_S = 30.0


class WorkerFailure(RuntimeError):
    """This worker cannot run units (its server, environment or filesystem): the unit goes back untried."""


def ensure_dir(path: Path, attempts: int = 10) -> None:
    """``path`` as a directory, also when another host made it a moment ago: on the shared NFS volume a pod can be
    told it exists by ``mkdir`` and that it does not by ``stat`` (2026-11-02: four pods failed 60 units in 42 s
    with FileExistsError on a ``units/`` another pod had just made)."""
    for attempt in range(attempts):
        try:
            path.mkdir(parents=True, exist_ok=True)
            return
        except FileExistsError:
            if attempt == attempts - 1:
                raise
            time.sleep(1.0)


def _run_bounded(command: str, environment: Mapping[str, str] | None, timeout_s: float, log: Path) -> int | None:
    """Run a recipe step in its own process group; its return code, or None when it ran past ``timeout_s`` (then
    its whole group is stopped)."""
    with log.open("ab") as out:
        process = subprocess.Popen(["bash", "-c", command], env=environment, stdout=out, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            return process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            stop_group(process, grace_s=10.0)
            return None
        except BaseException:
            stop_group(process, grace_s=10.0)
            raise


class PolicyServer:
    """The model's server on this worker's GPU, from the config's recipe; one checkpoint at a time."""

    def __init__(self, recipe: Mapping[str, Any], port: int, log_dir: Path, environment: Mapping[str, str] | None = None,
                 ready_timeout_s: float = SERVER_READY_TIMEOUT_S, gpu: str | None = None):
        self.recipe, self.port, self.ready_timeout_s = recipe, port, ready_timeout_s
        self.log_dir = Path(os.path.abspath(log_dir))
        self.environment = dict(environment) if environment is not None else None
        for name in ("CUDA_VISIBLE_DEVICES", OWNER_VARIABLE):
            if self.environment is not None and name in os.environ:
                self.environment[name] = os.environ[name]
        if gpu is not None:
            self.environment = {**(self.environment if self.environment is not None else os.environ),
                                "CUDA_VISIBLE_DEVICES": str(gpu)}
        self.checkpoint: str | None = None
        self.process: subprocess.Popen | None = None
        self.token = ""
        self.token_file: Path | None = None
        self.started_s = 0.0
        self.starts = 0
        self.incarnation = f"{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}-{secrets.token_hex(16)}"
        self.fields: dict[str, Any] = {}

    def client_set(self) -> list[str]:
        """The recipe's ``--set`` overrides for the simulator's policy block, filled in for this worker."""
        return [item.format(**self.fields) for item in self.recipe.get("client_set", [])]

    def ensure(self, checkpoint: str) -> None:
        if self.checkpoint == checkpoint and self.process is not None and self.process.poll() is None:
            return
        self.stop()
        try:
            self._start(checkpoint)
        except BaseException:
            self.stop()
            raise

    def _write_token(self) -> Path:
        """A new token in a file created readable by its owner only (never readable by others, not even for a moment),
        which :meth:`stop` removes."""
        token_file = self.log_dir / f"token-{self.port}"
        token_file.unlink(missing_ok=True)
        self.token = secrets.token_hex(24)
        descriptor = os.open(token_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self.token_file = token_file
        with os.fdopen(descriptor, "w") as handle:
            handle.write(self.token)
        return token_file

    def _start(self, checkpoint: str) -> None:
        start = time.monotonic()
        ensure_dir(self.log_dir)
        token_file = self._write_token()
        served = os.path.abspath(checkpoint) if not os.path.isabs(checkpoint) and os.path.exists(checkpoint) else checkpoint
        self.fields = {"checkpoint": shlex.quote(served), "port": self.port,
                       "checkpoint_id": hashlib.sha256(checkpoint.encode()).hexdigest()[:12],
                       "token_file": shlex.quote(str(token_file)), "worker_dir": shlex.quote(str(self.log_dir)),
                       "launcher": shlex.quote(str(PACKAGE_DIR))}
        self.starts += 1
        log = self.log_dir / f"server-{self.port}-{self.incarnation}-{self.starts:02d}.log"
        log.open("xb").close()
        if self.recipe.get("setup"):
            returncode = _run_bounded(self.recipe["setup"].format(**self.fields), None, self.ready_timeout_s, log)
            if returncode != 0:
                raise WorkerFailure(f"the recipe's setup {'timed out' if returncode is None else 'failed'} for "
                                    f"{checkpoint}; see {log}")
        server_from = log.stat().st_size
        self.process = subprocess.Popen(["bash", "-c", self.recipe["command"].format(**self.fields)],
                                        env=self.environment, stdout=log.open("ab"), stderr=subprocess.STDOUT,
                                        start_new_session=True)
        ready = re.compile(self.recipe.get("ready", "Server ready"))
        while not ready.search(log.read_bytes()[server_from:].decode(errors="replace")):
            if self.process.poll() is not None:
                raise WorkerFailure(f"policy server exited ({self.process.returncode}); see {log}")
            if time.monotonic() - start > self.ready_timeout_s:
                self.stop()
                raise WorkerFailure(f"policy server not ready after {self.ready_timeout_s:.0f} s; see {log}")
            time.sleep(5)
        if self.recipe.get("probe"):
            returncode = _run_bounded(self.recipe["probe"].format(**self.fields), self.environment, 120.0, log)
            if returncode != 0:
                self.stop()
                raise WorkerFailure(f"policy server printed its ready line but failed the recipe's probe; see {log}")
        if self.process.poll() is not None:
            raise WorkerFailure(f"policy server exited ({self.process.returncode}) after its ready line; see {log}")
        self.checkpoint, self.started_s = checkpoint, time.monotonic() - start

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            stop_group(self.process, grace_s=60.0)
        self.process, self.checkpoint = None, None
        if self.token_file is not None:
            self.token_file.unlink(missing_ok=True)
            self.token_file = None


def _rates(unit_dir: Path) -> dict[str, Any]:
    """Every rate (a ``*_hz`` setting) the unit's composed experiment states, with the values it takes, and the
    run's horizon: what the unit actually ran at."""
    found: dict[str, set] = {}

    def walk(node: Any) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                if str(key).endswith("_hz") and not isinstance(value, (Mapping, list)):
                    found.setdefault(str(key), set()).add(value)
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    composed = unit_dir / "composed_experiment.yaml"
    if composed.is_file():
        walk(yaml.safe_load(composed.read_text()))
    rates: dict[str, Any] = {key: sorted(values, key=str) for key, values in sorted(found.items())}
    manifest = unit_dir / "run_manifest.json"
    if manifest.is_file():
        rates["horizon"] = json.loads(manifest.read_text()).get("horizon")
    return rates


def _stepping(unit_dir: Path) -> dict[str, float]:
    """Seconds spent stepping and steps taken, from the experiment's timings Arena aggregates over its Runs."""
    for row in json.loads((unit_dir / "arena" / ARENA_EXPERIMENT_TIMINGS_FILENAME).read_text()).get("totals", []):
        if row.get("name") == "step":
            return {"stepping_s": row["total_ms"] / 1000.0, "steps": row["count"]}
    return {"stepping_s": 0.0, "steps": 0}


def _keep_claimed(claim_path: Path, stop: threading.Event) -> None:
    """Touch the claim every ``HEARTBEAT_S`` until ``stop``; a claim no longer this attempt's is left alone."""
    while not stop.wait(HEARTBEAT_S):
        try:
            queue_dir.heartbeat(claim_path)
        except FileNotFoundError:
            return


def server_address(model: ModelEntry, port: int) -> list[str]:
    """The ``--set`` that points ``model``'s client at this worker's server: ``remote_port`` for a client that takes a
    port (GR00T, RLDX, openpi), ``remote_url`` for one that takes a URL (the HTTP contract: Cosmos3 and the models of
    CONNECT_YOUR_POLICY.md), whichever its registry block has; none for a block with neither. A recipe's
    ``client_set`` comes after it and wins (the planner refuses a model with neither and no ``client_set`` for it).
    Until 2026-11-02 every unit got ``remote_port``, which a ``remote_url`` block refuses, so Cosmos3 could not run
    under the launcher. The URL keeps everything but its port; an IPv6 host keeps its brackets."""
    policy = model.policy or {}
    if "remote_url" in policy:
        url = urlsplit(str(policy["remote_url"]))
        host = f"[{url.hostname}]" if ":" in (url.hostname or "") else url.hostname
        return ["--set", f"remote_url={urlunsplit(url._replace(netloc=f'{host}:{port}'))}"]
    if "remote_port" in policy:
        return ["--set", f"remote_port={port}"]
    return []


LOG_TAIL_LINES = 20


def _log_tail(log: Path, lines: int = LOG_TAIL_LINES) -> list[str]:
    """The last ``lines`` lines of a unit's log: a failed attempt's cause as the runner printed it (a traceback, a
    simulator state gone non-finite), kept in its outcome, since a later attempt may be the one scored."""
    try:
        with log.open("rb") as handle:
            handle.seek(max(0, log.stat().st_size - 64 * 1024))
            return handle.read().decode(errors="replace").splitlines()[-lines:]
    except OSError as error:
        return [f"unreadable log: {error!r}"]


def run_unit(unit: Mapping[str, Any], unit_dir: Path, server: PolicyServer, model: str | Mapping[str, Any],
             deadline: float | None = None, environment: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Run one unit with ``run_benchmark_cell.py --slices`` into ``unit_dir``, stopped after ``deadline`` seconds,
    and say whether it is a result. ``model`` is ``plan.model_spec``: a registered model's key, or the entry the config
    registers, passed on as ``--model-entry``. ``environment`` is added to the simulator's (the config's
    ``backend.env``)."""
    if unit.get("cell") is None or not unit.get("slices"):
        return {"rc": None, "error": "not a cell unit: the launcher runs cells only (the sweep is not runnable yet)"}
    entry = resolve_model(model)
    command = [sys.executable, str(RUN_CELL), "--task", unit["task"], "--cell", unit["cell"],
               "--slices", json.dumps(unit["slices"]), "--version", unit_eval_version(unit),
               "--model", entry.key, *(["--model-entry", json.dumps(dict(model))] if isinstance(model, Mapping) else []),
               "--output", str(unit_dir), "--checkpoint", unit["checkpoint"], *server_address(entry, server.port)]
    if unit.get("submission", False):
        command += ["--submission", "--checkpoint-sha256", str(unit["checkpoint_sha256"])]
    for item in server.client_set():
        command += ["--set", item]
    command += ["--", "--viz", "none"]
    child_environment = {**os.environ, **dict(environment or {}), "DEXBENCH_POLICY_API_TOKEN": server.token}
    ensure_dir(unit_dir.parent)
    start = time.monotonic()
    log_path = unit_dir.parent / f"{unit_dir.name}.log"
    memory = gpu_memory.Peak(os.environ.get("CUDA_VISIBLE_DEVICES")).start()
    with log_path.open("ab") as log:
        process = subprocess.Popen(command, cwd=REPO_ROOT, env=child_environment, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            returncode = process.wait(timeout=deadline)
            timed_out = False
        except subprocess.TimeoutExpired:
            stop_group(process, grace_s=30.0)
            returncode, timed_out = process.returncode, True
        except BaseException:
            stop_group(process, grace_s=5.0)
            raise
        finally:
            gpu_memory_peak = memory.stop()
    if returncode != 0 and not timed_out and not (unit_dir / RUN_MANIFEST).exists():
        raise WorkerFailure(f"run_benchmark_cell.py refused {unit['id']} before running it (tree, frozen rows or "
                            f"model); see {log_path}")
    outcome: dict[str, Any] = {"rc": returncode, "wall_s": time.monotonic() - start, "gpu_memory": gpu_memory_peak}
    if timed_out:
        outcome.update(error=f"stopped at its deadline, {deadline:.0f} s", log_tail=_log_tail(log_path),
                       salvage=_salvage(unit_dir, unit["slices"]))
        return outcome
    try:
        outcome.update(_stepping(unit_dir), rates=_rates(unit_dir))
    except (OSError, ValueError, yaml.YAMLError) as error:
        outcome["metadata_error"] = repr(error)
    if returncode == 0:
        outcome.update(check_unit(unit_dir, unit["slices"]))
    else:
        outcome["complete"] = False
        stop_group(process, grace_s=30.0)
    if not outcome["complete"]:
        outcome["log_tail"] = _log_tail(log_path)
        outcome["salvage"] = _salvage(unit_dir, unit["slices"])
    return outcome


def _salvage(unit_dir: Path, slices: Mapping[str, Any]) -> dict[str, Any]:
    """The episodes an attempt that did not complete finished (``unit_check.finished_episodes``), which the driver's
    record accepts row by row (``acceptance``); what the unit wrote is the unit's, so an error reading it is reported,
    never raised (the record then charges every row of the attempt, as for an attempt that finished nothing)."""
    try:
        return finished_episodes(unit_dir, slices)
    except Exception as error:
        return {"error": repr(error)[:500]}


SENTINEL = "sentinel"


def _hold_sentinel(queue: Path, worker: str, lease_s: float) -> bool:
    """Whether this worker is the launch's sentinel: the one that stays once nothing is left to claim. Taken by
    creating ``queue/sentinel`` and kept by touching it every poll; a lease not renewed for ``lease_s`` (its worker
    died) is taken over. Two workers taking over at once may both stay; that costs a GPU, never a unit."""
    path = queue / SENTINEL
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            holder, age = path.read_text(), queue_dir.fs_now(queue) - path.stat().st_mtime
        except FileNotFoundError:
            return False
        if holder == worker:
            os.utime(path)
            return True
        if age < lease_s:
            return False
        temporary = queue / f".sentinel.{worker}.tmp"
        temporary.write_text(worker)
        os.replace(temporary, path)
        return True
    os.write(descriptor, worker.encode())
    os.close(descriptor)
    return True


def _sentinel_renews(queue: Path, poll_s: float) -> bool:
    """Whether the sentinel renews its lease within a poll and a half: then this worker may leave the tail to it."""
    path = queue / SENTINEL
    try:
        before = path.stat().st_mtime
    except FileNotFoundError:
        return False
    time.sleep(1.5 * poll_s)
    try:
        return path.stat().st_mtime > before
    except FileNotFoundError:
        return False


def run_worker(launch: Path, worker: str, server: PolicyServer, model: str | Mapping[str, Any],
               run: Callable[..., dict[str, Any]] = run_unit, stale_s: float = STALE_S, max_attempts: int = 2,
               plan_config: Mapping[str, Any] | None = None, poll_s: float = SENTINEL_POLL_S,
               environment: Mapping[str, str] | None = None) -> int:
    """Claim and run units until every unit has ended (as the sentinel) or nothing is left to claim and the
    sentinel is alive (as any other worker); returns how many of this worker's units failed for good."""
    queue = launch / "queue"
    failed = 0
    while True:
        try:
            queue_dir.requeue_stale(queue, stale_s, max_attempts=max_attempts)
        except OSError as error:
            if not queue_dir.lost_race(error):
                raise
            print(f"[eval_worker {worker}] recovery scan lost a race and moves on: {error}", file=sys.stderr, flush=True)
        held = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
        try:
            claimed = queue_dir.claim(queue, worker, prefer={"checkpoint": server.checkpoint} if server.checkpoint else None)
            if claimed is not None:
                _HELD[:] = [(queue, claimed[0])]
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, held)
        if claimed is None:
            if queue_dir.terminal(queue):
                return failed
            if _hold_sentinel(queue, worker, lease_s=3.0 * poll_s):
                time.sleep(poll_s)
            elif _sentinel_renews(queue, poll_s):
                return failed
            continue
        claim_path, record = claimed
        unit = record["unit"]
        unit_dir = launch / "units" / unit["id"] / record["attempt"]
        stop = threading.Event()
        threading.Thread(target=_keep_claimed, args=(claim_path, stop), daemon=True).start()
        try:
            server.ensure(unit["checkpoint"])
            outcome = run(unit, unit_dir, server, model, deadline_s(unit, plan_config), environment)
            outcome = {**outcome, "output": str(unit_dir), "server_load_s": server.started_s, "worker": worker}
        except Exception as error:
            stop.set()
            queue_dir.release(queue, claim_path)
            print(traceback.format_exc(), file=sys.stderr, flush=True)
            raise SystemExit(f"[eval_worker {worker}] stopping: {error!r}; {unit['id']} handed back untried")
        except BaseException:
            stop.set()
            queue_dir.release(queue, claim_path)
            raise
        finally:
            stop.set()
        ok = outcome.get("rc") == 0 and outcome.get("complete") is True
        _HELD.clear()
        state = queue_dir.finish(queue, claim_path, outcome, ok=ok, max_attempts=max_attempts)
        failed += state == "failed"
        print(f"[eval_worker {worker}] {unit['id']} {record['attempt']}: {state} {json.dumps(outcome)[:300]}", flush=True)


def wait_for_queue(launch: Path, timeout_s: float = 3600.0) -> dict:
    """Wait until ``python -m eval_launch --start`` has published the plan and queued every unit of it (a backend starts
    the workers and the start together), so no worker finds a half-filled queue empty and stops; returns the plan."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        plan_path = launch / "plan.json"
        if plan_path.is_file() and (launch / "queue").is_dir():
            plan = json.loads(plan_path.read_text())
            queued = set().union(*(queue_dir.unit_ids(launch / "queue", state) for state in queue_dir.STATES))
            if {unit["id"] for unit in plan["units"]} <= queued:
                return plan
        time.sleep(10)
    raise SystemExit(f"{launch}: no complete plan and queue after {timeout_s:.0f} s (python -m eval_launch --start)")


_HELD: list[tuple[Path, Path]] = []
"""The claim this worker holds while it runs a unit, for the SIGTERM handler."""


def _stop_on_sigterm(signum, frame) -> None:
    """Hand the unit back first, then unwind: a pod gets SIGKILL a grace period after SIGTERM, and stopping the
    simulator and server takes part of it (2026-11-02 rehearsal: a unit not yet handed back at SIGKILL waited out
    the 10 min stale timeout). The unwinding stops the simulator (run_unit) and the server (main's finally)."""
    for queue, claim_path in _HELD:
        queue_dir.release(queue, claim_path)
    _HELD.clear()
    raise SystemExit(128 + signum)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--launch", type=Path, required=True, help="The launch's output directory (plan.json, queue/).")
    parser.add_argument("--worker", required=True, help="This worker's name, unique in the launch (e.g. gpu0).")
    parser.add_argument("--port", type=int, required=True, help="The port this worker's policy server listens on.")
    parser.add_argument("--server-env", type=Path,
                        help="The environment to start the policy server in (env -0), saved by the backend before "
                             "Isaac Sim's launcher changed it, and removed once read (it may hold tokens); without "
                             "it the server inherits the worker's.")
    args = parser.parse_args(argv)
    signal.signal(signal.SIGTERM, _stop_on_sigterm)
    scope = f"{args.launch.resolve()}:{args.worker}"
    orphans = stop_orphans(scope)
    if orphans:
        print(f"[eval_worker {args.worker}] stopped {len(orphans)} processes a killed worker left: {orphans}", flush=True)
    claim_ownership(scope)
    plan = wait_for_queue(args.launch)
    config = plan["config"]
    recipe = (config.get("servers") or {}).get("recipe")
    if not recipe or "command" not in recipe:
        parser.error("the config's servers.recipe gives the policy server's command (and its ready line)")
    plan_config = config.get("plan") or {}
    server = PolicyServer(recipe, args.port, args.launch / "logs" / args.worker,
                          server_env.read(args.server_env) if args.server_env else None,
                          ready_timeout_s=float(plan_config.get("server_ready_timeout_s", SERVER_READY_TIMEOUT_S)))
    try:
        return 1 if run_worker(args.launch, args.worker, server, model_spec(config), plan_config=plan_config,
                               max_attempts=int(plan_config.get("max_attempts", 2)),
                               environment={str(k): str(v) for k, v in ((config.get("backend") or {}).get("env") or {}).items()}) else 0
    finally:
        server.stop()


if __name__ == "__main__":
    sys.exit(main())
