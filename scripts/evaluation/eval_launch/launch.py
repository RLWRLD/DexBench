#!/usr/bin/env python3
# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Launch a benchmark evaluation from one config: plan, queue the units, and follow the queue.

Kit-free. The workers (``eval_launch.eval_worker``, one per GPU) run the units; a backend starts them::

    python -m eval_launch <config.yaml> --plan-only [--write-plan plan.json]   # the units and the estimate; nothing written
    python -m eval_launch <config.yaml> --start      # <output>/plan.json and <output>/queue/; again: resumes, adds nothing
    python -m eval_launch <config.yaml> --start --tasks <task>,<task>   # plan these tasks too, queued behind the others
    python -m eval_launch <config.yaml> --status     # todo / claimed / done / failed, and why each failed unit failed
    python -m eval_launch <config.yaml> --local      # this host: --start, then one worker per GPU, until every unit ends
    python -m eval_launch <config.yaml> --resume [--retry-failed] [--requeue-claims]   # pick up a stopped launch
    python -m eval_launch <config.yaml> --score      # score the accepted units, per task, with the benchmark scorer
    python -m eval_launch <config.yaml> --smoke [--smoke-rows 2]   # one station, a few rows per simulator: pass or fail
                                                                    # on frames, timing and GPU memory (smoke.py)
    python -m eval_launch <config.yaml> --k8s-job <dir>   # <dir>/job.json, prepull.json and eval.yaml for the k8s backend
                                                          # (a Ray config: <dir>/ray.json, its Ray cluster as pods)

A launch is resumable by construction: its state is the queue in ``<output>/queue`` on a filesystem every worker sees.
A unit in ``done/`` never runs again, so a launch stopped for any reason (Ctrl-C, a lost node, a deleted Job, a
deadline) continues where it was with ``--resume``: claims whose worker is gone go back to the queue (after their
heartbeat goes stale, or at once with ``--requeue-claims`` when no worker is running), ``--retry-failed`` gives
failed units another round, and the local backend starts its workers again (on k8s, apply the Job again).

With ``backend.kind: local`` or ``ray``, ``--local`` and ``--resume`` run the launch's driver instead
(``python -m eval_launch.local_backend`` on this machine's GPUs, ``python -m eval_launch.ray_backend`` on stations Ray
places): no queue directory, the servers and simulators go where the config's ``servers.layout`` says, and the launch's
record is ``accepted/`` and ``state/`` (:mod:`eval_launch.acceptance`), which ``--status`` and ``--score`` read. A
local launch started before the driver (its units in ``queue/todo``) resumes with one worker per GPU, as it started.

Run with /isaac-sim/python.sh (or any Python with the dexbench_arena package importable; launch.sh picks one). How to
use it: USER_GUIDE.md here.
"""

from __future__ import annotations

import argparse
import json
import re
import os
import signal
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from dexbench_arena import REPO_ROOT
from dexbench_arena.benchmark.run_manifest import RUN_MANIFEST
from dexbench_arena.evaluation.evaluation_signature import code_revision

from eval_launch import PACKAGE_DIR, STATUS_VARIABLE, queue_dir, server_env, smoke, status_file
from eval_launch.acceptance import STATE, LaunchState, create_once, mapping_rows, rows_mapping, slice_rows
from eval_launch.k8s_backend import config_on_volume, render_job, render_prepull
from eval_launch.ray_k8s import render_ray
from eval_launch.layout import COLOCATED, parse_layout
from eval_launch.plan import (MANIFEST, RETIRED_KEYS, add_to_plan, check_inputs, check_recipe, eval_version, load_config,
                              load_timings, make_plan, missing_inputs, plan_eval_version, summary, task_list)
from eval_launch.process_group import stop_groups


PLANNING_KEYS = ("model", "tasks", "cells", "sweep", "version", "plan")
"""What a launch's units depend on. A launch resumes under a config that differs elsewhere (a fixed server recipe, a
newer tree, another backend or GPU count), never under one that would plan other units."""


def _publish(path: Path, data: dict) -> None:
    """Write ``path`` whole or not at all: workers read the plan while a launcher may be rewriting it."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=1) + "\n")
    os.replace(temporary, path)


LAUNCHES = "launches"
"""``<output>/launches/<UTC time>-<id>/``: one record per start of a launch (``--start``, ``--local``, ``--resume``, a
driver's start, and ``--k8s-job`` beside what it renders), each file created once: ``launch.json`` (the mode, the
command line, the source commit as ``code_revision`` reads it, the time) and ``config.json`` (the config as that start
resolved it), and for ``--k8s-job`` the rendered manifests. Environment maps keep their variable names only."""
ENV_KEYS = {"env", "env_vars", "secret_env"}
"""Keys whose values are environment variables (``backend.env``, ``backend.k8s.env``, ``backend.k8s.secret_env``,
Ray's ``runtime_env.env_vars``, a pod's ``env``): a launch record lists their names, never a value."""


def names_only(value: Any, key: str | None = None) -> Any:
    """``value`` with every environment map or list under :data:`ENV_KEYS` reduced to its sorted variable names."""
    if key in ENV_KEYS and isinstance(value, Mapping):
        return sorted(str(name) for name in value)
    if key in ENV_KEYS and isinstance(value, list):
        return sorted(str(item.get("name") if isinstance(item, Mapping) else item).split("=", 1)[0] for item in value)
    if isinstance(value, Mapping):
        return {name: names_only(item, str(name)) for name, item in value.items()}
    if isinstance(value, list):
        return [names_only(item) for item in value]
    return value


def record_launch(root: Path, config: Mapping[str, Any], mode: str,
                  manifests: Mapping[str, Mapping[str, Any]] | None = None) -> Path:
    """Write this start's record under ``root/launches`` (:data:`LAUNCHES`); returns its directory. Never a secret: the
    servers' environment is not in the config, and environment maps keep their names only (:func:`names_only`)."""
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    directory = root / LAUNCHES / f"{stamp}-{os.getpid()}-{os.urandom(3).hex()}"
    create_once(directory / "launch.json", {"mode": mode, "argv": list(sys.argv), "code": code_revision(REPO_ROOT),
                                            "started_at": stamp, "output": str(config.get("output"))})
    create_once(directory / "config.json", names_only(dict(config)))
    for name, manifest in (manifests or {}).items():
        create_once(directory / name, names_only(dict(manifest)))
    return directory


def _adopt(plan: dict, config: dict, plan_path: Path) -> dict:
    """``plan`` with ``config``'s runtime settings (everything but the planning keys): what the workers read when they
    start, so a resume runs with the recipe and backend it was resumed with. Refuses a config that would plan
    other units."""
    changed = [key for key in PLANNING_KEYS if key not in ("tasks", "version") and plan["config"].get(key) != config.get(key)]
    if plan_eval_version(plan) != eval_version(config):
        changed.append(f"version ({plan_eval_version(plan)}, the config asks for {eval_version(config)})")
    if changed:
        raise SystemExit(f"{plan_path} was planned with another {', '.join(changed)}; start a new output directory")
    runtime = {key: value for key, value in config.items() if key not in PLANNING_KEYS}
    plan = {key: value for key, value in plan.items() if key != "instructions"}
    return {**plan, "config": {**{k: v for k, v in plan["config"].items() if k in PLANNING_KEYS}, **runtime}}


TREE = REPO_ROOT
"""The tree this launcher runs from, which every unit's simulator runs in (the staged tree, on a cluster)."""


def launch_tree(config: Mapping[str, Any]) -> Path | None:
    """The tree the launch's units will read: ``backend.k8s.tree_dir`` for a Kubernetes launch (the staged tree on the
    shared volume) when this host sees it, None when it does not; :data:`TREE` otherwise."""
    tree_dir = ((config.get("backend") or {}).get("k8s") or {}).get("tree_dir")
    if tree_dir is None:
        return TREE
    return Path(str(tree_dir)) if Path(str(tree_dir)).is_dir() else None


def _missing_note(config: Mapping[str, Any], manifest: Path, states: Mapping[str, Any]) -> tuple[Path | None, dict]:
    """The tree the launch reads (:func:`launch_tree`) and what it lacks of the planned tasks' inputs (empty when this
    host cannot see it)."""
    tree = launch_tree(config)
    return tree, (missing_inputs(json.loads(Path(manifest).read_text()), states, tree) if tree is not None else {})


PLACEHOLDER = re.compile(r"<[A-Za-z][^<>]*>")
"""A value an example left for the user to fill in: ``<namespace>``, ``<registry>/...@sha256:<digest>``."""


def placeholders(config: Mapping[str, Any]) -> list[str]:
    """The cluster settings (``backend.image``, ``backend.k8s``, from the config or its site profile) that still hold
    an example's ``<...>`` placeholder, as ``key: value``."""
    found: list[str] = []

    def walk(value: Any, where: str) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                walk(item, f"{where}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{where}[{index}]")
        elif isinstance(value, str) and PLACEHOLDER.search(value):
            found.append(f"{where}: {value}")

    backend = config.get("backend") or {}
    walk({key: backend[key] for key in ("image", "k8s") if key in backend}, "backend")
    return found


def staged_commit_mismatch(config: Mapping[str, Any], tree: Path | None) -> str | None:
    """Why the pods would refuse the visible staged ``tree`` (its ``.source-commit`` is not ``backend.k8s.commit``,
    which every pod checks first), or None."""
    commit = ((config.get("backend") or {}).get("k8s") or {}).get("commit")
    if tree is None or tree == TREE or not commit:
        return None
    marker = tree / ".source-commit"
    staged = marker.read_text().strip() if marker.is_file() else None
    if staged != str(commit):
        return (f"{tree} is staged at {staged or 'no commit (no .source-commit)'}, the config's backend.k8s.commit is "
                f"{commit}: every pod would stop at its tree check")
    return None


def _check_tree(manifest: Path, states: dict) -> None:
    try:
        check_inputs(json.loads(Path(manifest).read_text()), states, TREE)
    except ValueError as error:
        raise SystemExit(f"refusing to plan: {error}") from None


def write_plan(config: dict, output: Path, tasks: list[str] | None, manifest: Path, timings: dict | None) -> dict:
    """Write or extend ``<output>/plan.json`` (and make the directories the units and logs go to); returns the plan.
    Tasks planned here are refused when :data:`TREE` lacks a file their units read (``plan.check_inputs``)."""
    plan_path = output / "plan.json"
    if plan_path.is_file():
        plan = _adopt(json.loads(plan_path.read_text()), config, plan_path)
        new = [task for task in (tasks or []) if task not in plan["tasks"]]
        if new:
            added = add_to_plan(plan, config, new, manifest, timings)
            _check_tree(manifest, {task: added["tasks"][task] for task in new})
            plan = added
    else:
        plan = make_plan(config if not tasks else {**config, "tasks": tasks}, manifest, timings)
        _check_tree(manifest, plan["tasks"])
    for directory in (output, output / "units", output / "logs"):
        directory.mkdir(parents=True, exist_ok=True)
    _publish(plan_path, plan)
    print(summary(plan))
    return plan


def _start(config: dict, output: Path, tasks: list[str] | None, manifest: Path, timings: dict | None) -> dict:
    """:func:`write_plan`, then put every planned unit the queue lacks into ``<output>/queue``."""
    plan = write_plan(config, output, tasks, manifest, timings)
    added = queue_dir.init(output / "queue", plan["units"])
    print(f"queued {added} new units in {output / 'queue'}; {queue_dir.status(output / 'queue')}")
    return plan


def adopt_plan(config: dict, output: Path) -> dict:
    """A stopped launch's plan with ``config``'s runtime settings, published for its workers; returns it."""
    plan_path = output / "plan.json"
    if not plan_path.is_file():
        raise SystemExit(f"{output} has no plan.json: nothing to resume (--start or --local begins a launch)")
    plan = _adopt(json.loads(plan_path.read_text()), config, plan_path)
    _publish(plan_path, plan)
    return plan


def ray_state(output: Path) -> LaunchState | None:
    """The Ray backend's record of the launch at ``output``, or None for a launch of another backend."""
    plan_path = output / "plan.json"
    if not (output / STATE / "drivers").is_dir() or not plan_path.is_file():
        return None
    plan = json.loads(plan_path.read_text())
    return LaunchState(output, plan["units"], int((plan["config"].get("plan") or {}).get("max_attempts", 2)))


def _status(output: Path) -> None:
    state = ray_state(output)
    if state is not None:
        print(f"driver: {state.status()}")
        records = state.failed_records()
    else:
        print(queue_dir.status(output / "queue"))
        records = [json.loads(path.read_text()) for path in sorted(p for p in (output / "queue" / "failed").glob("*.json")
                                                                    if not p.name.startswith("."))]
    for record in records:
        last = (record.get("outcomes") or [{}])[-1]
        rows = f" (failed rows {json.dumps(record['failed_rows'])})" if record.get("failed_rows") else ""
        print(f"failed {record['unit']['id']} after {record['attempts']} attempts{rows}: "
              f"{last.get('error') or 'rc ' + str(last.get('rc'))} {'; '.join(last.get('problems', []))[:300]}")


SCORER = REPO_ROOT / "scripts/evaluation/score_benchmark_cells.py"
WORKER_MODULE = "eval_launch.eval_worker"
"""What the local backend runs once per GPU (``python -m``)."""
RAY_MODULE = "eval_launch.ray_backend"
"""What runs a ``backend.kind: ray`` launch (``python -m``): its own module, so this one runs without Ray installed."""
DRIVER_MODULES = {"local": "eval_launch.local_backend", "ray": RAY_MODULE}
"""The driver module of each backend that runs a launch with one driver (``python -m``)."""


def queue_launch(output: Path) -> bool:
    """Whether ``output`` is a launch of the directory backend (its units queued in ``queue/todo``), which keeps it."""
    return (output / "queue" / "todo").is_dir() and ray_state(output) is None


def _resume(config: dict, output: Path, retry_failed: bool, requeue_claims: bool, max_attempts: int) -> None:
    """Get a stopped launch ready to run again: the plan takes the config's runtime settings, every planned unit the
    queue lacks (a start that died half way) is queued, and claims or failed units go back as asked."""
    queue = output / "queue"
    plan = adopt_plan(config, output)
    added = queue_dir.init(queue, plan["units"])
    if added:
        print(f"queued {added} planned units the queue lacked")
    if requeue_claims:
        moved = queue_dir.requeue_stale(queue, stale_s=0.0, max_attempts=max_attempts)
        print(f"requeued {len(moved)} claims")
    if retry_failed:
        print(f"back to the queue for another round: {len(queue_dir.retry_failed(queue))} failed units")
    print(queue_dir.status(queue))


def _gpus(backend: dict) -> list[str]:
    gpus = backend.get("gpus", 1)
    return [str(gpu) for gpu in gpus] if isinstance(gpus, list) else [str(index) for index in range(int(gpus))]


def run_local(config: dict, output: Path, server_environment: Mapping[str, str] | None, poll_s: float = 30.0) -> int:
    """The local backend: one worker per GPU of this host, each its own process group, restarted when it dies while
    units remain (``backend.local.restarts`` times per GPU); returns once every unit is done or failed. Ctrl-C or
    SIGTERM stops the workers and leaves the queue as it is: --resume picks it up. Each worker start gets its own
    copy of ``server_environment`` in a file it removes as it reads it (:mod:`eval_launch.server_env`); one a worker
    never read is removed here when the launcher returns."""
    backend = config["backend"]
    local = backend.get("local") or {}
    port_base, restarts = int(local.get("port_base", 5810)), int(local.get("restarts", 2))
    pythonpath = os.pathsep.join(str(path) for path in backend.get("pythonpath", []))
    queue = output / "queue"
    (output / "logs").mkdir(parents=True, exist_ok=True)
    workers: dict[str, dict] = {}
    stop = {"starting": False, "pending": False}
    environment_files: list[Path] = []

    def start(gpu: str, index: int) -> None:
        name = f"local{gpu}"
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
        environment["PYTHONPATH"] = os.pathsep.join(filter(None, (str(PACKAGE_DIR.parent), environment.get("PYTHONPATH"),
                                                                  pythonpath)))
        command = [sys.executable, "-m", WORKER_MODULE, "--launch", str(output), "--worker", name,
                   "--port", str(port_base + 10 * index)]
        if server_environment is not None:
            environment_files.append(server_env.write(server_environment))
            command += ["--server-env", str(environment_files[-1])]
        log = (output / "logs" / f"{name}.log").open("ab")
        stop["starting"] = True
        try:
            process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            entry = workers.setdefault(gpu, {"starts": 0})
            entry.update(process=process, starts=entry["starts"] + 1)
        finally:
            stop["starting"] = False
        if stop["pending"]:
            raise KeyboardInterrupt

    def interrupt(signum, frame):
        if stop["starting"]:
            stop["pending"] = True
            return
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    gpus = _gpus(backend)
    try:
        for index, gpu in enumerate(gpus):
            start(gpu, index)
        print(f"started {len(gpus)} workers on GPUs {', '.join(gpus)}; logs in {output / 'logs'}", flush=True)
        while True:
            time.sleep(poll_s)
            counts = queue_dir.status(queue)
            print(f"[{time.strftime('%H:%M:%S')}] {counts}", flush=True)
            running = [gpu for gpu, entry in workers.items() if entry["process"].poll() is None]
            if queue_dir.terminal(queue) and not running:
                break
            for index, gpu in enumerate(gpus):
                entry = workers[gpu]
                if entry["process"].poll() is None or entry["process"].returncode == 0 and not counts["todo"]:
                    continue
                if counts["todo"] and entry["starts"] <= restarts:
                    print(f"worker local{gpu} exited ({entry['process'].returncode}) with units waiting: starting it again",
                          flush=True)
                    start(gpu, index)
            if not any(entry["process"].poll() is None for entry in workers.values()) and not queue_dir.terminal(queue):
                print(f"every worker has stopped and units remain {queue_dir.status(queue)}: see {output / 'logs'}, then "
                      "--resume (--requeue-claims, as no worker runs)", flush=True)
                return 1
    except KeyboardInterrupt:
        print("stopping the workers; the queue stays as it is (--resume --requeue-claims to continue)", flush=True)
        stop_groups([entry["process"] for entry in workers.values()], grace_s=120.0)
        return 130
    finally:
        for path in environment_files:
            path.unlink(missing_ok=True)
    print(f"every unit has ended: {queue_dir.status(queue)}; --score next", flush=True)
    return 0 if not queue_dir.status(queue)["failed"] else 1


def done_records(output: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Every accepted unit's record with where it was read: the Ray backend's ``accepted/`` (its authority), else the
    directory queue's ``done/``. Both hold ``unit``, ``accepted`` and ``outcomes``."""
    state = ray_state(output)
    if state is not None:
        return [(state.accepted_path(record["unit"]["id"]), record) for record in state.accepted_records()]
    return [(path, json.loads(path.read_text()))
            for path in sorted(p for p in (output / "queue" / "done").glob("*.json") if not p.name.startswith("."))]


def accepted_parts(output: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """``{task: {cell: [part, ...]}}`` from the done records: every directory an accepted unit is read from, with what
    the scorer reads of it (``score_benchmark_cells.py --accepted``): ``output``; ``rows`` ``{run: [episode, ...]}``
    counted as its episodes count (``replay_source_record_index``), or None for every episode it ran (an attempt that
    completed); ``salvaged``, whether a crash ended it; ``retried``, the rows of it an earlier attempt was charged for."""
    found: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for path, record in done_records(output):
        unit, outcomes = record["unit"], record.get("outcomes", [])
        planned = slice_rows(unit.get("slices") or {})
        if "parts" in record:
            parts, retried = record["parts"], mapping_rows(record.get("retried_rows"))
        else:
            if "accepted" in record or any(outcome.get("attempt") for outcome in outcomes):
                accepted = [outcome.get("output") for outcome in outcomes
                            if outcome.get("attempt") == record.get("accepted") and outcome.get("output")]
                if len(accepted) != 1:
                    raise SystemExit(f"{path}: no single output for its accepted attempt {record.get('accepted')!r}")
                directory = accepted[0]
            else:
                directory = str(output / "units" / unit["id"])
            parts = [{"output": directory, "slices": unit.get("slices") or {}, "rows": rows_mapping(planned),
                      "salvaged": False}]
            failed_before = any(outcome.get("attempt") != record.get("accepted") for outcome in outcomes)
            retried = set(planned) if failed_before else set()
        given = [row for part in parts for row in mapping_rows(part["rows"])]
        if sorted(given) != sorted(planned):
            raise SystemExit(f"{path}: its parts give {len(given)} rows ({len(set(given))} distinct), the unit plans "
                             f"{len(planned)}; each planned row is read from one attempt")
        for part in parts:
            rows, starts = mapping_rows(part["rows"]), {run: bound[0] for run, bound in part["slices"].items()}
            ran = set(slice_rows(part["slices"]))
            if not rows <= ran:
                raise SystemExit(f"{path}: {part['output']} is given rows it did not run ({sorted(rows - ran)[:5]})")
            local = {run: [index - starts[run] for index in indices] for run, indices in rows_mapping(rows).items()}
            found[unit["task"]][unit["cell"]].append({
                "output": part["output"], "rows": None if rows == ran and not part["salvaged"] else local,
                "salvaged": bool(part["salvaged"]),
                "retried": {run: [index - starts[run] for index in indices]
                            for run, indices in rows_mapping(retried & rows).items()}})
    return found


def accepted_outputs(output: Path) -> dict[str, dict[str, list[str]]]:
    """``{task: {cell: [directory, ...]}}``: the scorer's inputs, every directory an accepted unit is read from
    (:func:`accepted_parts`)."""
    return {task: {cell: [part["output"] for part in parts] for cell, parts in cells.items()}
            for task, cells in accepted_parts(output).items()}


def failure_cause(outcome: Mapping[str, Any]) -> str:
    """One line for why an attempt failed: its error (a deadline, a lost worker), else its exit code, the unit
    check's first problems, and the last line its runner printed."""
    parts = [outcome.get("error") or f"rc {outcome.get('rc')}"]
    parts += list(outcome.get("problems") or [])[:3]
    if outcome.get("metadata_error"):
        parts.append(outcome["metadata_error"])
    tail = [line for line in outcome.get("log_tail") or [] if line.strip()]
    if tail:
        parts.append(f"last line: {tail[-1].strip()}")
    return "; ".join(str(part) for part in parts)[:500]


def retried_units(output: Path) -> dict[str, list[dict[str, Any]]]:
    """``{task: [unit, ...]}`` for every done unit whose accepted attempt came after a failed one: the unit, its
    cell, the accepted attempt, and each failed attempt with its cause and the outcome as recorded. A unit can fail
    one attempt for a reason that would fail the episode (a simulator state the policy drove non-finite) and pass
    the next, so the score says how many of its units that holds for."""
    found: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for _, record in done_records(output):
        failed = [outcome for outcome in record.get("outcomes", []) if outcome.get("attempt") != record.get("accepted")]
        if failed:
            unit = record["unit"]
            found[unit["task"]].append({
                "unit": unit["id"], "cell": unit["cell"], "accepted": record.get("accepted"),
                "failed_attempts": [{"attempt": outcome.get("attempt"), "worker": outcome.get("worker"),
                                     "cause": failure_cause(outcome), "outcome": outcome} for outcome in failed]})
    return found


def code_revisions(cells: Mapping[str, list[str]]) -> dict[str, list[str]]:
    """``{cell: [revision, ...]}`` for each cell whose accepted units ran more than one code revision (each unit
    output's run manifest ``code.revision``). A resume may take a newer tree, and the scorer only lists the
    revisions a model's units ran, so --score warns. An unreadable manifest counts as its own
    revision; the scorer refuses that unit anyway."""
    mixed = {}
    for cell, directories in sorted(cells.items()):
        revisions = set()
        for directory in directories:
            try:
                revisions.add(str(json.loads((Path(directory) / RUN_MANIFEST).read_text())["code"]["revision"]))
            except (OSError, ValueError, KeyError, TypeError) as error:
                revisions.add(f"unreadable ({type(error).__name__})")
        if len(revisions) > 1:
            mixed[cell] = sorted(revisions)
    return mixed


def score(output: Path, partial: bool) -> int:
    """Score every task of the launch from its accepted units only; refuse while any planned unit is not done.
    Next to each ``score_<task>.json``, ``retries_<task>.json`` lists the accepted units that had a failed attempt
    before them, with each failure's cause (:func:`retried_units`); the count is printed with the score.
    ``score_<task>.accepted.json`` is what the scorer reads of each directory (:func:`accepted_parts`), so each row is
    scored from the one attempt it was accepted from, and the score counts per cell the episodes from attempts a crash
    ended and the rows that needed more than one attempt."""
    plan = json.loads((output / "plan.json").read_text())
    planned = {unit["id"] for unit in plan["units"]}
    state = ray_state(output)
    missing = sorted(planned - {record["unit"]["id"] for _, record in done_records(output)})
    if missing:
        counts = state.status() if state is not None else queue_dir.status(output / "queue")
        message = f"{len(missing)} of {len(planned)} planned units are not done ({counts})"
        if state is not None:
            failed_rows = {record["unit"]["id"]: record["failed_rows"] for record in state.failed_records()}
            if failed_rows:
                message += f"; failed rows {json.dumps(failed_rows)}"
        if not partial:
            raise SystemExit(message + "; --partial scores what is done, and the scorer refuses an incomplete cell")
        print(message, flush=True)
    rc = 0
    parts = accepted_parts(output)
    outputs = {task: {cell: [part["output"] for part in cell_parts] for cell, cell_parts in cells.items()}
               for task, cells in parts.items()}
    retried = retried_units(output)
    for task, cells in sorted(outputs.items()):
        results = [f"{cell}={directory}" for cell, directories in sorted(cells.items()) for directory in directories]
        report = output / f"score_{task}.json"
        accepted = output / f"score_{task}.accepted.json"
        _publish(accepted, {part["output"]: {key: part[key] for key in ("rows", "salvaged", "retried")}
                            for cell_parts in parts[task].values() for part in cell_parts})
        command = [sys.executable, str(SCORER), "score", "--task", task, "--version", plan_eval_version(plan),
                   "--results", *results, "--accepted", str(accepted), "--json", str(report)]
        salvaged = sum(len(indices) for cell_parts in parts[task].values() for part in cell_parts if part["salvaged"]
                       for indices in (part["rows"] or {}).values())
        rerun = sum(len(indices) for cell_parts in parts[task].values() for part in cell_parts
                    for indices in part["retried"].values())
        print(f"scoring {task}: {len(results)} unit outputs, {salvaged} episodes from attempts a crash ended, {rerun} "
              f"rows run more than once -> {report}", flush=True)
        for cell, revisions in code_revisions(cells).items():
            print(f"warning: {task} {cell}: its accepted units ran {len(revisions)} code revisions "
                  f"({', '.join(revision[:12] for revision in revisions)}), e.g. across a resume onto a newer tree",
                  flush=True)
        rc |= subprocess.run(command, cwd=REPO_ROOT, check=False).returncode
        retries = output / f"retries_{task}.json"
        _publish(retries, {"task": task, "accepted_units": len(results), "retried_units": retried.get(task, [])})
        print(f"{task}: {len(retried.get(task, []))} of {len(results)} accepted units ran after a failed attempt "
              f"-> {retries}", flush=True)
        for entry in retried.get(task, []):
            for attempt in entry["failed_attempts"]:
                print(f"  {entry['unit']} {attempt['attempt']}: {attempt['cause']}", flush=True)
    unscored = sorted(set(plan["tasks"]) - set(outputs))
    if unscored:
        print(f"no accepted units yet for {', '.join(unscored)}", flush=True)
    return rc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true", help="Plan and print the estimate; write nothing.")
    mode.add_argument("--start", action="store_true", help="Write the plan and queue its units in the output directory.")
    mode.add_argument("--status", action="store_true", help="The queue's counts and every failed unit's reason.")
    mode.add_argument("--local", action="store_true",
                      help="Start (or resume) the launch: its driver (backend.kind local or ray), or one worker per GPU.")
    mode.add_argument("--resume", action="store_true",
                      help="Pick up a stopped launch; with the local backend, also start its workers again.")
    mode.add_argument("--score", action="store_true", help="Score every task from the accepted units only.")
    mode.add_argument("--smoke", action="store_true",
                      help="Before a launch (backend.kind local or ray): one station runs --smoke-rows rows on each of "
                           "its simulator slots at once, in <output>-smoke/<time>/, and the check says pass or fail on "
                           "frames, timing and GPU memory (eval_launch/smoke.py).")
    mode.add_argument("--k8s-job", type=Path, metavar="DIR",
                      help="Write the k8s backend's Job (job.json) and the config the pods read (eval.yaml) into DIR; "
                           "copy eval.yaml to the launch's output directory, then kubectl apply -f job.json. For a "
                           "config with backend.kind: ray, ray.json instead: a Ray head pod that runs the driver, "
                           "worker pods and the head's Service.")
    parser.add_argument("--write-plan", type=Path, help="With --plan-only: also write the plan as JSON here.")
    parser.add_argument("--existing-plan", action="store_true",
                        help="With --k8s-job: the config's output on the volume already has plan.json (a resume), so a "
                             "retired key is dropped from eval.yaml rather than refused.")
    parser.add_argument("--server-env", type=Path,
                        help="With --local/--resume: the environment the policy servers start in (env -0, saved before "
                             "/isaac-sim/python.sh; launch.sh does this). Read and removed at once, whatever the mode: "
                             "it holds the shell's tokens.")
    parser.add_argument("--retry-failed", action="store_true", help="With --resume: failed units get another round.")
    parser.add_argument("--requeue-claims", action="store_true",
                        help="With --resume: no worker is running, so every claimed unit goes back to the queue now.")
    parser.add_argument("--partial", action="store_true", help="With --score: score what is done so far.")
    parser.add_argument("--smoke-rows", type=int, default=2, help="With --smoke: rows per simulator slot (default 2).")
    parser.add_argument("--tasks", help="Comma-separated tasks: plan only these (or, with --start on a launch, add them; "
                                        "with --smoke, the tasks it picks its one cell from).")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--timings", type=Path,
                        help="Measured per-task slowdown and build time, over the config's plan.timings.")
    early = argparse.ArgumentParser(add_help=False)
    early.add_argument("--server-env", type=Path)
    saved = early.parse_known_args(argv)[0].server_env
    server_environment = server_env.read(saved) if saved else None
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except (ValueError, OSError, yaml.YAMLError) as error:
        parser.error(f"{args.config}: {error}")
    if args.start or args.local or args.resume or args.k8s_job or args.smoke:
        try:
            check_recipe(config)
        except ValueError as error:
            parser.error(f"{args.config}: {error}")
    output = Path(config.get("output") or "")
    if (args.start or args.status or args.local or args.resume or args.score or args.smoke) and not config.get("output"):
        parser.error("the config's output names the launch's directory")
    retired = sorted(set(config) & set(RETIRED_KEYS))
    if args.existing_plan and not args.k8s_job:
        parser.error("--existing-plan goes with --k8s-job: the other modes see the output directory themselves")
    if retired and not (args.status or args.score):
        planned = args.existing_plan if args.k8s_job else (
            (args.start or args.local or args.resume) and (output / "plan.json").is_file())
        if not planned:
            parser.error(f"{args.config}: remove {', '.join(retired)}: "
                         + "; ".join(f"{key}: {RETIRED_KEYS[key]}" for key in retired))
        print(f"ignoring {', '.join(retired)} of {args.config}: {output / 'plan.json'} was planned with it"
              + (" (--existing-plan)" if args.k8s_job else "") + ", and "
              + "; ".join(RETIRED_KEYS[key] for key in retired), flush=True)
        config = {key: value for key, value in config.items() if key not in RETIRED_KEYS}
    timings = None
    if args.plan_only or args.start or args.local or args.k8s_job or args.smoke:
        try:
            timings = load_timings(config, args.timings)
        except (OSError, ValueError) as error:
            parser.error(f"{args.config}: plan.timings: {error}")
    if args.smoke:
        if args.smoke_rows < 1:
            parser.error("--smoke-rows is at least 1")
        checked = {**config, "tasks": task_list(args.tasks)} if args.tasks else config
        try:
            return smoke.run(checked, args.smoke_rows, args.manifest, timings, server_environment, TREE)
        except (ValueError, SystemExit) as error:
            if isinstance(error, SystemExit) and not isinstance(error.code, str):
                raise
            print(f"{args.config}: smoke check not run: {error.code if isinstance(error, SystemExit) else error}",
                  file=sys.stderr, flush=True)
            return smoke.NOT_RUN
    tasks = task_list(args.tasks)
    kind = (config.get("backend") or {}).get("kind")
    driven = kind in DRIVER_MODULES and not (kind == "local" and queue_launch(output))
    if ((args.start or args.local or args.resume) and not driven) or (args.k8s_job and kind != "ray"):
        if ray_state(output) is not None:
            parser.error(f"{output} is a driver's launch (backend.kind: local or ray): a launch keeps the backend it "
                         "started with")
        try:
            layout = parse_layout(config.get("servers")).name
        except ValueError as error:
            parser.error(f"{args.config}: {error}")
        if layout != COLOCATED:
            parser.error(f"servers.layout {config['servers']['layout']!r} needs backend.kind: local or ray; this "
                         "backend runs one server and one simulator per GPU")
    if driven and (args.local or args.resume):
        command = [sys.executable, "-m", DRIVER_MODULES[kind], str(args.config), *(["--resume"] if args.resume else []),
                   *(["--retry-failed"] if args.retry_failed else []), *(["--tasks", ",".join(tasks)] if tasks else []),
                   "--manifest", str(args.manifest), *(["--timings", str(args.timings)] if args.timings else []),
                   *(["--server-env", server_env.handover(server_environment)] if server_environment is not None else [])]
        environment = {**os.environ, **({STATUS_VARIABLE: status_file()} if status_file() else {})}
        os.execve(sys.executable, command, environment)
    if args.k8s_job:
        manifest = "ray.json" if kind == "ray" else "job.json"
        unfilled = placeholders(config)
        if unfilled:
            site = ((config.get("backend") or {}).get("site") or {}).get("profile")
            parser.error(f"{args.config}: {', '.join(unfilled[:4])}{' ...' if len(unfilled) > 4 else ''} still hold an "
                         f"example's placeholder" + (f" (from the site profile {site}: copy it beside your config, fill it "
                                                     "in, and point backend.site at the copy)" if site else ""))
        try:
            if kind == "ray":
                launch_id = f"{config['backend']['k8s']['job_name']}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
                job = render_ray(config, config_on_volume(config), launch_id)
            else:
                job = render_job(config, config_on_volume(config))
            prepull = render_prepull(config) if config["backend"]["k8s"].get("prepull", True) else None
        except (ValueError, KeyError) as error:
            parser.error(f"{args.config}: {error}")
        try:
            states = make_plan(config, args.manifest, timings)["tasks"]
        except ValueError as error:
            parser.error(f"{args.config}: {error}")
        tree, missing = _missing_note(config, args.manifest, states)
        if missing:
            parser.error(f"{tree} lacks {len(missing)} files the planned tasks read ({', '.join(sorted(missing)[:3])}"
                         f"{', ...' if len(missing) > 3 else ''}): run {', '.join(sorted(set(missing.values())))} in it")
        mismatch = staged_commit_mismatch(config, tree)
        if mismatch:
            parser.error(f"{args.config}: {mismatch}")
        if tree is None:
            print(f"note: backend.k8s.tree_dir {config['backend']['k8s']['tree_dir']} is not visible here; pod 0 checks "
                  "its inputs when it plans")
        args.k8s_job.mkdir(parents=True, exist_ok=True)
        (args.k8s_job / "eval.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        (args.k8s_job / manifest).write_text(json.dumps(job, indent=1) + "\n")
        rendered = {manifest: job}
        if prepull is not None:
            (args.k8s_job / "prepull.json").write_text(json.dumps(prepull, indent=1) + "\n")
            rendered["prepull.json"] = prepull
        record = record_launch(args.k8s_job, config, "k8s-job", rendered)
        what = (f"a Ray head and {job['items'][2]['spec']['completions']} worker pods" if kind == "ray"
                else f"{config['backend']['gpus']} one-GPU workers")
        print(f"wrote {args.k8s_job / manifest} ({what}), "
              f"{str(args.k8s_job / 'prepull.json') + ' and ' if prepull is not None else '(no pre-pull: backend.k8s.prepull false) and '}"
              f"{args.k8s_job / 'eval.yaml'}; copy eval.yaml to {config_on_volume(config)}, and {record} to "
              f"{output / LAUNCHES}/ (the pods record their own start there too)")
    elif args.status:
        _status(output)
    elif args.score:
        return score(output, args.partial)
    elif args.local or args.resume:
        max_attempts = int((config.get("plan") or {}).get("max_attempts", 2))
        if args.resume:
            _resume(config, output, args.retry_failed, args.requeue_claims, max_attempts)
        else:
            _start(config, output, tasks, args.manifest, timings)
        record_launch(output, config, "resume" if args.resume else "local")
        if (config.get("backend") or {}).get("kind") == "local":
            return run_local(config, output, server_environment)
        if args.local:
            raise SystemExit("--local needs backend.kind: local")
        print("the queue is ready: start the backend's workers again (k8s: delete the old Job, apply job.json)")
    elif args.start:
        if driven:
            write_plan(config, output, tasks, args.manifest, timings)
        else:
            _start(config, output, tasks, args.manifest, timings)
        record_launch(output, config, "start")
    else:
        plan = make_plan(config if not tasks else {**config, "tasks": tasks}, args.manifest, timings)
        print(summary(plan))
        tree, missing = _missing_note(config, args.manifest, plan["tasks"])
        mismatch = staged_commit_mismatch(config, tree)
        if mismatch:
            print(f"note: {mismatch}")
        if missing:
            print(f"note: {tree} lacks {len(missing)} files the planned tasks read; a launch on it would be refused "
                  f"({', '.join(sorted(set(missing.values())))})")
        elif tree is None:
            print(f"note: backend.k8s.tree_dir {config['backend']['k8s']['tree_dir']} is not visible here; its inputs "
                  "are checked when the launch plans")
        if args.write_plan:
            args.write_plan.write_text(json.dumps(plan, indent=1) + "\n")
            print(f"wrote {args.write_plan}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
