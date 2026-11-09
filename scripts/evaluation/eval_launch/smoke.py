"""A smoke check of a launch config before the launch: one station of its layout runs a few rows on every simulator
slot at once, through the config's own driver (``local``, or ``ray``: a Ray instance on this host, or the cluster
``backend.ray.address`` names), and says pass or fail on what decides whether the full launch can run::

    launch.sh <config.yaml> --smoke [--smoke-rows 2]

- **frames** (the hard gate): every unit accepted (the run exited 0, every episode it was given ran, each with a video
  from every camera, and every camera rendered: ``unit_check``);
- **timing** (advisory up to the deadline): each unit's measured slowdown (wall time stepping over simulated time)
  against what the plan assumes for its task with ``servers.layout``'s simulators sharing a GPU. Within
  :data:`TIMING_TOLERANCE` of it passes; beyond, up to the plan's ``deadline_factor`` (what a launch's units may take
  before they are stopped), is a warning to measure this machine's costs; past that, the launch's units would hit their
  deadlines, and the check fails. A smoke unit's slowdown is a mean over a few rows, which its first steps inflate,
  and without ``plan.timings`` "planned" is the default (GR00T on an L40S), so it is not a gate below the deadline;
- **memory**: the peak each simulator's GPU reached (``gpu_memory``, the whole GPU, its server and neighbours
  included) below :data:`MEMORY_LIMIT` of the card, and, when the config states ``servers.gpu_memory_gb`` for every
  process on that GPU, within :data:`DECLARED_TOLERANCE` of what it states (the layout check trusts those figures). A
  GPU nothing could read (no ``nvidia-smi``) is "not measured", its own state, not a failure. A unit without step
  timings fails the timing check: the gate could not run.

Each item has a ``status``: ``pass``, ``fail``, ``advisory`` or ``not measured``; the check passes when nothing failed.

It writes nothing under the config's ``output``: the check is a launch of its own in ``<output>-smoke/<UTC time>/``
(its plan, its record, its units, ``smoke.json`` with the verdict), never scored. The units are ``--smoke-rows`` rows
per simulator slot of one planned cell, at the config's version (:func:`smoke_units`).
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import yaml

from eval_launch import server_env
from eval_launch.acceptance import LaunchState
from eval_launch.layout import gpu_count, parse_layout
from eval_launch.plan import (check_inputs, costs_from, deadline_factor, eval_version, estimate, make_plan,
                              version_counts)

TIMING_TOLERANCE = 1.5
"""A smoke unit runs a few rows, so its slowdown is noisy; more than 1.5 times the plan's (per simulator sharing the
GPU) means the plan's costs are not this machine's: measure them (USER_GUIDE.md, "Plan costs")."""
MEMORY_LIMIT = 0.95
"""The share of a GPU's memory a station may peak at: a card nearly full starves the renderer (black frames), as a
simulator beside a Cosmos3 server on one L40S did (2026-10-23)."""
DECLARED_TOLERANCE = 1.1
"""How far the measured peak may exceed what ``servers.gpu_memory_gb`` states for a GPU before those figures are
called wrong."""
DRIVERS = {"local": "eval_launch.local_backend", "ray": "eval_launch.ray_backend"}
MEMORY_LIMITS = ("a floor, not the launch's peak: each smoke unit builds the environment of one Run of one cell ({cell}), "
                 "while a launch's units build up to 60 Runs in turn in one Kit process, an earlier Run's scene still "
                 "alive during the next, and other tasks' scenes differ; a pass says nothing about them")
"""What the memory reading does not show, in the report and ``smoke.json``."""
NOT_RUN = 2
"""The exit code when the check could not run (a refusal, or a driver under which no unit ran); 1 is a check that
ran and failed, 0 one that passed, 130 one a signal stopped."""


def smoke_units(config: Mapping[str, Any], plan: Mapping[str, Any], manifest: Mapping[str, Any], rows: int,
                slots: int, build_s: Callable[[str], float]) -> list[dict[str, Any]]:
    """``slots`` units of ``rows`` rows each, a unit to a slot, from the planned cell whose longest Run has the most rows
    at the config's version (each Run a unit touches is an environment build), its Runs longest first; a unit takes
    rows of several Runs only where one Run has fewer. Each unit's cost (``expected_s``, ``overhead_s``, from which the
    drivers set its deadline) is its own, from the plan's costs for its rows and builds (``build_s`` per task), not the
    planned unit's it was cut from: a hung smoke simulator is stopped as soon as a unit of its size would be."""
    version = eval_version(config)
    cells = {(unit["task"], unit["cell"]): unit for unit in plan["units"] if unit["cell"]}
    if not cells:
        raise ValueError(f"no cell of {', '.join(plan['tasks']) or 'no task'} can be planned for the smoke: check the "
                         "config's tasks and cells, and --tasks")
    (task, cell), first = max(cells.items(), key=lambda item: max(version_counts(manifest, *item[0], version).values()))
    counts = version_counts(manifest, task, cell, version)
    units: list[dict[str, list[int]]] = [{}]
    for run in sorted(counts, key=lambda name: -counts[name]):
        start = 0
        while start < counts[run] and len(units) <= slots:
            taken = sum(stop - begin for begin, stop in units[-1].values())
            stop = min(counts[run], start + rows - taken)
            units[-1][run] = [start, stop]
            start = stop
            if taken + (stop - units[-1][run][0]) == rows:
                units.append({})
    units = [slices for slices in units if sum(stop - begin for begin, stop in slices.values()) == rows][:slots]
    if len(units) < slots:
        raise ValueError(f"{task} {cell} has too few rows for {slots} units of {rows}")
    kit_start_s, episode_s = float(plan["costs"]["kit_start_s"]), float(plan["tasks"][task]["episode_s"])
    return [{**first, "id": f"{task}.{cell}.smoke{index:02d}", "slices": slices, "runs": len(slices), "episodes": rows,
             "expected_s": kit_start_s + len(slices) * build_s(task) + rows * episode_s,
             "overhead_s": kit_start_s + len(slices) * build_s(task)}
            for index, slices in enumerate(units)]


def control_hz(manifest: Mapping[str, Any], task: str) -> float:
    """The rate each Run of ``task`` steps at: the protocol's (``policy_execution.environment.rby1_control_hz`` in the
    manifest), which ``model_run.compose`` writes into every Run. Not read from the composed file: it also keeps the
    shared block's own rate (30 beside the Runs' 60)."""
    return float(manifest["tasks"][task]["policy_execution"]["environment"]["rby1_control_hz"])


def _slowdown(outcome: Mapping[str, Any], hz: float) -> float | None:
    """Wall time stepping over simulated time: Arena's step timings over the steps at the task's control rate ``hz``."""
    if not outcome.get("steps") or not outcome.get("stepping_s"):
        return None
    return float(outcome["stepping_s"]) / (float(outcome["steps"]) / hz)


def verdict(config: Mapping[str, Any], plan: Mapping[str, Any], state: LaunchState,
            rates: Mapping[str, float]) -> dict[str, Any]:
    """Each check's items with their status and what they measured; ``ok`` when no item failed. ``rates``: each task's
    control rate (:func:`control_hz`)."""
    layout = parse_layout(config.get("servers"))
    accepted = {record["unit"]["id"]: record for record in state.accepted_records()}
    failed = {record["unit"]["id"]: record for record in state.failed_records()}
    limit = deadline_factor(config.get("plan"))
    frames, timing, memory = [], [], []
    for unit in plan["units"]:
        record = accepted.get(unit["id"]) or failed.get(unit["id"])
        outcomes = (record or {}).get("outcomes") or []
        last = outcomes[-1] if outcomes else {}
        if unit["id"] in accepted:
            frames.append({"unit": unit["id"], "status": "pass"})
        else:
            frames.append({"unit": unit["id"], "status": "fail",
                           "cause": (last.get("error") or f"rc {last.get('rc')}") if record else "never ran",
                           "problems": list(last.get("problems") or [])[:5]})
        task = plan["tasks"][unit["task"]]
        planned = float(task["episode_s"]) / float(task["cap_s"]) * layout.sims_per_gpu
        measured = _slowdown(last, rates[unit["task"]])
        entry = {"unit": unit["id"], "measured_slowdown": measured, "planned_slowdown": planned, "deadline_factor": limit}
        if measured is None:
            entry.update(status="fail", cause="no step timings in the unit's output: the timing could not be checked")
        elif measured > planned * limit:
            entry.update(status="fail", cause=f"{measured / planned:.1f}x the plan's: past the deadline factor "
                                              f"{limit:g}, the launch's units would be stopped")
        elif measured <= planned * TIMING_TOLERANCE:
            entry["status"] = "pass"
        else:
            entry.update(status="advisory", cause=f"{measured / planned:.1f}x the plan's: within the deadline "
                                                  f"({limit:g}x), but the plan's costs are not this machine's; measure "
                                                  "them (USER_GUIDE.md, \"Plan costs\")")
        timing.append(entry)
        memory.append(_memory_item(config, layout, unit["id"], last.get("gpu_memory")))
    checks = {"frames": frames, "timing": timing, "memory": memory}
    for items in checks.values():
        for item in items:
            item["ok"] = item["status"] != "fail"
    cells = sorted({f"{unit['task']} {unit['cell']}" for unit in plan["units"]})
    return {"ok": all(item["ok"] for items in checks.values() for item in items), "layout": layout.name, **checks,
            "memory_limits": MEMORY_LIMITS.format(cell=", ".join(cells))}


def _memory_item(config: Mapping[str, Any], layout: Any, unit_id: str, peak: Mapping[str, Any] | None) -> dict[str, Any]:
    entry: dict[str, Any] = {"unit": unit_id, "gpu_memory": peak}
    if peak is None:
        entry.update(status="not measured", cause="no reading (no nvidia-smi, or no one GPU, where the simulator ran)")
        return entry
    entry["share"] = peak["peak_used_mib"] / peak["total_mib"]
    if entry["share"] > MEMORY_LIMIT:
        entry.update(status="fail", cause=f"peaked at {entry['share']:.0%} of the GPU's memory")
        return entry
    entry["status"] = "pass"
    declared = (config.get("servers") or {}).get("gpu_memory_gb") or {}
    servers_here, sims_here = layout.processes_per_gpu()[-1 if layout.sim_gpus else 0]
    stated = [(servers_here, declared.get("server")), (sims_here, declared.get("sim"))]
    if declared and all(figure is not None for count, figure in stated if count):
        stated_gb = sum(count * float(figure) for count, figure in stated if count)
        entry["declared_gb"] = stated_gb
        if peak["peak_used_mib"] / 1024 > stated_gb * DECLARED_TOLERANCE:
            entry.update(status="fail", cause=f"peaked at {peak['peak_used_mib'] / 1024:.1f} GB, servers.gpu_memory_gb "
                                              f"states {stated_gb:g} GB for this GPU")
    elif declared:
        entry["cause"] = "servers.gpu_memory_gb does not state every process on this GPU: not compared with it"
    return entry


def report(result: Mapping[str, Any]) -> str:
    verdict_word = {"pass": "PASS", "fail": "FAIL", "not run": "NOT RUN (the check could not run)"}[result["result"]]
    lines = [f"smoke check, layout {result['layout']}: {verdict_word}"]
    labels = {"pass": "ok  ", "fail": "FAIL", "advisory": "warn", "not measured": "n/a "}
    for check in ("frames", "timing", "memory"):
        for item in result[check]:
            detail = item.get("cause") or ""
            if check == "timing":
                measured = item["measured_slowdown"]
                detail = (f"slowdown {measured:.2f}" if measured is not None else "slowdown not measured") + \
                    f" (the plan's {item['planned_slowdown']:.2f} on this layout)" + (f"; {item['cause']}" if
                                                                                       item.get("cause") else "")
            elif check == "memory" and item.get("gpu_memory"):
                peak = item["gpu_memory"]
                detail = (f"GPU {peak['gpu']} peaked at {peak['peak_used_mib'] / 1024:.1f} of "
                          f"{peak['total_mib'] / 1024:.1f} GB" + (f"; {item['cause']}" if item.get("cause") else ""))
            lines.append(f"  {check:<7}{labels[item['status']]} {item['unit']}: {detail}".rstrip())
    if result.get("memory_limits"):
        lines.append(f"  memory is {result['memory_limits']}")
    return "\n".join(lines)


def _run_driver(command: list[str], server_environment: Mapping[str, str] | None) -> tuple[int, str | None]:
    """Run the smoke's driver to its end and return its exit code and the signal that stopped the smoke, if any. The driver
    is in this process's group, so a Ctrl-C or a TERM to the launch's process group (what launch.sh and signal_group.sh
    send) reaches it too: this process does not stop for it (``subprocess.run`` would SIGKILL the driver 0.25 s into its
    teardown) but waits for the driver to stop its stations. The servers' environment goes as a descriptor whose file is
    already gone (``server_env.handover``)."""
    descriptors: tuple[int, ...] = ()
    if server_environment is not None:
        handed = server_env.handover(server_environment)
        descriptors = (int(handed.rsplit("/", 1)[1]),)
        command = [*command, "--server-env", handed]
    stopped: list[str] = []
    previous = {number: signal.signal(number, lambda signum, frame: stopped.append(signal.Signals(signum).name))
                for number in (signal.SIGINT, signal.SIGTERM)}
    try:
        process = subprocess.Popen(command, pass_fds=descriptors)
        for descriptor in descriptors:
            os.close(descriptor)
        descriptors = ()
        return process.wait(), (stopped[0] if stopped else None)
    finally:
        for descriptor in descriptors:
            os.close(descriptor)
        for number, handler in previous.items():
            signal.signal(number, handler)


def run(config: dict[str, Any], rows: int, manifest_path: Path, timings: Mapping[str, Any] | None,
        server_environment: Mapping[str, str] | None, tree: Path) -> int:
    """Plan the smoke launch beside the config's output, run it with the config's driver on one station, and print
    and write (``smoke.json``) the verdict; 0 when every check passed. Refused, like any launch, on a ``tree`` that
    lacks a file its units read (``plan.check_inputs``)."""
    kind = (config.get("backend") or {}).get("kind")
    if kind not in DRIVERS:
        raise SystemExit(f"--smoke runs the local or Ray driver, not backend.kind {kind!r}: give the config "
                         "backend.kind: ray (a Ray instance on this host, or backend.ray.address) to check a layout")
    layout = parse_layout(config.get("servers"))
    launch_output = Path(str(config["output"]))
    output = launch_output.with_name(f"{launch_output.name}-smoke") / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    gpus = config["backend"].get("gpus", layout.gpus)
    station_gpus = gpus[:layout.gpus] if isinstance(gpus, list) else layout.gpus
    if gpu_count(station_gpus) < layout.gpus:
        raise SystemExit(f"backend.gpus gives fewer than the {layout.gpus} GPUs one station of {layout.name} takes")
    smoke = {**config, "output": str(output), "backend": {**config["backend"], "gpus": station_gpus},
             "plan": {**(config.get("plan") or {}), "max_attempts": 1}}
    full = make_plan(config, manifest_path, timings)
    manifest = json.loads(Path(manifest_path).read_text())
    units = smoke_units(config, full, manifest, rows, layout.sims, costs_from(config, timings).build)
    task = units[0]["task"]
    smoke = {**smoke, "tasks": [task], "cells": {task: [units[0]["cell"]]}}
    plan = {**full, "config": smoke, "tasks": {task: full["tasks"][task]}, "units": units,
            "estimate": estimate(units, gpu_count(station_gpus)), "smoke": True}
    check_inputs(manifest, {task: {**full["tasks"][task], "cells": [units[0]["cell"]]}}, tree)
    output.mkdir(parents=True)
    (output / "plan.json").write_text(json.dumps(plan, indent=1) + "\n")
    config_path = output / "eval.yaml"
    config_path.write_text(yaml.safe_dump(smoke, sort_keys=False))
    print(f"smoke check: {len(units)} units of {rows} rows ({task} {units[0]['cell']}) on one {layout.name} station "
          f"-> {output}", flush=True)
    command = [sys.executable, "-m", DRIVERS[kind], str(config_path), "--resume", "--manifest", str(manifest_path)]
    driver_rc, interrupted = _run_driver(command, server_environment)
    if interrupted:
        (output / "smoke.json").write_text(json.dumps({"result": "stopped", "signal": interrupted,
                                                       "driver_rc": driver_rc, "output": str(output)}, indent=1) + "\n")
        print(f"smoke check stopped ({interrupted}) after its driver's teardown; nothing judged: {output}", flush=True)
        return 130
    result = {**verdict(smoke, plan, LaunchState(output, units, 1), {task: control_hz(manifest, task)}),
              "driver_rc": driver_rc, "output": str(output)}
    not_run = all(item.get("cause") == "never ran" for item in result["frames"])
    result["result"] = "not run" if not_run else ("pass" if result["ok"] else "fail")
    (output / "smoke.json").write_text(json.dumps(result, indent=1) + "\n")
    print(report(result), flush=True)
    if not_run:
        print(f"no smoke unit ran (the driver exited {driver_rc}): the check could not run; see {output}", flush=True)
        return NOT_RUN
    return 0 if result["ok"] else 1
