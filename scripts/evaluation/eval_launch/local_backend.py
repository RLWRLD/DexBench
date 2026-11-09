#!/usr/bin/env python3
# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""The local backend: the launch's driver (:mod:`eval_launch.driver`) on this machine's GPUs, without Ray.

    python -m eval_launch.local_backend <config.yaml> [--resume] [--retry-failed] [--tasks <task>,...]

(``launch.sh <config> --local`` or ``--resume`` runs it for a config with ``backend.kind: local``.) Any layout that
fits the machine's GPUs runs (:mod:`eval_launch.layout`); ``backend.kind: ray`` runs the same driver on several
machines or a Kubernetes cluster.

The GPUs are ``backend.gpus``: a list of device ids, a count (the first that many of those ``CUDA_VISIBLE_DEVICES``
or ``nvidia-smi`` lists, else ``0..count-1``), or, left out, every GPU this process sees. Station ``i`` takes the
``i``-th run of ``layout.gpus`` of them, bundle ``b`` of its layout the ``b``-th of its run; a fractional share of a
GPU is one of several processes given the same device.

Each station's server is a child process of the driver holding ``eval_worker.PolicyServer`` on its GPU, which it
starts or keeps for each checkpoint the driver asks for (JSON lines on its stdin and stdout). Each unit runs in a child
process of its own, one per simulator slot at a time, that calls ``eval_worker.run_unit`` unchanged with its slot's
GPU as its only ``CUDA_VISIBLE_DEVICES``, writes its outcome where a later driver can recover it, and replies on its
stdout. A slot that does not answer past its deadline is SIGKILLed; what it started (its Kit, in a session of its
own) carries the launch's owner tag and is swept (``process_group.stop_orphans``) before the slot takes another unit.
The children die with the driver (``PR_SET_PDEATHSIG``: SIGTERM, on which a slot stops its Kit and a server its
recipe's process), and what a killed one left is swept when the next driver starts.

Standard library only: this module never imports Ray.
"""

from __future__ import annotations

import ctypes
import importlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from concurrent.futures import FIRST_COMPLETED, Future
from concurrent.futures import wait as wait_futures
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from eval_launch import PACKAGE_DIR, driver, eval_worker, run_main, server_env
from eval_launch.driver import Lost, Raised, Station, owner_scope, run_slot
from eval_launch.layout import PORT_BASE, Layout, check_config, parse_layout
from eval_launch.plan import model_spec
from eval_launch.process_group import claim_ownership, stop_orphans

LOCAL_KEYS = {"port_base", "slot_restarts", "station_restarts", "setup_hook"}
"""``backend.local``: ``port_base`` (:data:`PORT_BASE`; station i's server listens on ``port_base + 10 i``),
``slot_restarts`` (3: losses in a row before a slot's station is rebuilt), ``station_restarts`` (2), and
``setup_hook``, a ``module.function`` every server and slot process calls first (as Ray's
``worker_process_setup_hook``; the tests put their stand-ins in place with it)."""
STOP_GRACE_S = 120.0
"""How long a stopped station's processes get to stop their Kit and server (each takes up to a minute) before SIGKILL."""
_PR_SET_PDEATHSIG = 1


def visible_gpus() -> list[str]:
    """The GPUs this process may use: ``CUDA_VISIBLE_DEVICES`` when it is set, else every GPU ``nvidia-smi`` lists,
    else none."""
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return [gpu.strip() for gpu in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if gpu.strip()]
    if shutil.which("nvidia-smi") is None:
        return []
    listed = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], capture_output=True, text=True,
                            timeout=60, check=False)
    return [line.strip() for line in listed.stdout.splitlines() if line.strip()] if listed.returncode == 0 else []


def local_gpus(backend: Mapping[str, Any]) -> list[str]:
    """The device ids of ``backend.gpus`` (see the module's docstring)."""
    gpus = backend.get("gpus")
    if isinstance(gpus, list):
        return [str(gpu) for gpu in gpus]
    visible = visible_gpus()
    if gpus is None:
        if not visible:
            raise ValueError("no GPU found (CUDA_VISIBLE_DEVICES, nvidia-smi): state backend.gpus")
        return visible
    if type(gpus) is not int or gpus < 1:
        raise ValueError(f"backend.gpus is a list of device ids or a count, not {gpus!r}")
    if not visible:
        return [str(index) for index in range(gpus)]
    if gpus > len(visible):
        raise ValueError(f"backend.gpus is {gpus}, and this machine shows {len(visible)} GPUs ({','.join(visible)})")
    return visible[:gpus]


def check_local_config(config: Mapping[str, Any]) -> Layout:
    """The config's layout, refused when it cannot run on this machine's GPUs; before anything is written."""
    backend = config.get("backend") or {}
    return check_config(config, "local", LOCAL_KEYS, len(local_gpus(backend)))


def _in_thread(function, *args) -> Future:
    """``function(*args)`` on a thread of its own; a future to what it returns or raises."""
    future: Future = Future()

    def target() -> None:
        try:
            future.set_result(function(*args))
        except BaseException as error:
            future.set_exception(error)

    threading.Thread(target=target, daemon=True).start()
    return future


def _done(value: Any) -> Future:
    future: Future = Future()
    future.set_result(value)
    return future


@dataclass
class LocalStation:
    """What the local executor holds for one station."""

    gpus: list[str]
    server: subprocess.Popen | None = None
    slots: dict[int, subprocess.Popen] = field(default_factory=dict)


class LocalExecutor:
    """A station is a run of this machine's GPUs, its server and each running unit a child process (:class:`driver.Executor`)."""

    kind = "local"

    def __init__(self, config: Mapping[str, Any], output: Path, server_environment: Mapping[str, str] | None = None):
        self.output = Path(output)
        self.server_environment = dict(server_environment) if server_environment is not None else None
        backend = config.get("backend") or {}
        local = dict(backend.get("local") or {})
        self.layout: Layout = parse_layout(config.get("servers"))
        self.gpu_ids = local_gpus(backend)
        self.gpus = len(self.gpu_ids)
        self.port_base = int(local.get("port_base", PORT_BASE))
        self.slot_restarts = int(local.get("slot_restarts", 3))
        self.station_restarts = int(local.get("station_restarts", 2))
        self.setup_hook = local.get("setup_hook")
        plan_config = config.get("plan") or {}
        self.ready_timeout_s = float(plan_config.get("server_ready_timeout_s", eval_worker.SERVER_READY_TIMEOUT_S))
        self.recipe = (config.get("servers") or {})["recipe"]
        self.model = model_spec(config)
        self.environment = {str(k): str(v) for k, v in (backend.get("env") or {}).items()}
        self.pythonpath = os.pathsep.join(filter(None, (str(PACKAGE_DIR.parent), os.environ.get("PYTHONPATH"),
                                                        *(str(path) for path in backend.get("pythonpath", [])))))
        self.scope = owner_scope(self.output)
        self.environment_files: list[Path] = []


    def start(self, epoch: int) -> list[int]:
        return stop_orphans(self.scope)

    def stop(self) -> None:
        for path in self.environment_files:
            path.unlink(missing_ok=True)
        self.environment_files.clear()

    def _child(self, role: str, gpu: str, log: Path) -> subprocess.Popen:
        """A server or slot process on ``gpu``, started from this (the driver's main) thread: ``PR_SET_PDEATHSIG``
        fires when the thread that started the child ends."""
        log.parent.mkdir(parents=True, exist_ok=True)
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu, "PYTHONPATH": self.pythonpath}
        with log.open("ab") as stderr:
            return subprocess.Popen([sys.executable, "-m", "eval_launch.local_backend", "--child", role],
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr, env=environment)

    def _header(self) -> dict[str, Any]:
        return {"scope": self.scope, "parent": os.getpid(), "hook": self.setup_hook}


    def place(self, station: Station) -> Future:
        size = self.layout.gpus
        station.held = LocalStation(self.gpu_ids[station.index * size:(station.index + 1) * size])
        return _done(None)

    def open(self, station: Station) -> None:
        held: LocalStation = station.held
        gpu = held.gpus[self.layout.server_slot().bundle]
        held.server = self._child("server", gpu, self.output / "logs" / station.name / "server-process.log")
        environment_file = None
        if self.server_environment is not None:
            environment_file = server_env.write(self.server_environment)
            self.environment_files.append(environment_file)
        try:
            _send(held.server.stdin, {**self._header(), "recipe": self.recipe, "port": station.port,
                                      "log_dir": str(self.output / "logs" / station.name), "gpu": gpu,
                                      "server_env": str(environment_file) if environment_file else None,
                                      "ready_timeout_s": self.ready_timeout_s})
        except BrokenPipeError:
            pass

    def ensure_server(self, station: Station, checkpoint: str) -> Future:
        process: subprocess.Popen = station.held.server
        gpu = station.held.gpus[self.layout.server_slot().bundle]

        def ask() -> dict[str, Any]:
            try:
                _send(process.stdin, {"op": "ensure", "checkpoint": checkpoint})
                line = process.stdout.readline()
            except (BrokenPipeError, ValueError, OSError) as error:
                raise Lost(f"its server process is gone: {error!r}") from None
            if not line:
                raise Lost(f"its server process exited ({process.poll()})")
            reply = json.loads(line)
            if "raised" in reply:
                raise Raised(reply["raised"])
            return {**reply["ok"], "gpu": gpu}

        return _in_thread(ask)

    def run(self, station: Station, slot: int, unit: Mapping[str, Any], token: str, server: Mapping[str, Any],
            deadline: float) -> Future:
        held: LocalStation = station.held
        gpu = held.gpus[self.layout.sim_slots()[slot].bundle]
        process = self._child("slot", gpu, self.output / "logs" / station.name / f"slot{slot}.log")
        held.slots[slot] = process
        try:
            _send(process.stdin, {**self._header(), "unit": dict(unit), "token": token, "output": str(self.output),
                                  "server": dict(server), "model": self.model, "deadline": deadline,
                                  "environment": self.environment, "worker": f"{station.name}.{slot}"})
            process.stdin.close()
        except BrokenPipeError:
            pass

        def collect() -> dict[str, Any]:
            lines = process.stdout.read().splitlines()
            returncode = process.wait()
            reply = json.loads(lines[-1]) if lines else {}
            if "outcome" in reply:
                return reply["outcome"]
            if "raised" in reply:
                raise Raised(reply["raised"])
            swept = stop_orphans(self.scope)
            raise Lost(f"its slot process ended ({returncode}) without a result"
                       + (f"; stopped {len(swept)} processes it left" if swept else ""))

        return _in_thread(collect)

    def probe(self, station: Station, slot: int) -> Future:
        return _done(None)

    def kill_slot(self, station: Station, slot: int) -> None:
        process = station.held.slots.get(slot) if station.held is not None else None
        if process is not None and process.poll() is None:
            process.kill()

    def tear_down(self, stations: list[Station], graceful: bool = False) -> None:
        """SIGTERM every server and slot process of the stations (a slot stops its Kit, a server its recipe's
        process), SIGKILL what is left after the grace time, then sweep what the killed ones left."""
        processes = [process for station in stations if station.held is not None
                     for process in (station.held.server, *station.held.slots.values())
                     if process is not None and process.poll() is None]
        for process in processes:
            for sig in (signal.SIGTERM, signal.SIGCONT):
                try:
                    process.send_signal(sig)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + (STOP_GRACE_S if graceful else 30.0)
        while any(process.poll() is None for process in processes) and time.monotonic() < deadline:
            time.sleep(0.2)
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait()
        for station in stations:
            station.held = None
        if processes:
            swept = stop_orphans(self.scope)
            if swept:
                print(f"[local] stopped {len(swept)} processes the stopped slots left: {swept}", flush=True)

    def quiet(self, stations: list[Station]) -> set[str]:
        """A station runs nothing when none of its slot processes (one per running unit) is alive."""
        return {station.name for station in stations if station.held is not None
                and all(process.poll() is not None for process in station.held.slots.values())}

    def release_nodes(self, kept: list[Station]) -> list[str]:
        """This machine's GPUs are no one else's to take back: a released station's are free once it is torn down."""
        return []

    def wait(self, handles: list[Future], timeout: float) -> list[Future]:
        done, _ = wait_futures(handles, timeout=timeout, return_when=FIRST_COMPLETED)
        return [handle for handle in handles if handle in done]

    def result(self, handle: Future) -> Any:
        return handle.result()


def _send(stream: IO[bytes], message: Mapping[str, Any]) -> None:
    stream.write((json.dumps(dict(message)) + "\n").encode())
    stream.flush()




def _stop_on_sigterm(signum, frame) -> None:
    raise SystemExit(128 + signum)


def _become_child(header: Mapping[str, Any]) -> IO[str]:
    """Set up a server or slot process: die with the driver, carry the launch's owner tag, run the setup hook; the
    stream its replies go on (its stdout; anything else it prints goes to its log)."""
    signal.signal(signal.SIGTERM, _stop_on_sigterm)
    try:
        ctypes.CDLL(None, use_errno=True).prctl(_PR_SET_PDEATHSIG, signal.SIGTERM)
    except (OSError, AttributeError):
        pass
    if os.getppid() != int(header["parent"]):
        raise SystemExit(1)
    replies = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)
    claim_ownership(str(header["scope"]))
    if header.get("hook"):
        module, _, name = str(header["hook"]).rpartition(".")
        getattr(importlib.import_module(module), name)()
    return replies


def _reply(replies: IO[str], message: Mapping[str, Any]) -> None:
    replies.write(json.dumps(dict(message)) + "\n")
    replies.flush()


def _serve(stdin: IO[str]) -> int:
    """A station's server process: start or keep the recipe's server for each checkpoint asked for, until its stdin
    closes or it is told to stop."""
    requests = os.fdopen(os.dup(stdin.fileno()), "r")
    os.dup2(os.open(os.devnull, os.O_RDONLY), stdin.fileno())
    header = json.loads(requests.readline())
    replies = _become_child(header)
    environment = server_env.read(header["server_env"]) if header.get("server_env") else None
    server = eval_worker.PolicyServer(header["recipe"], int(header["port"]), Path(header["log_dir"]), environment,
                                      ready_timeout_s=float(header["ready_timeout_s"]), gpu=str(header["gpu"]))
    try:
        for line in requests:
            checkpoint = json.loads(line)["checkpoint"]
            try:
                server.ensure(checkpoint)
            except Exception as error:
                _reply(replies, {"raised": repr(error)[:500]})
                continue
            _reply(replies, {"ok": {"checkpoint": checkpoint, "port": server.port, "token": server.token,
                                    "client_set": server.client_set(), "started_s": server.started_s,
                                    "pid": os.getpid()}})
    finally:
        server.stop()
    return 0


def _slot(stdin: IO[str]) -> int:
    """One unit on one slot: :func:`driver.run_slot`, its outcome the reply."""
    request = json.loads(stdin.read())
    replies = _become_child(request)
    try:
        outcome = run_slot(request["unit"], request["token"], request["output"], request["server"], request["model"],
                           float(request["deadline"]), request["environment"], request["worker"], {"pid": os.getpid()})
    except Exception as error:
        _reply(replies, {"raised": repr(error)[:500]})
        return 3
    _reply(replies, {"outcome": outcome})
    return 0


def main(argv: list[str] | None = None) -> int:
    return driver.main(argv, "local", LocalExecutor, check_local_config, __doc__)


if __name__ == "__main__":
    if sys.argv[1:3] in (["--child", "server"], ["--child", "slot"]):
        sys.exit((_serve if sys.argv[2] == "server" else _slot)(sys.stdin))
    run_main(main)
