#!/usr/bin/env python3
# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""The Ray backend: the launch's driver (:mod:`eval_launch.driver`) on stations that Ray places, on any machine Ray runs on.

    python -m eval_launch.ray_backend <config.yaml> [--resume] [--retry-failed] [--tasks <task>,...]

(``launch.sh <config> --local`` or ``--resume`` runs it for a config with ``backend.kind: ray``.) The same code runs on
a workstation (a Ray instance the driver starts on its GPUs), on Kubernetes pods or SLURM nodes that form a Ray cluster
(``backend.ray.address``); nothing here names a cluster, and the launch's state is files on the launch's output, read
and written by the driver alone. On one machine, ``backend.kind: local`` (:mod:`eval_launch.local_backend`) runs the
same driver without Ray.

A station is one placement group of one-GPU bundles packed onto one node. Its server is a :class:`ServerActor`, which
owns the recipe's server process (``eval_worker.PolicyServer``: setup, command, token, ready line, probe) and keeps it
for every unit of its checkpoint; each simulator slot is a :class:`SimActor`, which runs one unit at a time as
``run_benchmark_cell.py`` in its own Kit process (``eval_worker.run_unit``, unchanged) with the
``CUDA_VISIBLE_DEVICES`` Ray gave the actor: one GPU, its own. Ray decides where a station goes and restarts a dead
actor in place (``backend.ray.actor_restarts``); it never retries a unit (``max_task_retries=0``): the driver does,
through the launch's record (:mod:`eval_launch.acceptance`), so a retried unit gets a new attempt token and its own
``units/<id>/<token>/``.

A server starts in the environment the driver was handed (``--server-env``: the launch's, before Isaac Sim's paths),
minus its secrets: those are named in the config (``layout.secret_names``) and travel by name only, so no value sits
in Ray's metadata on the head; the server actor reads each one from its own process. On Kubernetes that is the worker
pod's, from the secret (``ray_k8s.render_ray``); a Ray instance the driver starts inherits the driver's; a node of
a cluster started by hand inherits the environment ``ray start`` ran in.

Processes an actor started outlive it when it is killed (each Kit and server runs in its own session); they carry the
launch's owner tag, so a new actor on the node and the executor's sweep at start and stop end them
(``process_group.stop_orphans``).
"""

from __future__ import annotations

import argparse
import os
import secrets
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import ray
from ray.exceptions import ActorDiedError, RayError, RayTaskError
from ray.util.placement_group import placement_group, placement_group_table, remove_placement_group
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy, PlacementGroupSchedulingStrategy

from eval_launch import driver, eval_worker, run_main
from eval_launch.driver import Gone, Lost, Raised, Station, join_pythonpath, owner_scope, run_slot
from eval_launch.layout import PORT_BASE, Layout, check_ray_config, gpu_count, parse_layout, secret_names
from eval_launch.plan import model_spec
from eval_launch.process_group import claim_ownership, stop_orphans


def _where() -> dict[str, Any]:
    return {"pid": os.getpid(), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpu_ids": ray.get_gpu_ids(),
            "node": ray.get_runtime_context().get_node_id()}


class ServerActor:
    """A station's policy server: the recipe's process on the actor's GPU, one checkpoint at a time."""

    def __init__(self, recipe: Mapping[str, Any], port: int, log_dir: str, environment: Mapping[str, str] | None,
                 secrets: list[str], ready_timeout_s: float, scope: str, pythonpath: list[str]):
        missing = [name for name in secrets if name not in os.environ]
        if missing:
            raise RuntimeError(f"{missing} (the config's secret_env) are not set on this Ray node: a k8s worker pod takes "
                               "them from its secret; elsewhere, run `ray start` where they are set")
        if environment is not None:
            environment = {**environment, **{name: os.environ[name] for name in secrets}}
        stop_orphans(scope)
        claim_ownership(scope)
        join_pythonpath(pythonpath)
        self.server = eval_worker.PolicyServer(recipe, port, Path(log_dir), environment, ready_timeout_s=ready_timeout_s,
                                               gpu=os.environ.get("CUDA_VISIBLE_DEVICES"))

    def ensure(self, checkpoint: str) -> dict[str, Any]:
        """Serve ``checkpoint`` (start or keep the server); what the simulators need to reach it."""
        self.server.ensure(checkpoint)
        return {"checkpoint": checkpoint, "port": self.server.port, "token": self.server.token,
                "client_set": self.server.client_set(), "started_s": self.server.started_s, **_where()}

    def where(self) -> dict[str, Any]:
        return _where()

    def stop(self) -> None:
        self.server.stop()


class SimActor:
    """One simulator slot: runs a unit at a time, each in its own Kit process on the actor's GPU."""

    def __init__(self, name: str, scope: str, pythonpath: list[str]):
        stop_orphans(scope)
        claim_ownership(scope)
        join_pythonpath(pythonpath)
        self.name = name

    def where(self) -> dict[str, Any]:
        return _where()

    def run(self, unit: Mapping[str, Any], token: str, output: str, server: Mapping[str, Any],
            model: str | Mapping[str, Any], deadline: float, environment: Mapping[str, str]) -> dict[str, Any]:
        return run_slot(unit, token, output, server, model, deadline, environment, self.name,
                        {"node": ray.get_runtime_context().get_node_id()})


def _sweep(scope: str) -> list[int]:
    return stop_orphans(scope)


ServerRemote = ray.remote(ServerActor)
SimRemote = ray.remote(SimActor)
SweepRemote = ray.remote(_sweep)

POD_RESOURCE = "eval_launch_pod_"
"""A worker pod's Ray node advertises one unit of ``<POD_RESOURCE><its pod name, non-alphanumerics as _>``
(``ray_k8s``): the name the driver releases it by, unique to that pod (a Job index can have two pods at once, a
restarted pod gets a new name)."""
JOIN_RESOURCE = "eval_launch_join_"
"""Each time a worker pod's Ray node joins, it also advertises one unit of ``<JOIN_RESOURCE><a random tag>``
(``ray_k8s``): which incarnation of the pod's node it is. A release names it, and the pod acts on a release of its
current node only."""


def _join(node: Mapping[str, Any] | None) -> str | None:
    """The join tag a Ray node advertises (:data:`JOIN_RESOURCE`), or None."""
    tags = [key[len(JOIN_RESOURCE):] for key in ((node or {}).get("Resources") or {}) if key.startswith(JOIN_RESOURCE)]
    return tags[0] if len(tags) == 1 else None


def _released_join(path: Path) -> str | None:
    """The join tag a release file names (``join <tag>``), or None."""
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return None
    return next((line.split(" ", 1)[1] for line in lines if line.startswith("join ")), None)


@dataclass
class RayStation:
    """What the Ray executor holds for one station."""

    pg: Any
    server: Any = None
    sims: list = field(default_factory=list)


class RayExecutor:
    """A station is a placement group; its server and each simulator slot are actors in it (:class:`driver.Executor`)."""

    kind = "ray"

    def __init__(self, config: Mapping[str, Any], output: Path, server_environment: Mapping[str, str] | None = None):
        self.config, self.output = config, Path(output)
        self.secrets = secret_names(config)
        self.server_environment = ({name: value for name, value in server_environment.items() if name not in self.secrets}
                                   if server_environment is not None else None)
        backend = config.get("backend") or {}
        self.ray_config = dict(backend.get("ray") or {})
        self.layout: Layout = parse_layout(config.get("servers"))
        self.gpus = gpu_count(backend.get("gpus", 1))
        self.port_base = int(self.ray_config.get("port_base", PORT_BASE))
        self.slot_restarts = int(self.ray_config.get("actor_restarts", 3))
        self.station_restarts = int(self.ray_config.get("station_restarts", 2))
        self.plan_config = config.get("plan") or {}
        self.recipe = (config.get("servers") or {})["recipe"]
        self.model = model_spec(config)
        self.environment = {str(k): str(v) for k, v in (backend.get("env") or {}).items()}
        self.pythonpath = [str(path) for path in backend.get("pythonpath", [])]
        self.scope = owner_scope(self.output)
        self.epoch = 0
        self.started_ray = False
        self.release_dir = Path(self.ray_config["release_dir"]) if self.ray_config.get("release_dir") else None
        self.fences: dict[str, Any] = {}
        """Released pods' fences, by pod name: held until this driver ends (:meth:`release_nodes`)."""

    def start(self, epoch: int) -> list[int]:
        self.epoch = epoch
        if not ray.is_initialized():
            options: dict[str, Any] = {"namespace": "eval_launch", "include_dashboard": False}
            if self.ray_config.get("runtime_env"):
                options["runtime_env"] = dict(self.ray_config["runtime_env"])
            if self.ray_config.get("address"):
                ray.init(address=str(self.ray_config["address"]), **options)
            else:
                gpus = (self.config.get("backend") or {}).get("gpus", 1)
                if isinstance(gpus, list):
                    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in gpus)
                ray.init(num_gpus=self.gpus, num_cpus=self.ray_config.get("num_cpus"),
                         object_store_memory=int(float(self.ray_config.get("object_store_gb", 1.0)) * 2**30), **options)
            self.started_ray = True
        swept = self._sweep_nodes()
        self._fence_released()
        return swept

    def _fence_released(self, timeout_s: float = 300.0) -> None:
        """Before any station is placed: every live pod an earlier driver released (it has not ended yet) is fenced,
        so it takes no station of this driver's, or has gone. Refuses to start otherwise (the k8s head then fails and
        its Job starts another): a station placed there would be stopped with the pod."""
        if self.release_dir is None:
            return
        deadline = time.monotonic() + timeout_s
        while True:
            self._recheck(self._pods())
            waiting = [name for name in self._pods() if (self.release_dir / name).exists() and name not in self.fences]
            if not waiting:
                return
            for name in self._fence(waiting, wait_s=5.0):
                self._publish_if_held(name)
            if time.monotonic() > deadline:
                raise RuntimeError(f"released worker pods {waiting} are still up and could not be fenced in "
                                   f"{timeout_s:.0f} s: not placing any station")

    def stop(self) -> None:
        if self.started_ray:
            ray.shutdown()
            self.started_ray = False

    def _sweep_nodes(self) -> list[int]:
        """Stop what dead actors of this launch left on every node (``process_group.stop_orphans``)."""
        refs = [SweepRemote.options(num_cpus=0, max_retries=0, scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node["NodeID"], soft=False)).remote(self.scope) for node in ray.nodes() if node.get("Alive")]
        swept = []
        for ref in refs:
            try:
                swept += ray.get(ref)
            except RayError as error:
                print(f"[ray] a node's sweep did not run: {_cause(error)}", flush=True)
        return swept

    def place(self, station: Station) -> Any:
        pg = placement_group(self.layout.bundles(), strategy="STRICT_PACK",
                             name=f"eval-{self.epoch}-{station.name}-{station.failures}")
        station.held = RayStation(pg)
        return pg.ready()

    def open(self, station: Station) -> None:
        held: RayStation = station.held

        def strategy(bundle: int) -> PlacementGroupSchedulingStrategy:
            return PlacementGroupSchedulingStrategy(placement_group=held.pg, placement_group_bundle_index=bundle)

        server_slot = self.layout.server_slot()
        held.server = ServerRemote.options(
            num_cpus=0, num_gpus=server_slot.gpu_share, max_restarts=self.slot_restarts, max_task_retries=0,
            scheduling_strategy=strategy(server_slot.bundle),
        ).remote(self.recipe, station.port, os.path.abspath(self.output / "logs" / station.name), self.server_environment,
                 self.secrets, float(self.plan_config.get("server_ready_timeout_s", eval_worker.SERVER_READY_TIMEOUT_S)), self.scope,
                 self.pythonpath)
        held.sims = [SimRemote.options(num_cpus=0, num_gpus=slot.gpu_share, max_restarts=self.slot_restarts,
                                       max_task_retries=0, scheduling_strategy=strategy(slot.bundle),
                                       ).remote(f"{station.name}.{index}", self.scope, self.pythonpath)
                     for index, slot in enumerate(self.layout.sim_slots())]

    def ensure_server(self, station: Station, checkpoint: str) -> Any:
        return station.held.server.ensure.remote(checkpoint)

    def run(self, station: Station, slot: int, unit: Mapping[str, Any], token: str, server: Mapping[str, Any],
            deadline: float) -> Any:
        return station.held.sims[slot].run.remote(unit, token, str(self.output), server, self.model, deadline,
                                                  self.environment)

    def probe(self, station: Station, slot: int) -> Any:
        return station.held.sims[slot].where.remote()

    def kill_slot(self, station: Station, slot: int) -> None:
        ray.kill(station.held.sims[slot], no_restart=False)

    def tear_down(self, stations: list[Station], graceful: bool = False) -> None:
        """Stop the stations' actors and free their GPUs. A killed actor leaves its server and Kit running (each in its
        own session), and Ray cannot interrupt a running call (``ray.cancel`` leaves a sync actor's call running), so:
        the servers are asked to stop first when ``graceful``; then every actor is killed, and its death awaited
        before the sweep, which stops only what a dead actor left."""
        held = [station.held for station in stations if station.held is not None]
        if graceful:
            stops = [entry.server.stop.remote() for entry in held if entry.server is not None]
            if stops:
                ray.wait(stops, num_returns=len(stops), timeout=120.0)
        actors = [actor for entry in held for actor in (entry.server, *entry.sims) if actor is not None]
        for actor in actors:
            ray.kill(actor, no_restart=True)
        probes = [actor.where.remote() for actor in actors]
        deadline = time.monotonic() + 60.0
        while probes and time.monotonic() < deadline:
            done, probes = ray.wait(probes, num_returns=len(probes), timeout=1.0)
        for entry in held:
            remove_placement_group(entry.pg)
        for station in stations:
            station.held = None
        if actors:
            swept = self._sweep_nodes()
            if swept:
                print(f"[ray] stopped {len(swept)} processes the stopped actors left: {swept}", flush=True)

    def quiet(self, stations: list[Station], timeout: float = 5.0) -> set[str]:
        """Ask each station's server and simulator actors where they are: an actor runs one call at a time, so an
        answer within ``timeout`` means no call of its was running (a unit, a server load). A station whose every actor
        answers is quiet; one with an actor that does not, or that is dead or restarting, is not."""
        asked = {station.name: [actor.where.remote() for actor in (station.held.server, *station.held.sims)
                                if actor is not None]
                 for station in stations if station.held is not None}
        refs = [ref for refs in asked.values() for ref in refs]
        done = set(ray.wait(refs, num_returns=len(refs), timeout=timeout)[0]) if refs else set()
        quiet = set()
        for name, refs in asked.items():
            if refs and all(ref in done for ref in refs):
                try:
                    ray.get(refs)
                except RayError:
                    continue
                quiet.add(name)
        return quiet

    def _pods(self) -> dict[str, dict[str, Any]]:
        """The live worker pods' Ray nodes by pod name (their Job index, from the resource each advertises)."""
        pods = {}
        for node in ray.nodes():
            names = [key[len(POD_RESOURCE):] for key in (node.get("Resources") or {}) if key.startswith(POD_RESOURCE)]
            if node.get("Alive") and len(names) == 1:
                pods[names[0]] = node
        return pods

    def _fence(self, names: list[str], wait_s: float) -> list[str]:
        """Claim each named pod's node whole: a placement group of one bundle of its pod resource and all its GPUs.
        Ray places it only once nothing on that node holds a GPU (every station there torn down, its placement group
        gone), and while it stands Ray places nothing else there. The ones placed within ``wait_s`` are kept (held
        until this driver ends) and returned; the others are removed again."""
        pods = self._pods()
        asked = {}
        for name in names:
            node = pods.get(name)
            gpus = float((node or {}).get("Resources", {}).get("GPU", 0))
            if node is None or name in self.fences or gpus <= 0:
                continue
            asked[name] = placement_group([{f"{POD_RESOURCE}{name}": 1, "GPU": gpus}], strategy="STRICT_PACK",
                                          name=f"eval-{self.epoch}-release-{name}-{secrets.token_hex(4)}")
        refs = {pg.ready(): name for name, pg in asked.items()}
        done = set(ray.wait(list(refs), num_returns=len(refs), timeout=wait_s)[0]) if refs else set()
        fenced = []
        for ref, name in refs.items():
            placed = False
            if ref in done:
                try:
                    ray.get(ref)
                    placed = True
                except RayError as error:
                    print(f"[ray] the fence of pod {name} failed: {_cause(error)}", flush=True)
            if placed:
                self.fences[name] = asked[name]
                fenced.append(name)
            else:
                remove_placement_group(asked[name])
        return fenced

    def _held(self, name: str, pods: Mapping[str, Mapping[str, Any]]) -> bool:
        """Whether this executor's fence of pod ``name`` stands now: Ray has placed it (``CREATED``), on the node the
        pod has now (the fence's bundle is on that node's id), and that node advertises a join tag. A fence Ray moved
        or is waiting to place again (the pod's node died; a new node of the pod may have joined) does not."""
        node = pods.get(name)
        if name not in self.fences or node is None or _join(node) is None:
            return False
        table = placement_group_table(self.fences[name])
        placed_on = set((table.get("bundles_to_node_id") or {}).values())
        return table.get("state") == "CREATED" and placed_on == {node.get("NodeID")}

    def _drop(self, name: str) -> None:
        """Give up the fence of pod ``name``: removed from Ray and forgotten, so the pod can be fenced again."""
        try:
            remove_placement_group(self.fences.pop(name))
        except (KeyError, RayError, ValueError) as error:
            print(f"[ray] removing pod {name}'s fence: {error}", flush=True)

    def _publish(self, name: str, node: Mapping[str, Any]) -> None:
        """Write ``<release_dir>/<pod>`` naming the node it releases (its join tag and Ray node id), in one rename."""
        self.release_dir.mkdir(parents=True, exist_ok=True)
        path = self.release_dir / name
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(f"join {_join(node)}\nnode {node.get('NodeID')}\n"
                             f"fenced {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
        os.replace(temporary, path)

    def _publish_if_held(self, name: str) -> bool:
        """Release pod ``name`` once its fence is checked against Ray, before anything is written; a fence that does not
        stand is dropped (the pod is fenced again on a later pass) and nothing is published."""
        pods = self._pods()
        if not self._held(name, pods):
            self._drop(name)
            print(f"[ray] pod {name}'s fence does not stand on its current node: not released", flush=True)
            return False
        if _released_join(self.release_dir / name) != _join(pods[name]):
            self._publish(name, pods[name])
        return True

    def _recheck(self, pods: Mapping[str, Mapping[str, Any]]) -> None:
        """Every fence this executor holds, checked against Ray now. One that no longer stands is dropped; if its pod
        is still up (a new node of it joined, or the fence was removed) its release is taken back, so the pod is
        fenced and released again from the start. A pod that has gone keeps its release file, so it never rejoins.
        One that stands while its release names an earlier node of the pod (the node died and a new one joined between
        the check and the write) is released again, naming the current node."""
        for name in list(self.fences):
            path = self.release_dir / name
            if self._held(name, pods):
                if path.exists() and _released_join(path) != _join(pods[name]):
                    self._publish(name, pods[name])
                continue
            self._drop(name)
            if name in pods:
                path.unlink(missing_ok=True)
                print(f"[ray] pod {name}'s fence was lost while the pod is up: its release is taken back", flush=True)

    def release_nodes(self, kept: list[Station]) -> list[str]:
        """Write ``<release_dir>/<pod>`` for each live worker pod whose whole node could be fenced (:meth:`_fence`) and
        whose fence is checked to stand on the pod's current node before the file is written (:meth:`_held`): its
        pod stops its Ray node, if the release names that node, and ends (``ray_k8s.RAY_WORKER_SCRIPT``). The proof
        that nothing runs there is Ray's own placement, not what this driver remembers of where its stations went (Ray
        moves a placement group's bundles when a node dies), and every fence held is checked again on each pass
        (:meth:`_recheck`). ``kept`` stations hold their GPUs, so their nodes cannot be fenced. Nothing outside
        Kubernetes (no ``release_dir``), and never the head's node (it advertises no pod resource)."""
        if self.release_dir is None:
            return []
        pods = self._pods()
        self._recheck(pods)
        candidates = [name for name in pods if name not in self.fences and not (self.release_dir / name).exists()]
        return [f"pod {name}" for name in self._fence(candidates, wait_s=5.0) if self._publish_if_held(name)]

    def wait(self, handles: list[Any], timeout: float) -> list[Any]:
        return ray.wait(handles, num_returns=1, timeout=timeout)[0]

    def result(self, handle: Any) -> Any:
        try:
            return ray.get(handle)
        except RayTaskError as error:
            raise Raised(_cause(error)) from None
        except ActorDiedError as error:
            raise Gone(_cause(error)) from None
        except RayError as error:
            raise Lost(f"{type(error).__name__}: {_cause(error)}") from None


def _cause(error: BaseException) -> str:
    cause = getattr(error, "cause", None)
    return repr(cause if cause is not None else error)[:500]


def _ray_options(parser: argparse.ArgumentParser) -> Callable[[argparse.Namespace, dict], None]:
    parser.add_argument("--ray-address",
                        help="Join this running Ray cluster (e.g. auto) whatever backend.ray.address says: the k8s head "
                             "pod runs the driver beside the Ray head it started (ray_k8s.render_ray).")
    parser.add_argument("--release-dir",
                        help="Where to name each worker pod whose stations are all released once nothing is left to "
                             "start, so it ends and frees its GPUs (ray_k8s.render_ray passes it).")

    def apply(args: argparse.Namespace, config: dict) -> None:
        backend = config.setdefault("backend", {})
        if args.ray_address:
            backend["ray"] = {**(backend.get("ray") or {}), "address": args.ray_address}
        if args.release_dir:
            backend["ray"] = {**(backend.get("ray") or {}), "release_dir": args.release_dir}

    return apply


def main(argv: list[str] | None = None) -> int:
    return driver.main(argv, "ray", RayExecutor, check_ray_config, __doc__, _ray_options)


if __name__ == "__main__":
    run_main(main)
