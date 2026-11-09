# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Where a launch's processes go on its GPUs: a policy server and the simulators it serves, as one placement shape.

A *station* is one policy server and its simulators. Its placement group has one bundle of one GPU per GPU it takes;
bundle 0 holds the server, and the simulators take the other bundles, or bundle 0 too when they share the server's
GPU. Every simulator is its own Kit process and sees one GPU only (``CUDA_VISIBLE_DEVICES``, which Ray sets for each
actor from its share of a bundle). The config's ``servers.layout`` names the shape:

- ``colocated`` (the default; RLDX, GR00T and pi0.5 on one L40S each): one GPU, the server and one simulator on it, the
  server kept for every unit of its checkpoint, as the directory backend's worker does;
- ``server+N``: two GPUs, the server alone on one, ``N`` simulators sharing the other (``server+1`` for Cosmos3,
  whose server takes 33 of an L40S's 46 GB; ``server+2`` for the two-simulator pilot);
- ``{sim_gpus: G, sims_per_gpu: N}``: the server alone on one GPU, ``N`` simulators on each of ``G`` more; with
  ``G: 0``, the server and ``N`` simulators on one GPU (``colocated+N``).

Ray's fractional GPUs only divide a GPU between processes for scheduling: nothing holds a process to its share of the
memory. So a layout that puts two processes on one GPU is checked against the memory the config states
(:func:`check_memory`), and refused when it cannot fit; the one case known so far is the Cosmos3 server with a
simulator beside it on one L40S (an M1 simulator beside it ran out of memory, 2026-10-23).

Ray-free and Kit-free: the planner, the tests, the k8s renderer and the driver with either executor read it.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

COLOCATED = "colocated"
_SERVER_PLUS = re.compile(r"server\+(\d+)")


@dataclass(frozen=True)
class Slot:
    """Where one process of a station runs: its bundle and its share of that bundle's GPU."""

    bundle: int
    gpu_share: float


@dataclass(frozen=True)
class Layout:
    name: str
    sim_gpus: int
    """GPUs for the simulators besides the server's; 0 when they share the server's GPU (``colocated``)."""
    sims_per_gpu: int

    @property
    def gpus(self) -> int:
        """GPUs one station takes."""
        return 1 + self.sim_gpus

    @property
    def sims(self) -> int:
        """Simulators one station runs at once."""
        return self.sims_per_gpu * max(1, self.sim_gpus)

    def bundles(self) -> list[dict[str, float]]:
        """The station's placement group: one bundle of one GPU per GPU (packed onto one node, so the simulators reach
        their server on 127.0.0.1)."""
        return [{"GPU": 1.0} for _ in range(self.gpus)]

    def server_slot(self) -> Slot:
        return Slot(0, _share(1 + self.sims_per_gpu) if self.sim_gpus == 0 else 1.0)

    def sim_slots(self) -> list[Slot]:
        if self.sim_gpus == 0:
            return [Slot(0, _share(1 + self.sims_per_gpu))] * self.sims_per_gpu
        return [Slot(1 + gpu, _share(self.sims_per_gpu)) for gpu in range(self.sim_gpus) for _ in range(self.sims_per_gpu)]

    def processes_per_gpu(self) -> list[tuple[int, int]]:
        """``[(servers, simulators)]`` on each of the station's GPUs."""
        if self.sim_gpus == 0:
            return [(1, self.sims_per_gpu)]
        return [(1, 0)] + [(0, self.sims_per_gpu)] * self.sim_gpus


def _share(processes: int) -> float:
    """One of ``processes`` equal shares of a GPU, rounded down to Ray's resource precision (1e-4) so that their sum
    never exceeds the bundle's one GPU."""
    return math.floor(10000 / processes) / 10000


MAX_SIMS_PER_GPU = 3
"""``servers.max_sims_per_gpu``'s default: the most simulators a layout may put on one GPU. Measured on L40S (46 GB):
4 simulators peaked at 33 GB with the GPU saturated (~93% utilisation) and steps 1.5-2.6x slower, against 1.3-2.1x at 3
(a simulators-per-GPU measurement, 2026-11-05). Another card states its own."""


def parse_layout(servers: Mapping[str, Any] | None) -> Layout:
    """The config's ``servers.layout`` (``colocated`` when it names none), refused when it puts more simulators on one
    GPU than ``servers.max_sims_per_gpu`` (:data:`MAX_SIMS_PER_GPU`). The memory check (:func:`check_memory`) and the
    driver's deadline scaling (times ``sims_per_gpu``) both read the layout this returns, so neither sees more."""
    layout = _parse_layout((servers or {}).get("layout", COLOCATED))
    cap = (servers or {}).get("max_sims_per_gpu", MAX_SIMS_PER_GPU)
    if type(cap) is not int or cap < 1:
        raise ValueError(f"servers.max_sims_per_gpu is a count of at least 1, not {cap!r}")
    if layout.sims_per_gpu > cap:
        raise ValueError(f"layout {layout.name} puts {layout.sims_per_gpu} simulators on one GPU, more than "
                         f"servers.max_sims_per_gpu ({cap})")
    return layout


def _parse_layout(value: Any) -> Layout:
    if value == COLOCATED:
        return Layout(COLOCATED, sim_gpus=0, sims_per_gpu=1)
    if isinstance(value, str):
        match = _SERVER_PLUS.fullmatch(value)
        if match and int(match.group(1)) >= 1:
            return Layout(value, sim_gpus=1, sims_per_gpu=int(match.group(1)))
    if isinstance(value, Mapping) and set(value) == {"sim_gpus", "sims_per_gpu"}:
        sim_gpus, sims_per_gpu = value["sim_gpus"], value["sims_per_gpu"]
        if type(sim_gpus) is int and type(sims_per_gpu) is int and sim_gpus >= 0 and sims_per_gpu >= 1:
            name = f"{COLOCATED}+{sims_per_gpu}" if sim_gpus == 0 else f"server+{sim_gpus}x{sims_per_gpu}"
            return Layout(name, sim_gpus=sim_gpus, sims_per_gpu=sims_per_gpu)
    raise ValueError(f"servers.layout is {COLOCATED!r}, 'server+N' (N >= 1) or {{sim_gpus: G, sims_per_gpu: N}} "
                     f"(G >= 0, N >= 1), not {value!r}")


def stations(layout: Layout, gpus: int) -> int:
    """How many stations ``gpus`` GPUs hold; refuses a count that leaves a GPU idle or holds none."""
    if gpus < layout.gpus or gpus % layout.gpus:
        raise ValueError(f"backend.gpus is {gpus}: layout {layout.name} takes {layout.gpus} GPUs per server, so give a "
                         f"multiple of {layout.gpus}")
    return gpus // layout.gpus


def check_memory(layout: Layout, servers: Mapping[str, Any] | None, card_gb: float | None) -> None:
    """Refuse a layout whose processes cannot fit in a GPU's memory."""
    memory = (servers or {}).get("gpu_memory_gb") or {}
    unknown = sorted(set(memory) - {"server", "sim"})
    if unknown:
        raise ValueError(f"servers.gpu_memory_gb has server and sim, not {unknown}")
    shared = any(sims > 1 for _, sims in layout.processes_per_gpu())
    beside_server = any(servers_here and sims > 1 for servers_here, sims in layout.processes_per_gpu())
    if shared and not (card_gb and "sim" in memory and ("server" in memory or not beside_server)):
        raise ValueError(f"layout {layout.name} runs {layout.sims_per_gpu} simulators on one GPU: state "
                         "servers.gpu_memory_gb.sim (and .server, when they share its GPU) and backend.gpu_memory_gb, so the launcher can check they fit "
                         "(Ray does not hold a process to its share of a GPU)")
    if not card_gb:
        return
    for index, (servers_here, sims_here) in enumerate(layout.processes_per_gpu()):
        if servers_here and "server" not in memory and sims_here == 1:
            continue
        need = servers_here * float(memory.get("server", 0.0)) + sims_here * float(memory.get("sim", 0.0))
        if need > float(card_gb):
            raise ValueError(f"layout {layout.name} puts {servers_here} server(s) and {sims_here} simulator(s) on GPU "
                             f"{index} of each station: {need:g} GB of a {float(card_gb):g} GB GPU "
                             "(servers.gpu_memory_gb); choose a layout that gives the server its own GPU")


def gpu_count(gpus: int | list) -> int:
    """``backend.gpus`` as a count: a number, or the list of GPU ids a local backend uses."""
    return len(gpus) if isinstance(gpus, list) else int(gpus)


def check_config(config: Mapping[str, Any], kind: str, keys: set[str], gpus: int) -> Layout:
    """The config's layout, refused when it cannot run on ``gpus`` GPUs (its GPU count, its memory) or names a key of
    ``backend.<kind>`` outside ``keys``; before anything is written."""
    backend = config.get("backend") or {}
    if backend.get("kind") != kind:
        raise ValueError(f"the {kind} backend runs a config with backend.kind: {kind}")
    layout = parse_layout(config.get("servers"))
    stations(layout, gpus)
    check_memory(layout, config.get("servers"), backend.get("gpu_memory_gb"))
    unknown = sorted(set(backend.get(kind) or {}) - keys)
    if unknown:
        raise ValueError(f"backend.{kind} has {sorted(keys)}, not {unknown}")
    return layout


def check_ray_config(config: Mapping[str, Any]) -> Layout:
    """A ``backend.kind: ray`` config's layout, refused when it cannot run (its GPU count, its memory); before anything
    is written. Here rather than in the Ray backend, so the k8s renderer checks a config without Ray installed."""
    layout = check_config(config, "ray", RAY_KEYS, gpu_count((config.get("backend") or {}).get("gpus", 1)))
    names = ((config.get("backend") or {}).get("ray") or {}).get("secret_env", [])
    if not isinstance(names, list) or not all(isinstance(name, str) and name for name in names):
        raise ValueError("backend.ray.secret_env lists variable names, e.g. [HF_TOKEN]: their values come from each Ray "
                         "node's own environment, never from the config")
    return layout


def secret_names(config: Mapping[str, Any]) -> list[str]:
    """The policy servers' secret variables on Ray: ``backend.k8s.secret_env``'s (each worker pod takes them from its
    Kubernetes secret) and ``backend.ray.secret_env``'s (a Ray cluster outside Kubernetes: each node's ``ray start``
    ran where they are set). The driver passes their names only; a server actor reads the values from its own
    process."""
    backend = config.get("backend") or {}
    return sorted({*((backend.get("k8s") or {}).get("secret_env") or {}),
                   *((backend.get("ray") or {}).get("secret_env") or [])})


PORT_BASE = 5810
"""Station i's server listens on ``port_base + 10 i``, on either executor, unless ``backend.<kind>.port_base`` says."""
RAY_KEYS = {"address", "num_cpus", "object_store_gb", "runtime_env", "port_base", "actor_restarts", "station_restarts",
            "secret_env", "release_dir"}
"""``backend.ray``: ``address`` of a running Ray cluster (none: start one on this host with ``backend.gpus``),
``num_cpus`` and ``object_store_gb`` (1) of that instance, ``runtime_env`` for its workers, ``port_base``
(:data:`PORT_BASE`), ``actor_restarts`` per actor (3), ``station_restarts`` per station (2), ``secret_env``: the
names of the servers' secret variables, which each node's environment holds (:func:`secret_names`), and
``release_dir``: where the driver names each Kubernetes worker pod it no longer needs (``ray_k8s`` sets it with
``--release-dir``; none elsewhere)."""
