# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""The Ray backend on Kubernetes: a launch as a Ray cluster of plain pods, a head pod that runs the driver
(:mod:`eval_launch.ray_backend`) beside the Ray head it starts, and one-station worker pods that join it.

``--k8s-job`` renders it for a config with ``backend.kind: ray`` (:func:`render_ray`); the per-GPU queue Job of
``backend.kind: k8s`` is :mod:`eval_launch.k8s_backend`, whose staged-tree script, labels and secret handling this
module reuses. Rendering needs no Ray: only the pods import it.
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any

from eval_launch.acceptance import EXIT_DRIVER_RUNNING, EXIT_ENDED_WITH_FAILURES
from eval_launch.k8s_backend import PACKAGE_PARENT, TREE, _environment, annotations, labels, pod_security, secret_environment
from eval_launch.layout import check_ray_config, gpu_count, parse_layout

POD_RESOURCE = "eval_launch_pod_"
"""= ``ray_backend.POD_RESOURCE`` (not imported: rendering needs no Ray): a worker pod's Ray node advertises one unit of
``<POD_RESOURCE><its pod name, non-alphanumerics as _>``, the name its driver releases it by."""
JOIN_RESOURCE = "eval_launch_join_"
"""= ``ray_backend.JOIN_RESOURCE``: each join of a worker pod's Ray node also advertises one unit of
``<JOIN_RESOURCE><a random tag>``, which a release names; the pod acts on a release of its current node only."""


RAY_HEAD_SCRIPT = """set -euo pipefail
trap 'exit 143' TERM
# The pod's environment, for the policy servers, before anything Isaac Sim's is added (the secret_env values are the
# worker pods'): owner-only, a copy per driver start, which the driver removes once read (server_env.read); a TERM
# removes both too.
E=${{TMPDIR:-/tmp}}/server.env
trap 'rm -f "$E" "$E.driver"' EXIT  # every way out, set -e's included
trap 'rm -f "$E" "$E.driver"; exit 143' TERM
(umask 077 && env -0 > "$E")
""" + TREE + """RAY=({python} -m ray.scripts.scripts)  # an array: the interpreter path is one word, whatever it holds
"${{RAY[@]}}" start --head --port={gcs_port} --node-ip-address="$POD_IP" --num-cpus=0 --num-gpus=0 \\
  --object-store-memory={object_store} --include-dashboard=false --disable-usage-stats
# The driver plans (or adopts the plan of) the launch and runs it; after its predecessor died, it ends what that one
# left (LaunchState.recover). Its lock outlives a dead pod on a network filesystem until the lease expires: wait for it.
rc={driver_running}
for try in $(seq {lock_tries}); do
  (umask 077 && cp "$E" "$E.driver")
  {package_parent}/eval_launch/signal_group.sh {python} -m eval_launch.ray_backend {config} --ray-address auto \\
    --server-env "$E.driver" --release-dir {release_dir} &
  driver=$!
  trap 'kill -TERM $driver 2>/dev/null; wait $driver; rm -f "$E" "$E.driver"; "${{RAY[@]}}" stop > /dev/null 2>&1; exit 143' TERM
  rc=0; wait $driver || rc=$?
  trap 'rm -f "$E" "$E.driver"; exit 143' TERM
  [ "$rc" = {driver_running} ] || break
  echo "another driver holds the launch; trying again in {lock_wait_s} s"
  sleep {lock_wait_s}
done
rm -f "$E" "$E.driver"
"${{RAY[@]}}" stop > /dev/null 2>&1 || true
case "$rc" in
  # Every unit has ended ({ended_with_failures}: some failed for good; --status says which): the workers may go. Any other
  # end (a crash, every station retired) fails this pod, and the Job starts a new head and driver (head_restarts).
  0|{ended_with_failures}) mkdir -p {marker_dir} && echo "$rc" > {marker} && exit 0 ;;
  *) exit "$rc" ;;
esac
"""
RAY_WORKER_SCRIPT = """set -euo pipefail
trap 'exit 143' TERM
""" + TREE + """RAY=({python} -m ray.scripts.scripts)  # an array: the interpreter path is one word, whatever it holds
trap '"${{RAY[@]}}" stop --force > /dev/null 2>&1; exit 143' TERM
# A Ray node of the launch until its head says every unit has ended, or its driver has released this pod: nothing was
# left to start, its stations were torn down and Ray placed the driver's fence of all this node's GPUs
# (ray_backend.RayExecutor.release_nodes), so its GPUs go back while the last units run elsewhere. The release names
# this pod by its own name (unique per pod: a Job index can have two pods for a moment), which Ray sees as a resource
# of this node; it is looked for before each join, so a released pod never joins again. It also names which join of
# the pod it releases (a random tag each join advertises, written by the driver once it has checked its fence stands
# on that node): a release of an earlier node of this pod does not stop the current one. A node whose head died (its
# raylet stops once the head is gone) joins the next head; what its dead actors' simulators left running, the next
# actor here stops.
POD=${{POD_NAME//[^A-Za-z0-9]/_}}
RELEASED={release_dir}/$POD
released() {{ [ -e "$RELEASED" ] && grep -qx "join $JOIN" "$RELEASED" 2> /dev/null; }}
while [ ! -e {marker} ] && [ ! -e "$RELEASED" ]; do
  JOIN=$(od -An -N8 -tx1 /dev/urandom | tr -d ' \\n')
  "${{RAY[@]}}" start --address={head}:{gcs_port} --node-ip-address="$POD_IP" --num-gpus={gpus} --num-cpus={cpus} \\
    --resources="{{\\"{pod_resource}$POD\\": 1, \\"{join_resource}$JOIN\\": 1}}" \\
    --object-store-memory={object_store} --disable-usage-stats --block &
  node=$!
  # The release ends the node from beside it; the wait stays in front, so a stopped node rejoins and a TERM is
  # handled as before
  ( while kill -0 $node 2> /dev/null && ! released; do sleep {poll_s}; done
    # Again until the node has gone: a stop that ran while the node was still starting may not have ended it
    while released && kill -0 $node 2> /dev/null; do "${{RAY[@]}}" stop --force > /dev/null 2>&1 || true; sleep {poll_s}; done ) &
  watcher=$!
  wait $node || true
  kill $watcher 2> /dev/null || true
  wait $watcher 2> /dev/null || true
  "${{RAY[@]}}" stop --force > /dev/null 2>&1 || true
  [ -e {marker} ] || [ -e "$RELEASED" ] || {{ echo "this Ray node stopped (its head is gone or not up yet): joining again in {rejoin_s} s"; sleep {rejoin_s}; }}
done
if [ -e {marker} ]; then echo "the launch has ended ({marker})"; else echo "released by the launch's driver ($RELEASED)"; fi
"""
RAY_K8S_KEYS = {"worker_gpus", "head_cpu", "head_memory", "head_cpu_limit", "head_memory_limit", "head_tolerations",
                "head_restarts", "worker_restarts", "gcs_port", "object_store_gb", "pythonpath", "lock_wait_s",
                "lock_tries", "rejoin_s", "release_poll_s"}
"""``backend.k8s.ray``, all optional: ``worker_gpus`` per worker pod (default: one station's GPUs; a multiple of them,
e.g. a whole node's), the head pod's ``head_cpu`` (2) and ``head_memory`` (8Gi) requested, ``head_cpu_limit`` and
``head_memory_limit`` (default: the requests) and ``head_tolerations`` (default: ``backend.k8s.tolerations``; a
cluster that refuses a GPU toleration on a pod without a GPU wants ``[]``), ``head_restarts`` (3) and
``worker_restarts`` per worker index (3), the head's ``gcs_port`` (6379), each node's ``object_store_gb`` (1),
``pythonpath``: where Ray is when the image lacks it (a ``pip install --no-deps --target`` directory on the volume),
how long a new head waits for a dead driver's lock (``lock_tries`` (30) times ``lock_wait_s`` (20)), and how long a
worker whose Ray node stopped waits before it joins again (``rejoin_s``, 15), and how often a worker looks for its
release (``release_poll_s``, 5)."""
RAY_ROLE = "eval-launch-ray-role"
RAY_LAUNCH = "eval-launch"
"""Pod labels: which launch a pod is part of and its role in the Ray cluster; the head's Service selects on them."""


def _scaled(quantity: Any, factor: int) -> str:
    """A Kubernetes quantity (``23``, ``500m``, ``170Gi``) times ``factor``."""
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([A-Za-z]*)", str(quantity))
    if not match:
        raise ValueError(f"backend.k8s: {quantity!r} is not a quantity this renderer can scale (e.g. 23, 500m, 170Gi)")
    return f"{float(match.group(1)) * factor:g}{match.group(2)}"


def _cpus(quantity: str) -> int:
    """Ray's CPU count for a pod's CPU quantity (its actors take none; this only sizes Ray's own worker pool)."""
    return max(1, int(float(quantity[:-1]) / 1000 if quantity.endswith("m") else float(quantity)))


def render_ray(config: Mapping[str, Any], config_path: str, launch_id: str) -> dict[str, Any]:
    """The Ray backend's cluster on Kubernetes for ``config`` (``backend.kind: ray`` with a ``backend.k8s`` block), as one
    ``List`` to apply: a Job for the head, an indexed Job of worker pods, and a headless Service the workers find the
    head by. ``launch_id`` names this rendering's end marker (``<output>/k8s/<launch_id>.finished``), so a marker an
    earlier rendering left never stops this one's workers."""
    check_ray_config(config)
    layout = parse_layout(config.get("servers"))
    backend = config["backend"]
    k8s = backend["k8s"]
    ray_k8s = dict(k8s.get("ray") or {})
    unknown = sorted(set(ray_k8s) - RAY_K8S_KEYS)
    if unknown:
        raise ValueError(f"backend.k8s.ray has {sorted(RAY_K8S_KEYS)}, not {unknown}")
    pod_labels = labels(k8s)
    missing = [key for key in ("cpu", "memory", "shm_size") if key not in k8s]
    if missing:
        raise ValueError(f"backend.k8s needs {missing}: CPU, memory and /dev/shm per GPU of a worker pod")
    gpus = gpu_count(backend["gpus"])
    worker_gpus = int(ray_k8s.get("worker_gpus", layout.gpus))
    if worker_gpus < layout.gpus or worker_gpus % layout.gpus or gpus % worker_gpus:
        raise ValueError(f"backend.k8s.ray.worker_gpus is {worker_gpus}: a worker pod holds whole stations of layout "
                         f"{layout.name} ({layout.gpus} GPUs each), and backend.gpus ({gpus}) whole worker pods")
    name = k8s["job_name"]
    output = str(config["output"])
    python = shlex.quote(str(k8s.get("python", "/isaac-sim/python.sh")))
    gcs_port = int(ray_k8s.get("gcs_port", 6379))
    object_store = int(float(ray_k8s.get("object_store_gb", 1)) * 2**30)
    marker_dir = PurePosixPath(output) / "k8s"
    marker = marker_dir / f"{launch_id}.finished"
    release_dir = marker_dir / f"{launch_id}.release"
    tree = {"tree": shlex.quote(k8s["tree_dir"]), "commit": shlex.quote(k8s["commit"]), "package_parent": PACKAGE_PARENT,
            "pythonpath": "".join(f":{shlex.quote(str(path))}" for path in ray_k8s.get("pythonpath", [])),
            "python": python, "gcs_port": gcs_port, "object_store": object_store, "marker": shlex.quote(str(marker))}
    head_script = RAY_HEAD_SCRIPT.format(**tree, config=shlex.quote(config_path), marker_dir=shlex.quote(str(marker_dir)),
                                         release_dir=shlex.quote(str(release_dir)),
                                         driver_running=EXIT_DRIVER_RUNNING, ended_with_failures=EXIT_ENDED_WITH_FAILURES,
                                         lock_tries=int(ray_k8s.get("lock_tries", 30)),
                                         lock_wait_s=int(ray_k8s.get("lock_wait_s", 20)))
    poll_s = float(ray_k8s.get("release_poll_s", 5))
    if not 0.1 <= poll_s <= 60:
        raise ValueError(f"backend.k8s.ray.release_poll_s is {poll_s}: seconds between a worker's looks for its release, "
                         "0.1 to 60")
    worker_cpu = _scaled(k8s["cpu"], worker_gpus)
    worker_script = RAY_WORKER_SCRIPT.format(**tree, head=f"{name}-head", gpus=worker_gpus, cpus=_cpus(worker_cpu),
                                             rejoin_s=f"{float(ray_k8s.get('rejoin_s', 15)):g}",
                                             release_dir=shlex.quote(str(release_dir)), pod_resource=POD_RESOURCE,
                                             join_resource=JOIN_RESOURCE,
                                             poll_s=f"{poll_s:g}")
    deadline = int(float(k8s.get("deadline_h", 24)) * 3600)

    def role_labels(role: str) -> dict[str, str]:
        return {**pod_labels, RAY_LAUNCH: name, RAY_ROLE: role}

    def pod(role: str, script: str, requests: dict[str, str], limits: dict[str, str], tolerations: list, shm: str,
            secrets: list, grace_s: int) -> dict[str, Any]:
        environment = [{"name": key, "value": value} for key, value in _environment(k8s).items()]
        environment.append({"name": "POD_IP", "valueFrom": {"fieldRef": {"fieldPath": "status.podIP"}}})
        environment.append({"name": "POD_NAME", "valueFrom": {"fieldRef": {"fieldPath": "metadata.name"}}})
        pod_context, container_context = pod_security(k8s)
        return {"metadata": {"labels": role_labels(role), "annotations": annotations(k8s)},
                "spec": {
                    "restartPolicy": "Never",
                    "securityContext": pod_context,
                    "terminationGracePeriodSeconds": grace_s,
                    "nodeSelector": dict(k8s.get("node_selector", {})),
                    "tolerations": list(tolerations),
                    "containers": [{
                        "name": role, "image": backend["image"], "command": ["/bin/bash", "-c", script],
                        "securityContext": container_context,
                        "env": environment + secrets,
                        "resources": {"requests": requests, "limits": limits},
                        "volumeMounts": [{"name": "data", "mountPath": k8s["volume_mount"]},
                                         {"name": "shm", "mountPath": "/dev/shm"}]}],
                    "volumes": [{"name": "data", "persistentVolumeClaim": {"claimName": k8s["volume_claim"]}},
                                {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": shm}}]}}

    head_requests = {"cpu": str(ray_k8s.get("head_cpu", 2)), "memory": str(ray_k8s.get("head_memory", "8Gi"))}
    head_limits = {"cpu": str(ray_k8s.get("head_cpu_limit", head_requests["cpu"])),
                   "memory": str(ray_k8s.get("head_memory_limit", head_requests["memory"]))}
    head_pod = pod("head", head_script, head_requests, head_limits,
                   ray_k8s.get("head_tolerations", k8s.get("tolerations", [])),
                   f"{float(ray_k8s.get('object_store_gb', 1)) + 1:g}Gi", [],
                   int(k8s.get("termination_grace_s", 180)))
    worker_resources = {"nvidia.com/gpu": str(worker_gpus), "cpu": worker_cpu, "memory": _scaled(k8s["memory"], worker_gpus)}
    worker_pod = pod("worker", worker_script, worker_resources, worker_resources, k8s.get("tolerations", []),
                     _scaled(k8s["shm_size"], worker_gpus), secret_environment(k8s), 30)
    worker_pod["spec"]["affinity"] = {"podAffinity": {"preferredDuringSchedulingIgnoredDuringExecution": [{
        "weight": 100, "podAffinityTerm": {"labelSelector": {"matchLabels": {RAY_LAUNCH: name, RAY_ROLE: "worker"}},
                                           "topologyKey": "kubernetes.io/hostname"}}]}}
    metadata = {"namespace": k8s["namespace"], "labels": {**pod_labels, RAY_LAUNCH: name}}
    service = {"apiVersion": "v1", "kind": "Service", "metadata": {"name": f"{name}-head", **metadata},
               "spec": {"clusterIP": "None", "publishNotReadyAddresses": True,
                        "selector": {RAY_LAUNCH: name, RAY_ROLE: "head"},
                        "ports": [{"name": "gcs", "port": gcs_port}]}}
    job_metadata = {**metadata, "annotations": annotations(k8s)}
    head = {"apiVersion": "batch/v1", "kind": "Job", "metadata": {"name": f"{name}-head", **job_metadata},
            "spec": {"completions": 1, "parallelism": 1, "backoffLimit": int(ray_k8s.get("head_restarts", 3)),
                     "activeDeadlineSeconds": deadline, "template": head_pod}}
    workers = gpus // worker_gpus
    worker_job = {"apiVersion": "batch/v1", "kind": "Job", "metadata": {"name": f"{name}-workers", **job_metadata},
                  "spec": {"completionMode": "Indexed", "completions": workers, "parallelism": workers,
                           "backoffLimitPerIndex": int(ray_k8s.get("worker_restarts", 3)),
                           "podReplacementPolicy": "Failed",
                           "activeDeadlineSeconds": deadline, "template": worker_pod}}
    return {"apiVersion": "v1", "kind": "List", "items": [service, head, worker_job]}
