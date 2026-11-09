# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""The ``k8s`` backend: an indexed Job of one-GPU pods, each one ``eval_launch.eval_worker`` on the launch's queue.

The Ray backend's cluster on the same Kubernetes cluster (a head pod and worker pods) is rendered by :mod:`eval_launch.ray_k8s`, which
shares this module's staged-tree script, labels and secret handling.

One GPU per pod: several Kit processes in one multi-GPU pod all rendered on GPU 0 (2026-10-23). Pod 0 also runs
``python -m eval_launch --start`` first, which writes the plan and fills the queue; every worker waits for the full queue.
A pod whose worker finds nothing left to claim exits, so GPUs free up as the tail drains; one (the sentinel) stays
until every unit is done or failed. Each index has its own retry budget (``backoffLimitPerIndex``): a pod lost to
the node is started again under the same worker name, and one index failing never stops the others. Everything
specific to a cluster (namespace, image, node selector, tolerations, the shared volume, the staged tree) comes from
the config's ``backend.k8s`` block (or its site profile, :mod:`eval_launch.site`); nothing here names a cluster. A
cluster's own label rule (an accounting label every workload must carry) is ``backend.k8s.required_labels``: a config
whose ``backend.k8s.labels`` lacks one of them, or gives it another value, is refused before any file is written.

``render_prepull`` gives a DaemonSet that pulls the image on every eligible node before a big launch (a cold pull
from another region took 15-20 min per node on 2026-11-02): apply it, wait until every pod is ready, delete it.
"""

from __future__ import annotations

import shlex
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any


PACKAGE_PARENT = "scripts/evaluation"
"""Where the ``eval_launch`` package sits in the tree; the pods put it on the path and run its modules."""
TREE = """R={tree}
test "$(cat $R/.source-commit)" = {commit} || {{ echo "tree $R is $(cat $R/.source-commit), the launch wants {commit}"; exit 2; }}
cd $R
L=$R/submodule/isaaclab_arena/submodules/IsaacLab/source
export PYTHONPATH=$R/{package_parent}:$R/source/dexbench_arena:$R/submodule/isaaclab_arena:$(ls -d $L/*/ | sed 's:/$::' | paste -sd:){pythonpath}
export ISAACLAB_PATH=$R/submodule/isaaclab_arena/submodules/IsaacLab ISAACLAB_ARENA_PATH=$R/submodule/isaaclab_arena
export DEXBENCH_ARENA_PATH=$R DEXBENCH_IMAGE_REVISION={commit} OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
"""
"""Every pod's start: the staged tree is the launch's commit (else the pod stops before any work), and the paths and
settings the launcher, the cell runner and Kit read."""
SCRIPT = """set -euo pipefail
trap 'exit 143' TERM  # PID 1 ignores a signal it has no trap for: a pod deleted while it starts must stop
# The pod's environment, for the policy server, before anything Isaac Sim's is added. It holds the secret_env values:
# owner-only, and the worker removes it once read (server_env.read); a pod that ends before the worker starts (a TERM
# while pod 0 plans, a refused tree) removes it on its way out.
E=${{TMPDIR:-/tmp}}/server.env
trap 'rm -f "$E"' EXIT
(umask 077 && env -0 > "$E")
""" + TREE + """if [ "$JOB_COMPLETION_INDEX" = 0 ]; then
  # In the background, waited for: bash runs a trap only once a foreground command returns, so a pod deleted while it
  # plans would have planned on until it was killed.
  {package_parent}/eval_launch/signal_group.sh {python} -m eval_launch {config} --start &
  starting=$!
  trap 'kill -TERM $starting 2>/dev/null; wait $starting; exit 143' TERM
  wait $starting
  trap 'exit 143' TERM
fi
# not exec {python}: Isaac Sim's python.sh would be PID 1, which ignores SIGTERM and passes nothing to the worker
exec {package_parent}/eval_launch/signal_group.sh {python} -m eval_launch.eval_worker --launch {output} --worker w$JOB_COMPLETION_INDEX --port {port} \\
  --server-env "$E"
"""


def labels(k8s: Mapping[str, Any]) -> dict[str, str]:
    """``backend.k8s.labels``, on every Job, DaemonSet and pod template the launcher makes; refused unless it holds
    every label of ``backend.k8s.required_labels`` with its value. The rule is the cluster's (its accounting reads
    the labels, e.g. a team's project label), so it lives in the site profile with the labels; the launcher checks,
    never adds, and requires none by default."""
    found = {str(key): str(value) for key, value in (k8s.get("labels") or {}).items()}
    required = {str(key): str(value) for key, value in (k8s.get("required_labels") or {}).items()}
    missing = {key: value for key, value in required.items() if found.get(key) != value}
    if missing:
        raise ValueError(f"backend.k8s.labels needs {missing}: backend.k8s.required_labels says every Job and pod the "
                         "launcher makes carries it")
    return found


def pod_security(k8s: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """``backend.k8s.pod_security_context`` and ``security_context`` (the container's), as the cluster wants them, e.g.
    ``{allowPrivilegeEscalation: false}`` where admission requires it; empty by default. Not ``capabilities: {drop:
    [ALL]}``: the image runs as root, and root without capabilities cannot read ``/isaac-sim``, so Kit does not start."""
    return dict(k8s.get("pod_security_context") or {}), dict(k8s.get("security_context") or {})


def annotations(k8s: Mapping[str, Any]) -> dict[str, str]:
    """``backend.k8s.annotations``: on every Job and pod template the launcher makes (e.g. a cluster's job-provenance
    keys); none by default."""
    return {str(key): str(value) for key, value in (k8s.get("annotations") or {}).items()}


def render_job(config: Mapping[str, Any], config_path: str) -> dict[str, Any]:
    """The Job manifest for ``config`` (``config_path``: where the pods read the config, on the shared volume)."""
    backend = config["backend"]
    k8s = backend["k8s"]
    job_labels = labels(k8s)
    gpus = int(backend["gpus"])
    output = str(config["output"])
    pythonpath = "".join(f":{shlex.quote(str(path))}" for path in backend.get("pythonpath", []))
    script = SCRIPT.format(tree=shlex.quote(k8s["tree_dir"]), commit=shlex.quote(k8s["commit"]), package_parent=PACKAGE_PARENT,
                           config=shlex.quote(config_path), output=shlex.quote(output), port=int(k8s.get("port", 5810)),
                           pythonpath=pythonpath, python=shlex.quote(str(k8s.get("python", "/isaac-sim/python.sh"))))
    missing = [key for key in ("cpu", "memory", "shm_size") if key not in k8s]
    if missing:
        raise ValueError(f"backend.k8s needs {missing}: each worker pod's CPU, memory and /dev/shm size")
    resources = {"nvidia.com/gpu": "1", "cpu": str(k8s["cpu"]), "memory": str(k8s["memory"])}
    environment = _environment(k8s)
    pod_context, container_context = pod_security(k8s)
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": k8s["job_name"], "namespace": k8s["namespace"], "labels": job_labels,
                     "annotations": annotations(k8s)},
        "spec": {
            "completionMode": "Indexed",
            "completions": gpus,
            "parallelism": gpus,
            "backoffLimitPerIndex": int(k8s.get("retries_per_worker", 2)),
            "activeDeadlineSeconds": int(float(k8s.get("deadline_h", 24)) * 3600),
            "template": {
                "metadata": {"labels": dict(job_labels), "annotations": annotations(k8s)},
                "spec": {
                    "restartPolicy": "Never",
                    "securityContext": pod_context,
                    "terminationGracePeriodSeconds": int(k8s.get("termination_grace_s", 120)),
                    "nodeSelector": dict(k8s.get("node_selector", {})),
                    "tolerations": list(k8s.get("tolerations", [])),
                    "containers": [{
                        "name": "worker",
                        "image": backend["image"],
                        "command": ["/bin/bash", "-c", script],
                        "securityContext": container_context,
                        "env": [{"name": name, "value": value} for name, value in environment.items()]
                        + secret_environment(k8s),
                        "resources": {"requests": resources, "limits": resources},
                        "volumeMounts": [{"name": "data", "mountPath": k8s["volume_mount"]},
                                         {"name": "shm", "mountPath": "/dev/shm"}],
                    }],
                    "volumes": [{"name": "data", "persistentVolumeClaim": {"claimName": k8s["volume_claim"]}},
                                {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": str(k8s["shm_size"])}}],
                },
            },
        },
    }


def secret_environment(k8s: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The pods' variables taken from Kubernetes secrets: ``backend.k8s.secret_env`` maps a variable to
    ``{secret: <name>, key: <key>}``, e.g. ``HF_TOKEN: {secret: hf-token, key: token}`` for gated Hub models. No config,
    Job file or log holds the value. The policy servers get it with the rest of the pod's environment, which passes
    through the container's own ``$TMPDIR/server.env`` (``/tmp`` in a pod; owner-only, removed by the worker once read,
    or by the script on its way out; not on the shared volume)."""
    entries = []
    for name, ref in (k8s.get("secret_env") or {}).items():
        if not isinstance(ref, Mapping) or not ref.get("secret") or not ref.get("key"):
            raise ValueError(f"backend.k8s.secret_env.{name} needs a secret and a key, e.g. {{secret: hf-token, key: token}}")
        entries.append({"name": str(name),
                        "valueFrom": {"secretKeyRef": {"name": str(ref["secret"]), "key": str(ref["key"])}}})
    return entries


def config_on_volume(config: Mapping[str, Any]) -> str:
    """Where pods read the launch's config: ``<output>/eval.yaml``, written by ``--k8s-job``."""
    return str(PurePosixPath(str(config["output"])) / "eval.yaml")


def render_prepull(config: Mapping[str, Any]) -> dict[str, Any]:
    """A DaemonSet that pulls the launch's image onto every node the Job may use and then idles. Its pods carry the
    cluster's annotations and security contexts as the Job's do, so a policy that requires them admits these too."""
    backend, k8s = config["backend"], config["backend"]["k8s"]
    name = f"{k8s['job_name']}-prepull"
    prepull_labels = {**labels(k8s), "app": name}
    pod_context, container_context = pod_security(k8s)
    return {
        "apiVersion": "apps/v1",
        "kind": "DaemonSet",
        "metadata": {"name": name, "namespace": k8s["namespace"], "labels": prepull_labels,
                     "annotations": annotations(k8s)},
        "spec": {
            "selector": {"matchLabels": {"app": name}},
            "template": {
                "metadata": {"labels": dict(prepull_labels), "annotations": annotations(k8s)},
                "spec": {
                    "securityContext": pod_context,
                    "nodeSelector": dict(k8s.get("node_selector", {})),
                    "tolerations": list(k8s.get("tolerations", [])),
                    "initContainers": [{"name": "pull", "image": backend["image"], "command": ["/bin/true"],
                                        "securityContext": dict(container_context),
                                        "resources": {"requests": {"cpu": "100m", "memory": "64Mi"}}}],
                    "containers": [{"name": "idle", "image": backend["image"], "command": ["/bin/sleep", "infinity"],
                                    "securityContext": dict(container_context),
                                    "resources": {"requests": {"cpu": "10m", "memory": "16Mi"}}}],
                },
            },
        },
    }



def _environment(k8s: Mapping[str, Any]) -> dict[str, str]:
    """Isaac Sim's own (EULA, privacy, root), then the cluster's (e.g. a scheduler's required variables)."""
    return {"ACCEPT_EULA": "Y", "PRIVACY_CONSENT": "Y", "OMNI_KIT_ALLOW_ROOT": "1", "NVIDIA_DRIVER_CAPABILITIES": "all",
            **{str(k): str(v) for k, v in (k8s.get("env") or {}).items()}}
