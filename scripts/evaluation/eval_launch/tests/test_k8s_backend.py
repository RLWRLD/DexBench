# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""The k8s backend's Job: one GPU per pod, as many pods as the config's GPUs, the staged tree checked before work."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from dexbench_arena import REPO_ROOT
from dexbench_arena.embodiments.rby1_wuji2.constants import RBY1_WUJI2_JOINT_ACTION_NAMES

from eval_launch import PACKAGE_DIR, k8s_backend, launch, ray_k8s
from eval_launch.plan import RECIPE_FIELDS, check_recipe, load_config
from eval_launch.tests.example_cluster import filled

EXAMPLE = PACKAGE_DIR / "examples/kubernetes.yaml"


def test_the_job_runs_one_worker_per_gpu_on_the_staged_tree():
    config = filled(load_config(EXAMPLE))
    job = k8s_backend.render_job(config, k8s_backend.config_on_volume(config))
    spec = job["spec"]
    assert spec["completionMode"] == "Indexed" and spec["completions"] == spec["parallelism"] == config["backend"]["gpus"]
    assert spec["backoffLimitPerIndex"] >= 1 and "backoffLimit" not in spec
    (container,) = spec["template"]["spec"]["containers"]
    assert container["resources"]["limits"]["nvidia.com/gpu"] == "1"
    script = container["command"][-1]
    assert f'= {config["backend"]["k8s"]["commit"]}' in script, "the pod refuses a tree of another commit"
    assert "-m eval_launch " in script and "--start" in script and "-m eval_launch.eval_worker --launch" in script
    assert yaml.safe_load(yaml.safe_dump(job)) == job


def test_every_job_and_pod_carries_the_required_labels_and_a_config_without_them_writes_nothing(tmp_path):
    """A cluster's label rule (e.g. a project label) on the Job, its pods and the pre-pull."""
    config = filled(load_config(EXAMPLE))
    required = {"cost-center": "evals"}
    config["backend"]["k8s"].update(labels={**required, "owner": "me"}, required_labels=required)
    job, prepull = k8s_backend.render_job(config, "/x/eval.yaml"), k8s_backend.render_prepull(config)
    for labels in (job["metadata"]["labels"], job["spec"]["template"]["metadata"]["labels"],
                   prepull["metadata"]["labels"], prepull["spec"]["template"]["metadata"]["labels"]):
        assert required.items() <= labels.items()
    path = tmp_path / "eval.yaml"
    for value in ("another-team", None):
        if value is None:
            del config["backend"]["k8s"]["labels"]["cost-center"]
        else:
            config["backend"]["k8s"]["labels"]["cost-center"] = value
        path.write_text(yaml.safe_dump({**config, "output": str(tmp_path / "launch")}))
        with pytest.raises(SystemExit):
            launch.main([str(path), "--k8s-job", str(tmp_path / "job")])
        assert not (tmp_path / "job").exists()
    del config["backend"]["k8s"]["required_labels"]
    assert "cost-center" not in k8s_backend.render_job(config, "/x/eval.yaml")["metadata"]["labels"]


@pytest.mark.parametrize("example", sorted(EXAMPLE.parent.glob("*.yaml")), ids=lambda path: path.stem)
def test_every_example_recipe_formats_with_only_the_worker_fields(example):
    """The fields eval_worker fills in; a recipe naming another would fail at its first server start. An example
    without a recipe says it is only planned, and the launcher refuses to start it."""
    config = load_config(example)
    if "plan-only" in example.read_text().splitlines()[0]:
        with pytest.raises(ValueError, match="can only be planned"):
            check_recipe(config)
        return
    check_recipe(config)
    recipe = config["servers"]["recipe"]
    fields = {"checkpoint": "/c", "checkpoint_id": "abc", "port": 5810, "token_file": "/t", "worker_dir": "/w",
              "launcher": "/l"}
    assert set(fields) == set(RECIPE_FIELDS)
    assert "5810" in recipe["command"].format(**fields)
    assert all("{" not in item.format(**fields) for item in recipe.get("client_set", []))


def test_a_recipe_brace_meant_for_the_shell_is_refused_unless_doubled():
    """${HOME} was a KeyError at every worker's first server start."""
    config = filled(load_config(EXAMPLE))
    config["servers"]["recipe"]["command"] = "cd ${HOME} && exec serve --port {port}"
    with pytest.raises(ValueError, match="doubled"):
        check_recipe(config)
    config["servers"]["recipe"]["command"] = "cd ${{HOME}} && exec serve --port {port}"
    check_recipe(config)


def test_the_model_client_path_and_the_prepull_come_from_the_config():
    config = filled(load_config(EXAMPLE))
    config["backend"]["pythonpath"] = ["/opt/model_client"]
    script = k8s_backend.render_job(config, "/x/eval.yaml")["spec"]["template"]["spec"]["containers"][0]["command"][-1]
    assert ":/opt/model_client" in next(line for line in script.splitlines() if line.startswith("export PYTHONPATH"))
    prepull = k8s_backend.render_prepull(config)
    assert prepull["kind"] == "DaemonSet"
    assert prepull["spec"]["template"]["spec"]["initContainers"][0]["image"] == config["backend"]["image"]
    assert prepull["spec"]["template"]["spec"]["nodeSelector"] == config["backend"]["k8s"]["node_selector"]


def test_a_config_without_prepull_renders_no_daemonset(tmp_path):
    config = filled(load_config(EXAMPLE))
    config["backend"]["k8s"]["prepull"] = False
    path = tmp_path / "eval.yaml"
    path.write_text(yaml.safe_dump({**config, "output": str(tmp_path / "launch")}))
    assert launch.main([str(path), "--k8s-job", str(tmp_path / "job")]) == 0
    assert (tmp_path / "job" / "job.json").exists() and not (tmp_path / "job" / "prepull.json").exists()


def test_a_sigterm_reaches_the_python_behind_a_wrapper_and_the_wrapper_waits_for_it(tmp_path):
    wrapper = tmp_path / "python.sh"
    wrapper.write_text(f'#!/bin/bash\n{sys.executable} "$@" || exit 1\n')
    wrapper.chmod(0o755)
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import signal, sys, time\n"
        "def stop(signum, frame):\n"
        "    time.sleep(1)  # handing the unit back\n"
        "    open(sys.argv[1], 'w').write('released')\n"
        "    sys.exit(128 + signum)\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        "open(sys.argv[1] + '.started', 'w').close()\n"
        "time.sleep(60)\n")
    released = tmp_path / "released"
    process = subprocess.Popen([str(PACKAGE_DIR / "signal_group.sh"), str(wrapper), str(worker),
                                str(released)])
    deadline = time.time() + 30
    while not Path(f"{released}.started").exists() and time.time() < deadline:
        time.sleep(0.1)
    process.send_signal(signal.SIGTERM)
    assert process.wait(timeout=30) != 0
    assert released.read_text() == "released"


def test_the_example_servers_joint_names_are_the_robots_action_joints():
    names = json.loads((REPO_ROOT / "source/dexbench_policy_server/dexbench_policy_server/rby1_wuji2_joint_names.json")
                       .read_text())
    assert names == list(RBY1_WUJI2_JOINT_ACTION_NAMES)


def test_a_secret_reaches_the_pods_by_reference_only():
    config = filled(load_config(EXAMPLE))
    config["backend"]["k8s"]["secret_env"] = {"HF_TOKEN": {"secret": "hf-token", "key": "token"}}
    job = k8s_backend.render_job(config, "/x/eval.yaml")
    env = job["spec"]["template"]["spec"]["containers"][0]["env"]
    assert {"name": "HF_TOKEN", "valueFrom": {"secretKeyRef": {"name": "hf-token", "key": "token"}}} in env
    config["backend"]["k8s"]["secret_env"] = {"HF_TOKEN": "hf-token"}
    with pytest.raises(ValueError, match="needs a secret and a key"):
        k8s_backend.render_job(config, "/x/eval.yaml")


RAY_EXAMPLE = PACKAGE_DIR / "examples/ray_kubernetes.yaml"


def _ray_config(**ray_k8s):
    config = filled(load_config(RAY_EXAMPLE))
    config["backend"]["k8s"]["job_name"] = "me-evl-ray"
    config["backend"]["k8s"]["ray"].update(ray_k8s)
    return config


def test_a_ray_launch_is_a_cpu_head_that_runs_the_driver_and_workers_of_whole_gpus():
    config = _ray_config(worker_gpus=8)
    config["backend"]["k8s"]["ray"].pop("head_tolerations", None)
    config["backend"]["gpus"] = 16
    config["backend"]["k8s"].update(labels={"team": "a", "owner": "me"}, required_labels={"team": "a"})
    config["backend"]["k8s"]["secret_env"] = {"HF_TOKEN": {"secret": "hf-token", "key": "token"}}
    service, head, workers = ray_k8s.render_ray(config, "/x/eval.yaml", "me-1")["items"]
    k8s = config["backend"]["k8s"]

    head_pod, worker_pod = head["spec"]["template"], workers["spec"]["template"]
    (head_container,), (worker_container,) = head_pod["spec"]["containers"], worker_pod["spec"]["containers"]
    assert "nvidia.com/gpu" not in head_container["resources"]["requests"]
    assert head_pod["spec"]["nodeSelector"] == k8s["node_selector"]
    assert k8s["tolerations"] and head_pod["spec"]["tolerations"] == k8s["tolerations"]
    script = head_container["command"][-1]
    assert f'= {k8s["commit"]}' in script, "the head refuses a tree of another commit"
    assert "start --head" in script and "-m eval_launch.ray_backend /x/eval.yaml --ray-address auto" in script
    assert head["spec"]["backoffLimit"] >= 1, "a dead head is replaced, and its driver recovers the launch"
    assert workers["spec"]["completionMode"] == "Indexed" and workers["spec"]["completions"] == 2
    assert worker_container["resources"]["limits"]["nvidia.com/gpu"] == "8"
    assert "--num-gpus=8" in worker_container["command"][-1]
    assert f"--address={service['metadata']['name']}:6379" in worker_container["command"][-1]
    assert service["spec"]["clusterIP"] == "None"
    assert service["spec"]["selector"].items() <= head_pod["metadata"]["labels"].items()
    assert not service["spec"]["selector"].items() <= worker_pod["metadata"]["labels"].items()
    assert {"name": "HF_TOKEN", "valueFrom": {"secretKeyRef": {"name": "hf-token", "key": "token"}}} in worker_container["env"]
    assert not any(entry["name"] == "HF_TOKEN" for entry in head_container["env"])
    for item in (service, head, workers, head_pod, worker_pod):
        assert k8s["required_labels"].items() <= item["metadata"]["labels"].items()


def test_the_head_takes_its_own_tolerations_and_limits_and_the_workers_keep_the_gpu_ones():
    config = _ray_config(worker_gpus=1)
    config["backend"]["k8s"]["ray"].update(head_tolerations=[], head_cpu="100m", head_memory="512Mi",
                                           head_cpu_limit="500m", head_memory_limit="4Gi")
    _, head, workers = ray_k8s.render_ray(config, "/x/eval.yaml", "me-1")["items"]
    head_pod, worker_pod = head["spec"]["template"]["spec"], workers["spec"]["template"]["spec"]
    assert head_pod["tolerations"] == []
    assert head_pod["containers"][0]["resources"] == {"requests": {"cpu": "100m", "memory": "512Mi"},
                                                      "limits": {"cpu": "500m", "memory": "4Gi"}}
    assert worker_pod["tolerations"] == config["backend"]["k8s"]["tolerations"]
    resources = worker_pod["containers"][0]["resources"]
    assert resources["requests"] == resources["limits"] and resources["limits"]["nvidia.com/gpu"] == "1"


def test_the_clusters_annotations_and_security_contexts_reach_every_job_and_pod():
    k8s_extra = {"annotations": {"bundle.example/job-kind": "eval"},
                 "security_context": {"allowPrivilegeEscalation": False},
                 "pod_security_context": {"seccompProfile": {"type": "RuntimeDefault"}}}
    ray_config = _ray_config(worker_gpus=1)
    ray_config["backend"]["k8s"].update(k8s_extra)
    _, head, workers = ray_k8s.render_ray(ray_config, "/x/eval.yaml", "me-1")["items"]
    job_config = filled(load_config(EXAMPLE))
    job_config["backend"]["k8s"].update(k8s_extra)
    job = k8s_backend.render_job(job_config, "/x/eval.yaml")
    prepull = k8s_backend.render_prepull(job_config)
    for item in (head, workers, job, prepull):
        template = item["spec"]["template"]
        assert item["metadata"]["annotations"] == template["metadata"]["annotations"] == k8s_extra["annotations"]
        assert template["spec"]["securityContext"] == k8s_extra["pod_security_context"]
        for container in template["spec"]["containers"] + template["spec"].get("initContainers", []):
            assert container["securityContext"] == k8s_extra["security_context"]


def test_a_worker_pod_holds_whole_stations_and_a_config_that_splits_one_writes_nothing(tmp_path):
    config = _ray_config(worker_gpus=3)
    config["servers"]["layout"] = "server+1"
    with pytest.raises(ValueError, match="whole stations"):
        ray_k8s.render_ray(config, "/x/eval.yaml", "me-1")
    config["backend"]["k8s"]["ray"]["worker_gpus"] = 4
    config["backend"]["gpus"] = 8
    (_, _, workers) = ray_k8s.render_ray(config, "/x/eval.yaml", "me-1")["items"]
    assert workers["spec"]["completions"] == 2
    config["servers"]["layout"] = {"sim_gpus": 3, "sims_per_gpu": 3}
    config["servers"]["gpu_memory_gb"], config["backend"]["gpu_memory_gb"] = {"server": 20, "sim": 10}, 46
    del config["backend"]["k8s"]["ray"]["worker_gpus"]
    (_, _, workers) = ray_k8s.render_ray(config, "/x/eval.yaml", "me-1")["items"]
    assert workers["spec"]["completions"] == 2
    assert workers["spec"]["template"]["spec"]["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] == "4"
    config["servers"]["layout"] = "server+1"
    config["backend"]["k8s"]["ray"]["worker_gpus"] = 3
    path = tmp_path / "eval.yaml"
    path.write_text(yaml.safe_dump({**config, "output": str(tmp_path / "launch")}))
    with pytest.raises(SystemExit):
        launch.main([str(path), "--k8s-job", str(tmp_path / "job")])
    assert not (tmp_path / "job").exists()
    config["backend"]["k8s"]["ray"]["worker_gpus"] = 4
    path.write_text(yaml.safe_dump({**config, "output": str(tmp_path / "launch")}))
    assert launch.main([str(path), "--k8s-job", str(tmp_path / "job")]) == 0
    assert json.loads((tmp_path / "job" / "ray.json").read_text())["kind"] == "List"


def test_a_worker_pod_that_ends_before_its_worker_leaves_no_saved_environment(tmp_path):
    """The pod saves its environment (secret_env values included) for the worker, which removes it once read. A pod
    that ends before that (here its tree is another commit's; a TERM while pod 0 plans is the same) removes it."""
    config = filled(load_config(EXAMPLE))
    tree, python, _, _ = _fake_tree(tmp_path, "0" * 40)
    config["backend"]["k8s"].update(tree_dir=str(tree), python=str(python))
    pod = _run_pod_script(k8s_backend.render_job(config, "/x/eval.yaml"), tmp_path)
    assert pod.wait(timeout=30) == 2, pod.stdout.read()
    assert not list(tmp_path.glob("server.env*")), "the servers' environment (it holds the pod's secrets) is removed"


def _fake_tree(tmp_path, commit):
    """A staged tree as the pod scripts read it, and a ``python`` that stands in for Ray's CLI and the driver: it logs
    each call, a node's ``ray start --block`` ends after a moment (its head went away; ``node_s`` in the tree's
    parent sets how long) or when ``ray stop`` stops the node of that pod, and the driver exits with the codes in
    ``driver_codes`` in turn (one line each), then 0."""
    tree = tmp_path / "tree"
    (tree / "submodule/isaaclab_arena/submodules/IsaacLab/source/isaaclab").mkdir(parents=True)
    (tree / ".source-commit").write_text(commit + "\n")
    package = tree / k8s_backend.PACKAGE_PARENT / "eval_launch"
    package.mkdir(parents=True)
    (package / "signal_group.sh").write_bytes((PACKAGE_DIR / "signal_group.sh").read_bytes())
    (package / "signal_group.sh").chmod(0o755)
    log, codes = tmp_path / "calls.log", tmp_path / "driver_codes"
    python = tmp_path / "python.sh"
    python.write_text(f"""#!/bin/bash
echo "$*" >> {log}
case "$*" in
  *"ray.scripts.scripts start"*--block*)
    sleep "$(cat {tmp_path}/node_s 2>/dev/null || echo 0.5)" & echo $! >> {tmp_path}/nodes.$POD_NAME
    wait; exit 1 ;;
  *"ray.scripts.scripts stop"*)
    kill $(cat {tmp_path}/nodes.$POD_NAME 2>/dev/null) 2>/dev/null; : > {tmp_path}/nodes.$POD_NAME; exit 0 ;;
  *ray.scripts.scripts*) exit 0 ;;
  *eval_launch.ray_backend*)
    test -s "$(echo "$*" | sed 's/.*--server-env //; s/ .*//')" || exit 9   # the driver's copy of the servers' environment
    case "$*" in *"--release-dir "*.release*) ;; *) exit 8 ;; esac   # where it names the pods it releases
    code=$(head -n1 {codes} 2>/dev/null); sed -i 1d {codes} 2>/dev/null; exit ${{code:-0}} ;;
esac
""")
    python.chmod(0o755)
    return tree, python, log, codes


def _run_pod_script(item, tmp_path, index="0", name="abcde"):
    script = item["spec"]["template"]["spec"]["containers"][0]["command"][-1]
    environment = {"PATH": os.environ["PATH"], "POD_IP": "127.0.0.1", "TMPDIR": str(tmp_path), "SECRET_FOR_SERVERS": "s",
                   "JOB_COMPLETION_INDEX": index, "POD_NAME": f"me-evl-ray-workers-{index}-{name}"}
    return subprocess.Popen(["/bin/bash", "-c", script], env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True)


def test_the_head_waits_out_a_dead_drivers_lock_and_its_end_lets_the_workers_go(tmp_path):
    """A new head pod finds the lock a killed driver held on the shared volume (exit 75) and starts the driver again;
    a driver that crashes (exit 1, as any uncaught error) fails the pod and lets no worker go (2026-11-05 on a cluster:
    a plan the driver could not summarise ended the launch); the driver's end writes the marker, and a worker node,
    which rejoins its head each time Ray stops, then exits."""
    config = _ray_config(lock_wait_s=0, rejoin_s=0.2, worker_gpus=8)
    config["backend"]["gpus"] = 8
    tree, python, log, codes = _fake_tree(tmp_path, config["backend"]["k8s"]["commit"])
    config["backend"]["k8s"].update(tree_dir=str(tree), python=str(python))
    config["output"] = str(tmp_path / "launch")
    _, head, workers = ray_k8s.render_ray(config, str(tmp_path / "eval.yaml"), "me-1")["items"]
    codes.write_text("75\n75\n1\n3\n")

    worker = _run_pod_script(workers, tmp_path)
    time.sleep(2)
    assert worker.poll() is None, "a worker waits for its head as long as the launch runs"
    head_process = _run_pod_script(head, tmp_path, "")
    assert head_process.wait(timeout=60) == 1, head_process.stdout.read()
    assert not (tmp_path / "launch/k8s/me-1.finished").exists(), "a crashed driver ends nothing: the workers stay"
    time.sleep(1)
    assert worker.poll() is None
    head_process = _run_pod_script(head, tmp_path, "")
    assert head_process.wait(timeout=60) == 0, head_process.stdout.read()
    assert worker.wait(timeout=60) == 0, worker.stdout.read()

    calls = log.read_text().splitlines()
    assert sum("eval_launch.ray_backend" in call for call in calls) == 4
    assert (tmp_path / "launch/k8s/me-1.finished").read_text().strip() == "3"
    assert sum("--block" in call for call in calls) >= 2, "a worker whose Ray node stopped joins again"
    assert not list(tmp_path.glob("server.env*")), "the servers' environment (it holds the pod's secrets) is removed"


def test_the_ray_scripts_run_an_interpreter_path_with_spaces_and_shell_syntax_as_one_word(tmp_path):
    """backend.k8s.python is a path of the image's, and the Ray command line is built from it: a path with a space, or
    with shell syntax in it, runs as that one interpreter (Ray started, the driver run), never split or evaluated."""
    config = _ray_config(lock_wait_s=0, worker_gpus=8)
    config["backend"]["gpus"] = 8
    tree, python, log, codes = _fake_tree(tmp_path, config["backend"]["k8s"]["commit"])
    odd = tmp_path / "isaac sim $(touch evaluated)" / "python.sh"
    odd.parent.mkdir()
    python.rename(odd)
    config["backend"]["k8s"].update(tree_dir=str(tree), python=str(odd))
    config["output"] = str(tmp_path / "launch")
    _, head, _ = ray_k8s.render_ray(config, str(tmp_path / "eval.yaml"), "me-1")["items"]
    head_process = _run_pod_script(head, tmp_path)
    assert head_process.wait(timeout=60) == 0, head_process.stdout.read()
    calls = log.read_text().splitlines()
    assert any(call.startswith("-m ray.scripts.scripts start --head") for call in calls), calls
    assert any("eval_launch.ray_backend" in call for call in calls)
    assert not list(tmp_path.rglob("evaluated")), "nothing in the path was run as a command"


def test_every_start_records_the_config_it_ran_with_its_commit_and_no_secret_value(tmp_path):
    """So we always know how a launch ran: each start writes a record of its own, once, with the resolved config, the
    source commit and (for --k8s-job) the rendered manifests; environment variables appear by name only."""
    config = filled(load_config(EXAMPLE))
    config["backend"]["env"] = {"SOME_TOKEN": "value-that-must-not-be-written"}
    config["backend"]["k8s"]["secret_env"] = {"HF_TOKEN": {"secret": "hf-token", "key": "token"}}
    config["output"] = str(tmp_path / "launch")
    path = tmp_path / "eval.yaml"
    path.write_text(yaml.safe_dump(config))
    assert launch.main([str(path), "--k8s-job", str(tmp_path / "job")]) == 0
    assert launch.main([str(path), "--start"]) == 0
    (rendered,) = sorted((tmp_path / "job" / launch.LAUNCHES).iterdir())
    (started,) = sorted((tmp_path / "launch" / launch.LAUNCHES).iterdir())
    for record, mode in ((rendered, "k8s-job"), (started, "start")):
        meta = json.loads((record / "launch.json").read_text())
        assert meta["mode"] == mode and {"revision", "revision_source"} <= set(meta["code"])
        recorded = json.loads((record / "config.json").read_text())
        assert recorded["model"] == config["model"] and recorded["servers"] == config["servers"]
        assert recorded["backend"]["env"] == ["SOME_TOKEN"] and recorded["backend"]["k8s"]["secret_env"] == ["HF_TOKEN"]
    job = json.loads((rendered / "job.json").read_text())
    env = job["spec"]["template"]["spec"]["containers"][0]["env"]
    assert "HF_TOKEN" in env and all(isinstance(name, str) for name in env)
    assert (rendered / "prepull.json").is_file()
    for file in [*(tmp_path / "job" / launch.LAUNCHES).rglob("*"), *(tmp_path / "launch" / launch.LAUNCHES).rglob("*")]:
        assert file.is_dir() or "value-that-must-not-be-written" not in file.read_text()
    before = (started / "config.json").read_bytes()
    assert launch.main([str(path), "--start"]) == 0
    assert len(list((tmp_path / "launch" / launch.LAUNCHES).iterdir())) == 2
    assert (started / "config.json").read_bytes() == before


def test_a_worker_its_driver_released_ends_and_only_that_pod_does(tmp_path):
    """Once the driver has released a worker pod (its node fenced, nothing left to start), the pod stops its Ray node,
    which runs until stopped, and exits 0, so its Indexed Job index succeeds and is not restarted, and its GPUs go
    back while the other workers run. The release names that pod alone: its Ray node advertises its pod name, and
    another pod of the same Job index (a replacement) is not stopped by it; the same pod started again ends without
    joining. It acts only on a release that names its node's current join: one naming another join of the pod (written
    for a node of it that has gone) leaves the node running. Another worker runs on until the launch ends."""
    config = _ray_config(lock_wait_s=0, rejoin_s=0.2, release_poll_s=0.2, worker_gpus=8)
    config["backend"]["gpus"] = 16
    tree, python, log, codes = _fake_tree(tmp_path, config["backend"]["k8s"]["commit"])
    config["backend"]["k8s"].update(tree_dir=str(tree), python=str(python))
    config["output"] = str(tmp_path / "launch")
    _, head, workers = ray_k8s.render_ray(config, str(tmp_path / "eval.yaml"), "me-1")["items"]
    (tmp_path / "node_s").write_text("120")
    released, other = _run_pod_script(workers, tmp_path, "0"), _run_pod_script(workers, tmp_path, "1")
    twin = _run_pod_script(workers, tmp_path, "0", name="fghij")
    time.sleep(1.5)
    assert released.poll() is None and other.poll() is None
    release_dir = tmp_path / "launch/k8s/me-1.release"
    release_dir.mkdir(parents=True)
    release = release_dir / "me_evl_ray_workers_0_abcde"
    release.write_text("join 0123456789abcdef\nnode n-gone\n")
    time.sleep(1.5)
    assert released.poll() is None, "a release of another join of the pod does not stop its current node"
    (join,) = {re.search(r'"eval_launch_join_([0-9a-f]+)"', call).group(1) for call in log.read_text().splitlines()
               if '"eval_launch_pod_me_evl_ray_workers_0_abcde"' in call}
    release.write_text(f"join {join}\nnode n-1\nfenced 2026-11-08T00:00:00Z\n")
    assert released.wait(timeout=30) == 0
    assert "released by the launch's driver" in released.stdout.read()
    time.sleep(1)
    assert other.poll() is None, "a worker that was not released stays until the launch ends"
    assert twin.poll() is None, "another pod of the same index is not the released one"
    again = _run_pod_script(workers, tmp_path, "0")
    assert again.wait(timeout=10) == 0
    calls = log.read_text().splitlines()
    starts = [call for call in calls if "--block" in call]
    assert any('"eval_launch_pod_me_evl_ray_workers_0_abcde": 1' in call for call in starts)
    assert any('"eval_launch_pod_me_evl_ray_workers_1_abcde": 1' in call for call in starts)
    for process in (other, twin):
        process.terminate()
        process.wait(timeout=30)


def test_a_copied_example_that_still_names_the_example_profile_is_refused(tmp_path, capsys):
    """A copy of an example keeps backend.site: examples/sites/kubernetes.yaml, which the package lookup finds: its
    <namespace>, <registry>/... and <rwx-claim> would reach the Job, which the API server refuses far from the config.
    --k8s-job refuses them, saying to copy the profile and point backend.site at the copy."""
    copied = tmp_path / "configs/local/eval.yaml"
    copied.parent.mkdir(parents=True)
    copied.write_bytes(EXAMPLE.read_bytes())
    with pytest.raises(SystemExit):
        launch.main([str(copied), "--k8s-job", str(tmp_path / "job")])
    error = capsys.readouterr().err
    assert "backend.k8s.namespace: <namespace>" in error and "point backend.site at the copy" in error
    assert not (tmp_path / "job").exists()
