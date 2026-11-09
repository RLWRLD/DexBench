# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""The launch's driver on both executors, with fake GPUs and stand-in simulators (``driver_stub.py``, each its own process
started by ``eval_worker.run_unit`` as a Kit would be): the local one (child processes on device ids that need not
exist) and the Ray one (a local Ray instance with ``num_gpus`` on a CPU host; skipped without Ray). Every layout puts
each simulator on one GPU of its own and its server where the layout says; each unit is accepted once through failed
attempts, slots killed mid-unit, slots that stop answering and a driver killed mid-launch; nothing depends on any
process's wall clock."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pytest
import yaml

pytest.importorskip("isaaclab_arena.evaluation.arena_experiment_result", reason="eval_worker needs IsaacLab-Arena on the path")

from eval_launch import driver, launch, local_backend  # noqa: E402
from eval_launch.acceptance import EXIT_DRIVER_RUNNING, EXIT_ENDED_WITH_FAILURES, LaunchState  # noqa: E402
from eval_launch.layout import parse_layout  # noqa: E402
from eval_launch.process_group import OWNER_VARIABLE  # noqa: E402
from eval_launch.tests import driver_stub  # noqa: E402

HOOK = "eval_launch.tests.driver_stub.setup"
CLIENT = "/opt/model-client"
"""A model's client package (``backend.pythonpath``): the simulator must import it, the recipe's setup checks it can."""
FAKE_GPUS = [str(11 + 2 * index) for index in range(4)]
"""The local executor's device ids: none exists, and each process's ``CUDA_VISIBLE_DEVICES`` must be one of them."""


@pytest.fixture(params=["local", pytest.param("ray", marks=pytest.mark.skipif(
    importlib.util.find_spec("ray") is None, reason="the Ray executor needs Ray (ray==2.58.1)"))])
def kind(request):
    return request.param


def _executor(kind, config, output):
    module = local_backend if kind == "local" else importlib.import_module("eval_launch.ray_backend")
    return (module.LocalExecutor if kind == "local" else module.RayExecutor)(config, output)


def _units(n):
    return [{"id": f"t.train.{i:03d}", "task": "t", "cell": "train", "slices": {"r": [i, i + 1]}, "checkpoint": "/a",
             "expected_s": 100 - i} for i in range(n)]


def _launch(tmp_path, monkeypatch, kind, units, stub_plan=None, layout="colocated", gpus=2, skew=False):
    """A driver launch's config and its plan.json, as ``write_plan`` would leave them. The local executor's processes
    inherit the test's environment; Ray's workers get theirs from the runtime environment."""
    plan_file = tmp_path / "stub_plan.json"
    plan_file.write_text(json.dumps(stub_plan or {}))
    monkeypatch.setenv("EVAL_TEST_PLAN", str(plan_file))
    if skew:
        monkeypatch.setenv("EVAL_TEST_SKEW", "1")
    pythonpath = os.pathsep.join(os.path.abspath(p) for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p)
    env_vars = {"EVAL_TEST_PLAN": str(plan_file), "PYTHONPATH": pythonpath, **({"EVAL_TEST_SKEW": "1"} if skew else {})}
    output = tmp_path / "launch"
    config = {"model": {"key": "gr00t", "checkpoint": "/a"}, "tasks": ["t"], "cells": "all", "version": "v1",
              "servers": {"layout": layout, "gpu_memory_gb": {"server": 10, "sim": 10},
                          "recipe": {"setup": f'case ":$PYTHONPATH:" in *":{CLIENT}:"*) ;; *) exit 1 ;; esac',
                                     "command": "echo Server ready; exec sleep 600"}},
              "plan": {"max_attempts": 2, "min_deadline_s": 15, "deadline_factor": 0},
              "backend": {"kind": kind, "gpus": FAKE_GPUS[:gpus] if kind == "local" else gpus, "gpu_memory_gb": 46,
                          "pythonpath": [CLIENT]},
              "output": str(output)}
    port_base = 20000 + os.getpid() % 1000 * 30
    if kind == "local":
        config["backend"]["local"] = {"station_restarts": 1, "port_base": port_base, "setup_hook": HOOK}
    else:
        config["backend"]["ray"] = {"num_cpus": 8, "station_restarts": 1, "port_base": port_base,
                                    "runtime_env": {"worker_process_setup_hook": HOOK, "env_vars": env_vars}}
    output.mkdir()
    plan = {"config": config, "tasks": {"t": {}}, "units": units, "eval_version": "v1"}
    (output / "plan.json").write_text(json.dumps(plan))
    return config, output, plan


def _run(config, output, plan, **kwargs):
    kwargs = {"poll_s": 0.2, **kwargs}
    return driver.Driver(config, output, plan, _executor(config["backend"]["kind"], config, output), **kwargs).run()


def _ends(output, unit_id):
    directory = output / "state" / "attempts" / unit_id
    return [json.loads(path.read_text())["kind"] for path in sorted(directory.glob("*.end.json"))]


def _left_running(output):
    """Processes the launch's slots and servers started (its servers, its stand-in simulators: each carries the launch's
    owner tag) that are still running. A launch must leave none, whatever killed its slots."""
    tag = f"{OWNER_VARIABLE}={driver.owner_scope(output)}|".encode()
    found = []
    for entry in os.scandir("/proc"):
        if entry.name.isdigit():
            try:
                if any(item.startswith(tag) for item in Path(f"/proc/{entry.name}/environ").read_bytes().split(b"\0")):
                    found.append(int(entry.name))
            except OSError:
                continue
    return found


@pytest.mark.parametrize("layout, gpus", [("colocated", 2), ("server+1", 4), ("server+2", 4),
                                          ({"sim_gpus": 2, "sims_per_gpu": 2}, 3)],
                         ids=["colocated", "server+1", "server+2", "2x2"])
def test_every_simulator_sees_one_gpu_and_its_server_is_where_the_layout_says(tmp_path, monkeypatch, kind, layout, gpus):
    units = _units(8)
    config, output, plan = _launch(tmp_path, monkeypatch, kind, units, {u["id"]: ["slow:1"] for u in units},
                                   layout=layout, gpus=gpus)
    assert _run(config, output, plan) == 0
    records = LaunchState(output, units).accepted_records()
    assert len(records) == 8
    by_station = defaultdict(set)
    for record in records:
        (outcome,) = [o for o in record["outcomes"] if o["attempt"] == record["accepted"]]
        seen = json.loads((Path(outcome["output"]) / driver_stub.SEEN).read_text())
        assert seen["gpu"] == outcome["gpu"] and seen["gpu"].isdigit(), seen
        assert seen["token"] and seen["attempt"] == record["accepted"]
        assert seen["pythonpath"].split(os.pathsep)[-1] == CLIENT
        assert any(arg.startswith("remote_port=") for arg in seen["argv"])
        station = outcome["worker"].split(".")[0]
        by_station[station].add((outcome["worker"], seen["gpu"], outcome["server_gpu"]))
    shape = parse_layout({"layout": layout})
    gpu_sets = []
    for station, slots in by_station.items():
        (server_gpu,) = {server for _, _, server in slots}
        sim_gpus = Counter(gpu for _, gpu, _ in {(w, g, s) for w, g, s in slots})
        if shape.sim_gpus == 0:
            assert set(sim_gpus) == {server_gpu}
        else:
            assert server_gpu not in sim_gpus and len(sim_gpus) <= shape.sim_gpus
            assert all(count <= shape.sims_per_gpu for count in sim_gpus.values())
        gpu_sets.append({server_gpu, *sim_gpus})
        if kind == "local":
            index = int(station[1:])
            mine = FAKE_GPUS[index * shape.gpus:(index + 1) * shape.gpus]
            assert server_gpu == mine[0] and set(sim_gpus) <= set(mine[1:] or mine[:1])
    assert all(not a & b for i, a in enumerate(gpu_sets) for b in gpu_sets[i + 1:])
    assert not _left_running(output)


def test_each_unit_is_accepted_once_through_failures_lost_slots_and_revoked_ones(tmp_path, monkeypatch, kind):
    """A failed attempt, a slot SIGKILLed mid-unit (its Kit left running), a slot that stops answering (SIGSTOP:
    revoked past its deadline, killed, restarted), a hung simulator stopped at its deadline, and a unit that fails
    every attempt: every other unit ends with one accepted attempt, and nothing the dead slots started is left."""
    units = _units(6)
    plan = {"t.train.000": ["kill_actor"], "t.train.001": ["fail"], "t.train.002": ["freeze_actor"],
            "t.train.003": ["fail", "fail"], "t.train.005": ["hang"]}
    config, output, launch_plan = _launch(tmp_path, monkeypatch, kind, units, plan)
    assert _run(config, output, launch_plan, revoke_grace_s=5.0) == EXIT_ENDED_WITH_FAILURES
    state = LaunchState(output, units)
    assert state.status() == {"todo": 0, "running": 0, "done": 5, "failed": 1}
    assert {unit_id: _ends(output, unit_id) for unit_id in state.units} == {
        "t.train.000": ["lost", "accepted"], "t.train.001": ["failed", "accepted"],
        "t.train.002": ["revoked", "accepted"], "t.train.003": ["failed", "failed"],
        "t.train.004": ["accepted"], "t.train.005": ["failed", "accepted"]}
    for record in state.accepted_records():
        accepted = [o for o in record["outcomes"] if o["attempt"] == record["accepted"]]
        assert len(accepted) == 1 and Path(accepted[0]["output"]).name == record["accepted"]
    deadline = json.loads(next((output / "state/attempts/t.train.005").glob("001.end.json")).read_text())
    assert "deadline" in deadline["outcome"]["error"]
    assert sum(len(cells["train"]) for cells in launch.accepted_outputs(output).values()) == 5
    assert {entry["unit"] for entry in launch.retried_units(output)["t"]} == {
        "t.train.000", "t.train.001", "t.train.002", "t.train.005"}
    assert not _left_running(output)


def test_a_station_that_cannot_run_hands_units_back_untried_and_a_resume_runs_them(tmp_path, monkeypatch, kind):
    """The cell runner refuses before running (the tree, its rows or the model): released, never charged; the station
    is rebuilt, then retired; the launch stops with the unit waiting, and runs it on --resume once fixed."""
    units = _units(1)
    config, output, plan = _launch(tmp_path, monkeypatch, kind, units, {"t.train.000": ["refuse"] * 3}, gpus=1)
    assert _run(config, output, plan) == 1
    state = LaunchState(output, units)
    assert state.status()["todo"] == 1 and state.units["t.train.000"].charged() == 0
    assert _ends(output, "t.train.000") == ["released", "released"]
    (tmp_path / "stub_plan.json").write_text("{}")
    assert _run(config, output, plan) == 0
    assert LaunchState(output, units).status()["done"] == 1
    assert not _left_running(output)


def _driver(config_path, module):
    return subprocess.Popen([sys.executable, "-m", module, str(config_path), "--resume"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)


def test_a_driver_killed_mid_launch_loses_no_unit_and_accepts_none_twice(tmp_path, monkeypatch, kind):
    """SIGKILL to the driver (its Ray instance goes with it) while units run; the next driver accepts what finished and
    was checked, charges what it cannot tell, and runs the rest."""
    units = _units(8)
    config, output, _ = _launch(tmp_path, monkeypatch, kind, units, {u["id"]: ["slow:4"] for u in units})
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(yaml.safe_dump(config))
    first = _driver(config_path, "eval_launch")
    deadline = time.monotonic() + 240
    while len(list((output / "accepted").glob("*.json"))) < 2 and time.monotonic() < deadline:
        assert first.poll() is None, first.stdout.read().decode()[-3000:]
        time.sleep(0.2)
    os.killpg(first.pid, signal.SIGKILL)
    first.wait()
    before = len(list((output / "accepted").glob("*.json")))
    assert 2 <= before < 8
    second = _driver(config_path, launch.DRIVER_MODULES[kind])
    log = second.communicate(timeout=300)[0].decode()
    assert second.returncode == 0, log[-3000:]
    state = LaunchState(output, units)
    assert state.status() == {"todo": 0, "running": 0, "done": 8, "failed": 0}
    tokens = [record["accepted"] for record in state.accepted_records()]
    assert len(set(tokens)) == 8 and len(list((output / "accepted").glob("*.json"))) == 8
    assert {token.split("-")[2] for token in tokens} <= {"e1", "e2"}
    assert all(not record.live() for record in state.units.values())
    assert "recovered what the last driver left" in log or before == len([t for t in tokens if "-e1-" in t])
    assert not _left_running(output)


def test_a_second_driver_on_a_running_launch_exits_with_the_code_a_restarting_caller_waits_on(tmp_path, monkeypatch,
                                                                                               kind):
    """The k8s head restarts its driver until it exits 0 or EXIT_ENDED_WITH_FAILURES; one that finds another driver
    holding the launch exits EXIT_DRIVER_RUNNING and changes nothing, on either executor."""
    units = _units(2)
    config, output, _ = _launch(tmp_path, monkeypatch, kind, units, {u["id"]: ["slow:4"] for u in units})
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(yaml.safe_dump(config))
    first = _driver(config_path, launch.DRIVER_MODULES[kind])
    deadline = time.monotonic() + 240
    while not list((output / "state" / "attempts").glob("*/*.begin.json")) and time.monotonic() < deadline:
        assert first.poll() is None, first.stdout.read().decode()[-3000:]
        time.sleep(0.2)
    second = _driver(config_path, launch.DRIVER_MODULES[kind])
    log = second.communicate(timeout=120)[0].decode()
    assert second.returncode == EXIT_DRIVER_RUNNING, log[-3000:]
    assert first.wait(timeout=300) == 0, first.stdout.read().decode()[-3000:]
    assert LaunchState(output, units).status() == {"todo": 0, "running": 0, "done": 2, "failed": 0}
    assert not _left_running(output)


def test_skewed_wall_clocks_in_every_process_change_nothing(tmp_path, monkeypatch, kind):
    """Every slot's and server's clock off by up to 30 days (a different offset each), the driver's by a week, the
    stand-ins' file stamps random: the same failures, revocations and acceptances as with agreeing clocks."""
    units = _units(4)
    plan = {"t.train.000": ["fail"], "t.train.001": ["freeze_actor"], "t.train.002": ["kill_actor"]}
    config, output, launch_plan = _launch(tmp_path, monkeypatch, kind, units, plan, skew=True)
    real = time.time
    monkeypatch.setattr(time, "time", lambda: real() - 7 * 86400)
    assert _run(config, output, launch_plan, revoke_grace_s=5.0) == 0
    state = LaunchState(output, units)
    assert state.status() == {"todo": 0, "running": 0, "done": 4, "failed": 0}
    assert {unit_id: _ends(output, unit_id) for unit_id in state.units} == {
        "t.train.000": ["failed", "accepted"], "t.train.001": ["revoked", "accepted"],
        "t.train.002": ["lost", "accepted"], "t.train.003": ["accepted"]}
    assert not _left_running(output)


def test_a_crashed_or_lost_attempt_keeps_its_finished_episodes_and_the_next_runs_only_the_rest(tmp_path, monkeypatch, kind):
    """One unit's simulator crashes after two of its four episodes (its slot reports what it finished), another's
    slot is killed after three (the driver reads what it finished): on either executor, each next attempt runs only the
    rows left, and every row is accepted from exactly one attempt."""
    pytest.importorskip("av", reason="the salvage reads the attempt's videos with PyAV")
    units = [{**unit, "slices": {"r": [0, 4]}} for unit in _units(2)]
    config, output, plan = _launch(tmp_path, monkeypatch, kind, units, {"t.train.000": ["crash_after:2"], "t.train.001": ["kill_after:3"]})
    assert _run(config, output, plan) == 0
    state = LaunchState(output, units)
    assert {unit_id: _ends(output, unit_id) for unit_id in state.units} == {
        "t.train.000": ["failed", "accepted"], "t.train.001": ["lost", "accepted"]}
    for unit_id, left in (("t.train.000", [2, 4]), ("t.train.001", [3, 4])):
        record = state.units[unit_id]
        first, second = (begin["attempt"] for _, begin in sorted(record.begins.items()))
        seen = json.loads((output / "units" / unit_id / second / driver_stub.SEEN).read_text())
        assert json.loads(seen["argv"][seen["argv"].index("--slices") + 1]) == {"r": left}
        assert {row: token for row, token in record.rows.items()} == {
            **{("r", i): first for i in range(left[0])}, **{("r", i): second for i in range(*left)}}
        (accepted,) = [r for r in state.accepted_records() if r["unit"]["id"] == unit_id]
        assert [part["rows"] for part in accepted["parts"]] == [{"r": list(range(left[0]))}, {"r": list(range(*left))}]
    assert not _left_running(output)


def test_the_local_backend_runs_without_ray_and_finds_its_gpus(monkeypatch, tmp_path):
    """No Ray in a local launch's processes; its GPUs are the ones listed, or the first of those visible."""
    imported = subprocess.run([sys.executable, "-c", "import sys, eval_launch.local_backend; "
                               "print(sorted(m for m in sys.modules if m == 'ray' or m.startswith('ray.')))"],
                              capture_output=True, text=True, check=True)
    assert imported.stdout.strip() == "[]"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,6")
    assert local_backend.local_gpus({}) == ["4", "6"]
    assert local_backend.local_gpus({"gpus": 1}) == ["4"]
    assert local_backend.local_gpus({"gpus": ["1", "3"]}) == ["1", "3"]
    with pytest.raises(ValueError, match="shows 2 GPUs"):
        local_backend.local_gpus({"gpus": 3})
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    monkeypatch.setenv("PATH", str(tmp_path))
    assert local_backend.local_gpus({"gpus": 2}) == ["0", "1"]
    with pytest.raises(ValueError, match="state backend.gpus"):
        local_backend.local_gpus({})


def test_a_unit_sharing_its_gpu_with_other_simulators_gets_their_share_of_time(tmp_path, monkeypatch, kind):
    """The plan's costs are one simulator per GPU; with two on one GPU each unit's deadline doubles, on either executor (2026-11-05, L40S:
    an H2 unit with four simulators on its GPU was stopped at the one-simulator deadline)."""
    units = _units(1)
    config, output, plan = _launch(tmp_path, monkeypatch, kind, units, {"t.train.000": ["hang"]}, layout="server+2", gpus=2)
    assert _run(config, output, plan) == 0
    first = json.loads(next((output / "state/attempts/t.train.000").glob("001.end.json")).read_text())
    assert first["outcome"]["error"] == "stopped at its deadline, 30 s"
    assert not _left_running(output)


@pytest.mark.skipif(importlib.util.find_spec("ray") is None, reason="the Ray executor needs Ray (ray==2.58.1)")
def test_a_ray_server_takes_its_secrets_from_its_own_node_and_ray_carries_their_names_only(tmp_path, monkeypatch):
    """The servers' environment reaches each server actor as an argument, which Ray keeps on the head: a variable the
    config names a secret goes by name only, and the server gets the value of the node its actor runs on (here the
    workers' runtime environment stands in for a worker pod's secretKeyRef)."""
    ray_backend = importlib.import_module("eval_launch.ray_backend")
    units = _units(2)
    config, output, plan = _launch(tmp_path, monkeypatch, "ray", units)
    config["backend"]["ray"]["runtime_env"]["env_vars"]["HF_TOKEN"] = "the-nodes-value"
    config["backend"]["k8s"] = {"secret_env": {"HF_TOKEN": {"secret": "hf-token", "key": "token"}}}
    config["servers"]["recipe"]["command"] = 'printf %s "$HF_TOKEN" > {worker_dir}/seen-token; echo Server ready; exec sleep 600'
    sent = []
    real = ray_backend.ServerRemote

    class Spy:
        def options(self, **options):
            remote = real.options(**options)

            class Recorded:
                def remote(self, *args):
                    sent.append(repr(args))
                    return remote.remote(*args)
            return Recorded()

    monkeypatch.setattr(ray_backend, "ServerRemote", Spy())
    shell = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "HF_TOKEN": "the-shells-value"}
    executor = ray_backend.RayExecutor(config, output, shell)
    assert driver.Driver(config, output, plan, executor, poll_s=0.2).run() == 0
    assert sent and not any("the-shells-value" in args or "the-nodes-value" in args for args in sent)
    assert all("'HF_TOKEN'" in args for args in sent)
    seen = {path.read_text() for path in (output / "logs").glob("*/seen-token")}
    assert seen == {"the-nodes-value"}
    assert not _left_running(output)


def test_once_nothing_is_left_to_start_idle_stations_go_one_stays_and_a_late_failure_runs_again(tmp_path, monkeypatch,
                                                                                                  kind, capsys):
    """Every unit has started and the quick ones have ended while one still runs: the idle stations are torn down (their
    GPUs go back; on Kubernetes their pods end), except one, and the running unit's station is kept. That unit then
    fails, and its retry runs on a station that stayed; every unit is accepted and nothing is left running."""
    units = _units(4)
    config, output, plan = _launch(tmp_path, monkeypatch, kind, units, {"t.train.000": ["slow_fail:6"]}, gpus=4)
    assert _run(config, output, plan, release_every_s=0.0) == 0
    state = LaunchState(output, units)
    assert state.status() == {"todo": 0, "running": 0, "done": 4, "failed": 0}
    assert _ends(output, "t.train.000") == ["failed", "accepted"]
    log = capsys.readouterr().out
    released = [line for line in log.splitlines() if "stations with nothing left to run" in line]
    assert released, log[-3000:]
    gone = {name for line in released for name in line.split("(", 1)[1].split(")", 1)[0].split(", ")}
    assert len(gone) == 2, released
    record = state.units["t.train.000"]
    first, second = (begin["station"] for _, begin in sorted(record.begins.items()))
    assert first not in gone and second not in gone
    assert not _left_running(output)


def test_a_lone_station_running_the_last_unit_is_kept_and_its_failure_runs_again_there(tmp_path, monkeypatch, kind,
                                                                                       capsys):
    """With one station there is no spare to keep: the slow unit's station stays while it runs (a busy station is
    never released), its failure runs again there, and the launch ends with every unit accepted."""
    units = _units(2)
    config, output, plan = _launch(tmp_path, monkeypatch, kind, units, {"t.train.000": ["slow_fail:3"]}, gpus=1)
    assert _run(config, output, plan, release_every_s=0.0) == 0
    assert _ends(output, "t.train.000") == ["failed", "accepted"]
    assert "stations with nothing left to run" not in capsys.readouterr().out
    assert not _left_running(output)


@pytest.mark.skipif(importlib.util.find_spec("ray") is None, reason="the Ray executor needs Ray (ray==2.58.1)")
def test_a_worker_pod_is_released_only_once_ray_can_fence_its_whole_node(tmp_path, monkeypatch, capsys):
    """The node of a pod (here this host's Ray instance, advertising pod w-0's resource and the tag of its join) is
    released only when Ray can place a fence of all its GPUs there: never while a station's placement group holds one,
    once after the station is torn down (the fence then stands, so nothing more is placed there), naming that join;
    never outside Kubernetes (no release directory). A fence that stops standing while the pod is up is dropped and its
    release taken back, then the pod is fenced and released again; a release naming another join of the pod is
    written again for the current one."""
    ray = pytest.importorskip("ray")
    ray_backend = importlib.import_module("eval_launch.ray_backend")
    units = _units(1)
    config, output, _ = _launch(tmp_path, monkeypatch, "ray", units)
    ray.init(num_gpus=2, num_cpus=4, resources={f"{ray_backend.POD_RESOURCE}w_0": 1, f"{ray_backend.JOIN_RESOURCE}j1": 1},
             include_dashboard=False, namespace="eval_launch")
    try:
        assert ray_backend.RayExecutor(config, output).release_nodes([]) == []
        release_dir = tmp_path / "launch/k8s/me-1.release"
        config["backend"]["ray"]["release_dir"] = str(release_dir)
        executor = ray_backend.RayExecutor(config, output)
        station = driver.Station(0, 0)
        ray.get(executor.place(station), timeout=60)
        assert executor.release_nodes([station]) == []
        assert not release_dir.exists() or not any(release_dir.iterdir())
        executor.tear_down([station])
        deadline = time.monotonic() + 60
        released = []
        while not released and time.monotonic() < deadline:
            released = executor.release_nodes([])
        assert released == ["pod w_0"] and sorted(path.name for path in release_dir.iterdir()) == ["w_0"]
        assert "join j1" in (release_dir / "w_0").read_text().splitlines()
        lost = executor.fences["w_0"]
        ray.util.remove_placement_group(lost)
        while ray.util.placement_group_table(lost).get("state") != "REMOVED":
            time.sleep(0.1)
        assert executor.release_nodes([]) == ["pod w_0"]
        assert "taken back" in capsys.readouterr().out
        assert executor.fences["w_0"].id != lost.id
        assert ray.util.placement_group_table(executor.fences["w_0"])["state"] == "CREATED"
        again = driver.Station(1, 10)
        ready = executor.place(again)
        assert not ray.wait([ready], timeout=5)[0], "the fenced node takes no new placement group"
        assert executor.release_nodes([]) == []
        later = ray_backend.RayExecutor(config, output)
        with pytest.raises(RuntimeError, match="could not be fenced"):
            later._fence_released(timeout_s=1.0)
        ray.util.remove_placement_group(executor.fences.pop("w_0"))
        later._fence_released(timeout_s=60.0)
        assert set(later.fences) == {"w_0"}
        (release_dir / "w_0").write_text("join an-earlier-node\n")
        assert later.release_nodes([]) == []
        assert "join j1" in (release_dir / "w_0").read_text().splitlines(), "released again, naming the current node"
    finally:
        ray.shutdown()


def test_no_station_goes_while_the_running_units_lack_their_spares(tmp_path, monkeypatch, kind, capsys):
    """Two units run on two of three stations and the third is idle: one spare per running unit is two, so nothing is
    released; once one of them has ended, one unit runs and one idle station is its spare, and only then does a
    station go."""
    units = _units(3)
    config, output, plan = _launch(tmp_path, monkeypatch, kind, units,
                                   {"t.train.000": ["slow:4"], "t.train.001": ["slow:8"]}, gpus=3)
    assert _run(config, output, plan, release_every_s=0.0) == 0
    assert LaunchState(output, units).status() == {"todo": 0, "running": 0, "done": 3, "failed": 0}
    releasing = [line for line in capsys.readouterr().out.splitlines() if line.startswith(f"[{kind}] releasing")]
    assert releasing and all(" 1 units running on " in line for line in releasing), releasing
    assert not _left_running(output)


class _QuietExecutor:
    """An executor that runs nothing: it records what the driver asks of a release pass, and answers the actor probe
    for the stations it is told answer."""

    kind, gpus, port_base, slot_restarts, station_restarts = "fake", 3, 0, 1, 1

    def __init__(self, answering):
        self.answering, self.torn, self.node_passes = set(answering), [], 0

    def quiet(self, stations):
        return {station.name for station in stations} & self.answering

    def tear_down(self, stations, graceful=False):
        self.torn += [station.name for station in stations]

    def release_nodes(self, kept):
        self.node_passes += 1
        return []


def _release_pass(tmp_path, monkeypatch, answering):
    units = _units(2)
    config, output, plan = _launch(tmp_path, monkeypatch, "local", units)
    executor = _QuietExecutor(answering)
    run = driver.Driver(config, output, plan, executor, release_every_s=0.0)
    run.stations = [driver.Station(index, 0) for index in range(3)]
    for station in run.stations:
        station.ready = True
    running, spare, gone = run.stations
    gone.released = True
    running.busy[0] = "handle"
    run.inflight["handle"] = driver.Inflight(attempt=None, station=running, slot=0, revoke_at=float("inf"))
    monkeypatch.setattr(run.state, "pending", lambda: [])
    run._release_idle()
    return executor


def test_a_pass_that_only_lets_go_of_nodes_still_needs_its_spare_to_answer(tmp_path, monkeypatch):
    """A unit runs on one station, one idle station is its spare and the third was released before: no station is a
    candidate, yet the pass asks the spare, and lets go of no node while it does not answer. The running station is
    never torn down."""
    silent = _release_pass(tmp_path, monkeypatch, answering=set())
    assert silent.node_passes == 0 and silent.torn == []
    (tmp_path / "b").mkdir()
    answers = _release_pass(tmp_path / "b", monkeypatch, answering={"s1"})
    assert answers.node_passes == 1 and answers.torn == []
