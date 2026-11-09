"""The smoke check (``--smoke``): one station of the config's layout runs a few rows on every simulator slot through
the config's own driver, and the check passes or fails on frames, timing and GPU memory. Here the local driver with
stand-in simulators (``driver_stub``) and a GPU memory reading the test sets."""

from __future__ import annotations

import importlib.util
import json
import os
import signal
import sys
import threading
import time

import pytest
import yaml

pytest.importorskip("isaaclab_arena.evaluation.arena_experiment_result", reason="eval_worker needs IsaacLab-Arena on the path")

from eval_launch import launch, smoke  # noqa: E402

HOOK = "eval_launch.tests.driver_stub.setup"


@pytest.fixture(params=["local", pytest.param("ray", marks=pytest.mark.skipif(
    importlib.util.find_spec("ray") is None, reason="the Ray executor needs Ray (ray==2.58.1)"))])
def kind(request):
    return request.param


def _config(tmp_path, port_base, kind="local"):
    config = {"model": {"key": "gr00t", "checkpoint": "/a"}, "tasks": ["r2"], "cells": "all", "version": "v1",
              "servers": {"layout": {"sim_gpus": 0, "sims_per_gpu": 2}, "gpu_memory_gb": {"server": 10, "sim": 10},
                          "recipe": {"command": "echo Server ready; exec sleep 600"}},
              "plan": {"min_deadline_s": 60},
              "backend": {"kind": kind, "gpus": ["11", "13"] if kind == "local" else 2, "gpu_memory_gb": 46},
              "output": str(tmp_path / "launch")}
    if kind == "local":
        config["backend"]["local"] = {"port_base": port_base, "setup_hook": HOOK, "station_restarts": 0}
    else:
        pythonpath = os.pathsep.join(os.path.abspath(p) for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p)
        env_vars = {name: os.environ[name] for name in ("EVAL_TEST_PLAN", "EVAL_TEST_GPU_MEMORY")}
        config["backend"]["ray"] = {"num_cpus": 8, "port_base": port_base, "station_restarts": 0,
                                    "runtime_env": {"worker_process_setup_hook": HOOK,
                                                    "env_vars": {**env_vars, "PYTHONPATH": pythonpath}}}
    return config


@pytest.mark.parametrize("slowdown, memory, statuses", [
    (9.0, "30000,46068", {"frames": "pass", "timing": "pass", "memory": "pass"}),
    (20.0, "30000,46068", {"frames": "pass", "timing": "advisory", "memory": "pass"}),
    (9.0, None, {"frames": "pass", "timing": "pass", "memory": "not measured"}),
    (40.0, "45500,46068", {"frames": "pass", "timing": "fail", "memory": "fail"}),
    (None, "30000,46068", {"frames": "pass", "timing": "fail", "memory": "pass"}),
])
def test_a_smoke_check_runs_one_station_and_says_what_failed(tmp_path, monkeypatch, capsys, kind, slowdown, memory,
                                                             statuses):
    """colocated+2: two slots, two units of two rows each at once. The plan's slowdown is the default 5 per simulator,
    10 with two sharing a GPU: 20 is past its 1.5x tolerance but within the deadline factor 3 (a warning), 40 past
    that (a failure). 45.5 of 46 GB is past the memory limit, and past the 30 GB the config states for the GPU
    (10 server + 2 x 10); no reading at all is "not measured", which fails nothing."""
    stub_plan = tmp_path / "stub_plan.json"
    stub_plan.write_text(json.dumps({f"r2.train.smoke{index:02d}": [f"timed:{slowdown}"] if slowdown else ["ok"]
                                     for index in range(2)}))
    monkeypatch.setenv("EVAL_TEST_PLAN", str(stub_plan))
    monkeypatch.setenv("EVAL_TEST_GPU_MEMORY", memory or "none")
    path = tmp_path / "eval.yaml"
    config = _config(tmp_path, 21000 + os.getpid() % 500 * 20, kind)
    config["output"] += "/"
    path.write_text(yaml.safe_dump(config))
    rc = launch.main([str(path), "--smoke", "--smoke-rows", "2"])
    (smoke_dir,) = (tmp_path / "launch-smoke").iterdir()
    result = json.loads((smoke_dir / "smoke.json").read_text())
    passed = "fail" not in statuses.values()
    assert rc == (0 if passed else 1) and result["ok"] == passed and result["result"] == ("pass" if passed else "fail")
    for check, status in statuses.items():
        assert [item["status"] for item in result[check]] == [status, status], (check, result[check])
    plan = json.loads((smoke_dir / "plan.json").read_text())
    assert [sum(end - start for start, end in unit["slices"].values()) for unit in plan["units"]] == [2, 2]
    episode_s, costs = plan["tasks"]["r2"]["episode_s"], plan["costs"]
    assert [unit["expected_s"] for unit in plan["units"]] == [
        costs["kit_start_s"] + len(unit["slices"]) * costs["build_s"] + 2 * episode_s for unit in plan["units"]]
    assert not (tmp_path / "launch").exists(), "the config's own output is left alone"
    out = capsys.readouterr().out
    assert ("PASS" if passed else "FAIL") in out
    assert "a floor, not the launch's peak" in result["memory_limits"] and "r2 train" in result["memory_limits"]
    assert "memory is a floor" in out


def test_a_smoke_check_needs_a_driver_backend(tmp_path, capsys):
    config = _config(tmp_path, 5810)
    config["backend"] = {"kind": "k8s", "gpus": 2}
    path = tmp_path / "eval.yaml"
    path.write_text(yaml.safe_dump(config))
    assert launch.main([str(path), "--smoke"]) == smoke.NOT_RUN
    assert "local or Ray driver" in capsys.readouterr().err
    path.write_text(yaml.safe_dump(_config(tmp_path, 5810)))
    assert launch.main([str(path), "--smoke", "--tasks", "nosuchtask"]) == smoke.NOT_RUN
    assert "can be planned for the smoke" in capsys.readouterr().err


def test_a_stopped_smoke_waits_for_its_drivers_teardown_and_leaves_no_saved_environment(tmp_path, monkeypatch):
    """Ctrl-C reaches the driver too (it is in the launcher's process group): the smoke must not kill it 0.25 s into
    its teardown, as subprocess.run does, but wait for it and then say it stopped (130). The servers' environment it
    handed the driver is on no disk, read or not."""
    (tmp_path / "slow_driver.py").write_text(
        "import pathlib, sys, time\ntime.sleep(1.5)\npathlib.Path(sys.argv[1]).write_text('torn down')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, (str(tmp_path), os.environ.get("PYTHONPATH")))))
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    monkeypatch.setattr(smoke.server_env.tempfile, "tempdir", None)
    marker = tmp_path / "marker"
    threading.Timer(0.5, lambda: os.kill(os.getpid(), signal.SIGINT)).start()
    started = time.monotonic()
    rc, stopped = smoke._run_driver([sys.executable, "-m", "slow_driver", str(marker)], {"HF_TOKEN": "s3cret"})
    assert stopped == "SIGINT" and rc == 0 and marker.read_text() == "torn down", "the driver finished its teardown"
    assert time.monotonic() - started >= 1.4
    assert not list((tmp_path / "tmp").iterdir()), "the servers' environment is on no disk"


def test_past_the_deadline_fails_even_when_the_deadline_factor_is_below_the_tolerance():
    """With plan.deadline_factor 1.2, a unit at 1.4x the plan's slowdown is past its deadline though within the 1.5x
    tolerance: it fails; 1.1x passes."""
    from eval_launch.acceptance import LaunchState

    plan = {"tasks": {"r2": {"episode_s": 50.0, "cap_s": 10.0}},
            "units": [{"id": f"r2.train.smoke0{index}", "task": "r2", "cell": "train"} for index in range(2)]}
    config = {"servers": {}, "plan": {"deadline_factor": 1.2}}

    class Recorded(LaunchState):
        def __init__(self, slowdowns):
            self.slowdowns = slowdowns

        def accepted_records(self):
            return [{"unit": {"id": unit_id}, "outcomes": [{"stepping_s": slowdown * 10.0, "steps": 600}]}
                    for unit_id, slowdown in self.slowdowns.items()]

        def failed_records(self):
            return []

    result = smoke.verdict(config, plan, Recorded({"r2.train.smoke00": 5.0 * 1.4, "r2.train.smoke01": 5.0 * 1.1}),
                           {"r2": 60.0})
    assert [item["status"] for item in result["timing"]] == ["fail", "pass"]
