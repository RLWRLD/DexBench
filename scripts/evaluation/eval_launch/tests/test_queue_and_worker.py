# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""The queue hands each unit to one worker, longest first and the loaded checkpoint first; a claim is fenced to its
attempt, so a late or recovered attempt is never credited; a dead worker's unit comes back and counts as an attempt; a
process that dies mid-step leaves nothing lost and nothing run twice; one worker stays until every unit has ended;
the worker accepts only complete units, stops on its own failures, and keeps its server on its own GPU; a launch
resumes with its new runtime settings and scores only accepted attempts of every planned unit."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("isaaclab_arena.evaluation.arena_experiment_result", reason="eval_worker needs IsaacLab-Arena on the path")

from dexbench_arena.benchmark.models import ModelEntry, resolve_model  # noqa: E402

from eval_launch import eval_worker, launch, process_group, queue_dir  # noqa: E402


def _units(n, checkpoint="/a"):
    return [{"id": f"t.train.{i:03d}", "task": "t", "cell": "train", "slices": {"r": [i, i + 1]},
             "checkpoint": checkpoint, "expected_s": 100 - i} for i in range(n)]


def _age(path, seconds):
    os.utime(path, (time.time() - seconds, time.time() - seconds))


def test_units_are_claimed_once_each_in_plan_order_and_a_restart_adds_nothing(tmp_path):
    assert queue_dir.init(tmp_path, _units(3)) == 3
    assert queue_dir.init(tmp_path, _units(3)) == 0
    claims = [queue_dir.claim(tmp_path, f"w{i}") for i in range(4)]
    assert [record["unit"]["id"] for _, record in claims[:3]] == [u["id"] for u in _units(3)]
    assert claims[3] is None
    assert queue_dir.status(tmp_path) == {"todo": 0, "claimed": 3, "done": 0, "failed": 0}


def test_a_worker_takes_its_loaded_checkpoint_first(tmp_path):
    queue_dir.init(tmp_path, _units(2, "/a") + [{**_units(1, "/b")[0], "id": "t.train.900"}])
    _, record = queue_dir.claim(tmp_path, "w", prefer={"checkpoint": "/b"})
    assert record["unit"]["id"] == "t.train.900"


def test_a_failed_unit_gets_one_more_attempt_then_stays_failed(tmp_path):
    queue_dir.init(tmp_path, _units(1))
    path, _ = queue_dir.claim(tmp_path, "w")
    assert queue_dir.finish(tmp_path, path, {"rc": 1}, ok=False) == "todo"
    path, record = queue_dir.claim(tmp_path, "w")
    assert record["attempts"] == 2
    assert queue_dir.finish(tmp_path, path, {"rc": 1}, ok=False) == "failed"
    failed = json.loads(next((tmp_path / "failed").glob("*.json")).read_text())
    assert [outcome["rc"] for outcome in failed["outcomes"]] == [1, 1]


def test_a_new_claim_has_a_fresh_heartbeat_whatever_its_wait_in_the_queue(tmp_path):
    """Review 2026-11-02: a rename kept the queue file's hours-old mtime, so a stale scan could take a claim at birth."""
    queue_dir.init(tmp_path, _units(1))
    _age(next((tmp_path / "todo").iterdir()), 3600)
    path, _ = queue_dir.claim(tmp_path, "w")
    assert queue_dir.requeue_stale(tmp_path, stale_s=600) == [] and path.exists()


def test_a_live_claim_stays_whatever_the_checking_hosts_clock_says(tmp_path, monkeypatch):
    """On NFS the file server sets mtimes; a pod whose clock ran ahead of it saw live claims as stale.
    Ages are measured on the filesystem's clock."""
    queue_dir.init(tmp_path, _units(1))
    path, _ = queue_dir.claim(tmp_path, "w")
    real_time = time.time
    monkeypatch.setattr(queue_dir.time, "time", lambda: real_time() + 3600)
    assert queue_dir.requeue_stale(tmp_path, stale_s=600) == [] and path.exists()
    assert not list(tmp_path.glob(".clock*"))


def test_a_claim_without_a_heartbeat_goes_back_to_the_queue(tmp_path):
    queue_dir.init(tmp_path, _units(2))
    stale, _ = queue_dir.claim(tmp_path, "dead")
    live, _ = queue_dir.claim(tmp_path, "alive")
    _age(stale, 3600)
    assert queue_dir.requeue_stale(tmp_path, stale_s=600) == ["t.train.000"]
    assert live.exists()
    assert queue_dir.status(tmp_path)["todo"] == 1


def test_a_late_attempt_is_never_credited_even_under_the_same_worker_name(tmp_path):
    """A restarted pod keeps its worker name; its new attempt is still another claim."""
    queue_dir.init(tmp_path, _units(1))
    old, first = queue_dir.claim(tmp_path, "w0")
    _age(old, 3600)
    queue_dir.requeue_stale(tmp_path, stale_s=600)
    new, second = queue_dir.claim(tmp_path, "w0")
    assert old != new and first["attempt"] != second["attempt"]
    assert queue_dir.finish(tmp_path, old, {"rc": 0}, ok=True) == "superseded"
    assert queue_dir.finish(tmp_path, new, {"rc": 0}, ok=True) == "done"
    done = json.loads(next((tmp_path / "done").glob("*.json")).read_text())
    assert done["accepted"] == second["attempt"]


def test_a_step_that_died_half_way_loses_no_unit_and_reruns_no_accepted_one(tmp_path):
    queue_dir.init(tmp_path, _units(2))
    path, _ = queue_dir.claim(tmp_path, "w")
    queue_dir.finish(tmp_path, path, {"rc": 0}, ok=True)
    residue = path.with_name(f".finish.{path.name}")
    shutil.copy(next((tmp_path / "done").iterdir()), residue)
    _age(residue, 3600)
    waiting = next(p for p in (tmp_path / "todo").iterdir())
    hidden = tmp_path / "claimed" / f".claiming.{waiting.name}.dead"
    os.rename(waiting, hidden)
    _age(hidden, 3600)
    assert queue_dir.status(tmp_path) == {"todo": 0, "claimed": 1, "done": 1, "failed": 0}
    assert queue_dir.requeue_stale(tmp_path, stale_s=600) == ["t.train.001"]
    assert queue_dir.status(tmp_path) == {"todo": 1, "claimed": 0, "done": 1, "failed": 0}
    assert not list((tmp_path / "claimed").iterdir())


def test_a_claim_under_way_is_never_recovered_however_long_its_unit_waited(tmp_path, monkeypatch):
    """A unit that waited an hour in todo/ is renamed to a hidden transition with that hour-old mtime; recovery must
    see the claim being made, not an abandoned step."""
    queue_dir.init(tmp_path, _units(1))
    _age(next((tmp_path / "todo").iterdir()), 3600)
    publish, recovered = queue_dir._write_new, []
    def recovery_runs_mid_claim(path, data):
        monkeypatch.setattr(queue_dir, "_write_new", publish)
        recovered.extend(queue_dir.requeue_stale(tmp_path, stale_s=600))
        publish(path, data)
    monkeypatch.setattr(queue_dir, "_write_new", recovery_runs_mid_claim)
    assert queue_dir.claim(tmp_path, "w") is not None
    assert recovered == []
    assert queue_dir.status(tmp_path) == {"todo": 0, "claimed": 1, "done": 0, "failed": 0}


def test_a_recovery_scan_right_after_the_claims_rename_leaves_the_claim_alone(tmp_path, monkeypatch):
    """The scan can also run between the rename into the hidden name and the touch after it; the hidden file must
    already look freshly taken then. A launch whose pods waited 25 min for GPUs lost 7 workers to this, each
    unit requeued while its claim was being written."""
    queue_dir.init(tmp_path, _units(1))
    _age(next((tmp_path / "todo").iterdir()), 3600)
    rename, recovered = os.rename, []
    def recovery_runs_after_rename(src, dst):
        rename(src, dst)
        if ".claiming." in str(dst):
            recovered.extend(queue_dir.requeue_stale(tmp_path, stale_s=600))
    monkeypatch.setattr(queue_dir.os, "rename", recovery_runs_after_rename)
    assert queue_dir.claim(tmp_path, "w") is not None
    monkeypatch.setattr(queue_dir.os, "rename", rename)
    assert recovered == []
    assert queue_dir.status(tmp_path) == {"todo": 0, "claimed": 1, "done": 0, "failed": 0}


def test_a_scan_that_loses_a_stale_claim_to_another_scan_moves_on(tmp_path, monkeypatch):
    """A scan's own takes have the claimer's window: right after the rename to ``.stale.`` (and to ``.recover.``)
    the hidden file still carries the dead claim's old mtime, so another worker's scan takes it and requeues the
    unit. The scan that lost it records nothing and returns; the unit is requeued once."""
    queue_dir.init(tmp_path, _units(1))
    target, _ = queue_dir.claim(tmp_path, "dead")
    _age(target, 3600)
    rename, recovered, interrupted, errors = os.rename, [], set(), []
    def another_scan_after_the_rename(src, dst):
        rename(src, dst)
        step = next((step for step in (".stale.", ".recover.") if step in str(dst)), None)
        if step and step not in interrupted:
            interrupted.add(step)
            try:
                recovered.extend(queue_dir.requeue_stale(tmp_path, stale_s=600))
            except OSError as error:
                errors.append(error)
    monkeypatch.setattr(queue_dir.os, "rename", another_scan_after_the_rename)
    recovered.extend(queue_dir.requeue_stale(tmp_path, stale_s=600))
    monkeypatch.setattr(queue_dir.os, "rename", rename)
    assert interrupted == {".stale.", ".recover."} and errors == []
    assert recovered == ["t.train.000"]
    assert queue_dir.status(tmp_path) == {"todo": 1, "claimed": 0, "done": 0, "failed": 0}
    assert not list((tmp_path / "claimed").iterdir())


def test_a_claim_survives_a_scan_that_took_its_hidden_file_before_or_after_publishing(tmp_path, monkeypatch):
    """A pod whose clock runs behind stamps a fresh hidden file as old, so another pod's scan can take it anyway:
    before the claim is published the unit goes back to todo/ and the claimer moves on; after, the scan removes
    only the residue and the claim stands. The worker survives both (a launch on 2026-11-04)."""
    queue_dir.init(tmp_path, _units(2))
    read, write = queue_dir.Path.read_text, queue_dir._write_new
    def taken_before_publishing(self, *args, **kwargs):
        if ".claiming." in self.name and not taken_before_publishing.done:
            taken_before_publishing.done = True
            _age(self, 3600)
            queue_dir.requeue_stale(tmp_path, stale_s=600)
        return read(self, *args, **kwargs)
    taken_before_publishing.done = False
    monkeypatch.setattr(queue_dir.Path, "read_text", taken_before_publishing)
    first = queue_dir.claim(tmp_path, "w")
    monkeypatch.setattr(queue_dir.Path, "read_text", read)
    assert first is not None and first[1]["unit"]["id"] == "t.train.001"
    def residue_removed_after_publishing(path, data):
        write(path, data)
        for hidden in (tmp_path / "claimed").glob(".claiming.*"):
            hidden.unlink()
    monkeypatch.setattr(queue_dir, "_write_new", residue_removed_after_publishing)
    second = queue_dir.claim(tmp_path, "w")
    monkeypatch.setattr(queue_dir, "_write_new", write)
    assert second is not None and second[0].exists()
    assert queue_dir.status(tmp_path) == {"todo": 0, "claimed": 2, "done": 0, "failed": 0}


def test_a_unit_moving_back_to_the_queue_during_a_start_is_not_queued_twice(tmp_path, monkeypatch):
    """--start on a live launch: a claim released into todo/ while the queue is listed is in neither listing of that
    scan."""
    queue_dir.init(tmp_path, _units(2))
    path, _ = queue_dir.claim(tmp_path, "w")
    listing = queue_dir._ids

    def released_mid_scan(root):
        found = listing(root)
        if path.exists():
            queue_dir.release(tmp_path, path)
            found["claimed"] = set()
        return found

    monkeypatch.setattr(queue_dir, "_ids", released_mid_scan)
    assert queue_dir.init(tmp_path, _units(2)) == 0
    monkeypatch.setattr(queue_dir, "_ids", listing)
    assert queue_dir.status(tmp_path) == {"todo": 2, "claimed": 0, "done": 0, "failed": 0}
    assert len(list((tmp_path / "todo").iterdir())) == 2


def test_processes_claiming_at_once_run_every_unit_exactly_once(tmp_path):
    """Eight processes claim, release (a worker handing a unit back) and finish 240 units at the same time, each also
    scanning for stale claims: every unit ends done once, none twice (real concurrency;
    the race tests above replay one order in one process)."""
    queue_dir.init(tmp_path, _units(240))
    script = (
        "import random, sys\n"
        "from eval_launch import queue_dir\n"
        "root, worker = sys.argv[1], sys.argv[2]\n"
        "random.seed(worker)\n"
        "while (claimed := queue_dir.claim(root, worker)):\n"
        "    queue_dir.requeue_stale(root, stale_s=600)\n"
        "    if random.random() < 0.2:\n"
        "        assert queue_dir.release(root, claimed[0])\n"
        "    else:\n"
        "        assert queue_dir.finish(root, claimed[0], {'rc': 0, 'worker': worker}, ok=True) == 'done'\n"
        "    print(claimed[1]['unit']['id'], flush=True)\n")
    workers = [subprocess.Popen([sys.executable, "-c", script, str(tmp_path), f"w{i}"], stdout=subprocess.PIPE, text=True)
               for i in range(8)]
    claims = [line for worker in workers for line in worker.communicate(timeout=300)[0].split()]
    assert all(worker.returncode == 0 for worker in workers)
    assert queue_dir.status(tmp_path) == {"todo": 0, "claimed": 0, "done": 240, "failed": 0}
    records = [json.loads(path.read_text()) for path in (tmp_path / "done").iterdir()]
    assert sorted(record["unit"]["id"] for record in records) == sorted(unit["id"] for unit in _units(240))
    assert all(len(record["outcomes"]) == 1 for record in records)
    assert len(claims) >= 240 and len({record["outcomes"][0]["worker"] for record in records}) > 1


def test_a_unit_that_keeps_losing_its_worker_ends_failed_and_can_be_retried(tmp_path):
    queue_dir.init(tmp_path, _units(1))
    for _ in range(2):
        path, _ = queue_dir.claim(tmp_path, "w")
        _age(path, 3600)
        queue_dir.requeue_stale(tmp_path, stale_s=600, max_attempts=2)
    assert queue_dir.status(tmp_path) == {"todo": 0, "claimed": 0, "done": 0, "failed": 1}
    assert queue_dir.terminal(tmp_path)
    assert queue_dir.retry_failed(tmp_path) == ["t.train.000"]
    _, record = queue_dir.claim(tmp_path, "w")
    assert record["attempts"] == 1 and len(record["outcomes"]) == 2


class _Server:
    """Stands in for the policy server: counts the loads, one per checkpoint change."""

    def __init__(self):
        self.checkpoint, self.port, self.token, self.started_s, self.loads = None, 5810, "t", 1.0, 0

    def ensure(self, checkpoint):
        if checkpoint != self.checkpoint:
            self.checkpoint, self.loads = checkpoint, self.loads + 1

    def client_set(self):
        return []


def test_a_worker_runs_every_unit_keeps_failed_attempts_and_loads_each_checkpoint_once(tmp_path):
    units = _units(3, "/a") + [{**u, "id": f"t.train.9{i:02d}"} for i, u in enumerate(_units(2, "/b"))]
    queue_dir.init(tmp_path / "queue", units)
    seen = []

    def run(unit, unit_dir, server, model, deadline, environment):
        unit_dir.mkdir(parents=True)
        seen.append((unit["id"], server.checkpoint))
        first_try = sum(1 for uid, _ in seen if uid == unit["id"]) == 1
        if unit["id"] == "t.train.001" and first_try:
            return {"rc": 1, "complete": False}
        return {"rc": 0, "complete": True}

    server = _Server()
    assert eval_worker.run_worker(tmp_path, "w", server, "gr00t", run=run) == 0
    assert queue_dir.status(tmp_path / "queue") == {"todo": 0, "claimed": 0, "done": 5, "failed": 0}
    assert server.loads == 2, seen
    assert len(list((tmp_path / "units" / "t.train.001").iterdir())) == 2
    done = json.loads(next((tmp_path / "queue" / "done").glob("*t.train.001.json")).read_text())
    assert [o.get("rc") for o in done["outcomes"]] == [1, 0] and done["outcomes"][-1]["worker"] == "w"
    assert done["outcomes"][-1]["output"].endswith(done["accepted"])


@pytest.mark.parametrize("failure", [eval_worker.WorkerFailure("policy server exited (1)"),
                                     FileExistsError(17, "File exists")], ids=["server", "filesystem"])
def test_a_worker_that_cannot_run_hands_the_unit_back_untried_and_stops(tmp_path, failure):
    """2026-11-02: a server that died at start in every pod, then four pods whose filesystem raised on every unit
    (60 units failed in 42 s); now the first unit goes back untried and the worker stops."""
    queue_dir.init(tmp_path / "queue", _units(3))

    def broken(*args):
        raise failure

    with pytest.raises(SystemExit, match="handed back untried"):
        eval_worker.run_worker(tmp_path, "w", _Server(), "gr00t", run=broken)
    assert queue_dir.status(tmp_path / "queue") == {"todo": 3, "claimed": 0, "done": 0, "failed": 0}
    assert all(json.loads(p.read_text())["attempts"] == 0 for p in (tmp_path / "queue" / "todo").iterdir())


def test_an_incomplete_unit_or_a_sweep_unit_is_a_failed_attempt_never_a_result(tmp_path):
    queue_dir.init(tmp_path / "queue", _units(1))
    eval_worker.run_worker(tmp_path, "w", _Server(), "gr00t", run=lambda *a: {"rc": 0, "complete": False})
    assert queue_dir.status(tmp_path / "queue")["failed"] == 1
    sweep = {**_units(1)[0], "cell": None, "slices": None}
    assert eval_worker.run_unit(sweep, tmp_path / "u", _Server(), "gr00t")["rc"] is None


def test_the_sentinel_takes_over_a_dead_one_and_runs_a_dead_workers_unit(tmp_path):
    """Nothing to claim, a claim still out and a sentinel that stopped renewing: the worker waits out the lease,
    becomes the sentinel, requeues the claim once stale, runs it, and only then exits."""
    queue = tmp_path / "queue"
    queue_dir.init(queue, _units(1))
    dead, _ = queue_dir.claim(queue, "dead")
    (queue / eval_worker.SENTINEL).write_text("gone")
    ran = []

    def run(unit, *args):
        ran.append(unit["id"])
        return {"rc": 0, "complete": True}

    assert eval_worker.run_worker(tmp_path, "w", _Server(), "gr00t", run=run, stale_s=0.5, poll_s=0.05) == 0
    assert ran == ["t.train.000"] and queue_dir.status(queue)["done"] == 1
    (queue / eval_worker.SENTINEL).write_text("alive")
    assert not eval_worker._sentinel_renews(queue, poll_s=0.05)


def test_the_client_is_pointed_at_the_workers_server_by_the_field_its_policy_block_has():
    assert eval_worker.server_address(resolve_model("gr00t"), 5830) == ["--set", "remote_port=5830"]
    assert eval_worker.server_address(resolve_model("cosmos"), 5830) == ["--set", "remote_url=http://127.0.0.1:5830"]
    ipv6 = ModelEntry(key="ipv6", label="", note="", policy={"remote_url": "http://[::1]:5811/v1?x=1"})
    assert eval_worker.server_address(ipv6, 5830) == ["--set", "remote_url=http://[::1]:5830/v1?x=1"]


@pytest.mark.parametrize("submission", [False, True])
def test_a_model_the_config_registers_reaches_the_cell_runner_whole(tmp_path, monkeypatch, submission):
    """A config model (no models.py entry) goes to run_benchmark_cell.py as --model-entry, its key as --model, and its
    client is pointed at the worker's server by its own block."""
    runner = tmp_path / "record_argv.py"
    runner.write_text("import json, sys\nopen(sys.argv[0] + '.argv', 'w').write(json.dumps(sys.argv[1:]))\n")
    monkeypatch.setattr(eval_worker, "RUN_CELL", runner)
    monkeypatch.setattr(eval_worker, "check_unit", lambda *args: {"complete": True})
    spec = {"key": "mymodel", "label": "My Model", "group": "community",
            "policy": {"type": "dexbench_arena_cosmos.policy.cosmos3_joint_policy.Cosmos3JointPolicy",
                       "remote_url": "http://127.0.0.1:5811", "policy_device": "cpu"}}
    unit = {**_units(1)[0], "submission": submission, "checkpoint_sha256": "b" * 64}
    outcome = eval_worker.run_unit(unit, tmp_path / "u" / "a1", _Server(), spec, deadline=5.0)
    assert outcome["rc"] == 0
    argv = json.loads((tmp_path / "record_argv.py.argv").read_text())
    assert argv[argv.index("--model") + 1] == "mymodel"
    assert json.loads(argv[argv.index("--model-entry") + 1]) == spec
    assert ("--submission" in argv) == submission
    if submission:
        assert argv[argv.index("--checkpoint-sha256") + 1] == "b" * 64
    assert f"remote_url=http://127.0.0.1:{_Server().port}" in argv


def test_a_server_ready_line_from_an_earlier_start_does_not_count(tmp_path, monkeypatch):
    """Replacement instances in the same process and clock second keep separate logs and cannot inherit readiness."""
    monkeypatch.setattr(eval_worker, "time", SimpleNamespace(
        strftime=lambda fmt: "20261106T193513", monotonic=time.monotonic,
        sleep=lambda seconds: time.sleep(min(seconds, 0.01))))
    good = eval_worker.PolicyServer({"command": "echo Server ready; sleep 30"}, 5810, tmp_path)
    good.ensure("/a")
    good.stop()
    first_log = next(tmp_path.glob("server-5810-*.log"))
    first_bytes = first_log.read_bytes()
    dead = eval_worker.PolicyServer({"command": "exit 3"}, 5810, tmp_path)
    with pytest.raises(eval_worker.WorkerFailure, match="exited"):
        dead.ensure("/b")
    assert len(list(tmp_path.glob("server-5810-*.log"))) == 2
    early = eval_worker.PolicyServer({"setup": "echo Server ready", "command": "exit 3"}, 5810, tmp_path)
    with pytest.raises(eval_worker.WorkerFailure, match="exited"):
        early.ensure("/c")
    assert len(list(tmp_path.glob("server-5810-*.log"))) == 3
    assert first_log.read_bytes() == first_bytes


def test_the_servers_token_file_is_its_owners_only_from_the_start_and_goes_with_the_server(tmp_path, monkeypatch):
    """The token file is created with mode 0600 (never written first and narrowed after), the server reads it, and it
    is removed when the server stops, and when a start fails."""
    created = []
    real_open = os.open
    monkeypatch.setattr(eval_worker.os, "open", lambda path, flags, mode=0o777: created.append((Path(path).name, mode))
                        or real_open(path, flags, mode))
    seen = tmp_path / "seen"
    server = eval_worker.PolicyServer({"command": f"stat -c %a {{token_file}} > {seen}; cat {{token_file}} >> {seen}; "
                                                  "echo Server ready; sleep 30"}, 5811, tmp_path / "logs")
    server.ensure("/a")
    assert [entry for entry in created if entry[0].startswith("token-")] == [("token-5811", 0o600)]
    assert seen.read_text().split() == ["600", server.token]
    server.stop()
    assert not (tmp_path / "logs" / "token-5811").exists()
    failed = eval_worker.PolicyServer({"setup": "exit 1", "command": "true"}, 5812, tmp_path / "logs")
    with pytest.raises(eval_worker.WorkerFailure, match="setup"):
        failed.ensure("/b")
    assert not (tmp_path / "logs" / "token-5812").exists()


def test_a_token_file_whose_write_fails_is_removed_with_the_failed_start(tmp_path, monkeypatch):
    """NFS reports a full disk or quota at close: the file exists, the write failed. The start fails and the file goes."""
    real_fdopen = os.fdopen

    def failing_fdopen(descriptor, *args, **kwargs):
        handle = real_fdopen(descriptor, *args, **kwargs)
        handle.write = lambda text: (_ for _ in ()).throw(OSError(28, "No space left on device"))
        return handle

    monkeypatch.setattr(eval_worker.os, "fdopen", failing_fdopen)
    server = eval_worker.PolicyServer({"command": "echo Server ready; sleep 30"}, 5813, tmp_path / "logs")
    with pytest.raises(OSError, match="No space"):
        server.ensure("/a")
    assert not (tmp_path / "logs" / "token-5813").exists()


def test_the_server_runs_on_its_workers_gpu_and_its_setup_is_bounded(tmp_path, monkeypatch):
    """The saved server environment predates the worker's GPU; the worker's is put over it (review 2026-11-02)."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    seen = tmp_path / "gpu"
    server = eval_worker.PolicyServer({"command": f"echo $CUDA_VISIBLE_DEVICES > {seen}; echo Server ready; sleep 30",
                                       "client_set": ["policy_config_yaml_path={worker_dir}/c/policy.yaml"]},
                                      5810, tmp_path, environment={"PATH": os.environ["PATH"]})
    server.ensure("/a")
    server.stop()
    assert seen.read_text().strip() == "3"
    assert server.client_set() == [f"policy_config_yaml_path={tmp_path}/c/policy.yaml"]
    slow = eval_worker.PolicyServer({"setup": "sleep 30", "command": "true"}, 5810, tmp_path, ready_timeout_s=1.0)
    start = time.time()
    with pytest.raises(eval_worker.WorkerFailure, match="timed out"):
        slow.ensure("/a")
    assert time.time() - start < 20


@pytest.mark.parametrize("wall_jump_s", [-3601.0, 3601.0])
def test_server_readiness_uses_elapsed_time_when_the_wall_clock_jumps(tmp_path, monkeypatch, wall_jump_s):
    """A ready server survives either clock jump; a stalled one still exhausts its actual budget."""
    reads = []

    def wall_time():
        value = 100000.0 + (wall_jump_s if reads else 0.0)
        reads.append(value)
        return value

    monkeypatch.setattr(eval_worker, "time", SimpleNamespace(
        time=wall_time, monotonic=time.monotonic, strftime=time.strftime,
        sleep=lambda seconds: time.sleep(min(seconds, 0.01))))
    server = eval_worker.PolicyServer(
        {"command": "sleep 0.05; echo Server ready; exec sleep 30"}, 5810, tmp_path / "ready",
        ready_timeout_s=1800.0)
    start = time.monotonic()
    try:
        server.ensure("/a")
        assert server.checkpoint == "/a"
        assert 0 <= server.started_s <= time.monotonic() - start
    finally:
        server.stop()

    stalled = eval_worker.PolicyServer(
        {"command": "exec sleep 30"}, 5811, tmp_path / "stalled", ready_timeout_s=0.1)
    start = time.monotonic()
    try:
        with pytest.raises(eval_worker.WorkerFailure, match="not ready after"):
            stalled.ensure("/a")
        assert time.monotonic() - start >= 0.1
    finally:
        stalled.stop()


@pytest.mark.parametrize("wall_jump_s", [-3601.0, 3601.0])
@pytest.mark.parametrize("publish_queue", [False, True])
def test_queue_wait_keeps_its_budget_when_the_wall_clock_jumps(tmp_path, monkeypatch, wall_jump_s, publish_queue):
    elapsed = 0.0
    plan = {"units": _units(1)}

    def sleep(seconds):
        nonlocal elapsed
        elapsed += seconds
        assert elapsed <= 20.0, "a backward wall-clock jump extended the queue wait"
        if publish_queue:
            (tmp_path / "plan.json").write_text(json.dumps(plan))
            queue_dir.init(tmp_path / "queue", plan["units"])

    monkeypatch.setattr(eval_worker, "time", SimpleNamespace(
        time=lambda: 100000.0 + elapsed + (wall_jump_s if elapsed else 0.0),
        monotonic=lambda: elapsed, sleep=sleep))
    if publish_queue:
        assert eval_worker.wait_for_queue(tmp_path, timeout_s=20.0) == plan
        assert elapsed == 10.0
    else:
        with pytest.raises(SystemExit, match="no complete plan and queue after 20 s"):
            eval_worker.wait_for_queue(tmp_path, timeout_s=20.0)
        assert elapsed == 20.0


@pytest.mark.parametrize("wall_jump_s", [-3601.0, 3601.0])
def test_unit_duration_uses_elapsed_time_when_the_wall_clock_jumps(tmp_path, monkeypatch, wall_jump_s):
    """A real child exits; the reported duration stays within the parent's elapsed interval."""
    runner = tmp_path / "complete.py"
    runner.write_text("import time\ntime.sleep(0.02)\n")
    monkeypatch.setattr(eval_worker, "RUN_CELL", runner)
    monkeypatch.setattr(eval_worker, "check_unit", lambda *args: {"complete": True})
    reads = []

    def wall_time():
        value = 100000.0 + (wall_jump_s if reads else 0.0)
        reads.append(value)
        return value

    monkeypatch.setattr(eval_worker, "time", SimpleNamespace(time=wall_time, monotonic=time.monotonic))
    start = time.monotonic()
    outcome = eval_worker.run_unit(_units(1)[0], tmp_path / "u" / "a1", _Server(), "gr00t", deadline=5.0)
    assert outcome["rc"] == 0 and outcome["complete"]
    assert 0.02 <= outcome["wall_s"] <= time.monotonic() - start


def test_a_stopped_group_leaves_no_member_behind_even_one_that_ignores_sigterm():
    """The leader exits on SIGTERM, its child ignores it: the group is not stopped until the child is gone too."""
    process = subprocess.Popen(["bash", "-c", "(trap '' TERM; sleep 300) & sleep 300"], start_new_session=True)
    time.sleep(0.5)
    assert process_group.group_alive(process.pid)
    process_group.stop_group(process, grace_s=1.0)
    assert not process_group.group_alive(process.pid)


def test_a_restarted_worker_stops_what_a_killed_one_left_and_nothing_else(tmp_path):
    """SIGKILL gives a worker no chance to stop its simulator and server; on a host they kept the port and GPU memory
    its replacement needed. Only a dead owner's tagged processes are stopped."""
    worker = ("import subprocess, sys, time\n"
              "from eval_launch import process_group\n"
              "process_group.claim_ownership(sys.argv[1])\n"
              "child = subprocess.Popen(['sleep', '300'], start_new_session=True)  # its simulator, in its own session\n"
              "print(child.pid, flush=True)\n"
              "time.sleep(300)\n")
    killed, live = (subprocess.Popen([sys.executable, "-c", worker, f"{tmp_path}:{name}"], stdout=subprocess.PIPE, text=True)
                    for name in ("w0", "w1"))
    left, kept = int(killed.stdout.readline()), int(live.stdout.readline())
    killed.kill()
    killed.wait()
    assert process_group._member(left)
    assert process_group.stop_orphans(f"{tmp_path}:w0", grace_s=5.0) == [left]
    assert not process_group._member(left)
    assert process_group.stop_orphans(f"{tmp_path}:w1", grace_s=5.0) == [] and process_group._member(kept)
    live.kill()
    live.wait()
    process_group.stop_orphans(f"{tmp_path}:w1", grace_s=5.0)


def test_a_hung_unit_is_stopped_at_its_deadline_and_a_failed_one_keeps_its_cause(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_worker, "RUN_CELL", tmp_path / "hang.py")
    (tmp_path / "hang.py").write_text("import time\ntime.sleep(60)\n")
    start = time.time()
    outcome = eval_worker.run_unit(_units(1)[0], tmp_path / "u" / "a1", _Server(), "gr00t", deadline=1.0)
    assert time.time() - start < 30 and "deadline" in outcome["error"]
    (tmp_path / "fail.py").write_text("import pathlib, sys\n"
                                      "pathlib.Path(sys.argv[sys.argv.index('--output') + 1]).mkdir(parents=True)\n"
                                      "(pathlib.Path(sys.argv[sys.argv.index('--output') + 1]) / 'run_manifest.json').write_text('{}')\n"
                                      "sys.exit('simulator state is not finite')\n")
    monkeypatch.setattr(eval_worker, "RUN_CELL", tmp_path / "fail.py")
    outcome = eval_worker.run_unit(_units(1)[0], tmp_path / "u" / "a2", _Server(), "gr00t", deadline=30.0)
    assert outcome["rc"] == 1 and outcome["complete"] is False
    assert outcome["log_tail"][-1] == "simulator state is not finite"


def test_a_unit_refused_before_it_runs_goes_back_untried_and_stops_the_worker(tmp_path, monkeypatch):
    """A refusal (no run manifest written) is the tree's, rows' or model's, the same for every unit: it must not spend
    each unit's attempts in seconds (2026-11-02: an R2 launch on a tree with no R2 rows)."""
    monkeypatch.setattr(eval_worker, "RUN_CELL", tmp_path / "refuse.py")
    (tmp_path / "refuse.py").write_text("import sys\nsys.exit('Refusal: rows not under the tree')\n")
    queue_dir.init(tmp_path / "queue", _units(3))
    with pytest.raises(SystemExit, match="refused"):
        eval_worker.run_worker(tmp_path, "w", _Server(), "gr00t")
    assert queue_dir.status(tmp_path / "queue") == {"todo": 3, "claimed": 0, "done": 0, "failed": 0}
    assert all(json.loads(p.read_text())["attempts"] == 0 for p in (tmp_path / "queue" / "todo").iterdir())


def _launch(tmp_path, units, config=None):
    output = tmp_path / "launch"
    output.mkdir()
    config = config or {"model": {"key": "gr00t", "checkpoint": "/a"}, "tasks": ["t"], "plan": {}, "version": "v1-plus",
                        "servers": {"recipe": {"command": "old"}}, "backend": {"kind": "local", "gpus": [0, 1]}}
    (output / "plan.json").write_text(json.dumps({"config": config, "tasks": {"t": {}}, "units": units}))
    queue_dir.init(output / "queue", units)
    return output, config


def test_scoring_reads_only_accepted_attempts_and_reports_the_failed_ones_before_them(tmp_path):
    output, _ = _launch(tmp_path, _units(2))
    path, first = queue_dir.claim(output / "queue", "w")
    failed = {"rc": 1, "output": "/units/t.train.000/first", "log_tail": ["Traceback ...", "simulator state is not finite"]}
    queue_dir.finish(output / "queue", path, failed, ok=False)
    path, record = queue_dir.claim(output / "queue", "w")
    queue_dir.finish(output / "queue", path, {"rc": 0, "output": f"/u/{record['attempt']}"}, ok=True)
    assert launch.accepted_outputs(output) == {"t": {"train": [f"/u/{record['attempt']}"]}}
    (retried,) = launch.retried_units(output)["t"]
    assert retried["unit"] == "t.train.000" and retried["accepted"] == record["attempt"]
    (attempt,) = retried["failed_attempts"]
    assert attempt["attempt"] == first["attempt"] and attempt["cause"] == "rc 1; last line: simulator state is not finite"
    with pytest.raises(SystemExit, match="1 of 2 planned units are not done"):
        launch.score(output, partial=False)
    done = next((output / "queue" / "done").iterdir())
    broken = json.loads(done.read_text())
    broken["outcomes"][-1].pop("output")
    done.write_text(json.dumps(broken))
    with pytest.raises(SystemExit, match="no single output"):
        launch.accepted_outputs(output)


def test_scoring_names_a_cell_whose_accepted_units_ran_more_than_one_code_revision(tmp_path):
    """--resume may take a newer tree; the scorer only lists the revisions."""
    directories = []
    for index, revision in enumerate(("a" * 40, "b" * 40, "b" * 40)):
        directory = tmp_path / f"u{index}"
        directory.mkdir()
        (directory / "run_manifest.json").write_text(json.dumps({"code": {"revision": revision}}))
        directories.append(str(directory))
    assert launch.code_revisions({"train": directories, "ga_s": directories[1:]}) == {"train": ["a" * 40, "b" * 40]}


def test_a_resume_takes_the_new_recipe_queues_what_is_missing_and_refuses_another_plan(tmp_path):
    """Review 2026-11-02: workers read the recipe from plan.json, so a resume must write the fixed one there."""
    output, config = _launch(tmp_path, _units(3))
    next((output / "queue" / "todo").iterdir()).unlink()
    fixed = {**config, "servers": {"recipe": {"command": "fixed"}}}
    launch._resume(fixed, output, retry_failed=False, requeue_claims=False, max_attempts=2)
    plan = json.loads((output / "plan.json").read_text())
    assert plan["config"]["servers"]["recipe"]["command"] == "fixed"
    assert queue_dir.status(output / "queue")["todo"] == 3
    with pytest.raises(SystemExit, match="another model"):
        launch._resume({**config, "model": {"key": "rldx11", "checkpoint": "/x"}}, output, False, False, 2)


def test_the_local_backend_runs_until_every_unit_ends_and_resumes(tmp_path, monkeypatch):
    """One worker per GPU, each a process; a launch stopped half way finishes on --resume without rerunning a unit.
    The servers' saved environment (it may hold tokens) reaches each worker in a file of its own, gone once read."""
    (tmp_path / "stand_in_worker.py").write_text(
        "import sys, os\n"
        "from eval_launch import queue_dir, server_env\n"
        "args = dict(zip(sys.argv[1::2], sys.argv[2::2]))\n"
        "token = server_env.read(args['--server-env'])['HF_TOKEN']\n"
        "queue = args['--launch'] + '/queue'\n"
        "while (claimed := queue_dir.claim(queue, args['--worker'])):\n"
        "    outcome = {'rc': 0, 'gpu': os.environ['CUDA_VISIBLE_DEVICES'], 'token': token,\n"
        "               'file_left': os.path.exists(args['--server-env'])}\n"
        "    queue_dir.finish(queue, claimed[0], outcome, ok=True)\n"
        "    if os.environ.get('STOP_AFTER_ONE'): sys.exit(1)\n")
    monkeypatch.setattr(launch, "WORKER_MODULE", "stand_in_worker")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, (str(tmp_path), os.environ.get("PYTHONPATH")))))
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    output, config = _launch(tmp_path, _units(6))
    config = {**config, "backend": {**config["backend"], "local": {"restarts": 0}}}
    monkeypatch.setenv("STOP_AFTER_ONE", "1")
    assert launch.run_local(config, output, {"HF_TOKEN": "secret"}, poll_s=0.2) == 1
    assert queue_dir.status(output / "queue")["done"] == 2
    monkeypatch.delenv("STOP_AFTER_ONE")
    launch._resume(config, output, retry_failed=False, requeue_claims=True, max_attempts=2)
    assert launch.run_local(config, output, {"HF_TOKEN": "secret"}, poll_s=0.2) == 0
    records = [json.loads(p.read_text()) for p in (output / "queue" / "done").iterdir()]
    assert len(records) == 6 and all(len(r["outcomes"]) == 1 for r in records)
    assert {r["outcomes"][0]["gpu"] for r in records} <= {"0", "1"}
    assert {(r["outcomes"][0]["token"], r["outcomes"][0]["file_left"]) for r in records} == {("secret", False)}
    assert not list((tmp_path / "tmp").iterdir())


def test_the_launcher_removes_the_saved_environment_as_it_starts(tmp_path):
    """launch.sh's file holds the shell's tokens; it was never deleted, one more in /tmp per run.
    Removed even when the arguments are refused."""
    output, config = _launch(tmp_path, _units(1))
    path = tmp_path / "eval.yaml"
    path.write_text(json.dumps({**config, "output": str(output)}))
    for argv in (["--status"], ["--no-such-mode"]):
        saved = tmp_path / "server.env"
        saved.write_text("HF_TOKEN=secret\0")
        try:
            launch.main([str(path), *argv, "--server-env", str(saved)])
        except SystemExit:
            pass
        assert not saved.exists(), argv


def test_a_stop_while_the_local_workers_start_stops_the_ones_started(tmp_path, monkeypatch):
    """SIGTERM right after the first worker's process is created, before it is recorded: run_local stops it and starts
    no other (the first worker went on unwatched)."""
    (tmp_path / "sleeping_worker.py").write_text("import time\ntime.sleep(300)\n")
    monkeypatch.setattr(launch, "WORKER_MODULE", "sleeping_worker")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, (str(tmp_path), os.environ.get("PYTHONPATH")))))
    started = []
    real_popen = subprocess.Popen

    def popen_then_sigterm(*args, **kwargs):
        started.append(real_popen(*args, **kwargs))
        os.kill(os.getpid(), signal.SIGTERM)
        return started[-1]

    monkeypatch.setattr(launch.subprocess, "Popen", popen_then_sigterm)
    output, config = _launch(tmp_path, _units(2))
    previous = signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)
    try:
        assert launch.run_local(config, output, None, poll_s=0.2) == 130
    finally:
        signal.signal(signal.SIGTERM, previous[0])
        signal.signal(signal.SIGINT, previous[1])
    assert len(started) == 1
    assert not process_group.group_alive(started[0].pid)


def test_a_sigterm_while_a_unit_is_claimed_hands_that_unit_back(tmp_path, monkeypatch):
    """SIGTERM between the claim's publication and the worker recording it."""
    queue_dir.init(tmp_path / "queue", _units(1))
    real_claim = queue_dir.claim

    def claim_then_sigterm(*args, **kwargs):
        claimed = real_claim(*args, **kwargs)
        os.kill(os.getpid(), signal.SIGTERM)
        return claimed

    monkeypatch.setattr(eval_worker.queue_dir, "claim", claim_then_sigterm)
    previous = signal.signal(signal.SIGTERM, eval_worker._stop_on_sigterm)
    try:
        with pytest.raises(SystemExit):
            eval_worker.run_worker(tmp_path, "w", _Server(), "gr00t", run=lambda *a: pytest.fail("ran the unit"))
    finally:
        signal.signal(signal.SIGTERM, previous)
    assert queue_dir.status(tmp_path / "queue") == {"todo": 1, "claimed": 0, "done": 0, "failed": 0}


def test_sigterm_hands_the_unit_back_before_anything_else(tmp_path):
    """2026-11-02 rehearsal: a gracefully deleted pod's unit waited out the stale timeout, because the worker spent
    the pod's grace period stopping its simulator before it handed the unit back."""
    queue_dir.init(tmp_path / "queue", _units(1))

    def run(unit, *args):
        eval_worker._stop_on_sigterm(signal.SIGTERM, None)

    with pytest.raises(SystemExit):
        eval_worker.run_worker(tmp_path, "w", _Server(), "gr00t", run=run)
    assert queue_dir.status(tmp_path / "queue") == {"todo": 1, "claimed": 0, "done": 0, "failed": 0}
    assert json.loads(next((tmp_path / "queue" / "todo").iterdir()).read_text())["attempts"] == 0
