# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""The launch plan: every row once, any task the manifest has, the reasons a task is skipped, and the unit cutting."""

from __future__ import annotations

import copy
import json
import re

import pytest
import yaml

from dexbench_arena.benchmark.models import MODELS_BY_KEY, ModelEntry

from eval_launch import plan
from eval_launch.tests.example_cluster import filled

MANIFEST = json.loads(plan.MANIFEST.read_text())
TASKS = list(MANIFEST["tasks"])


def _config(**changes):
    return {"model": {"key": "gr00t", "checkpoint": "/ckpt"}, "tasks": "all", "backend": {"gpus": 1}, **changes}


def _manifest(tmp_path, manifest):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def _rows(task, cell, version="v1"):
    """A cell's rows at ``version``: v1 half of every 120-row cell, v1-plus all of them."""
    return sum(plan.version_counts(MANIFEST, task, cell, version).values())


@pytest.mark.parametrize("version", ["v1", "v1-plus"])
def test_a_plan_of_every_task_runs_every_row_of_every_cell_once(version):
    result = plan.make_plan(_config(version=version))
    runnable = [task for task, state in result["tasks"].items() if state["runnable"]]
    assert runnable, "no task on the spec is runnable"
    expected = sum(_rows(task, cell, version) for task in runnable for cell in MANIFEST["tasks"][task]["cells"])
    assert sum(unit["episodes"] for unit in result["units"]) == expected
    assert result["eval_version"] == version and {unit["version"] for unit in result["units"] if unit["cell"]} == {version}
    plan.check_coverage(MANIFEST, result["units"])
    costs = [unit["expected_s"] for unit in result["units"]]
    assert costs == sorted(costs, reverse=True), "units go longest first"


def test_a_task_added_to_the_spec_is_planned_with_no_launcher_change(tmp_path, monkeypatch):
    manifest = copy.deepcopy(MANIFEST)
    source = TASKS[0]
    manifest["tasks"]["new_task"] = copy.deepcopy(manifest["tasks"][source])
    pinned_version_rows = plan.pinned_version_rows
    monkeypatch.setattr(plan, "pinned_version_rows", lambda manifest, task, cell, version: pinned_version_rows(
        MANIFEST, source if task == "new_task" else task, cell, version))
    result = plan.make_plan(_config(), _manifest(tmp_path, manifest), pinned=lambda task, section: True)
    added = [unit for unit in result["units"] if unit["task"] == "new_task"]
    assert sum(unit["episodes"] for unit in added) == sum(_rows(source, cell) for cell in MANIFEST["tasks"][source]["cells"])


def test_a_task_that_cannot_run_yet_is_skipped_with_its_reason(tmp_path):
    manifest = copy.deepcopy(MANIFEST)
    pending, unpinned = TASKS[0], TASKS[1]
    manifest["tasks"][pending]["rows_pending"] = "rows not recorded yet"
    result = plan.make_plan(_config(tasks=[pending, unpinned, "not_a_task"]), _manifest(tmp_path, manifest),
                            pinned=lambda task, section: task != unpinned)
    reasons = {task: state.get("reason", "") for task, state in result["tasks"].items()}
    assert "rows pending" in reasons[pending]
    assert "no pinned frozen set" in reasons[unpinned]
    assert "not on the benchmark spec" in reasons["not_a_task"]
    assert result["units"] == []


def test_a_long_run_is_cut_into_slices_and_short_runs_share_a_unit():
    costs = plan.costs_from(_config())
    units = plan._cell_units("t", "train", {"long": 30, "a": 1, "b": 1, "c": 1}, episode_s=300.0, build_s=120.0,
                             costs=costs, unit_s=1300.0)
    long_units = [unit for unit in units if "long" in unit["slices"]]
    assert len(long_units) > 1 and all(len(unit["slices"]) == 1 for unit in long_units)
    assert all((end - start) * 300.0 >= costs.min_slice_builds * 120.0
               for unit in long_units for start, end in unit["slices"].values())
    (short,) = [unit for unit in units if "long" not in unit["slices"]]
    assert set(short["slices"]) == {"a", "b", "c"}
    tight = plan._cell_units("t", "train", {"a": 1, "b": 1, "c": 1}, episode_s=300.0, build_s=120.0, costs=costs,
                             unit_s=900.0)
    assert [len(unit["slices"]) for unit in tight] == [2, 1], "a group closes before a Run would take it past the target"


def test_more_gpus_than_units_cuts_the_units_smaller():
    runnable = [task for task, state in plan.make_plan(_config())["tasks"].items() if state["runnable"]]
    task = min(runnable, key=lambda name: len(plan.make_plan(_config(tasks=[name]))["units"]))
    few = plan.make_plan(_config(tasks=[task], backend={"gpus": 1}))
    assert len(few["units"]) < 80
    many = plan.make_plan(_config(tasks=[task], backend={"gpus": 80}))
    assert len(many["units"]) > len(few["units"])
    plan.check_coverage(MANIFEST, many["units"])


def test_a_checkpoint_map_names_every_task_it_runs():
    config = {"model": {"key": "gr00t", "checkpoints": {TASKS[0]: "/a"}}, "tasks": TASKS[:2], "backend": {"gpus": 1}}
    with pytest.raises(ValueError, match="names no checkpoint"):
        plan.make_plan(config)
    config["model"]["checkpoints"][TASKS[1]] = "/b"
    result = plan.make_plan(config)
    assert {unit["checkpoint"] for unit in result["units"] if unit["task"] == TASKS[1]} == {"/b"}


def test_cells_as_a_list_or_per_task_plan_the_same_rows():
    task = TASKS[0]
    cell = next(iter(MANIFEST["tasks"][task]["cells"]))
    for cells in ([cell], {task: [cell]}):
        units = plan.make_plan(_config(tasks=[task], cells=cells))["units"]
        assert {unit["cell"] for unit in units} == {cell}
    with pytest.raises(ValueError, match="cells is all"):
        plan.make_plan(_config(tasks=[task], cells=cell))


def test_tasks_added_to_a_plan_go_behind_its_units_and_a_planned_task_is_refused():
    first = plan.make_plan(_config(tasks=[TASKS[0]]))
    grown = plan.add_to_plan(first, _config(), [TASKS[1]])
    assert grown["units"][:len(first["units"])] == first["units"]
    assert {unit["task"] for unit in grown["units"][len(first["units"]):]} == {TASKS[1]}
    with pytest.raises(ValueError, match="already in the plan"):
        plan.add_to_plan(grown, _config(), [TASKS[0]])


def test_a_pilot_model_or_a_sweep_is_refused_before_anything_is_queued(tmp_path, monkeypatch):
    """A pilot runs the first rows of each cell (``ModelEntry.rows``), which explicit slices cannot express; a sweep
    unit has no cell to run (review 2026-11-02)."""

    pilot = next(key for key, entry in MODELS_BY_KEY.items() if entry.rows is not None)
    with pytest.raises(ValueError, match="pilot"):
        plan.make_plan(_config(model={"key": pilot, "checkpoint": "/c"}))
    path = tmp_path / "eval.yaml"
    path.write_text(json.dumps({**_config(), "sweep": True}))
    with pytest.raises(ValueError, match="sweep"):
        plan.load_config(path)


def test_a_model_the_workers_could_not_point_at_their_servers_is_refused(monkeypatch):
    """No remote_url or remote_port in the block and none in the recipe's client_set."""
    entry = ModelEntry(key="noaddress", label="", note="", policy={"type": "x", "policy_variant": "noaddress"})
    monkeypatch.setitem(MODELS_BY_KEY, "noaddress", entry)
    config = _config(model={"key": "noaddress", "checkpoint": "/c"}, tasks=TASKS[:1])
    with pytest.raises(ValueError, match="no remote_url or remote_port"):
        plan.make_plan(config)
    config["servers"] = {"recipe": {"client_set": ["remote_url=http://127.0.0.1:{port}"]}}
    assert plan.make_plan(config)["units"]


MY_MODEL = {"key": "mymodel", "label": "My Model", "checkpoint": "/c",
            "policy": {"type": "dexbench_arena_cosmos.policy.cosmos3_joint_policy.Cosmos3JointPolicy",
                       "remote_url": "http://127.0.0.1:5811", "policy_device": "cpu"}}


def test_a_model_the_config_registers_is_planned_without_a_registry_entry(tmp_path):
    """Config-only registration: a key of its own, its client's block; reported apart (group community),
    never the headline, and never under a registered model's key."""
    path = tmp_path / "eval.yaml"
    path.write_text(json.dumps(_config(model=MY_MODEL, tasks=TASKS[:1])))
    config = plan.load_config(path)
    assert plan.model_spec(config) == {key: MY_MODEL[key] for key in ("key", "label", "policy")}
    entry = plan.model_entry(config)
    assert entry.group == "community" and entry.policy["policy_variant"] == "mymodel" and entry.horizon is not None
    assert plan.make_plan(config)["units"]
    assert plan.model_spec(_config()) == "gr00t", "a registered model goes by its key alone"
    for model, refused in (({**MY_MODEL, "key": "gr00t"}, "registered"), ({**MY_MODEL, "group": "headline"}, "headline"),
                           ({**MY_MODEL, "policy": {**MY_MODEL["policy"], "policy_variant": "x"}}, "policy_variant"),
                           ({**MY_MODEL, "server": "x"}, "not \\['server'\\]")):
        path.write_text(json.dumps(_config(model=model, tasks=TASKS[:1])))
        with pytest.raises(ValueError, match=refused):
            plan.load_config(path)


def test_measured_costs_come_only_from_the_config():
    assert plan.load_timings(_config()) is None
    measured = plan.load_timings(_config(plan={"timings": "timings/l40s_example.json"}))
    assert measured["tasks"]
    costs = plan.make_plan(_config())["tasks"]
    assert all("config default" in state["cost_source"] for state in costs.values() if state["runnable"])



@pytest.mark.parametrize("path", sorted((plan.PACKAGE_DIR / "examples").glob("*.yaml")), ids=lambda path: path.name)
def test_every_example_config_loads_and_its_recipe_fills_in(path):
    config = plan.load_config(path)
    plan.eval_version(config)
    if (config.get("servers") or {}).get("recipe"):
        plan.check_recipe(config)


def test_the_retired_instructions_key_is_refused_on_a_new_launch_and_ignored_on_a_resume(tmp_path, capsys):
    """Launches of 2026-11-04 were planned with ``instructions.training_tasks``: their configs must still
    resume, while a new launch naming it is told the check is gone."""
    from eval_launch import launch
    config = filled(plan.load_config(plan.PACKAGE_DIR / "examples/kubernetes.yaml"))
    config = {**config, "output": str(tmp_path / "launch"),
              "instructions": {"training_tasks": "/shared/run/meta/tasks.jsonl"}}
    path = tmp_path / "eval.yaml"
    path.write_text(yaml.safe_dump(config))
    for mode in (["--plan-only"], ["--start"], ["--resume"], ["--k8s-job", str(tmp_path / "job")]):
        with pytest.raises(SystemExit):
            launch.main([str(path), *mode])
        assert "remove instructions" in capsys.readouterr().err
    assert not (tmp_path / "job").exists()
    assert launch.main([str(path), "--k8s-job", str(tmp_path / "job"), "--existing-plan"]) == 0
    assert "ignoring instructions" in capsys.readouterr().out
    assert "instructions" not in yaml.safe_load((tmp_path / "job/eval.yaml").read_text())
    stored = {"config": {key: config.get(key) for key in launch.PLANNING_KEYS} | {"instructions": config["instructions"]},
              "eval_version": "v1", "units": [], "tasks": {}, "instructions": {"*": ["Pick the can."]}}
    (tmp_path / "launch").mkdir()
    (tmp_path / "launch/plan.json").write_text(json.dumps(stored))
    assert launch.main([str(path), "--resume"]) == 0
    assert "ignoring instructions" in capsys.readouterr().out
    resumed = json.loads((tmp_path / "launch/plan.json").read_text())
    assert "instructions" not in resumed and "instructions" not in resumed["config"]


def _pre_version_plan(tasks):
    """A plan as the launcher wrote it before versions: every row, and no eval_version, config version or unit version."""
    made = plan.make_plan(_config(tasks=tasks, version="v1-plus"))
    made.pop("eval_version")
    made["config"].pop("version")
    for unit in made["units"]:
        unit.pop("version")
    return made


def test_a_plan_from_before_versions_resumes_and_takes_tasks_under_version_v1_plus(tmp_path):
    """Such a plan ran every row: a config saying ``version: v1-plus`` resumes it and adds tasks at v1-plus, and one
    that omits the version (v1) is refused before anything is planned."""
    from eval_launch import launch
    output = tmp_path / "launch"
    output.mkdir()
    (output / "plan.json").write_text(json.dumps(_pre_version_plan([TASKS[0]])))
    with pytest.raises(SystemExit, match=r"another version \(v1-plus, the config asks for v1\)"):
        launch._start(_config(), output, [TASKS[1]], plan.MANIFEST, None)
    grown = launch._start(_config(version="v1-plus"), output, [TASKS[1]], plan.MANIFEST, None)
    added = [unit for unit in grown["units"] if unit["task"] == TASKS[1]]
    assert added and {unit["version"] for unit in added} == {"v1-plus"}
    assert sum(unit["episodes"] for unit in grown["units"]) == sum(
        _rows(task, cell, "v1-plus") for task in TASKS[:2] for cell in MANIFEST["tasks"][task]["cells"])
    plan.check_coverage(MANIFEST, grown["units"])


@pytest.mark.parametrize("planned, resumed", [({}, {"version": "v1"}), ({"version": "v1"}, {})],
                         ids=["omitted-then-explicit", "explicit-then-omitted"])
def test_v1_written_out_or_left_to_the_default_is_the_same_plan(tmp_path, planned, resumed):
    from eval_launch import launch
    output = tmp_path / "launch"
    output.mkdir()
    first = launch._start(_config(tasks=[TASKS[0]], **planned), output, None, plan.MANIFEST, None)
    grown = launch._start(_config(**resumed), output, [TASKS[1]], plan.MANIFEST, None)
    assert grown["units"][:len(first["units"])] == first["units"]
    assert {unit["version"] for unit in grown["units"]} == {"v1"}


def test_a_launch_on_a_tree_without_a_planned_tasks_inputs_is_refused_before_anything_is_written(tmp_path, monkeypatch,
                                                                                                  staged_tree, capsys):
    """A tree staged with the rows but not DB-H3's input package had every H3 unit fail at its start (2026-11-08): the
    planner refuses such a tree, naming the file and the command that installs it."""
    import shutil

    from eval_launch import launch

    tree = tmp_path / "tree"
    shutil.copytree(staged_tree, tree)
    h3_inputs = json.loads(plan.MANIFEST.read_text())["tasks"]["h3"]["runtime_inputs"]
    for path in h3_inputs:
        (tree / path).unlink()
    monkeypatch.setattr(launch, "TREE", tree)
    path = tmp_path / "eval.yaml"
    path.write_text(json.dumps({**_config(tasks=["h3"], output=str(tmp_path / "launch")),
                                "servers": {"recipe": {"command": "serve --port {port}"}}}))
    with pytest.raises(SystemExit, match="db_h3_inputs/manifest.json.*--eval-conditions"):
        launch.main([str(path), "--start"])
    assert not (tmp_path / "launch" / "plan.json").exists()
    assert launch.main([str(path), "--plan-only"]) == 0 and "--eval-conditions" in capsys.readouterr().out
    for missing in h3_inputs:
        (tree / missing).parent.mkdir(parents=True, exist_ok=True)
        (tree / missing).touch()
    assert launch.main([str(path), "--start"]) == 0


def test_a_kubernetes_launch_is_checked_against_its_staged_tree_before_the_job_is_written(tmp_path, staged_tree,
                                                                                        capsys):
    """--plan-only and --k8s-job check the tree the pods will read (backend.k8s.tree_dir), not this checkout, when this
    host sees it: --k8s-job refuses a tree that lacks a planned task's input before any GPU is held, and writes nothing.
    A tree_dir this host does not see is said so."""
    import shutil

    from eval_launch import PACKAGE_DIR, launch

    tree = tmp_path / "volume/trees/abc"
    shutil.copytree(staged_tree, tree)
    h3_inputs = json.loads(plan.MANIFEST.read_text())["tasks"]["h3"]["runtime_inputs"]
    for path in h3_inputs:
        (tree / path).unlink()
    config = filled(plan.load_config(PACKAGE_DIR / "examples/kubernetes.yaml"))
    config.update(tasks=["h3"], output=str(tmp_path / "volume/launch"))
    config["backend"]["k8s"]["tree_dir"] = str(tree)
    path = tmp_path / "eval.yaml"
    path.write_text(json.dumps(config))
    assert launch.main([str(path), "--plan-only"]) == 0
    assert f"note: {tree} lacks 1 files" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        launch.main([str(path), "--k8s-job", str(tmp_path / "job")])
    assert re.search(r"db_h3_inputs/manifest.json.*--eval-conditions", capsys.readouterr().err)
    assert not (tmp_path / "job").exists()
    for path_ in h3_inputs:
        (tree / path_).parent.mkdir(parents=True, exist_ok=True)
        (tree / path_).touch()
    (tree / ".source-commit").write_text("0" * 40 + "\n")
    with pytest.raises(SystemExit):
        launch.main([str(path), "--k8s-job", str(tmp_path / "job")])
    assert "is staged at 0000" in capsys.readouterr().err and not (tmp_path / "job").exists()
    (tree / ".source-commit").write_text(config["backend"]["k8s"]["commit"] + "\n")
    assert launch.main([str(path), "--k8s-job", str(tmp_path / "job")]) == 0
    config["backend"]["k8s"]["tree_dir"] = str(tmp_path / "elsewhere")
    path.write_text(json.dumps(config))
    assert launch.main([str(path), "--plan-only"]) == 0
    assert "is not visible here" in capsys.readouterr().out


def test_a_model_block_that_describes_a_model_without_registering_one_is_refused(tmp_path):
    """Without policy, model.key names the registered model, which runs under its own label and group; a label, note or
    group the config states would be dropped without a word (a fine-tuned GR00T reported as the headline one)."""
    path = tmp_path / "eval.yaml"
    path.write_text(json.dumps(_config(model={"key": "gr00t", "label": "Our fine-tuned GR00T", "group": "community",
                                              "checkpoint": "/ours"})))
    with pytest.raises(ValueError, match="model.group, model.label describe a model this config registers"):
        plan.load_config(path)
    path.write_text(json.dumps(_config(model={"key": "gr00t", "checkpoint": "/ours"})))
    assert plan.model_entry(plan.load_config(path)).key == "gr00t"


@pytest.mark.parametrize("policy, message", [
    ({"type": "a.B", "remote_api_token": "literal-secret"}, "remote_api_token is written"),
    ({"type": "a.B", "remote_api_token": "${oc.env:DEXBENCH_POLICY_API_TOKEN,literal-secret}"}, "remote_api_token is"),
    ({"type": "a.B", "remote_api_token": "${oc.env:DEXBENCH_POLICY_API_TOKEN,null}literal"}, "remote_api_token is"),
    ({"type": "a.B", "remote_api_token": "${oc.env:MY_TOKEN}"}, "remote_api_token is"),
    (["a.B"], "policy is the client's block"),
])
def test_a_config_models_client_block_is_refused_when_it_would_leak_a_token_or_is_not_a_mapping(tmp_path, policy,
                                                                                                 message):
    path = tmp_path / "eval.yaml"
    path.write_text(json.dumps(_config(model={"key": "mine", "checkpoint": "/c", "policy": policy})))
    with pytest.raises(ValueError, match=message):
        plan.load_config(path)
    path.write_text(json.dumps(_config(model={"key": "mine", "checkpoint": "/c", "policy": {
        "type": "a.B", "remote_api_token": "${oc.env:DEXBENCH_POLICY_API_TOKEN,null}"}})))
    assert plan.model_entry(plan.load_config(path)).policy["remote_api_token"].startswith("${oc.env:")


def test_a_relative_output_is_this_hosts_directory_from_wherever_the_launcher_runs(tmp_path, monkeypatch):
    """The cell runner starts in the tree's root with --output under the launch's output: a relative output (the
    examples' outputs/eval/...) named another directory there when the launcher ran from elsewhere. A Kubernetes
    launch's output is the pods' path and stays as given."""
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "eval.yaml"
    path.write_text(json.dumps(_config(output="outputs/eval/run")))
    assert plan.load_config(path)["output"] == str(tmp_path / "outputs/eval/run")
    path.write_text(json.dumps(_config(output="launch", backend={"kind": "k8s", "gpus": 1})))
    assert plan.load_config(path)["output"] == "launch"


def test_a_missing_timings_file_is_refused_before_a_kubernetes_job_is_written(tmp_path, capsys):
    """Pod 0 loads plan.timings at --start; a file that is not there would fail every retry while the GPU pods wait."""
    from eval_launch import PACKAGE_DIR, launch

    config = filled(plan.load_config(PACKAGE_DIR / "examples/kubernetes.yaml"))
    config["plan"] = {**(config.get("plan") or {}), "timings": "timings/no_such_file.json"}
    path = tmp_path / "eval.yaml"
    path.write_text(json.dumps(config))
    for mode in (["--plan-only"], ["--k8s-job", str(tmp_path / "job")]):
        with pytest.raises(SystemExit) as stopped:
            launch.main([str(path), *mode])
        assert stopped.value.code == 2 and "no_such_file.json" in capsys.readouterr().err
    assert not (tmp_path / "job").exists()


@pytest.mark.parametrize("group", ["ablation", "trial", "headline", "outside"])
def test_a_config_model_takes_the_community_group_only(tmp_path, group):
    """One group for a model a config registers (the maintainers' rule, 2026-11-09): community. An ablation, a trial
    or a headline model is an entry in benchmark/models.py, reviewed with the manifest."""
    path = tmp_path / "eval.yaml"
    path.write_text(json.dumps(_config(model={"key": "mine", "checkpoint": "/c", "group": group,
                                              "policy": {"type": "a.B"}})))
    with pytest.raises(ValueError, match="group is community"):
        plan.load_config(path)
    path.write_text(json.dumps(_config(model={"key": "mine", "checkpoint": "/c", "group": "community",
                                              "policy": {"type": "a.B"}})))
    assert plan.model_entry(plan.load_config(path)).group == "community"


def test_spaced_tasks_reach_a_local_launchs_driver_whole(tmp_path, monkeypatch):
    """--local --tasks "r2, r3": the launcher plans both, and so must the driver it execs into (it used to split the
    raw string, so " r3" was skipped and only r2 ran)."""
    from eval_launch import driver, launch, local_backend

    path = tmp_path / "eval.yaml"
    path.write_text(json.dumps(_config(output=str(tmp_path / "launch"), backend={"kind": "local", "gpus": 1},
                                       servers={"recipe": {"command": "serve --port {port}"}})))
    hop: list = []

    def no_exec(executable, command, environment):
        hop.append(command)
        raise SystemExit(0)

    monkeypatch.setattr(launch.os, "execve", no_exec)
    with pytest.raises(SystemExit):
        launch.main([str(path), "--local", "--tasks", "r2, r3"])
    (command,) = hop
    assert command[command.index("--tasks") + 1] == "r2,r3"
    planned: list = []

    def no_plan(config, output, tasks, *rest):
        planned.append(tasks)
        raise SystemExit(0)

    monkeypatch.setattr(driver, "write_plan", no_plan)
    with pytest.raises(SystemExit):
        local_backend.main([str(path), "--tasks", "r2, r3"])
    assert planned == [["r2", "r3"]]


def test_submission_recording_is_preserved_in_plan_units():
    config = _config(submission=True)
    config["model"]["checkpoint_sha256"] = "b" * 64
    result = plan.make_plan(config)
    assert result["units"] and all(unit["submission"] and unit["checkpoint_sha256"] == "b" * 64 for unit in result["units"])
    with pytest.raises(ValueError, match="cannot change submission"):
        plan.add_to_plan(result, {**config, "submission": False}, ["m1"])
