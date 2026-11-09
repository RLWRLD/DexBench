# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""The scorer and the frozen sets, and the benchmark's files: published without clobbering, with a recorded draw
that is numbers or nothing. Nothing here imports Isaac Lab or Arena."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from functools import partial
from pathlib import Path

import pytest

from dexbench_arena.benchmark import Refusal, frozen
from dexbench_arena.benchmark.models import MODELS_BY_KEY, REQUIRED_MODELS
from dexbench_arena.benchmark.protocol import cell_rows, version_rows
from dexbench_arena.evaluation.artifacts import is_finite, jsonl_bytes, numeric_draw, publish_new

REPO = Path(__file__).resolve().parents[3]
CONFIGS = REPO / "scripts" / "evaluation" / "experiment_configs" / "benchmark"
MANIFEST = json.loads((CONFIGS / "benchmark_manifest.json").read_text())
PINNED_DIR = frozen.PINNED_DIR


def _tool(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _scorer():
    pytest.importorskip("scipy", reason="the scorer's Beta quantiles need scipy")
    return _tool("score_benchmark_cells", REPO / "scripts/evaluation/score_benchmark_cells.py")


DRAWER = {"axis": "m1_drawer", "key": "lg_fh25eae_detergent_drawer", "fit": {"scene_layout": "tabletop"}}


def _source_row(cell: str, index: int) -> dict:
    """Frozen row ``index`` of an M1 cell: the object and a draw unique to the cell and index."""
    yaw = 0.0 if cell == "train" else 0.11
    return {"task_object": DRAWER, "variations": {
        "drawer.pose": [0.0, index * 1e-4, 0.0], "washer.pose": [0.0, 0.0, 0.0],
        "dexbench_rby1_wuji2_joint.base_pose": [0.0, 0.0, yaw], "drawer.mass": [0.55],
    }}


@pytest.fixture(autouse=True)
def _pins(tmp_path, monkeypatch):
    """Each test's frozen sets stand in for the checked-in pins."""
    monkeypatch.setattr(frozen, "PINNED_DIR", tmp_path / "pins")


def _pin(record: dict, section: str = "headline", manifest: dict = MANIFEST):
    """Pin ``record`` as M1's ``section`` set; a headline set gets its version rows pinned beside it, as freeze writes them."""
    target = frozen.pinned_path("m1", section)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record))
    if section == "headline" and record:
        frozen.version_pin_path("m1").write_bytes(frozen.version_pin_bytes(frozen.version_pin(manifest, "m1", target.read_bytes())))
    return target


def _frozen_rows(tmp_path):
    """Write M1's frozen headline rows under ``tmp_path`` (the runner's directory) and pin them; returns the pin."""
    for cell, cell_entry in MANIFEST["tasks"]["m1"]["cells"].items():
        for row in cell_entry["rows"].values():
            path = tmp_path / row["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(json.dumps(_source_row(cell, index)) + "\n" for index in range(row["rows"])))
    return _pin(frozen.freeze(MANIFEST, "m1", tmp_path))


def _ran_as_declared(cell_entry: dict, row: str, model: str) -> dict:
    """What a model's Run of the cell records when it runs as declared: its environment fields and settings."""
    environment = {**cell_entry["declared"], **cell_entry["rows"][row]["environment"], "record_dataset": True}
    execution = next((task["policy_execution"] for task in MANIFEST["tasks"].values()
                      if any(cell["eval"] == cell_entry["eval"] for cell in task["cells"].values())), None)
    policy = {"policy_variant": model, "action_chunk_length": MANIFEST["execution_horizon"],
              **((execution or {}).get("policy") or {}),
              **{field: MODELS_BY_KEY[model].policy[field] for field in ("relative_joint_slices", "relative_action_groups", "denoising_timesteps")
                 if field in MODELS_BY_KEY[model].policy}}
    return {"environment_cfg": environment, "evaluation_settings": {
        "version": 3, "physics": {"type": "isaaclab_newton.physics.newton_manager_cfg.NewtonCfg"},
        "control": {"sim_dt": 1 / 240, "decimation": round(240 / (environment.get("rby1_control_hz") or 30))},
        "episode_length_s": environment["episode_length_s"],
        "terminations": {"time_out": {"func": "isaaclab.envs.mdp.time_out", "params": {}}}, "events": {},
        "code": {"revision": "abc123", "assets": "def456"}, "policy": policy}}


def _result_file(directory: Path) -> Path:
    return directory / "arena" / "arena_experiment_result.json"


def _results(tmp_path, entry, successes, frozen_sha256="unchecked"):
    """One run directory per cell and model, as run_benchmark_cell.py leaves it: its run manifest and the
    experiment runner's result under arena/. Every Run replays its frozen rows, which come back in reverse
    order (they finish out of order)."""
    results = []
    for cell, cell_entry in entry["cells"].items():
        for model in (key for key, declared in entry["models"].items()
                      if declared["available"] and declared.get("group") != "trial"):
            directory = tmp_path / "results" / f"{cell}_{model}"
            (directory / "arena").mkdir(parents=True)
            (directory / "run_manifest.json").write_text(json.dumps({
                "version": 1, "task": "m1", "section": "headline", "cell": cell, "model": model,
                "frozen_sha256": frozen_sha256, "arena_output": "arena", "checkpoint": f"{model}-step-1",
                "deployment": {}, "code": {"revision": "abc123"}, "cell_config_sha256": "c" * 64}))
            runs = {}
            for name, row in cell_entry["rows"].items():
                episodes = [{"success": index < successes[cell], "replay_source_record_index": index, **_source_row(cell, index),
                             **_ran_as_declared(cell_entry, name, model)} for index in reversed(range(row["rows"]))]
                runs[name] = {"policy_variant": model, "status": "completed", "rebuilds": [{"index": 0, "episodes": episodes}]}
            _result_file(directory).write_text(json.dumps({"runs": runs}))
            results.append((cell, directory))
    return results


def _edit_episodes(results, change, only=lambda directory: True):
    """Apply ``change`` to every episode of each result directory ``only`` picks; returns ``results``."""
    for _cell, directory in results:
        if not only(directory):
            continue
        path = _result_file(directory)
        data = json.loads(path.read_text())
        for run in data["runs"].values():
            for episode in run["rebuilds"][0]["episodes"]:
                change(episode)
        path.write_text(json.dumps(data))
    return results


def _refreeze(tmp_path, change_row):
    """Rewrite every frozen M1 row through ``change_row`` and freeze the set again."""
    for cell_entry in MANIFEST["tasks"]["m1"]["cells"].values():
        for row in cell_entry["rows"].values():
            path = tmp_path / row["path"]
            rows = [change_row(json.loads(line)) for line in path.read_text().splitlines()]
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    _pin(frozen.freeze(MANIFEST, "m1", tmp_path))


def _score(scorer, tmp_path, results, version="v1-plus"):
    digest, sources = frozen.load_frozen_set(frozen.pinned_path("m1"), MANIFEST, "m1", tmp_path)
    return scorer.score(MANIFEST, "m1", scorer.collect(MANIFEST, "m1", results), sources, digest, version)


def test_the_scorer_reports_each_cell_and_every_required_model(tmp_path):
    scorer = _scorer()
    entry = MANIFEST["tasks"]["m1"]
    _frozen_rows(tmp_path)
    results = _results(tmp_path, entry, {"train": 30, "ga_s": 12})
    gr00t = _score(scorer, tmp_path, results)["models"]["gr00t"]
    assert (gr00t["cells"]["train"]["k"], gr00t["cells"]["train"]["n"]) == (30, 120) and gr00t["headline"]
    assert gr00t["cells"]["train"]["objects"] == {DRAWER["key"]: {"k": 30, "n": 120}}
    assert gr00t["macro_sr"] == pytest.approx((30 / 120 + 12 / 120) / 2)
    partial = _score(scorer, tmp_path, [(cell, path) for cell, path in results if not path.name.endswith("_cosmos")])
    assert partial["status"] == "partial" and set(partial["models"]) == {model.key for model in REQUIRED_MODELS}
    assert partial["models"]["cosmos"]["missing_cells"] == ["ga_s", "train"] and partial["models"]["cosmos"]["available"]


def test_a_challenge_track_is_scored_beside_the_headline_and_never_in_it(tmp_path):
    """A task declaring a challenge (DB-L1's every bag, beside its two-bag headline) gets each model's k/n per cell of
    its rows' challenge_success, by the headline's interval convention; the headline is unchanged, and a row without
    the verdict is refused, not scored."""
    scorer = _scorer()
    manifest = json.loads(json.dumps(MANIFEST))
    entry = manifest["tasks"]["m1"]
    entry["challenge"] = "every object placed"
    _frozen_rows(tmp_path)
    results = _results(tmp_path, entry, {"train": 30, "ga_s": 12})
    _edit_episodes(results, lambda episode: episode.update(
        challenge_success=episode["success"] and episode["replay_source_record_index"] % 3 == 0))

    def report():
        digest, sources = frozen.load_frozen_set(frozen.pinned_path("m1"), manifest, "m1", tmp_path)
        return scorer.score(manifest, "m1", scorer.collect(manifest, "m1", results), sources, digest, "v1-plus")

    scored = report()
    expected = {}
    for cell, directory in results:
        if directory.name.endswith("_gr00t"):
            runs = json.loads(_result_file(directory).read_text())["runs"].values()
            expected[cell] = sum(e["challenge_success"] for run in runs for e in run["rebuilds"][0]["episodes"])
    assert 0 < expected["train"] < 30 and 0 < expected["ga_s"] < 12
    assert scored["models"]["gr00t"]["cells"]["train"]["k"] == 30
    track = scored["challenge"]
    assert track["rule"] == "every object placed"
    for cell, k in expected.items():
        cell_track = track["models"]["gr00t"]["cells"][cell]
        assert (cell_track["k"], cell_track["n"]) == (k, 120)
        assert cell_track["ci95"] == pytest.approx(list(scorer.beta_interval(k, 120)))
    assert track["models"]["gr00t"]["macro_sr"] == pytest.approx((expected["train"] + expected["ga_s"]) / 240)
    assert "challenge track (never in the headline): every object placed" in scorer.format_report(manifest, "m1", scored)
    _edit_episodes(results, lambda episode: episode.pop("challenge_success"), only=lambda d: d.name == "train_gr00t")
    with pytest.raises(scorer.Refusal, match="records no challenge_success"):
        report()


def test_a_row_judged_by_another_rule_is_refused_and_the_challenge_track_names_a_deviation(tmp_path):
    """A field of the task's scoring_environment (DB-L1's l1_headline_bags) decides what a row's success means, so a
    row that ran it at another value, or without it (a cell file run without composing), is refused rather than
    scored as a deviation. A deviation of any other field still scores, and the challenge track names it too. M1's
    newton_njmax stands in for the scoring field, at its declared value."""
    scorer = _scorer()
    manifest = json.loads(json.dumps(MANIFEST))
    entry = manifest["tasks"]["m1"]
    entry["challenge"] = "every object placed"
    limit = entry["cells"]["train"]["declared"]["newton_njmax"]
    entry["scoring_environment"] = {"newton_njmax": limit}
    _frozen_rows(tmp_path)

    def report(results):
        digest, sources = frozen.load_frozen_set(frozen.pinned_path("m1"), manifest, "m1", tmp_path)
        return scorer.score(manifest, "m1", scorer.collect(manifest, "m1", results), sources, digest)

    def ran(label, change):
        results = _results(tmp_path / label, entry, {"train": 30, "ga_s": 12})
        _edit_episodes(results, lambda episode: episode.update(challenge_success=episode["success"]))
        return _edit_episodes(results, change, only=lambda d: d.name == "train_gr00t")

    assert report(ran("as_declared", lambda episode: None))["status"] == "complete"
    for label, change in (("another_value", lambda episode: episode["environment_cfg"].update(newton_njmax=limit + 1)),
                          ("unset", lambda episode: episode["environment_cfg"].update(newton_njmax=None))):
        with pytest.raises(scorer.Refusal, match=r"ran newton_njmax=.*\(scoring_environment\)"):
            report(ran(label, change))

    deviated = report(ran("render", lambda episode: episode["environment_cfg"].update(camera_scale=1.0)))
    assert deviated["status"] == "non-headline"
    track = deviated["challenge"]["models"]["gr00t"]["cells"]["train"]
    assert any("ran camera_scale=1.0" in deviation for deviation in track["deviations"]), track
    assert "gr00t: " in (text := scorer.format_report(manifest, "m1", deviated)) and "NON-HEADLINE: " in text.split(
        "challenge track")[1]


def test_the_command_line_names_the_likely_mistakes_and_leaves_other_errors_alone(tmp_path, capsys):
    scorer = _scorer()
    entry = MANIFEST["tasks"]["m1"]
    _frozen_rows(tmp_path)
    results = _results(tmp_path, entry, {"train": 30, "ga_s": 12})
    train = next(path for cell, path in results if cell == "train")
    assert scorer.main(["score", "--task", "db_m1", "--root", str(tmp_path), "--results", f"train={train}"]) == 2
    assert "its tasks are" in capsys.readouterr().err
    assert scorer.main(["score", "--task", "m1", "--root", str(tmp_path / "elsewhere"), "--results", f"train={train}"]) == 2
    assert "download_datasets.sh --eval-conditions" in capsys.readouterr().err
    (train / "run_manifest.json").write_text('{"version": ')
    with pytest.raises(ValueError, match=rf"{train.name}/run_manifest.json: not valid JSON"):
        scorer.main(["score", "--task", "m1", "--root", str(tmp_path), "--results", f"train={train}"])


def test_the_interval_is_the_equal_tailed_beta_posterior():
    scorer = _scorer()
    none, every = scorer.beta_interval(0, 120), scorer.beta_interval(120, 120)
    assert 0.0 < none[0] < 1e-3 and none[1] == pytest.approx(0.0300, abs=2e-4)
    assert every[0] == pytest.approx(1 - none[1]) and every[1] == pytest.approx(1 - none[0])
    assert [round(100 * v, 1) for v in scorer.beta_interval(10, 10)] == [71.5, 99.8]


def test_the_scorer_refuses_a_run_it_cannot_bind_to_its_model_cell_and_rows(tmp_path, capsys):
    scorer = _scorer()
    entry = MANIFEST["tasks"]["m1"]
    _frozen_rows(tmp_path)
    results = _results(tmp_path, entry, {"train": 30, "ga_s": 12})
    with pytest.raises(ValueError, match="are in both"):
        scorer.collect(MANIFEST, "m1", results + results[:2])
    gr00t_train = next(path for cell, path in results if cell == "train" and path.name.endswith("_gr00t"))
    with pytest.raises(ValueError, match="given for m1 ga_s"):
        scorer.collect(MANIFEST, "m1", [("ga_s", gr00t_train)])
    with pytest.raises(ValueError, match="ran against frozen set"):
        scorer.collect(MANIFEST, "m1", results, frozen_sha256="0" * 64)
    assert scorer.main(["score", "--task", "m1", "--root", str(tmp_path), "--results", f"train={gr00t_train}"]) == 2
    assert capsys.readouterr().err.startswith("refused:")
    not_run = tmp_path / "results" / "not_run"
    not_run.mkdir()
    (not_run / "run_manifest.json").write_text((gr00t_train / "run_manifest.json").read_text())
    with pytest.raises(ValueError, match="has not run"):
        scorer.collect(MANIFEST, "m1", [("train", not_run)])

    cosmos_train = _result_file(next(path for cell, path in results if cell == "train" and path.name.endswith("_cosmos")))
    original = json.loads(cosmos_train.read_text())
    cosmos = next(iter(original["runs"]))

    def refused(change, match):
        data = json.loads(json.dumps(original))
        change(data["runs"][cosmos])
        cosmos_train.write_text(json.dumps(data))
        with pytest.raises(ValueError, match=match):
            _score(scorer, tmp_path, results)
        cosmos_train.write_text(json.dumps(original))

    def episodes(change):
        return lambda run: change(run["rebuilds"][0]["episodes"])

    refused(episodes(lambda e: e[0].update(variations={"drawer.grasp_disturbance": [-1]})), "frozen row's draws")
    refused(episodes(lambda e: e.pop()), "missing")
    refused(episodes(lambda e: e.__setitem__(0, dict(e[1]))), "ran twice")
    refused(episodes(lambda e: e[0].update(success=None)), "no success value")
    refused(episodes(lambda e: e[0]["evaluation_settings"].update(physics={"type": "isaaclab_physx.PhysxCfg"})),
            "different evaluation settings")
    refused(lambda run: run.update(policy_variant="gr00t"), "its run manifest names 'cosmos'")

    gr00t_ga_s = next(path for cell, path in results if cell == "ga_s" and path.name.endswith("_gr00t"))
    record = json.loads((gr00t_ga_s / "run_manifest.json").read_text())
    (gr00t_ga_s / "run_manifest.json").write_text(json.dumps({**record, "checkpoint": "gr00t-step-2"}))
    with pytest.raises(ValueError, match="one model's score is one checkpoint"):
        scorer.collect(MANIFEST, "m1", results)
    (gr00t_ga_s / "run_manifest.json").write_text(json.dumps({**record, "checkpoint": None}))
    with pytest.raises(ValueError, match="records no checkpoint"):
        scorer.collect(MANIFEST, "m1", results)


def test_a_run_split_into_row_slices_scores_as_the_whole_run(tmp_path):
    """DB-M1's cells are one 120-row Run each; run_benchmark_cell.py --shard splits one over several jobs. Each
    part's episodes count from its slice's start; joined, they score exactly as the unsplit run. Parts that
    overlap or leave a gap are refused."""
    scorer = _scorer()
    entry = MANIFEST["tasks"]["m1"]
    _frozen_rows(tmp_path)
    results = _results(tmp_path, entry, {"train": 30, "ga_s": 12})
    whole = _score(scorer, tmp_path, results)
    gr00t_train = next(path for cell, path in results if cell == "train" and path.name.endswith("_gr00t"))
    manifest = json.loads((gr00t_train / "run_manifest.json").read_text())
    (run_name, run), = json.loads(_result_file(gr00t_train).read_text())["runs"].items()

    def part(label, start, end):
        directory = tmp_path / "parts" / label
        (directory / "arena").mkdir(parents=True)
        (directory / "run_manifest.json").write_text(json.dumps({**manifest, "slices": {run_name: [start, end]}}))
        episodes = [{**episode, "replay_source_record_index": episode["replay_source_record_index"] - start}
                    for episode in run["rebuilds"][0]["episodes"] if start <= episode["replay_source_record_index"] < end]
        _result_file(directory).write_text(json.dumps({"runs": {run_name: {**run, "rebuilds": [{"index": 0, "episodes": episodes}]}}}))
        return ("train", directory)

    others = [(cell, path) for cell, path in results if path != gr00t_train]
    split = _score(scorer, tmp_path, others + [part("a", 0, 50), part("b", 50, 120)])
    assert split["models"]["gr00t"] == whole["models"]["gr00t"]
    with pytest.raises(ValueError, match="are in both"):
        scorer.collect(MANIFEST, "m1", others + [part("c", 0, 60), part("d", 50, 120)])
    with pytest.raises(ValueError, match="missing"):
        _score(scorer, tmp_path, others + [part("e", 0, 50), part("f", 60, 120)])
    with pytest.raises(ValueError, match="outside its 40 rows"):
        broken = part("g", 0, 40)
        data = json.loads(_result_file(broken[1]).read_text())
        data["runs"][run_name]["rebuilds"][0]["episodes"][0]["replay_source_record_index"] = 45
        _result_file(broken[1]).write_text(json.dumps(data))
        scorer.collect(MANIFEST, "m1", [broken])


def test_a_cell_from_attempts_a_crash_ended_scores_each_accepted_row_once_and_counts_the_salvage(tmp_path):
    """An eval launch accepts each row from one attempt (episode-level salvage): a slice whose simulator crashed after
    some episodes has no experiment result, only Arena's per-episode lines, and lines past what the launch accepted;
    the next attempt ran the rows left. Scored with what the launch accepted, the parts score as the whole Run, and the
    cell counts the salvaged episodes and the rows that ran more than once. A row accepted from two attempts, a row
    accepted from none, and a row the launch says an attempt finished that its directory lacks are refused."""
    scorer = _scorer()
    entry = MANIFEST["tasks"]["m1"]
    _frozen_rows(tmp_path)
    results = _results(tmp_path, entry, {"train": 30, "ga_s": 12})
    whole = _score(scorer, tmp_path, results)
    gr00t_train = next(path for cell, path in results if cell == "train" and path.name.endswith("_gr00t"))
    manifest = json.loads((gr00t_train / "run_manifest.json").read_text())
    (run_name, run), = json.loads(_result_file(gr00t_train).read_text())["runs"].items()
    by_row = {episode["replay_source_record_index"]: episode for episode in run["rebuilds"][0]["episodes"]}

    def crashed(label, start, end, finished):
        """An attempt of rows [start, end) that finished ``finished`` of them and crashed in the next one's episode."""
        directory = tmp_path / "parts" / label
        (directory / "arena" / run_name).mkdir(parents=True)
        (directory / "run_manifest.json").write_text(json.dumps({**manifest, "slices": {run_name: [start, end]},
                                                                 "policy": {"policy_variant": "gr00t"}}))
        lines = [json.dumps({**by_row[start + i], "replay_source_record_index": i}) for i in range(finished)]
        (directory / "arena" / run_name / "episode_results_rebuild0.jsonl").write_text("\n".join(lines) + '\n{"env_id": 0, "succ')
        return ("train", directory)

    def complete(label, start, end):
        directory = tmp_path / "parts" / label
        (directory / "arena").mkdir(parents=True)
        (directory / "run_manifest.json").write_text(json.dumps({**manifest, "slices": {run_name: [start, end]}}))
        episodes = [{**by_row[i], "replay_source_record_index": i - start} for i in range(start, end)]
        _result_file(directory).write_text(json.dumps({"runs": {run_name: {**run, "rebuilds": [{"index": 0, "episodes": episodes}]}}}))
        return ("train", directory)

    def scored(parts, accepted):
        digest, sources = frozen.load_frozen_set(frozen.pinned_path("m1"), MANIFEST, "m1", tmp_path)
        others = [(cell, path) for cell, path in results if path != gr00t_train]
        accepted = {str(parts[label][1]): value for label, value in accepted.items()}
        return scorer.score(MANIFEST, "m1", scorer.collect(MANIFEST, "m1", others + list(parts.values()), accepted=accepted),
                            sources, digest, "v1-plus")

    parts = {"first": crashed("first", 0, 120, 50), "second": complete("second", 48, 120)}
    accepted = {"first": {"rows": {run_name: list(range(48))}, "salvaged": True, "retried": {}},
                "second": {"rows": None, "salvaged": False, "retried": {run_name: [0]}}}
    report = scored(parts, accepted)
    gr00t = report["models"]["gr00t"]["cells"]["train"]
    assert {key: gr00t[key] for key in ("k", "n", "sr")} == {key: whole["models"]["gr00t"]["cells"]["train"][key]
                                                            for key in ("k", "n", "sr")}
    assert gr00t["attempts"] == {"salvaged_episodes": 48, "retried_rows": 1}
    assert "attempts" not in report["models"]["cosmos"]["cells"]["train"]
    assert "(48 salvaged, 1 rerun)" in scorer.format_report(MANIFEST, "m1", report)
    with pytest.raises(ValueError, match="are in both"):
        scored({"first": parts["first"], "second": complete("again", 47, 120)}, accepted)
    with pytest.raises(ValueError, match="missing"):
        scored(parts, {**accepted, "first": {**accepted["first"], "rows": {run_name: list(range(47))}}})
    with pytest.raises(ValueError, match="holds 0 episodes replaying 50"):
        scored(parts, {**accepted, "first": {**accepted["first"], "rows": {run_name: list(range(51))}}})


def test_v1_rows_balance_alternating_and_ordered_rows_and_vary_across_shared_runs():
    """v1 keeps half of a cell. A Run whose rows alternate and run in order (DB-M5: row i at compartment i mod 2,
    pile rank i) keeps both compartments evenly and ranks across the whole Run, where its first rows would keep the
    low half; Runs that share their draws row by row (DB-L3's GA-S settings) keep different rows; v1-plus keeps all."""
    m5 = version_rows({f"part_{i}": 24 for i in range(5)}, "v1")
    assert sum(map(len, m5.values())) == 60
    for rows in m5.values():
        assert sum(index % 2 for index in rows) == len(rows) / 2
        assert max(rows) - min(rows) >= 20
    shared = version_rows({f"setting_{i}": 6 for i in range(20)}, "v1")
    assert sum(map(len, shared.values())) == 60
    counts = [sum(index in rows for rows in shared.values()) for index in range(6)]
    assert min(counts) >= 8 and max(counts) <= 12, counts
    assert version_rows({"a": 70, "b": 50}, "v1-plus") == {"a": tuple(range(70)), "b": tuple(range(50))}


def test_v1_keeps_half_of_every_objects_rows_in_every_pinned_cell():
    """A cell of 1-row Runs listed by object (DB-R2 GA-OS: 84 Runs over four objects) keeps half of each object's
    rows at v1, where splitting over the Runs alone would give the first objects the tied remainders; and every
    committed version pin is what the pin writer gives for the manifest, so each pinned cell keeps half of each
    object."""
    quota = {f"{obj}_{i}": 1 for obj in ("a", "b") for i in range(15)} | {f"{obj}_{i}": 1 for obj in ("c", "d") for i in range(45)}
    objects = {name: name.split("_")[0] for name in quota}
    kept = version_rows(quota, "v1", objects)
    assert {obj: sum(len(rows) for name, rows in kept.items() if objects[name] == obj) for obj in "abcd"} == \
        {"a": 8, "b": 8, "c": 22, "d": 22}
    for task, spec in MANIFEST["tasks"].items():
        pin = frozen.version_pin_path(task)
        if not pin.is_file():
            continue
        assert pin.read_bytes() == frozen.version_pin_bytes(frozen.version_pin(MANIFEST, task, frozen.pinned_path(task).read_bytes())), task
        for cell, entry in spec["cells"].items():
            per_object: dict[str, list[int]] = {}
            pinned = frozen.pinned_version_rows(MANIFEST, task, cell, "v1")
            for name, row in entry["rows"].items():
                counts = per_object.setdefault(row["object"], [0, 0])
                counts[0] += len(pinned.get(name, ())); counts[1] += row["rows"]
            total = sum(rows for _, rows in per_object.values())
            for obj, (v1, rows) in per_object.items():
                assert abs(v1 - rows * sum(k for k, _ in per_object.values()) / total) < 1, (task, cell, obj, v1, rows)


@pytest.mark.parametrize("quota", [{"a": 120}, {f"r{i}": 30 for i in range(4)}, {"r0": 60, "r1": 60}],
                         ids=["1x120", "4x30", "2x60"])
def test_a_pilots_rows_are_rows_every_headline_model_ran(quota):
    """A pilot (Astra's trials, 10 rows of a cell) is compared with the headline models on its rows, so they must be
    rows those models ran at either version: they are, and the runner and the scorer take them from one helper."""
    pinned = {"v1": version_rows(quota, "v1")}.__getitem__
    pilot = cell_rows(quota, 10, "v1", pinned)
    assert sum(map(len, pilot.values())) == 10 and pilot == cell_rows(quota, 10, "v1-plus", pinned)
    for version in ("v1", "v1-plus"):
        reference = cell_rows(quota, None, version, pinned)
        assert all(set(rows) <= set(reference[name]) for name, rows in pilot.items()), (version, pilot)


def test_every_db_m5_run_keeps_both_compartments_evenly_at_v1(monkeypatch):
    """DB-M5's rows alternate between its two compartments (row i at i mod 2); its pinned v1 rows keep half of each
    in every Run. An odd keep would not (7 to 6 of 26), and no DB-M5 Run has one."""
    monkeypatch.setattr(frozen, "PINNED_DIR", PINNED_DIR)
    for cell in MANIFEST["tasks"]["m5"]["cells"]:
        for name, rows in frozen.pinned_version_rows(MANIFEST, "m5", cell, "v1").items():
            assert len(rows) % 2 == 0 and 2 * sum(index % 2 for index in rows) == len(rows), (cell, name, rows)


def test_the_v1_rows_are_read_from_the_pin_and_not_from_the_manifests_run_order(tmp_path):
    """A cell's v1 rows are pinned beside its frozen set: a manifest that lists the cell's Runs in another order
    selects the same rows of each row file, and a pin made for another frozen set is refused. The pin is read only
    for a version that runs fewer than the cell's rows, so v1-plus, or v1 on a cell of 60 rows, needs none."""
    manifest = json.loads(json.dumps(MANIFEST))
    cell = manifest["tasks"]["m4"]["cells"]["ga_s"]
    quota = {name: row["rows"] for name, row in cell["rows"].items()}
    frozen.pinned_path("m4").parent.mkdir(parents=True)
    frozen.pinned_path("m4").write_text("{}")
    frozen.version_pin_path("m4").write_bytes(frozen.version_pin_bytes(
        frozen.version_pin(manifest, "m4", frozen.pinned_path("m4").read_bytes())))
    pin = partial(frozen.pinned_version_rows, manifest, "m4", "ga_s")
    pinned = cell_rows(quota, None, "v1", pin)
    assert pinned == version_rows(quota, "v1")
    cell["rows"] = dict(reversed(cell["rows"].items()))
    reordered = {name: row["rows"] for name, row in cell["rows"].items()}
    assert version_rows(reordered, "v1") != pinned
    assert cell_rows(reordered, None, "v1", pin) == pinned
    every_row = {name: tuple(range(count)) for name, count in quota.items()}
    good = frozen.version_pin_path("m4").read_text()
    frozen.version_pin_path("m4").write_text(good.replace('"v1":', '"v2":'))
    with pytest.raises(Refusal, match=r"pins rows of \['v2'\], which are not versions"):
        cell_rows(quota, None, "v1", pin)
    frozen.version_pin_path("m4").write_text(good)
    frozen.pinned_path("m4").write_text("{ }")
    with pytest.raises(ValueError, match="pinned again with the set"):
        cell_rows(quota, None, "v1", pin)
    assert cell_rows(quota, None, "v1-plus", pin) == every_row
    frozen.version_pin_path("m4").unlink()
    with pytest.raises(ValueError, match="no v1 rows are pinned"):
        cell_rows(quota, None, "v1", pin)
    assert cell_rows(quota, None, "v1-plus", pin) == every_row
    assert cell_rows({"a": 30, "b": 30}, None, "v1", pin) == {"a": tuple(range(30)), "b": tuple(range(30))}


@pytest.mark.parametrize("task", [task for task in MANIFEST["tasks"] if (PINNED_DIR / frozen.pinned_path(task).name).is_file()])
def test_every_pinned_set_has_its_v1_rows_pinned_beside_it(task, monkeypatch):
    """The checked-in version pins load: made for the checked-in frozen set, 60 rows per cell from its row files."""
    monkeypatch.setattr(frozen, "PINNED_DIR", PINNED_DIR)
    for cell, cell_entry in MANIFEST["tasks"][task]["cells"].items():
        quota = {name: row["rows"] for name, row in cell_entry["rows"].items()}
        assert sum(map(len, cell_rows(quota, None, "v1", partial(frozen.pinned_version_rows, MANIFEST, task, cell)).values())) == 60


def test_a_v1_score_comes_from_a_v1_run_or_from_the_v1_rows_of_a_v1_plus_run(tmp_path):
    """The default version scores each cell's v1 rows: a v1-plus run (every row) is scored on those rows only, a v1
    run (its rows file cut by write_rows, replayed in its own order) gives the same report, and a v1 run is not a
    v1-plus score."""
    scorer = _scorer()
    entry = MANIFEST["tasks"]["m1"]
    _frozen_rows(tmp_path)
    results = _results(tmp_path, entry, {"train": 30, "ga_s": 12})
    from_plus = _score(scorer, tmp_path, results, "v1")
    for cell, successes in {"train": 30, "ga_s": 12}.items():
        kept = version_rows({name: row["rows"] for name, row in entry["cells"][cell]["rows"].items()}, "v1")
        expected_k = sum(index < successes for rows in kept.values() for index in rows)
        assert (from_plus["models"]["gr00t"]["cells"][cell]["k"], from_plus["models"]["gr00t"]["cells"][cell]["n"]) == (expected_k, 60)
    assert from_plus["version"] == "v1" and from_plus["rows_per_cell"] == 60

    def as_v1(cell, directory):
        manifest = json.loads((directory / "run_manifest.json").read_text())
        runs = json.loads(_result_file(directory).read_text())["runs"]
        quota = {name: row["rows"] for name, row in entry["cells"][cell]["rows"].items()}
        kept = version_rows(quota, "v1")
        v1_directory = tmp_path / "v1" / directory.name
        (v1_directory / "arena").mkdir(parents=True)
        for name, rows in kept.items():
            source = entry["cells"][cell]["rows"][name]["path"]
            record = frozen.write_rows(source, rows, v1_directory / "rows" / f"{name}.jsonl", tmp_path)
            assert record["rows"] == list(rows)
        (v1_directory / "run_manifest.json").write_text(json.dumps(
            {**manifest, "eval_version": "v1", "row_indices": {name: list(rows) for name, rows in kept.items()}}))
        v1_runs = {}
        for name, rows in kept.items():
            by_index = {episode["replay_source_record_index"]: episode for episode in runs[name]["rebuilds"][0]["episodes"]}
            episodes = [{**by_index[index], "replay_source_record_index": position}
                        for position, index in reversed(list(enumerate(rows)))]
            v1_runs[name] = {**runs[name], "rebuilds": [{"index": 0, "episodes": episodes}]}
        _result_file(v1_directory).write_text(json.dumps({"runs": v1_runs}))
        return cell, v1_directory

    v1_results = [as_v1(cell, directory) for cell, directory in results]
    assert _score(scorer, tmp_path, v1_results, "v1")["models"] == from_plus["models"]
    with pytest.raises(ValueError, match="missing"):
        _score(scorer, tmp_path, v1_results, "v1-plus")


def test_an_ablation_never_enters_the_headline_or_makes_it_partial(tmp_path):
    """With every required model present, the report is complete whether the ablation ran or not; the
    ablation's results, even ones off the headline models' settings, change nothing in the headline."""
    scorer = _scorer()
    manifest = json.loads(json.dumps(MANIFEST))
    for model in REQUIRED_MODELS:
        if not model.available:
            del manifest["tasks"]["m1"]["models"][model.key]
    entry = manifest["tasks"]["m1"]
    _frozen_rows(tmp_path)
    results = _results(tmp_path, entry, {"train": 30, "ga_s": 12})
    digest, sources = frozen.load_frozen_set(frozen.pinned_path("m1"), MANIFEST, "m1", tmp_path)

    def report_of(chosen):
        return scorer.score(manifest, "m1", scorer.collect(manifest, "m1", chosen), sources, digest)

    with_ablation = report_of(results)
    without = report_of([(cell, path) for cell, path in results if not path.name.endswith(("_gr00t_rel", "_cosmos_rel"))])
    assert with_ablation["status"] == without["status"] == "complete"
    assert with_ablation["models"] == without["models"] and without["ablations"] == {}
    def off_settings(episode):
        episode["evaluation_settings"]["control"]["decimation"] = 8
        episode["success"] = True

    _edit_episodes(results, off_settings, only=lambda path: path.name.endswith("_gr00t_rel"))
    changed = report_of(results)
    assert changed["status"] == "complete" and changed["models"] == without["models"]
    assert any("other evaluation settings" in d for d in changed["ablations"]["gr00t_rel"]["cells"]["train"]["deviations"])


def test_a_model_its_eval_config_registered_is_scored_apart_with_its_entry(tmp_path):
    """Config-only registration: runs whose manifests record a config entry are scored under community,
    with the entry, and change nothing in the headline; a config entry under a registered key is refused."""
    import shutil

    scorer = _scorer()
    entry = MANIFEST["tasks"]["m1"]
    _frozen_rows(tmp_path)
    results = _results(tmp_path, entry, {"train": 30, "ga_s": 12})
    digest, sources = frozen.load_frozen_set(frozen.pinned_path("m1"), MANIFEST, "m1", tmp_path)
    model_entry = {"key": "mymodel", "label": "My Model", "note": "", "group": "community", "cells": None,
                   "task_policy": {}, "rows": None, "registration": "config",
                   "policy": {**MODELS_BY_KEY["cosmos"].policy, "policy_variant": "mymodel"}}

    def as_config_model(directory, key):
        copy = directory.with_name(directory.name.replace("_cosmos", f"_config_{key}"))
        shutil.copytree(directory, copy)
        run_manifest = json.loads((copy / "run_manifest.json").read_text())
        (copy / "run_manifest.json").write_text(json.dumps({**run_manifest, "model": key,
                                                            "model_entry": {**model_entry, "key": key}}))
        data = json.loads(_result_file(copy).read_text())
        for run in data["runs"].values():
            run["policy_variant"] = key
            for episode in run["rebuilds"][0]["episodes"]:
                episode["evaluation_settings"]["policy"]["policy_variant"] = key
        _result_file(copy).write_text(json.dumps(data))
        return copy

    def report_of(chosen):
        manifest = scorer.with_config_models(MANIFEST, "m1", chosen)
        return scorer.score(manifest, "m1", scorer.collect(manifest, "m1", chosen), sources, digest)

    cosmos = [(cell, path) for cell, path in results if path.name.endswith("_cosmos")]
    mine = [(cell, as_config_model(path, "mymodel")) for cell, path in cosmos]
    with_mine, without = report_of(results + mine), report_of(results)
    assert with_mine["models"] == without["models"] and with_mine["status"] == without["status"]
    assert with_mine["community"]["mymodel"]["entry"]["policy"]["policy_variant"] == "mymodel"
    assert with_mine["community"]["mymodel"]["headline"] is False
    assert with_mine["community"]["mymodel"]["cells"]["train"]["k"] == without["models"]["cosmos"]["cells"]["train"]["k"]
    bare = mine[0][1] / "run_manifest.json"
    bare.write_text(json.dumps({key: value for key, value in json.loads(bare.read_text()).items() if key != "model_entry"}))
    with pytest.raises(Refusal, match="records no config model entry"):
        report_of(results + mine[:1])
    with pytest.raises(Refusal, match="does not record .* config entry"):
        report_of(results + mine)
    coded = mine[0][1] / "run_manifest.json"
    coded.write_text(json.dumps({**json.loads(coded.read_text()), "model_entry": {**model_entry,
                                                                                 "registration": "code"}}))
    with pytest.raises(Refusal, match="does not record .* config entry"):
        report_of(results + mine)
    clash = [(cell, as_config_model(path, "gr00t_rel")) for cell, path in cosmos[:1]]
    with pytest.raises(Refusal, match="never scored as the registered model"):
        report_of(results[:1] + clash)


def test_rows_recorded_before_a_rename_or_a_new_cfg_field_score_with_rows_recorded_after(tmp_path, monkeypatch):
    """One model's rows name a cfg value by its old name (the manifest's legacy_values maps it) and have no
    ``mimic_action_noise`` key; the others have the current name and the field null. Same settings: one cell,
    every model headline. So are rows recorded with and without the task's
    success criteria, but not two models' rows that record different ones, and rows that differ only in an
    option of the environment's other tasks."""
    scorer = _scorer()
    entry = MANIFEST["tasks"]["m1"]
    _frozen_rows(tmp_path)
    current = entry["cells"]["train"]["declared"]["embodiment"]
    results = _results(tmp_path, entry, {"train": 30, "ga_s": 12})
    _edit_episodes(results, lambda episode: episode["environment_cfg"].update(embodiment="old_name"),
                   only=lambda directory: directory.name.endswith("_gr00t"))
    _edit_episodes(results, lambda episode: episode["environment_cfg"].update(mimic_action_noise=None),
                   only=lambda directory: not directory.name.endswith("_gr00t"))
    with pytest.raises(ValueError, match="different evaluation settings"):
        _score(scorer, tmp_path, results)
    monkeypatch.setitem(entry, "legacy_values", {"embodiment": {"old_name": current}})
    report = _score(scorer, tmp_path, results)
    assert all(data["headline"] for data in report["models"].values() if data["available"])

    def criteria(max_rim_offset_m):
        seated = {"func": "dexbench_arena.tasks.terminations.plates_seated_in_stack",
                  "params": {"max_rim_offset_m": max_rim_offset_m}}
        return lambda episode: episode["evaluation_settings"].update(success_criteria={
            "criteria": [{"name": "success", "sequences": {"default_sequence": [seated]}}],
            "subtasks_are_sequential": False, "desired_subtask_success_state": [True]})

    _edit_episodes(results, criteria(0.01), only=lambda directory: not directory.name.endswith("_gr00t"))
    report = _score(scorer, tmp_path, results)
    assert all(data["headline"] for data in report["models"].values() if data["available"])

    _edit_episodes(results, criteria(0.01), only=lambda directory: directory.name.endswith("_gr00t"))
    _edit_episodes(results, lambda episode: episode["environment_cfg"].update(other_task_option=1.0),
                   only=lambda directory: directory.name.endswith("_gr00t"))
    with pytest.raises(ValueError, match="different evaluation settings"):
        _score(scorer, tmp_path, results)
    monkeypatch.setitem(entry, "other_task_fields", ["other_task_option"])
    report = _score(scorer, tmp_path, results)
    assert all(data["headline"] for data in report["models"].values() if data["available"])
    _edit_episodes(results, criteria(0.05), only=lambda directory: directory.name.endswith("_gr00t"))
    with pytest.raises(ValueError, match="different evaluation settings"):
        _score(scorer, tmp_path, results)

    def interleaved(parity, max_rim_offset_m):
        def change(episode):
            if episode["replay_source_record_index"] % 2 == parity:
                criteria(max_rim_offset_m)(episode)
            else:
                episode["evaluation_settings"].pop("success_criteria", None)
        return change

    _edit_episodes(results, interleaved(0, 0.01), only=lambda directory: directory.name.endswith("_gr00t"))
    _edit_episodes(results, interleaved(1, 0.05), only=lambda directory: not directory.name.endswith("_gr00t"))
    with pytest.raises(ValueError, match="different evaluation settings"):
        _score(scorer, tmp_path, results)


def test_an_r2_row_is_compared_on_r2s_settings_not_on_r3s_defaults():
    """R2 and R3 share retail_v1's config, so every R2 row records R3's options too. The EA rows record R3's facing
    as 175 mm; a row whose R3 derives it from the carton records null. Neither is an R2
    setting (the manifest's other_task_fields): both rows meet every R2 cell's declaration and sign alike. A field
    every task's scene reads still separates them, whatever its name."""
    scorer = _scorer()
    entry = MANIFEST["tasks"]["r2"]
    for cell_entry in entry["cells"].values():
        row = next(iter(cell_entry["rows"]))
        declared = {**cell_entry["declared"], **cell_entry["rows"][row]["environment"]}

        def row_with(**fields):
            episode = _ran_as_declared(cell_entry, row, "gr00t")
            episode["environment_cfg"].update(fields)
            problems = scorer.episode_problems(episode, declared, cell_entry["backend"], "gr00t", MANIFEST["execution_horizon"],
                                               entry.get("legacy_values"), entry["other_task_fields"],
                                               entry.get("policy_execution"))
            return problems, scorer.evaluation_signature(episode, entry.get("legacy_values"),
                                                         other_task_fields=entry["other_task_fields"])

        ea = row_with(facing_width_mm=175.0, r3_target_asset="dexbench_retail_e1_fc_r3t2", box_jitter_xy=0.0)
        assert ea == row_with(facing_width_mm=None) and ea[0] == []
        moved, moved_signature = row_with(facing_width_mm=175.0, r3_supply_offset_x=0.1)
        assert moved_signature != ea[1] and any("r3_supply_offset_x=0.1" in problem for problem in moved)


def test_an_episode_off_its_cells_approved_settings_scores_only_as_non_headline(tmp_path):
    """Agreement among models is not enough: a change every model makes alike (a shorter horizon, another cap,
    another backend, another render, a solver setting) still departs from the cell's approved settings. The
    cell scores, but the report names it non-headline. Unknown settings are not scorable at all."""
    scorer = _scorer()
    entry = MANIFEST["tasks"]["m1"]
    _frozen_rows(tmp_path)

    def every_model(change, label):
        return _edit_episodes(_results(tmp_path / label, entry, {"train": 30, "ga_s": 12}), change)

    def horizon(episode):
        if episode["evaluation_settings"]["policy"]:
            episode["evaluation_settings"]["policy"]["action_chunk_length"] = 32

    for change, named, label in (
        (horizon, "ran action_chunk_length=32", "horizon"),
        (lambda episode: episode["environment_cfg"].update(episode_length_s=40.0), "ran episode_length_s=40.0", "cap"),
        (lambda episode: episode["environment_cfg"].update(newton_num_substeps=4), "ran newton_num_substeps=4", "solver"),
        (lambda episode: episode["evaluation_settings"]["policy"].update(relative_joint_slices=[[0, 7]]),
         "ran relative_joint_slices=[[0, 7]], model gr00t declares []", "representation"),
        (lambda episode: episode["evaluation_settings"]["policy"].get("policy_variant") == "gr00t"
         and episode["evaluation_settings"]["policy"].update(relative_action_groups=["left_arm_joints"]),
         "ran relative_action_groups=['left_arm_joints'], model gr00t declares []", "gr00t representation"),
        (lambda episode: episode["environment_cfg"].update(camera_scale=1.0), "ran camera_scale=1.0", "render"),
        (lambda episode: episode["evaluation_settings"]["control"].update(decimation=8), "its cell runs at 60 Hz", "control rate"),
        (lambda episode: episode["evaluation_settings"]["policy"].update(policy_hz=None, policy_action_interpolation="none"),
         "its cell asks", "policy rate"),
        (lambda episode: episode["evaluation_settings"].update(physics={"type": "isaaclab_physx.PhysxCfg"}),
         "declares NewtonCfg", "backend"),
    ):
        report = _score(scorer, tmp_path, every_model(change, label))
        assert report["status"] == "non-headline", label
        gr00t = report["models"]["gr00t"]
        assert not gr00t["headline"] and gr00t["cells"]["train"]["k"] == 30, label
        assert any(named in deviation for deviation in gr00t["cells"]["train"]["deviations"]), (label, gr00t["cells"]["train"]["deviations"])

    report = _score(scorer, tmp_path, every_model(
        lambda episode: episode["evaluation_settings"]["policy"].get("denoising_timesteps")
        and episode["evaluation_settings"]["policy"].update(denoising_timesteps=[0.0, 0.25, 0.5, 0.75]), "rldx steps"))
    rldx = report["models"]["rldx11"]
    assert not rldx["headline"] and any("ran denoising_timesteps=[0.0, 0.25, 0.5, 0.75]" in deviation
                                        for deviation in rldx["cells"]["train"]["deviations"]), rldx["cells"]["train"]

    for change, match, label in (
        (lambda episode: episode.pop("evaluation_settings"), "settings are unknown", "unknown"),
        (lambda episode: episode["evaluation_settings"].update(version=1), "not known in full", "v1"),
        (lambda episode: episode["evaluation_settings"].update(version=2), "before nested term parameters were kept whole", "v2"),
        (lambda episode: episode["evaluation_settings"]["policy"].pop("relative_action_groups", None),
         "records no served relative_action_groups", "unstamped gr00t"),
        (lambda episode: episode["evaluation_settings"]["policy"].pop("denoising_timesteps", None),
         "records no denoising_timesteps", "unstamped rldx"),
        (lambda episode: episode["evaluation_settings"].pop("terminations"), r"without \['terminations'\]", "partial"),
    ):
        with pytest.raises(ValueError, match=match):
            _score(scorer, tmp_path, every_model(change, label))


def _shifted(row):
    """The row redrawn: the source replaced and re-frozen after the results were made."""
    draw = row["variations"]["drawer.pose"]
    return {**row, "variations": {**row["variations"], "drawer.pose": [draw[0] + 0.01, *draw[1:]]}}


def _drawless(row):
    """The row as a fixed-setting cell (R2's GA-S and GA-OS: the lane and ring point fixed) writes it."""
    return {key: value for key, value in row.items() if key != "variations"}


def _fitted(row):
    return {**row, "task_object": {**row["task_object"], "fit": {"seat_depth_m": 0.25}}}


@pytest.mark.parametrize("change_row, change_episode, refused", [
    (None, lambda episode: episode.update(_source_row("train", episode["replay_source_record_index"])),
     "did not replay its frozen row's draws"),
    (_shifted, lambda episode: None, "did not replay its frozen row's draws"),
    (None, lambda episode: episode.update(task_object={**episode["task_object"], "key": "another_drawer"}), "ran object"),
    (None, lambda episode: episode.update(placement={"box": [0.0, 0.0, 0.0]}), "relation placement"),
    (_fitted, lambda episode: episode.update(task_object={**episode["task_object"], "fit": {"seat_depth_m": 0.30}}),
     "another object fit"),
    (_drawless, lambda episode: episode.pop("variations"), None),
    (_drawless, lambda episode: episode.update(variations={}), None),
    (_drawless, lambda episode: None, "did not replay its frozen row's draws"),
], ids=["agreeing", "refrozen", "object", "placement", "fit", "no_draws", "empty_draws", "one_side_draws"])
def test_a_score_is_bound_to_the_frozen_rows_it_replayed(tmp_path, change_row, change_episode, refused):
    """Results count only as replays of the frozen rows: agreeing models, a re-frozen source, a wrong object,
    a relation placement or another fit do not pass."""
    scorer = _scorer()
    _frozen_rows(tmp_path)
    if change_row is not None:
        _refreeze(tmp_path, change_row)
    results = _edit_episodes(_results(tmp_path, MANIFEST["tasks"]["m1"], {"train": 30, "ga_s": 12}), change_episode)
    if refused is None:
        assert _score(scorer, tmp_path, results)["models"]["gr00t"]["headline"]
    else:
        with pytest.raises(ValueError, match=refused):
            _score(scorer, tmp_path, results)


@pytest.mark.parametrize("source_version,episode_version,refused", [
    ("3.0.0", "3.1.4", False),
    ("1.4.0", "3.0.0", True),
    ("3.0.0", "1.4.0", True),
    ("3.0.0", None, True),
    (None, "3.0.0", True),
    ("1.4.0", "1.4.0", True),
])
def test_scoring_enforces_declared_physical_major_on_both_source_and_episode(tmp_path, source_version,
                                                                          episode_version, refused):
    """The real scorer reads the contract from data, not a hard-coded task or object name."""
    scorer = _scorer()
    manifest = json.loads(json.dumps(MANIFEST))
    manifest["tasks"]["m1"]["asset_major_contracts"] = {DRAWER["axis"]: {DRAWER["key"]: 3}}
    _frozen_rows(tmp_path)
    _refreeze(tmp_path, lambda row: {**row, "task_object": {**row["task_object"], "asset_version": source_version}})
    results = _results(tmp_path, manifest["tasks"]["m1"], {"train": 30, "ga_s": 12})
    _edit_episodes(results, lambda episode: episode["task_object"].update(asset_version=episode_version))
    digest, sources = frozen.load_frozen_set(frozen.pinned_path("m1"), manifest, "m1", tmp_path)
    if refused:
        with pytest.raises(ValueError, match="asset MAJOR"):
            scorer.score(manifest, "m1", scorer.collect(manifest, "m1", results), sources, digest)
    else:
        assert scorer.score(manifest, "m1", scorer.collect(manifest, "m1", results), sources, digest)["status"] == "complete"


@pytest.mark.parametrize("source_version,episode_version,current_version,renames,refused", [
    ("1.0.0", "2.0.0", "2.0.0", [], True),
    ("1.0.0", "2.0.0", "2.0.0", [[1, 2, "CHANGELOG.md"]], False),
    ("1.0.0", "3.0.0", "3.0.0", [[1, 2, "CHANGELOG.md"]], True),
    ("2.0.0", None, "2.0.0", [], True),
    ("2.0.0", "2.0.0", None, [], True),
    (None, None, "3.0.0", [], False),
])
def test_scoring_checks_known_versions_without_a_required_major(tmp_path, source_version, episode_version,
                                                               current_version, renames, refused):
    scorer = _scorer()
    manifest = json.loads(json.dumps(MANIFEST))
    entry = manifest["tasks"]["m1"]
    entry.pop("asset_major_contracts", None)
    entry["asset_version_contracts"] = {DRAWER["axis"]: {DRAWER["key"]: {
        "version": current_version, "rename_only": renames}}}
    _frozen_rows(tmp_path)
    _refreeze(tmp_path, lambda row: {**row, "task_object": {**row["task_object"], "asset_version": source_version}})
    results = _results(tmp_path, entry, {"train": 30, "ga_s": 12})
    _edit_episodes(results, lambda episode: episode["task_object"].update(asset_version=episode_version))
    digest, sources = frozen.load_frozen_set(frozen.pinned_path("m1"), manifest, "m1", tmp_path)
    if refused:
        with pytest.raises(ValueError, match="asset MAJOR"):
            scorer.score(manifest, "m1", scorer.collect(manifest, "m1", results), sources, digest)
    else:
        assert scorer.score(manifest, "m1", scorer.collect(manifest, "m1", results), sources, digest)["status"] == "complete"


def test_an_h3_episode_scores_with_the_utensil_version_it_records(tmp_path):
    """An H3 episode records its utensil package's sidecar version (``task_object``); the manifest's H3 contract
    must name the same version, or the scorer refuses every H3 episode. The frozen H3 rows record none."""
    from dexbench_arena import ASSETS_DIR
    from dexbench_arena.assets.physics_variants import asset_package_version
    from dexbench_arena.tasks.h3_objects import H3_OBJECTS, H3_PACKAGES

    scorer = _scorer()
    entry = MANIFEST["tasks"]["h3"]

    def task_object(key, version):
        return {"axis": "h3_object", "key": key, "fit": {"target": H3_OBJECTS[key]["target"]}, "asset_version": version}

    for cell_entry in entry["cells"].values():
        for row in cell_entry["rows"].values():
            path = tmp_path / row["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(json.dumps({"task_object": task_object(row["object"], None),
                                                "variations": {"start": [float(index)]}}) + "\n"
                                    for index in range(row["rows"])))
    pin = frozen.pinned_path("h3")
    pin.parent.mkdir(parents=True, exist_ok=True)
    pin.write_text(json.dumps(frozen.freeze(MANIFEST, "h3", tmp_path)))
    frozen.version_pin_path("h3").write_bytes(frozen.version_pin_bytes(frozen.version_pin(MANIFEST, "h3", pin.read_bytes())))
    results = []
    for cell, cell_entry in entry["cells"].items():
        for model in (key for key, declared in entry["models"].items()
                      if declared["available"] and declared.get("group") != "trial"):
            directory = tmp_path / "results" / f"{cell}_{model}"
            (directory / "arena").mkdir(parents=True)
            (directory / "run_manifest.json").write_text(json.dumps({
                "version": 1, "task": "h3", "section": "headline", "cell": cell, "model": model,
                "frozen_sha256": "unchecked", "arena_output": "arena", "checkpoint": f"{model}-step-1",
                "deployment": {}, "code": {"revision": "abc123"}, "cell_config_sha256": "c" * 64}))
            runs = {}
            for name, row in cell_entry["rows"].items():
                recorded = asset_package_version(ASSETS_DIR / H3_PACKAGES[H3_OBJECTS[row["object"]]["asset"]]["usd_path"])
                episodes = [{"success": index == 0, "replay_source_record_index": index,
                             "task_object": task_object(row["object"], recorded), "variations": {"start": [float(index)]},
                             **_ran_as_declared(cell_entry, name, model)} for index in range(row["rows"])]
                runs[name] = {"policy_variant": model, "status": "completed", "rebuilds": [{"index": 0, "episodes": episodes}]}
            _result_file(directory).write_text(json.dumps({"runs": runs}))
            results.append((cell, directory))
    digest, sources = frozen.load_frozen_set(pin, MANIFEST, "h3", tmp_path)
    scored = scorer.score(MANIFEST, "h3", scorer.collect(MANIFEST, "h3", results), sources, digest)
    assert scored["status"] == "complete"
    assert all(set(model["cells"]) == set(entry["cells"]) for model in scored["models"].values())


def test_a_frozen_set_lists_each_row_file_and_refuses_a_changed_or_unlisted_one(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    expected = frozen.expected_rows(MANIFEST, "m1")
    for relative, rows in expected.items():
        Path(relative).parent.mkdir(parents=True, exist_ok=True)
        Path(relative).write_text("".join(json.dumps({"i": index}) + "\n" for index in range(rows)))
    record = frozen.freeze(MANIFEST, "m1")
    assert set(record) == set(expected)
    target = _pin({})
    with pytest.raises(ValueError, match="empty"):
        frozen.load_frozen_set(target, MANIFEST, "m1")
    target.write_text(json.dumps(dict(list(record.items())[:1])))
    with pytest.raises(ValueError, match="not m1's headline set"):
        frozen.load_frozen_set(target, MANIFEST, "m1")
    target.write_text(json.dumps(record))
    some_row_file = next(iter(expected))
    frozen.check_frozen_row_file(some_row_file)
    with pytest.raises(ValueError, match="missing; freeze needs every row file of the sensitivity set"):
        frozen.freeze(MANIFEST, "m1", section="sensitivity")

    Path(some_row_file).write_text(Path(some_row_file).read_text().replace('"i": 0', '"i": 9'))
    with pytest.raises(ValueError, match="changed since"):
        frozen.check_frozen_row_file(some_row_file)
    target.unlink()
    with pytest.raises(ValueError, match="not in any pinned frozen set"):
        frozen.check_frozen_row_file(some_row_file)
    Path(some_row_file).write_text("{}\n")
    with pytest.raises(ValueError, match="1 rows, the protocol replays 120"):
        frozen.freeze(MANIFEST, "m1")


def test_freeze_writes_the_v1_rows_beside_its_candidate_and_they_load_once_committed(tmp_path, capsys):
    """``score_benchmark_cells.py freeze`` writes ``<candidate>_versions.json``: the v1 rows of each row file, for the
    candidate's bytes. Committed as the pins (``db_m1.json``, ``versions_db_m1.json``), they load, and a second freeze
    says both match."""
    scorer = _scorer()
    for cell_entry in MANIFEST["tasks"]["m1"]["cells"].values():
        for row in cell_entry["rows"].values():
            path = tmp_path / row["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(json.dumps({"i": index}) + "\n" for index in range(row["rows"])))
    candidate = tmp_path / "candidate.json"
    assert scorer.main(["freeze", "--task", "m1", "--root", str(tmp_path), "--output", str(candidate)]) == 0
    pin = json.loads((tmp_path / "candidate_versions.json").read_text())
    assert pin["frozen_set"] == "db_m1.json" and pin["frozen_sha256"] == hashlib.sha256(candidate.read_bytes()).hexdigest()
    for cell_entry in MANIFEST["tasks"]["m1"]["cells"].values():
        quota = {name: row["rows"] for name, row in cell_entry["rows"].items()}
        assert {cell_entry["rows"][name]["path"]: list(rows) for name, rows in version_rows(quota, "v1").items()} == {
            name: rows for name, rows in pin["versions"]["v1"].items() if name in
            {row["path"] for row in cell_entry["rows"].values()}}
    frozen.PINNED_DIR.mkdir(parents=True)
    frozen.pinned_path("m1").write_bytes(candidate.read_bytes())
    frozen.version_pin_path("m1").write_bytes((tmp_path / "candidate_versions.json").read_bytes())
    for cell in MANIFEST["tasks"]["m1"]["cells"]:
        assert sum(map(len, frozen.pinned_version_rows(MANIFEST, "m1", cell, "v1").values())) == 60
    capsys.readouterr()
    assert scorer.main(["freeze", "--task", "m1", "--root", str(tmp_path), "--output", str(tmp_path / "again.json")]) == 0
    printed = capsys.readouterr().out
    assert "matches the pinned set" in printed and "match the pinned rows" in printed, printed


@pytest.mark.parametrize("section", frozen.SECTIONS)
@pytest.mark.parametrize("task", MANIFEST["tasks"])
def test_the_pinned_frozen_sets_list_exactly_the_declared_row_files(task, section):
    """A spec change that moves or resizes a row file needs a new pin, committed on purpose. A task whose rows are
    not recorded yet declares it (spec.RowsPending) and has no pin; committing its first pin ends that state."""
    path = PINNED_DIR / frozen.pinned_path(task, section).name
    pending = MANIFEST["tasks"][task]["rows_pending"] or (
        section == "sensitivity" and MANIFEST["tasks"][task]["sensitivity"].get("rows_pending"))
    if pending:
        assert not path.is_file(), f"{task} has a pinned set; drop its rows_pending"
        return
    pinned = json.loads(path.read_text())
    assert {name: record["rows"] for name, record in pinned.items()} == frozen.expected_rows(MANIFEST, task, section)


@pytest.mark.parametrize("task", sorted({"m1", *(name for name, entry in MANIFEST["tasks"].items() if entry["rows_pending"])}))
def test_a_task_without_a_pinned_set_is_never_scored(task, tmp_path):
    """No pin (a task whose rows are pending, or a pin removed): the scorer refuses before it counts anything."""
    with pytest.raises(FileNotFoundError, match=frozen.pinned_path(task).name):
        _scorer().main(["score", "--task", task, "--results", f"train={tmp_path}"])


def test_a_published_file_is_written_once_and_never_replaced(tmp_path):
    target = tmp_path / "rows" / "tc1.jsonl"
    rows = [{"variations": {"k": [0.1, 0.2]}}, {"variations": {"k": [0.3, 0.4]}}]
    assert publish_new(target, jsonl_bytes(rows)) is True
    assert publish_new(target, jsonl_bytes(rows)) is False
    with pytest.raises(FileExistsError, match="other content"):
        publish_new(target, jsonl_bytes(rows[:1]))
    assert target.read_bytes() == jsonl_bytes(rows)
    assert sorted(path.name for path in target.parent.iterdir()) == ["tc1.jsonl"]


def test_a_numeric_draw_is_finite_numbers_only():
    assert numeric_draw([0.01, -2, 0.0]) == [0.01, -2.0, 0.0] and numeric_draw(1.5) == [1.5]
    for bad in (None, "0.1", True, [0.1, None], [0.1, float("inf")], [], [0.1, "x"]):
        assert numeric_draw(bad) is None, bad
    assert is_finite({"a": [1.0, "x", None]}) and not is_finite({"a": [1.0, float("nan")]})
