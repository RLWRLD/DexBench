"""Submission validation through the CLI with shuffled native-format trajectory evidence."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest

h5py = pytest.importorskip("h5py", reason="offline submission extra")
np = pytest.importorskip("numpy", reason="offline submission extra")
pytest.importorskip("jsonschema", reason="offline submission extra")
pytest.importorskip("scipy", reason="official scorer's Beta intervals")

from dexbench_arena.benchmark import frozen
from dexbench_arena.benchmark.reference_settings import evaluation_contract_sha256, reference_path
from dexbench_arena.benchmark.models import MODELS_BY_KEY, model_record
from dexbench_arena.evaluation.artifacts import file_sha256
from test_benchmark_scoring import MANIFEST, REPO, _frozen_rows, _results, _tool


def _write_trajectory(path, episodes):
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        data.attrs["env_args"] = json.dumps({"env_name": "synthetic-test"})
        for index, episode in enumerate(reversed(episodes)):
            demo = data.create_group(f"demo_{index}")
            demo.attrs["num_samples"] = 2
            for field in ("env_id", "episode_in_env"):
                demo.create_dataset(f"episode_id/{field}", data=[episode[field]])
            for field in ("actions", "processed_actions", "obs"):
                demo.create_dataset(field, data=np.zeros((2, 2)))
            for group, count in (("initial_state", 1), ("states", 2)):
                for field, width in (("root_pose", 7), ("root_velocity", 6), ("joint_position", 2), ("joint_velocity", 2)):
                    values = np.zeros((count, width))
                    if field == "root_pose":
                        values[:, -1] = 1
                    demo.create_dataset(f"{group}/articulation/robot/{field}", data=values)
            demo.create_dataset("success", data=[episode["success"]])


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(frozen, "PINNED_DIR", tmp_path / "pins")
    pin = _frozen_rows(tmp_path)
    monkeypatch.syspath_prepend(str(REPO / "scripts/evaluation"))
    cli = _tool("validate_leaderboard_submission", REPO / "scripts/evaluation/validate_leaderboard_submission.py")
    manifest_path = tmp_path / "trusted_manifest.json"
    manifest_path.write_text(json.dumps({**MANIFEST, "tasks": {"m1": MANIFEST["tasks"]["m1"]}}))
    entry = copy.deepcopy(MANIFEST["tasks"]["m1"])
    entry["models"] = {"gr00t": entry["models"]["gr00t"]}
    results = _results(tmp_path, entry, {"train": 80, "ga_s": 40}, file_sha256(pin))
    submission = json.loads((REPO / "docs/leaderboard/submission.example.json").read_text())
    submission["benchmark"]["manifest_sha256"] = file_sha256(manifest_path)
    submission["benchmark"]["eval_version"] = "v1-plus"
    submission["benchmark"]["evaluation_sha256"] = {"m1": evaluation_contract_sha256(MANIFEST, "m1")}
    model_path = tmp_path / "model_entry.json"
    model_path.write_text(json.dumps({"key": "gr00t"}))
    submission["model_entry"] = {"path": "model_entry.json", "sha256": file_sha256(model_path)}
    submission["checkpoints"]["m1"]["label"] = "gr00t-step-1"
    card = tmp_path / "method_card.md"
    card.write_text("Synthetic integration fixture; not a real policy result.\n")
    submission["method_card"]["sha256"] = file_sha256(card)
    submission["runs"] = []
    for cell, directory in results:
        composed = directory / "composed_experiment.yaml"
        composed.write_text("runs: {}\n")
        manifest = json.loads((directory / "run_manifest.json").read_text())
        manifest.update(record_dataset=True, record_camera_video=True, eval_version="v1-plus",
                        model_entry=model_record(MODELS_BY_KEY["gr00t"]),
                        evaluation_contract_sha256=evaluation_contract_sha256(MANIFEST, "m1"),
                        code={"revision": "0" * 40, "dirty": False}, policy={"policy_variant": "gr00t"},
                        composed_sha256=file_sha256(composed), benchmark_manifest_sha256=file_sha256(manifest_path))
        (directory / "run_manifest.json").write_text(json.dumps(manifest))
        result_path = directory / "arena/arena_experiment_result.json"
        result = json.loads(result_path.read_text())
        for name, run in result["runs"].items():
            episodes = run["rebuilds"][0]["episodes"]
            for episode in episodes:
                row = episode["replay_source_record_index"]
                episode.update(env_id=row % 2, episode_in_env=row // 2, episode_length=2)
                episode["evaluation_settings"]["code"].update(revision="0" * 40, dirty=False)
                episode["evaluation_settings"]["success_criteria"] = {"func": "synthetic.success", "params": {"threshold": 0.1}}
            output = directory / "arena" / name
            output.mkdir()
            _write_trajectory(output / f"dataset_{name}_rebuild0.hdf5", episodes)
            for episode in episodes:
                filename = (f"robot-cam-rebuild0-env{episode['env_id']}-robot_pov_left_cam_rgb-"
                            f"episode-{episode['episode_in_env']}.mp4")
                (output / filename).write_bytes(b"synthetic video placeholder")
            (output / "robot-cam-rebuild0-env0-wrist_left_cam_rgb-episode-0.mp4").write_bytes(b"optional wrist")
        result_path.write_text(json.dumps(result))
        listed = cli.inventory(directory)
        listed["directory"] = str(directory.relative_to(tmp_path))
        submission["runs"].append(listed)
    candidate = cli.freeze_reference(manifest_path, tmp_path, "m1", "gr00t", results)
    reference_path("m1").write_text(json.dumps(candidate))
    path = tmp_path / "submission.json"
    path.write_text(json.dumps(submission))
    return cli, path, manifest_path, submission


def _validate(bundle, capsys):
    cli, path, manifest, submission = bundle
    path.write_text(json.dumps(submission))
    status = cli.main(["validate", str(path), "--manifest", str(manifest), "--conditions-root", str(path.parent)])
    captured = capsys.readouterr()
    return status, captured


def _write_manifest(bundle, manifest):
    """Update the operator manifest and entrant declarations, leaving the reference pin untouched."""
    _, path, manifest_path, submission = bundle
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2))
    digest = file_sha256(manifest_path)
    submission["benchmark"]["manifest_sha256"] = digest
    submission["benchmark"]["evaluation_sha256"] = {task: evaluation_contract_sha256(manifest, task) for task in manifest["tasks"]}
    for run in submission["runs"]:
        metadata_path = path.parent / run["directory"] / "run_manifest.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["benchmark_manifest_sha256"] = digest
        metadata["evaluation_contract_sha256"] = evaluation_contract_sha256(manifest, run["task"])
        metadata_path.write_text(json.dumps(metadata))
        run["manifest_sha256"] = file_sha256(metadata_path)


def test_submission_scores_one_model_and_joins_every_success_and_failure(bundle, capsys):
    status, output = _validate(bundle, capsys)
    assert status == 0, output.err
    report = json.loads(output.out)
    assert report["tasks"]["m1"]["status"] == "complete"
    assert report["tasks"]["m1"]["models"]["gr00t"]["macro_sr"] == pytest.approx(0.5)
    assert report["overall_sr"] == pytest.approx(0.5)
    assert report["status"] == "validated"
    assert report["verification"] == "artifact validation only"
    assert report["evaluation_settings"]["status"] == "matched release references"
    assert report["evaluation_settings"]["references"]["m1"]["sha256"] == file_sha256(reference_path("m1"))
    assert len(report["trajectories"]) == 240
    assert sum(row["success"] for row in report["trajectories"]) == 120
    assert all(row["control_dt_s"] == pytest.approx(1 / 60) for row in report["trajectories"])
    assert report["trajectories"][0]["frozen_row_index"] == 0


def test_inventory_accepts_a_relative_run_path(bundle, capsys, monkeypatch):
    cli, path, _, submission = bundle
    monkeypatch.chdir(path.parent)
    run = submission["runs"][0]
    assert cli.main(["inventory", "--run", run["directory"]]) == 0
    assert json.loads(capsys.readouterr().out) == run


@pytest.mark.parametrize("damage, message", [
    ("missing_failure", "missing trajectories"),
    ("nan", "non-finite"),
    ("length", "trajectory length differs"),
    ("states", "missing trajectory field states"),
    ("verdict", "success differs"),
    ("identity", "duplicate trajectory"),
    ("external", "HDF5 links"),
    ("hash", "SHA-256 mismatch"),
])
def test_invalid_trajectory_refuses_the_submission(bundle, capsys, damage, message):
    _, path, _, submission = bundle
    run = submission["runs"][0]
    artifact = run["trajectories"][0]
    trajectory = path.parent / run["directory"] / artifact["path"]
    with h5py.File(trajectory, "a") as handle:
        demo = handle["data/demo_0"]
        if damage == "missing_failure":
            name = next(name for name, group in handle["data"].items() if not group["success"][0])
            del handle["data"][name]
        elif damage == "nan":
            demo["actions"][0, 0] = np.nan
        elif damage == "length":
            demo.attrs["num_samples"] = 1
        elif damage == "states":
            del demo["states"]
        elif damage == "verdict":
            demo["success"][0] = not demo["success"][0]
        elif damage == "identity":
            for field in ("env_id", "episode_in_env"):
                handle[f"data/demo_1/episode_id/{field}"][0] = demo[f"episode_id/{field}"][0]
        elif damage == "external":
            demo["external"] = h5py.ExternalLink("outside.hdf5", "/state")
        else:
            demo["actions"][0, 0] = 2
    if damage != "hash":
        artifact["sha256"] = file_sha256(trajectory)
    status, output = _validate(bundle, capsys)
    assert status == 1
    assert message in output.err


def test_missing_cell_and_escaping_artifact_are_refused(bundle, capsys):
    submission = bundle[3]
    removed = submission["runs"].pop()
    status, output = _validate(bundle, capsys)
    assert status == 1 and "incomplete" in output.err
    submission["runs"].append(removed)
    submission["runs"][0]["trajectories"][0]["path"] = "../../../../../outside.hdf5"
    status, output = _validate(bundle, capsys)
    assert status == 1 and "escapes bundle" in output.err


def test_v1_scoring_keeps_evidence_for_all_submitted_rollouts(bundle, capsys):
    bundle[3]["benchmark"]["eval_version"] = "v1"
    status, output = _validate(bundle, capsys)
    assert status == 0, output.err
    report = json.loads(output.out)
    assert report["tasks"]["m1"]["models"]["gr00t"]["cells"]["train"]["n"] == 60
    assert len(report["trajectories"]) == 240


def test_overall_uses_the_operator_suite_and_rejects_policy_changes(bundle, capsys):
    cli, path, manifest_path, submission = bundle
    manifest = json.loads(manifest_path.read_text())
    manifest["tasks"] = {"m1": manifest["tasks"]["m1"]}
    _write_manifest(bundle, manifest)
    status, output = _validate(bundle, capsys)
    assert status == 0, output.err
    report = json.loads(output.out)
    assert report["coverage"]["complete"] is True
    assert report["overall_sr"] == pytest.approx(0.5)

    run = submission["runs"][1]
    metadata_path = path.parent / run["directory"] / "run_manifest.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["policy"]["denoising_steps"] = 100
    metadata_path.write_text(json.dumps(metadata))
    run["manifest_sha256"] = file_sha256(metadata_path)
    status, output = _validate(bundle, capsys)
    assert status == 1 and "settings change" in output.err


@pytest.mark.parametrize("field", ["physics", "terminations", "events", "success_criteria", "missing_criteria"])
def test_changed_evaluation_settings_are_refused_even_when_all_rows_agree(bundle, capsys, field):
    _, path, _, submission = bundle
    for run in submission["runs"]:
        result_path = path.parent / run["directory"] / "arena/arena_experiment_result.json"
        result = json.loads(result_path.read_text())
        for arena_run in result["runs"].values():
            for rebuild in arena_run["rebuilds"]:
                for episode in rebuild["episodes"]:
                    settings = episode["evaluation_settings"]
                    if field == "missing_criteria":
                        del settings["success_criteria"]
                    else:
                        settings[field]["unapproved_setting"] = 999
        result_path.write_text(json.dumps(result))
        run["result_sha256"] = file_sha256(result_path)
    status, output = _validate(bundle, capsys)
    assert status == 1
    assert ("success_criteria" if field == "missing_criteria" else "differ from release reference") in output.err


@pytest.mark.parametrize("damage", ["missing", "evaluation", "frozen", "cell", "row", "digest", "old_version"])
def test_missing_stale_or_partial_release_references_are_refused(bundle, capsys, damage):
    pin = reference_path("m1")
    reference = json.loads(pin.read_text())
    if damage == "missing":
        pin.unlink()
    else:
        if damage in {"evaluation", "frozen"}:
            reference[damage + "_sha256"] = "f" * 64
        elif damage == "old_version":
            reference["version"] = 1
            reference["manifest_sha256"] = reference.pop("evaluation_sha256")
        elif damage == "cell":
            reference["rows"].pop("ga_s")
        elif damage == "row":
            next(iter(reference["rows"]["train"].values())).pop()
        else:
            next(iter(reference["rows"]["train"].values()))[0] = "invalid"
        pin.write_text(json.dumps(reference))
    status, output = _validate(bundle, capsys)
    assert status == 1
    assert "reference settings" in output.err


def test_reference_cli_publishes_only_complete_candidates_without_clobbering(bundle, capsys):
    cli, path, manifest, submission = bundle
    output = path.parent / "reference_candidate.json"
    command = ["freeze-reference", "--task", "m1", "--model", "gr00t",
               "--manifest", str(manifest), "--conditions-root", str(path.parent), "--json", str(output),
               "--results", *[f"{run['cell']}={path.parent / run['directory']}" for run in submission["runs"]]]
    assert cli.main(command) == 0, capsys.readouterr().err
    candidate = json.loads(output.read_text())
    assert candidate == json.loads(reference_path("m1").read_text())
    assert len(candidate["provenance"]["runs"]) == 2
    assert candidate["provenance"]["manifest_sha256"] == file_sha256(manifest)
    assert "manifest_sha256" not in candidate
    assert cli.main(command[:-1]) == 1
    assert "every v1-plus row" in capsys.readouterr().err
    output.write_text("existing reviewed evidence")
    assert cli.main(command) == 1
    assert output.read_text() == "existing reviewed evidence"
    capsys.readouterr()
    command[command.index("--json") + 1] = str(reference_path("m1"))
    assert cli.main(command) == 1
    assert "outside the pin" in capsys.readouterr().err


def test_reference_comparison_uses_original_rows_and_checks_unscored_v1_plus_rows(bundle, capsys):
    _, path, manifest_path, submission = bundle
    submission["benchmark"]["eval_version"] = "v1"
    for run in submission["runs"]:
        directory = path.parent / run["directory"]
        metadata_path = directory / "run_manifest.json"
        metadata = json.loads(metadata_path.read_text())
        result_path = directory / "arena/arena_experiment_result.json"
        result = json.loads(result_path.read_text())
        metadata["row_indices"] = {}
        for name, arena_run in result["runs"].items():
            episodes = arena_run["rebuilds"][0]["episodes"]
            metadata["row_indices"][name] = list(reversed(range(len(episodes))))
            for episode in episodes:
                episode["replay_source_record_index"] = len(episodes) - 1 - episode["replay_source_record_index"]
        metadata_path.write_text(json.dumps(metadata))
        result_path.write_text(json.dumps(result))
        run["manifest_sha256"] = file_sha256(metadata_path)
        run["result_sha256"] = file_sha256(result_path)
    status, output = _validate(bundle, capsys)
    assert status == 0, output.err

    run = submission["runs"][0]
    directory = path.parent / run["directory"]
    metadata = json.loads((directory / "run_manifest.json").read_text())
    result_path = directory / "arena/arena_experiment_result.json"
    result = json.loads(result_path.read_text())
    chosen = frozen.pinned_version_rows(json.loads(manifest_path.read_text()), "m1", run["cell"], "v1")
    for name, arena_run in result["runs"].items():
        episode = next(ep for ep in arena_run["rebuilds"][0]["episodes"]
                       if metadata["row_indices"][name][ep["replay_source_record_index"]] not in chosen[name])
        episode["evaluation_settings"]["physics"]["solver_iterations"] = 999
        break
    result_path.write_text(json.dumps(result))
    run["result_sha256"] = file_sha256(result_path)
    status, output = _validate(bundle, capsys)
    assert status == 1
    assert "differ from release reference" in output.err


@pytest.mark.parametrize("change", ["models", "sensitivity", "asset_version_contracts", "asset_major_contracts", "formatting"])
def test_unrelated_manifest_changes_keep_existing_reference_pins(bundle, capsys, change):
    manifest = json.loads(bundle[2].read_text())
    entry = manifest["tasks"]["m1"]
    pin_bytes = reference_path("m1").read_bytes()
    if change == "models":
        entry["models"]["new_entrant"] = copy.deepcopy(entry["models"]["gr00t"])
    elif change == "sensitivity":
        entry["sensitivity"] = {"reviewed_metadata": "updated"}
    elif change == "asset_version_contracts":
        entry["asset_version_contracts"]["m1_drawer"]["lg_fh25eae_detergent_drawer"]["version"] = "3.0.1"
    elif change == "asset_major_contracts":
        entry.setdefault("asset_major_contracts", {})["unused_axis"] = {"unused_object": 1}
    _write_manifest(bundle, manifest)
    status, output = _validate(bundle, capsys)
    assert status == 0, output.err
    assert reference_path("m1").read_bytes() == pin_bytes


@pytest.mark.parametrize("field", ["cells", "policy_execution", "scoring_environment", "legacy_values", "other_task_fields"])
def test_changed_task_evaluation_contract_requires_a_new_reference(bundle, capsys, field):
    manifest = json.loads(bundle[2].read_text())
    entry = manifest["tasks"]["m1"]
    if field == "cells":
        entry["cells"]["train"]["declared"]["episode_length_s"] += 1
    elif field == "policy_execution":
        entry[field]["policy"]["policy_hz"] = 15
    elif field == "scoring_environment":
        entry[field] = {"rby1_control_hz": 30}
    elif field == "legacy_values":
        entry[field] = {"start_pose": {"old": "teleop_ready"}}
    else:
        entry[field] = ["another_task_option"]
    _write_manifest(bundle, manifest)
    status, output = _validate(bundle, capsys)
    assert status == 1
    assert "reference settings do not match the task evaluation contract" in output.err


def test_asset_compatibility_is_still_enforced_without_invalidating_the_reference(bundle, capsys):
    manifest = json.loads(bundle[2].read_text())
    manifest["tasks"]["m1"]["asset_major_contracts"] = {"m1_drawer": {"lg_fh25eae_detergent_drawer": 4}}
    _write_manifest(bundle, manifest)
    status, output = _validate(bundle, capsys)
    assert status == 1
    assert "asset MAJOR" in output.err
    assert "reference settings" not in output.err


def test_model_metadata_changes_do_not_require_rewriting_submitted_evidence(bundle, capsys):
    manifest_path = bundle[2]
    manifest = json.loads(manifest_path.read_text())
    manifest["tasks"]["m1"]["sensitivity"] = {"updated": True}
    manifest_path.write_text(json.dumps(manifest))
    status, output = _validate(bundle, capsys)
    assert status == 0, output.err


@pytest.mark.parametrize("section, field, value", [
    ("training", "track", "open-data"),
    ("model", "scope", "specialist"),
    ("training", "additional_data", True),
])
def test_only_standard_data_generalist_submissions_are_accepted(bundle, capsys, section, field, value):
    bundle[3][section][field] = value
    status, output = _validate(bundle, capsys)
    assert status == 1
    assert "Submission refused" in output.err


def test_every_submitted_task_requires_the_same_checkpoint(bundle, capsys):
    manifest = json.loads(bundle[2].read_text())
    manifest["tasks"]["m2"] = copy.deepcopy(MANIFEST["tasks"]["m2"])
    _write_manifest(bundle, manifest)
    submission = bundle[3]
    submission["runs"].append({**copy.deepcopy(submission["runs"][0]), "task": "m2"})
    submission["checkpoints"]["m2"] = {**submission["checkpoints"]["m1"], "sha256": "e" * 64}
    status, output = _validate(bundle, capsys)
    assert status == 1
    assert "same checkpoint on every task" in output.err


def test_partial_suite_is_refused(bundle, capsys):
    manifest = json.loads(bundle[2].read_text())
    manifest["tasks"]["m2"] = copy.deepcopy(MANIFEST["tasks"]["m2"])
    _write_manifest(bundle, manifest)
    status, output = _validate(bundle, capsys)
    assert status == 1 and "full trusted task suite" in output.err


@pytest.fixture
def community_bundle(bundle):
    from dexbench_arena.benchmark.models import config_model
    cli, path, manifest_path, submission = bundle
    key = "outside-generalist"
    policy = dict(MODELS_BY_KEY["gr00t"].policy)
    policy["policy_variant"] = key
    config = {"key": key, "label": "Outside generalist", "group": "community", "policy": policy}
    record = model_record(config_model(config))
    model_path = path.parent / "model_entry.json"
    model_path.write_text(json.dumps(config))
    submission["model_entry"]["sha256"] = file_sha256(model_path)
    submission["model"]["key"] = key
    for run in submission["runs"]:
        directory = path.parent / run["directory"]
        metadata_path = directory / "run_manifest.json"
        metadata = json.loads(metadata_path.read_text())
        metadata.update(model=key, model_entry=record, group="community",
                        checkpoint_sha256=submission["checkpoints"]["m1"]["sha256"])
        metadata["policy"]["policy_variant"] = key
        metadata_path.write_text(json.dumps(metadata))
        run["manifest_sha256"] = file_sha256(metadata_path)
        result_path = directory / "arena/arena_experiment_result.json"
        result = json.loads(result_path.read_text())
        for arena_run in result["runs"].values():
            arena_run["policy_variant"] = key
            for rebuild in arena_run["rebuilds"]:
                for episode in rebuild["episodes"]:
                    episode["evaluation_settings"]["policy"]["policy_variant"] = key
        result_path.write_text(json.dumps(result))
        run["result_sha256"] = file_sha256(result_path)
    return bundle


def test_outside_model_validates_without_changing_the_release_manifest(community_bundle, capsys):
    status, output = _validate(community_bundle, capsys)
    assert status == 0, output.err
    report = json.loads(output.out)
    assert report["overall_sr"] == pytest.approx(0.5)
    assert report["tasks"]["m1"]["models"]["outside-generalist"]["rank_eligible"]
    assert not report["tasks"]["m1"]["models"]["outside-generalist"]["headline"]
    assert "outside-generalist" not in json.loads(community_bundle[2].read_text())["tasks"]["m1"]["models"]
    _, path, _, submission = community_bundle
    run = submission["runs"][0]
    metadata_path = path.parent / run["directory"] / "run_manifest.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["model_entry"]["policy"]["action_chunk_length"] = 999
    metadata_path.write_text(json.dumps(metadata))
    run["manifest_sha256"] = file_sha256(metadata_path)
    status, output = _validate(community_bundle, capsys)
    assert status == 1 and "model_entry differs" in output.err


def test_registered_models_rank_beside_baselines_without_changing_completeness(community_bundle):
    from dexbench_arena.benchmark.leaderboard_models import with_registered_models
    cli, path, manifest_path, submission = community_bundle
    scoring = _tool("registered_scoring", REPO / "scripts/evaluation/score_benchmark_cells.py")
    manifest = json.loads(manifest_path.read_text())
    entry = manifest["tasks"]["m1"]
    entry["models"] = {"gr00t": entry["models"]["gr00t"]}
    results = [(run["cell"], path.parent / run["directory"]) for run in submission["runs"]]
    for cell, directory in list(results):
        baseline = path.parent / (cell + "-baseline")
        shutil.copytree(directory, baseline)
        metadata_path = baseline / "run_manifest.json"
        metadata = json.loads(metadata_path.read_text())
        metadata.update(model="gr00t", model_entry=model_record(MODELS_BY_KEY["gr00t"]), group="headline")
        metadata_path.write_text(json.dumps(metadata))
        result_path = baseline / "arena/arena_experiment_result.json"
        result_path.write_text(result_path.read_text().replace("outside-generalist", "gr00t"))
        results.append((cell, baseline))
    manifest = scoring.with_config_models(manifest, "m1", results)
    registry = path.parent / "models"
    registry.mkdir()
    config = json.loads((path.parent / "model_entry.json").read_text())
    checkpoint = submission["checkpoints"]["m1"]
    (registry / "outside-generalist.json").write_text(json.dumps({"schema_version": 1, "group": "registered",
        "model_entry": config, "checkpoint_sha256": checkpoint["sha256"], "checkpoint_label": checkpoint["label"]}))
    manifest = with_registered_models(manifest, "m1", results, registry)
    digest, sources = frozen.load_frozen_set(frozen.pinned_path("m1"), manifest, "m1", path.parent)
    episodes = scoring.collect(manifest, "m1", results, digest)
    report = scoring.score(manifest, "m1", episodes, sources, digest, "v1-plus")
    assert report["status"] == "complete"
    assert set(report["models"]) == {"gr00t", "outside-generalist"}
    assert report["models"]["outside-generalist"]["rank_eligible"]
    assert "outside-generalist:" in scoring.format_report(manifest, "m1", report)
    output = path.parent / "registered-score.json"
    assert scoring.main(["--manifest", str(manifest_path), "score", "--task", "m1", "--version", "v1-plus",
                         "--root", str(path.parent), "--model-registry", str(registry), "--json", str(output),
                         "--results", *[f"{cell}={directory}" for cell, directory in results]]) == 0
    assert json.loads(output.read_text())["models"]["outside-generalist"]["group"] == "registered"
    del episodes["ga_s"]["outside-generalist"]
    report = scoring.score(manifest, "m1", episodes, sources, digest, "v1-plus")
    assert report["status"] == "complete"
    assert not report["models"]["outside-generalist"]["complete"]
    assert not report["models"]["outside-generalist"]["rank_eligible"]
    assert "outside-generalist:" in scoring.format_report(manifest, "m1", report)


def test_v1_references_cover_only_observed_rows(bundle, capsys):
    cli, path, manifest_path, submission = bundle
    manifest = json.loads(manifest_path.read_text())
    results = []
    for run in submission["runs"]:
        directory = path.parent / run["directory"]
        metadata_path = directory / "run_manifest.json"
        metadata = json.loads(metadata_path.read_text())
        chosen = frozen.pinned_version_rows(manifest, "m1", run["cell"], "v1")
        metadata.update(eval_version="v1", row_indices=chosen)
        metadata_path.write_text(json.dumps(metadata))
        result_path = directory / "arena/arena_experiment_result.json"
        result = json.loads(result_path.read_text())
        for name, arena_run in result["runs"].items():
            episodes = [ep for ep in arena_run["rebuilds"][0]["episodes"] if ep["replay_source_record_index"] in chosen[name]]
            for ep in episodes:
                ep["replay_source_record_index"] = chosen[name].index(ep["replay_source_record_index"])
            arena_run["rebuilds"][0]["episodes"] = episodes
            _write_trajectory(directory / "arena" / name / f"dataset_{name}_rebuild0.hdf5", episodes)
        result_path.write_text(json.dumps(result))
        listed = cli.inventory(directory)
        listed["directory"] = run["directory"]
        run.update(listed)
        results.append((run["cell"], directory))
    candidate = cli.freeze_reference(manifest_path, path.parent, "m1", "gr00t", results, "v1")
    assert candidate["version"] == 3
    assert sum(value is not None for runs in candidate["rows"].values() for values in runs.values() for value in values) == 120
    reference_path("m1").write_text(json.dumps(candidate))
    submission["benchmark"]["eval_version"] = "v1"
    status, output = _validate(bundle, capsys)
    assert status == 0, output.err
    from dexbench_arena.benchmark.reference_settings import check_reference
    from dexbench_arena.benchmark import Refusal
    name, values = next(iter(candidate["rows"]["train"].items()))
    with pytest.raises(Refusal, match="no reference settings"):
        check_reference(manifest, "m1", "gr00t", {"train": {"gr00t": [{"_row": name,
            "replay_source_record_index": values.index(None)}]}}, candidate["rows"])


@pytest.mark.parametrize("dirty", [True, None])
def test_dirty_or_unknown_evaluator_checkout_is_refused(bundle, capsys, dirty):
    _, path, _, submission = bundle
    run = submission["runs"][0]
    metadata_path = path.parent / run["directory"] / "run_manifest.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["code"]["dirty"] = dirty
    metadata_path.write_text(json.dumps(metadata))
    run["manifest_sha256"] = file_sha256(metadata_path)
    status, output = _validate(bundle, capsys)
    assert status == 1 and "clean evaluator checkout" in output.err


def test_model_entry_schema_is_checked_before_run_validation(community_bundle, capsys):
    _, path, _, submission = community_bundle
    model_path = path.parent / "model_entry.json"
    entry = json.loads(model_path.read_text())
    entry["key"] = "not_a_registry_slug"
    model_path.write_text(json.dumps(entry))
    submission["model_entry"]["sha256"] = file_sha256(model_path)
    status, output = _validate(community_bundle, capsys)
    assert status == 1 and "model_entry:" in output.err


@pytest.mark.parametrize("success", [False, True])
@pytest.mark.parametrize("damage", ["unlisted", "missing", "empty", "camera", "run", "rebuild", "env", "episode"])
def test_submission_requires_designated_video_for_every_episode(bundle, capsys, success, damage):
    _, path, _, submission = bundle
    run = submission["runs"][0]
    directory = path.parent / run["directory"]
    result = json.loads((directory / "arena/arena_experiment_result.json").read_text())
    name, result_run = next(iter(result["runs"].items()))
    rebuild = result_run["rebuilds"][0]
    episode = next(row for row in rebuild["episodes"] if row["success"] is success)
    filename = (f"robot-cam-rebuild{rebuild['index']}-env{episode['env_id']}-"
                f"robot_pov_left_cam_rgb-episode-{episode['episode_in_env']}.mp4")
    relative = Path("arena") / name / filename
    artifact = next(video for video in run["videos"] if video["path"] == str(relative))
    video = directory / relative
    if damage == "unlisted":
        run["videos"].remove(artifact)
    elif damage == "missing":
        video.unlink()
    elif damage == "empty":
        video.write_bytes(b"")
        artifact["sha256"] = file_sha256(video)
    else:
        replacements = {
            "camera": ("robot_pov_left_cam_rgb", "robot_pov_right_cam_rgb"),
            "run": (f"arena/{name}/", "arena/another_run/"),
            "rebuild": (f"rebuild{rebuild['index']}-", "rebuild999-"),
            "env": (f"-env{episode['env_id']}-", "-env999-"),
            "episode": (f"-episode-{episode['episode_in_env']}.mp4", "-episode-999.mp4"),
        }
        old, new = replacements[damage]
        artifact["path"] = artifact["path"].replace(old, new)
        destination = directory / artifact["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        video.rename(destination)
    status, output = _validate(bundle, capsys)
    assert status == 1
    expected = "missing artifact" if damage == "missing" else "empty rollout video" if damage == "empty" else "missing required robot_pov_left_cam_rgb video"
    assert expected in output.err
