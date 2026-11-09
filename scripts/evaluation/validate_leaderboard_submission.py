#!/usr/bin/env python3
"""Inventory or validate a local leaderboard submission, without launching the simulator.

See docs/Leaderboard_Submission.md. Validation recalculates scores from original
episode results and checks every trajectory; publication requires maintainer review.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from score_benchmark_cells import collect, run_rows, score_submission, with_config_models

from dexbench_arena.benchmark import Refusal
from dexbench_arena.benchmark.frozen import load_frozen_set, pinned_path
from dexbench_arena.benchmark.run_manifest import read_run_manifest
from dexbench_arena.benchmark.reference_settings import evaluation_contract_sha256, load_reference, reference_candidate, reference_path
from dexbench_arena.benchmark.submission import bundle_path, checked_file, load_submission, validate_run
from dexbench_arena.evaluation.artifacts import file_sha256, publish_new

CONFIGS = Path(__file__).resolve().parent / "experiment_configs" / "benchmark"


def inventory(directory: Path) -> dict:
    """List original run files and their hashes; do not infer episode identity from demo order."""
    directory_label = str(directory)
    directory = directory.resolve()
    metadata = read_run_manifest(directory)
    result = directory / "arena/arena_experiment_result.json"
    runs = json.loads(result.read_text())["runs"]
    trajectories, videos = [], []
    for run_name, run in runs.items():
        rebuilds = {rebuild["index"] for rebuild in run["rebuilds"]}
        run_directory = bundle_path(directory, f"arena/{run_name}")
        for path in sorted(run_directory.rglob("*.hdf5")):
            match = re.search(r"_rebuild(\d+)(?:_|$)", path.stem)
            if match is None or int(match[1]) not in rebuilds:
                raise Refusal(f"{path}: cannot identify rebuild; supply an explicit reviewed inventory")
            trajectories.append({"path": str(path.relative_to(directory)), "sha256": file_sha256(path),
                                 "run": run_name, "rebuild": int(match[1])})
        videos.extend({"path": str(path.relative_to(directory)), "sha256": file_sha256(path)}
                      for path in sorted(run_directory.rglob("*.mp4")))
    return {"task": metadata["task"], "cell": metadata["cell"], "directory": directory_label,
            "manifest_sha256": file_sha256(directory / "run_manifest.json"),
            "result_sha256": file_sha256(result), "trajectories": trajectories, "videos": videos}


def validate(path: Path, manifest_path: Path, conditions_root: Path) -> dict:
    submission = load_submission(path)
    root = path.resolve().parent
    manifest = json.loads(manifest_path.read_text())
    checked_file(root, submission["method_card"])
    model_entry = json.loads(checked_file(root, submission["model_entry"]).read_text())
    contracts = {task: evaluation_contract_sha256(manifest, task) for task in manifest["tasks"]}
    if submission["benchmark"]["evaluation_sha256"] != contracts:
        raise Refusal("submission evaluation contracts differ from the trusted task suite")
    tasks = {run["task"] for run in submission["runs"]}
    if tasks != set(manifest["tasks"]):
        raise Refusal("a submission must cover the full trusted task suite")
    if set(submission["checkpoints"]) != tasks:
        raise Refusal("declare exactly one checkpoint per submitted task")
    if len({(c["sha256"], c["label"]) for c in submission["checkpoints"].values()}) != 1:
        raise Refusal("a generalist submission must use the same checkpoint on every task (hash and recorded label)")
    results, evidence, directories, policies = {}, [], set(), {}
    for run in submission["runs"]:
        directory, trajectories = validate_run(root, run, submission, manifest_path)
        if directory in directories:
            raise Refusal(f"duplicate run directory: {directory}")
        directories.add(directory)
        metadata = read_run_manifest(directory)
        policy = json.dumps(metadata["policy"], sort_keys=True)
        if policies.setdefault(run["task"], policy) != policy:
            raise Refusal(f"{run['task']}: policy settings change between cells or shards")
        for trajectory in trajectories:
            row_count = manifest["tasks"][run["task"]]["cells"][run["cell"]]["rows"][trajectory["run"]]["rows"]
            indices = run_rows(metadata, trajectory["run"], row_count)
            local_index = trajectory["replay_source_record_index"]
            if type(local_index) is not int or not 0 <= local_index < len(indices):
                raise Refusal("trajectory references an invalid replay row")
            evidence.append({**trajectory, "task": run["task"], "cell": run["cell"],
                             "directory": run["directory"], "frozen_row_index": indices[local_index]})
        results.setdefault(run["task"], []).append((run["cell"], directory))
    reports, references = {}, {}
    model = submission["model"]["key"]
    version = submission["benchmark"]["eval_version"]
    for task, task_results in sorted(results.items()):
        digest, sources = load_frozen_set(pinned_path(task), manifest, task, conditions_root)
        reference_digest, reference_rows = load_reference(manifest, task, digest)
        scoring_manifest = with_config_models(manifest, task, task_results)
        episodes = collect(scoring_manifest, task, task_results, digest)
        reports[task] = score_submission(scoring_manifest, task, model, episodes, sources, digest, version,
                                         reference_rows=reference_rows)
        references[task] = {"file": reference_path(task).name, "sha256": reference_digest}
    complete = tasks == set(manifest["tasks"])
    return {"status": "validated", "verification": "artifact validation only",
            "version": version, "model": submission["model"], "model_entry": model_entry, "data_track": submission["training"]["track"],
            "coverage": {"submitted": sorted(tasks), "required": sorted(manifest["tasks"]), "complete": complete},
            "overall_sr": sum(r["models"][model]["macro_sr"] for r in reports.values()) / len(reports) if complete else None,
            "evaluation_settings": {"status": "matched release references", "references": references},
            "tasks": reports, "trajectories": evidence}


def freeze_reference(manifest_path: Path, conditions_root: Path, task: str, model: str,
                     results: list[tuple[str, Path]], version: str = "v1-plus") -> dict:
    """Make an operator candidate from full trusted evaluation runs, never install a pin."""
    manifest = json.loads(manifest_path.read_text())
    digest, sources = load_frozen_set(pinned_path(task), manifest, task, conditions_root)
    episodes = collect(manifest, task, results, digest)
    candidate = reference_candidate(manifest, task, model, episodes, digest, version)
    score_submission(manifest, task, model, episodes, sources, digest, version,
                     reference_rows=candidate["rows"])
    candidate["provenance"] = {"manifest_sha256": file_sha256(manifest_path), "model": model, "runs": [
        {"cell": cell, "manifest_sha256": file_sha256(directory / "run_manifest.json"),
         "result_sha256": file_sha256(directory / read_run_manifest(directory)["arena_output"] / "arena_experiment_result.json")}
        for cell, directory in results]}
    return candidate


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("inventory", help="Print one runs[] entry from a completed run directory.")
    listing.add_argument("--run", required=True, type=Path)
    check = commands.add_parser("validate", help="Validate evidence and recompute one entrant's scores.")
    check.add_argument("submission", type=Path)
    check.add_argument("--manifest", type=Path, default=CONFIGS / "benchmark_manifest.json",
                       help="Operator-selected trusted manifest, never a participant-supplied replacement.")
    check.add_argument("--conditions-root", type=Path, default=Path("."), help="Root of the official downloaded frozen rows.")
    check.add_argument("--json", type=Path, help="Publish report without replacing an existing different file.")
    freeze = commands.add_parser("freeze-reference", help="Operator only: create a candidate from trusted complete v1 or v1-plus runs.")
    freeze.add_argument("--version", choices=("v1", "v1-plus"), default="v1-plus")
    freeze.add_argument("--task", required=True)
    freeze.add_argument("--model", required=True)
    freeze.add_argument("--results", nargs="+", required=True, metavar="CELL=DIRECTORY")
    freeze.add_argument("--manifest", type=Path, default=CONFIGS / "benchmark_manifest.json")
    freeze.add_argument("--conditions-root", type=Path, default=Path("."))
    freeze.add_argument("--json", type=Path, required=True, help="Candidate output; review and commit beside the frozen pin.")
    args = parser.parse_args(argv)
    try:
        if args.command == "inventory":
            result = inventory(args.run)
        elif args.command == "freeze-reference":
            results = [(cell, Path(directory)) for cell, directory in (item.split("=", 1) for item in args.results)]
            if args.json.resolve() == reference_path(args.task).resolve():
                raise Refusal("write a reference candidate outside the pin; review it before committing")
            result = freeze_reference(args.manifest, args.conditions_root, args.task, args.model, results, args.version)
        else:
            result = validate(args.submission, args.manifest, args.conditions_root)
        payload = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode()
        if args.command != "inventory" and args.json:
            publish_new(args.json, payload)
        else:
            print(payload.decode(), end="")
    except (Refusal, ValueError, KeyError, TypeError, OSError) as error:
        print(f"Submission refused: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
