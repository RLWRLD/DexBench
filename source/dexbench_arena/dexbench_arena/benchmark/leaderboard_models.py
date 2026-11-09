"""Data-only admission of config models and maintainer-owned leaderboard registrations."""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

from dexbench_arena.benchmark import Refusal
from dexbench_arena.benchmark.models import model_record, resolve_model
from dexbench_arena.benchmark.run_manifest import read_run_manifest


def submission_model(entry: dict) -> dict:
    """Normalize the same config block accepted by --model-entry, without importing a policy."""
    if not isinstance(entry, dict) or not isinstance(entry.get("key"), str):
        raise Refusal("model_entry needs a model key")
    try:
        return model_record(resolve_model(entry["key"] if set(entry) == {"key"} else entry))
    except (ValueError, TypeError, KeyError) as error:
        raise Refusal(f"model_entry: {error}") from error


def with_registered_models(manifest: dict, task: str, results: list[tuple[str, Path]], registry: Path | None) -> dict:
    """Promote only exact reviewed config entries/checkpoints, never participant-declared groups."""
    if registry is None:
        return manifest
    if not registry.is_dir():
        raise Refusal(f"missing leaderboard model registry: {registry}")
    result = copy.deepcopy(manifest)
    for path in sorted(registry.glob("*.json")):
        record = json.loads(path.read_text())
        if set(record) != {"schema_version", "group", "model_entry", "checkpoint_sha256", "checkpoint_label"} or record.get("schema_version") != 1 or record.get("group") != "registered":
            raise Refusal(f"invalid leaderboard registration: {path}")
        entry = submission_model(record["model_entry"])
        key = entry["key"]
        if path.name != f"{key}.json" or entry["registration"] != "config":
            raise Refusal(f"registration must name a config model: {path}")
        if re.fullmatch(r"[a-f0-9]{64}", str(record["checkpoint_sha256"])) is None or not record["checkpoint_label"]:
            raise Refusal(f"invalid registered checkpoint: {path}")
        declared = result["tasks"][task]["models"].get(key)
        if declared is None:
            continue
        if declared.get("entry") != entry:
            raise Refusal(f"{key}: model entry differs from the reviewed registration")
        for _, directory in results:
            run = read_run_manifest(directory)
            if run.get("model") == key and (run.get("checkpoint") != record["checkpoint_label"] or
                                          run.get("checkpoint_sha256") != record["checkpoint_sha256"]):
                raise Refusal(f"{key}: recorded checkpoint differs from the reviewed registration")
        declared["group"] = "registered"
    return result
