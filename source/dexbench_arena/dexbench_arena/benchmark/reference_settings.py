"""Operator-reviewed evaluation signatures, pinned beside each task's frozen rows."""

from __future__ import annotations

import json
import re
from functools import partial
from pathlib import Path

from dexbench_arena.benchmark import Refusal
from dexbench_arena.benchmark.frozen import pinned_path, pinned_version_rows
from dexbench_arena.benchmark.protocol import cell_rows
from dexbench_arena.evaluation.artifacts import file_sha256, recorded, sha256_bytes
from dexbench_arena.evaluation.evaluation_signature import (
    SUCCESS_CRITERIA_KEY,
    evaluation_signature,
    settings_problems,
)


EVALUATION_FIELDS = ("cells", "policy_execution", "scoring_environment", "legacy_values", "other_task_fields")
"""Task fields defining the reference evaluation contract; model and sensitivity metadata are independent."""


def evaluation_contract_sha256(manifest: dict, task: str) -> str:
    """Canonical hash of this task's evaluation contract, independent of JSON formatting."""
    entry = manifest["tasks"][task]
    contract = {field: entry.get(field) for field in EVALUATION_FIELDS}
    return sha256_bytes(json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def reference_path(task: str) -> Path:
    """Use a separate prefix so frozen-row readers never mistake signatures for row files."""
    return pinned_path(task).with_name(f"settings_db_{task}.json")


def row_signature(episode: dict, entry: dict) -> str:
    """Use the scorer's normalization, but require explicitly recorded success criteria."""
    settings = recorded(episode, "evaluation_settings")
    problems = settings_problems(settings)
    if problems:
        raise Refusal("reference settings: " + "; ".join(problems))
    if settings.get(SUCCESS_CRITERIA_KEY) is None:
        raise Refusal("reference settings require recorded success_criteria")
    return evaluation_signature(episode, entry.get("legacy_values"),
                                other_task_fields=entry.get("other_task_fields", ()))


def reference_candidate(manifest: dict, task: str, model: str, episodes: dict,
                        frozen_sha256: str, version: str = "v1-plus") -> dict:
    """Observed row signatures; unobserved v1-plus rows remain unknown in a v1 candidate."""
    entry = manifest["tasks"][task]
    rows = {cell: {name: [None] * row["rows"] for name, row in cfg["rows"].items()}
            for cell, cfg in entry["cells"].items()}
    for cell, models in episodes.items():
        if cell not in rows or set(models) != {model}:
            raise Refusal("reference runs must contain one model and only declared cells")
        for episode in models[model]:
            name, index = episode["_row"], episode.get("replay_source_record_index")
            if name not in rows[cell] or type(index) is not int or not 0 <= index < len(rows[cell][name]):
                raise Refusal(f"{task} {cell}: invalid reference row {name}:{index}")
            if rows[cell][name][index] is not None:
                raise Refusal(f"{task} {cell}: duplicate reference row {name}:{index}")
            rows[cell][name][index] = row_signature(episode, entry)
    for cell, cfg in entry["cells"].items():
        required = cell_rows({name: row["rows"] for name, row in cfg["rows"].items()}, None, version,
                             partial(pinned_version_rows, manifest, task, cell))
        if any(rows[cell][name][index] is None for name, indices in required.items() for index in indices):
            raise Refusal(f"reference runs must cover every {version} row of every cell")
    return {"version": 2 if version == "v1-plus" else 3, **({"eval_version": version} if version != "v1-plus" else {}), "task": task, "evaluation_sha256": evaluation_contract_sha256(manifest, task),
            "frozen_sha256": frozen_sha256, "rows": rows}


def load_reference(manifest: dict, task: str, frozen_sha256: str) -> tuple[str, dict]:
    """Read only the release checkout's pin; refuse stale, partial or malformed references."""
    path = reference_path(task)
    if not path.is_file():
        raise Refusal(f"{task}: missing release reference settings {path}; an operator must freeze and review trusted runs")
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or data.get("version") not in {2, 3} or (data["version"] == 3 and data.get("eval_version") != "v1"):
        raise Refusal(f"{task}: unsupported reference settings version")
    version = "v1" if data["version"] == 3 else "v1-plus"
    expected = {"task": task, "evaluation_sha256": evaluation_contract_sha256(manifest, task),
                "frozen_sha256": frozen_sha256}
    if not isinstance(data, dict) or any(data.get(key) != value for key, value in expected.items()):
        raise Refusal(f"{task}: reference settings do not match the task evaluation contract and frozen set (version 2 or 3 required)")
    rows = data.get("rows")
    cells = manifest["tasks"][task]["cells"]
    if not isinstance(rows, dict) or set(rows) != set(cells):
        raise Refusal(f"{task}: reference settings must list exactly the declared cells")
    for cell, cfg in cells.items():
        runs = rows[cell]
        if not isinstance(runs, dict) or set(runs) != set(cfg["rows"]):
            raise Refusal(f"{task} {cell}: reference settings must list exactly the declared runs")
        required = cell_rows({name: row["rows"] for name, row in cfg["rows"].items()}, None, version,
                             partial(pinned_version_rows, manifest, task, cell))
        for name, row in cfg["rows"].items():
            values = runs[name]
            if (not isinstance(values, list) or len(values) != row["rows"]
                    or any(value is not None and (not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None) for value in values)
                    or any(values[index] is None for index in required.get(name, ()))):
                raise Refusal(f"{task} {cell} {name}: reference settings require one signature per frozen row")
    return file_sha256(path), rows


def check_reference(manifest: dict, task: str, model: str, episodes: dict, rows: dict) -> None:
    """Compare every submitted episode, including extra v1-plus rows scored as v1."""
    entry = manifest["tasks"][task]
    for cell, models in episodes.items():
        for episode in models[model]:
            name, index = episode["_row"], episode.get("replay_source_record_index")
            signatures = rows.get(cell, {}).get(name, [])
            if type(index) is not int or not 0 <= index < len(signatures) or signatures[index] is None:
                raise Refusal(f"{task} {cell}: no reference settings for row {name}:{index}")
            if row_signature(episode, entry) != signatures[index]:
                raise Refusal(f"{task} {cell}: evaluation settings differ from release reference at {name}:{index}")
