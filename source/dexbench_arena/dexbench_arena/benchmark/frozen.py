# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""The frozen sets: every row file a task's cells (and, apart, its sweep) replay, with SHA-256 and row count."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path

from dexbench_arena.benchmark import Refusal
from dexbench_arena.benchmark.protocol import CONDITIONS_ROOT, EVAL_VERSIONS, version_rows
from dexbench_arena.evaluation.artifacts import file_sha256, publish_new, read_jsonl, sha256_bytes

PINNED_DIR = Path(__file__).resolve().parent / "frozen"
"""The checked-in frozen sets, one file per task and section."""


def digest_and_count(path: str | Path) -> tuple[str, int]:
    """SHA-256 and row count of one read of ``path``'s bytes."""
    digest, rows = read_jsonl(path)
    return digest, len(rows)


INPUTS_PIN = PINNED_DIR / "inputs.json"
"""The start-state sets the cells read beside their rows (DB-R1's evaluation layouts, DB-M5's piles):
``{directory: {"files", "sha256"}}`` by :func:`tree_digest`, each directory as the cells name it, under
``outputs/db_conditions/``. The rows bind each set already (a DB-R1 row's ``task_object.fit.layouts_sha256``, a
DB-M5 row's ``parts_bin_clutter_set_sha256``), but only once a build reads it; this pin lets ``download_datasets.sh
--eval-conditions`` refuse a revision that lacks a set, or holds another, before anything is installed. Not named
``db_*.json``, so nothing that reads every frozen set takes it for one."""


def tree_digest(directory: str | Path) -> tuple[str, int]:
    """SHA-256 of every file under ``directory``, and their count: the digest of one ``<path relative to directory>
    <file SHA-256>`` line per file, sorted by path, so a renamed, added, removed or edited file changes it."""
    root = Path(directory)
    files = sorted(path for path in root.rglob("*") if path.is_file())
    lines = "".join(f"{path.relative_to(root).as_posix()} {file_sha256(path)}\n" for path in files)
    return sha256_bytes(lines.encode()), len(files)


def input_problems(conditions_root: str | Path, pins: dict | None = None) -> list[str]:
    """Why the start-state sets under ``conditions_root`` (a tree's ``outputs/db_conditions``, or a download's
    ``eval/db_conditions``) are not the pinned ones (``pins``, by default :data:`INPUTS_PIN`'s): one line per set
    that is missing or whose files differ; empty when every set matches."""
    pins = json.loads(INPUTS_PIN.read_text()) if pins is None else pins
    problems = []
    for name, record in pins.items():
        directory = Path(conditions_root) / os.path.relpath(os.path.normpath(name), CONDITIONS_ROOT)
        if not directory.is_dir():
            problems.append(f"{name}: not in {conditions_root}")
        elif (found := tree_digest(directory)) != (record["sha256"], record["files"]):
            problems.append(f"{name}: {found[1]} files, sha256 {found[0][:12]}; {INPUTS_PIN.name} pins "
                            f"{record['files']} files, sha256 {record['sha256'][:12]}")
    return problems


SECTIONS = ("headline", "sensitivity")


def pinned_path(task: str, section: str = "headline") -> Path:
    """The checked-in frozen set of ``task``'s ``section``."""
    if section not in SECTIONS:
        raise ValueError(f"section {section!r} is not one of {SECTIONS}")
    return PINNED_DIR / (f"db_{task}.json" if section == "headline" else f"db_{task}_sensitivity.json")


def version_pin_path(task: str) -> Path:
    """The version rows pinned beside ``task``'s headline set: ``versions_db_<task>.json``, named so that nothing that
    reads every frozen set (``db_*.json``: :func:`check_frozen_row_file`, ``download_datasets.sh``) takes it for one.
    In ``frozen/`` itself rather than a subdirectory, since the package data pattern (``benchmark/frozen/*.json`` in
    ``setup.py``) already carries it and a new pattern there would need a new Arena base image."""
    return PINNED_DIR / f"versions_{pinned_path(task).name}"


def version_pin(manifest: dict, task: str, frozen_bytes: bytes) -> dict:
    """The version pin of ``task``'s headline set whose file holds ``frozen_bytes``: the set's name and SHA-256, and
    for each version that runs fewer than a cell's rows ``{row file: [frozen row indices]}`` (``protocol.version_rows``
    over the manifest's Runs, split over its objects first; a Run the version gives none is left out). ``score_benchmark_cells.py freeze`` writes it
    beside its candidate set, and it is committed with the set."""
    versions: dict[str, dict[str, list[int]]] = {}
    for version, count in EVAL_VERSIONS.items():
        for cell in manifest["tasks"][task]["cells"].values():
            quota = {name: row["rows"] for name, row in cell["rows"].items()}
            if count < sum(quota.values()):
                objects = {name: row["object"] for name, row in cell["rows"].items()}
                for name, rows in version_rows(quota, version, objects).items():
                    versions.setdefault(version, {})[cell["rows"][name]["path"]] = list(rows)
    return {"frozen_set": pinned_path(task).name, "frozen_sha256": sha256_bytes(frozen_bytes), "versions": versions}


def version_pin_bytes(pin: dict) -> bytes:
    """A version pin as it is committed: one line per row file, so a diff shows which file's rows changed."""
    versions = ",\n".join(
        f' {json.dumps(version)}: {{\n' + ",\n".join(f"  {json.dumps(name)}: {json.dumps(rows)}" for name, rows in files.items())
        + "\n }" for version, files in pin["versions"].items())
    return (f'{{"frozen_set": {json.dumps(pin["frozen_set"])}, "frozen_sha256": {json.dumps(pin["frozen_sha256"])},\n'
            f'"versions": {{\n{versions}\n}}}}\n').encode()


def pinned_version_rows(manifest: dict, task: str, cell: str, version: str) -> dict[str, tuple[int, ...]]:
    """``{run: frozen row indices}`` of ``task``'s ``cell`` at ``version`` as its version pin lists them, in the
    manifest's Run order. ``protocol.cell_rows`` reads it only for a version that runs fewer than the cell's rows, so
    a version that runs them all (v1-plus) needs no pin, and a stale or missing one does not refuse it."""
    path = version_pin_path(task)
    if not path.is_file():
        raise Refusal(f"{path}: missing, so no {version} rows are pinned for {task}; score_benchmark_cells.py freeze "
                      "writes them with the frozen set")
    pin = json.loads(path.read_text())
    frozen_sha256 = sha256_bytes(pinned_path(task).read_bytes())
    if pin.get("frozen_sha256") != frozen_sha256:
        raise Refusal(f"{path}: pinned for {pin.get('frozen_set')} at sha256 {str(pin.get('frozen_sha256'))[:12]}, the "
                      f"pinned set is {frozen_sha256[:12]}; the version rows are pinned again with the set "
                      "(score_benchmark_cells.py freeze writes both)")
    files = {os.path.normpath(name) for name in expected_rows(manifest, task)}
    entry = manifest["tasks"][task]["cells"][cell]
    runs = {os.path.normpath(row["path"]): name for name, row in entry["rows"].items()}
    unknown_versions = sorted(set(pin["versions"]) - set(EVAL_VERSIONS))
    if unknown_versions:
        raise Refusal(f"{path}: pins rows of {unknown_versions}, which are not versions ({sorted(EVAL_VERSIONS)})")
    if version not in pin["versions"]:
        raise Refusal(f"{path}: no {version} rows are pinned for {task} (it pins {sorted(pin['versions'])}); "
                      "score_benchmark_cells.py freeze writes them with the frozen set")
    by_file = pin["versions"][version]
    unknown = sorted(name for name in by_file if os.path.normpath(name) not in files)
    if unknown:
        raise Refusal(f"{path}: {version} rows of {unknown[:3]}, which are not row files of {task}'s headline set")
    chosen = {runs[os.path.normpath(name)]: tuple(rows) for name, rows in by_file.items() if os.path.normpath(name) in runs}
    for name, rows in chosen.items():
        count = entry["rows"][name]["rows"]
        if (not rows or list(rows) != sorted(set(rows))
                or not all(isinstance(i, int) and not isinstance(i, bool) and 0 <= i < count for i in rows)):
            raise Refusal(f"{path}: {task} {cell} {name}: {version} rows {list(rows)[:5]}... are not distinct sorted "
                          f"rows of its {count}")
    if sum(map(len, chosen.values())) != EVAL_VERSIONS[version]:
        raise Refusal(f"{path}: {task} {cell} pins {sum(map(len, chosen.values()))} {version} rows, the version "
                      f"runs {EVAL_VERSIONS[version]}")
    return {name: chosen[name] for name in entry["rows"] if name in chosen}


def candidate_path(manifest: dict, task: str, section: str = "headline") -> str:
    """Where ``freeze`` writes a candidate set by default: next to the task's rows directories."""
    entry = manifest["tasks"][task]
    first = next(iter(next(iter(entry["cells"].values()))["rows"].values()))["path"]
    return str(Path(first).parent.parent / pinned_path(task, section).name.replace(".json", "_frozen_candidate.json"))


def expected_rows(manifest: dict, task: str, section: str = "headline") -> dict[str, int]:
    """``{row file: rows}`` of every file a section replays, as the manifest declares them."""
    entry = manifest["tasks"][task]
    if section == "headline":
        return {row["path"]: row["rows"] for cell in entry["cells"].values() for row in cell["rows"].values()}
    if section == "sensitivity":
        return {path: entry["sensitivity"]["rows_per_path"] for path in entry["sensitivity"]["paths"]}
    raise ValueError(f"section {section!r} is not one of {SECTIONS}")


def freeze(manifest: dict, task: str, root: str | Path = ".", section: str = "headline") -> dict:
    """``{row file: {"sha256", "rows"}}`` for a section; refuses a missing file or one with other than its row count,
    and a task whose cap is still provisional (every row records the cap it replays under)."""
    if manifest["tasks"][task].get("cap_provisional"):
        raise ValueError(f"{task}: its episode cap is provisional (no training demos yet); derive it "
                         f"(derive_episode_cap.py --write {task}) and regenerate before recording or freezing rows")
    frozen = {}
    for relative, rows in expected_rows(manifest, task, section).items():
        path = Path(root) / relative
        if not path.is_file():
            raise Refusal(f"{relative}: missing; freeze needs every row file of the {section} set")
        digest, count = digest_and_count(path)
        if count != rows:
            raise Refusal(f"{relative}: {count} rows, the protocol replays {rows}")
        frozen[relative] = {"sha256": digest, "rows": rows}
    return frozen


SLICE_SUFFIX = ".slice.json"
"""The sidecar of a rows file cut from a frozen file: ``<rows>.jsonl`` has ``<rows>.slice.json`` beside it."""


def _slice_sidecar(path: str | Path) -> Path:
    return Path(path).with_suffix(SLICE_SUFFIX)


def _sidecar_rows(record: dict) -> list[int]:
    """The source rows a sidecar names: ``rows``, or ``start:end`` in a sidecar without ``rows``."""
    return list(record["rows"]) if "rows" in record else list(range(record["start"], record["end"]))


def write_rows(source: str, rows: Sequence[int], target: str | Path, root: str | Path = ".") -> dict:
    """Write the frozen rows file ``source``'s rows ``rows`` (indices, in order; as the manifest names it, read under
    ``root``) to ``target``, byte for byte, with a sidecar naming the source, the rows and both SHA-256s; returns it."""
    lines = (Path(root) / source).read_bytes().splitlines(keepends=True)
    rows = list(rows)
    if (not rows or len(set(rows)) != len(rows)
            or not all(isinstance(index, int) and not isinstance(index, bool) and 0 <= index < len(lines) for index in rows)):
        raise Refusal(f"{source}: rows {rows[:5]} are not distinct rows of its {len(lines)}")
    data = b"".join(lines[index] for index in rows)
    record = {"source": source, "rows": rows, "sha256": sha256_bytes(data), "source_sha256": sha256_bytes(b"".join(lines))}
    publish_new(target, data)
    publish_new(_slice_sidecar(target), (json.dumps(record) + "\n").encode())
    return record


def check_frozen_row_file(path: str | Path) -> None:
    """Refuse a rows file that no pinned frozen set (``benchmark/frozen/``) lists, or that changed since."""
    sidecar = _slice_sidecar(path)
    if Path(path).suffix == ".jsonl" and sidecar.is_file():
        record = json.loads(sidecar.read_text())
        check_frozen_row_file(record["source"])
        lines = Path(record["source"]).read_bytes().splitlines(keepends=True)
        rows = _sidecar_rows(record)
        if Path(path).read_bytes() != b"".join(lines[index] for index in rows):
            raise Refusal(f"{path}: not rows {rows[:5]}... of {record['source']} as frozen")
        return
    key = os.path.normpath(str(path))
    for frozen_file in sorted(PINNED_DIR.glob("db_*.json")):
        records = {os.path.normpath(name): record for name, record in json.loads(frozen_file.read_text()).items()}
        if key in records:
            record = records[key]
            if digest_and_count(path) != (record["sha256"], record["rows"]):
                raise Refusal(f"{path}: changed since {frozen_file.name} froze it ({record['rows']} rows, sha256 {record['sha256'][:12]})")
            return
    raise Refusal(f"{path}: not in any pinned frozen set ({PINNED_DIR}); a benchmark run replays only rows a "
                     "checked-in set lists (score_benchmark_cells.py freeze writes a candidate to commit)")


def load_frozen_set(frozen_file: str | Path, manifest: dict, task: str, root: str | Path = ".",
                    section: str = "headline") -> tuple[str, dict[str, list[dict]]]:
    """``(the frozen set's SHA-256, {row file: its rows})``, refusing anything but the exact declared set."""
    data = Path(frozen_file).read_bytes()
    frozen = json.loads(data)
    expected = expected_rows(manifest, task, section)
    if not frozen:
        raise Refusal(f"{frozen_file}: empty; it must list the {len(expected)} row files of {task}'s {section} set")
    listed = {os.path.normpath(name): record for name, record in frozen.items()}
    declared = {os.path.normpath(name): rows for name, rows in expected.items()}
    if set(listed) != set(declared):
        missing, extra = sorted(set(declared) - set(listed)), sorted(set(listed) - set(declared))
        raise Refusal(f"{frozen_file}: not {task}'s {section} set (missing {missing[:3]}, extra {extra[:3]})")
    sources = {}
    for name, rows in expected.items():
        record = listed[os.path.normpath(name)]
        if record["rows"] != rows:
            raise Refusal(f"{name}: frozen with {record['rows']} rows, the protocol replays {rows}")
        if not (Path(root) / name).is_file():
            raise Refusal(f"{name}: not under {Path(root).resolve()}; fetch the frozen rows with "
                          "./scripts/download_datasets.sh --eval-conditions")
        digest, parsed = read_jsonl(Path(root) / name)
        if (digest, len(parsed)) != (record["sha256"], rows):
            raise Refusal(f"{name}: changed since {frozen_file} froze it")
        sources[name] = parsed
    return sha256_bytes(data), sources
