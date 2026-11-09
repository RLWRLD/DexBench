# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Teleop collection from a recorded start-state dump: one row at a time, held until a demo of it is kept."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from isaaclab_arena.variations.recorded_variation_samples import RebuildVariationRecord
from isaaclab_arena.variations.variation_replay_scheduler import VariationReplayScheduler

from dexbench_arena.benchmark.protocol import COLLECTION_SEED_RANGE
from dexbench_arena.embodiments.episode_metadata import install_episode_metadata
from dexbench_arena.evaluation.artifacts import file_sha256, read_jsonl

DATASET_METADATA_KEY = "dexbench_collection_rows"
"""The recording's ``env_args`` key: the rows file and which row each kept demo started from."""
EPISODE_RECORD_KEY = "collection_row"
"""The episode row's field (``<stem>.episodes.jsonl``): the row each take started from, discarded takes too."""


class CollectionRowsExhausted(RuntimeError):
    """Every row of the file has a kept demo: the collection is complete and starts no other episode."""


@dataclass(frozen=True)
class CollectionRows:
    """A rows file a session collects from, as read before Kit starts."""

    path: str
    """The file as the command line names it."""
    sha256: str
    rows: int
    first_row: int = 0
    """The row the session starts at (a session that resumes a collection)."""
    manifest_run: str | None = None
    """``<task>/batch_<b>/<run>`` when the file is a collection dump's row file in the benchmark manifest."""
    environment: Mapping[str, Any] = field(default_factory=dict)
    """That Run's environment fields (its object and spatial setting), which the build must set alike."""
    requeue_path: str | None = None
    """``--collection_requeue_file``: rows to hold again before the next one in file order (the module docstring)."""

    @classmethod
    def read(cls, path: str | Path, first_row: int = 0, manifest: Mapping[str, Any] | None = None,
             requeue_path: str | Path | None = None) -> CollectionRows:
        """Read ``path`` (an episode-result JSONL); refuses an empty file or a first row outside it."""
        source = Path(path)
        if source.suffix.lower() != ".jsonl" or not source.is_file():
            raise ValueError(f"--collection_rows names {path}, which is no episode-result JSONL file")
        records = read_jsonl(source)[1]
        count = len(records)
        if count == 0:
            raise ValueError(f"{path} holds no rows")
        if not 0 <= first_row < count:
            raise ValueError(f"--collection_first_row {first_row} is outside {path}'s rows 0..{count - 1}")
        found = manifest_collection_run(manifest, source) if manifest is not None else None
        run, environment = (f"{found[0]}/{found[1]}", dict(found[2]["environment"])) if found else (None, {})
        if found is not None and found[2]["rows"] != count:
            raise ValueError(f"{path} holds {count} rows; the manifest's collection Run {run} keeps {found[2]['rows']}")
        if found is None and manifest is not None and records[0].get("seed") in COLLECTION_SEED_RANGE:
            run, environment = _run_by_recorded_environment(manifest, path, records)
        return cls(str(path), file_sha256(source), count, first_row, run, environment,
                   None if requeue_path is None else str(requeue_path))

    def requeued(self) -> list[int]:
        """The rows the requeue file lists, in the order appended; an unreadable file lists none, and a line that is
        no row of this file is left out (with a note), since a session must not stop on the console's file."""
        if self.requeue_path is None:
            return []
        try:
            lines = Path(self.requeue_path).read_text().split()
        except OSError:
            return []
        rows = []
        for line in lines:
            if line.isdigit() and int(line) < self.rows:
                rows.append(int(line))
            else:
                print(f"[DexBench collection] {self.requeue_path}: {line!r} is no row of {self.path}; left out",
                      flush=True)
        return rows

    def build_differences(self, build_cfg: Any) -> list[str]:
        """The manifest Run's environment fields that the build (its environment config) sets otherwise."""
        return [f"{name}: the rows' Run {self.manifest_run} sets {value!r}, this build {getattr(build_cfg, name, None)!r}"
                for name, value in self.environment.items() if getattr(build_cfg, name, None) != value]

    def provenance(self) -> dict[str, Any]:
        return {"rows_file": self.path, "rows_sha256": self.sha256, "rows": self.rows, "first_row": self.first_row,
                "manifest_run": self.manifest_run}


REPO_ROOT = Path(__file__).resolve().parents[4]
"""The checkout the manifest's repo-relative row paths (``outputs/db_conditions/...``) are relative to."""


def _collection_runs(manifest: Mapping[str, Any]) -> Iterable[tuple[str, str, Mapping[str, Any]]]:
    for task, entry in manifest.get("tasks", {}).items():
        for batch in (entry.get("collection") or {}).get("batches") or ():
            for run, row in batch["rows"].items():
                yield task, f"batch_{batch['batch']}/{run}", row


def collection_conditions(env_args, demos, conditions):
    """``rerecord_from_conditions.py --condition_rows collection``: each source demo's condition row, the one its
    collection session recorded it from (``env_args`` ``DATASET_METADATA_KEY``'s ``demo_rows``), keyed by the source
    demo's name and naming that row (``condition_row``); a demo whose row ``conditions`` lacks is left out (the
    caller refuses a selected one). ValueError for a demo the session names no row for."""
    rows = (env_args.get(DATASET_METADATA_KEY) or {}).get("demo_rows") or {}
    missing = [name for name in demos if name not in rows]
    if missing:
        raise ValueError(f"--condition_rows collection: the source names no collection row for {missing}")
    names = {name: f"demo_{int(rows[name])}" for name in demos}
    return {name: {**conditions[row], "condition_row": row} for name, row in names.items() if row in conditions}


def manifest_collection_run(manifest: Mapping[str, Any], path: str | Path) -> tuple[str, str, Mapping[str, Any]] | None:
    """``(task, "batch_<b>/<run>", manifest row)`` of the collection Run whose row file ``path`` is; None for any other file."""
    resolved = Path(path).resolve()
    for task, batch_run, row in _collection_runs(manifest):
        recorded = Path(row["path"])
        if resolved in (recorded.resolve(), (REPO_ROOT / recorded).resolve()):
            return task, batch_run, row
    return None


def _run_by_recorded_environment(manifest: Mapping[str, Any], path: str | Path,
                                 records: Sequence[Mapping[str, Any]]) -> tuple[str, dict[str, Any]]:
    """The collection Run whose environment fields ``records`` were recorded with, for a dump read from a path the
    manifest does not name. Refuses rows that match no Run, or Runs that set their fields differently."""
    recorded = [row.get("environment_cfg") or {} for row in records]
    matches = [(f"{task}/{batch_run}", dict(row["environment"])) for task, batch_run, row in _collection_runs(manifest)
               if all(all(cfg.get(name) == value for name, value in row["environment"].items()) for cfg in recorded)]
    environments = {tuple(sorted(environment.items())) for _, environment in matches}
    if len(environments) != 1:
        raise ValueError(f"{path} holds rows recorded at collection seeds, but its rows' environment_cfg matches "
                         f"{'no collection Run' if not matches else 'Runs that differ: ' + str(sorted(name for name, _ in matches))} "
                         "in the benchmark manifest; collect from the manifest's row file")
    names = sorted(name for name, _ in matches)
    return (names[0] if len(names) == 1 else f"{names[0]} (or {', '.join(names[1:])})"), matches[0][1]


def embodiment_mismatches(path: str | Path, embodiment: str, embodiments: Iterable[str]) -> list[str]:
    """Why ``path``'s draws do not fit a build on ``embodiment``: each other embodiment (of ``embodiments``, the
    registered ones) whose draws the rows record (``<name>.base_pose``). Arena binds a recorded draw only to the
    build's own variation of that name, and a dump recorded with the joint twin names the twin's.
    """
    others = set(embodiments) - {embodiment}
    found = sorted({key for row in read_jsonl(path)[1] for key in (row.get("variations") or {}) if key.split(".", 1)[0] in others})
    if not found:
        return []
    return [f"{path} records {found}, another robot's draws; this build runs {embodiment}. Record the dump with the "
            "robot the station drives (CollectionSpec.environment), or build with the rows' robot"]


def variations_for_embodiment(variations: Mapping[str, Any], source: str, target: str) -> dict[str, Any]:
    """``variations`` (flat ``<asset>.<variation>...`` keys) with those registered on embodiment ``source`` named for
    ``target``: an embodiment's variation carries its name (``<embodiment>.base_pose``), so the same draw on a build
    with another robot (the station's IK twin, for a range that names the joint twin) is the other robot's key."""
    return {(f"{target}.{key.split('.', 1)[1]}" if key.split(".", 1)[0] == source else key): value
            for key, value in variations.items()}


class RowHoldingReplayScheduler(VariationReplayScheduler):
    """Arena's replay scheduler, holding one row until a demo of it is kept (the module's docstring)."""

    def __init__(self, variation_record: RebuildVariationRecord, rows: CollectionRows) -> None:
        super().__init__(variation_record)
        self._hold(rows)

    def _hold(self, rows: CollectionRows) -> None:
        if rows.rows != self.variation_record.num_recorded_episodes:
            raise ValueError(f"{rows.path} holds {rows.rows} rows; the replay loaded "
                             f"{self.variation_record.num_recorded_episodes}")
        self.rows = rows
        self._next_in_order = rows.first_row
        self._requeue_read = 0
        self._takes = 0
        self._started = 0
        self._assigned: dict[int, int] = {}
        self.kept: dict[str, int] = {}
        """Demo name -> the row it started from, for every kept demo of this session."""
        self._row = self._next_row()

    def _next_row(self) -> int:
        """The row to hold next: the first one appended to the requeue file since the last read, else the next in
        file order that has no kept demo of this session (``rows.rows`` past the last: exhausted)."""
        requeued = self.rows.requeued()
        if len(requeued) > self._requeue_read:
            row = requeued[self._requeue_read]
            self._requeue_read += 1
            print(f"[DexBench collection] row {row} offered again (requeued in {self.rows.requeue_path})", flush=True)
            return row
        while self._next_in_order in self.kept.values():
            self._next_in_order += 1
        row = self._next_in_order
        self._next_in_order += 1
        return row

    @property
    def held_row(self) -> int:
        return self._row

    @property
    def exhausted(self) -> bool:
        return self._row >= self.rows.rows

    @property
    def num_assignments_started(self) -> int:
        return self._started

    @property
    def num_assignments_completed(self) -> int:
        return self._started - len(self._assigned)

    def assign_new_episodes(self, env_ids: Sequence[int]) -> None:
        for raw_env_id in env_ids:
            env_id = int(raw_env_id)
            assert env_id not in self._assigned, f"Environment {env_id} already has an active variation-record assignment."
            if self._assigned:
                raise ValueError("a collection session runs one env: two takes would start from the same row")
            if self.exhausted and len(self.rows.requeued()) > self._requeue_read:
                self._row = self._next_row()
            if self.exhausted:
                raise CollectionRowsExhausted(f"every row of {self.rows.path} from {self.rows.first_row} has a kept demo; "
                                              "the collection is complete")
            self._assigned[env_id] = self._row
            self._started += 1
            self._takes += 1
            print(f"[DexBench collection] row {self._row} of {self.rows.path} ({self.rows.rows} rows, sha256 "
                  f"{self.rows.sha256[:12]}) presented, take {self._takes} of it", flush=True)

    def complete_episodes(self, env_ids: Sequence[int]) -> None:
        for raw_env_id in env_ids:
            self._assigned.pop(int(raw_env_id))

    def source_record_index_for_env(self, env_id: int) -> int:
        return self._assigned[int(env_id)]

    def advance(self, demo: str) -> None:
        """A demo of the held row was kept as ``demo``: note it, and hold the next row from the next episode on."""
        self.kept[demo] = self._row
        print(f"[DexBench collection] row {self._row} kept as {demo} after {self._takes} take(s); "
              f"{max(0, self.rows.rows - self._next_in_order)} row(s) left", flush=True)
        self._row = self._next_row()
        self._takes = 0

    def episode_record(self, env_id: int) -> dict[str, Any]:
        """The episode row's ``collection_row``: the row the finishing take started from."""
        if int(env_id) not in self._assigned:
            return {}
        return {EPISODE_RECORD_KEY: {"row": self._assigned[int(env_id)], "rows_sha256": self.rows.sha256}}

    def dataset_metadata(self, pending_demo: str | None = None) -> dict[str, Any]:
        """The recording's ``env_args`` entry: the rows file and each kept demo's row, ``pending_demo`` (the one
        being exported as a success) included."""
        demos = dict(self.kept)
        if pending_demo is not None:
            demos[pending_demo] = self._row
        return {**self.rows.provenance(), "demo_rows": demos}


def hold_rows(scheduler: VariationReplayScheduler, rows: CollectionRows) -> RowHoldingReplayScheduler:
    """Make the scheduler Arena's builder made hold ``rows`` (the module's docstring: a pinned-Arena workaround)."""
    if not isinstance(scheduler, VariationReplayScheduler):
        raise ValueError("--collection_rows needs Arena's replay scheduler; the build replays no rows")
    if scheduler.num_assignments_started:
        raise ValueError("the replay scheduler has started episodes; hold the rows before the first reset")
    scheduler.__class__ = RowHoldingReplayScheduler
    scheduler._hold(rows)
    return scheduler


def record_collection_row(env, env_id: int) -> dict[str, Any]:
    """Episode recorder term: the row the take started from; nothing for an env not collecting from rows."""
    scheduler = getattr(env, "variation_replay_scheduler", None)
    return scheduler.episode_record(env_id) if isinstance(scheduler, RowHoldingReplayScheduler) else {}


def install_collection_recording(env, scheduler: RowHoldingReplayScheduler) -> bool:
    """Advance ``scheduler`` on every demo the recorder exports as a success, and stamp the recording with the rows."""
    recorder = getattr(env, "recorder_manager", None)
    if recorder is None or not getattr(recorder, "active_terms", None):
        return False
    export: Callable[..., None] = recorder.export_episodes

    def export_and_advance(*args, **kwargs) -> None:
        count = recorder.exported_successful_episode_count
        export(*args, **kwargs)
        if recorder.exported_successful_episode_count > count:
            scheduler.advance(f"demo_{recorder.exported_successful_episode_count - 1}")

    def metadata() -> dict[str, Any]:
        succeeded = bool(recorder.get_episode(0).success)
        return scheduler.dataset_metadata(f"demo_{recorder.exported_successful_episode_count}" if succeeded else None)

    recorder.export_episodes = export_and_advance
    install_episode_metadata(env, DATASET_METADATA_KEY, metadata)
    return True
