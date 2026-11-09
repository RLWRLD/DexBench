# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""The benchmark's fixed numbers and rules, defined once for every task (docs/Eval_Protocol.md)."""

from __future__ import annotations

import json
import math
import statistics
import zlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

CONDITIONS_ROOT = "outputs/db_conditions"


CELLS = ("train", "ga_o", "ga_s", "ga_os")
"""The four object x spatial cells, in report order: Train, GA-O, GA-S, GA-OS. A task's cells unless its spec says."""
PHYSICAL_CELLS = ("ga_p", "ga_sp")
"""The cells a physical axis adds, after the four in report order: GA-P, GA-SP."""
CELL_LABELS = {"train": "Train", "ga_o": "GA-O", "ga_s": "GA-S", "ga_os": "GA-OS", "ga_p": "GA-P", "ga_sp": "GA-SP"}
CELL_OBJECTS = {"train": "train", "ga_o": "eval", "ga_s": "train", "ga_os": "eval", "ga_p": "train", "ga_sp": "train"}
"""Which ObjectAxis split a cell runs."""
CELL_SPATIAL = {"train": "training", "ga_o": "training", "ga_s": "beyond", "ga_os": "beyond", "ga_p": "training",
                "ga_sp": "beyond"}
"""Which spatial range a cell draws from: the training range, or the range beyond it."""
CELL_PHYSICAL = {cell: "beyond" if cell in PHYSICAL_CELLS else "training" for cell in CELL_LABELS}
"""Which physical range a cell draws from, in a task that has a physical axis."""
ROWS_PER_CELL = 120
"""Start-state rows per cell, divided evenly over the cell's Runs (objects x fixed spatial settings) where
they divide; otherwise the spec declares each Run's quota (``row_quotas``)."""
EVAL_VERSIONS = {"v1": 60, "v1-plus": ROWS_PER_CELL}
"""The benchmark's versions and the rows each runs per cell: ``v1`` a fixed half of every cell's frozen rows
(:func:`version_rows`), ``v1-plus`` all of them. Both replay the same frozen sets, so a ``v1-plus`` run also holds a
``v1`` score."""
DEFAULT_EVAL_VERSION = "v1"
PRE_VERSION_EVAL_VERSION = "v1-plus"
"""The version of a launch plan or unit that names none (no ``eval_version``, no unit ``version``): every row. A
config that resumes or extends such a plan says ``version: v1-plus``."""
CONDITION_EPISODE_S = 10.0
"""A condition recording's episode: the reset pose held this long, then the settle check."""
CONDITION_OVERRECORD = 1.5
"""Condition recordings record this many times the rows they keep (divided by the share of
draws a beyond-range filter keeps), so the settle filter has rows to spare."""
CELL_SEED_RANGE = range(100_000, 10_100_000)
"""Where the cells' Run seeds lie (:func:`cell_run_seed`): clear of every other seed the benchmark uses, the
collection's (:data:`COLLECTION_SEED_RANGE`), the sweep's (:data:`SENSITIVITY_SEED` on), DB-M1's frozen 9000-9120
and training seeds (1000-2999), DB-M4's training seeds (7301-8701), DB-M3's (6000-6079), the live
sessions' (:data:`LIVE_COLLECTION_SEED`, 101-301) and DB-R2 v1's (20000-29999); and below 2**31, as
Isaac Lab's numpy seeding needs."""


def cell_run_seed(task: str, cell: str, run: str) -> int:
    """``environment_builder.seed`` of one Run of a cell, for its condition recording and its evaluation."""
    return CELL_SEED_RANGE.start + zlib.crc32(f"{task}/{cell}/{run}".encode()) % len(CELL_SEED_RANGE)


CONTROL_HZ = 30.0
"""The rate the training data was recorded at (``cli.DEXBENCH_DEFAULT_CONTROL_HZ``), which every cell file states as
its ``rby1_control_hz`` so that ``DEXBENCH_CONTROL_HZ`` cannot move it; a model's run of a GA cell runs at its
:class:`PolicyExecution` control rate instead, and a trial at this one. The scorer holds each episode's recorded
control period to it."""


@dataclass(frozen=True)
class PolicyExecution:
    """How a task's cells run a model's policy (``policy.policy_rate``): a ``control_hz`` control step, the policy
    asked at ``policy_hz`` on that step's observation, the steps between running ``action_interpolation``.
    ``run_benchmark_cell`` writes :meth:`policy_fields` over a model's policy block and :meth:`environment_fields`
    over its environment (``model_run.compose``; a trial keeps the task's own rate, asked every step); the cells'
    declaration and the manifest's ``policy_execution`` say the same, the policy stamps what it ran, and
    ``evaluation_signature.declaration_problems`` holds the stamp to it."""

    control_hz: int
    policy_hz: int
    action_interpolation: str

    def __post_init__(self) -> None:
        if self.control_hz % self.policy_hz:
            raise ValueError(f"policy_hz {self.policy_hz} does not divide control_hz {self.control_hz}")

    def policy_fields(self) -> dict[str, object]:
        return {"policy_hz": self.policy_hz, "policy_action_interpolation": self.action_interpolation}

    def environment_fields(self) -> dict[str, object]:
        """The control rate, and the cameras refreshed only for the frames the policy reads (``camera_hz``)."""
        return {"rby1_control_hz": self.control_hz, "camera_hz": self.policy_hz}


GA_POLICY_EXECUTION = PolicyExecution(control_hz=60, policy_hz=30, action_interpolation="linear")
"""Every GA task: the policy at its 30 Hz training rate, the joint targets at 60 Hz, linear between two answers.
Why: docs/Eval_Protocol.md, "Policy at its training rate"."""

RENDER_VARIABLES = ("DEXBENCH_RENDER_INTERVAL", "DEXBENCH_DLSS_MODE")
"""Variables that change what the cameras see (``cli.apply_dexbench_env_settings``) and that no cell declares:
a benchmark run refuses to start with either set."""
LAUNCHER_HOOK_VARIABLES = ("DEXBENCH_WATER", "DEXBENCH_HIDE_PRIMS", "DEXBENCH_RENDER_TRANSLUCENCY",
                           "DEXBENCH_RENDER_REFLECTIONS", "DEXBENCH_RECORD_EXPORT_ON_RESET")
"""The teleop and recording hooks of the launcher (``_dexbench_arena_script``): the teleop water, prims hidden after
reset, RTX translucency and reflections, and a take exported at every operator reset. None is a cell's setting: a task
that needs water owns it (DB-H2's ``lift_and_pour``), and a pour scene's glass renders through its environment's
``POUR_RENDER_SETTINGS``. An evaluation refuses each, and every ``DEXBENCH_WATER_*`` (:func:`launcher_hook_variables`)."""


def launcher_hook_variables(environ) -> list[str]:
    """The launcher hook variables (:data:`LAUNCHER_HOOK_VARIABLES`, ``DEXBENCH_WATER_*``) set in ``environ``."""
    return sorted(name for name in environ if name in LAUNCHER_HOOK_VARIABLES or name.startswith("DEXBENCH_WATER_"))

BACKEND_PHYSICS = {"NEWTON": "NewtonCfg"}
"""The physics cfg class an ``environment_builder.presets`` value composes; a row's ``evaluation_settings["physics"]``
names it, and the scorer holds it to the cell's declared backend. Newton only: DexBench builds on nothing else."""


LIVE_COLLECTION_SEED = 42
"""Arena's default ``environment_builder.seed`` (``--seed``): a teleop or recording session that draws its start
states live. A task with a ``CollectionSpec`` collects from a recorded start-state dump instead."""
COLLECTION_SEED_RANGE = range(8000, 9000)
"""Where a task's collection seeds (``CollectionSpec.seed_base`` + the Run's index) lie: clear of the cells'
(:data:`CELL_SEED_RANGE`, and DB-M1's frozen 9000-9120), the sweep's (:data:`SENSITIVITY_SEED` on), DB-M1's
training seeds (1000-2999) and the live sessions' (:data:`LIVE_COLLECTION_SEED`, 101-301)."""
COLLECTION_BATCH_SEED_STRIDE = 10
"""Seeds between one batch of a collection and the next: Run ``i`` of batch ``b`` records at
``seed_base + b * COLLECTION_BATCH_SEED_STRIDE + i``, so a task has at most this many Runs and its batches never
share a seed."""
COLLECTION_MAX_BATCHES = 10
"""Batches a task's collection may declare: with the tasks' seed bases 100 apart, ten batches of ten seeds fill
a task's block and no task's seeds reach another's."""


SENSITIVITY_EPISODES = 200
"""Recorded draws per task in the sensitivity sweep, replayed for every model."""
SENSITIVITY_REBUILDS = 20
"""Environment builds the sweep spreads its draws over. The light variations are build-time
at the pin (one draw per build), so the sweep sees this many lighting draws, each with
SENSITIVITY_EPISODES / SENSITIVITY_REBUILDS run-time draws (camera, spatial) under it."""
NUISANCE_FACTORS = ("light_intensity", "color_temperature", "camera_extrinsics", "camera_intrinsics", "camera_rotation",
                    "texture", "distractors")
"""The standard nuisance factors (RoboLab, RoboDojo, Bench2Dex). Every task's sweep either covers each
with variation keys or states it unavailable at the pin with the reason; none is claimed silently."""
VISUAL_FACTORS = ("light_intensity", "color_temperature", "camera_extrinsics", "camera_intrinsics", "camera_rotation", "texture")
"""The factors that change only what the cameras see. The sweep's recording keeps the rows whose scene
settled under the hold-pose policy; a visual factor cannot change whether a scene settles, so that
filter cannot select on it, and the kept draws stay uniform. A sweep claims visual factors only
(``spec.Sensitivity`` refuses another; the filter applies the generated sweep plan); a physical one (distractors) would need
its own unfiltered recording."""
SENSITIVITY_SEED = 1016
"""``environment_builder.seed`` of the sweep's recording; rebuild ``i`` runs at seed + i (Arena). That seeds the
run-time draws only: the build-time light draws come from torch's global state before Isaac Lab seeds it,
so a re-recording at this seed gives other lights until IsaacLab-Arena#1467 lands. The replay
is unaffected, since every row records its build's light draw and the replay binds it."""


EXECUTION_HORIZON = 16
"""Actions every model executes from one chunk before it is queried again, whatever its own
chunk length: one setting for every model, so the comparison does not reward a longer open loop."""


DEMO_LENGTHS_DIR = Path(__file__).resolve().parent / "demo_lengths"
"""``db_<task>.json`` per task: the per-episode lengths ``derive_episode_cap.py --write`` read from its training set."""
DEMO_SET_EPISODES = 240
"""Episodes in a task's EA training set, the set its cap is derived from."""
CAP_ROUND_S = 5.0
CAP_MEDIAN_FACTOR = 2.0
CAP_LONGEST_FACTOR = 1.25


def _ceil_to(value: float, step: float) -> float:
    return math.ceil(value / step - 1e-9) * step


def episode_cap_s(median_s: float, longest_s: float, hold_s: float = 0.0) -> float:
    """The benchmark's episode cap for a task whose training demos have this median and longest duration."""
    raw = max(CAP_MEDIAN_FACTOR * median_s, CAP_LONGEST_FACTOR * longest_s) + hold_s
    return _ceil_to(raw, CAP_ROUND_S)


@dataclass(frozen=True)
class DemoDurations:
    """What a task's cap is derived from: its training set's episode durations, summarised."""

    source: str
    """Where the set lives and what the lengths were read from (for a provisional cap: where the cap comes from)."""
    n: int
    median_s: float | None
    p90_s: float | None
    longest_s: float | None
    manifest_sha256: str | None = None
    """SHA-256 of the set's own manifest file, so a re-collected set shows up as a different one."""
    revision: str | None = None
    """The dataset repository commit the set was read at, where the set is published."""
    files: tuple[tuple[str, str], ...] = ()
    """(path at ``revision``, SHA-256) of each training file: the cohort a model is trained on, pinned
    file by file where the set is published as a few files rather than behind a manifest."""
    lengths_file: str | None = None
    """``benchmark/demo_lengths/<name>.json`` holding the per-episode lengths."""
    provisional_cap_s: float | None = None
    """The stated cap of a task with no training set yet (:meth:`provisional`); None once derived from demos."""

    @property
    def is_provisional(self) -> bool:
        return self.provisional_cap_s is not None

    @classmethod
    def provisional(cls, cap_s: float, source: str) -> "DemoDurations":
        """A stated cap for a task with no training demos yet: no episodes, no median, p90 or longest."""
        if isinstance(cap_s, bool) or not isinstance(cap_s, (int, float)) or not math.isfinite(cap_s) or cap_s <= 0:
            raise ValueError(f"a provisional cap is a finite, positive number of seconds, got {cap_s!r}")
        if not source:
            raise ValueError("a provisional cap says where it comes from")
        return cls(source=source, n=0, median_s=None, p90_s=None, longest_s=None, provisional_cap_s=float(cap_s))

    @classmethod
    def from_steps(cls, steps: Sequence[int], control_hz: float = 30.0, **fields) -> "DemoDurations":
        if not steps or not (isinstance(control_hz, (int, float)) and math.isfinite(control_hz) and control_hz > 0):
            raise ValueError("demo durations need episodes and a finite, positive control rate")
        if any(isinstance(step, bool) or not isinstance(step, int) or step <= 0 for step in steps):
            raise ValueError("demo durations need whole, positive lengths in control steps")
        seconds = [step / control_hz for step in steps]
        return cls(n=len(seconds), median_s=round(statistics.median(seconds), 3),
                   p90_s=round(statistics.quantiles(seconds, n=10, method="inclusive")[8], 3), longest_s=round(max(seconds), 3), **fields)

    @classmethod
    def from_record(cls, record: dict, **fields) -> "DemoDurations":
        """The durations of a lengths record (what ``derive_episode_cap.py`` writes)."""
        return cls.from_steps([steps for _name, steps in record["episodes"]], record["control_hz"],
                              source=record["source"], manifest_sha256=record["manifest_sha256"], **fields)

    @classmethod
    def from_lengths_file(cls, name: str, **fields) -> "DemoDurations":
        """The durations of ``DEMO_LENGTHS_DIR/<name>``."""
        return cls.from_record(json.loads((DEMO_LENGTHS_DIR / name).read_text()), lengths_file=name, **fields)

    def cap_s(self, hold_s: float = 0.0) -> float:
        if self.provisional_cap_s is not None:
            return self.provisional_cap_s + hold_s
        return episode_cap_s(self.median_s, self.longest_s, hold_s)


def subset_quota(quota: dict[str, int], rows: int) -> dict[str, int]:
    """``rows`` of a cell split over its Runs (``quota``: each Run's rows) in proportion to their rows, the largest
    remainders first and ties to the earlier Run; Runs that get none are left out. :func:`version_rows` splits a
    version's rows over the frozen rows this way, and :func:`cell_rows` a pilot's over the v1 rows. It counts rows
    only; which rows each Run keeps is the caller's choice."""
    total = sum(quota.values())
    if not 0 < rows <= total:
        raise ValueError(f"a subset of {rows} rows from a cell of {total}")
    exact = {name: rows * count / total for name, count in quota.items()}
    share = {name: int(value) for name, value in exact.items()}
    order = sorted(quota, key=lambda name: (-(exact[name] - share[name]), list(quota).index(name)))
    for name in order[: rows - sum(share.values())]:
        share[name] += 1
    return {name: count for name, count in share.items() if count}


def version_rows(quota: dict[str, int], version: str,
                 objects: Mapping[str, str] | None = None) -> dict[str, tuple[int, ...]]:
    """``{run: frozen row indices}`` a cell replays at ``version`` (:data:`EVAL_VERSIONS`); Runs given none are left out."""
    if version not in EVAL_VERSIONS:
        raise ValueError(f"version {version!r} is not one of {sorted(EVAL_VERSIONS)}")
    total = sum(quota.values())
    rows_kept = min(EVAL_VERSIONS[version], total)
    if objects is None or len(set(objects[name] for name in quota)) < 2:
        share = subset_quota(quota, rows_kept)
    else:
        per_object: dict[str, int] = {}
        for name, count in quota.items():
            per_object[objects[name]] = per_object.get(objects[name], 0) + count
        share = {}
        for obj, keep in subset_quota(per_object, rows_kept).items():
            share.update(subset_quota({name: count for name, count in quota.items() if objects[name] == obj}, keep))
    rows = {}
    for position, (name, count) in enumerate(quota.items()):
        keep = share.get(name, 0)
        if keep == count:
            rows[name] = tuple(range(count))
        elif keep:
            pairs, offset = (keep + 1) // 2, position * count // len(quota)
            picked = [(start + step) % count for pair in range(pairs)
                      for start in [pair * count // pairs + offset] for step in (0, 1)][:keep]
            rows[name] = tuple(sorted(picked))
    return rows


PILOT_VERSION = min(EVAL_VERSIONS, key=EVAL_VERSIONS.get)
"""The version a pilot's rows are taken from: the one that runs the fewest rows, so they are rows every headline
model ran at any version."""


def cell_rows(quota: dict[str, int], pilot_rows: int | None, version: str,
              pinned: Callable[[str], Mapping[str, Sequence[int]]]) -> dict[str, tuple[int, ...]]:
    """``{run: frozen row indices}`` a model runs of a cell (``quota``: each Run's frozen rows) and is scored on."""

    def rows_at(name: str) -> dict[str, tuple[int, ...]]:
        if name not in EVAL_VERSIONS:
            raise ValueError(f"version {name!r} is not one of {sorted(EVAL_VERSIONS)}")
        if EVAL_VERSIONS[name] >= sum(quota.values()):
            return {run: tuple(range(count)) for run, count in quota.items()}
        return {run: tuple(rows) for run, rows in pinned(name).items()}

    kept = rows_at(version)
    if not pilot_rows:
        return kept
    kept = rows_at(PILOT_VERSION)
    share = subset_quota({name: len(rows) for name, rows in kept.items()}, pilot_rows)
    return {name: kept[name][:count] for name, count in share.items()}
