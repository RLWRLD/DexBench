# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Each benchmark task's declaration: what it shares, its spatial ranges, its cap's demo set and its sweep."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from dexbench_arena.benchmark.models import BASELINE_POLICY
from dexbench_arena.benchmark.protocol import (
    CELL_LABELS,
    CELL_PHYSICAL,
    CELL_SPATIAL,
    CELLS,
    COLLECTION_MAX_BATCHES,
    COLLECTION_SEED_RANGE,
    CONDITION_EPISODE_S,
    CONDITION_OVERRECORD,
    DEMO_SET_EPISODES,
    GA_POLICY_EXECUTION,
    NUISANCE_FACTORS,
    PHYSICAL_CELLS,
    VISUAL_FACTORS,
    DemoDurations,
    PolicyExecution,
)
from dexbench_arena.deformables.pack_invoice_scene_randomization import (
    PAPER_PHYSICS_ARGS,
    TRAINING_RANGES,
    PackInvoiceSceneRanges,
    beyond_settings,
    scene_fields,
)
from dexbench_arena.embodiments.rby1_wuji2.constants import RBY1_WUJI2_JOINT_EMBODIMENT
from dexbench_arena.layouts.db_m1_tabletop_envelope import TABLETOP_GA_S_SCALE, TABLETOP_TRAIN_BOUNDS_M_RAD
from dexbench_arena.tasks.h3_objects import H3_INPUTS_DIR
from dexbench_arena.variations.variation_draws import dotted_items

_EMPTY: Mapping[str, Any] = MappingProxyType({})
_NEWTON = MappingProxyType({"presets": "NEWTON"})


@dataclass(frozen=True)
class SpatialSetting:
    """One spatial setting of a range: fixed environment fields and Arena variation keys, run as its own Runs
    (``<object>_<key>``)."""

    key: str
    environment: Mapping[str, Any]
    note: str = ""
    variations: Mapping[str, Any] = _EMPTY
    """Arena variation keys the setting's Runs enable and bound (a draw the setting keeps, or fixes as a point)."""


@dataclass(frozen=True)
class SpatialRange:
    """The training range, or the range beyond it."""

    note: str
    """What it is, with its numbers and why (it goes into the files' headers and the doc)."""
    environment: Mapping[str, Any] = _EMPTY
    """Environment fields every Run of the range sets (a draw enabled through a config field)."""
    variations: Mapping[str, Any] = _EMPTY
    """Arena variation keys (``<asset>.<variation>.<field>``) the range enables and bounds."""
    settings: tuple[SpatialSetting, ...] = ()
    """Fixed settings run as separate Runs (DB-R2's base ring); empty for a range drawn per episode. A setting's
    variation keys are its Runs' own; the range's ``variations`` hold only what every Run shares."""
    outside: tuple[str, tuple[float, ...], tuple[float, ...]] | None = None
    """``(key, low, high)``: keep only the recorded rows whose ``key`` draw lies outside this box
    (the training range) in at least one component -- a beyond range recorded as a wider box."""
    outside_keep_share: float = 1.0
    """The share of a wider box's uniform draws the ``outside`` filter keeps, for over-recording."""


@dataclass(frozen=True)
class PhysicalRange:
    """A physical axis's training range, or the range beyond it: drawn per episode on top of the cell's spatial range,
    so it has no fixed settings and no outside box of its own."""

    note: str
    """What it is, with its numbers and why (it goes into the files' headers and the doc)."""
    environment: Mapping[str, Any] = _EMPTY
    """Environment fields every Run of the range sets (a draw enabled through a config field)."""
    variations: Mapping[str, Any] = _EMPTY
    """Arena variation keys the range enables and bounds."""


@dataclass(frozen=True)
class FrozenRows:
    """Cells whose rows a tool writes from frozen inputs (scenes or seeds) instead of a condition recording."""

    cells: tuple[str, ...]
    command: str
    note: str
    invalid_starts: str
    """What happens to a frozen row whose start state turns out invalid (no condition recording filters it)."""


@dataclass(frozen=True)
class RowsPending:
    """A task whose cells are declared and generated but whose rows are not recorded yet."""

    reason: str
    """Why the rows are not recorded yet, and what has to happen first."""


@dataclass(frozen=True)
class CollectionBatch:
    """A batch declared otherwise than the collection's default: at a range other than the task's training range, or
    collecting part of the training set (its share replaces as many demos of an earlier batch)."""

    rows: int = DEMO_SET_EPISODES
    """The rows the batch's files hold, one start state each: its demos, plus its reserve where it keeps one."""
    row_quotas: Mapping[str, int] = _EMPTY
    """Run name -> rows where ``rows`` does not split evenly over the Runs; else an even split (the collection's
    ``row_quotas`` are the whole training set's)."""
    training: SpatialRange | None = None
    """The training range the batch was recorded at, where it differs from the task's; None: the task's."""
    keep: Mapping[str, int] = _EMPTY
    """Run name -> the demos the set takes from the Run's file: its first rows, in file order. The rows after them
    are the Run's reserve, in a fixed order: a row whose demo cannot be kept is replaced by the next reserve row,
    never by a repeat. Empty: every row is a demo."""

    def __post_init__(self) -> None:
        if self.rows < 1:
            raise ValueError(f"a collection batch holds at least one row, got {self.rows}")
        if self.row_quotas and sum(self.row_quotas.values()) != self.rows:
            raise ValueError(f"a batch's row_quotas sum to its {self.rows} rows, not {sum(self.row_quotas.values())}")
        if self.keep and not (set(self.keep) <= set(self.row_quotas)
                              and all(0 < count <= self.row_quotas[name] for name, count in self.keep.items())):
            raise ValueError("a batch keeps 1..its quota of each Run's rows, and names its quotas to say so")


@dataclass(frozen=True)
class CollectionSpec:
    """A training set recorded from a start-state dump: one kept demo per row."""

    seed_base: int
    """Batch 0's first Run's ``environment_builder.seed``; Run ``i`` (in the manifest's order) of batch ``b`` records
    at ``seed_base + b * protocol.COLLECTION_BATCH_SEED_STRIDE + i``, inside ``protocol.COLLECTION_SEED_RANGE``."""
    note: str
    """How the training set's demos split over the Runs, and where that split comes from."""
    row_quotas: Mapping[str, int] = _EMPTY
    """Run name -> rows where the training set does not split evenly over the Runs; else an even split."""
    overrecord: float = CONDITION_OVERRECORD
    """Episodes the dump records per kept row, so the settle filter has rows to spare."""
    episode_s: float = CONDITION_EPISODE_S
    """How long each of the dump's episodes holds the reset pose before the settle check: the protocol's, unless the
    task's start has nothing that moves for longer (DB-L1: every item waits parked; its note says why)."""
    environment: Mapping[str, Any] = MappingProxyType({"embodiment": "dexbench_rby1_wuji2"})
    """Fields the dump records with over the task's: the Pink IK twin, so each row names its draws as a recording
    build does (``dexbench_rby1_wuji2.base_pose``) and a recording replays the file as it is. The benchmark's rows name the joint twin, which the policies run."""

    batches: int = 1
    """How many 240-demo sets the team collects, each from a dump of its own: batch ``b`` has its own seeds, its own
    recording (``db_<task>_collect_conditions_batch<b>.yaml``) and its own rows (``<rows dir>/batch_<b>/``). Raise it
    and regenerate to add a batch; the batches recorded so far keep their files."""
    batch_declarations: Mapping[int, CollectionBatch] = _EMPTY
    """Batch -> how it differs from the default (the whole training set over the task's training range): a batch
    recorded at another range keeps the range it was recorded at, so its file and rows stay what they are."""

    def __post_init__(self) -> None:
        if not 1 <= self.batches <= COLLECTION_MAX_BATCHES:
            raise ValueError(f"a collection declares 1..{COLLECTION_MAX_BATCHES} batches, got {self.batches}")
        if not set(self.batch_declarations) <= set(range(self.batches)):
            raise ValueError(f"batch_declarations name batches {sorted(self.batch_declarations)}, outside 0..{self.batches - 1}")
        if self.seed_base not in COLLECTION_SEED_RANGE:
            raise ValueError(f"a collection's seeds lie in {COLLECTION_SEED_RANGE}, clear of the cells', the sweep's "
                             f"and the live sessions'; got seed_base {self.seed_base}")
        if not (math.isfinite(self.overrecord) and self.overrecord >= 1.0):
            raise ValueError(f"a collection records at least the rows it keeps (overrecord >= 1), got {self.overrecord}")
        if not (math.isfinite(self.episode_s) and 0.0 < self.episode_s <= CONDITION_EPISODE_S):
            raise ValueError(f"a collection's condition episode lasts (0, {CONDITION_EPISODE_S}] s, got {self.episode_s}")
        if not self.note:
            raise ValueError("a collection says how its rows split and why")

    def batch(self, batch: int) -> CollectionBatch:
        """How batch ``batch`` is declared (the default unless ``batch_declarations`` names it)."""
        return self.batch_declarations.get(batch, _DEFAULT_BATCH)


_DEFAULT_BATCH = CollectionBatch()


@dataclass(frozen=True)
class Sensitivity:
    """The "+alpha" sweep: recorded once with the baseline policy, replayed for every model."""

    object_key: str
    factors: Mapping[str, str]
    """Factor key (``<asset>.<variation>``) -> its range and why, for the doc and the headers."""
    variations: Mapping[str, Any]
    """The sweep's whole variation block: the factors, plus the task's training-range draws."""
    covers: Mapping[str, tuple[str, ...]] = _EMPTY
    """Standard factor (``protocol.NUISANCE_FACTORS``) -> the variation keys that sweep it here."""
    unavailable: Mapping[str, str] = _EMPTY
    """Standard factor -> why this scene cannot sweep it at the pin (and the smallest addition)."""
    environment: Mapping[str, Any] = _EMPTY
    notes: tuple[str, ...] = ()
    legacy_unlabelled_paths: bool = False
    """Preserve a published sweep whose paths have no layout qualification; others use their object axis layout."""
    rows_pending: RowsPending | None = None
    """The optional sweep may remain unfrozen after the headline cells are ready."""

    def __post_init__(self) -> None:
        physical = sorted(set(self.covers) - set(VISUAL_FACTORS))
        if physical:
            raise ValueError(f"a sweep covers visual factors only (protocol.VISUAL_FACTORS); {physical} would be "
                             "selected on by the settle filter")
        if set(self.covers) | set(self.unavailable) != set(NUISANCE_FACTORS) or set(self.covers) & set(self.unavailable):
            raise ValueError(f"a sweep covers or states unavailable each of {NUISANCE_FACTORS} once; it covers "
                             f"{sorted(self.covers)} and states {sorted(self.unavailable)}")
        if not all(self.unavailable.values()):
            raise ValueError("an unavailable factor states why")
        if {key for keys in self.covers.values() for key in keys} != set(self.factors):
            raise ValueError("the covered factors' variation keys are the sweep's factors, no more and no fewer")


@dataclass(frozen=True)
class BenchmarkTaskSpec:
    task: str
    """File token: ``db_<task>_...yaml``."""
    name: str
    title: str
    environment: Mapping[str, Any]
    """Fields every Run shares (``type`` is the environment's registered name)."""
    training: SpatialRange
    beyond: SpatialRange
    durations: DemoDurations
    sensitivity: Sensitivity
    cells: tuple[str, ...] = CELLS
    environment_builder: Mapping[str, Any] = _NEWTON
    baseline_policy: Mapping[str, Any] = BASELINE_POLICY
    frozen: FrozenRows | None = None
    rows_pending: RowsPending | None = None
    """Declared when the task's rows are not recorded yet, so it has no pinned frozen set (:class:`RowsPending`)."""
    hold_s: float = 1.0
    """Success dwell the demos do not contain, added to the cap: DB-R2's strict tier holds 1 s, which its demos, kept
    by the collection gate, do not. A task whose demos already contain its hold (DB-R3, kept by the gate it is
    scored by), whose success holds otherwise, or whose cap is frozen without it, says so."""
    row_quotas: Mapping[str, Mapping[str, int]] = _EMPTY
    """Cell -> {row name: rows} where a cell's rows do not split evenly over its Runs; else an even split."""
    recorded_cell_seeds: Mapping[str, int] = _EMPTY
    """Cell -> the one seed its pinned rows were recorded at, for a cell recorded at one seed for all its Runs: its Runs
    keep that seed, so the generated conditions file is the one the rows came from and the filter's seed check holds.
    Its Runs share the seed's draws row by row, as recorded."""
    cell_seed_bases: Mapping[str, int] = _EMPTY
    """Cell -> the seed its Runs count from: Run ``i`` (in the cell's order) records at ``base + i``. The protocol
    message's cell seeds (Train 9000, GA-O 9100, GA-S 9200, GA-OS 9300), one per Run so a cell's Runs never share
    their draws (a cell's 120 rows all distinct). The cells of a task declaring none take ``cell_run_seed``."""
    collection: CollectionSpec | None = None
    """The training collection's start-state dump; None for a task whose training starts do not come from one."""
    collection_settings: Mapping[str, str] = _EMPTY
    """Environment field -> why the hand-written collection file (``db_<task>_collect.yaml``) sets it otherwise than
    the Train cell, for a teleop scene the benchmark does not run (DB-R1's headset scene and its own layout sets).
    The collection/Train pair check compares every other field."""
    policy_execution: PolicyExecution | None = GA_POLICY_EXECUTION
    """How a model's run of the cells and the sweep runs its policy (``model_run.compose``); one protocol for every
    GA task. None asks it every step of the task's own control rate."""
    scoring_environment: Mapping[str, Any] = _EMPTY
    """Environment fields that change only how an episode is judged, not what it runs (DB-L1's ``l1_headline_bags``):
    set on a model's run of the cells and the sweep (``model_run.compose``) and in the cells' declaration, like
    ``policy_execution``, and in no generated file, so the condition and collection recordings keep the
    environment's own success."""
    challenge: str | None = None
    """A challenge track scored beside the headline and never mixed into it: its rule, in words. Each episode records
    its verdict as ``challenge_success`` next to its ``success`` (the headline's), so one run yields both; the scorer
    refuses a row without it. DB-L1 alone declares one (headline the first two bags, challenge every bag).
    The same "first N of M, with all M as the challenge" split could serve another multi-object task later; nothing
    beyond this field and the scorer's block is generic yet."""
    physical_training: PhysicalRange | None = None
    """The physical axis's training range, which every cell but GA-P and GA-SP draws; None: the task has no such axis."""
    physical_beyond: PhysicalRange | None = None
    """The physical axis's range beyond training, which GA-P and GA-SP draw."""
    condition_episode_s: float = CONDITION_EPISODE_S
    """How long each episode of the cells' and the sweep's condition recordings holds the reset pose before the settle
    check: the protocol's, unless the task's start has nothing that moves for longer (DB-L1, as its collection dump's
    ``CollectionSpec.episode_s``). The rows record it; the replay's cap is ``cap_s``, not this."""
    runtime_inputs: Mapping[str, str] = _EMPTY
    """Files a Run reads from the tree it runs in besides its frozen rows, each with the command that installs it: an
    environment that loads its objects from an input package (DB-H3's in-hand bundles). The manifest carries them and
    the eval launcher refuses to plan a launch on a tree that lacks one."""
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not (math.isfinite(self.condition_episode_s) and 0.0 < self.condition_episode_s <= CONDITION_EPISODE_S):
            raise ValueError(f"{self.name}: a condition episode lasts (0, {CONDITION_EPISODE_S}] s, "
                             f"got {self.condition_episode_s}")
        if not all(self.collection_settings.values()):
            raise ValueError(f"{self.name}: a collection setting the Train cell does not share says why")
        unknown = sorted(set(self.cells) - set(CELL_LABELS))
        if unknown:
            raise ValueError(f"{self.name}: no cells {unknown}; the protocol's are {list(CELL_LABELS)}")
        physical = (self.physical_training, self.physical_beyond)
        if (physical[0] is None) != (physical[1] is None):
            raise ValueError(f"{self.name}: a physical axis declares both its training range and the range beyond it")
        if (physical[0] is not None) != bool(set(self.cells) & set(PHYSICAL_CELLS)):
            raise ValueError(f"{self.name}: the cells {PHYSICAL_CELLS} go with a physical axis, and a physical axis with them")
        if physical[0] is None:
            return
        for spatial in (self.training, self.beyond):
            for range_ in physical:
                shared = (set(spatial.environment) & set(range_.environment)) | (set(spatial.variations) & set(range_.variations))
                if shared:
                    raise ValueError(f"{self.name}: the spatial and physical ranges both set {sorted(shared)}")

    @property
    def cap_s(self) -> float:
        """The episode cap every evaluation and sweep Run states."""
        return self.durations.cap_s(self.hold_s)

    def physical_for(self, cell: str) -> PhysicalRange | None:
        """The physical range ``cell`` draws from; None in a task without a physical axis."""
        if self.physical_training is None:
            return None
        return self.physical_training if CELL_PHYSICAL[cell] == "training" else self.physical_beyond

    def range_for(self, cell: str) -> SpatialRange:
        """What ``cell`` draws: its spatial range, with its physical range's fields and variations added."""
        spatial = self.training if CELL_SPATIAL[cell] == "training" else self.beyond
        physical = self.physical_for(cell)
        if physical is None:
            return spatial
        return dataclasses.replace(spatial, environment={**spatial.environment, **physical.environment},
                                   variations={**spatial.variations, **physical.variations})


_HEAD_CAMERAS = ("robot_pov_left_cam", "robot_pov_right_cam")
_PALM_CAMERAS = ("wrist_left_cam", "wrist_right_cam")
"""The palm cameras every sweep renders (``wrist_cameras: true``); the embodiment registers their variations with the
head cameras' (``PALM_CAMERA_RIGS``)."""
_SWEEP_CAMERAS = _HEAD_CAMERAS + _PALM_CAMERAS
_ROBOT = RBY1_WUJI2_JOINT_EMBODIMENT


def _camera_variations() -> dict[str, Any]:
    """The head and palm cameras' extrinsics variations, each drawn independently per camera (one variation per camera,
    as the RBY1-Wuji2 embodiment's ``add_camera_variations`` registers them): a translation uniform per axis in the
    camera's optical frame (+X right, +Y down, +Z forward), added to its mount, so a palm camera keeps following its
    wrist. The ranges are ``HEAD_CAMERA_RANGE_M`` and ``PALM_CAMERA_RANGE_M``."""
    variations: dict[str, Any] = {}
    for cameras, (low, high) in ((_HEAD_CAMERAS, HEAD_CAMERA_RANGE_M), (_PALM_CAMERAS, PALM_CAMERA_RANGE_M)):
        for camera in cameras:
            extrinsics = f"{_ROBOT}.camera_extrinsics_{camera}"
            variations[f"{extrinsics}.enabled"] = True
            variations[f"{extrinsics}.sampler_cfg.low"] = list(low)
            variations[f"{extrinsics}.sampler_cfg.high"] = list(high)
    return variations


HEAD_CAMERA_RANGE_M = ((-0.03, -0.03, -0.03), (0.03, 0.03, 0.03))
PALM_CAMERA_RANGE_M = ((-0.005, -0.005, 0.0), (0.005, 0.005, 0.005))
_EXTRINSICS_KEYS = tuple(f"{_ROBOT}.camera_extrinsics_{camera}" for camera in _SWEEP_CAMERAS)
_CAMERA_FACTORS = {
    **{f"{_ROBOT}.camera_extrinsics_{camera}":
       "head camera translation, +-30 mm per axis, each camera drawn independently" for camera in _HEAD_CAMERAS},
    **{f"{_ROBOT}.camera_extrinsics_{camera}":
       "palm camera translation, x, y +-5 mm and z (optical axis) 0 to +5 mm, each camera drawn independently"
       for camera in _PALM_CAMERAS},
}
_CAMERA_COVERS = {"camera_extrinsics": _EXTRINSICS_KEYS}
_UNAVAILABLE = {
    "camera_intrinsics": "held by choice: each camera keeps its calibrated intrinsics, which a mount change does not "
                         "alter (Arena's CameraIntrinsicsVariation exists at the pin)",
    "camera_rotation": "CameraExtrinsicsVariation draws a translation only at the pin and on Arena main; rotation "
                       "coordinates are an upstream ask",
    "texture": "no material or texture variation at the pin or Arena main; the smallest addition is a run-time "
               "material-swap variation of ours",
}
INDOOR_COLOR_TEMPERATURE_K = (2700.0, 6500.0)

BASE_XY_TRAINING_M = 0.02
BASE_YAW_TRAINING_DEG = 3.0
BASE_YAW_BEYOND_DEG = 6.0


def _base_pose(yaw_low_deg: float, yaw_high_deg: float, xy_m: float = BASE_XY_TRAINING_M,
               y_m: tuple[float, float] | None = None) -> dict[str, Any]:
    """The base's Arena pose variation (``<embodiment>.base_pose``, which every DexBench scene attaches to its robot,
    scene_environment_base): x uniform in [-xy_m, xy_m] about the resolved base, y in ``y_m`` (default the same as x),
    yaw uniform in [low, high] deg about the robot's facing (a point where the two are equal). DB-H1's ``_h1_draws`` is
    the same variation."""
    low, high = (round(math.radians(yaw), 7) + 0.0 for yaw in (yaw_low_deg, yaw_high_deg))
    y_low, y_high = y_m if y_m is not None else (-xy_m, xy_m)
    return {
        f"{_ROBOT}.base_pose.enabled": True,
        f"{_ROBOT}.base_pose.sampler_cfg.low": [-xy_m, y_low, low],
        f"{_ROBOT}.base_pose.sampler_cfg.high": [xy_m, y_high, high],
    }


_BASE_TRAINING = _base_pose(-BASE_YAW_TRAINING_DEG, BASE_YAW_TRAINING_DEG)


def _yaw_key(yaw_deg: float) -> str:
    """``base_yaw_p6``, ``base_yaw_m5p8``: a turned-base setting's Run-name suffix (no dot or minus in a Run name)."""
    return f"base_yaw_{'p' if yaw_deg > 0 else 'm'}{abs(yaw_deg):g}".replace(".", "p")


def _spread(blocks: tuple[tuple[str, int], ...], settings: tuple[str, ...]) -> dict[str, int]:
    """``{<block>_<setting>: rows}``: each block (an object, or an object in a lane) keeps its rows, split over the
    settings as evenly as they go: every setting its floor share, and the remainder one row each to the next settings
    of ``settings`` in turn, the turn running on across the blocks in order. So every setting's total differs from
    another's by one row at most, whatever the blocks' sizes."""
    quotas: dict[str, int] = {}
    turn = 0
    for block, rows in blocks:
        share, extra = divmod(rows, len(settings))
        if not share:
            raise ValueError(f"{block}: {rows} rows leave a setting of {len(settings)} empty")
        bonus = {(turn + step) % len(settings) for step in range(extra)}
        turn = (turn + extra) % len(settings)
        for index, setting in enumerate(settings):
            quotas[f"{block}_{setting}"] = share + (index in bonus)
    return quotas


_COMPASS = (("e", 0), ("ne", 45), ("n", 90), ("nw", 135), ("w", 180), ("sw", 225), ("s", 270), ("se", 315))
"""The eight world directions a fixed beyond setting points in (DB-R3, DB-M4), by the angle from world +x."""


def _compass(radius_m: float, angle_deg: float) -> tuple[float, float]:
    """``(dx, dy)`` ``radius_m`` toward world ``angle_deg``, to 0.1 mm."""
    angle = math.radians(angle_deg)
    return (round(radius_m * math.cos(angle), 4) + 0.0, round(radius_m * math.sin(angle), 4) + 0.0)



R1_EVAL_LAYOUTS = "outputs/db_conditions/db_r1_layouts/eval"

DB_R1 = BenchmarkTaskSpec(
    task="r1",
    name="DB-R1",
    title="retail_v1, pick_fruit_from_display at the Fruit anchor",
    environment=MappingProxyType({
        "type": "dexbench_retail_v1_environment",
        "embodiment": _ROBOT,
        "arm_mode": "dual_arm",
        "task": "pick_fruit_from_display",
        "scene_layout": "fc_260821",
        "anchor": "Fruit",
        "start_pose": "teleop_home",
        "rby1_control_hz": 30,
        "with_full_items": True,
        "r1_layouts": R1_EVAL_LAYOUTS,
    }),
    hold_s=0.0,
    cells=("train", "ga_o"),
    training=SpatialRange(
        note=(
            "the pile's start layout: one dense settled layer of the kind (30 fruit; apple 26, pear 22, bell "
            "pepper 18), five shape variants mixed, on the divided tray's pile compartment, drawn from the kind's "
            "evaluation seed stream, apart from the collection's; the robot at teleop_home, no base draw "
            "(db_r1_collect.yaml's range, which the 240 training demos cover)"
        ),
    ),
    beyond=SpatialRange(
        note="none: DB-R1 declares no range beyond its training layouts, so it has no GA-S or GA-OS cell",
    ),
    frozen=FrozenRows(
        cells=("train", "ga_o"),
        command=(
            f"python3 scripts/tools/write_r1_layout_rows.py {R1_EVAL_LAYOUTS} --r1_target_sku <kind>_pile "
            "--out outputs/db_conditions/db_r1_fc_260821_<cell>/<kind>_pile.jsonl, for each Run"
        ),
        note=(
            "the 240 evaluation layouts, 30 per kind: of each kind's 60 baked layouts (bake_r1_layouts.py "
            "--phase evaluation, each reloaded three times within 3 mm), the 30 with the lowest seeds, by seed order "
            "alone; one row per layout, in seed order"
        ),
        invalid_starts=(
            "checked before freezing: every layout reloaded in place three times on this build's physics "
            "(bake_r1_layouts.py --check: no fruit moves more than 3 mm in 2 s of loading or 4 s idle, "
            "none leaves the tray); a start that still fails counts as a failure for every model alike"
        ),
    ),
    durations=DemoDurations.from_lengths_file("db_r1.json", revision="03542f21aac00b972c03443911b05567a539b7eb"),
    sensitivity=Sensitivity(
        legacy_unlabelled_paths=True,
        object_key="plum_pile",
        factors={
            "dexbench_retail_v1_lighting.intensity_scale": "x0.25-1.5 of the delivered lights (RETAIL_V1_LIGHT_INTENSITY_SCALE_RANGE)",
            "dexbench_retail_v1_lighting.color_temperature": "2700-6500 K on every light (RETAIL_V1_LIGHT_COLOR_TEMPERATURE_RANGE_K)",
            **_CAMERA_FACTORS,
        },
        variations={
            "dexbench_retail_v1_lighting.intensity_scale.enabled": True,
            "dexbench_retail_v1_lighting.color_temperature.enabled": True,
            **_camera_variations(),
        },
        covers={"light_intensity": ("dexbench_retail_v1_lighting.intensity_scale",),
                "color_temperature": ("dexbench_retail_v1_lighting.color_temperature",),
                **_CAMERA_COVERS},
        unavailable={**_UNAVAILABLE, "distractors": "the other fruit of the pile are the task's: none may leave the "
                                                    "tray"},
    ),
    collection_settings=MappingProxyType({
        "r1_layouts": "collection drew from its own baked layout sets (db_r1_layouts/batch_1/<kind>), apart from the "
                      "cells'",
        "r1_minimal_scene": "the headset scene: only the display, its cart and basket, none of the room drawn",
        "with_full_items": "the stocked backdrop is off in the headset scene; the camera re-record that made the "
                           "training images put it back, as the cells run",
    }),
    notes=("The spatial axis has its training layouts and no range beyond them: Train and GA-O only, 240 episodes "
           "per model.",),
)


R2_YAW_RING_DEG = (-0.5, 3.0)
R2_YAW_BEYOND_DEG = (6.0, -4.0)
_R2_BASE_RING_TURN = _base_pose(*R2_YAW_RING_DEG, xy_m=0.0)

_R2_LANES = tuple(
    SpatialSetting(key=f"mf{lane}", environment={"r2_target_lane": lane}, note=f"target in MiniFit lane MF{lane}")
    for lane in (3, 4, 5)
)
_R2_RING = tuple(
    SpatialSetting(
        key=f"base_{direction}",
        environment={"base_offset_x": round(0.04 * math.cos(math.radians(angle)), 4) + 0.0,
                     "base_offset_y": round(0.04 * math.sin(math.radians(angle)), 4) + 0.0},
        variations=_R2_BASE_RING_TURN,
        note=f"4.0 cm toward world {angle} deg, re-aimed, turned -0.5 to +3 deg",
    )
    for direction, angle in (("e", 0), ("w", 180), ("sw", 225), ("s", 270), ("se", 315))
) + tuple(
    SpatialSetting(key=_yaw_key(yaw), environment={}, variations=_base_pose(yaw, yaw),
                   note=f"turned {abs(yaw):g} deg {'left (+z)' if yaw > 0 else 'right (-z)'} from its facing, x, y drawn +-2 cm")
    for yaw in R2_YAW_BEYOND_DEG
)
_R2_LANE_RING = tuple(
    SpatialSetting(key=f"{lane.key}_{point.key}", environment={**lane.environment, **point.environment},
                   variations=point.variations, note=f"{lane.note}, base {point.note}")
    for lane in _R2_LANES for point in _R2_RING
)

_R2_TRAIN_DRINKS = ("tc1", "tc2", "tp1")
_R2_TURNED_KEEP = {"tc1_mf3": 12, "tc1_mf4": 18, "tc1_mf5": 10, "tc2_mf3": 12, "tc2_mf4": 16, "tc2_mf5": 12,
                   "tp1_mf3": 12, "tp1_mf4": 16, "tp1_mf5": 12}
_R2_TURNED_FILE = {name: 30 if name == "tc1_mf3" else 2 * rows for name, rows in _R2_TURNED_KEEP.items()}
_R2_TRAINING_XY = SpatialRange(
    note=(
        "the target in MiniFit lane MF3, MF4 or MF5, the 240 training demos' lanes, 40 rows each; the base "
        "drawn uniformly from the square x, y in [-2, 2] cm, no yaw (dexbench_rby1_wuji2_joint.base_pose); "
        "reach is checked over a 3 cm disk, which contains it"
    ),
    environment={"base_jitter_radius": 0.02},
    settings=_R2_LANES,
)
_R2_HELD_OUT_DRINKS = ("ec1", "ec2", "ep2", "ep3")


def _r2_row_quotas() -> dict[str, dict[str, int]]:
    """Train: 40 rows per lane over the three drinks, the odd row on the lane's own drink (14 + 13 + 13), so
    each drink also gets 40. GA-S: each drink and lane block keeps Train's 14 or 13, and GA-OS's held-out drink
    and lane blocks 10 each (120 over 4 x 3), split over the block's seven settings (the five ring points e, w,
    sw, s, se, then the base turned +6 and -4 deg) by ``_spread``: 2, or 1 with the rest in turn, across the
    blocks lane by lane. Each drink and lane keeps its 40 (GA-S) or 30 (GA-OS), and each setting gets 17 or 18
    of the cell's 120. GA-O splits evenly (4 drinks x 3 lanes x 10)."""
    lanes = tuple(lane.key for lane in _R2_LANES)
    points = tuple(point.key for point in _R2_RING)
    train = {f"{drink}_{lane}": 13 + (drink_index == lane_index)
             for lane_index, lane in enumerate(lanes) for drink_index, drink in enumerate(_R2_TRAIN_DRINKS)}
    held_out = tuple((f"{drink}_{lane}", 10) for lane in lanes for drink in _R2_HELD_OUT_DRINKS)
    return {"train": train, "ga_s": _spread(tuple(train.items()), points), "ga_os": _spread(held_out, points)}


DB_R2 = BenchmarkTaskSpec(
    task="r2",
    name="DB-R2",
    title="retail_v1, retrieve_beverage_from_clutter at the Fridge anchor",
    environment=MappingProxyType({
        "type": "dexbench_retail_v1_environment",
        "embodiment": _ROBOT,
        "arm_mode": "dual_arm",
        "task": "retrieve_beverage_from_clutter",
        "scene_layout": "fc_260821",
        "anchor": "Fridge",
        "start_pose": "teleop_home",
        "rby1_control_hz": 30,
        "with_full_items": True,
        "fridge_shelf3_raise_m": 0.065,
        "r2_stock_rule": "unique_target",
        "r2_stock_fillers": "TC2,TC1",
        "success_tier": "strict",
    }),
    training=SpatialRange(
        note=(
            "the target in MiniFit lane MF3, MF4 or MF5, the 240 training demos' lanes, 40 rows each; the base "
            "drawn uniformly from the square x, y in [-2, 2] cm and turned uniformly in [-3, 3] deg "
            "(dexbench_rby1_wuji2_joint.base_pose); reach is checked over a 3 cm disk, which contains the x, y "
            "draw, at no turn; the settle filter guards the turned starts"
        ),
        variations=_BASE_TRAINING,
        settings=_R2_LANES,
    ),
    beyond=SpatialRange(
        note=(
            "the target in MF3, MF4 or MF5 and the base fixed 4 cm out, toward e, w, sw, s and se, turned at its "
            "training draw on the left and at most 0.5 deg right (R2_YAW_RING_DEG); n, ne "
            "and nw are out, since the fingers meet fridge_01/02 or the door frame at reset from teleop_home on every "
            "lane. At MF3, IK reaches TC2's grasp poses from s and se, with nominal joint-limit margins. GA-OS's held-out drinks are reach-checked at MF4 "
            "only: at MF3 and MF5 the settle filter guards their starts, but reach is unverified. Or the base at its "
            "training x, y draw and turned 6 deg left (base_yaw_p6) or 4 deg right (base_yaw_m4), reach-checked on "
            "every lane (R2_YAW_* says why the right side is narrower)"
        ),
        settings=_R2_LANE_RING,
    ),
    durations=DemoDurations.from_lengths_file("db_r2.json", revision="2f154c827da379b285694729776f45332998901a"),
    sensitivity=Sensitivity(
        object_key="TC2",
        factors={
            "dexbench_retail_v1_lighting.intensity_scale": "x0.25-1.5 of the delivered lights (RETAIL_V1_LIGHT_INTENSITY_SCALE_RANGE)",
            "dexbench_retail_v1_lighting.color_temperature": "2700-6500 K on every light (RETAIL_V1_LIGHT_COLOR_TEMPERATURE_RANGE_K)",
            **_CAMERA_FACTORS,
        },
        variations={
            "dexbench_retail_v1_lighting.intensity_scale.enabled": True,
            "dexbench_retail_v1_lighting.color_temperature.enabled": True,
            **_camera_variations(),
        },
        covers={"light_intensity": ("dexbench_retail_v1_lighting.intensity_scale",),
                "color_temperature": ("dexbench_retail_v1_lighting.color_temperature",),
                **_CAMERA_COVERS},
        unavailable={**_UNAVAILABLE, "distractors": "the neighbours and the stock are part of the task: a moved or removed "
                                                    "one is the failure the neighbour gate scores"},
        environment={"base_jitter_radius": 0.02, "r2_target_lane": 4},
        notes=("the base keeps the EA training draw (x, y +-2 cm, no turn), not the Train cell's turned one: the "
               "frozen sensitivity rows were recorded with it",),
    ),
    row_quotas=_r2_row_quotas(),
    collection=CollectionSpec(
        seed_base=8201,
        note=("80 demos per drink (TC1, TC2 and TP1) and 80 per lane: the Train cell's quotas doubled, 28 + 26 + 26 "
              "per lane with the extra rows on the lane's own drink. Batch 1 (8211-8219) is 120 demos over the turned "
              "training range, in the training set's drink x lane counts, each Run's "
              "file twice its demos (tc1_mf3 2.5 times) with the demos first and the reserve after"),
        row_quotas={name: 2 * rows for name, rows in _r2_row_quotas()["train"].items()},
        batches=2,
        batch_declarations={0: CollectionBatch(training=_R2_TRAINING_XY),
                            1: CollectionBatch(rows=sum(_R2_TURNED_FILE.values()), row_quotas=_R2_TURNED_FILE,
                                               keep=_R2_TURNED_KEEP)},
    ),
)


_R3_BASE_CENTRE = (0.05, -0.1)


_R3_CARTON_TRAINING = {"box_jitter_xy": 0.02}
_R3_TRAINING_XY = SpatialRange(
    note=(
        "the base drawn uniformly from the square x, y in [-2, 2] cm around the carry layout's base, no yaw "
        "(dexbench_rby1_wuji2_joint.base_pose), and the carton in the top tote from x, y in [-20, 20] mm, no yaw "
        "(target_carton.pose): db_r3_collect.yaml's range, which the 240 training demos (60 per carton, R3-T1..T4) "
        "cover, reachable and contact-free at reset"
    ),
    environment={"base_jitter_radius": 0.02, "box_jitter_xy": 0.02},
)
R3_YAW_TRAINING_DEG = (-2.5, 3.0)
R3_YAW_COMPASS_DEG = (-1.0, 3.0)
R3_YAW_BEYOND_RIGHT_DEG = -5.8
_R3_BASE_TRAINING = _base_pose(*R3_YAW_TRAINING_DEG)
_R3_BASE_TRAINING_TURN = _base_pose(*R3_YAW_COMPASS_DEG, xy_m=0.0)
_R3_BASE_TURNED = (
    (_yaw_key(BASE_YAW_BEYOND_DEG), _base_pose(BASE_YAW_BEYOND_DEG, BASE_YAW_BEYOND_DEG),
     f"turned {BASE_YAW_BEYOND_DEG:g} deg left (+z) from its facing, x, y drawn +-2 cm"),
    (_yaw_key(R3_YAW_BEYOND_RIGHT_DEG),
     _base_pose(R3_YAW_BEYOND_RIGHT_DEG, R3_YAW_BEYOND_RIGHT_DEG, y_m=(-BASE_XY_TRAINING_M, 0.0)),
     f"turned {-R3_YAW_BEYOND_RIGHT_DEG:g} deg right (-z) from its facing, x drawn +-2 cm, y in [-2, 0] cm (away from "
     "tote_06)"),
)
_R3_TRAINING_TURN3 = SpatialRange(
    note=(
        "the base drawn uniformly from the square x, y in [-2, 2] cm around the carry layout's base and turned "
        "uniformly in [-3, 3] deg (dexbench_rby1_wuji2_joint.base_pose), and the carton in the top tote from x, y "
        "in [-20, 20] mm, no yaw (target_carton.pose): the range batch 3 is recorded at (db_r3_collect.yaml turns "
        "[-2.5, 3] deg, R3_YAW_TRAINING_DEG), which the 240 training demos (60 per carton, R3-T1..T4) cover; the "
        "x, y draws reachable and contact-free at reset, the turn guarded by the collection dump's settle filter"
    ),
    environment=_R3_CARTON_TRAINING,
    variations=_BASE_TRAINING,
)
_R3_STUDY = "spatial study"


def _r3_base_setting(direction: str, angle_deg: float, radius_m: float) -> SpatialSetting:
    """The base fixed ``radius_m`` from the carry layout's base toward world ``angle_deg`` (base_offset_x/y, which
    re-aims the yaw at the gondola) and turned from there in R3_YAW_COMPASS_DEG ([-1, 3] deg: the training turn,
    narrower to the right where the hand nears tote_06), the carton at the training draw."""
    dx, dy = _compass(radius_m, angle_deg)
    return SpatialSetting(
        key=f"base_{direction}",
        environment={"base_offset_x": round(_R3_BASE_CENTRE[0] + dx, 4) + 0.0,
                     "base_offset_y": round(_R3_BASE_CENTRE[1] + dy, 4) + 0.0,
                     **_R3_CARTON_TRAINING},
        variations=_R3_BASE_TRAINING_TURN,
        note=(f"base {radius_m * 100:.1f} cm toward world {angle_deg:g} deg, re-aimed, turned -1 to +3 deg, carton drawn +-2 cm ({_R3_STUDY}: "
              "no reset contact at 4.5 cm in any direction; reach checked at 4 and 5 cm)"),
    )


def _r3_carton_setting(direction: str, angle_deg: float, radius_m: float) -> SpatialSetting:
    """The carton fixed ``radius_m`` from its nominal spot in the top tote toward world ``angle_deg`` (box_offset_xy,
    the tote stays), no carton draw; the base at the training draw (x, y and turn)."""
    return SpatialSetting(
        key=f"carton_{direction}",
        environment={"box_offset_xy": list(_compass(radius_m, angle_deg))},
        variations=_R3_BASE_TRAINING,
        note=(f"carton {radius_m * 100:.1f} cm toward world {angle_deg:g} deg in the tote, base drawn +-2 cm and -2.5 to +3 deg "
              f"({_R3_STUDY}: every carton fits the tote at 4 cm; reach checked at the e, n, w and s points, at no "
              "turn)"),
    )


R3_BEYOND_SETTINGS: tuple[SpatialSetting, ...] = (
    *(_r3_base_setting(direction, angle, 0.045) for direction, angle in _COMPASS),
    *(_r3_carton_setting(direction, angle, 0.04) for direction, angle in _COMPASS),
    *(SpatialSetting(key=key, environment=_R3_CARTON_TRAINING, variations=variations,
                     note=f"base {note}, carton drawn +-2 cm (reset contact: tote_06 clearance >= 1 mm, all 9 cartons)")
      for key, variations, note in _R3_BASE_TURNED),
)
_R3_TRAIN_CARTONS = tuple(f"dexbench_retail_e1_fc_r3t{n}" for n in (1, 2, 3, 4))
_R3_CARTONS = {"ga_s": _R3_TRAIN_CARTONS,
               "ga_os": tuple(f"dexbench_retail_e1_fc_r3e{n}" for n in (1, 2, 4, 5, 6))}


def _r3_row_quotas() -> dict[str, dict[str, int]]:
    """GA-S (4 cartons x 18 settings = 72 Runs) and GA-OS (5 x 18 = 90) hold 120 rows each, which their Runs do not
    divide. Each carton takes its even share (30, 24), split over its 18 settings by ``_spread``: 1 row each and the
    rest (12, 6) one each in turn, through the base and carton settings of e, n, w, s, ne, nw, sw, se, a direction's
    two together, then the base turned +6 deg and R3_YAW_BEYOND_RIGHT_DEG, across the cartons in the ObjectAxis order. Every setting gets
    6 or 7 of the cell's 120: the base and the carton settings 54 each, the turned base 12, and a direction's base and
    carton settings the same."""
    compass = [_COMPASS[index][0] for index in (*range(0, 8, 2), *range(1, 8, 2))]
    order = (*(f"{axis}_{direction}" for direction in compass for axis in ("base", "carton")),
             *(key for key, _, _ in _R3_BASE_TURNED))
    return {cell: _spread(tuple((key, 120 // len(keys)) for key in keys), order) for cell, keys in _R3_CARTONS.items()}


DB_R3 = BenchmarkTaskSpec(
    task="r3",
    name="DB-R3",
    title="retail_v1, place_box_in_slot at the Gondola anchor",
    environment=MappingProxyType({
        "type": "dexbench_retail_v1_environment",
        "embodiment": _ROBOT,
        "arm_mode": "dual_arm",
        "task": "place_box_in_slot",
        "scene_layout": "fc_260821",
        "anchor": "Gondola",
        "start_pose": "r3_carry",
        "base_offset_x": _R3_BASE_CENTRE[0],
        "base_offset_y": _R3_BASE_CENTRE[1],
        "rby1_control_hz": 30,
        "with_full_items": True,
        "facing_width_mm": None,
    }),
    training=SpatialRange(
        note=(
            "the base drawn uniformly from the square x, y in [-2, 2] cm around the carry layout's base and turned "
            "uniformly in [-2.5, 3] deg (dexbench_rby1_wuji2_joint.base_pose; narrower to the right, where the r3_carry "
            "hand nears tote_06's wall), and the carton in the top tote from x, y in [-20, 20] mm, no yaw "
            "(target_carton.pose): db_r3_collect.yaml's range, which the 240 training demos (60 per carton, R3-T1..T4) "
            "cover; reachable at every corner and clear of contact at reset (Newton contact query, all 9 "
            "cartons)"
        ),
        environment=_R3_CARTON_TRAINING,
        variations=_R3_BASE_TRAINING,
    ),
    beyond=SpatialRange(
        note=(
            "one axis fixed off the training range and the other at its training draw: the base 4.5 cm out from the "
            "carry layout's base toward each of e, ne, n, nw, w, sw, s, se (base_offset_x/y, yaw re-aimed at the "
            "gondola, then turned -1 to +3 deg), the carton drawn +-2 cm; or the carton 4 cm out in the tote toward "
            "each of the same eight (box_offset_xy), the base at its training draw; or the base turned 6 deg left at "
            "its training x, y draw (base_yaw_p6), or R3_YAW_BEYOND_RIGHT_DEG right with y in [-2, 0] cm "
            "(base_yaw_m*), the carton drawn +-2 cm. Every GA-S scene lies outside the training box on the fixed axis. "
            "Reach (all 9 cartons) holds at every extreme; reset contact (Newton) clears tote_06 by "
            ">= 1 mm at every extreme. Every carton fits the tote"
        ),
        settings=R3_BEYOND_SETTINGS,
    ),
    durations=DemoDurations.from_lengths_file("db_r3.json", revision="6420de63e15e3af42d78208fb4ba8c66c109a138"),
    hold_s=0.0,
    row_quotas=_r3_row_quotas(),
    sensitivity=Sensitivity(
        object_key="dexbench_retail_e1_fc_r3t2",
        factors={
            "dexbench_retail_v1_lighting.intensity_scale": "x0.25-1.5 of the delivered lights (RETAIL_V1_LIGHT_INTENSITY_SCALE_RANGE)",
            "dexbench_retail_v1_lighting.color_temperature": "2700-6500 K on every light (RETAIL_V1_LIGHT_COLOR_TEMPERATURE_RANGE_K)",
            **_CAMERA_FACTORS,
        },
        variations={
            "dexbench_retail_v1_lighting.intensity_scale.enabled": True,
            "dexbench_retail_v1_lighting.color_temperature.enabled": True,
            **_camera_variations(),
            **_R3_BASE_TRAINING,
        },
        covers={"light_intensity": ("dexbench_retail_v1_lighting.intensity_scale",),
                "color_temperature": ("dexbench_retail_v1_lighting.color_temperature",),
                **_CAMERA_COVERS},
        unavailable={**_UNAVAILABLE, "distractors": "the six neighbours are part of the slot task: they fix the "
                                                    "facing the carton goes into"},
        environment=_R3_CARTON_TRAINING,
    ),
    collection=CollectionSpec(
        seed_base=8101,
        note=("60 demos per training carton (R3-T1..T4) over the training range, recorded 75 per carton (batch 0 at "
              "8101-8104, batch 1 at 8111-8114, batch 2 at 8121-8124; batch 3 at 8131-8134 is 30 demos per carton over "
              "the turned training range, each file 60 rows, the demos first and the reserve after). Recorded with "
              "the IK twin, waist IK on"),
        overrecord=1.25,
        environment=MappingProxyType({"embodiment": "dexbench_rby1_wuji2", "waist_ik": True}),
        batches=4,
        batch_declarations={**{batch: CollectionBatch(training=_R3_TRAINING_XY) for batch in range(3)},
                            3: CollectionBatch(rows=240, row_quotas={key: 60 for key in _R3_TRAIN_CARTONS},
                                               keep={key: 30 for key in _R3_TRAIN_CARTONS},
                                               training=_R3_TRAINING_TURN3)},
    ),
    notes=("The training set is 240 demos, 60 per training carton over the training range; each cell replays 120 rows.",),
)


_L2_CAGE_TRAINING = ((0.0, 0.0, -0.01), (1.0, 0.08, 0.01))
DB_L2 = BenchmarkTaskSpec(
    task="l2",
    name="DB-L2",
    title="E2 logistics, lift_to_conveyor at the A1_CAGE anchor",
    environment=MappingProxyType({
        "type": "dexbench_logistics_center_e2_environment",
        "embodiment": _ROBOT,
        "task": "lift_to_conveyor",
        "anchor": "A1_CAGE",
        "rby1_control_hz": 30,
        "cage_box_alt_scales": [0.85, 1.15],
        "base_jitter_radius": 0.0,
        "newton_num_substeps": 8,
    }),
    hold_s=0.0,
    recorded_cell_seeds={"train": 9000, "ga_o": 9100, "ga_s": 9200, "ga_os": 9300},
    training=SpatialRange(
        note=(
            "cage stack: box set uniform over the three scales, depth 0-80 mm toward the back wall, lateral +-10 mm; "
            "the box's own pose on the tier over its whole safe range (cage_l2_package_pose_variation_range)"
        ),
        variations={
            "cage_l2_package.cage_stack.enabled": True,
            "cage_l2_package.cage_stack.sampler_cfg.low": list(_L2_CAGE_TRAINING[0]),
            "cage_l2_package.cage_stack.sampler_cfg.high": list(_L2_CAGE_TRAINING[1]),
            "cage_l2_package.pose.enabled": True,
        },
    ),
    beyond=SpatialRange(
        note=(
            "cage stack recorded over depth -50 to +88 mm and lateral +-13 mm, the widest box every set and "
            "every object fits (CageStackVariation.validate_range with the 1.15 set), keeping the rows outside "
            "the training box: in practice the stack up to 50 mm nearer the robot, or 1-3 mm further out sideways "
            "or deeper. The box pose already spans its safe range in training, so it has no beyond"
        ),
        variations={
            "cage_l2_package.cage_stack.enabled": True,
            "cage_l2_package.cage_stack.sampler_cfg.low": [0.0, -0.05, -0.013],
            "cage_l2_package.cage_stack.sampler_cfg.high": [1.0, 0.088, 0.013],
            "cage_l2_package.pose.enabled": True,
        },
        outside=("cage_l2_package.cage_stack", *_L2_CAGE_TRAINING),
        outside_keep_share=0.55,
    ),
    collection=CollectionSpec(
        seed_base=8301,
        note="120 demos per training box (train_0 and train_1), as the training set",
    ),
    durations=DemoDurations.from_lengths_file("db_l2.json", revision="5121312cc648228283e554eff5f673d10d475272"),
    sensitivity=Sensitivity(
        legacy_unlabelled_paths=True,
        object_key="train_0",
        factors={
            "high_bay_light_01.intensity": "x0.25-1.5 of the high-bay's 6302 (DexBenchInteriorLightBase), the high-bay nearest the station, over HB2's infeed end and the cage's front",
            "high_bay_light_01.color_temperature": "2700-6500 K",
            "high_bay_light_04.intensity": "x0.25-1.5, the next nearest, north of the cage",
            "high_bay_light_04.color_temperature": "2700-6500 K",
            **_CAMERA_FACTORS,
        },
        variations={
            **{f"high_bay_light_0{n}.{factor}.enabled": True for n in (1, 4) for factor in ("intensity", "color_temperature")},
            **{f"high_bay_light_0{n}.color_temperature.sampler_cfg.low": [INDOOR_COLOR_TEMPERATURE_K[0]] for n in (1, 4)},
            **{f"high_bay_light_0{n}.color_temperature.sampler_cfg.high": [INDOOR_COLOR_TEMPERATURE_K[1]] for n in (1, 4)},
            **_camera_variations(),
            "cage_l2_package.cage_stack.enabled": True,
            "cage_l2_package.cage_stack.sampler_cfg.low": list(_L2_CAGE_TRAINING[0]),
            "cage_l2_package.cage_stack.sampler_cfg.high": list(_L2_CAGE_TRAINING[1]),
            "cage_l2_package.pose.enabled": True,
        },
        covers={"light_intensity": ("high_bay_light_01.intensity", "high_bay_light_04.intensity"),
                "color_temperature": ("high_bay_light_01.color_temperature", "high_bay_light_04.color_temperature"),
                **_CAMERA_COVERS},
        unavailable={**_UNAVAILABLE, "distractors": "the cage boxes carry the box, so removing one is not a distraction"},
        notes=("no scene-wide light factor: E2's twelve lights are separate Arena lights, so the sweep varies the two "
               "high-bays nearest the station; a grouped light variation is the smallest addition",),
    ),
)


_L1_YAW_TRAINING_DEG = [0.0, 30.0]
_L1_YAW_BEYOND_DEG = [60.0, 90.0]
_L1_TRAIN_SETS = ("train_a", "train_b", "train_c", "train_d")
_L1_KEEP = 60
_L1_FILE_ROWS = 75
_L1_STATION = "the IK twin, waist IK on"
DB_L1 = BenchmarkTaskSpec(
    task="l1",
    name="DB-L1",
    title="E2 logistics, pick_polybag_from_conveyor at the A_L1 anchor on HB2",
    environment=MappingProxyType({
        "type": "dexbench_logistics_e2_l1_environment",
        "embodiment": _ROBOT,
        "task": "pick_polybag_from_conveyor",
        "anchor": "A_L1",
        "rby1_control_hz": 30,
        "belt_speed": 0.2,
        "l1_stream_mode": "episode",
        "l1_stream_bags": 3,
        "l1_stream_boxes": 2,
        "l1_stream_lead": "bag",
        "l1_item_y": [0.80, 0.95],
        "l1_spacing": [0.72, 0.88],
        "polybag_coupling": "two_way",
        "polybag_contact_ke": 3000.0,
        "polybag_contact_kd": 1.0,
        "polybag_collider_ke": 3000.0,
        "polybag_collider_kd": 1.0,
        "polybag_contact_mu": 1.0,
        "newton_solver": "newton",
        "polybag_stepping": "a",
        "polybag_vbd_iterations": 8,
        "polybag_collider_mu": 3.0,
        "l1_bag_wrist_link_collision": "off",
        "polybag_contact_barrier": "off",
        "polybag_vbd_init": "contact",
        "polybag_film_hessian": "psd",
        "polybag_tet_poisson": 0.45,
        "polybag_tet_damping": 10.0,
    }),
    training=SpatialRange(
        note=(
            "every bag's long side 0-30 deg off the belt's direction (l1_bag_yaw_off_belt_deg, either way along it), the "
            "boxes the same; each item's centre across HB2 uniform in y 0.80-0.95 m and the belt's run between releases "
            "0.72-0.88 m, drawn per episode by the layout draw (l1_park_slab.stream_layout) with the three bags, the two "
            "boxes and their order"
        ),
        environment={"l1_bag_yaw_off_belt_deg": _L1_YAW_TRAINING_DEG},
    ),
    beyond=SpatialRange(
        note=(
            "every bag's long side 60-90 deg off the belt's direction, across it (the boxes keep the training band); the "
            "rest as in training. 30-60 deg is in neither range"
        ),
        environment={"l1_bag_yaw_off_belt_deg": _L1_YAW_BEYOND_DEG},
    ),
    challenge="every bag of the episode (three) placed and no box disturbed: the rule the episode ends on",
    scoring_environment={"l1_headline_bags": 2},
    durations=DemoDurations.from_lengths_file("db_l1.json", revision="cb1ae405a2a63574d108aa3d12936f07ecb67f94"),
    hold_s=0.0,
    condition_episode_s=2.0,
    sensitivity=Sensitivity(
        legacy_unlabelled_paths=True,
        object_key="train_a",
        factors={
            "high_bay_light_02.intensity": "x0.25-1.5 of the high-bay's 6302 (DexBenchInteriorLightBase), the high-bay nearest the station, over HB2 at the pick",
            "high_bay_light_02.color_temperature": "2700-6500 K",
            "high_bay_light_05.intensity": "x0.25-1.5, the next nearest, north of the station",
            "high_bay_light_05.color_temperature": "2700-6500 K",
            **_CAMERA_FACTORS,
        },
        variations={
            **{f"high_bay_light_0{n}.{factor}.enabled": True for n in (2, 5) for factor in ("intensity", "color_temperature")},
            **{f"high_bay_light_0{n}.color_temperature.sampler_cfg.low": [INDOOR_COLOR_TEMPERATURE_K[0]] for n in (2, 5)},
            **{f"high_bay_light_0{n}.color_temperature.sampler_cfg.high": [INDOOR_COLOR_TEMPERATURE_K[1]] for n in (2, 5)},
            **_camera_variations(),
        },
        covers={"light_intensity": ("high_bay_light_02.intensity", "high_bay_light_05.intensity"),
                "color_temperature": ("high_bay_light_02.color_temperature", "high_bay_light_05.color_temperature"),
                **_CAMERA_COVERS},
        unavailable={**_UNAVAILABLE, "distractors": "the boxes on the belt are the task's distractors: one disturbed is "
                                                    "the failure the episode scores"},
        environment={"l1_bag_yaw_off_belt_deg": _L1_YAW_TRAINING_DEG},
        notes=("no scene-wide light factor: E2's twelve lights are separate Arena lights, so the sweep varies the two "
               "high-bays nearest A_L1 (2.70, 1.60): high_bay_light_02 at (3, 1), 0.67 m off, and high_bay_light_05 at "
               "(3, 3), 1.43 m off",),
    ),
    collection=CollectionSpec(
        seed_base=8401,
        note=("60 demos per training set (train_a..train_d), the Train cell's Runs. Of each Run's 75 rows the first 60 "
              "are its demos and the last 15 its reserve. Recorded with " + _L1_STATION),
        overrecord=1.1,
        episode_s=2.0,
        environment=MappingProxyType({"embodiment": "dexbench_rby1_wuji2", "waist_ik": True}),
        batch_declarations={0: CollectionBatch(rows=_L1_FILE_ROWS * len(_L1_TRAIN_SETS),
                                               row_quotas={key: _L1_FILE_ROWS for key in _L1_TRAIN_SETS},
                                               keep={key: _L1_KEEP for key in _L1_TRAIN_SETS})},
    ),
    notes=("GA-O holds out colour (eval_c, eval_d: purple and light blue on training sizes) and size (eval_a, eval_b: "
           "grey 35 x 45 and black 20 x 40 in training colours); an unseen bag profile joins it as a further set later, "
           "a new Run whose rows are recorded then. The training set is 240 demos, 60 per training set.",),
)



def _h1_draws(radius_m: float) -> dict[str, Any]:
    """The H1 scene set's variation block at a range: base, source stack and destination stack, base yaw +-5 deg."""
    yaw = round(math.radians(5.0), 7)
    return {
        f"{_ROBOT}.base_pose.enabled": True,
        f"{_ROBOT}.base_pose.sampler_cfg.low": [-radius_m, -radius_m, -yaw],
        f"{_ROBOT}.base_pose.sampler_cfg.high": [radius_m, radius_m, yaw],
        "table_plate_00.source_stack_pose.enabled": True,
        "table_plate_00.source_stack_pose.sampler_cfg.low": [-radius_m, -radius_m, 0.0],
        "table_plate_00.source_stack_pose.sampler_cfg.high": [radius_m, radius_m, 0.0],
        "clearing_plate_00.dest_stack_pose.enabled": True,
        "clearing_plate_00.dest_stack_pose.sampler_cfg.low": [-radius_m, -radius_m, 0.0],
        "clearing_plate_00.dest_stack_pose.sampler_cfg.high": [radius_m, radius_m, 0.0],
    }


DB_H1 = BenchmarkTaskSpec(
    task="h1",
    name="DB-H1",
    title="E4 banquet hall, stack_plates at A_h1",
    environment=MappingProxyType({
        "type": "dexbench_hotel_fb_e4_environment",
        "embodiment": _ROBOT,
        "task": "stack_plates",
        "anchor": "A_h1",
        "rby1_control_hz": 30,
        "hide_replay_prims": True,
        "stack_alignment_point": "rim",
        "success_follows_dest_draw": True,
    }),
    hold_s=0.0,
    training=SpatialRange(
        note="base, source stack and destination stack each uniform by area in a 0-3 cm disk, base yaw +-5 deg (the H1 scene set's Train draws, both 240-demo sets' range)",
        variations=_h1_draws(0.03),
    ),
    beyond=SpatialRange(
        note="each uniform by area in a 3-5 cm ring, base yaw +-5 deg (the H1 scene set's GA-S draws)",
        variations=_h1_draws(0.05),
    ),
    frozen=FrozenRows(
        cells=CELLS,
        command=(
            "scripts/tools/write_db_h1_eval_conditions.py --definitions <set>/eval_scene_definitions.json "
            "--states <set>/initial_states.hdf5 --output_dir outputs/db_conditions"
        ),
        note=(
            "the benchmark's 480 of the 1,200 frozen scenes: for each cell and dish, the 40 with the lowest scene "
            "numbers (db_h1_selected_scenes.json, sha256 69f71fd3...), outcome-blind, none matching a draw of either "
            "240-demo training set (EA db_h1_training_draws.json, GA db_h1_ga_training_draws.json)"
        ),
        invalid_starts=(
            "kept: the 1,200 states were validated before selection (each is its dish's nominal plus its draws), and a "
            "start that still fails counts as a failure for every model alike"
        ),
    ),
    durations=DemoDurations.from_lengths_file("db_h1.json", revision="c1337d61ffeeac9db5a20991fda17bb5c5936006"),
    sensitivity=Sensitivity(
        object_key="harmynta_deep_plate_22",
        factors={
            "dexbench_hotel_fb_e4_lighting.intensity_scale": "x0.25-1.5 of the delivered lights (LIGHT_INTENSITY_SCALE_RANGE)",
            "dexbench_hotel_fb_e4_lighting.color_temperature": "3000-9000 K (LIGHT_COLOR_TEMPERATURE_RANGE_K)",
            **_CAMERA_FACTORS,
        },
        variations={
            "dexbench_hotel_fb_e4_lighting.intensity_scale.enabled": True,
            "dexbench_hotel_fb_e4_lighting.color_temperature.enabled": True,
            **_camera_variations(),
            **_h1_draws(0.021),
        },
        covers={"light_intensity": ("dexbench_hotel_fb_e4_lighting.intensity_scale",),
                "color_temperature": ("dexbench_hotel_fb_e4_lighting.color_temperature",),
                **_CAMERA_COVERS},
        unavailable={**_UNAVAILABLE, "distractors": "every rigid object on the tables is a dish of the task"},
    ),
)


def _m1_tabletop_draws(scale: float) -> dict[str, Any]:
    """The box the frozen rows of a cell are drawn in: ``scale`` times
    ``layouts.db_m1_tabletop_envelope.TABLETOP_TRAIN_BOUNDS_M_RAD``, the values the row writer draws with."""
    draws = {}
    for key, bounds in TABLETOP_TRAIN_BOUNDS_M_RAD.items():
        half = [round(bound * scale, 7) for bound in bounds]
        draws.update({f"{key}.enabled": True, f"{key}.sampler_cfg.low": [-value for value in half],
                      f"{key}.sampler_cfg.high": half})
    return draws


def _m1_tabletop_envelope(scale: float) -> str:
    """``_m1_tabletop_draws(scale)`` in words, for the cell notes."""
    words = []
    for key, (half_x, half_y, half_yaw) in TABLETOP_TRAIN_BOUNDS_M_RAD.items():
        asset, variation = key.split(".")
        xy = (f"XY +-{half_x * scale * 100:g} cm" if half_x == half_y
              else f"X +-{half_x * scale * 100:g} cm, Y +-{half_y * scale * 100:g} cm")
        words.append(f"{'base' if variation == 'base_pose' else asset} {xy}, yaw +-{math.degrees(half_yaw) * scale:g} deg")
    return "; ".join(words)


_M1_TABLETOP_NOMINAL_DRAWS = {
    field: value
    for key in (f"{_ROBOT}.base_pose", "drawer.pose", "washer.pose")
    for field, value in ((f"{key}.enabled", True), (f"{key}.sampler_cfg.low", [0.0, 0.0, 0.0]),
                         (f"{key}.sampler_cfg.high", [0.0, 0.0, 0.0]))
}

DB_M1 = BenchmarkTaskSpec(
    task="m1",
    name="DB-M1",
    title="E3 manufacturing, tabletop pick_insert_drawer at A_m1",
    environment=MappingProxyType({
        "type": "dexbench_manufacturing_e3_m1_environment",
        "embodiment": _ROBOT,
        "task": "pick_insert_drawer",
        "anchor": "A_m1",
        "start_pose": "teleop_ready",
        "m1_tabletop_base_xy": [0.15, -0.9757],
        "m1_tabletop_base_yaw_deg": 180.0,
        "m1_tabletop_crouch_deg": 55.0,
        "m1_tabletop_drawer_mass_kg": 0.55,
        "m1_physics_profile": "damped_contacts",
        "rby1_control_hz": 30,
        "camera_hz": 30,
    }),
    hold_s=0.0,
    cells=("train", "ga_s"),
    training=SpatialRange(
        note=(f"outcome-blind draws inside {_m1_tabletop_envelope(1.0)}, "
              "conditioned on complete worktop/pallet support; the success-selected training set is nonuniform "
              "inside this envelope and reports its actual coverage; the frozen rows supply every draw"),
        variations=_m1_tabletop_draws(1.0),
    ),
    beyond=SpatialRange(
        note=(f"the supported part of the {TABLETOP_GA_S_SCALE:g}x envelope: {_m1_tabletop_envelope(TABLETOP_GA_S_SCALE)}; "
              "every row is outside the training envelope on at least one coordinate, with no policy-success filtering"),
        variations=_m1_tabletop_draws(TABLETOP_GA_S_SCALE),
    ),
    frozen=FrozenRows(
        cells=("train", "ga_s"),
        command="scripts/tools/write_db_m1_tabletop_eval_conditions.py --output_dir outputs/db_conditions",
        note="120 support-constrained draws per cell, at independent cell_run_seed seeds; no training pose is reused",
        invalid_starts=("candidate rows must hold for 10 s under the hold-pose policy before freezing; support filtering "
                        "alone is not a physical qualification; do not replace rows based on manipulation success"),
    ),
    durations=DemoDurations.from_lengths_file("db_m1.json", revision="93bf004e9535b5def24c424edde5042909b1ede1"),
    sensitivity=Sensitivity(
        object_key="lg_fh25eae_detergent_drawer",
        factors={
            "dexbench_tube_light.intensity": "x0.25-1.5 of the tube's 60000 (DexBenchInteriorLightBase); the room's authored lights stay on",
            "dexbench_tube_light.color_temperature": "2700-6500 K",
            **_CAMERA_FACTORS,
        },
        variations={
            "dexbench_tube_light.intensity.enabled": True,
            "dexbench_tube_light.color_temperature.enabled": True,
            "dexbench_tube_light.color_temperature.sampler_cfg.low": [INDOOR_COLOR_TEMPERATURE_K[0]],
            "dexbench_tube_light.color_temperature.sampler_cfg.high": [INDOOR_COLOR_TEMPERATURE_K[1]],
            **_camera_variations(),
            **_M1_TABLETOP_NOMINAL_DRAWS,
        },
        covers={"light_intensity": ("dexbench_tube_light.intensity",),
                "color_temperature": ("dexbench_tube_light.color_temperature",),
                **_CAMERA_COVERS},
        unavailable={**_UNAVAILABLE, "distractors": "the scene holds the washer and the drawer only"},
    ),
    notes=("550 g drawer starts flat on the worktop. Full insertion and a quiet 0.5 s hold; hand contact is permitted.",
           "EA insert_drawer remains available; its frozen seeds and cap are not tabletop evaluation conditions.",
           "Sensitivity probes keep the nominal physical start while varying appearance and camera parameters."),
)



def _l3_fields(ranges: PackInvoiceSceneRanges) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(environment, dotted variations)`` that draw every pack-invoice axis in ``ranges``."""
    environment, variations = scene_fields(ranges)
    return environment, dict(dotted_items(variations))


def _l3_beyond_settings() -> tuple[SpatialSetting, ...]:
    """GA-S: one Run per axis, side and factor (``beyond_settings``), each setting only what differs from training."""
    train_env, train_var = _l3_fields(TRAINING_RANGES)
    settings = []
    for key, note, ranges in beyond_settings():
        environment, variations = _l3_fields(ranges)
        settings.append(SpatialSetting(
            key=key, note=note,
            environment={k: v for k, v in environment.items() if v != train_env[k]},
            variations={k: v for k, v in variations.items() if v != train_var[k]},
        ))
    return tuple(settings)


def _l3_range_note(ranges: PackInvoiceSceneRanges) -> str:
    def side(axis: str) -> str:
        low, high = getattr(ranges, axis)
        unit, scale = ("deg", 1.0) if axis.endswith("_deg") else ("cm", 100.0)
        return f"{axis.removesuffix('_deg')} {low * scale:+g}..{high * scale:+g} {unit}"
    return ", ".join(side(axis) for axis in ranges.__dataclass_fields__)


def _l3_ga_s_quotas(settings: tuple[SpatialSetting, ...], rows: int = 120) -> dict[str, dict[str, int]]:
    """GA-S's rows over its Runs: an even share each, the remainder one more to the first Runs (axis order)."""
    share, extra = divmod(rows, len(settings))
    return {"ga_s": {f"invoice_202x276_{setting.key}": share + (index < extra) for index, setting in enumerate(settings)}}


_L3_TRAIN_ENV, _L3_TRAIN_VAR = _l3_fields(TRAINING_RANGES)
_L3_GA_S = _l3_beyond_settings()
DB_L3 = BenchmarkTaskSpec(
    task="l3",
    name="DB-L3",
    title="E2 logistics, pack_invoice at the A3_BENCH anchor",
    environment=MappingProxyType({
        "type": "dexbench_logistics_center_e2_environment",
        "embodiment": _ROBOT,
        "task": "pack_invoice",
        "anchor": "A3_BENCH",
        "rby1_control_hz": 30,
        **PAPER_PHYSICS_ARGS,
    }),
    recorded_cell_seeds={"train": 9000, "ga_s": 9200},
    cells=("train", "ga_s"),
    training=SpatialRange(
        note=(
            "each axis uniform in its own interval about the default layout, half of how far, searched one axis "
            "at a time, the robot still reaches and the left ego camera keeps the hand, cut so the pile and the box "
            "stay on the worktop and apart at every corner (pack_invoice_scene_randomization.TRAINING_RANGES; "
            "world x, y, base included): "
            + _l3_range_note(TRAINING_RANGES)
        ),
        environment=_L3_TRAIN_ENV,
        variations=_L3_TRAIN_VAR,
    ),
    beyond=SpatialRange(
        note=(
            "one axis fixed outside training, at 1.5 times the training extent on one side, the others drawn "
            "in training (beyond_settings), one Run per axis and side"
        ),
        environment=_L3_TRAIN_ENV,
        variations=_L3_TRAIN_VAR,
        settings=_L3_GA_S,
    ),
    row_quotas=_l3_ga_s_quotas(_L3_GA_S),
    durations=DemoDurations.from_lengths_file("db_l3.json"),
    sensitivity=Sensitivity(
        object_key="invoice_202x276",
        factors={
            "task_light_strip_bench.intensity": "x0.25-1.5 of the strip under the packing bench's top shelf (1900, DexBenchStripLight)",
            "task_light_strip_bench.color_temperature": "2700-6500 K",
            "high_bay_light_06.intensity": "x0.25-1.5 of the high-bay over the packing bench (6302, DexBenchHighBayLight)",
            "high_bay_light_06.color_temperature": "2700-6500 K",
            **_CAMERA_FACTORS,
        },
        variations={
            **{f"{light}.{factor}.enabled": True for light in ("task_light_strip_bench", "high_bay_light_06")
               for factor in ("intensity", "color_temperature")},
            **{f"{light}.color_temperature.sampler_cfg.low": [INDOOR_COLOR_TEMPERATURE_K[0]]
               for light in ("task_light_strip_bench", "high_bay_light_06")},
            **{f"{light}.color_temperature.sampler_cfg.high": [INDOOR_COLOR_TEMPERATURE_K[1]]
               for light in ("task_light_strip_bench", "high_bay_light_06")},
            **_camera_variations(),
            **_L3_TRAIN_VAR,
        },
        environment=_L3_TRAIN_ENV,
        covers={"light_intensity": ("task_light_strip_bench.intensity", "high_bay_light_06.intensity"),
                "color_temperature": ("task_light_strip_bench.color_temperature", "high_bay_light_06.color_temperature"),
                **_CAMERA_COVERS},
        unavailable={**_UNAVAILABLE, "distractors": "the bench holds the task's pile and box and its fixed fixtures only"},
    ),
    notes=(
        "The object axis has one sheet and no held-out one: Train and GA-S only, 240 episodes per model.",
    ),
)



_M4_LEG = "leg.station_pose"
"""The leg and the five pieces of its holder, moved together ([dx, dy, dyaw])."""
_M4_CABINET = "base_cabinet.station_pose"
"""The cabinet and its pallet on the workbench, moved together ([dx, dy, dyaw])."""


def _m4_station(key: str, low_xy: tuple[float, float], high_xy: tuple[float, float]) -> dict[str, Any]:
    """A station draw, x, y uniform in [low, high] (a point where the two are equal), never turned."""
    return {f"{key}.enabled": True, f"{key}.sampler_cfg.low": [*low_xy, 0.0], f"{key}.sampler_cfg.high": [*high_xy, 0.0]}


M4_LEG_TRAINING_M = 0.005
M4_CABINET_TRAINING_M = 0.02
_M4_LEG_TRAINING = _m4_station(_M4_LEG, (-M4_LEG_TRAINING_M,) * 2, (M4_LEG_TRAINING_M,) * 2)
_M4_CABINET_TRAINING = _m4_station(_M4_CABINET, (-M4_CABINET_TRAINING_M,) * 2, (M4_CABINET_TRAINING_M,) * 2)
_M4_BASE_TRAINING = _base_pose(0.0, 0.0)
M4_LEG_BEYOND_M = 0.015
M4_CABINET_BEYOND_M = (0.03, 0.028)


def _m4_cabinet_point(angle_deg: float) -> tuple[float, float]:
    """The rectangle's point toward world ``angle_deg``: an edge's middle on an axis, else a corner."""
    dx, dy = _compass(1.0, angle_deg)
    return tuple(math.copysign(limit, value) if abs(value) > 1e-9 else 0.0 for value, limit in zip((dx, dy), M4_CABINET_BEYOND_M))


M4_BEYOND_SETTINGS: tuple[SpatialSetting, ...] = (
    *(SpatialSetting(
        key=f"cabinet_{direction}",
        environment={},
        variations={**_M4_BASE_TRAINING, **_M4_LEG_TRAINING,
                    **_m4_station(_M4_CABINET, _m4_cabinet_point(angle), _m4_cabinet_point(angle))},
        note=(f"cabinet fixed at x, y {tuple(round(v * 1000) for v in _m4_cabinet_point(angle))} mm (toward world "
              f"{angle:g} deg), base drawn +-20 mm, leg +-5 mm"),
    ) for direction, angle in _COMPASS),
    *(SpatialSetting(
        key=f"leg_{direction}",
        environment={},
        variations={**_M4_BASE_TRAINING, **_M4_CABINET_TRAINING,
                    **_m4_station(_M4_LEG, _compass(M4_LEG_BEYOND_M, angle), _compass(M4_LEG_BEYOND_M, angle))},
        note=f"leg {M4_LEG_BEYOND_M * 1000:g} mm toward world {angle:g} deg, base drawn +-20 mm, cabinet +-20 mm",
    ) for direction, angle in _COMPASS),
)


def _m4_row_quotas() -> dict[str, int]:
    """GA-S: the one leg's 120 rows over its 16 settings by ``_spread``, 7 each and one more for the first eight in
    turn: the cabinet and the leg settings of e, n, w, s, a direction's two together, then the diagonals. The cabinet
    and the leg settings get 60 each."""
    compass = [_COMPASS[index][0] for index in (*range(0, 8, 2), *range(1, 8, 2))]
    order = tuple(f"{axis}_{direction}" for direction in compass for axis in ("cabinet", "leg"))
    return _spread((("lg_fx25_leveling_leg", 120),), order)


DB_M4 = BenchmarkTaskSpec(
    task="m4",
    name="DB-M4",
    title="E3 manufacturing, hand_start_leg at A_m4",
    environment=MappingProxyType({
        "type": "dexbench_manufacturing_e3_environment",
        "embodiment": _ROBOT,
        "task": "hand_start_leg",
        "anchor": "A_m4",
        "rby1_control_hz": 30,
    }),
    hold_s=0.0,
    cells=("train", "ga_s"),
    training=SpatialRange(
        note=(
            "the base drawn uniformly from the square x, y in [-20, 20] mm, no yaw "
            "(dexbench_rby1_wuji2_joint.base_pose), the leg and its holder from x, y in [-5, 5] mm (leg.station_pose), "
            "and the cabinet and its pallet from x, y in [-20, 20] mm (base_cabinet.station_pose), nothing turned: "
            "db_m4_collect.yaml's range, the draws of every attempt the 240 training demos were kept from"
        ),
        variations={**_M4_BASE_TRAINING, **_M4_LEG_TRAINING, **_M4_CABINET_TRAINING},
    ),
    beyond=SpatialRange(
        note=(
            "one station fixed off its training square and the other draws at their training ranges: the leg 15 mm out "
            "toward each of e, ne, n, nw, w, sw, s, se; or the cabinet at x +-30 mm, y +-28 mm and the four corners, "
            "the widest y its pallet keeps on the worktop (CABINET_STATION_MAX_ABS_XY_M). Every GA-S scene lies "
            "outside the training box on the fixed station, by a narrow step: the cabinet 10 mm past its training "
            "square in x and 8 mm in y (its worktop's limit), the leg 10 mm on an axis and 8 mm past the square's "
            "corner on a diagonal; the cabinet's x and the leg stay this close by choice (their domains allow 250 and "
            "80 mm), so a small Train/GA-S gap is expected"
        ),
        settings=M4_BEYOND_SETTINGS,
    ),
    row_quotas={"ga_s": _m4_row_quotas()},
    durations=DemoDurations.from_lengths_file("db_m4.json", revision="f8e4ae7dd3d8bea343041b5c8b8548e5bfc7f491"),
    sensitivity=Sensitivity(
        object_key="lg_fx25_leveling_leg",
        factors={
            "dexbench_tube_light.intensity": "x0.25-1.5 of the tube's 60000 (DexBenchInteriorLightBase); the room's authored lights stay on",
            "dexbench_tube_light.color_temperature": "2700-6500 K",
            **_CAMERA_FACTORS,
        },
        variations={
            "dexbench_tube_light.intensity.enabled": True,
            "dexbench_tube_light.color_temperature.enabled": True,
            "dexbench_tube_light.color_temperature.sampler_cfg.low": [INDOOR_COLOR_TEMPERATURE_K[0]],
            "dexbench_tube_light.color_temperature.sampler_cfg.high": [INDOOR_COLOR_TEMPERATURE_K[1]],
            **_camera_variations(),
            **_M4_BASE_TRAINING, **_M4_LEG_TRAINING, **_M4_CABINET_TRAINING,
        },
        covers={"light_intensity": ("dexbench_tube_light.intensity",),
                "color_temperature": ("dexbench_tube_light.color_temperature",),
                **_CAMERA_COVERS},
        unavailable={**_UNAVAILABLE, "distractors": "the scene holds the leg, its holder and the cabinet on its pallet "
                                                    "only, each part of the task"},
    ),
    notes=("The object axis has one leg and no held-out one: Train and GA-S only, 240 episodes per model.",),
)


M3_TRAINING_M = 0.01
_M3_DRAWS = {
    "power_driver.pickup_position.enabled": True,
    "power_driver.pickup_position.sampler_cfg.low": [-M3_TRAINING_M, -M3_TRAINING_M, 0.0],
    "power_driver.pickup_position.sampler_cfg.high": [M3_TRAINING_M, M3_TRAINING_M, 0.0],
    "center_hinge_fixture.position.enabled": True,
    "center_hinge_fixture.position.sampler_cfg.low": [-M3_TRAINING_M, -M3_TRAINING_M],
    "center_hinge_fixture.position.sampler_cfg.high": [M3_TRAINING_M, M3_TRAINING_M],
}
M3_TRAIN_BUILDS = 8
_M3_TRAIN_BUILDS = tuple(SpatialSetting(key=f"build{index}", environment={},
                                        note="one build: one station draw, the base and the driver drawn per reset")
                         for index in range(M3_TRAIN_BUILDS))
M3_BASE_BEYOND_M = 0.03
_M3_BEYOND_DIRECTIONS = _COMPASS


def _m3_base_setting(direction: str, angle_deg: float) -> SpatialSetting:
    """The base fixed ``M3_BASE_BEYOND_M`` from A_m3 toward world ``angle_deg`` (base_offset_x/y, re-aimed at the
    station), the driver and the station at their training draws."""
    dx, dy = _compass(M3_BASE_BEYOND_M, angle_deg)
    return SpatialSetting(
        key=f"base_{direction}",
        environment={"base_offset_x": dx, "base_offset_y": dy},
        variations=_M3_DRAWS,
        note=f"base fixed {M3_BASE_BEYOND_M * 1000:g} mm toward world {angle_deg:g} deg, re-aimed; driver and station drawn +-10 mm",
    )


DB_M3 = BenchmarkTaskSpec(
    task="m3",
    name="DB-M3",
    title="E3 manufacturing, power_driver_fastening at A_m3",
    environment=MappingProxyType({
        "type": "dexbench_manufacturing_e3_environment",
        "embodiment": _ROBOT,
        "task": "power_driver_fastening",
        "anchor": "A_m3",
        "rby1_control_hz": 30,
    }),
    hold_s=0.0,
    cells=("train", "ga_s"),
    training=SpatialRange(
        note=(
            "the base drawn uniformly from the square x, y in [-10, 10] mm, no yaw "
            "(dexbench_rby1_wuji2_joint.base_pose), the driver on its workbench from x, y in [-10, 10] mm "
            "(power_driver.pickup_position) at every reset, and the station from x, y in [-10, 10] mm "
            "(center_hinge_fixture.position) once per build, nothing turned: db_m3_collect.yaml's range, the one the "
            "240 training demos were recorded at, each at its limit in the M3 staging. "
            f"{M3_TRAIN_BUILDS} Runs of {120 // M3_TRAIN_BUILDS} rows, each its own build, so the cell holds "
            f"{M3_TRAIN_BUILDS} station positions"
        ),
        environment={"base_jitter_radius": M3_TRAINING_M},
        variations=_M3_DRAWS,
        settings=_M3_TRAIN_BUILDS,
    ),
    beyond=SpatialRange(
        note=(
            f"the base fixed {M3_BASE_BEYOND_M * 1000:g} mm from A_m3 toward each of "
            f"{', '.join(direction for direction, _ in _M3_BEYOND_DIRECTIONS)} (base_offset_x/y, yaw re-aimed at the "
            "station), the driver and the station at their training draws; each Run is one build, one station position. "
            f"Only the base moves out: {(M3_BASE_BEYOND_M - M3_TRAINING_M) * 1000:g} mm past its training square on an "
            f"axis, and on a diagonal {(M3_BASE_BEYOND_M / math.sqrt(2) - M3_TRAINING_M) * 1000:.0f} mm in each axis "
            f"({(M3_BASE_BEYOND_M - M3_TRAINING_M * math.sqrt(2)) * 1000:.0f} mm past the square's corner). The "
            "driver and the station are at their staging limits at +-10 mm (the driver's PlanarPoseDomain, "
            "M3StationPositionVariation), so this margin is what the scene allows"
        ),
        settings=tuple(_m3_base_setting(direction, angle) for direction, angle in _M3_BEYOND_DIRECTIONS),
    ),
    durations=DemoDurations.from_lengths_file("db_m3.json", revision="afbe89b7bdfaabc7b992ed15e5d63cd070088dd5"),
    sensitivity=Sensitivity(
        legacy_unlabelled_paths=True,
        object_key="center_hinge_fixture",
        factors={
            "dexbench_tube_light.intensity": "x0.25-1.5 of the tube's 60000 (DexBenchInteriorLightBase); the room's authored lights and the station's 700 fill stay on",
            "dexbench_tube_light.color_temperature": "2700-6500 K",
            **_CAMERA_FACTORS,
        },
        variations={
            "dexbench_tube_light.intensity.enabled": True,
            "dexbench_tube_light.color_temperature.enabled": True,
            "dexbench_tube_light.color_temperature.sampler_cfg.low": [INDOOR_COLOR_TEMPERATURE_K[0]],
            "dexbench_tube_light.color_temperature.sampler_cfg.high": [INDOOR_COLOR_TEMPERATURE_K[1]],
            **_camera_variations(),
            **_M3_DRAWS,
        },
        covers={"light_intensity": ("dexbench_tube_light.intensity",),
                "color_temperature": ("dexbench_tube_light.color_temperature",),
                **_CAMERA_COVERS},
        unavailable={**_UNAVAILABLE, "distractors": "the driver is the one movable object, and the task's"},
        environment={"base_jitter_radius": M3_TRAINING_M},
    ),
    notes=("The object axis has one station and no held-out one: Train and GA-S only, 240 episodes per model.",),
)


M2_PLUG_CENTER_M = (-0.008, 0.008)
M2_TRAIN_RADIUS_M = 0.004
M2_GA_S_RADIUS_M = (0.004, 0.006)
M2_TRAIN_YAW_DEG = (-10.0, 20.0)
M2_GA_S_YAW_DEG = (25.0, 30.0)
M2_TRAIN_BASE = (0.010, 1.0)
M2_GA_S_BASE = (0.0125, 1.25)


def _m2_draws(radius_m: float, yaw_deg: tuple[float, float], base: tuple[float, float]) -> dict[str, Any]:
    """The tape spot's box about the disk (x, y) and its yaw interval, and the base's box: the bounds the eval files
    state. The frozen rows draw inside them by area (``spatial_conditions``); Arena's uniform sampler draws boxes."""
    cx, cy = M2_PLUG_CENTER_M
    yaw = tuple(round(math.radians(value), 7) + 0.0 for value in yaw_deg)
    return {
        "plug.tape_spot.enabled": True,
        "plug.tape_spot.sampler_cfg.low": [round(cx - radius_m, 7), round(cy - radius_m, 7), yaw[0]],
        "plug.tape_spot.sampler_cfg.high": [round(cx + radius_m, 7), round(cy + radius_m, 7), yaw[1]],
        **_base_pose(-base[1], base[1], base[0]),
    }


def _m2_cable_spot(yaw_deg: tuple[float, float]) -> dict[str, Any]:
    """Where a range's builds author the anchored cable (``m2_wire_rest_spot``): the disk's centre turned to the middle
    of the range's yaw. Each reset sets the cable to the shape a build at its own draw would author
    (``m2_wire_reset_shape``); the first link alone stays as built, so the closer the build, the closer the shape.
    The task's environment holds the Train range's (its cells, the sweep); GA-S sets its own."""
    return {"m2_wire_rest_spot": [*M2_PLUG_CENTER_M, round(math.radians(sum(yaw_deg) / 2), 7)]}


DB_M2 = BenchmarkTaskSpec(
    task="m2",
    name="DB-M2",
    title="E3 manufacturing, insert_connector at A_m2",
    environment=MappingProxyType({
        "type": "dexbench_manufacturing_e3_m2_environment",
        "embodiment": _ROBOT,
        "task": "insert_connector",
        "anchor": "A_m2",
        "rby1_control_hz": 30,
        "arm_mode": "dual_arm",
        "m2_plug_condim": 3,
        "m2_fingertip_condim": 4,
        "m2_wire_collision": "panel",
        **_m2_cable_spot(M2_TRAIN_YAW_DEG),
    }),
    hold_s=0.0,
    cells=("train", "ga_s"),
    training=SpatialRange(
        note=(
            f"the plug's tape spot uniform by area in a {M2_TRAIN_RADIUS_M * 1000:g} mm disk about "
            f"({M2_PLUG_CENTER_M[0] * 1000:g}, {M2_PLUG_CENTER_M[1] * 1000:+g}) mm, turned {M2_TRAIN_YAW_DEG[0]:g} to "
            f"{M2_TRAIN_YAW_DEG[1]:+g} deg (plug.tape_spot), and the base x, y in [-{M2_TRAIN_BASE[0] * 1000:g}, "
            f"{M2_TRAIN_BASE[0] * 1000:g}] mm, turned +-{M2_TRAIN_BASE[1]:g} deg: the profile the 240 training demos "
            "were collected on"
        ),
        variations=_m2_draws(M2_TRAIN_RADIUS_M, M2_TRAIN_YAW_DEG, M2_TRAIN_BASE),
    ),
    beyond=SpatialRange(
        note=(
            f"every draw out together: the tape spot uniform by area in the {M2_GA_S_RADIUS_M[0] * 1000:g}-"
            f"{M2_GA_S_RADIUS_M[1] * 1000:g} mm ring about the same centre, turned {M2_GA_S_YAW_DEG[0]:+g} to "
            f"{M2_GA_S_YAW_DEG[1]:+g} deg, past a held-out +{M2_TRAIN_YAW_DEG[1]:g} to {M2_GA_S_YAW_DEG[0]:+g} deg gap, "
            f"and the base in the wider box +-{M2_GA_S_BASE[0] * 1000:g} mm, +-{M2_GA_S_BASE[1]:g} deg (its overlap "
            "with the training box included). The ring starts where the disk ends, so the two share only that circle, "
            "and no GA-S draw lies in the training range. One-sided: the S2 tape domain stops yaw at 30 deg either "
            "way"
        ),
        environment=_m2_cable_spot(M2_GA_S_YAW_DEG),
        variations=_m2_draws(M2_GA_S_RADIUS_M[1], M2_GA_S_YAW_DEG, M2_GA_S_BASE),
    ),
    frozen=FrozenRows(
        cells=("train", "ga_s"),
        command="scripts/tools/write_db_m2_eval_conditions.py --output_dir outputs/db_conditions",
        note=("each cell's 120 draws stratified over the six coordinates at the Run's seed (spatial_conditions, "
              "outcome-blind), as the training set's were at its own placement seeds"),
        invalid_starts=(
            "checked before freezing (check_db_m2_starts.py): every row replayed one build per Run, as the cells do, "
            "for 10 s under the hold-pose policy, and all 240 are kept; a start that still fails counts as a failure "
            "for every model alike"
        ),
    ),
    durations=DemoDurations.from_lengths_file("db_m2.json", revision="f31170d9ae568b5ef4b3f028943d74df76a609ad"),
    sensitivity=Sensitivity(
        object_key="S2",
        factors={
            "dexbench_tube_light.intensity": "x0.25-1.5 of the tube's 60000 (DexBenchInteriorLightBase); the room's authored lights stay on",
            "dexbench_tube_light.color_temperature": "2700-6500 K",
            **_CAMERA_FACTORS,
        },
        variations={
            "dexbench_tube_light.intensity.enabled": True,
            "dexbench_tube_light.color_temperature.enabled": True,
            "dexbench_tube_light.color_temperature.sampler_cfg.low": [INDOOR_COLOR_TEMPERATURE_K[0]],
            "dexbench_tube_light.color_temperature.sampler_cfg.high": [INDOOR_COLOR_TEMPERATURE_K[1]],
            **_camera_variations(),
            **_m2_draws(M2_TRAIN_RADIUS_M / math.sqrt(2), M2_TRAIN_YAW_DEG, M2_TRAIN_BASE),
        },
        covers={"light_intensity": ("dexbench_tube_light.intensity",),
                "color_temperature": ("dexbench_tube_light.color_temperature",),
                **_CAMERA_COVERS},
        unavailable={**_UNAVAILABLE, "distractors": "the scene holds the liner, its mated neighbour and the task's plug "
                                                    "only, each part of the task"},
    ),
    notes=("The object axis has one connector in the benchmark, S2, the training set's; S1 is unfit (no tape spot, "
           "demos or collector; refused), so no GA-O: "
           "Train and GA-S only, 240 episodes per model.",),
)



def _h2_base(xy_m: float, yaw_deg: float) -> dict[str, Any]:
    """The base's draw about the 1.90 m pour base (A_h2 + 0.15 m in y): x, y uniform in +-xy_m, yaw in +-yaw_deg."""
    yaw = round(math.radians(yaw_deg), 7)
    return {
        f"{_ROBOT}.base_pose.enabled": True,
        f"{_ROBOT}.base_pose.sampler_cfg.low": [-xy_m, -xy_m, -yaw],
        f"{_ROBOT}.base_pose.sampler_cfg.high": [xy_m, xy_m, yaw],
    }


def _h2_props(jug_xy_m: float, jug_yaw_deg: float, goblet_xy_m: float) -> dict[str, Any]:
    """The jug's and the target goblet's draws (``<jug>.pour_pose``, ``pour_goblet.pour_pose``), by config field."""
    return {"pour_jug_jitter_xy": jug_xy_m, "pour_jug_jitter_yaw_deg": jug_yaw_deg, "pour_goblet_jitter_xy": goblet_xy_m}


_H2_COMPASS = (("e", 0), ("n", 90), ("w", 180), ("s", 270), ("ne", 45), ("nw", 135), ("sw", 225), ("se", 315))
_H2_JUG_BEYOND_M = {"e": 0.04, "ne": 0.04, "n": 0.04, "nw": 0.04, "w": 0.03, "sw": 0.04, "s": 0.03, "se": 0.04}
"""Toward w the IDEALISK spout meets a goblet drawn toward it at 4 cm; toward s, a grasp knocks the jug over."""
_H2_JUG_YAW_BEYOND_DEG = (-15.0, 15.0)
_H2_BASE_BEYOND_M = 0.02
_H2_BASE_YAW_BEYOND_DEG = (7.0, -7.0)
_H2_GOBLET_BEYOND_M = 0.02
_H2_PROPS_TRAINING = _h2_props(0.02, 10.0, 0.01)
_H2_BASE_TRAINING = _h2_base(0.01, 5.0)


def _h2_base_at(dx: float, dy: float, yaw_low_deg: float, yaw_high_deg: float, xy_m: float = 0.0) -> dict[str, Any]:
    """The base's draw centred on (dx, dy): x, y in +-xy_m about it, yaw in [low, high] deg. A fixed point is part of
    the recorded draw (no base_offset_x, which would also turn the base toward its anchor's target)."""
    low, high = (round(math.radians(yaw), 7) + 0.0 for yaw in (yaw_low_deg, yaw_high_deg))
    return {
        f"{_ROBOT}.base_pose.enabled": True,
        f"{_ROBOT}.base_pose.sampler_cfg.low": [round(dx - xy_m, 4) + 0.0, round(dy - xy_m, 4) + 0.0, low],
        f"{_ROBOT}.base_pose.sampler_cfg.high": [round(dx + xy_m, 4) + 0.0, round(dy + xy_m, 4) + 0.0, high],
    }


def _h2_drawn_but(field_name: str) -> dict[str, Any]:
    """The props' training draws without the one a fixed point holds: that field falls to its default 0, no draw."""
    return {key: value for key, value in _H2_PROPS_TRAINING.items() if key != field_name}


def _h2_signed_key(prefix: str, value: float) -> str:
    """``jug_yaw_m15``, ``base_yaw_p7``: a fixed turn's setting key (no dot or minus in a Run name)."""
    return f"{prefix}_{'p' if value > 0 else 'm'}{abs(value):g}".replace(".", "p")


def _h2_beyond_settings() -> tuple[SpatialSetting, ...]:
    """The 28 GA-S fixed points: the jug 8 directions and 2 turns, the base 8 directions and 2 turns, the goblet 8."""
    settings = []
    for direction, angle in _H2_COMPASS:
        dx, dy = _compass(_H2_JUG_BEYOND_M[direction], angle)
        settings.append(SpatialSetting(
            key=f"jug_{direction}",
            environment={**_h2_drawn_but("pour_jug_jitter_xy"), "pour_jug_offset_x": dx, "pour_jug_offset_y": dy},
            variations=_H2_BASE_TRAINING,
            note=f"jug {_H2_JUG_BEYOND_M[direction] * 100:g} cm toward world {angle} deg, turned +-10 deg; goblet and base drawn",
        ))
    for yaw in _H2_JUG_YAW_BEYOND_DEG:
        settings.append(SpatialSetting(
            key=_h2_signed_key("jug_yaw", yaw),
            environment={**_h2_drawn_but("pour_jug_jitter_yaw_deg"), "pour_jug_offset_yaw_deg": yaw},
            variations=_H2_BASE_TRAINING,
            note=f"jug turned {yaw:+g} deg, x, y drawn +-2 cm; goblet and base drawn",
        ))
    for direction, angle in _H2_COMPASS:
        dx, dy = _compass(_H2_BASE_BEYOND_M, angle)
        settings.append(SpatialSetting(
            key=f"base_{direction}",
            environment=_H2_PROPS_TRAINING,
            variations=_h2_base_at(dx, dy, -5.0, 5.0),
            note=f"base 2 cm toward world {angle} deg, turned +-5 deg; jug and goblet drawn",
        ))
    for yaw in _H2_BASE_YAW_BEYOND_DEG:
        settings.append(SpatialSetting(
            key=_h2_signed_key("base_yaw", yaw),
            environment=_H2_PROPS_TRAINING,
            variations=_h2_base_at(0.0, 0.0, yaw, yaw, xy_m=0.01),
            note=f"base turned {yaw:+g} deg, x, y drawn +-1 cm; jug and goblet drawn",
        ))
    for direction, angle in _H2_COMPASS:
        dx, dy = _compass(_H2_GOBLET_BEYOND_M, angle)
        settings.append(SpatialSetting(
            key=f"goblet_{direction}",
            environment={**_h2_drawn_but("pour_goblet_jitter_xy"), "pour_goblet_offset_x": dx, "pour_goblet_offset_y": dy},
            variations=_H2_BASE_TRAINING,
            note=f"goblet 2 cm toward world {angle} deg; jug and base drawn",
        ))
    return tuple(settings)


H2_BEYOND_SETTINGS = _h2_beyond_settings()
_H2_AXES = {
    "jug": tuple(f"jug_{d}" for d, _ in _H2_COMPASS),
    "jug_yaw": tuple(_h2_signed_key("jug_yaw", y) for y in _H2_JUG_YAW_BEYOND_DEG),
    "base": tuple(f"base_{d}" for d, _ in _H2_COMPASS),
    "base_yaw": tuple(_h2_signed_key("base_yaw", y) for y in _H2_BASE_YAW_BEYOND_DEG),
    "goblet": tuple(f"goblet_{d}" for d, _ in _H2_COMPASS),
}
_H2_JUGS = {"ga_s": ("idealisk", "d00527_x15"), "ga_os": ("steel_pitcher_large", "steel_pitcher_footed")}
"""The cells' jugs in the object axis's order (e4_hotel_fb_pour.POUR_JUG_OBJECT_AXIS): the training jugs, the held-out."""


def _h2_row_quotas() -> dict[str, dict[str, int]]:
    """GA-S and GA-OS: 2 jugs x 28 settings = 56 Runs for 120 rows. Each jug takes 60, each of the five axes 12 of
    them. A two-point axis (the turns) gives each point 6 per jug; an eight-direction axis gives four directions 2 and
    four 1, the other jug the reverse, so every direction gets 3 of the cell's rows and every turn 12."""
    quotas: dict[str, dict[str, int]] = {}
    for cell, jugs in _H2_JUGS.items():
        rows: dict[str, int] = {}
        for index, jug in enumerate(jugs):
            for keys in _H2_AXES.values():
                if len(keys) == 2:
                    rows.update({f"{jug}_{key}": 6 for key in keys})
                else:
                    rows.update({f"{jug}_{key}": 2 if (position < 4) == (index == 0) else 1 for position, key in enumerate(keys)})
            if sum(v for k, v in rows.items() if k.startswith(f"{jug}_")) != 60:
                raise ValueError(f"DB-H2 {cell} {jug}: quotas do not sum to 60")
        quotas[cell] = rows
    return quotas


DB_H2 = BenchmarkTaskSpec(
    task="h2",
    name="DB-H2",
    title="E4 banquet hall, lift_and_pour at A_h2",
    environment=MappingProxyType({
        "type": "dexbench_hotel_fb_e4_pour_environment",
        "embodiment": _ROBOT,
        "task": "lift_and_pour",
        "anchor": "A_h2",
        "rby1_control_hz": 30,
        "idealisk_pour": True,
        "base_offset_y": 0.15,
        "pour_station_offset_y": 0.05,
        "start_pose": "teleop_ready",
        "newton_nconmax": 16384,
        "newton_njmax": 65536,
        "newton_num_substeps": 8,
    }),
    training=SpatialRange(
        note=(
            "jug x, y +-2 cm and yaw +-10 deg, target goblet x, y +-1 cm, base x, y +-1 cm and yaw +-5 deg, each uniform "
            "and independent"
        ),
        environment=_H2_PROPS_TRAINING,
        variations=_H2_BASE_TRAINING,
    ),
    beyond=SpatialRange(
        note=(
            "one axis fixed off its training range and every other at its training draw: the jug 4 cm toward world "
            "e, ne, n, nw, sw and se and 3 cm toward w and s (at 4 cm the IDEALISK spout meets a goblet drawn toward w, "
            "and a grasp knocks the jug over toward the robot); or the jug turned -15 or +15 deg; or the base "
            "2 cm toward each of the eight, turned at its draw; or the base turned +7 or -7 deg; or the goblet 2 cm "
            "toward each of the eight. Every point lies outside the training box on its axis and is checked at the "
            "1.90 m base with every other axis at its training extremes, all four jugs: the placement on the trolley, "
            "the base >= 17 mm from the trolley's columns and the curtain, the wrist poses within the arm's reach, and "
            "the start at rest on Newton with no hand touching a prop"
        ),
        settings=H2_BEYOND_SETTINGS,
    ),
    durations=DemoDurations.from_lengths_file("db_h2.json", revision="016681a0e1ce429f9605af3f55c96120d15e958d"),
    hold_s=0.0,
    cell_seed_bases={"train": 9000, "ga_o": 9100, "ga_s": 9200, "ga_os": 9300},
    row_quotas=_h2_row_quotas(),
    sensitivity=Sensitivity(
        object_key="idealisk",
        factors={
            "dexbench_hotel_fb_e4_lighting.intensity_scale": "x0.25-1.5 of the delivered lights (LIGHT_INTENSITY_SCALE_RANGE)",
            "dexbench_hotel_fb_e4_lighting.color_temperature": "3000-9000 K (LIGHT_COLOR_TEMPERATURE_RANGE_K)",
            **_CAMERA_FACTORS,
        },
        variations={
            "dexbench_hotel_fb_e4_lighting.intensity_scale.enabled": True,
            "dexbench_hotel_fb_e4_lighting.color_temperature.enabled": True,
            **_camera_variations(),
            **_h2_base(0.01, 5.0),
        },
        environment=_h2_props(0.02, 10.0, 0.01),
        covers={"light_intensity": ("dexbench_hotel_fb_e4_lighting.intensity_scale",),
                "color_temperature": ("dexbench_hotel_fb_e4_lighting.color_temperature",),
                **_CAMERA_COVERS},
        unavailable={**_UNAVAILABLE, "distractors": "the station holds the jug and the target goblet only"},
    ),
)




M5_TRAINING_COMPARTMENTS = (0, 3)
"""The far-right and the near-left compartment, the diagonal the 240 demos were collected on."""
M5_BEYOND_COMPARTMENTS = (1, 2)
"""The near-right and the far-left, the other diagonal."""
M5_PILE_SEED_QUANTILES = (0.0, 0.6)
M5_PILES = "outputs/db_conditions/db_m5_piles"
"""The dbclutter pile set every Run draws its pile from: RLWRLD/DexBench-eval db_m5/clutter_all_x40 (e72dc4c4),
the set the 240 demos were collected on, placed with the rows (eval/db_conditions/db_m5_piles) so
``download_datasets.sh --eval-conditions`` installs it. Each row records the set's hash
(``parts_bin_clutter_set_sha256``) and the build refuses a row of another set."""


def _m5_draws(compartments: tuple[int, ...]) -> dict[str, Any]:
    return {"kit_box_empty_compartments": list(compartments), "parts_bin_seed_quantiles": list(M5_PILE_SEED_QUANTILES)}


DB_M5 = BenchmarkTaskSpec(
    task="m5",
    name="DB-M5",
    title="E3 manufacturing, place_part_in_kit_box at A_m5",
    environment=MappingProxyType({
        "type": "dexbench_manufacturing_m5_moving_conveyor_kitting_environment",
        "embodiment": _ROBOT,
        "task": "place_part_in_kit_box",
        "anchor": "A_m5",
        "rby1_control_hz": 30,
        "arm_mode": "left",
        "parts_bin_clutter_dir": M5_PILES,
    }),
    hold_s=0.0,
    training=SpatialRange(
        note=(
            "the kit box's empty compartment drawn from the far-right and the near-left (belt_kit_box.empty_compartment"
            ", kit_box_empty_compartments [0, 3]) and the parts-bin pile from the first 24 of the part's 40 dbclutter "
            "seeds by rank (dexbench_assembly_line_e3.parts_bin_clutter_seed, parts_bin_seed_quantiles [0, 0.6]): "
            "db_m5_collect.yaml's range, every start of which one of the 240 training demos was collected at"
        ),
        environment=_m5_draws(M5_TRAINING_COMPARTMENTS),
    ),
    beyond=SpatialRange(
        note=(
            "the other diagonal: the near-right or the far-left compartment empty (kit_box_empty_compartments [1, 2]), "
            "no demo of which exists, at the same piles, so a GA-S row differs from a Train start in the compartment "
            "alone; each range has one near and one far, one left and one right compartment"
        ),
        environment=_m5_draws(M5_BEYOND_COMPARTMENTS),
    ),
    frozen=FrozenRows(
        cells=CELLS,
        command=f"scripts/tools/write_db_m5_eval_conditions.py --parts_bin_clutter_dir {M5_PILES} --output_dir outputs/db_conditions",
        note=(
            "each part of the cell's split an even share of the 120 rows (Train and GA-S 24, GA-O and GA-OS 12), "
            "spread over the range's two compartments and the 24 piles by a fixed rule (row i: compartment i mod 2, "
            "pile of rank floor(24 i / n)), outcome-blind; written, not recorded, because a condition recording's settle "
            "check fails on the kit box the belt carries"
        ),
        invalid_starts=(
            "checked before freezing (check_db_m5_starts.py): every row replayed one build per Run, as the cells do, "
            "for 10 s under the hold-pose policy, with every object at rest at the cap but the kit box and its parts, "
            "which the belt carries; all 480 are kept; a start that still fails counts as a failure for every model "
            "alike"
        ),
    ),
    durations=DemoDurations.from_lengths_file("db_m5.json", revision="9c058fc29d368959c1c4fb922101295585967bf2"),
    sensitivity=Sensitivity(
        object_key="bolt_m24",
        factors={
            "dexbench_tube_light.intensity": "x0.25-1.5 of the tube's 60000 (DexBenchInteriorLightBase); the room's authored lights stay on",
            "dexbench_tube_light.color_temperature": "2700-6500 K",
            **_CAMERA_FACTORS,
        },
        variations={
            "dexbench_tube_light.intensity.enabled": True,
            "dexbench_tube_light.color_temperature.enabled": True,
            "dexbench_tube_light.color_temperature.sampler_cfg.low": [INDOOR_COLOR_TEMPERATURE_K[0]],
            "dexbench_tube_light.color_temperature.sampler_cfg.high": [INDOOR_COLOR_TEMPERATURE_K[1]],
            **_camera_variations(),
        },
        environment=_m5_draws(M5_TRAINING_COMPARTMENTS),
        covers={"light_intensity": ("dexbench_tube_light.intensity",),
                "color_temperature": ("dexbench_tube_light.color_temperature",),
                **_CAMERA_COVERS},
        unavailable={**_UNAVAILABLE, "distractors": "the scene holds the parts-bin pile, the kit box and the three parts "
                                                    "loaded in it, each part of the task"},
    ),
    collection_settings={
        "parts_bin_clutter_dir": (
            "the collection read the same pile set (db_m5/clutter_all_x40) from where its tarballs unpack, "
            "outputs/clutter/all_x40; the cells read it from beside their rows. Every row and start records the set's "
            "hash, so both read one set"
        ),
    },
    notes=(
        "GA-O and GA-OS hold out ten parts, each the shape of a training part at another size (db_m5_layout's split); "
        "four cells, 480 episodes per model.",
    ),
)



_L4_TRAINING_SET = "b80e00f1eaa2011fd2bf9d2a7d9d431f315b3f17"
DB_L4 = BenchmarkTaskSpec(
    task="l4",
    name="DB-L4",
    title="E2 logistics, deliver_parcels_barcode_up at A_L4 on HB2 and FB1",
    environment=MappingProxyType({
        "type": "dexbench_logistics_e2_l4_environment",
        "embodiment": _ROBOT,
        "task": "deliver_parcels_barcode_up",
        "anchor": "A_L4",
        "rby1_control_hz": 30,
        "belt_speed": 0.45,
        "end_on_parcel_failure": True,
    }),
    hold_s=0.0,
    cells=("train", "ga_s", "ga_p", "ga_sp"),
    training=SpatialRange(
        note=(
            "each parcel put down on HB2 at one of its in-domain rotations, drawn per slot at every reset "
            "(L4RotationVariation): barcode up, down or to a side, square to the belt (16 for a box, 8 for a bag, "
            "e2_logistics_l4_layout.rotation_labels), the rotations the training set's sequences use, each ten "
            "times per parcel"
        ),
        environment={"l4_rotation_draw": "id"},
    ),
    beyond=SpatialRange(
        note=(
            "each parcel put down at a held-out rotation: an in-domain one turned 45 deg about up, so its edges run "
            "diagonal to the belt (16 for a box, 8 for a bag); no training sequence puts a parcel down so. A box left "
            "alone squares up against the plate's edge and ends off to one side; a bag mostly keeps its turn. So GA-S "
            "measures the whole episode after a held-out put-down, not the held-out rotation alone: the training demos "
            "first reach a box at a median 1.6 s (p90 1.9 s), and a policy that reaches as they do meets the turned "
            "box, a later one a box already squared up off centre or tipped over"
        ),
        environment={"l4_rotation_draw": "ood"},
    ),
    physical_training=PhysicalRange(
        note=(
            "each parcel at its nominal mass (boxes 1.0 kg, bags 0.08-0.30 kg as weighed) times an in-domain "
            "multiplier, x0.5, x0.71, x1, x1.41 or x2, drawn per slot at every reset (L4MassVariation), the "
            "multipliers the training set's sequences use"
        ),
        environment={"l4_mass_draw": "id"},
    ),
    physical_beyond=PhysicalRange(
        note=(
            "each parcel at a held-out multiplier, x0.25, x0.35, x2.83 or x4: a step of sqrt(2) past each end of "
            "the training range on a log scale"
        ),
        environment={"l4_mass_draw": "ood"},
    ),
    frozen=FrozenRows(
        cells=("train", "ga_s", "ga_p", "ga_sp"),
        command="scripts/tools/write_db_l4_eval_conditions.py --output_dir outputs/db_conditions",
        note=("each evaluation sequence's 10 rows at the Run's seed: every slot's rotation and mass multiplier from "
              "the cell's domains, stratified over the domain, outcome-blind"),
        invalid_starts=(
            "a row's start state, its first parcel on HB2 (the rest are put down as the episode goes), is landed and "
            "carried by HB2 1 s after the put-down, as the sweep's settle check judges it. What the parcel does "
            "after (it stops at the plate's edge, is pushed onto the plate, squares up, tips over) is part of the "
            "task, and the relay judges delivery as usual: a parcel fails only by falling, by the timeout or by "
            "reaching FB1 barcode not up. Every candidate rotation is checked on its own, nobody touching it "
            "(smoke --check_put_downs, at a model run's 60 Hz, each at x0.25, x1 and x4): all 240 are valid starts at "
            "every mass, so none is dropped. A start that still fails counts as a failure for every model alike"
        ),
    ),
    durations=DemoDurations.from_lengths_file("db_l4.json", revision=_L4_TRAINING_SET),
    condition_episode_s=1.0,
    sensitivity=Sensitivity(
        object_key="eval_00",
        factors={
            "high_bay_light_02.intensity": "x0.25-1.5 of the high-bay's 6302 (DexBenchInteriorLightBase), the high-bay nearest the station, 0.67 m off A_L4",
            "high_bay_light_02.color_temperature": "2700-6500 K",
            "high_bay_light_05.intensity": "x0.25-1.5, the next nearest, 1.37 m off",
            "high_bay_light_05.color_temperature": "2700-6500 K",
            **_CAMERA_FACTORS,
        },
        variations={
            **{f"high_bay_light_0{n}.{factor}.enabled": True for n in (2, 5) for factor in ("intensity", "color_temperature")},
            **{f"high_bay_light_0{n}.color_temperature.sampler_cfg.low": [INDOOR_COLOR_TEMPERATURE_K[0]] for n in (2, 5)},
            **{f"high_bay_light_0{n}.color_temperature.sampler_cfg.high": [INDOOR_COLOR_TEMPERATURE_K[1]] for n in (2, 5)},
            **_camera_variations(),
        },
        covers={"light_intensity": ("high_bay_light_02.intensity", "high_bay_light_05.intensity"),
                "color_temperature": ("high_bay_light_02.color_temperature", "high_bay_light_05.color_temperature"),
                **_CAMERA_COVERS},
        unavailable={**_UNAVAILABLE, "distractors": "the sequence's other parcels wait parked out of view; one put "
                                                    "down beside the task's would be a second task item, not a "
                                                    "distraction"},
        environment={"l4_rotation_draw": "id", "l4_mass_draw": "id"},
        notes=("DB-L1's sweep: the same two high-bays, which are also the two nearest A_L4 (3.191, 1.640)",),
    ),
    notes=("The object axis is the 12 evaluation sequences (L4_SEQUENCE_AXIS), all of training parcels, so no GA-O: "
           "Train, GA-S, GA-P and GA-SP, 480 episodes per model. Scored on success (all five parcels delivered); "
           "the task reports parcels_delivered beside it, never scored.",),
)


DB_H3 = BenchmarkTaskSpec(
    task="h3",
    name="DB-H3",
    title="E4 banquet hall, extract one utensil from the held bundle",
    environment=MappingProxyType({
        "type": "dexbench_hotel_fb_e4_h3_environment",
        "embodiment": _ROBOT,
        "task": "extract_utensil",
        "anchor": "A_h3",
        "rby1_control_hz": 30,
        "wrist_cameras": True,
        "camera_scale": 0.5,
        "head_yaw_deg": 5.0,
        "head_pitch_deg": 45.0,
        "h3_drive_profile": "collection_v1",
    }),
    cells=("train", "ga_os"),
    training=SpatialRange(
        note="saved in-hand clutter grasps from the collection range, excluding every physical start in the 240 selected training demos",
    ),
    beyond=SpatialRange(
        note="held-out utensil trios with unseen geometry-dependent in-hand grasps from the same generation procedure; coupled object/layout generalization, not an independently held-out angle range",
    ),
    frozen=FrozenRows(
        cells=("train", "ga_os"),
        command="scripts/tools/write_db_h3_eval_conditions.py --inputs outputs/db_h3_inputs --output_dir outputs/db_conditions",
        note="10 distinct state-target pairs per trio/target, 120 per cell; a physical grasp may be reused for another target",
        invalid_starts="validate each saved start before freezing; a later invalid start remains a failure for every model",
    ),
    durations=DemoDurations.from_lengths_file("db_h3.json", revision="5ace70ba5aa285b2092076361d616f0745a7e157"),
    runtime_inputs=MappingProxyType({
        f"{H3_INPUTS_DIR}/manifest.json": "scripts/download_datasets.sh --eval-conditions"}),
    hold_s=0.0,
    baseline_policy=MappingProxyType({
        "type": "dexbench_arena.policy.hold_pose_policy.HoldPosePolicy", "hold": "targets",
    }),
    sensitivity=Sensitivity(
        object_key="train_01_spoon",
        factors=_CAMERA_FACTORS,
        variations=_camera_variations(),
        covers=_CAMERA_COVERS,
        unavailable={
            **_UNAVAILABLE,
            "light_intensity": "the collection disk light has a fixed 60000 intensity and no registered variation",
            "color_temperature": "the collection disk light uses fixed white RGB and no registered variation",
            "distractors": "the three utensils are task objects; no distractor variation is supplied",
        },
    ),
    notes=(
        "Training: 240 demos, 80 per target and 20 per trio/target; no target marker in the policy images.",
        "No collection dump: starts come from frozen saved-state rows.",
        "Success: target released inside the bin for 0.5 s while both other utensils remain held; no low-motion threshold.",
        "The support fingers track saved init_q commands with the collection gains, collision exclusions and isotropic-max utensil inertia.",
    ),
)

TASKS: tuple[BenchmarkTaskSpec, ...] = (DB_R1, DB_R2, DB_R3, DB_L1, DB_L2, DB_L4, DB_M1, DB_M2, DB_H1, DB_M4, DB_M3, DB_M5, DB_H3, DB_L3, DB_H2)
TASKS_BY_NAME = {spec.task: spec for spec in TASKS}
