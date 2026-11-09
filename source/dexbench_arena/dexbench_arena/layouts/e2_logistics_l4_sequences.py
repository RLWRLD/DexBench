# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""DB-L4's sequences: the parcels an episode puts down, for the benchmark's Runs and for its training set."""

from __future__ import annotations

import itertools
import json
import random
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

from dexbench_arena import REPO_ROOT
from dexbench_arena.assets.e2_logistics_cell import E2_L4_PARCELS
from dexbench_arena.layouts.e2_logistics_l4_layout import (
    L4_PARCELS_PER_EPISODE,
    MASS_MULTIPLIERS,
    rotation_labels,
)
from dexbench_arena.tasks.object_axis import ObjectAxis, TaskObject

TRAINING_SEQUENCES_PATH = REPO_ROOT / "scripts" / "evaluation" / "experiment_configs" / "db_l4_training_sequences.json"
SEQUENCE_SEED = 20261021
TRAINING_SET_SOURCE = "DB-L4 GA training set: episode i is train_<i>"
TRAINING_SET_SHA256 = "008b8329532c02f33a060a2f6bbc79b8b58ce286d6aa7b4068931e7ac18df1c7"
BOXES = tuple(parcel.name for parcel in E2_L4_PARCELS if not parcel.is_polybag)
BAGS = tuple(parcel.name for parcel in E2_L4_PARCELS if parcel.is_polybag)
EVAL_SEQUENCES_BY_BAGS = {2: 8, 1: 4}
TRAIN_SEQUENCES_BY_BAGS = {2: 160, 1: 80}
TRAIN_ROTATION_REPEATS_PER_SLOT = 2
_EVAL_BAG_SLOTS = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2), (1, 3), (2, 4), (0,), (1,), (3,), (4,))
_ATTEMPTS = 1000


@dataclass(frozen=True)
class L4Sequence:
    """One sequence: its id (``eval_00``, ``train_000``), its parcels in order, and for a training sequence the
    rotation each is put down at and the mass multiplier each weighs."""

    id: str
    objects: tuple[str, ...]
    rotations: tuple[str, ...] | None = None
    masses: tuple[float, ...] | None = None


EVAL_SEQUENCES = (
    L4Sequence("eval_00", ("box_3", "box_2", "box_1", "box_1", "polybag_4")),
    L4Sequence("eval_01", ("polybag_2", "box_1", "polybag_2", "box_5", "box_2")),
    L4Sequence("eval_02", ("polybag_3", "box_4", "box_3", "box_4", "polybag_3")),
    L4Sequence("eval_03", ("box_2", "box_5", "polybag_4", "box_1", "polybag_5")),
    L4Sequence("eval_04", ("box_4", "polybag_1", "polybag_5", "box_3", "box_5")),
    L4Sequence("eval_05", ("box_5", "box_1", "polybag_1", "polybag_4", "box_5")),
    L4Sequence("eval_06", ("box_1", "box_4", "box_5", "polybag_3", "polybag_2")),
    L4Sequence("eval_07", ("box_3", "box_5", "box_4", "polybag_5", "box_3")),
    L4Sequence("eval_08", ("polybag_1", "polybag_3", "box_4", "box_2", "box_3")),
    L4Sequence("eval_09", ("box_2", "polybag_2", "box_2", "box_5", "box_4")),
    L4Sequence("eval_10", ("box_1", "polybag_5", "box_2", "polybag_1", "box_1")),
    L4Sequence("eval_11", ("polybag_4", "box_3", "box_3", "box_2", "box_4")),
)


def _sequence_fit(task_object: TaskObject, cfg) -> dict:
    """The parcels the sequence puts down, in order."""
    del cfg
    return {"parcels": list(next(s for s in EVAL_SEQUENCES if s.id == task_object.key).objects)}


L4_SEQUENCE_AXIS = ObjectAxis(
    field="l4_sequence",
    default=EVAL_SEQUENCES[0].id,
    objects=tuple(TaskObject(key=sequence.id, asset="E2_L4_PARCELS", split="train", label="parcels", size_m=())
                  for sequence in EVAL_SEQUENCES),
    size_axes="none: each parcel has its own size (E2_L4_PARCELS)",
    fit=_sequence_fit,
)


def _fill(rng: random.Random, bag_slots: list[tuple[int, ...]], per_slot) -> list[list[str]]:
    """Sequences with bags in ``bag_slots`` (one tuple per sequence), each slot's boxes and bags drawn from
    ``per_slot(slot, boxes, bags)``, which returns that slot's box names and bag names to shuffle in."""
    rows = [[None] * L4_PARCELS_PER_EPISODE for _ in bag_slots]
    for slot in range(L4_PARCELS_PER_EPISODE):
        bag_rows = [i for i, slots in enumerate(bag_slots) if slot in slots]
        box_rows = [i for i, slots in enumerate(bag_slots) if slot not in slots]
        boxes, bags = per_slot(slot, len(box_rows), len(bag_rows))
        rng.shuffle(boxes)
        rng.shuffle(bags)
        for i, name in zip(box_rows, boxes):
            rows[i][slot] = name
        for i, name in zip(bag_rows, bags):
            rows[i][slot] = name
    return rows


def _distinct(rows, taken=frozenset()) -> bool:
    keys = [tuple(row) for row in rows]
    return len(set(keys)) == len(keys) and not set(keys) & taken


def _eval_sequences(rng: random.Random) -> list[L4Sequence]:
    def per_slot(slot, n_boxes, n_bags):
        return ([BOXES[(slot * n_boxes + k) % len(BOXES)] for k in range(n_boxes)],
                [BAGS[(slot * n_bags + k) % len(BAGS)] for k in range(n_bags)])

    for _ in range(_ATTEMPTS):
        bag_slots = list(_EVAL_BAG_SLOTS)
        rng.shuffle(bag_slots)
        rows = _fill(rng, bag_slots, per_slot)
        if _distinct(rows):
            return [L4Sequence(f"eval_{i:02d}", tuple(row)) for i, row in enumerate(rows)]
    raise RuntimeError(f"no distinct evaluation set in {_ATTEMPTS} draws")


def _training_sequences(rng: random.Random, taken: frozenset) -> list[L4Sequence]:
    pairs = list(itertools.combinations(range(L4_PARCELS_PER_EPISODE), 2))
    bag_slots = ([pair for pair in pairs for _ in range(TRAIN_SEQUENCES_BY_BAGS[2] // len(pairs))]
                 + [(slot,) for slot in range(L4_PARCELS_PER_EPISODE)
                    for _ in range(TRAIN_SEQUENCES_BY_BAGS[1] // L4_PARCELS_PER_EPISODE)])

    def per_slot(slot, n_boxes, n_bags):
        return ([name for name in BOXES for _ in range(n_boxes // len(BOXES))],
                [name for name in BAGS for _ in range(n_bags // len(BAGS))])

    for _ in range(_ATTEMPTS):
        rng.shuffle(bag_slots)
        rows = _fill(rng, bag_slots, per_slot)
        if _distinct(rows, taken):
            break
    else:
        raise RuntimeError(f"no distinct training set in {_ATTEMPTS} draws")
    rotations = [[None] * L4_PARCELS_PER_EPISODE for _ in rows]
    for slot in range(L4_PARCELS_PER_EPISODE):
        for name in BOXES + BAGS:
            cells = [i for i, row in enumerate(rows) if row[slot] == name]
            labels = [label for label in rotation_labels(name) for _ in range(TRAIN_ROTATION_REPEATS_PER_SLOT)]
            rng.shuffle(labels)
            for i, label in zip(cells, labels):
                rotations[i][slot] = label
    masses = _training_masses(rng, rows)
    return [L4Sequence(f"train_{i:03d}", tuple(row), tuple(rotation), tuple(mass))
            for i, (row, rotation, mass) in enumerate(zip(rows, rotations, masses))]


def _training_masses(rng: random.Random, rows) -> list[list[float]]:
    """Each training parcel's in-domain mass multiplier: in every (parcel kind, slot), each value its even share,
    the remainder going to values in turn from slot to slot, so every value's total over the five slots is the
    same; shuffled within the (kind, slot)."""
    values = MASS_MULTIPLIERS["id"]
    masses = [[None] * L4_PARCELS_PER_EPISODE for _ in rows]
    for name in BOXES + BAGS:
        for slot in range(L4_PARCELS_PER_EPISODE):
            cells = [i for i, row in enumerate(rows) if row[slot] == name]
            share, extra = divmod(len(cells), len(values))
            pool = [value for value in values for _ in range(share)]
            pool += [values[(slot * extra + k) % len(values)] for k in range(extra)]
            rng.shuffle(pool)
            for i, value in zip(cells, pool):
                masses[i][slot] = value
    return masses


def build_sequences(seed: int = SEQUENCE_SEED) -> dict[str, list[L4Sequence]]:
    """``{"eval": [...], "train": [...]}``, drawn from ``random.Random(seed)``: the evaluation set first, then the
    training set around it."""
    rng = random.Random(seed)
    eval_sequences = _eval_sequences(rng)
    return {"eval": eval_sequences,
            "train": _training_sequences(rng, frozenset(sequence.objects for sequence in eval_sequences))}


def _bag_counts(sequences) -> Counter:
    return Counter(sum(name in BAGS for name in sequence.objects) for sequence in sequences)


def check_sequences(sets: dict[str, list[L4Sequence]]) -> list[str]:
    """Every rule of the module docstring that ``sets`` breaks, in words; empty when it keeps them all."""
    problems = []
    eval_set, train = sets["eval"], sets["train"]
    for label, sequences, by_bags in (("eval", eval_set, EVAL_SEQUENCES_BY_BAGS), ("train", train, TRAIN_SEQUENCES_BY_BAGS)):
        if any(len(sequence.objects) != L4_PARCELS_PER_EPISODE for sequence in sequences):
            problems.append(f"{label}: a sequence does not hold {L4_PARCELS_PER_EPISODE} parcels")
        if dict(_bag_counts(sequences)) != by_bags:
            problems.append(f"{label}: sequences by bag count {dict(_bag_counts(sequences))}, not {by_bags}")
        if not _distinct([sequence.objects for sequence in sequences]):
            problems.append(f"{label}: two sequences are the same")
    if {sequence.objects for sequence in eval_set} & {sequence.objects for sequence in train}:
        problems.append("a training sequence is an evaluation one")
    totals = Counter(name for sequence in eval_set for name in sequence.objects)
    if any(totals[name] != 8 for name in BOXES) or any(totals[name] != 4 for name in BAGS):
        problems.append(f"eval: parcels in all {dict(totals)}, not 8 of each box and 4 of each bag")
    for slot in range(L4_PARCELS_PER_EPISODE):
        if sum(sequence.objects[slot] in BAGS for sequence in eval_set) != 4:
            problems.append(f"eval: slot {slot} does not hold four bags")
        counts = Counter(sequence.objects[slot] for sequence in train)
        expected = {**{name: 32 for name in BOXES}, **{name: 16 for name in BAGS}}
        if dict(counts) != expected:
            problems.append(f"train: slot {slot} holds {dict(counts)}, not each box 32 times and each bag 16")
        rotations = Counter((sequence.objects[slot], (sequence.rotations or (None,) * L4_PARCELS_PER_EPISODE)[slot])
                            for sequence in train)
        expected = {(name, label): TRAIN_ROTATION_REPEATS_PER_SLOT for name in BOXES + BAGS for label in rotation_labels(name)}
        if dict(rotations) != expected:
            wrong = sorted(set(rotations.items()) ^ set(expected.items()), key=str)[:3]
            problems.append(f"train: slot {slot} is not every (parcel, in-domain rotation) twice, e.g. {wrong}")
    values = MASS_MULTIPLIERS["id"]
    totals = Counter((name, mass) for sequence in train for name, mass in zip(sequence.objects, sequence.masses or ()))
    expected = {**{(name, value): 160 // len(values) for name in BOXES for value in values},
                **{(name, value): 80 // len(values) for name in BAGS for value in values}}
    if dict(totals) != expected:
        problems.append("train: not every (parcel, in-domain mass) 32 times for a box and 16 for a bag")
    for slot in range(L4_PARCELS_PER_EPISODE):
        per_slot = Counter((sequence.objects[slot], (sequence.masses or (None,) * L4_PARCELS_PER_EPISODE)[slot])
                           for sequence in train)
        for name, count in [(name, 32) for name in BOXES] + [(name, 16) for name in BAGS]:
            share = count // len(values)
            if not all(per_slot[(name, value)] in (share, share + 1) for value in values):
                problems.append(f"train: slot {slot}'s {name} masses are not {share} or {share + 1} of each value")
    if any(sequence.rotations is not None or sequence.masses is not None for sequence in eval_set):
        problems.append("eval: a sequence fixes its rotations or masses; a Run draws them")
    return problems


def training_sequences_text(train: list[L4Sequence], seed: int = SEQUENCE_SEED) -> str:
    """The training file: its source and digest, then one sequence per line."""
    def line(sequence: L4Sequence) -> str:
        return "        " + json.dumps({"id": sequence.id, "objects": list(sequence.objects),
                                        "rotations": list(sequence.rotations), "masses": list(sequence.masses)})

    parts = ["{",
             '    "_generated": "from build_sequences(seed); do not edit",',
             f'    "seed": {seed},',
             f"    \"source\": {json.dumps(TRAINING_SET_SOURCE)},",
             f'    "source_sha256": "{TRAINING_SET_SHA256}",',
             '    "sequences": [',
             ",\n".join(line(sequence) for sequence in train),
             "    ]",
             "}"]
    return "\n".join(parts) + "\n"


@lru_cache(maxsize=1)
def load_l4_sequences() -> dict[str, L4Sequence]:
    """Every sequence by id: :data:`EVAL_SEQUENCES`, then the training file's, each training one checked to name five
    known parcels with an in-domain rotation and mass for each."""
    data = json.loads(TRAINING_SEQUENCES_PATH.read_text())
    sequences = {sequence.id: sequence for sequence in EVAL_SEQUENCES}
    for entry in data["sequences"]:
        sequence = L4Sequence(entry["id"], tuple(entry["objects"]), tuple(entry["rotations"]), tuple(entry["masses"]))
        unknown = sorted(set(sequence.objects) - set(BOXES + BAGS))
        if len(sequence.objects) != L4_PARCELS_PER_EPISODE or unknown or sequence.id in sequences:
            raise ValueError(f"{TRAINING_SEQUENCES_PATH.name}: {sequence.id} names {list(sequence.objects)}")
        if any(label not in rotation_labels(name) for name, label in zip(sequence.objects, sequence.rotations)):
            raise ValueError(f"{TRAINING_SEQUENCES_PATH.name}: {sequence.id} puts a parcel down outside its in-domain rotations")
        if any(mass not in MASS_MULTIPLIERS["id"] for mass in sequence.masses):
            raise ValueError(f"{TRAINING_SEQUENCES_PATH.name}: {sequence.id} weighs a parcel outside the in-domain masses")
        sequences[sequence.id] = sequence
    return sequences
