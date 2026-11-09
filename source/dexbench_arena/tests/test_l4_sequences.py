# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""DB-L4's evaluation table and training file are what its generator makes, and keep the benchmark's quotas."""

from collections import Counter

from dexbench_arena.layouts.e2_logistics_l4_sequences import (
    EVAL_SEQUENCES,
    L4_SEQUENCE_AXIS,
    TRAINING_SEQUENCES_PATH,
    build_sequences,
    check_sequences,
    load_l4_sequences,
    training_sequences_text,
)


def test_the_checked_in_sequences_are_the_generators_and_keep_every_rule():
    sets = build_sequences()
    assert check_sequences(sets) == []
    assert tuple(sets["eval"]) == EVAL_SEQUENCES
    assert TRAINING_SEQUENCES_PATH.read_text() == training_sequences_text(sets["train"])
    assert L4_SEQUENCE_AXIS.train() == tuple(sequence.id for sequence in EVAL_SEQUENCES)
    loaded = load_l4_sequences()
    assert len(loaded) == 252
    train = [sequence for key, sequence in loaded.items() if key.startswith("train_")]
    assert set(Counter(pair for sequence in train for pair in zip(sequence.objects, sequence.rotations)).values()) == {10}
    masses = Counter(pair for sequence in train for pair in zip(sequence.objects, sequence.masses))
    assert {count for (name, _), count in masses.items() if not name.startswith("polybag")} == {32}
    assert {count for (name, _), count in masses.items() if name.startswith("polybag")} == {16}


def test_check_sequences_names_a_broken_quota():
    sets = build_sequences()
    sets["train"] = sets["train"][:-1] + [sets["eval"][0]]
    problems = check_sequences(sets)
    assert any("training sequence is an evaluation one" in problem for problem in problems)
    assert any("train: sequences by bag count" in problem or "train: slot" in problem for problem in problems)
