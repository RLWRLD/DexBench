# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""DB-L4's rotation draw puts each slot's drawn pose in the relay's table, records it, and cannot be switched off."""

from types import SimpleNamespace

import pytest

pytest.importorskip("isaaclab_arena.variations.variations_hydra")

import torch  # noqa: E402
from isaaclab.managers import EventTermCfg  # noqa: E402
from isaaclab_arena.variations import variations_hydra  # noqa: E402
from isaaclab_arena.variations.variation_recorder import VariationRecorder  # noqa: E402

from dexbench_arena.environments.manipulation.e2_logistics_l4_environment import l4_sequence_rotations  # noqa: E402
from dexbench_arena.layouts.e2_logistics_l4_layout import ROTATION_LABELS, rotation_labels  # noqa: E402
from dexbench_arena.variations.db_l4_rotation_variation import (  # noqa: E402
    ApplyL4RotationFromSampler,
    L4RotationVariation,
)
from dexbench_arena.variations.variation_samples_recorder import VariationSamplesRecorder  # noqa: E402

LABELS = rotation_labels("box_1", "ood")


def _variation(slot: int = 1) -> L4RotationVariation:
    return L4RotationVariation(slot, LABELS, tuple((float(i), 0.0, 0.8, 0.0, 0.0, 0.0, 1.0) for i in range(len(LABELS))))


def test_a_run_config_cannot_disable_the_draw():
    variation = _variation()
    with pytest.raises(ValueError, match="l4_rotation_draw"):
        variations_hydra.apply_overrides({"l4_park_slab": [variation]}, ["l4_park_slab.rotation_slot_1.enabled=false"])
    with pytest.raises(ValueError, match="l4_rotation_draw"):
        variation.disable()
    assert variation.enabled


def test_the_draw_fills_its_slot_of_the_relay_table_and_the_demo_records_the_label_index():
    variation = _variation()
    recorder = VariationRecorder()
    recorder.attach({"l4_park_slab": [variation]})
    table = torch.full((3, 5, 7), -1.0)
    relay = SimpleNamespace(func=SimpleNamespace(stage_poses=table))
    env = SimpleNamespace(num_envs=3, device="cpu", event_manager=SimpleNamespace(get_term_cfg=lambda name: relay),
                          variation_recorder=recorder, get_episode_index=lambda env_id: 0)
    recorder.bind_env(env)
    name, cfg = variation.build_event_cfg()
    term = ApplyL4RotationFromSampler(EventTermCfg(func=ApplyL4RotationFromSampler, params=cfg.params), env)
    term(env, torch.tensor([0, 2]), **cfg.params)
    drawn = table[[0, 2], 1, 0].long().tolist()
    assert (table[1] == -1.0).all() and (table[:, [0, 2, 3, 4]] == -1.0).all()
    recording = VariationSamplesRecorder.__new__(VariationSamplesRecorder)
    recording._env = env
    key, values = recording.record_post_reset([0, 2])
    assert key == "variations"
    assert values["l4_park_slab/rotation_slot_1"].reshape(-1).tolist() == [
        float(ROTATION_LABELS.index(LABELS[i])) for i in drawn]


def test_an_evaluation_sequence_needs_its_rotations_drawn_or_given():
    def cfg(**fields):
        return SimpleNamespace(**{"l4_sequence": None, "l4_rotations": None, "l4_rotation_draw": None, **fields})

    with pytest.raises(ValueError, match="evaluation sequence"):
        l4_sequence_rotations(cfg(l4_sequence="eval_00"))
    objects, labels = l4_sequence_rotations(cfg(l4_sequence="eval_00", l4_rotation_draw="ood"))
    assert all(label in rotation_labels(name, "ood") for name, label in zip(objects, labels))
    with pytest.raises(ValueError, match="give one"):
        l4_sequence_rotations(cfg(l4_sequence="eval_00", l4_rotation_draw="id", l4_rotations="up/far"))
    with pytest.raises(ValueError, match="not a put-down rotation of polybag_2"):
        l4_sequence_rotations(cfg(l4_sequence="train_000", l4_rotations="near/up,up/far,up/far,up/far,up/far"))
