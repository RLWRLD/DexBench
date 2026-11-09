# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""A condition recording's settle field is each environment's own (Kit-free; needs torch and Arena)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
condition_recording = pytest.importorskip("dexbench_arena.tasks.condition_recording")


def test_each_env_records_its_own_settle_and_termination(monkeypatch):
    """Env 0 settled and timed out; env 1 still has a box moving and dropped it: neither inherits the other's."""
    settled = {"box": torch.tensor([True, False]), "neighbour": torch.tensor([True, True])}
    monkeypatch.setattr(condition_recording, "compute_objects_settled_mask",
                        lambda world, scene, names, lin, ang: settled[names[0]])
    terms = {"time_out": torch.tensor([True, False]), "object_dropped": torch.tensor([False, True])}
    env = SimpleNamespace(
        arena_world=None,
        scene=SimpleNamespace(rigid_objects={"neighbour": object(), "box": object()}),
        termination_manager=SimpleNamespace(active_terms=list(terms), get_term=terms.__getitem__),
    )
    first, second = (condition_recording.record_settle(env, env_id)["settle"] for env_id in (0, 1))
    assert (first["settled"], first["unsettled"], first["terminated_by"]) == (True, [], ["time_out"])
    assert (second["settled"], second["unsettled"], second["terminated_by"]) == (False, ["box"], ["object_dropped"])
    assert condition_recording.record_settle(env, 1)["settle"] == second
    assert condition_recording.record_settle(env, 0)["settle"] == first


def test_a_belt_carried_object_is_at_rest_when_it_moves_with_its_belt():
    """Arena's own velocity test, in the belt's frame for what the environment says a belt carries: a box riding the
    0.1 m/s belt has settled, a part sliding along it has not, and an object off the belt is judged in the world."""
    belt = (0.1, 0.0, 0.0)
    linear = {
        "kit_box": [0.1, 0.0, 0.0],
        "kit_box_part": [0.15, 0.0, 0.0],
        "bin_part": [0.0, 0.0, 0.0],
        "loose_box": [0.1, 0.0, 0.0],
    }
    world = SimpleNamespace(
        get_root_linear_velocity_w=lambda name: torch.tensor([linear[name]]),
        get_root_angular_velocity_w=lambda name: torch.zeros(1, 3),
    )
    env = SimpleNamespace(
        arena_world=world,
        scene=SimpleNamespace(rigid_objects=dict.fromkeys(linear), deformable_objects={}, num_envs=1, device="cpu"),
        termination_manager=SimpleNamespace(active_terms=["time_out"], get_term=lambda name: torch.tensor([True])),
    )
    carried = {"kit_box": belt, "kit_box_part": belt}
    settle = condition_recording.record_settle(env, 0, belt_carried=carried)["settle"]
    assert (settle["settled"], settle["unsettled"]) == (False, ["kit_box_part", "loose_box"])
    assert settle["relative_to_belt"] == {"kit_box": [0.1, 0.0, 0.0], "kit_box_part": [0.1, 0.0, 0.0]}

    plain = condition_recording.record_settle(env, 0)["settle"]
    assert plain["unsettled"] == ["kit_box", "kit_box_part", "loose_box"] and "relative_to_belt" not in plain
    with pytest.raises(ValueError, match=r"\['ghost'\] are no rigid object"):
        condition_recording.record_settle(env, 0, belt_carried={"ghost": belt})
