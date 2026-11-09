# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""The strict tiers' hold: seconds of control steps, independent of control rate and read count."""

from types import SimpleNamespace

import pytest
import torch

from dexbench_arena.tasks.predicates.hold import SustainedHold, reset_sustained, sustained, sustained_count


@pytest.mark.parametrize("hz", [20, 30, 60])
def test_hold_uses_control_dt_and_ignores_duplicate_reads(hz):
    hold = SustainedHold(torch.tensor([True]))
    for step in range(1, hz // 2 + 1):
        expected = step == hz // 2
        for _ in range(3):
            assert hold.update(torch.tensor([True]), step, 1 / hz, 0.5).item() == expected


def test_false_or_missing_step_breaks_continuity():
    hold = SustainedHold(torch.tensor([True]))
    for step in range(1, 15):
        hold.update(torch.tensor([True]), step, 1 / 30, 0.5)
    assert not hold.update(torch.tensor([False]), 15, 1 / 30, 0.5)
    assert not hold.update(torch.tensor([True]), 16, 1 / 30, 0.5)
    hold.update(torch.tensor([True]), 18, 1 / 30, 0.5)
    assert hold.count.item() == 1


def test_partial_reset_clears_even_a_successful_pose_at_same_global_step():
    hold = SustainedHold(torch.tensor([True, True]))
    for step in range(1, 16):
        hold.update(torch.tensor([True, True]), step, 1 / 30, 0.5)
    hold.reset(torch.tensor([0]))
    assert hold.update(torch.tensor([True, True]), 15, 1 / 30, 0.5).tolist() == [False, True]
    hold.reset()
    assert hold.count.tolist() == [0, 0]


def fake_env(hz, num_envs=1):
    env = SimpleNamespace(
        step_dt=1.0 / hz, common_step_counter=0, episode_length_buf=torch.zeros(num_envs, dtype=torch.long)
    )
    env.unwrapped = env
    return env


def first_success_s(hz, ok_from_s, ok_until_s, hold_time_s, reads_per_step, duration_s=3.0):
    """Run one episode of a condition true on [ok_from_s, ok_until_s) of sim time; return when the hold fires."""
    env = fake_env(hz)
    for step in range(1, int(round(duration_s * hz)) + 1):
        env.common_step_counter = step
        env.episode_length_buf[:] = step
        t = step / hz
        ok = torch.tensor([ok_from_s <= t < ok_until_s])
        results = {bool(sustained(env, ok, hold_time_s, "strict")[0]) for _ in range(reads_per_step)}
        assert len(results) == 1, "a second read in the same step changed the answer"
        if results.pop():
            return t
    return None


@pytest.mark.parametrize("reads_per_step", [1, 3])
def test_same_trajectory_same_result_at_30_and_60_hz(reads_per_step):
    """A 1 s hold of a condition that starts at 0.2 s fires near 1.2 s at either rate; a 0.8 s spell never does."""
    held = {hz: first_success_s(hz, 0.2, 10.0, 1.0, reads_per_step) for hz in (30, 60)}
    assert all(t is not None and 1.15 < t < 1.2 for t in held.values()), held
    assert abs(held[30] - held[60]) <= 1 / 30
    assert {hz: first_success_s(hz, 0.2, 1.0, 1.0, reads_per_step) for hz in (30, 60)} == {30: None, 60: None}


def test_a_reset_env_starts_its_hold_over_without_a_reset_event():
    env = fake_env(30, num_envs=2)
    for step in range(1, 11):
        env.common_step_counter = step
        env.episode_length_buf[:] = step
        sustained(env, torch.tensor([True, True]), 1.0, "strict")
    env.episode_length_buf[0] = 0
    assert sustained(env, torch.tensor([True, True]), 1.0, "strict").tolist() == [False, False]
    assert sustained_count(env, "strict").tolist() == [0, 10]
    env.common_step_counter = 11
    env.episode_length_buf += 1
    sustained(env, torch.tensor([True, True]), 1.0, "strict")
    assert sustained_count(env, "strict").tolist() == [1, 11]
    reset_sustained(env, None, "strict")
    assert sustained_count(env, "strict").tolist() == [0, 0]


def test_zero_hold_is_the_single_step_tier():
    env = fake_env(30)
    ok = torch.tensor([True])
    assert sustained(env, ok, 0.0, "collection") is ok
    assert sustained_count(env, "collection") is None


def share_success_step(hz, rest_steps, hold_time_s=1.0, share=0.9, duration_s=3.0):
    """Run one episode whose condition holds on the steps in ``rest_steps``; return the step a share hold fires."""
    env = fake_env(hz)
    for step in range(1, int(round(duration_s * hz)) + 1):
        env.common_step_counter = step
        env.episode_length_buf[:] = step
        if bool(sustained(env, torch.tensor([step in rest_steps]), hold_time_s, "share", share)[0]):
            return step
    return None


@pytest.mark.parametrize("hz", [30, 60])
def test_a_share_hold_passes_a_second_broken_by_a_tenth_and_not_by_more(hz):
    """90% of a sliding 1 s window: at 60 Hz 54 of the last 60 steps, wherever the misses fall."""
    every = set(range(1, 3 * hz + 1))
    allowed = round(0.1 * hz)
    gaps = {hz // 3 + 2 * k for k in range(allowed)}
    assert share_success_step(hz, every) == hz
    assert share_success_step(hz, every - gaps) == hz
    assert share_success_step(hz, {s for s in every if s % 10 not in (0, 1)}, duration_s=3.0) is None
    env = fake_env(hz)
    fired = None
    for step in range(1, 3 * hz + 1):
        env.common_step_counter = step
        env.episode_length_buf[:] = step
        if fired is None and bool(sustained(env, torch.tensor([step in every - gaps]), 1.0, "run")[0]):
            fired = step
    assert fired == max(gaps) + hz


def test_a_share_hold_never_reaches_back_across_a_reset():
    env = fake_env(60, num_envs=2)
    for step in range(1, 51):
        env.common_step_counter = step
        env.episode_length_buf[:] = step
        sustained(env, torch.tensor([True, True]), 1.0, "share", 0.9)
    env.episode_length_buf[0] = 0
    sustained(env, torch.tensor([True, True]), 1.0, "share", 0.9)
    assert sustained_count(env, "share").tolist() == [0, 50]
    for step in range(51, 61):
        env.common_step_counter = step
        env.episode_length_buf += 1
        result = sustained(env, torch.tensor([True, True]), 1.0, "share", 0.9)
    assert result.tolist() == [False, True]
    assert sustained_count(env, "share").tolist() == [10, 60]
