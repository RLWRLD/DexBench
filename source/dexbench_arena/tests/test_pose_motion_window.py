# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Settling must follow actual motion, with no history leaking across resets."""

import importlib.util
from pathlib import Path

import pytest
import torch

PATH = Path(__file__).resolve().parents[1] / "dexbench_arena/tasks/_pose_motion_window.py"
SPEC = importlib.util.spec_from_file_location("pose_motion_window", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
PoseMotionWindow = MODULE.PoseMotionWindow


def test_requires_full_window_and_rejects_motion_even_if_it_returns_to_start():
    window = PoseMotionWindow(0.25, 1 / 30)
    positions = torch.zeros(1, 5, 3)
    for i in range(9):
        speed = window.update(positions, torch.tensor([i]))
        assert bool((speed < 0.1).all()) == (i == 8)
    positions[:, 2, 0] += 0.01
    assert window.update(positions, torch.tensor([9]))[0, 2] > 0.1
    positions[:, 2, 0] -= 0.01
    for i in range(10, 18):
        assert window.update(positions, torch.tensor([i]))[0, 2] > 0.1
    assert bool((window.update(positions, torch.tensor([18])) < 0.1).all())


def test_constant_translation_uses_the_same_speed_limit():
    window = PoseMotionWindow(0.25, 1 / 30)
    positions = torch.zeros(2, 5, 3)
    velocity = torch.tensor([0.05, 0.15])
    for i in range(12):
        positions[:, :, 0] = velocity[:, None] * i / 30
        speed = window.update(positions, torch.tensor([i, i]))
    torch.testing.assert_close(speed[:, 0], velocity)


def test_duplicate_evaluations_and_independent_resets():
    window = PoseMotionWindow(0.25, 1 / 30)
    positions = torch.zeros(2, 5, 3)
    for i in range(9):
        speed = window.update(positions, torch.tensor([i, i]))
    for _ in range(12):
        torch.testing.assert_close(window.update(positions, torch.tensor([8, 8])), speed)
    reset = window.update(positions, torch.tensor([0, 9]))
    assert bool(torch.isinf(reset[0]).all())
    assert bool((reset[1] == 0).all())
    for i in range(1, 9):
        result = window.update(positions, torch.tensor([i, i + 9]))
        assert bool((result[0] < 0.1).all()) == (i == 8)
        assert bool((result[1] < 0.1).all())


def test_skipped_samples_cannot_prove_stability():
    window = PoseMotionWindow(0.25, 1 / 30)
    positions = torch.zeros(1, 5, 3)
    for i in range(9):
        window.update(positions, torch.tensor([i]))
    assert bool(torch.isinf(window.update(positions, torch.tensor([10]))).all())


@pytest.mark.parametrize("duration,dt", [(0, 0.03), (0.25, 0), (float("nan"), 0.03)])
def test_invalid_durations_are_rejected(duration, dt):
    with pytest.raises(ValueError):
        PoseMotionWindow(duration, dt)
