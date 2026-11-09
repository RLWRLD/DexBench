# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
from dexbench_arena.fluids.pour_evaluation import Cavity, measure


def test_cavity_capacity_and_real_boundaries():
    cone = Cavity([0, .1], [0, .05])
    assert cone.capacity_ml == pytest.approx(np.pi*.05**2*.1/3*1e6)
    p = np.array([[.04, 0, .01], [0, 0, .05], [0, 0, .1], [0, 0, -.01]])
    assert cone.contains(p, np.zeros(3), np.eye(3)).tolist() == [False, True, False, False]


def test_rotation_and_nonuniform_mass_accounting():
    cup, jug = Cavity([.1, .2], [.05, .05]), Cavity([0, .2], [.1, .1])
    rot = np.array([[0., 0, 1], [0, 1, 0], [-1, 0, 0]])
    origin = np.array([1., 0, .3])
    p = np.array([[0, 0, .15], [0, 0, .25], [.2, 0, .05], [np.nan, 0, 0],
                  np.array([0, 0, .1]) @ rot.T+origin])
    masses = np.array([.2, .02, .03, .01, .54])
    row, masks = measure(p, masses, 1000, cup, (np.zeros(3), np.eye(3)), jug, (origin, rot))
    assert row['target_ml'] == pytest.approx(200)
    assert row['source_ml'] == pytest.approx(540)
    assert row['spilled_ml'] == pytest.approx(30)
    assert row['inflight_ml'] == pytest.approx(20)
    assert row['invalid_ml'] == pytest.approx(10)
    assert sum(row[k+'_ml'] for k in masks) == pytest.approx(800)
    assert np.all(sum(masks.values()) == 1)


def test_source_overlap_takes_precedence():
    c = Cavity([0, 1], [1, 1]); pose = (np.zeros(3), np.eye(3))
    row, _ = measure([[0, 0, .5]], [.001], 1000, c, pose, c, pose)
    assert row['target_count'] == 0 and row['source_count'] == 1
