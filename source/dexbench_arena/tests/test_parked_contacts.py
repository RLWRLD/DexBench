# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Run the actual Warp contact-filter kernel without Kit or a GPU."""

from types import SimpleNamespace

import numpy as np
import pytest

wp = pytest.importorskip("warp")

from dexbench_arena.assets.parked_contacts import ParkedBodyContactFilter


def make_scene():
    poses = np.zeros((6, 7), dtype=np.float32)
    poses[:, 6] = 1
    poses[:, 0] = [0, -30, -60, 10, -20, -50]
    model = SimpleNamespace(
        device="cpu", body_q=wp.array(poses, dtype=wp.transform, device="cpu"),
        body_label=[f"/env_{e}/box_{s}/Body" for e in range(2) for s in range(3)],
    )
    mapping = np.array([[-1, 0, 1, 2], [-1, 3, 4, 5]], dtype=np.int32)
    mjw_model = SimpleNamespace(
        opt=SimpleNamespace(run_collision_detection=True),
        callback=SimpleNamespace(contactfilter=None),
        geom_bodyid=wp.array([0, 1, 2, 3], dtype=int, device="cpu"),
    )
    solver = SimpleNamespace(model=model, mjw_model=mjw_model, use_mujoco_cpu=False,
                             mjc_body_to_newton=wp.array(mapping, dtype=int, device="cpu"))
    positions = np.zeros((2, 4, 3), dtype=np.float32)
    positions[:, 1:] = poses[:, :3].reshape(2, 3, 3)
    pairs = [[1, 0], [2, 0], [3, 0], [0, 0]] * 2
    data = SimpleNamespace(
        naconmax=8, nacon=wp.array([8], dtype=int, device="cpu"),
        xpos=wp.array(positions, dtype=wp.vec3, device="cpu"),
        contact=SimpleNamespace(
            worldid=wp.array([0] * 4 + [1] * 4, dtype=int, device="cpu"),
            geom=wp.array(pairs, dtype=wp.vec2i, device="cpu"),
            type=wp.full(8, 3, dtype=int, device="cpu"),
            efc_address=wp.full((8, 4), 7, dtype=int, device="cpu"),
        ),
    )
    offsets = [(f"/env_.*/box_{s}/.*", -90 if s == 0 else 0, 0, 0) for s in range(3)]
    return solver, data, offsets


def test_initial_switch_partial_reset_and_saved_pose_restore():
    solver, data, offsets = make_scene()
    callback = ParkedBodyContactFilter(solver, offsets)
    saved = data.xpos.numpy().copy()

    def check(expected):
        data.contact.type.fill_(3)
        data.contact.efc_address.fill_(7)
        callback(solver.mjw_model, data)
        assert data.contact.type.numpy().tolist() == expected
        disabled = np.array(expected) == 0
        assert np.all(data.contact.efc_address.numpy()[disabled] == -1)
        assert np.all(data.contact.efc_address.numpy()[~disabled] == 7)

    check([3, 0, 0, 3, 3, 0, 0, 3])
    poses = saved.copy()
    poses[0, 1, 0], poses[0, 2, 0] = -90, 0
    data.xpos.assign(poses)
    check([0, 3, 0, 3, 3, 0, 0, 3])
    poses[1, 1, 0], poses[1, 3, 0] = -80, 10
    data.xpos.assign(poses)
    check([0, 3, 0, 3, 0, 0, 3, 3])
    poses[0] = saved[0]
    data.xpos.assign(poses)
    check([3, 0, 0, 3, 0, 0, 3, 3])
    data.xpos.assign(saved)
    check([3, 0, 0, 3, 3, 0, 0, 3])


def test_filter_chains_and_rebuilds_without_reusing_old_buffers():
    solver, data, offsets = make_scene()
    calls = []
    solver.mjw_model.callback.contactfilter = lambda m, d: calls.append(d)
    first = ParkedBodyContactFilter(solver, offsets)
    first(solver.mjw_model, data)
    assert calls == [data]
    rebuilt, new_data, offsets = make_scene()
    second = ParkedBodyContactFilter(rebuilt, offsets)
    assert second.parked.ptr != first.parked.ptr
    second(rebuilt.mjw_model, new_data)
    np.testing.assert_array_equal(new_data.contact.type.numpy(), data.contact.type.numpy())


def test_missing_body_and_unsupported_solver_fail_explicitly():
    solver, _, offsets = make_scene()
    with pytest.raises(ValueError, match="No Newton body"):
        ParkedBodyContactFilter(solver, [("/missing", 0, 0, 0)])
    solver.use_mujoco_cpu = True
    with pytest.raises(ValueError, match="native MuJoCo-Warp"):
        ParkedBodyContactFilter(solver, offsets)
    solver.use_mujoco_cpu = False
    solver.mjw_model.opt.run_collision_detection = False
    with pytest.raises(ValueError, match="native MuJoCo-Warp"):
        ParkedBodyContactFilter(solver, offsets)
