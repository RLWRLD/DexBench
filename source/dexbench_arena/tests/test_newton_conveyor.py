# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""A BeltDriveCfg drives the collider it names, at its velocity in its frame, with its grip."""

from __future__ import annotations

import math

import numpy as np
import pytest

wp = pytest.importorskip("warp")
newton = pytest.importorskip("newton")
pytest.importorskip("isaaclab_newton", reason="the conveyor manager subclasses Isaac Lab's MJWarp manager")

from dexbench_arena.assets.newton_conveyor import BeltDriveCfg, build_belt_treadmill

BELT_LABEL = "/World/envs/env_{env}/Conveyor/Belt/TopRun"
FRAME_LABEL = "/World/envs/env_{env}/Conveyor/Frame"
AUTHORED_MU = 0.5
FRAME_MU = 0.8
YAW_DEG = 40.0


def _model(envs: int = 2):
    """Each env: a kinematic belt yawed YAW_DEG on a FREE joint (as the USD parse gives one) and a static frame."""
    yaw = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.radians(YAW_DEG))
    builder = newton.ModelBuilder()
    shapes, frames = [], []
    for env in range(envs):
        body = builder.add_link(xform=wp.transform(wp.vec3(0.0, 3.0 * env, 0.7), yaw), mass=100.0, is_kinematic=True)
        shapes.append(builder.add_shape_box(body, hx=2.0, hy=0.4, hz=0.05, label=BELT_LABEL.format(env=env),
                                            cfg=newton.ModelBuilder.ShapeConfig(mu=AUTHORED_MU)))
        builder.add_articulation([builder.add_joint_free(body)])
        frames.append(builder.add_shape_box(-1, hx=2.0, hy=0.5, hz=0.3, label=FRAME_LABEL.format(env=env),
                                            cfg=newton.ModelBuilder.ShapeConfig(mu=FRAME_MU)))
    model = builder.finalize(device="cpu")
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    return model, state, shapes, frames, yaw


def _apply_to_a_moved_state(model, state, treadmill, shapes):
    """Each belt's (FREE joint velocity, body twist, body pose, rest pose) after ``treadmill.apply`` on a state whose
    belts were knocked off their pose and given a wrong velocity, as MuJoCo's own integration would leave them."""
    bodies = model.shape_body.numpy()[shapes]
    joints = [int(np.nonzero(model.joint_child.numpy() == b)[0][0]) for b in bodies]
    qd_starts = model.joint_qd_start.numpy()[joints]
    rest = state.body_q.numpy()[bodies].copy()

    body_q = state.body_q.numpy()
    body_q[bodies, :3] += 0.3
    state.body_q.assign(body_q)
    state.joint_qd.assign(np.full(state.joint_qd.shape, 9.0, dtype=np.float32))
    state.body_qd.assign(np.full((state.body_qd.shape[0], 6), 9.0, dtype=np.float32))
    treadmill.apply(state)

    joint_qd, body_qd, body_q = state.joint_qd.numpy(), state.body_qd.numpy(), state.body_q.numpy()
    return [(joint_qd[s:s + 6], body_qd[b], body_q[b], r) for s, b, r in zip(qd_starts, bodies, rest)]


def test_a_belt_frame_belt_runs_along_its_own_axis_with_its_grip():
    model, state, shapes, frames, yaw = _model()
    treadmill, friction = build_belt_treadmill(
        model, state,
        (BeltDriveCfg(shape_pattern=r"/Belt/TopRun$", velocity=(0.1, 0.0, 0.0), frame="belt", friction=1.0),))

    along = np.array(wp.quat_rotate(yaw, wp.vec3(0.1, 0.0, 0.0)))
    driven = _apply_to_a_moved_state(model, state, treadmill, shapes)
    assert len(driven) == len(shapes)
    for joint_qd, body_qd, body_q, rest in driven:
        assert np.allclose(joint_qd[:3], along, atol=1e-7) and not joint_qd[3:].any()
        assert np.allclose(body_qd[:3], along, atol=1e-7) and not body_qd[3:].any()
        assert np.array_equal(body_q, rest)
    mu = model.shape_material_mu.numpy()
    assert friction and np.allclose(mu[shapes], 1.0) and np.allclose(mu[frames], FRAME_MU)


def test_a_world_frame_belt_keeps_its_velocity_and_its_authored_grip():
    model, state, shapes, _, _ = _model()
    treadmill, friction = build_belt_treadmill(
        model, state, (BeltDriveCfg(shape_pattern=r"/Belt/TopRun$", velocity=(0.45, 0.0, 0.0)),))

    for joint_qd, body_qd, body_q, rest in _apply_to_a_moved_state(model, state, treadmill, shapes):
        assert np.array_equal(joint_qd, np.float32([0.45, 0.0, 0.0, 0.0, 0.0, 0.0]))
        assert np.array_equal(body_qd, joint_qd) and np.array_equal(body_q, rest)
    assert not friction and np.allclose(model.shape_material_mu.numpy()[shapes], AUTHORED_MU)


def test_a_pattern_that_names_no_colliding_shape_is_refused():
    model, state, _, _, _ = _model(envs=1)
    with pytest.raises(ValueError, match="no shape colliding"):
        build_belt_treadmill(model, state, (BeltDriveCfg(shape_pattern=r"/Roller$", velocity=(0.1, 0.0, 0.0)),))


def test_driving_belts_keeps_arenas_newton_preset_and_its_manager(simulation_app):
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import (
        ArenaPhysicsCfg,
        NewtonArenaMJWarpManager,
    )

    from dexbench_arena.assets.newton_conveyor_manager import drive_conveyor_belts

    physics = ArenaPhysicsCfg().newton
    physics.solver_cfg.ls_iterations = 7
    belts = (BeltDriveCfg(shape_pattern=r"/Belt/TopRun$", velocity=(0.1, 0.0, 0.0)),)
    driven = drive_conveyor_belts(physics, belts)

    assert driven.solver_cfg.belts == belts
    assert issubclass(driven.class_type, NewtonArenaMJWarpManager)
    kept = {k: v for k, v in driven.solver_cfg.to_dict().items() if k not in ("class_type", "belts")}
    assert kept == {k: v for k, v in physics.solver_cfg.to_dict().items() if k != "class_type"}
