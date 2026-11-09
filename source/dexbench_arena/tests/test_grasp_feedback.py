# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Support feedback is invariant to world placement and protects the named target."""

from types import SimpleNamespace

import pytest

pytest.importorskip("isaaclab.managers")

import torch
import warp as wp
from isaaclab.managers import EventTermCfg
from isaaclab.utils.math import quat_apply, quat_from_angle_axis

from dexbench_arena.tasks.predicates.grasp_feedback import SupportGraspFeedback


class Scene(dict):
    pass


def fixture():
    rotations = quat_from_angle_axis(torch.tensor([0., torch.pi / 2]), torch.tensor([[0., 0., 1.]] * 2))
    origins = torch.tensor([[0., 0., 0.], [5., -3., 2.]])
    local_points = torch.tensor([[0., 0., 0.], [.1, 0., 0.], [.15, 0., 0.]])
    positions = quat_apply(rotations[:, None].expand(-1, 3, -1), local_points[None].expand(2, -1, -1))
    positions += origins[:, None]
    robot = SimpleNamespace(
        body_names=['wrist', 'thumb', 'index'],
        find_bodies=lambda pattern: ([['wrist', 'thumb', 'index'].index(pattern)], [pattern]),
        data=SimpleNamespace(body_pos_w=positions, body_quat_w=rotations[:, None].expand(-1, 3, -1)),
    )
    scene = Scene(robot=robot)
    scene.env_origins = origins
    for name, local in [('fork', [.1, 0., 0.]), ('knife', [.1, .05, 0.])]:
        root = quat_apply(rotations, torch.tensor([local] * 2)) + origins
        scene[name] = SimpleNamespace(data=SimpleNamespace(
            root_pos_w=wp.from_torch(root.contiguous(), dtype=wp.vec3),
            root_quat_w=wp.from_torch(rotations.contiguous(), dtype=wp.quat),
        ))
    calls = []
    action = SimpleNamespace(update_support_grasp=lambda *args: calls.append(args))
    env = SimpleNamespace(scene=scene, num_envs=2, device='cpu', step_dt=1 / 60,
                          action_manager=SimpleNamespace(get_term=lambda name: action))
    cfg = EventTermCfg(func=SupportGraspFeedback, mode='interval', is_global_time=True,
                       interval_range_s=(1 / 30, 1 / 30), params=dict(
                           robot_name='robot', reference_body='wrist', finger_body_patterns=['thumb', 'index'],
                           object_names=['fork', 'knife'], object_bounds=[[[-.02, -.01, -.01], [.02, .01, .01]]] * 2,
                           target_object='knife', support_action='joint_pos', feedback_dt=1 / 30))
    return env, cfg, calls


def test_feedback_rotates_centers_and_distances_and_preserves_target_identity():
    env, cfg, calls = fixture()
    term = SupportGraspFeedback(cfg, env)
    term(env, None, **cfg.params)
    relative, target, distances, dt = calls[0]
    torch.testing.assert_close(relative, torch.tensor([[[.1, 0., 0.], [.1, .05, 0.]]] * 2), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(distances, torch.tensor([[[0., .04], [.03, .05]]] * 2), atol=1e-6, rtol=1e-5)
    assert target.tolist() == [[False, True], [False, True]]
    assert dt == 1 / 30
    with pytest.raises(ValueError, match='global interval'):
        term(env, torch.tensor([1]), **cfg.params)


def test_feedback_rejects_cadence_that_changes_filter_time():
    env, cfg, _ = fixture()
    cfg.interval_range_s = (0., 0.)
    with pytest.raises(ValueError, match='fixed global interval'):
        SupportGraspFeedback(cfg, env)
    cfg.params['feedback_dt'] = .025
    cfg.interval_range_s = (.025, .025)
    with pytest.raises(ValueError, match='integer number'):
        SupportGraspFeedback(cfg, env)
