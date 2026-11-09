# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""H3 placement does not read velocity and requires retained neighbours."""
from types import SimpleNamespace
import pytest
pytest.importorskip('isaaclab.managers')
import torch
import warp as wp
from dexbench_arena.tasks.extract_utensil import extracted_utensil_placed, utensil_dropped
from dexbench_arena.environments.manipulation.newton_shape_contact import ShapeContactRequest, set_shape_contact

class Scene(dict):
    env_origins = torch.zeros((1, 3))

def test_placement_does_not_read_velocity_and_refuses_lost_neighbour():
    scene = Scene(robot=SimpleNamespace(body_names=['wrist'], data=SimpleNamespace(
        body_pos_w=torch.tensor([[[3.55, .60, 1.10]]]))))
    positions = {'spoon': [3.40, .27, .762], 'fork': [3.55, .60, 1.], 'knife': [3.56, .60, 1.]}
    for name, pos in positions.items():
        scene[name] = SimpleNamespace(data=SimpleNamespace(
            root_pos_w=wp.from_torch(torch.tensor([pos]), dtype=wp.vec3),
            root_quat_w=wp.from_torch(torch.tensor([[0., 0., 0., 1.]]), dtype=wp.quat)))
    env = SimpleNamespace(scene=scene, device='cpu', step_dt=1/60, common_step_counter=1,
                          episode_length_buf=torch.tensor([1]))
    params = dict(object_names=list(positions), object_bounds=[[[-.005]*3, [.005]*3]]*3, reference_body='wrist')
    assert extracted_utensil_placed(env, **params, target_object='spoon', hold_time_s=0).item()
    scene['robot'].data.body_pos_w[:, 0, 2] = .85
    scene['fork'].data.root_pos_w = wp.from_torch(torch.tensor([[3.55, .60, .757]]), dtype=wp.vec3)
    env.common_step_counter += 1
    assert not extracted_utensil_placed(env, **params, target_object='spoon', hold_time_s=0).item()
    assert not utensil_dropped(env, **params).item()
    scene['fork'].data.root_pos_w = wp.from_torch(torch.tensor([[3.55, .60, .50]]), dtype=wp.vec3)
    env.common_step_counter += 1
    assert not extracted_utensil_placed(env, **params, target_object='spoon', hold_time_s=0).item()
    assert utensil_dropped(env, **params).item()

def test_optional_support_friction_changes_only_selected_colliders():
    builder = SimpleNamespace(shape_label=['/Robot/left_palm/collision', '/Robot/right_palm/collision'],
        shape_flags=[2, 2], shape_body=[0, 1], body_label=['/Robot/left_palm', '/Robot/right_palm'],
        shape_material_ke=[1., 1.], shape_material_kd=[1., 1.], shape_material_mu=[5., 5.],
        custom_attributes={'mujoco:geom_solmix': SimpleNamespace(values={})})
    set_shape_contact(builder, ShapeContactRequest('left', '/left_palm$', 40000., 400., 1., match='body', friction=1.1))
    assert builder.shape_material_mu == [1.1, 5.]
    set_shape_contact(builder, ShapeContactRequest('both', '/collision$', 40000., 400., 1.))
    assert builder.shape_material_mu == [1.1, 5.]
