# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Behavior of the measured screw state and real motor/contact/guide mechanism."""

import math
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("isaaclab")
from dexbench_arena.layouts.db_m3_layout import BOLT_NAMES, SEATED_HEADS
from dexbench_arena.tasks.power_driver_torque import ContactTorqueFastening, TorqueFasteningState
from dexbench_arena.tasks.power_driver_model import FasteningCfg


@pytest.fixture
def torque_scene():
    pose = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]).repeat(2, 1)
    joint_state = torch.zeros(2, 2)
    command = Mock()
    tool = SimpleNamespace(
        joint_names=["SpindleMotor", "TriggerPress"],
        data=SimpleNamespace(
            joint_pos=SimpleNamespace(torch=joint_state),
            joint_vel=SimpleNamespace(torch=joint_state.clone()),
            applied_torque=SimpleNamespace(torch=joint_state.clone()),
            root_link_pose_w=SimpleNamespace(torch=pose),
        ),
        actuators=SimpleNamespace(target_command=command),
        write_joint_state_to_sim_index=Mock(),
    )
    contact = torch.zeros(2, 2, 1, 6, 3)
    trigger_contact = torch.zeros(2, 2, 1, 1, 3)

    class Scene(dict):
        env_origins = torch.zeros(2, 3)

    scene = Scene(
        power_driver=tool,
        driver_trigger_contact=SimpleNamespace(data=SimpleNamespace(
            normal_force_matrix_w_history=SimpleNamespace(torch=trigger_contact))),
        driver_bolt_contact=SimpleNamespace(
            data=SimpleNamespace(normal_force_matrix_w_history=SimpleNamespace(torch=contact))
        ),
    )
    for index, name in enumerate(BOLT_NAMES):
        body_pose = pose[:, None].clone()
        body_pose[:, 0, :3] = torch.tensor(SEATED_HEADS[index])
        scene[name] = SimpleNamespace(
            joint_names=["ScrewRotation", "ThreadTravel"],
            body_names=["Screw"],
            data=SimpleNamespace(
                joint_pos=SimpleNamespace(torch=torch.zeros(2, 2)), body_link_pose_w=SimpleNamespace(torch=body_pose)
            ),
            write_joint_state_to_sim_index=Mock(),
        )
    env = SimpleNamespace(scene=scene, num_envs=2, device="cpu", common_step_counter=0)
    term = ContactTorqueFastening(
        SimpleNamespace(params={"fastening_cfg": {}, "bit_extension_m": 0.12, "motor_enabled": True}), env
    )

    return env, term, command, joint_state, contact, trigger_contact


def test_success_rechecks_current_joints_before_interval_event(torque_scene):
    from dexbench_arena.tasks.power_driver_fastening import all_bolts_seated

    env, term, command, *_ = torque_scene
    env.event_manager = SimpleNamespace(get_term_cfg=lambda name: SimpleNamespace(func=term))
    for i in term.screw_ids:
        term.bolts[i].data.joint_pos.torch[:] = torch.tensor([8 * math.pi, .005])
    assert not all_bolts_seated(env).any()
    term(env, None, {})
    assert all_bolts_seated(env).tolist() == [True, True]
    before_commands = command.set_velocity_index.call_count
    before_turns = term.model.turns.clone()
    term.bolts[1].data.joint_pos.torch[0] = torch.tensor([8 * math.pi - .203, .0049596])
    assert all_bolts_seated(env).tolist() == [False, True]
    assert all_bolts_seated(env).tolist() == [False, True]
    assert torch.equal(term.model.turns, before_turns)
    assert command.set_velocity_index.call_count == before_commands
    term.bolts[1].data.joint_pos.torch[0] = torch.tensor([8 * math.pi, .005])
    assert all_bolts_seated(env).all()
    term.bolts[0].data.joint_pos.torch[1, 0] = torch.nan
    assert all_bolts_seated(env).tolist() == [True, False]


def test_motor_starts_only_on_current_active_head_contact_and_brakes_on_release(torque_scene):
    env, term, command, joint_state, contact, trigger_contact = torque_scene
    def step(enabled=True, direction=1.0):
        env.common_step_counter += 1
        term(env, None, {}, motor_enabled=enabled, motor_direction=direction)
        return command.set_velocity_index.call_args.kwargs["value"].flatten()

    contact[:, 1] = 100.0
    assert not step().any()
    contact[0, 0, 0, 0, 1] = 1.0
    contact[1, 0, 0, 5, 1] = 100.0
    assert not step().any()
    joint_state[:, 1] = .003
    trigger_contact[:, 1] = 10.
    assert not step().any()
    trigger_contact[:, 0, 0, 0, 1] = 1.
    assert step().tolist() == pytest.approx([-8 * math.pi, 0.0])
    joint_state[:, 1] = 0.
    assert not step().any()
    joint_state[:, 1] = .003
    trigger_contact[:, 0] = 0.
    assert not step().any()
    trigger_contact[:, 0, 0, 0, 1] = 1.
    contact[0, 0] = 0.0
    contact[1, 0, 0, 1, 1] = 1.0
    assert step(direction=-1.0).tolist() == pytest.approx([0.0, 8 * math.pi])
    assert not step(enabled=False).any()
    contact[:, 0, 0, 0, 1] = 1.0
    step()
    term.reset(torch.tensor([0]))
    assert term.motor_target_speed_rad_s.tolist() == pytest.approx([0.0, 8 * math.pi])
    assert not command.set_velocity_index.call_args.kwargs["value"].any()
    assert command.set_velocity_index.call_args.kwargs["env_ids"].tolist() == [0]
    contact[:, 0] = 0.01
    assert not step().any()


def test_torque_state_requires_physical_rotation_and_pitch_and_preserves_partial_reset():
    heads = torch.zeros(6, 3)
    state = TorqueFasteningState(heads, 2)
    angles = torch.zeros(2, 6)
    travel = torch.zeros_like(angles)
    positions = heads.expand(2, -1, -1).clone()
    forces = torch.ones_like(angles)
    tip = torch.zeros(2, 3)
    axis = torch.tensor([[0.0, 1.0, 0.0]]).expand(2, -1)
    with pytest.raises(RuntimeError, match="measured screw joints"):
        state.advance(tip, axis, forces, 100.0)
    angles[:, 0] = 8 * math.pi
    travel[:, 0] = 0.005
    state.observe(angles, travel, positions, forces, tip, axis)
    assert not state.success.any()
    angles[:, 1] = 8 * math.pi
    state.observe(angles, travel, positions, forces, tip, axis)
    assert not state.success.any()
    travel[:, 1] = 0.005
    state.observe(angles, travel, positions, forces, tip, axis)
    assert state.success.all() and not state.turns[:, 2:].any()
    state.reset(torch.tensor([0]))
    assert state.success.tolist() == [False, True]
    assert not state.turns[0].any()
    angles[:, 0] = 4 * math.pi
    travel[:, 0] = 0.0025
    state.observe(angles, travel, positions, forces, tip, axis)
    assert state.turns[:, 0].tolist() == [2.0, 2.0]
    assert not state.success.any()


@pytest.mark.parametrize("tolerance,expected", [(.0001, False), (.001, False), (.004, True), (.01, True)])
def test_engagement_uses_configured_axial_window_without_changing_seating(tolerance, expected):
    state = TorqueFasteningState(torch.zeros(6, 3), 2, FasteningCfg(axial_tolerance_m=tolerance))
    angles = torch.zeros(2, 6)
    travel = torch.zeros_like(angles)
    positions = torch.zeros(2, 6, 3)
    forces = torch.ones_like(angles)
    tip = torch.tensor([[0., .003, 0.], [0., -.003, 0.]])
    axis = torch.tensor([[0., 1., 0.]]).expand(2, -1)
    state.observe(angles, travel, positions, forces, tip, axis)
    assert state.engaged[:, :2].tolist() == [[expected, expected]] * 2
    assert not state.engaged[:, 2:].any()
    assert not state.success.any()
    tip[:, 1] = torch.tensor([tolerance, -tolerance])
    state.observe(angles, travel, positions, forces, tip, axis)
    assert state.engaged[:, :2].all()
