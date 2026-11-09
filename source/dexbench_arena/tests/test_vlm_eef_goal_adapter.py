# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""L1's hands in the torso frame and as changes: what the model reads, what its goals become, what the controller gets."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("isaaclab.utils.math", exc_type=ImportError)
pytest.importorskip("pinocchio", exc_type=ImportError)

from dexbench_arena_vlm_agent import adapters  # noqa: E402
from dexbench_arena_vlm_agent.upstream.action_adapter import InvalidAgentResponse  # noqa: E402

START = {"torso_1": 0.8029, "torso_2": -1.6058, "torso_3": 0.8029,
         "left_arm_0": 0.489, "left_arm_1": 0.419, "left_arm_3": -1.815, "left_arm_4": -0.052, "left_arm_5": -0.524,
         "left_arm_6": 0.384, "right_arm_0": 0.524, "right_arm_1": -0.489, "right_arm_3": -1.867, "right_arm_4": -0.052,
         "right_arm_5": -0.524, "right_arm_6": -0.506}


class _Controller:
    """Records the base-frame wrists of each action."""

    def __init__(self):
        self.wrists = None

    def reset(self, env_ids):
        pass

    def action(self, joint_pos, joint_pos_target, wrists, passed):
        self.wrists = {hand: (position[0].double().numpy(), rotation[0].double().numpy())
                       for hand, (position, rotation) in wrists.items()}
        return torch.zeros(1, len(adapters.JOINT_NAMES))


def _env(monkeypatch):
    names = list(adapters._rby1_wuji2_pink_kinematics()._all_joint_names)
    joint_pos = torch.tensor([[START.get(name, 0.0) for name in names]], dtype=torch.float64)
    term = SimpleNamespace(_joint_ids=[names.index(name) for name in adapters.JOINT_NAMES])
    data = SimpleNamespace(joint_names=names, joint_pos=joint_pos, joint_pos_target=joint_pos.clone(),
                           soft_joint_pos_limits=torch.tensor([[[-3.0, 3.0]] * len(names)], dtype=torch.float64))
    env = SimpleNamespace(num_envs=1, scene={"robot": SimpleNamespace(data=data)}, step_dt=1 / 30, device="cpu",
                          max_episode_length=900, episode_length_buf=torch.zeros(1, dtype=torch.long),
                          action_manager=SimpleNamespace(active_terms=[adapters.JOINT_ACTION_TERM],
                                                         get_term=lambda name: term))
    env.unwrapped = env
    controller = _Controller()
    monkeypatch.setattr(adapters.DexBenchRby1Wuji2JointEmbodiment, "eef_arm_ik_controller", lambda env, profile: controller)
    return env, controller, names


def _adapter(env, **options):
    adapter = adapters.EefGoalAdapter(max_steps=150, hand="shapes", ik_profile="generation", linear_speed=0.10,
                                      angular_speed_deg=90.0, joint_speed=0.6, grasp_s=1.5, **options)
    assert adapter.is_robot_action_space_supported(env)
    return adapter


def _points(state, hand):
    return np.array([state[f"{hand}_{axis}"] for axis in "xyz"])


def _link(env, name):
    """A link's base-frame pose (position, rotation matrix) at the stand-in's joints, from the Pink model directly."""
    kinematics = adapters._rby1_wuji2_pink_kinematics()
    kinematics.update(env.scene["robot"].data.joint_pos[0].numpy())
    pose = kinematics.get_transform_frame_to_world(name)
    return np.array(pose.translation), np.array(pose.rotation)


def test_torso_frame_hands_ride_the_torso_and_reach_the_controller_as_base_frame_wrists(monkeypatch):
    env, controller, names = _env(monkeypatch)
    torso, base = _adapter(env, reference_frame="torso"), _adapter(env)
    before_torso, before_base = torso.state(), base.state()
    env.scene["robot"].data.joint_pos[0, names.index("torso_5")] = 0.5
    after_torso, after_base = torso.state(), base.state()
    for hand in adapters.HANDS:
        np.testing.assert_allclose(_points(after_torso, hand), _points(before_torso, hand), atol=2e-4)
        assert np.linalg.norm(_points(after_base, hand) - _points(before_base, hand)) > 0.05
    assert after_torso["torso_frame_in_base"]["yaw_deg"] == pytest.approx(math.degrees(0.5), abs=0.1)
    assert "torso_frame_in_base" not in after_base
    torso.command_to_action(env, torso.compute_command(torso.hold_goal(after_torso), after_torso))
    for hand in adapters.HANDS:
        position, rotation = _link(env, f"{hand}_wrist_frame")
        np.testing.assert_allclose(controller.wrists[hand][0], position, atol=1e-3)
        np.testing.assert_allclose(controller.wrists[hand][1], rotation, atol=2e-3)


def test_a_change_starts_at_the_measured_hand_and_is_refused_past_its_bounds_or_the_frame_box(monkeypatch):
    env, _, _ = _env(monkeypatch)
    delta = _adapter(env, reference_frame="torso", controller="eef_delta")
    assert "right_dx" in delta.response_model.model_fields and "right_x" not in delta.response_model.model_fields
    state = delta.state()
    measured = _link(env, adapters.TORSO_FRAME_LINK)[1].T @ _link(env, "right_wrist_frame")[1]
    goal = delta.decode_command(delta.response_model(note="n", right_dx=0.05, right_dyaw_deg=30.0, right_grasp=0.4), state)
    right = len(adapters.EEF_AXES) + len(delta._suffixes)
    np.testing.assert_allclose(goal[right:right + 3], _points(state, "right") + [0.05, 0.0, 0.0], atol=2e-4)
    np.testing.assert_allclose(adapters._matrix("right", *goal[right + 3:right + 6]),
                               adapters._matrix_from_turn(0.0, 0.0, 30.0) @ measured, atol=1e-9)
    np.testing.assert_allclose(goal[right + 6:right + 6 + len(delta._suffixes)], 0.4 * delta._shape_vec["claw"])
    np.testing.assert_allclose(goal[:3], _points(state, "left"), atol=2e-4)
    with pytest.raises(InvalidAgentResponse, match="right_dx=0.6 is outside"):
        delta.decode_command(delta.response_model(note="n", right_dx=0.6), state)
    absolute = _adapter(env, reference_frame="torso")
    absolute.decode_command(absolute.response_model(note="n", left_y=0.862, right_y=-0.884, right_z=-0.489,
                                                    left_x=0.796), absolute.state())
    in_base = _adapter(env, controller="eef_delta")
    with pytest.raises(InvalidAgentResponse, match="right_dz=-0.5 would end the grasp point at z="):
        in_base.decode_command(in_base.response_model(note="n", right_dz=-0.5), in_base.state())
    monkeypatch.setitem(adapters.WORKSPACES, "base", (np.array([-0.10, -0.80, 1.0]), np.array([0.95, 0.80, 1.60])))
    state = in_base.state()
    np.testing.assert_allclose(in_base.hold_goal(state)[right:right + 3], _points(state, "right"), atol=2e-4)
    in_base.decode_command(in_base.response_model(note="n", right_dx=0.01), state)
    with pytest.raises(InvalidAgentResponse, match="right_dz=0.01 would end"):
        in_base.decode_command(in_base.response_model(note="n", right_dz=0.01), state)


def test_each_frame_and_goal_kind_gets_its_own_prompt_sentences_and_goal_fields(monkeypatch):
    env, _, _ = _env(monkeypatch)
    assert {"right_x", "right_yaw_deg", "right_grasp_shape"} <= set(_adapter(env).response_model.model_fields)
    for frame in adapters.REFERENCE_FRAMES:
        for controller in adapters.EEF_CONTROLLERS:
            prompt = _adapter(env, reference_frame=frame, controller=controller).system_prompt("recipe", 5.0)
            assert ("in the torso frame" in prompt) == (frame == "torso")
            assert ("about the torso frame's axes" in prompt) == ((frame, controller) == ("torso", "eef_abs"))
            assert ("a change from where the hand is" in prompt) == (controller == "eef_delta")
            assert "`<hand>_grasp_shape`, one of claw, cup" in prompt
            assert "{" not in prompt and "}" not in prompt
