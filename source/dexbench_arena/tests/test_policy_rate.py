# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""A policy asked every second control step: when it is asked, what each step commands, and its chunk's pace."""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from isaaclab_arena.policy.policy_base import PolicyBase

from dexbench_arena.evaluation.evaluation_signature import POLICY_SETTINGS_ATTRIBUTE
from dexbench_arena.policy.policy_rate import PolicyRateCfg, PolicyRateScheduler
from dexbench_arena.policy.remote_joint_policy import Codec, RemoteJointPolicy, Transport

JOINTS = ["a", "b"]


@dataclass
class ProbeCfg(PolicyRateCfg):
    policy_device: str = "cpu"
    robot_asset_name: str = "robot"
    joint_action_term: str = "joint_pos"
    policy_variant: str = "probe"
    scheduler: str = "chunk"


class Requests(Transport):
    def __init__(self):
        self.count = 0

    def info(self):
        return None

    def infer(self, request, env_ids):
        self.count += 1
        return self.count


class Ramp(Codec):
    """Chunk n (from 1) holds 10 n + k at its k-th action, so a value names the chunk and the action."""

    joint_names = JOINTS
    action_horizon = 4

    def __init__(self):
        self.observed = []
        self.encoded = []

    def check(self, info):
        pass

    def observe(self, observation):
        self.observed.append(observation["step"])

    def encode(self, observation, env_ids, task_description):
        self.encoded.append(observation["step"])

    def decode(self, response, num_envs):
        chunk = 10.0 * response + np.arange(self.action_horizon, dtype=np.float32)
        return np.repeat(chunk[None, :, None], len(JOINTS), axis=2).repeat(num_envs, axis=0)


class Probe(RemoteJointPolicy, PolicyBase[ProbeCfg]):
    name = "probe"


def probe(policy_hz=None, interpolation="none"):
    codec = Ramp()
    policy = Probe(ProbeCfg(policy_hz=policy_hz, policy_action_interpolation=interpolation), Requests(), codec,
                   action_chunk_length=3)
    policy.set_task_description("probe")
    return policy, codec


def env(control_hz, num_envs=1):
    live = SimpleNamespace(
        num_envs=num_envs,
        step_dt=1.0 / control_hz,
        scene={"robot": SimpleNamespace(joint_names=JOINTS)},
        action_manager=SimpleNamespace(total_action_dim=2, get_term=lambda _: SimpleNamespace(_joint_names=JOINTS)),
    )
    return SimpleNamespace(unwrapped=live)


def run(policy, wrapped, steps, first=0):
    observations = ({"step": step, "policy": {"robot_joint_pos": torch.zeros(1, 2)}} for step in range(first, first + steps))
    return [float(policy.get_action(wrapped, observation)[0, 0]) for observation in observations]


def test_linear_asks_every_second_step_and_reaches_each_answer_from_the_last():
    policy, codec = probe(30, "linear")
    wrapped = env(60.0)

    assert run(policy, wrapped, 8) == [10, 10, 10.5, 11, 11.5, 12, 16, 20]
    assert codec.observed == [0, 2, 4, 6]
    assert codec.encoded == [0, 6]
    stamp = getattr(wrapped.unwrapped, POLICY_SETTINGS_ATTRIBUTE)
    assert (stamp["policy_hz"], stamp["policy_action_interpolation"], stamp["steps_per_query"]) == (30, "linear", 2)

    policy.reset(torch.tensor([0]))
    assert run(policy, wrapped, 4, first=100) == [30, 30, 30.5, 31]
    assert codec.observed[-2:] == [100, 102]


def test_hold_and_every_step():
    held, _ = probe(30, "none")
    assert run(held, env(60.0), 6) == [10, 10, 11, 11, 12, 12]
    every, _ = probe()
    assert run(every, env(30.0), 4) == [10, 11, 12, 20]


@pytest.mark.parametrize("control_hz,policy_hz,interpolation", [(60.0, 25, "linear"), (30.0, 30, "linear"),
                                                                 (60.0, None, "linear"), (60.0, 30, "cubic")])
def test_refused_rates(control_hz, policy_hz, interpolation):
    with pytest.raises(ValueError):
        PolicyRateScheduler(None, control_hz, policy_hz, interpolation, num_envs=1)


def test_one_phase_needs_one_env():
    policy, _ = probe(30, "linear")
    observation = {"step": 0, "policy": {"robot_joint_pos": torch.zeros(2, 2)}}
    with pytest.raises(ValueError, match="one env"):
        policy.get_action(env(60.0, num_envs=2), observation)
