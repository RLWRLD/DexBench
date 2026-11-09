# SPDX-License-Identifier: Apache-2.0
"""Mimic with action noise executes the noisy command and records the clean one beside it; a generation
run sets its noise and whether it keeps failed attempts from its config."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("isaaclab.sim", exc_type=ImportError)
import torch  # noqa: E402

from dexbench_arena.embodiments.common.clean_action_labels import CleanActionsRecorder  # noqa: E402
from dexbench_arena.embodiments.rby1_wuji2.constants import RBY1_WUJI2_JOINT_ACTION_NAMES  # noqa: E402
from dexbench_arena.embodiments.rby1_wuji2.recorders import Wuji2CleanJointTargetsRecorder  # noqa: E402
from dexbench_arena.embodiments.rby1_wuji2.rby1_wuji2 import (  # noqa: E402
    Rby1Wuji2MimicEnv,
    Rby1Wuji2PinkInverseKinematicsAction,
)

HAND_WIDTH = 20


class _Term(Rby1Wuji2PinkInverseKinematicsAction):
    """The Wuji2 Pink term without a robot: its "IK" returns the wrist targets it was given,
    and its waist limiter state advances on every solve, as the real one does."""

    action_dim = 14 + 2 * HAND_WIDTH
    hand_joint_dim = 2 * HAND_WIDTH

    def __init__(self, env, executed):
        self._env = env
        self._raw_actions = executed.clone()
        self._waist_ik_previous_target = torch.tensor([[1.0]])
        self._set_task_targets(executed[:, :14])

    def _extract_controlled_frame_poses(self, actions):
        return actions[:, :14]

    def _transform_poses_to_base_link_frame(self, poses):
        return poses

    def _set_task_targets(self, poses):
        self.targets = poses.clone()

    def _compute_ik_solutions(self):
        self._waist_ik_previous_target = self._waist_ik_previous_target + 1
        return self.targets.clone()


def _generate_one_step(noise: float):
    """target_eef_pose_to_action, then env.step's action processing and pre-step recording."""
    torch.manual_seed(0)
    mimic = object.__new__(Rby1Wuji2MimicEnv)
    mimic.cfg = SimpleNamespace(subtask_configs={"right": [SimpleNamespace(action_noise=noise)]})
    pose = torch.eye(4)
    pose[:3, 3] = torch.tensor([0.4, 0.1, 1.0])
    hands = {side: torch.full((HAND_WIDTH,), 0.2) for side in ("left", "right")}
    executed = mimic.target_eef_pose_to_action(
        {"left": pose, "right": pose}, hands, {"left": noise, "right": noise}, env_id=0
    ).unsqueeze(0)
    term = _Term(mimic, executed)
    mimic.action_manager = SimpleNamespace(action=executed, active_terms=["upper_body_ik"], get_term=lambda name: term)
    term._take_clean_actions(executed)
    solution = term._solve_for(term._clean_actions) if term._clean_actions is not None else None
    _, recorded = CleanActionsRecorder.record_pre_step(SimpleNamespace(_env=mimic))
    term._controlled_joint_ids = list(range(14))
    term.clean_processed_actions = solution
    names = list(RBY1_WUJI2_JOINT_ACTION_NAMES)
    targets = torch.zeros(1, len(names))
    mimic.scene = {"robot": SimpleNamespace(
        find_joints=lambda requested, preserve_order: (list(range(len(requested))), list(requested)),
        actuators=SimpleNamespace(target_command=SimpleNamespace(position=SimpleNamespace(torch=targets))),
    )}
    _, recorded_targets = Wuji2CleanJointTargetsRecorder.record_post_step(
        SimpleNamespace(_env=mimic, joint_names=names))
    return executed, recorded, solution, term, recorded_targets


def test_noise_is_executed_and_the_clean_target_is_recorded():
    executed, recorded, solution, term, targets = _generate_one_step(noise=0.01)
    assert not torch.equal(executed, recorded)
    assert torch.equal(recorded[:, 14:], executed[:, 14:])
    assert torch.equal(solution, recorded[:, :14])
    assert torch.equal(term.targets, executed[:, :14])
    assert torch.equal(term._waist_ik_previous_target, torch.tensor([[1.0]]))
    assert torch.equal(targets[:, :14], solution) and not targets[:, 14:].any()

    executed, recorded, solution, _, targets = _generate_one_step(noise=0.0)
    assert recorded is None and targets is None and solution is None


def test_a_run_sets_every_subtasks_noise_and_the_retired_variable_is_refused(monkeypatch):
    from dexbench_arena.environments.manipulation.scene_environment_base import (
        DexBenchEnvironmentCfg,
        mimic_action_noise_callback,
    )

    composed = SimpleNamespace(subtask_configs={"right": [SimpleNamespace(action_noise=0.0)] * 2,
                                                "left": [SimpleNamespace(action_noise=0.0)]})
    previous = lambda cfg: cfg  # noqa: E731
    assert mimic_action_noise_callback(previous, DexBenchEnvironmentCfg()) is previous
    noisy = mimic_action_noise_callback(previous, DexBenchEnvironmentCfg(mimic_action_noise=0.005))(composed)
    assert [s.action_noise for subtasks in noisy.subtask_configs.values() for s in subtasks] == [0.005] * 3
    with pytest.raises(ValueError, match="finite value >= 0"):
        mimic_action_noise_callback(previous, DexBenchEnvironmentCfg(mimic_action_noise=-1.0))
    monkeypatch.setenv("DEXBENCH_MIMIC_ACTION_NOISE", "0.005")
    with pytest.raises(ValueError, match="--mimic_action_noise"):
        mimic_action_noise_callback(previous, DexBenchEnvironmentCfg())


def test_a_run_keeps_or_drops_failed_attempts_and_the_retired_variable_is_refused(monkeypatch):
    from dexbench_arena.environments.manipulation.scene_environment_base import (
        DexBenchEnvironmentCfg,
        mimic_keep_failed_callback,
    )

    def composed(keep):
        return SimpleNamespace(datagen_config=SimpleNamespace(generation_keep_failed=keep))

    previous = lambda cfg: cfg  # noqa: E731
    assert mimic_keep_failed_callback(previous, DexBenchEnvironmentCfg()) is previous
    keep = mimic_keep_failed_callback(previous, DexBenchEnvironmentCfg(mimic_keep_failed=True))
    drop = mimic_keep_failed_callback(previous, DexBenchEnvironmentCfg(mimic_keep_failed=False))
    assert keep(composed(False)).datagen_config.generation_keep_failed is True
    assert drop(composed(True)).datagen_config.generation_keep_failed is False
    assert keep(SimpleNamespace()) is not None
    monkeypatch.setenv("DEXBENCH_MIMIC_KEEP_FAILED", "1")
    with pytest.raises(ValueError, match="--mimic_keep_failed"):
        mimic_keep_failed_callback(previous, DexBenchEnvironmentCfg())
