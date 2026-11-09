# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Complete joint commands across Pink and direct-joint recording paths."""

from types import SimpleNamespace

import pytest
import torch

from dexbench_arena.embodiments.rby1_wuji2.constants import RBY1_WUJI2_JOINT_ACTION_NAMES


def test_recorder_config_import_after_configclass_submodule(simulation_app, monkeypatch):
    """Import order must not turn the configclass decorator into a module."""
    import importlib
    import isaaclab.utils

    decorator_module = importlib.import_module("isaaclab.utils.configclass")
    monkeypatch.setitem(vars(isaaclab.utils), "configclass", decorator_module)
    assert isaaclab.utils.configclass is decorator_module
    from dexbench_arena.embodiments.rby1_wuji2 import recorders

    recorders = importlib.reload(recorders)
    cfg = recorders.Wuji2JointTargetsRecorderCfg()
    assert cfg.class_type is recorders.Wuji2JointTargetsRecorder


class Robot:
    def __init__(self, names, targets):
        self.names = names
        self.actuators = SimpleNamespace(target_command=SimpleNamespace(position=SimpleNamespace(torch=targets)))
        self.data = SimpleNamespace(joint_pos=torch.full_like(targets, -99.0))

    def find_joints(self, requested, preserve_order=False):
        assert preserve_order
        found = [name for name in requested if name in self.names]
        return [self.names.index(name) for name in found], found


@pytest.mark.parametrize("scrambled", [False, True])
@pytest.mark.parametrize("raw_width", [54, 62], ids=["pink", "rrc"])
def test_records_complete_targets_in_public_order(simulation_app, tmp_path, scrambled, raw_width):
    from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler
    from dexbench_arena.embodiments.rby1_wuji2.recorders import Wuji2JointTargetsRecorder, Wuji2JointTargetsRecorderCfg

    public_names = list(RBY1_WUJI2_JOINT_ACTION_NAMES)
    names = public_names + ["wheel_fr", "wheel_fl", "wheel_rr", "wheel_rl"]
    if scrambled:
        names = names[::2][::-1] + names[1::2]
    expected = torch.arange(124, dtype=torch.float32).reshape(2, 62) / 100
    targets = torch.zeros(2, len(names))
    for i, name in enumerate(public_names):
        targets[:, names.index(name)] = expected[:, i]
    robot = Robot(names, targets)
    raw_actions = torch.zeros(2, raw_width)
    env = SimpleNamespace(scene={"robot": robot}, action_manager=SimpleNamespace(action=raw_actions))
    recorder = Wuji2JointTargetsRecorder(Wuji2JointTargetsRecorderCfg(), env)
    key, first = recorder.record_post_step()
    assert key == "processed_actions"
    torch.testing.assert_close(first, expected)
    assert env.action_manager.action.shape == (2, raw_width)

    for name in public_names[-6:]:
        targets[:, names.index(name)] += 0.25
    _, after_reset = recorder.record_post_step()
    torch.testing.assert_close(first, expected)
    expected_after_reset = expected.clone()
    expected_after_reset[:, -6:] += 0.25
    torch.testing.assert_close(after_reset, expected_after_reset)

    episode = EpisodeData()
    for frame in (first, after_reset):
        episode.add("actions", raw_actions[0])
        episode.add(key, frame[0])
    handler = HDF5DatasetFileHandler()
    path = str(tmp_path / "recording.hdf5")
    handler.create(path, env_name="wuji2_recording_test")
    episode.pre_export()
    handler.write_episode(episode)
    handler.close()
    handler.open(path)
    loaded = handler.load_episode("demo_0", "cpu")
    assert loaded.data["actions"].shape == (2, raw_width)
    torch.testing.assert_close(loaded.data[key], torch.stack([expected[0], expected_after_reset[0]]))
    handler.close()


def test_direct_joint_targets_match_existing_processed_actions(simulation_app):
    from isaaclab.envs.mdp.recorders.recorders import PostStepProcessedActionsRecorder
    from isaaclab.envs.mdp.recorders.recorders_cfg import PostStepProcessedActionsRecorderCfg
    from dexbench_arena.embodiments.rby1_wuji2.recorders import Wuji2JointTargetsRecorder, Wuji2JointTargetsRecorderCfg

    command = torch.linspace(-0.5, 0.5, 62).reshape(1, 62)
    robot = Robot(list(RBY1_WUJI2_JOINT_ACTION_NAMES), command.clone())
    env = SimpleNamespace(
        scene={"robot": robot},
        action_manager=SimpleNamespace(
            active_terms=["joint_pos"], get_term=lambda _: SimpleNamespace(processed_actions=command)
        ),
    )
    old = PostStepProcessedActionsRecorder(PostStepProcessedActionsRecorderCfg(), env)
    new = Wuji2JointTargetsRecorder(Wuji2JointTargetsRecorderCfg(), env)
    torch.testing.assert_close(old.record_post_step()[1], new.record_post_step()[1])


def test_missing_torso_joint_fails_instead_of_padding(simulation_app):
    from dexbench_arena.embodiments.rby1_wuji2.recorders import Wuji2JointTargetsRecorder, Wuji2JointTargetsRecorderCfg

    robot = Robot(list(RBY1_WUJI2_JOINT_ACTION_NAMES[:-1]), torch.zeros(1, 61))
    recorder = Wuji2JointTargetsRecorder(Wuji2JointTargetsRecorderCfg(), SimpleNamespace(scene={"robot": robot}))
    with pytest.raises(ValueError, match="torso_5"):
        recorder.record_post_step()


@pytest.mark.parametrize("cameras", [False, True])
def test_config_preserves_camera_and_mimic_terms(simulation_app, cameras):
    from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
    from isaaclab_arena.terms.recorders import ArenaEnvRecorderManagerCfg
    from dexbench_arena.embodiments.rby1_wuji2.recorders import configure_wuji2_recorders, Wuji2JointTargetsRecorder

    demo = ArenaEnvRecorderManagerCfg() if cameras else None
    mimic = ActionStateRecorderManagerCfg()
    extra_term = object()
    mimic.task_specific_term = extra_term
    active_recorders = object()
    cfg = SimpleNamespace(demo_recorder_config=demo, mimic_recorder_config=mimic, recorders=active_recorders)
    configure_wuji2_recorders(cfg, embodiment="dexbench_rby1_wuji2")
    assert cfg.recorders is active_recorders
    assert cfg.mimic_recorder_config is mimic
    assert mimic.task_specific_term is extra_term
    for recorder_cfg in (cfg.demo_recorder_config, cfg.mimic_recorder_config):
        assert recorder_cfg.record_post_step_processed_actions.class_type is Wuji2JointTargetsRecorder
    if cameras:
        assert cfg.demo_recorder_config is demo
        assert demo.record_pre_step_flat_camera_observations is not None
    else:
        assert not hasattr(cfg.demo_recorder_config, "record_pre_step_flat_camera_observations")


def test_mimic_fallback_gets_complete_joint_recorder(simulation_app):
    from dexbench_arena.embodiments.rby1_wuji2.recorders import configure_wuji2_recorders, Wuji2JointTargetsRecorder

    cfg = SimpleNamespace(demo_recorder_config=None, mimic_recorder_config=None)
    configure_wuji2_recorders(cfg, embodiment="dexbench_rby1_wuji2")
    assert cfg.mimic_recorder_config.record_post_step_processed_actions.class_type is Wuji2JointTargetsRecorder


def test_evaluation_recorder_is_overridden_only_when_it_records_per_step(simulation_app):
    """--record_dataset brings the generic processed-actions term; a metric-only recorder stays metric-only."""
    from isaaclab_arena.terms.recorders import ArenaEnvRecorderManagerCfg
    from dexbench_arena.embodiments.rby1_wuji2.recorders import configure_wuji2_recorders, Wuji2JointTargetsRecorder

    per_step = ArenaEnvRecorderManagerCfg()
    cfg = SimpleNamespace(demo_recorder_config=None, mimic_recorder_config=None, recorders=per_step)
    configure_wuji2_recorders(cfg, embodiment="dexbench_rby1_wuji2")
    assert cfg.recorders is per_step
    assert per_step.record_post_step_processed_actions.class_type is Wuji2JointTargetsRecorder
    assert per_step.record_pre_step_flat_camera_observations is not None

    metric_only = SimpleNamespace(success=object())
    cfg = SimpleNamespace(demo_recorder_config=None, mimic_recorder_config=None, recorders=metric_only)
    configure_wuji2_recorders(cfg, embodiment="dexbench_rby1_wuji2")
    assert vars(metric_only) == {"success": metric_only.success}
