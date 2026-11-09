# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Joint torque recording: Isaac Lab's applied effort in joint order, wiring, metadata and the switch."""

from types import SimpleNamespace

import pytest


class _Robot:
    def __init__(self, joint_names, applied_effort, computed_effort=None):
        self.joint_names = list(joint_names)
        computed = applied_effort if computed_effort is None else computed_effort
        self.actuators = SimpleNamespace(
            applied_effort=SimpleNamespace(torch=applied_effort), computed_effort=SimpleNamespace(torch=computed)
        )


def _env(robot):
    return SimpleNamespace(scene={"robot": robot}, recorder_manager=SimpleNamespace(get_ep_meta=dict))


def _recorder_module():
    """The recorder needs Isaac Lab's manager classes, not a running Kit."""
    pytest.importorskip("isaaclab.managers", exc_type=ImportError)
    from dexbench_arena.embodiments import joint_torque

    return joint_torque


def test_records_applied_effort_in_joint_order():
    import torch

    module = _recorder_module()
    JointTorqueRecorder, JointTorqueRecorderCfg = module.JointTorqueRecorder, module.JointTorqueRecorderCfg

    names = ["torso_0", "right_arm_0", "right_thumb_finger_joint1"]
    effort = torch.tensor([[1.0, -100.0, 5.0], [0.0, 2.5, -5.0]])
    env = _env(_Robot(names, effort))
    key, value = JointTorqueRecorder(JointTorqueRecorderCfg(), env).record_post_step()

    assert key == "joint_torque" and list(value) == ["applied_effort"]
    torch.testing.assert_close(value["applied_effort"], effort)
    effort.zero_()
    assert value["applied_effort"][0, 1] == -100.0
    meta = env.recorder_manager.get_ep_meta()["joint_torque"]
    assert meta["joint_names"] == names
    assert meta["fields"] == ["applied_effort"]
    assert "applied_effort" in meta["source"] and "estimate" in meta["source"]


def test_a_non_finite_estimate_is_recorded_as_nan_not_as_the_limit():
    import torch

    module = _recorder_module()
    nan = float("nan")
    computed = torch.tensor([[250.0, 0.0, 0.0], [nan, nan, nan]])
    applied = torch.tensor([[100.0, 0.0, 0.0], [-100.0, -1.0, -3.0]])
    env = _env(_Robot(["right_arm_0", "wheel_fl", "right_thumb_cmc_abd"], applied, computed))
    _, value = module.JointTorqueRecorder(module.JointTorqueRecorderCfg(), env).record_post_step()
    recorded = value["applied_effort"]
    torch.testing.assert_close(recorded[0], torch.tensor([100.0, 0.0, 0.0]))
    assert torch.isnan(recorded[1]).all()


@pytest.mark.kit
def test_embodiment_adds_the_recorder_term(simulation_app):
    from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg

    from dexbench_arena.embodiments.joint_torque import JointTorqueRecorderCfg
    from dexbench_arena.embodiments.rby1_wuji2.rby1_wuji2 import DexBenchRby1Wuji2JointEmbodiment

    embodiment = DexBenchRby1Wuji2JointEmbodiment()
    embodiment.enable_joint_torque_recording()

    demo = ActionStateRecorderManagerCfg()
    demo.record_variation_samples = SimpleNamespace(marker=True)
    metric_only = SimpleNamespace()
    env_cfg = SimpleNamespace(demo_recorder_config=demo, mimic_recorder_config=demo, recorders=metric_only)
    embodiment.configure_env_recorders(env_cfg)
    demo = env_cfg.demo_recorder_config
    assert env_cfg.mimic_recorder_config is demo
    assert isinstance(demo, ActionStateRecorderManagerCfg)
    assert isinstance(demo.record_post_step_joint_torque, JointTorqueRecorderCfg)
    assert demo.record_post_step_joint_torque.asset_name == "robot"
    assert demo.record_variation_samples.marker
    copied = demo.copy()
    assert isinstance(copied.record_post_step_joint_torque, JointTorqueRecorderCfg)
    assert copied.record_post_step_processed_actions is not None
    assert not hasattr(metric_only, "record_post_step_joint_torque")
    env_cfg.recorders = ActionStateRecorderManagerCfg()
    embodiment.configure_env_recorders(env_cfg)
    assert isinstance(env_cfg.recorders.record_post_step_joint_torque, JointTorqueRecorderCfg)


@pytest.mark.kit
def test_recording_is_off_unless_enabled(simulation_app):
    from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg

    from dexbench_arena.embodiments.rby1_wuji2.rby1_wuji2 import DexBenchRby1Wuji2JointEmbodiment

    demo = ActionStateRecorderManagerCfg()
    DexBenchRby1Wuji2JointEmbodiment().configure_env_recorders(SimpleNamespace(demo_recorder_config=demo))
    assert not hasattr(demo, "record_post_step_joint_torque")


class _Embodiment:
    name = "with_joint_torque"

    def __init__(self):
        self.enabled = False

    def enable_joint_torque_recording(self):
        self.enabled = True


def _configure(setting, embodiment=None):
    from dexbench_arena.environments.joint_torque import configure_environment_joint_torque

    environment = SimpleNamespace(embodiment=embodiment if embodiment is not None else _Embodiment())
    configure_environment_joint_torque(environment, SimpleNamespace(record_joint_torque=setting))
    return environment.embodiment


def test_recording_is_on_by_default_where_declared():
    assert _configure(None).enabled
    assert _configure(True).enabled
    assert not _configure(False).enabled


def test_default_skips_other_embodiments_but_an_explicit_request_fails(capsys):
    other = SimpleNamespace(name="other_robot")
    _configure(None, embodiment=other)
    assert "--record_joint_torque is off" in capsys.readouterr().out
    with pytest.raises(ValueError, match="other_robot"):
        _configure(True, embodiment=other)
    _configure(False, embodiment=other)


def test_without_an_embodiment_only_an_explicit_request_fails(capsys):
    from dexbench_arena.environments.joint_torque import configure_environment_joint_torque

    for setting in (None, False):
        configure_environment_joint_torque(SimpleNamespace(embodiment=None), SimpleNamespace(record_joint_torque=setting))
    assert capsys.readouterr().out == ""
    with pytest.raises(ValueError, match="has none"):
        configure_environment_joint_torque(SimpleNamespace(embodiment=None), SimpleNamespace(record_joint_torque=True))


def test_cli_parser_declares_the_switch_once():
    import argparse

    from dexbench_arena.environments.joint_torque import add_joint_torque_cli_args

    parser = argparse.ArgumentParser()
    add_joint_torque_cli_args(parser)
    add_joint_torque_cli_args(parser)
    assert parser.parse_args([]).record_joint_torque is None
    assert parser.parse_args(["--record_joint_torque"]).record_joint_torque is True
    assert parser.parse_args(["--no-record_joint_torque"]).record_joint_torque is False
