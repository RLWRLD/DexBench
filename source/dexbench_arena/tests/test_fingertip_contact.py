# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Fingertip contact recording: link-frame force split, row order, wiring and refusals."""

import math
import re
from types import SimpleNamespace

import pytest


class _Scene(dict):
    def __init__(self, robot, sensor):
        super().__init__(robot=robot)
        self.sensors = {"fingertip_contact": sensor}


def _env(robot, sensor):
    return SimpleNamespace(scene=_Scene(robot, sensor), recorder_manager=SimpleNamespace(get_ep_meta=dict))


class _Robot:
    def __init__(self, body_names, quat_xyzw):
        import torch

        self.body_names = list(body_names)
        self.data = SimpleNamespace(body_link_quat_w=SimpleNamespace(torch=torch.as_tensor(quat_xyzw, dtype=torch.float32)))

    def find_bodies(self, names, preserve_order=False):
        assert preserve_order
        return [self.body_names.index(name) for name in names], list(names)


def _sensor(sensed_names, net_w, friction_w):
    """Isaac Lab's Newton contact sensor with ``track_friction_forces=True``, as the recorder reads it."""
    data = SimpleNamespace(net_forces_w=SimpleNamespace(torch=net_w), net_friction_forces_w=SimpleNamespace(torch=friction_w))
    return SimpleNamespace(sensor_names=list(sensed_names), data=data)


@pytest.mark.kit
def test_records_link_frame_split_in_declared_rows(simulation_app):
    import torch

    from dexbench_arena.embodiments.fingertip_contact import FingertipContactRecorder, FingertipContactRecorderCfg

    declared = ["left_pad", "left_distal", "right_pad"]
    sensed = ["right_pad", "left_pad", "left_distal"]
    robot_bodies = ["base", "left_distal", "left_pad", "right_pad"]
    s = math.sqrt(0.5)
    identity = [0.0, 0.0, 0.0, 1.0]
    yaw90 = [0.0, 0.0, s, s]
    quat = [[identity, identity, yaw90, identity]] * 2

    net_w = torch.zeros(2, 3, 3)
    friction_w = torch.zeros(2, 3, 3)
    net_w[1, 1] = torch.tensor([1.0, -3.0, 0.0])
    friction_w[1, 1] = torch.tensor([1.0, 0.0, 0.0])
    net_w[0, 0] = torch.tensor([0.5, 0.0, 2.0])
    friction_w[0, 0] = torch.tensor([0.5, 0.0, 0.0])

    env = _env(_Robot(robot_bodies, quat), _sensor(sensed, net_w, friction_w))
    recorder = FingertipContactRecorder(FingertipContactRecorderCfg(body_names=declared), env)
    key, value = recorder.record_post_step()

    assert key == "fingertip_contact"
    assert set(value) == {"net_force", "normal_force", "tangential_force"}
    for tensor in value.values():
        assert tensor.shape == (2, 3, 3)
    torch.testing.assert_close(value["net_force"][1, 0], torch.tensor([-3.0, -1.0, 0.0]))
    torch.testing.assert_close(value["tangential_force"][1, 0], torch.tensor([0.0, -1.0, 0.0]))
    torch.testing.assert_close(value["normal_force"][1, 0], torch.tensor([-3.0, 0.0, 0.0]))
    torch.testing.assert_close(value["net_force"][0, 2], torch.tensor([0.5, 0.0, 2.0]))
    torch.testing.assert_close(value["normal_force"][0, 2], torch.tensor([0.0, 0.0, 2.0]))
    torch.testing.assert_close(value["net_force"], value["normal_force"] + value["tangential_force"])
    torch.testing.assert_close(value["net_force"][1, 0].norm(), net_w[1, 1].norm())
    assert env.recorder_manager.get_ep_meta()["fingertip_contact"]["body_names"] == declared


@pytest.mark.kit
def test_a_zero_net_force_keeps_its_friction_split(simulation_app):
    """Two contacts can cancel to zero net force and still carry friction."""
    import torch

    from dexbench_arena.embodiments.fingertip_contact import FingertipContactRecorder, FingertipContactRecorderCfg

    sensor = _sensor(["pad"], torch.zeros(1, 1, 3), torch.tensor([[[0.3, 0.0, 0.0]]]))
    env = _env(_Robot(["pad"], [[[0.0, 0.0, 0.0, 1.0]]]), sensor)
    _, value = FingertipContactRecorder(FingertipContactRecorderCfg(body_names=["pad"]), env).record_post_step()
    torch.testing.assert_close(value["net_force"][0, 0], torch.zeros(3))
    torch.testing.assert_close(value["tangential_force"][0, 0], torch.tensor([0.3, 0.0, 0.0]))
    torch.testing.assert_close(value["normal_force"][0, 0], torch.tensor([-0.3, 0.0, 0.0]))


@pytest.mark.kit
def test_refuses_a_sensor_without_the_friction_sum(simulation_app):
    import torch

    from dexbench_arena.embodiments.fingertip_contact import FingertipContactRecorder, FingertipContactRecorderCfg

    data = SimpleNamespace(net_forces_w=SimpleNamespace(torch=torch.zeros(1, 1, 3)), net_friction_forces_w=None)
    env = _env(_Robot(["pad"], [[[0.0, 0.0, 0.0, 1.0]]]), SimpleNamespace(sensor_names=["pad"], data=data))
    recorder = FingertipContactRecorder(FingertipContactRecorderCfg(body_names=["pad"]), env)
    with pytest.raises(RuntimeError, match="track_friction_forces"):
        recorder.record_post_step()


@pytest.mark.kit
def test_refuses_a_declared_body_the_sensor_does_not_sense(simulation_app):
    import torch

    from dexbench_arena.embodiments.fingertip_contact import FingertipContactRecorder, FingertipContactRecorderCfg

    env = _env(
        _Robot(["pad", "distal"], [[[0.0, 0.0, 0.0, 1.0]] * 2]),
        _sensor(["pad"], torch.zeros(1, 1, 3), torch.zeros(1, 1, 3)),
    )
    recorder = FingertipContactRecorder(FingertipContactRecorderCfg(body_names=["pad", "distal"]), env)
    with pytest.raises(RuntimeError, match="distal"):
        recorder.record_post_step()


@pytest.mark.kit
def test_embodiment_adds_the_sensor_and_its_recorder(simulation_app):
    from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg

    from dexbench_arena.embodiments.fingertip_contact import FingertipContactRecorderCfg
    from dexbench_arena.embodiments.rby1_wuji2.constants import (
        RBY1_WUJI2_FINGERTIP_DISTAL_BODY_NAMES,
        RBY1_WUJI2_FINGERTIP_PAD_BODY_NAMES,
    )
    from dexbench_arena.embodiments.rby1_wuji2.rby1_wuji2 import DexBenchRby1Wuji2JointEmbodiment

    embodiment = DexBenchRby1Wuji2JointEmbodiment()
    declared = embodiment.scene_config.fingertip_contact
    embodiment.enable_fingertip_contact_recording()
    assert embodiment.scene_config.fingertip_contact is declared

    names = RBY1_WUJI2_FINGERTIP_PAD_BODY_NAMES + RBY1_WUJI2_FINGERTIP_DISTAL_BODY_NAMES
    assert len(names) == 20 and names[0] == "left_thumb_tip_sensor_frame" and names[-1] == "right_pinky_distal"
    sensor = embodiment.scene_config.fingertip_contact
    pattern = re.compile(sensor.prim_path.replace("{ENV_REGEX_NS}", r"/World/envs/env_\d+"))
    assert all(pattern.fullmatch(f"/World/envs/env_3/Robot/{name}") for name in names)
    assert not pattern.fullmatch("/World/envs/env_0/Robot/right_index_finger_middle")
    assert not sensor.filter_prim_paths_expr

    demo = ActionStateRecorderManagerCfg()
    demo.record_variation_samples = SimpleNamespace(marker=True)
    metric_only = SimpleNamespace()
    env_cfg = _newton_cfg(embodiment, demo_recorder_config=demo, mimic_recorder_config=demo, recorders=metric_only)
    embodiment.configure_env_recorders(env_cfg)
    demo = env_cfg.demo_recorder_config
    assert env_cfg.mimic_recorder_config is demo
    assert isinstance(demo, ActionStateRecorderManagerCfg)
    assert isinstance(demo.record_post_step_fingertip_contact, FingertipContactRecorderCfg)
    assert demo.record_post_step_fingertip_contact.body_names == list(names)
    assert env_cfg.scene.fingertip_contact.track_friction_forces
    assert demo.record_variation_samples.marker
    copied = demo.copy()
    assert isinstance(copied.record_post_step_fingertip_contact, FingertipContactRecorderCfg)
    assert copied.record_post_step_processed_actions is not None
    assert not hasattr(metric_only, "record_post_step_fingertip_contact")
    env_cfg.recorders = ActionStateRecorderManagerCfg()
    embodiment.configure_env_recorders(env_cfg)
    assert isinstance(env_cfg.recorders.record_post_step_fingertip_contact, FingertipContactRecorderCfg)


def _env_cfg(physics, embodiment, **recorders):
    return SimpleNamespace(
        sim=SimpleNamespace(physics=physics),
        scene=SimpleNamespace(fingertip_contact=embodiment.scene_config.fingertip_contact),
        **recorders,
    )


def _newton_cfg(embodiment, **recorders):
    from isaaclab_newton.physics.newton_manager_cfg import NewtonCfg

    return _env_cfg(NewtonCfg(), embodiment, **recorders)


@pytest.mark.kit
def test_deformables_are_left_out(simulation_app, capsys):
    """Coupled MJWarp + VBD refuses contact sensors, and its soft-contact forces need a converged solve."""
    from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
    from isaaclab_contrib.custom_coupling.newton_manager_cfg import CoupledMJWarpVBDSolverCfg
    from isaaclab_newton.physics.newton_manager_cfg import NewtonCfg

    from dexbench_arena.embodiments.rby1_wuji2.rby1_wuji2 import DexBenchRby1Wuji2JointEmbodiment

    def coupled_cfg(embodiment):
        return _env_cfg(
            NewtonCfg(solver_cfg=CoupledMJWarpVBDSolverCfg()),
            embodiment,
            demo_recorder_config=ActionStateRecorderManagerCfg(),
            observations=embodiment.observation_config,
        )

    embodiment = DexBenchRby1Wuji2JointEmbodiment()
    embodiment.enable_fingertip_contact_recording()
    env_cfg = coupled_cfg(embodiment)
    assert env_cfg.observations.policy.fingertip_contact_force is not None
    embodiment.configure_env_recorders(env_cfg)
    assert env_cfg.scene.fingertip_contact is None
    assert env_cfg.observations.policy.fingertip_contact_force is None
    assert env_cfg.observations.policy.hand_joint_state is not None
    assert not hasattr(env_cfg.demo_recorder_config, "record_post_step_fingertip_contact")
    assert "--record_fingertip_contact is off" in capsys.readouterr().out

    required = DexBenchRby1Wuji2JointEmbodiment()
    required.enable_fingertip_contact_recording(required=True)
    with pytest.raises(ValueError, match="deformables"):
        required.configure_env_recorders(coupled_cfg(required))


@pytest.mark.kit
def test_an_environment_that_couples_the_solver_leaves_it_out(simulation_app, capsys):
    """An environment's Newton hook can put the scene on coupled MJWarp + VBD (a polybag): the sensor still goes."""
    from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
    from isaaclab_contrib.custom_coupling.newton_manager_cfg import CoupledMJWarpVBDSolverCfg
    from isaaclab_newton.physics import MJWarpSolverCfg
    from isaaclab_newton.physics.newton_manager_cfg import NewtonCfg

    from dexbench_arena.embodiments.rby1_wuji2.rby1_wuji2 import DexBenchRby1Wuji2JointEmbodiment
    from dexbench_arena.environments.manipulation.scene_environment_base import (
        DexBenchEnvironmentCfg,
        _with_dexbench_settings,
    )

    class CouplingEnvironment:
        def apply_newton_sim_settings(self, env_cfg, cfg):
            physics = env_cfg.sim.physics
            physics.solver_cfg = CoupledMJWarpVBDSolverCfg(rigid_solver_cfg=physics.solver_cfg)

    embodiment = DexBenchRby1Wuji2JointEmbodiment()
    embodiment.enable_fingertip_contact_recording()
    env_cfg = _env_cfg(
        NewtonCfg(solver_cfg=MJWarpSolverCfg(nconmax=1, njmax=1)),
        embodiment,
        demo_recorder_config=ActionStateRecorderManagerCfg(),
        observations=embodiment.observation_config,
    )
    settings = SimpleNamespace(cfg=DexBenchEnvironmentCfg(), embodiment=embodiment, output_base_dir=None,
                               factory=CouplingEnvironment(), replayed_variation_samples=None)
    _with_dexbench_settings(None, settings)(env_cfg)
    assert isinstance(env_cfg.sim.physics.solver_cfg, CoupledMJWarpVBDSolverCfg)
    assert env_cfg.scene.fingertip_contact is None
    assert not hasattr(env_cfg.demo_recorder_config, "record_post_step_fingertip_contact")
    assert "--record_fingertip_contact is off: fingertip contact force is not recorded with deformables" in (
        capsys.readouterr().out
    )


@pytest.mark.kit
def test_recording_is_off_unless_enabled(simulation_app):
    from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg

    from dexbench_arena.embodiments.rby1_wuji2.rby1_wuji2 import DexBenchRby1Wuji2JointEmbodiment

    embodiment = DexBenchRby1Wuji2JointEmbodiment()
    demo = ActionStateRecorderManagerCfg()
    embodiment.configure_env_recorders(_newton_cfg(embodiment, demo_recorder_config=demo))
    assert not hasattr(demo, "record_post_step_fingertip_contact")
    assert embodiment.scene_config.fingertip_contact is not None


@pytest.mark.kit
def test_episode_metadata_names_the_rows(simulation_app):
    from dexbench_arena.embodiments.fingertip_contact import FingertipContactRecorder, FingertipContactRecorderCfg

    import torch

    robot = _Robot(["a", "b"], [[[0.0, 0.0, 0.0, 1.0]] * 2])
    sensor = _sensor(["a", "b"], torch.zeros(1, 2, 3), torch.zeros(1, 2, 3))
    env = _env(robot, sensor)
    FingertipContactRecorder(FingertipContactRecorderCfg(body_names=["a", "b"]), env).record_post_step()
    meta = env.recorder_manager.get_ep_meta()["fingertip_contact"]
    assert meta["body_names"] == ["a", "b"]
    assert meta["fields"] == ["net_force", "normal_force", "tangential_force"]
    assert meta["frame"] == "body link frame"
    assert "last physics substep" in meta["sampled"]


class _Embodiment:
    def __init__(self):
        self.enabled = False
        self.required = None

    def enable_fingertip_contact_recording(self, required=False):
        self.enabled = True
        self.required = required


def _configure(setting, embodiment=None):
    from dexbench_arena.environments.fingertip_contact import configure_environment_fingertip_contact

    environment = SimpleNamespace(embodiment=embodiment if embodiment is not None else _Embodiment())
    configure_environment_fingertip_contact(environment, SimpleNamespace(record_fingertip_contact=setting))
    return environment.embodiment


def test_recording_is_asked_for_by_default_and_required_when_named():
    default = _configure(None)
    assert default.enabled and default.required is False
    assert _configure(True).required is True
    assert not _configure(False).enabled


def test_default_skips_an_embodiment_without_fingertips_but_an_explicit_request_fails(capsys):
    assert not hasattr(_configure(None, embodiment=SimpleNamespace()), "enabled")
    assert "--record_fingertip_contact is off" in capsys.readouterr().out
    with pytest.raises(ValueError, match="fingertip contact bodies"):
        _configure(True, embodiment=SimpleNamespace())
    from dexbench_arena.environments.fingertip_contact import configure_environment_fingertip_contact

    configure_environment_fingertip_contact(SimpleNamespace(embodiment=None), SimpleNamespace(record_fingertip_contact=None))


def test_cli_parser_declares_the_switch_once():
    import argparse

    from dexbench_arena.environments.fingertip_contact import add_fingertip_contact_cli_args

    parser = argparse.ArgumentParser()
    add_fingertip_contact_cli_args(parser)
    add_fingertip_contact_cli_args(parser)
    assert parser.parse_args([]).record_fingertip_contact is None
    assert parser.parse_args(["--record_fingertip_contact"]).record_fingertip_contact is True
    assert parser.parse_args(["--no-record_fingertip_contact"]).record_fingertip_contact is False
