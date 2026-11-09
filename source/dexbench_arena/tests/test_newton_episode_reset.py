# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""One episode lifetime on Newton: the model survives episodes, saved state reaches MuJoCo."""

import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from dexbench_arena.assets import newton_episode_reset as lifetime
from dexbench_arena.embodiments.common.mimic_boundaries import ReplaysSubtaskBoundaries

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts/imitation_learning"


def _newton_env():
    """A stand-in env whose hard reset drops the gain the way Newton's re-finalize does."""
    cfg_module = pytest.importorskip("isaaclab_newton.physics.newton_manager_cfg")
    physics = cfg_module.NewtonCfg()
    model = {"hand_stiffness": 350.0}

    def hard_reset():
        model["hand_stiffness"] = 0.365533

    env = SimpleNamespace(cfg=SimpleNamespace(sim=SimpleNamespace(physics=physics)), sim=SimpleNamespace(reset=hard_reset))
    return env, model


@pytest.mark.parametrize("requested", [None, False, True])
def test_the_hard_reset_keeps_the_newton_model_unless_asked(requested):
    env, model = _newton_env()
    expected = bool(requested)
    for _ in range(3):
        assert lifetime.reset_sim_for_episode(env, requested) is expected
    assert model["hand_stiffness"] == (0.365533 if expected else 350.0)


class _RecordingEnv:
    """Isaac Lab's ManagerBasedRLEnv reset paths (bb0c8e1b), reduced to a log."""

    def __init__(self, log):
        self.log = log
        self.common_step_counter = 0
        self.action_manager = SimpleNamespace(active_terms=[])

    def _reset_idx(self, env_ids):
        self.log.append(("_reset_idx", env_ids))

    def reset(self, seed=None, env_ids=None, options=None):
        self._reset_idx(env_ids)
        self.log.append(("forward",))
        return "obs", {}

    def reset_to(self, state, env_ids, seed=None, is_relative=False):
        self._reset_idx(env_ids)
        self.log.append(("reset_to", state, env_ids, is_relative))
        return "obs", {}

    def step(self, action):
        self.log.append(("physics", action))
        terminated = action["terminated"]
        if terminated:
            self._reset_idx(terminated)
        return "obs", 0.0, False, False, {}


class _MimicEnv(ReplaysSubtaskBoundaries, _RecordingEnv):
    pass


@pytest.mark.parametrize("entry_point", [f"{__name__}:_RecordingEnv", _MimicEnv], ids=["string", "mimic-class"])
@pytest.mark.parametrize("env_ids", [None, [1]])
def test_every_newton_reset_ends_with_the_saved_state_sync(monkeypatch, entry_point, env_ids):
    pytest.importorskip("gymnasium")
    log = []
    monkeypatch.setattr(lifetime, "synchronize_newton_episode_reset", lambda ids=None: log.append(("sync", ids)))
    env_class = lifetime.newton_entry_point(entry_point)
    assert lifetime.newton_entry_point(env_class) is env_class
    env = env_class(log)
    for episode in range(3):
        assert env.reset_to({"episode": episode}, env_ids, is_relative=True) == ("obs", {})
    assert log == [entry for episode in range(3)
                   for entry in (("_reset_idx", env_ids), ("sync", env_ids),
                                 ("reset_to", {"episode": episode}, env_ids, True), ("sync", env_ids))]
    if entry_point is _MimicEnv:
        assert isinstance(env, _MimicEnv) and env._replay_episode_index == 2

    log.clear()
    assert env.reset(env_ids=env_ids) == ("obs", {})
    env.step({"terminated": []})
    env.step({"terminated": [0]})
    env.step({"terminated": []})
    assert log == [("_reset_idx", env_ids), ("sync", env_ids), ("forward",),
                   ("physics", {"terminated": []}),
                   ("physics", {"terminated": [0]}), ("_reset_idx", [0]), ("sync", [0]),
                   ("physics", {"terminated": []})]


def test_saved_grasp_refuses_external_restore_before_reset_side_effects(monkeypatch):
    pytest.importorskip("isaaclab")
    from dexbench_arena.embodiments.common.grasp_action import SavedGraspJointPositionAction

    log = []
    monkeypatch.setattr(lifetime, "synchronize_newton_episode_reset", lambda ids=None: log.append(("sync", ids)))
    env = lifetime.newton_entry_point(_RecordingEnv)(log)
    action = SimpleNamespace(validate_scene_state_restore=lambda state, **kwargs:
        SavedGraspJointPositionAction.validate_scene_state_restore(None, state, **kwargs))
    env.action_manager = SimpleNamespace(active_terms=['support'], get_term=lambda name: action)
    with pytest.raises(ValueError, match="scene-only env.reset_to"):
        env.reset_to({'articulation': {}}, [0], seed=77, is_relative=True)
    assert log == []
    env.reset(env_ids=[0])
    assert log == [("_reset_idx", [0]), ("sync", [0]), ("forward",)]


def test_annotation_starts_episodes_like_rerecord(tmp_path, monkeypatch):
    """annotate_demos.py through the wrapper's gym.make: Isaac Lab's script hard-resets
    before every episode and has no flag for it. Every DexBench environment runs on
    Newton, so the wrapper drops it without reading the backend."""
    h5py = pytest.importorskip("h5py")
    monkeypatch.syspath_prepend(str(SCRIPTS))

    joints = ["torso_0", "head_0", "left_arm_0", "wheel_fl"]
    recording = tmp_path / "demos.hdf5"
    with h5py.File(recording, "w") as source:
        source.create_group("data").attrs["env_args"] = json.dumps(
            {"dexbench_rrc_dataset": {"state_joint_names": joints, "action_joint_names": joints[:3]}})

    class NewtonCfg:
        pass

    events = []
    model = {"hand_stiffness": 350.0}

    def hard_reset():
        model["hand_stiffness"] = 0.365533
        events.append("hard_reset")

    term = SimpleNamespace(_joint_ids=[0, 1, 2])
    env = SimpleNamespace(
        cfg=SimpleNamespace(sim=SimpleNamespace(physics=NewtonCfg(), dt=1 / 60), decimation=2),
        sim=SimpleNamespace(reset=hard_reset),
        scene={"robot": SimpleNamespace(joint_names=joints)},
        action_manager=SimpleNamespace(active_terms=["joint_pos"], get_term=lambda name: term),
        recorder_manager=SimpleNamespace(reset=lambda: events.append("recorder_reset"), get_ep_meta=lambda: {}),
        reset_to=lambda state, ids, seed=None, is_relative=False: events.append((state, ids, is_relative)),
    )
    monkeypatch.setitem(sys.modules, "gymnasium", SimpleNamespace(make=lambda *a, **kw: SimpleNamespace(unwrapped=env)))
    helper = runpy.run_path(str(SCRIPTS / "_dexbench_arena_script.py"))
    monkeypatch.setitem(helper["_on_env_made"].__globals__, "_after_app_launch", lambda hook: hook())
    helper["_install_annotation_episode_reset"](["--viz", "none", "--input_file", str(recording)])

    gym = sys.modules["gymnasium"]
    annotated = gym.make("dexbench_logistics_center_e2_environment", cfg=env.cfg).unwrapped
    episodes = ({"seed": 31}, {"seed": 47})
    for initial_state in episodes:
        annotated.sim.reset()
        annotated.recorder_manager.reset()
        annotated.reset_to(initial_state, None, is_relative=True)
    expected = []
    for initial_state in episodes:
        expected += ["recorder_reset", (initial_state, None, True)]
    assert events == expected
    assert model["hand_stiffness"] == 350.0

    with h5py.File(recording, "r+") as source:
        source["data"].attrs["env_args"] = json.dumps(
            {"dexbench_rrc_dataset": {"state_joint_names": joints[::-1], "action_joint_names": joints[:3]}})
    with pytest.raises(ValueError, match="state joint order"):
        gym.make("dexbench_logistics_center_e2_environment", cfg=env.cfg)

    with h5py.File(recording, "r+") as source:
        source["data"].attrs["env_args"] = json.dumps({"dexbench_recording": {"physics_backend": "OtherCfg"}})
    env.recorder_manager.get_ep_meta = lambda: {"dexbench_rrc_dataset": {"state_joint_names": joints}}
    with pytest.raises(ValueError, match="recorded on OtherCfg"):
        gym.make("dexbench_logistics_center_e2_environment", cfg=env.cfg)


def test_annotation_of_a_pink_ik_recording_checks_only_its_state_order(tmp_path, monkeypatch):
    """An R2-style IsaacTeleop source (Pink IK, its contract naming the 62 joint targets) under
    annotate_demos.py, whose MimicRecorderManagerCfg records no joint targets: nothing to compare
    the action names with, so it annotates; a scrambled state order is still refused."""
    h5py = pytest.importorskip("h5py")
    monkeypatch.syspath_prepend(str(SCRIPTS))

    joints = ["torso_0", "head_0", "left_arm_0", "left_hand_0", "wheel_fl"]
    recording = tmp_path / "r2.hdf5"

    def write(state_names):
        with h5py.File(recording, "w") as source:
            source.create_group("data").attrs["env_args"] = json.dumps(
                {"dexbench_rrc_dataset": {"state_joint_names": state_names, "action_joint_names": joints[:4]}})

    pink = SimpleNamespace()
    env = SimpleNamespace(
        cfg=SimpleNamespace(sim=SimpleNamespace(physics=object(), dt=1 / 60), decimation=2,
                            recorders=SimpleNamespace(record_post_step_processed_actions=object())),
        sim=SimpleNamespace(reset=lambda: None),
        scene={"robot": SimpleNamespace(joint_names=joints)},
        action_manager=SimpleNamespace(active_terms=["upper_body_ik"], get_term=lambda name: pink),
        recorder_manager=SimpleNamespace(get_ep_meta=lambda: {}),
    )
    monkeypatch.setitem(sys.modules, "gymnasium", SimpleNamespace(make=lambda *a, **kw: SimpleNamespace(unwrapped=env)))
    helper = runpy.run_path(str(SCRIPTS / "_dexbench_arena_script.py"))
    monkeypatch.setitem(helper["_on_env_made"].__globals__, "_after_app_launch", lambda hook: hook())
    helper["_install_annotation_episode_reset"](["--viz", "none", "--input_file", str(recording)])
    gym = sys.modules["gymnasium"]

    write(joints)
    assert gym.make("dexbench_retail_v1_environment", cfg=env.cfg).unwrapped is env
    write(joints[::-1])
    with pytest.raises(ValueError, match="state joint order"):
        gym.make("dexbench_retail_v1_environment", cfg=env.cfg)


@pytest.fixture
def saved_state_sync_scene(monkeypatch):
    """Exercise the pinned conversion kernel on CPU, including free poses."""
    wp = pytest.importorskip("warp")
    newton = pytest.importorskip("newton")
    from dexbench_arena.assets import newton_episode_reset as compat

    wp.init()
    world = newton.ModelBuilder()
    arm = world.add_link(label="arm")
    world.add_articulation([world.add_joint_revolute(-1, arm)])
    dish = world.add_link(label="dish")
    world.add_articulation([world.add_joint_free(dish)])
    builder = newton.ModelBuilder()
    for i in range(3):
        builder.add_world(world, label_prefix=f"env{i}")
    model = builder.finalize(device="cpu")
    model.joint_target_ke.fill_(350.0)
    model.joint_effort_limit.fill_(3.0)
    state = model.state()
    poses = np.array([[0.1 + i, .01 * (i + 1), .02, .5, 0., 0., np.sin(.2), np.cos(.2)]
                      for i in range(3)], dtype=np.float32)
    velocities = np.tile(np.array([.3, .4, .5, .6, 0., 0., 0.], dtype=np.float32), (3, 1))
    state.joint_q.assign(poses.reshape(-1))
    state.joint_qd.assign(velocities.reshape(-1))
    solver = object.__new__(newton.solvers.SolverMuJoCo)
    solver.model = model
    solver.enable_sleeping = False
    solver.use_mujoco_cpu = False
    solver.update_data_interval = 3
    solver._step = 16
    solver.mj_q_start = wp.array([0, 1], dtype=wp.int32, device="cpu")
    solver.mj_qd_start = wp.array([0, 1], dtype=wp.int32, device="cpu")
    buffers = {name: wp.full((3, 2), 17.0, dtype=wp.float32, device="cpu")
               for name in ("qacc_warmstart", "qfrc_applied", "ctrl", "act")}
    buffers["xfrc_applied"] = wp.full((3, 2), wp.spatial_vector(*([17.0] * 6)), device="cpu")
    solver.mjw_data = SimpleNamespace(
        nworld=3, qpos=wp.full((3, 8), 17.0, device="cpu"),
        qvel=wp.full((3, 7), 17.0, device="cpu"), **buffers,
    )
    manager = SimpleNamespace(_solver=solver, get_model=lambda: model, get_state_0=lambda: state)
    monkeypatch.setitem(sys.modules, "isaaclab_newton.physics", SimpleNamespace(NewtonManager=manager))
    return compat, wp, newton, manager, poses, velocities, buffers


@pytest.mark.parametrize("interval", [0, 1, 2, 3])
@pytest.mark.parametrize("env_ids", [None, [1], []])
def test_saved_state_sync_handles_periodic_boundaries_without_rebuilding(saved_state_sync_scene, interval, env_ids):
    compat, wp, _, manager, poses, velocities, buffers = saved_state_sync_scene
    solver, model, state = manager._solver, manager.get_model(), manager.get_state_0()
    solver.update_data_interval = interval
    selected = list(range(3)) if env_ids is None else env_ids
    mask = None if env_ids is None else wp.array(
        [i in env_ids for i in range(3)] + [False], dtype=wp.bool, device="cpu"
    )
    solver.reset(state, world_mask=mask, flags=0)
    np.testing.assert_array_equal(solver.mjw_data.qpos.numpy(), 17.0)
    np.testing.assert_array_equal(solver.mjw_data.qvel.numpy(), 17.0)
    previous_history = {name: buffer.numpy().copy() for name, buffer in buffers.items()}

    assert compat.synchronize_newton_episode_reset(env_ids) is bool(selected)
    expected_qpos, expected_qvel = np.full((3, 8), 17.0), np.full((3, 7), 17.0)
    for i in selected:
        expected_qpos[i] = poses[i, [0, 1, 2, 3, 7, 4, 5, 6]]
        expected_qvel[i] = velocities[i]
    np.testing.assert_allclose(solver.mjw_data.qpos.numpy(), expected_qpos, atol=1e-7)
    np.testing.assert_allclose(solver.mjw_data.qvel.numpy(), expected_qvel, atol=1e-7)
    np.testing.assert_array_equal(state.joint_q.numpy(), poses.reshape(-1))
    np.testing.assert_array_equal(state.joint_qd.numpy(), velocities.reshape(-1))
    np.testing.assert_array_equal(model.joint_target_ke.numpy(), 350.0)
    np.testing.assert_array_equal(model.joint_effort_limit.numpy(), 3.0)
    assert solver._step == 16
    for name, buffer in buffers.items():
        np.testing.assert_array_equal(buffer.numpy(), previous_history[name])


def test_native_cpu_sync_preserves_an_unselected_template(saved_state_sync_scene):
    compat, _, _, manager, poses, velocities, _ = saved_state_sync_scene
    solver = manager._solver
    solver.use_mujoco_cpu = True
    solver._total_loop_joint_coords = 0
    solver.mj_data = SimpleNamespace(qpos=np.full(8, 17.0), qvel=np.full(7, 17.0))
    assert not compat.synchronize_newton_episode_reset([1])
    np.testing.assert_array_equal(solver.mj_data.qpos, 17.0)
    assert compat.synchronize_newton_episode_reset([0])
    np.testing.assert_allclose(solver.mj_data.qpos, poses[0, [0, 1, 2, 3, 7, 4, 5, 6]], atol=1e-7)
    np.testing.assert_allclose(solver.mj_data.qvel, velocities[0], atol=1e-7)


@pytest.mark.parametrize("selection", [[-1], [3], [True], [1.5]])
def test_invalid_reset_selection_cannot_overwrite_another_world(saved_state_sync_scene, selection):
    compat, _, _, manager, _, _, _ = saved_state_sync_scene
    with pytest.raises(ValueError, match="valid environment indices"):
        compat.synchronize_newton_episode_reset(selection)
    np.testing.assert_array_equal(manager._solver.mjw_data.qpos.numpy(), 17.0)


def test_other_newton_solvers_do_not_require_mujoco_sync(saved_state_sync_scene):
    compat, _, _, manager, _, _, _ = saved_state_sync_scene
    manager._solver = object()
    assert not compat.synchronize_newton_episode_reset()


def test_the_coupled_managers_rigid_solver_gets_the_saved_state(saved_state_sync_scene):
    """The coupled MuJoCo-Warp + VBD manager (any polybag scene) keeps a placeholder in NewtonManager's solver slot
    and MuJoCo-Warp as its own rigid solver; the reset state reaches that one."""
    compat, _, _, manager, poses, velocities, _ = saved_state_sync_scene
    solver = manager._solver
    manager._solver = object()
    manager._sim = SimpleNamespace(physics_manager=SimpleNamespace(_rigid_solver=solver))
    assert compat.synchronize_newton_episode_reset([1])
    qpos = solver.mjw_data.qpos.numpy()
    np.testing.assert_allclose(qpos[1], poses[1, [0, 1, 2, 3, 7, 4, 5, 6]], atol=1e-7)
    np.testing.assert_allclose(solver.mjw_data.qvel.numpy()[1], velocities[1], atol=1e-7)
    np.testing.assert_array_equal(qpos[[0, 2]], 17.0)


def test_the_paper_couplers_wrapped_solver_gets_the_saved_state(saved_state_sync_scene):
    """DB-L3's PaperProxyCoupler takes NewtonManager's solver slot and keeps MuJoCo as its rigid_solver; every episode
    reset still reaches MuJoCo, so the next episode does not start at its predecessor's pose."""
    compat, _, _, manager, poses, velocities, _ = saved_state_sync_scene
    solver = manager._solver
    manager._solver = SimpleNamespace(rigid_solver=solver)
    assert compat.synchronize_newton_episode_reset([1])
    qpos = solver.mjw_data.qpos.numpy()
    np.testing.assert_allclose(qpos[1], poses[1, [0, 1, 2, 3, 7, 4, 5, 6]], atol=1e-7)
    np.testing.assert_allclose(solver.mjw_data.qvel.numpy()[1], velocities[1], atol=1e-7)
    np.testing.assert_array_equal(qpos[[0, 2]], 17.0)
