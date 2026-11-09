# SPDX-License-Identifier: Apache-2.0
"""record_demos.py records what the teleop preprocessing pipeline replays."""

from __future__ import annotations

import asyncio
import importlib
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts" / "imitation_learning"
ENVIRONMENT = "dexbench_logistics_center_e2_environment"
SOURCE = {"schema_version": 1, "git_sha": "d" * 40, "dirty": False, "dirty_paths": [],
          "dirty_paths_outside_image": [], "branch": "main", "environment": {"DEXBENCH_CONTROL_HZ": "30"}}


def test_record_demos_records_what_the_pipeline_replays(tmp_path, monkeypatch):
    datasets = pytest.importorskip("isaaclab.utils.datasets")
    pytest.importorskip("isaaclab.managers")
    gym = pytest.importorskip("gymnasium")
    import torch

    from dexbench_arena import isaaclab_interop
    from dexbench_arena.embodiments.rby1_wuji2.constants import (
        RBY1_WUJI2_HAND_JOINT_ACTION_NAMES,
        RBY1_WUJI2_JOINT_ACTION_NAMES,
        RBY1_WUJI2_PINK_CONTROLLED_JOINT_NAMES,
        RBY1_WUJI2_TORSO_JOINT_NAMES,
    )
    from dexbench_arena.embodiments.rby1_wuji2.recorders import (
        Wuji2JointTargetsRecorderCfg,
        install_joint_names_recording_metadata,
    )

    monkeypatch.syspath_prepend(str(SCRIPTS))
    metadata = importlib.import_module("dexbench_arena.recording.rerecord_metadata")
    physics = importlib.import_module("dexbench_arena.recording.physics_runtime")
    contract_key = metadata.JOINT_NAMES_KEY
    monkeypatch.setattr(metadata, "take_capture_source", lambda root: (SOURCE, None))
    monkeypatch.setattr(metadata, "commit_on", lambda *args, **kwargs: True)

    solved = [*RBY1_WUJI2_PINK_CONTROLLED_JOINT_NAMES, *RBY1_WUJI2_HAND_JOINT_ACTION_NAMES]
    joints = [*solved, *RBY1_WUJI2_TORSO_JOINT_NAMES, "wheel_fl"]

    def built(values):
        return SimpleNamespace(torch=torch.tensor([values]))

    count = len(joints)
    data = SimpleNamespace(
        joint_stiffness=built([400.0] * (count - 1) + [0.0]), joint_damping=built([40.0] * count),
        joint_armature=built([0.01] * count), joint_friction_coeff=built([0.0] * count),
        joint_viscous_friction_coeff=built([0.0] * count),
        joint_effort_limits=built([100.0] * (count - 1) + [float("inf")]), joint_vel_limits=built([5.0] * count),
    )
    wheels = SimpleNamespace(joint_names=["wheel_fl"], stiffness=torch.tensor([[0.0]]),
                             damping=torch.tensor([[20.0]]), effort_limit=torch.tensor([[float("inf")]]))
    robot = SimpleNamespace(joint_names=joints, data=data, actuators={"wheels": wheels})
    monkeypatch.setenv("DEXBENCH_RRC_MAX_STEP_DEG", "5")
    monkeypatch.setenv("DEXBENCH_HF_TOKEN", "hf_secret")

    class Scene:
        sensors = {}

        def __getitem__(self, name):
            return robot

    term = SimpleNamespace(_controlled_joint_names=solved)
    env = SimpleNamespace(scene=Scene(), action_manager=SimpleNamespace(active_terms=["upper_body_ik"],
                                                                        get_term=lambda name: term),
                          recorder_manager=SimpleNamespace(get_ep_meta=dict))

    def make(*args, cfg, **kwargs):
        env.cfg = cfg
        install_joint_names_recording_metadata(env, None, embodiment="dexbench_rby1_wuji2")
        return SimpleNamespace(unwrapped=env)

    monkeypatch.setattr(gym, "make", make)
    monkeypatch.setattr(isaaclab_interop, "BUILDER", SimpleNamespace(arena_env=SimpleNamespace(
        task=SimpleNamespace(get_task_description=lambda: "Lift the box onto the conveyor."),
        embodiment=SimpleNamespace(name="dexbench_rby1_wuji2"))))

    from isaaclab.app import AppLauncher

    monkeypatch.setattr(AppLauncher, "__init__", lambda self, *args, **kwargs: None)

    dataset = tmp_path / "e2" / "session_001.hdf5"
    helper = runpy.run_path(str(SCRIPTS / "_dexbench_arena_script.py"))
    helper["_install_recording_provenance"](
        ["--viz", "kit", "--device", "cpu", "--enable_cameras", "--dataset_file", str(dataset)],
        ENVIRONMENT,
        ["--anchor", "A1_CAGE", "--task", "lift_to_conveyor", "--teleop_session", "--teleop_device", "openxr",
         "--embodiment", "dexbench_rby1_wuji2", "--arm_mode", "dual_arm"],
    )
    assert gym.make is make
    AppLauncher()
    recorders = SimpleNamespace(record_post_step_processed_actions=Wuji2JointTargetsRecorderCfg())
    cfg = SimpleNamespace(recorders=recorders, decimation=4,
                          sim=SimpleNamespace(dt=1 / 120, physics=SimpleNamespace(solver_type=1)))
    gym.make(ENVIRONMENT, cfg=cfg)

    assert json.loads(dataset.with_suffix(".job.json").read_text()) == {
        "env": ENVIRONMENT,
        "args": ["--anchor", "A1_CAGE", "--task", "lift_to_conveyor", "--embodiment", "dexbench_rby1_wuji2",
                 "--arm_mode", "dual_arm"],
        "source": SOURCE,
        "task": "Lift the box onto the conveyor.",
    }

    handler = cfg.recorders.dataset_file_handler_class_type()
    handler.create(str(dataset), env_name=ENVIRONMENT)
    handler.add_env_args(env.recorder_manager.get_ep_meta())
    episode = datasets.EpisodeData()
    episode.data["actions"] = torch.zeros((1, 54))
    handler.write_episode(episode)
    handler.close()
    import h5py

    with h5py.File(dataset, "r") as recording:
        env_args = json.loads(recording["data"].attrs["env_args"])
    assert env_args[metadata.CAPTURE_SOURCE_KEY] == SOURCE
    recorded = env_args["dexbench_recording"]
    assert {key: recorded[key] for key in physics.physics_runtime_metadata(cfg)} == physics.physics_runtime_metadata(cfg)
    drive = recorded["robot_drive"]
    assert drive["joints"]["joint_names"] == joints
    assert drive["joints"]["stiffness"][0] == 400.0 and drive["joints"]["stiffness"][-1] == 0.0
    assert drive["joints"]["effort_limit"][-1] is None
    assert drive["actuators"]["wheels"] == {"class": "SimpleNamespace", "joint_names": ["wheel_fl"],
                                            "stiffness": [0.0], "damping": [20.0], "effort_limit": [None]}
    assert recorded["dexbench_environment"]["DEXBENCH_RRC_MAX_STEP_DEG"] == "5"
    assert recorded["dexbench_environment"]["DEXBENCH_HF_TOKEN"] == "<redacted>"
    assert env_args[contract_key] == {
        "schema_version": 1, "robot": "rby1_wuji2", "profile": "rby1_wuji2", "embodiment": "dexbench_rby1_wuji2",
        "state_joint_names": joints, "action_joint_names": list(RBY1_WUJI2_JOINT_ACTION_NAMES),
    }


def test_a_launch_that_skips_app_launcher_fails_with_hooks_queued(monkeypatch):
    """IsaacLab#8090 scripts start Kit through launch_simulation; the queued hooks would never run."""
    app = pytest.importorskip("isaaclab.app")
    from isaaclab.app import AppLauncher

    monkeypatch.setattr(AppLauncher, "__init__", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(app, "launch_simulation", app.launch_simulation, raising=False)
    helper = runpy.run_path(str(SCRIPTS / "_dexbench_arena_script.py"))
    ran = []
    helper["_after_app_launch"](lambda: ran.append(True))
    from isaaclab.app import launch_simulation

    with pytest.raises(RuntimeError, match="IsaacLab#8090"):
        launch_simulation(None, {})
    AppLauncher()
    assert ran == [True]


def test_concurrent_generation_attempts_log_their_own_picks(tmp_path, monkeypatch):
    """generate_dataset.py runs one asyncio task per env over one DataGenerator; each attempt's picks
    go to its own env's episode and its own log row, in the order the attempts finish."""
    gym = pytest.importorskip("gymnasium")
    data_generator = pytest.importorskip("isaaclab_mimic.datagen.data_generator")
    import torch
    from isaaclab.app import AppLauncher

    monkeypatch.syspath_prepend(str(SCRIPTS))
    metadata = importlib.import_module("dexbench_arena.recording.rerecord_metadata")
    physics = importlib.import_module("dexbench_arena.recording.physics_runtime")
    monkeypatch.setattr(metadata, "take_capture_source", lambda root: (SOURCE, None))
    monkeypatch.setattr(metadata, "as_built_metadata", lambda env: {})
    monkeypatch.setattr(physics, "physics_runtime_metadata", lambda cfg: {"physics_backend": "test"})

    generator_type = data_generator.DataGenerator

    async def generate(self, env_id, picks, success):
        for pick in picks:
            self.select_source_demo(eef_name="right", pick=pick)
            await asyncio.sleep(0)
        return {"success": success}

    monkeypatch.setattr(generator_type, "generate", generate)
    monkeypatch.setattr(generator_type, "select_source_demo", lambda self, eef_name, pick: pick)
    monkeypatch.setattr(AppLauncher, "__init__", lambda self, *args, **kwargs: None)
    episodes = []
    env = SimpleNamespace(device="cpu", recorder_manager=SimpleNamespace(
        get_ep_meta=dict,
        add_to_episodes=lambda key, value, env_ids: episodes.append((env_ids, key, int(value))),
    ))
    monkeypatch.setattr(gym, "make", lambda *args, cfg, **kwargs: SimpleNamespace(unwrapped=env))

    helper = runpy.run_path(str(SCRIPTS / "_dexbench_arena_script.py"))
    helper["_install_generation_provenance"](["--input_file", str(tmp_path / "missing.hdf5")])
    AppLauncher()
    recorders = SimpleNamespace(dataset_export_dir_path=str(tmp_path / "out"), dataset_filename="tc1")
    gym.make("dexbench_env", cfg=SimpleNamespace(recorders=recorders))

    generator = object.__new__(generator_type)
    generator.env = env

    async def two_envs():
        return await asyncio.gather(generator.generate(0, [1, 2, 3], True), generator.generate(1, [7, 8], False))

    asyncio.run(two_envs())
    rows = [json.loads(line) for line in (tmp_path / "out" / "tc1.attempts.jsonl").read_text().splitlines()]
    assert rows == [{"success": False, "selected_src_demo_inds": [7, 8], "selected_src_demo_eefs": ["right", "right"]},
                    {"success": True, "selected_src_demo_inds": [1, 2, 3], "selected_src_demo_eefs": ["right"] * 3}]
    by_env = {env_id: [value for ids, _, value in episodes if ids == [env_id]] for env_id in (0, 1)}
    assert by_env == {0: [1, 2, 3], 1: [7, 8]}
    assert {key for _, key, _ in episodes} == {"dexbench_src_demo_inds/right"}
    assert env.recorder_manager.get_ep_meta()["dexbench_mimic_generation_run"]["code"] == SOURCE
