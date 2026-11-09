# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""``--record_dataset`` and the typed ``record_dataset`` fields compose the evaluation recorder."""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[3]
SHIM = ROOT / "scripts" / "imitation_learning" / "_dexbench_arena_script.py"

isaaclab = pytest.importorskip("isaaclab.managers.recorder_manager", reason="needs Isaac Lab (the DexBench container)")
pytest.importorskip("isaaclab_arena.terms.recorders", reason="needs IsaacLab-Arena on the path")

from isaaclab.managers import DatasetExportMode  # noqa: E402
from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg, RecorderTermCfg  # noqa: E402
from isaaclab_arena.evaluation import run_execution  # noqa: E402
from isaaclab_arena.utils.configclass import combine_configclass_instances, make_configclass  # noqa: E402
from isaaclab_arena.terms.recorders import ArenaEnvRecorderManagerCfg, make_trajectory_recorder_terms_cfg  # noqa: E402

from dexbench_arena.environments.manipulation.scene_environment_base import DexBenchEnvironmentCfg  # noqa: E402
from dexbench_arena.evaluation import dataset_recording as recording  # noqa: E402

ISAACLAB_TERMS = {
    "record_initial_state",
    "record_post_step_states",
    "record_pre_step_actions",
    "record_pre_step_flat_policy_observations",
    "record_post_step_processed_actions",
}
ARENA_EXTRA_TERM = "record_post_step_flat_policy_action_observations"


def _term_names(cfg) -> set[str]:
    return {f.name for f in dataclasses.fields(cfg) if f.name.startswith("record_")}


class _SuccessTermCfg(RecorderTermCfg):
    """Stands in for the metric recorder term the builder composes."""


def _metric_only_recorders(filename: str = "dataset_20260921_120000_rank0"):
    """What ``ArenaEnvBuilder.compose_manager_cfg`` leaves in ``env_cfg.recorders``."""
    cfg_cls = make_configclass(
        "RecorderManagerCfg", [("success", _SuccessTermCfg, _SuccessTermCfg())], bases=(RecorderManagerBaseCfg,)
    )
    cfg = cfg_cls()
    cfg.dataset_filename = filename
    return cfg


@pytest.fixture(autouse=True)
def _fresh_run_dirs(monkeypatch):
    monkeypatch.setattr(recording, "_RUN_DIRS", {})


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    recording.add_dataset_recording_cli_args(parser)
    return parser.parse_args(argv)


def test_flags_default_off_and_images_imply_recording():
    assert recording.dataset_recording_request(_parse([])) == (False, False)
    assert recording.dataset_recording_request(_parse(["--record_dataset"])) == (True, False)
    assert recording.dataset_recording_request(_parse(["--record_dataset_images"])) == (True, True)
    assert recording.dataset_recording_request(NS()) == (False, False)


def test_terms_without_images_drop_only_the_camera_term():
    without = recording.make_dataset_recorder_terms_cfg(include_images=False)
    with_images = recording.make_dataset_recorder_terms_cfg(include_images=True)

    upstream = _term_names(ArenaEnvRecorderManagerCfg())
    assert recording.CAMERA_RECORDER_TERM in upstream, "upstream renamed the camera term"
    assert _term_names(with_images) == upstream
    assert _term_names(without) == upstream - {recording.CAMERA_RECORDER_TERM}
    assert ISAACLAB_TERMS | {ARENA_EXTRA_TERM} <= _term_names(without)


def test_no_request_leaves_recorders_untouched():
    env_cfg = NS(recorders=_metric_only_recorders())
    before = env_cfg.recorders
    recording.apply_dataset_recording(env_cfg, NS(), enable_cameras=True, output_base_dir="outputs")
    assert env_cfg.recorders is before


@pytest.mark.parametrize("include_images", [False, True])
def test_recording_extends_metric_terms_and_exports_into_the_run_dir(tmp_path, include_images):
    env_cfg = NS(recorders=_metric_only_recorders("dataset_20260921_120000_rank3"))
    args = NS(record_dataset=True, record_dataset_images=include_images)

    recording.apply_dataset_recording(env_cfg, args, enable_cameras=True, output_base_dir=str(tmp_path))

    cfg = env_cfg.recorders
    assert isinstance(cfg, RecorderManagerBaseCfg)
    assert isinstance(cfg.success, _SuccessTermCfg), "metric term lost"
    assert (recording.CAMERA_RECORDER_TERM in _term_names(cfg)) is include_images
    assert ISAACLAB_TERMS | {ARENA_EXTRA_TERM} <= _term_names(cfg)
    assert cfg.dataset_filename == "dataset_20260921_120000_rank3"
    assert cfg.dataset_export_dir_path == recording.run_output_dir(str(tmp_path))
    assert Path(cfg.dataset_export_dir_path).parent == tmp_path
    assert cfg.dataset_export_mode == DatasetExportMode.EXPORT_ALL
    assert cfg.export_in_record_pre_reset and not cfg.export_in_close
    assert cfg.dataset_compression


@pytest.mark.parametrize("include_images", [False, True])
def test_dataset_recording_extends_the_builders_trajectory_terms(tmp_path, include_images):
    base = combine_configclass_instances(
        "RecorderManagerCfg", _metric_only_recorders(),
        make_trajectory_recorder_terms_cfg(frame_transformer_names=()),
        bases=(RecorderManagerBaseCfg,),
    )
    custom_action_recorder = "test.custom_actions:Recorder"
    base.record_pre_step_actions.class_type = custom_action_recorder
    env_cfg = NS(recorders=base)
    recording.apply_dataset_recording(
        env_cfg, NS(record_dataset=True, record_dataset_images=include_images),
        enable_cameras=include_images, export_dir=str(tmp_path),
    )
    cfg = env_cfg.recorders
    assert ISAACLAB_TERMS | {ARENA_EXTRA_TERM, "record_episode_id", "record_gripper_state"} <= _term_names(cfg)
    assert cfg.record_episode_id == base.record_episode_id
    assert cfg.record_pre_step_actions.class_type == custom_action_recorder
    assert cfg.success == base.success
    assert (recording.CAMERA_RECORDER_TERM in _term_names(cfg)) is include_images
    assert cfg.dataset_filename == base.dataset_filename
    assert cfg.dataset_export_dir_path == str(tmp_path)
    assert cfg.dataset_export_mode == DatasetExportMode.EXPORT_ALL


def test_images_need_cameras():
    env_cfg = NS(recorders=_metric_only_recorders())
    args = NS(record_dataset=False, record_dataset_images=True)
    with pytest.raises(ValueError, match="--enable_cameras"):
        recording.apply_dataset_recording(env_cfg, args, enable_cameras=False, output_base_dir="outputs")
    args.record_dataset_images = False
    args.record_dataset = True
    recording.apply_dataset_recording(env_cfg, args, enable_cameras=False, output_base_dir="outputs")
    assert recording.CAMERA_RECORDER_TERM not in _term_names(env_cfg.recorders)


def test_run_dir_is_stamped_once_per_base_dir(monkeypatch):
    from isaaclab_arena.video import video_recording

    stamps = iter(["a", "b", "c"])
    monkeypatch.setattr(video_recording, "timestamped_run_dir", lambda base: f"{base}/{next(stamps)}")

    assert recording.run_output_dir("outputs") == "outputs/a"
    assert recording.run_output_dir("outputs") == "outputs/a"
    assert recording.run_output_dir("elsewhere") == "elsewhere/b"


def test_shared_run_dir_makes_upstream_reuse_the_first_stamp(monkeypatch):
    from isaaclab_arena.video import video_recording

    original = video_recording.timestamped_run_dir
    monkeypatch.setattr(video_recording, "timestamped_run_dir", original)
    recording.install_shared_run_dir()
    recording.install_shared_run_dir()
    shared = video_recording.timestamped_run_dir
    assert shared is not original and shared._dexbench_original is original

    upstream_dir = shared("outputs")
    assert recording.run_output_dir("outputs") == upstream_dir
    assert Path(upstream_dir).parent == Path("outputs")


@pytest.fixture
def shim(monkeypatch):
    """The launcher's environment hook, with every global it patches restored on teardown."""
    import runpy

    import isaaclab_arena_environments.cli as environments_cli
    from isaaclab.app import AppLauncher
    from isaaclab_arena.video import video_recording

    monkeypatch.setattr(AppLauncher, "__init__", lambda self, *args, **kwargs: None)
    monkeypatch.setattr(video_recording, "timestamped_run_dir", video_recording.timestamped_run_dir)
    monkeypatch.setattr(run_execution, "build_and_run", run_execution.build_and_run)
    from isaaclab_arena.recording.arena_recorder_manager import ArenaRecorderManager

    monkeypatch.setattr(ArenaRecorderManager, "add_to_episodes", ArenaRecorderManager.add_to_episodes)
    originals = {
        name: getattr(environments_cli, name)
        for name in ("ensure_environments_registered", "get_isaaclab_arena_environments_cli_parser")
    }
    for name, value in originals.items():
        monkeypatch.setattr(environments_cli, name, value)
    for name in (
        "isaaclab_arena.evaluation.run_execution",
        "isaaclab_arena.evaluation.legacy_graph_environment_cli",
    ):
        monkeypatch.delitem(sys.modules, name, raising=False)
    import isaaclab_arena.evaluation as evaluation_package

    for attr in ("run_execution", "legacy_graph_environment_cli"):
        monkeypatch.delattr(evaluation_package, attr, raising=False)
    namespace = runpy.run_path(str(SHIM))

    def install(**kwargs):
        for name, value in originals.items():
            setattr(environments_cli, name, value)
        namespace["_register_dexbench_environments_with_arena"](**kwargs)
        return environments_cli

    return install


def _parser_flags(cli_module) -> set[str]:
    parser = cli_module.get_isaaclab_arena_environments_cli_parser(argparse.ArgumentParser())
    return {opt for action in parser._actions for opt in action.option_strings}


def _patched() -> tuple[bool, bool]:
    from isaaclab_arena.video import video_recording

    run_execution_module = sys.modules.get("isaaclab_arena.evaluation.run_execution")
    capture = getattr(run_execution_module, "build_and_run", None)
    return (
        getattr(video_recording.timestamped_run_dir, "_dexbench_shared", False),
        getattr(capture, "_dexbench_captures_output_dir", False),
    )


def test_shim_routes_evaluation_scripts_to_the_evaluation_parser(shim, monkeypatch):
    """The upstream runner's parser entrypoint gains the flags only on the evaluation path."""
    monkeypatch.setattr(sys, "argv", ["policy_runner.py"])
    assert {"--record_dataset", "--record_dataset_images"} <= _parser_flags(
        shim(evaluation=True, script_name="policy_runner.py")
    )
    assert not ({"--record_dataset", "--record_dataset_images"} & _parser_flags(shim(evaluation=False)))


def test_upstream_cli_resolves_dexbench_environments(shim, monkeypatch):
    """Arena's own parser and registry see DexBench's environments next to its examples."""
    argv = ["policy_runner.py", "--record_dataset", "dexbench_hotel_fb_e4_environment", "--task", "stack_plates"]
    monkeypatch.setattr(sys, "argv", argv)
    cli_module = shim(evaluation=True, script_name="policy_runner.py")
    assert sys.modules["isaaclab_arena_environments.cli"] is cli_module
    assert Path(cli_module.__file__).name == "cli.py"

    from isaaclab_arena.assets.registries import EnvironmentRegistry

    cli_module.ensure_environments_registered()
    names = set(EnvironmentRegistry().get_all_keys())
    assert "dexbench_hotel_fb_e4_environment" in names
    assert any(not name.startswith("dexbench_") for name in names), "Arena's own examples are registered too"

    args = cli_module.get_isaaclab_arena_environments_cli_parser().parse_args(argv[1:])
    assert (args.example_environment, args.task, args.record_dataset) == (
        "dexbench_hotel_fb_e4_environment",
        "stack_plates",
        True,
    )


def test_policy_runner_without_the_flags_patches_nothing(shim, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["policy_runner.py", "--policy_type", "zero_action", "--enable_cameras"])
    shim(evaluation=True, script_name="policy_runner.py")
    assert _patched() == (False, False)


@pytest.mark.parametrize("flag", ["--record_dataset", "--record_dataset_images", "--record_dataset_im"])
def test_policy_runner_with_a_flag_shares_the_run_dir_only(shim, monkeypatch, flag):
    monkeypatch.setattr(sys, "argv", ["policy_runner.py", flag, "--enable_cameras"])
    shim(evaluation=True, script_name="policy_runner.py")
    assert _patched() == (True, False)


@pytest.mark.parametrize(
    "cameras, ok",
    [([], False), (["--enable_cameras"], True), (["--record_camera_video"], True)],
    ids=["none", "enable_cameras", "record_camera_video"],
)
def test_frames_without_cameras_fail_before_kit(monkeypatch, cameras, ok):
    monkeypatch.setattr(recording, "install_shared_run_dir", lambda: None)
    argv = ["--record_dataset_images", *cameras]
    if ok:
        recording.install_dataset_recording_patches("policy_runner.py", argv, after_launch=lambda _: None)
    else:
        with pytest.raises(SystemExit, match="--enable_cameras"):
            recording.install_dataset_recording_patches("policy_runner.py", argv, after_launch=lambda _: None)


def test_the_typed_runner_gets_the_capture_and_binds_the_dexbench_cli(shim, monkeypatch):
    """The capture imports run_execution once the app is up, after the parser is wrapped."""
    from isaaclab.app import AppLauncher

    script = "isaaclab_arena.evaluation.experiment_runner"
    monkeypatch.setattr(sys, "argv", [script, "--experiment_config", "x.yaml"])
    cli_module = shim(evaluation=True, script_name=script)
    assert _patched() == (False, False)
    AppLauncher()
    assert _patched() == (False, True)
    legacy = sys.modules["isaaclab_arena.evaluation.legacy_graph_environment_cli"]
    assert legacy.get_isaaclab_arena_environments_cli_parser is cli_module.get_isaaclab_arena_environments_cli_parser




def test_typed_cfg_fields_ask_for_recording():
    assert recording.dataset_recording_request(DexBenchEnvironmentCfg()) == (False, False)
    assert recording.dataset_recording_request(DexBenchEnvironmentCfg(record_dataset=True)) == (True, False)
    assert recording.dataset_recording_request(
        DexBenchEnvironmentCfg(record_dataset_images=True, enable_cameras=True)
    ) == (True, True)


def test_run_output_dir_is_captured_only_while_build_and_run_executes(monkeypatch):
    seen = []

    def fake_build_and_run(cfg, output_dir, video_cfg=None):
        seen.append(recording.current_run_output_dir())
        return "result"

    monkeypatch.setattr(run_execution, "build_and_run", fake_build_and_run)
    recording.install_run_output_dir_capture()
    recording.install_run_output_dir_capture()
    assert run_execution.build_and_run._dexbench_original is fake_build_and_run

    assert recording.current_run_output_dir() is None
    assert run_execution.build_and_run(NS(name="r"), Path("/exp/r")) == "result"
    assert seen == ["/exp/r"]
    assert recording.current_run_output_dir() is None


def test_the_effective_experiment_is_written_before_its_runs(monkeypatch, tmp_path):
    from isaaclab_arena.evaluation.arena_experiment_config_loader import load_arena_experiment_from_config_file

    import dexbench_arena.environments  # noqa: F401  -- registers the environment types the YAML names

    experiment_yaml = tmp_path / "exp" / recording.EXPERIMENT_YAML_FILENAME
    seen = []

    def fake_execute_experiment(experiment_cfg, output_dir, record_viewport_video=False, **kwargs):
        seen.append(experiment_yaml.is_file())
        return ["results"]

    monkeypatch.setattr(run_execution, "execute_experiment", fake_execute_experiment)
    recording.install_experiment_yaml_snapshot()
    recording.install_experiment_yaml_snapshot()
    assert run_execution.execute_experiment._dexbench_original is fake_execute_experiment

    source = ROOT / "scripts" / "evaluation" / "experiment_configs" / "e4_hotel_fb_variations.yaml"
    experiment = load_arena_experiment_from_config_file(
        str(source), device="cpu", overrides=["shared.rollout_limit.num_steps=7"]
    )
    assert run_execution.execute_experiment(experiment, output_dir=tmp_path / "exp") == ["results"]
    assert seen == [True]
    assert load_arena_experiment_from_config_file(str(experiment_yaml), device="cpu").runs == experiment.runs


def _sim_cfg_stub(recorders):
    """Enough of an env cfg for the environment's settings callback."""
    return NS(
        sim=NS(physics=NS(), render=NS()),
        scene=NS(),
        events=None,
        num_rerenders_on_reset=0,
        recorders=recorders,
        demo_recorder_config=None,
        mimic_recorder_config=None,
    )


@pytest.mark.parametrize("record", [False, True])
def test_typed_build_records_into_the_run_dir(simulation_app, tmp_path, record):
    """The typed path: settings composed under a captured run dir record into it, under upstream's name."""
    from dexbench_arena.cli import apply_dexbench_env_settings

    cfg = DexBenchEnvironmentCfg(record_dataset=record, enable_cameras=True)
    run_dir = tmp_path / "exp" / "myrun"
    token = recording._CURRENT_RUN.set(recording._TypedRun(str(run_dir)))
    try:
        env_cfg = _sim_cfg_stub(_metric_only_recorders("dataset_myrun_rebuild1"))
        apply_dexbench_env_settings(env_cfg, cfg, embodiment=NS(enable_cameras=True))
    finally:
        recording._CURRENT_RUN.reset(token)

    assert env_cfg.sim.dt * env_cfg.decimation == pytest.approx(1 / 30)
    assert env_cfg.recorders.dataset_filename == "dataset_myrun_rebuild1"
    if record:
        assert ISAACLAB_TERMS <= _term_names(env_cfg.recorders)
        assert recording.CAMERA_RECORDER_TERM not in _term_names(env_cfg.recorders)
        assert env_cfg.recorders.dataset_export_dir_path == str(run_dir)
    else:
        assert not (ISAACLAB_TERMS & _term_names(env_cfg.recorders))
        assert env_cfg.recorders.dataset_export_dir_path == "/tmp/isaaclab/logs"


def test_typed_build_without_a_captured_run_dir_keeps_upstreams_directory():
    from dexbench_arena.cli import apply_dexbench_env_settings

    env_cfg = _sim_cfg_stub(_metric_only_recorders())
    apply_dexbench_env_settings(env_cfg, DexBenchEnvironmentCfg(record_dataset=True), embodiment=NS(enable_cameras=False))
    assert ISAACLAB_TERMS <= _term_names(env_cfg.recorders)
    assert env_cfg.recorders.dataset_export_dir_path == "/tmp/isaaclab/logs"


@pytest.mark.parametrize("name", ["dexbench_rby1_wuji2", "dexbench_rby1_wuji2_joint"])
def test_typed_build_records_complete_wuji2_joint_targets(simulation_app, name):
    """The build applies the embodiment's recorder override (cli.apply_embodiment_recorders) after
    the recording terms, as scene_environment_base's settings callback does. The Wuji2 classes
    provide one (configure_env_recorders)."""
    from isaaclab_arena.assets.registries import AssetRegistry

    from dexbench_arena.cli import apply_dexbench_env_settings, apply_embodiment_recorders
    from dexbench_arena.embodiments.rby1_wuji2 import rby1_wuji2 as _rby1_wuji2  # noqa: F401  (registers)
    from dexbench_arena.embodiments.rby1_wuji2.recorders import Wuji2JointTargetsRecorderCfg

    embodiment = NS(name=name, enable_cameras=False)
    embodiment.configure_env_recorders = AssetRegistry().get_asset_by_name(name)().configure_env_recorders
    env_cfg = _sim_cfg_stub(_metric_only_recorders())
    apply_dexbench_env_settings(
        env_cfg, DexBenchEnvironmentCfg(record_dataset=True), embodiment=embodiment
    )
    apply_embodiment_recorders(env_cfg, embodiment)
    assert type(env_cfg.recorders.record_post_step_processed_actions) is Wuji2JointTargetsRecorderCfg


class _CountingMask:
    """An env's ``active_episode_mask`` that counts the lookups the recorder makes."""

    def __init__(self, mask):
        self.mask, self.lookups = mask, 0

    def __getitem__(self, index):
        self.lookups += 1
        return self.mask[index]


def _recorder(mask):
    import torch
    from isaaclab.managers.recorder_manager import RecorderTerm
    from isaaclab_arena.recording.arena_recorder_manager import ArenaRecorderManager

    class _Idle(RecorderTerm):
        pass

    cfg = make_configclass("Cfg", [("idle", RecorderTermCfg, RecorderTermCfg(class_type=_Idle))],
                           bases=(RecorderManagerBaseCfg,))()
    cfg.dataset_export_mode = DatasetExportMode.EXPORT_NONE
    env = NS(num_envs=4, device="cpu", sim=NS(is_playing=lambda: True), active_episode_mask=_CountingMask(mask),
             cfg=NS(sim=NS(dt=0.01, render_interval=1), decimation=1, scene=NS(num_envs=4)))
    return ArenaRecorderManager(cfg, env), env.active_episode_mask


def _stored(recorder) -> dict:
    """Every episode's data as nested lists, comparable with ==."""
    def plain(value):
        if isinstance(value, dict):
            return {key: plain(item) for key, item in value.items()}
        return [item.tolist() for item in value]
    return {env_id: plain(episode.data) for env_id, episode in recorder._episodes.items() if not episode.is_empty()}


def test_the_recorder_workaround_stores_what_arena_stores_and_looks_up_rows_once(monkeypatch):
    """Arena's own add_to_episodes and the workaround store the same episodes for the same calls; the
    workaround looks the active rows up once per call where Arena does once per tensor leaf."""
    import torch
    import warp as wp
    from isaaclab_arena.recording.arena_recorder_manager import ArenaRecorderManager

    wp.init()
    nested = lambda: {"articulation": {"robot": {"joint_position": torch.tensor([[30.0], [20.0], [10.0]]),
                                                 "joint_velocity": torch.tensor([[3.0], [2.0], [1.0]])}},
                      "warp": wp.from_torch(torch.tensor([300.0, 200.0, 100.0]))}
    calls = [
        ("states", nested, [3, 2, 1]),
        ("states", nested, torch.tensor([3, 2, 1])),
        ("actions", lambda: torch.arange(4.0), None),
        ("actions", lambda: torch.arange(2.0), slice(2, 4)),
        ("obs", lambda: {"policy": torch.ones(2, 3)}, [0, 2]),
        ("empty", lambda: {}, None),
    ]
    stored, lookups = {}, {}
    for name, install in (("arena", False), ("workaround", True)):
        monkeypatch.setattr(ArenaRecorderManager, "add_to_episodes", ArenaRecorderManager.add_to_episodes)
        if install:
            recording.install_recorder_rows_once_per_call()
        recorder, mask = _recorder(torch.tensor([False, True, False, True]))
        try:
            for key, value, env_ids in calls:
                if not (name == "arena" and isinstance(env_ids, slice)):
                    recorder.add_to_episodes(key, value(), env_ids=env_ids)
            if name == "arena":
                recorder.add_to_episodes("actions", torch.arange(2.0), env_ids=[2, 3])
            lookups[name] = mask.lookups
            stored[name] = _stored(recorder)
        finally:
            recorder.close()
    assert stored["workaround"] == stored["arena"]
    assert sorted(stored["arena"]) == [1, 3]
    assert lookups["workaround"] == 6 and lookups["arena"] > lookups["workaround"]


def test_the_recorder_workaround_refuses_another_method(monkeypatch):
    """After a pin move the method it re-implements may be gone or changed; it says so instead of installing."""
    from isaaclab_arena.recording.arena_recorder_manager import ArenaRecorderManager

    def add_to_episodes(self, key, value, env_ids=None):
        pass

    monkeypatch.setattr(ArenaRecorderManager, "add_to_episodes", add_to_episodes)
    with pytest.raises(RuntimeError, match="the Arena pin moved"):
        recording.install_recorder_rows_once_per_call()
    assert ArenaRecorderManager.add_to_episodes is add_to_episodes
