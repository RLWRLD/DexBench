"""Exercise the submission switch with a Kit-free upstream loader stub."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from dexbench_arena.benchmark.models import MODELS_BY_KEY
from dexbench_arena.benchmark.protocol import GA_POLICY_EXECUTION


def test_submission_adds_native_trajectory_recording_and_refuses_opt_out(monkeypatch):
    loader = ModuleType("isaaclab_arena.evaluation.arena_experiment_config_loader")
    loader.load_arena_experiment_from_config_file = lambda *args, **kwargs: None
    environments = ModuleType("dexbench_arena.environments")
    scene = ModuleType("dexbench_arena.environments.manipulation.scene_environment_base")
    scene.environment_cfg_snapshot = lambda cfg: cfg
    for name, module in ((loader.__name__, loader), (environments.__name__, environments), (scene.__name__, scene)):
        monkeypatch.setitem(sys.modules, name, module)
    root = Path(__file__).resolve().parents[3]
    path = root / "source/dexbench_arena/dexbench_arena/benchmark/model_run.py"
    spec = importlib.util.spec_from_file_location("submission_config_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    baseline = {"policy_type": "zero_action"}
    source = {"runs": {"test": {"policy": baseline, "environment": {}, "environment_builder": {"seed": 42}}}}
    model = MODELS_BY_KEY["gr00t"]
    kwargs = dict(record_dataset=True, execution=GA_POLICY_EXECUTION)
    ordinary = module.compose(source, model, model.policy, baseline, **kwargs)
    submitted = module.compose(source, model, model.policy, baseline, submission=True, **kwargs)
    assert submitted["runs"]["test"]["environment_builder"].pop("record_trajectories") is True
    assert submitted == ordinary
    assert source["runs"]["test"]["environment"] == {}
    with pytest.raises(ValueError, match="requires record_dataset"):
        module.compose(source, model, model.policy, baseline, submission=True)
    with pytest.raises(ValueError, match="requires a cell"):
        module.prepare("m1", "train", "gr00t", "unused", manifest_path="unused", configs="unused",
                       checkpoint="weights", record_camera_video=False, submission=True)

    monkeypatch.setitem(sys.modules, "dexbench_arena.benchmark.model_run", module)
    runner_spec = importlib.util.spec_from_file_location("submission_runner_under_test", root / "scripts/evaluation/run_benchmark_cell.py")
    runner = importlib.util.module_from_spec(runner_spec)
    runner_spec.loader.exec_module(runner)
    seen = {}

    def prepare(*args, **kwargs):
        seen.update(kwargs)
        return {"label": "test", "frozen_sha256": "0" * 64}

    monkeypatch.setattr(runner, "prepare", prepare)
    assert runner.main(["--task", "m1", "--cell", "train", "--model", "gr00t", "--output", "unused",
                        "--checkpoint", "weights", "--submission", "--prepare_only"]) == 0
    assert seen["submission"] is True and seen["record_dataset"] is True and seen["record_camera_video"] is True
