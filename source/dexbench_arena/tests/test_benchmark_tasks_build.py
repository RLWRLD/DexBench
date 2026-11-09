# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Every benchmark task builds on Newton, resets and steps with finite state (Kit tier)."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("isaaclab")
pytest.importorskip("isaaclab_arena")

pytestmark = pytest.mark.usefixtures("simulation_app")

BENCHMARK = Path(__file__).resolve().parents[3] / "scripts" / "evaluation" / "experiment_configs" / "benchmark"
_MANIFEST = json.loads((BENCHMARK / "benchmark_manifest.json").read_text())
TASKS = {key: BENCHMARK / entry["cells"]["train"]["eval"] for key, entry in _MANIFEST["tasks"].items()}
STEPS = 180


REPO_ROOT = BENCHMARK.parents[3]
FETCH = "scripts/download_datasets.sh --eval-conditions"


def _under_outputs(path: str) -> bool:
    """Whether ``path``, relative to the repository root or absolute, lies under the gitignored outputs/."""
    return bool(path) and (REPO_ROOT / path).resolve().is_relative_to((REPO_ROOT / "outputs").resolve())


def _missing_inputs(environment) -> list[str]:
    """The paths under outputs/ the Run names, alone or in a list, that this checkout does not have."""
    missing = []
    for spec in dataclasses.fields(environment):
        value = getattr(environment, spec.name)
        for path in value if isinstance(value, (list, tuple)) else (value,):
            if isinstance(path, str) and _under_outputs(path) and not (REPO_ROOT / path).exists():
                missing.append(path)
    return missing


def _tensors(value):
    """Every tensor in an observation, however its groups nest."""
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensors(item)


def _record_hold_lengths(monkeypatch) -> dict:
    """Patch both sustained-hold kinds to note the control steps each hold needs; return that record, per hold."""
    from dexbench_arena.tasks.predicates import hold

    lengths = {}
    for kind in (hold.SustainedHold, hold.SustainedShareHold):

        def update(self, ok, step, step_dt, hold_time_s, *args, _update=kind.update, **kwargs):
            lengths[self] = math.ceil(hold_time_s / step_dt - 1e-9)
            return _update(self, ok, step, step_dt, hold_time_s, *args, **kwargs)

        monkeypatch.setattr(kind, "update", update)
    return lengths


def _assert_finite(obs, when):
    for tensor in _tensors(obs):
        if tensor.is_floating_point():
            assert torch.isfinite(tensor).all(), f"non-finite observation {when}"


@pytest.mark.parametrize("task", TASKS, ids=TASKS.keys())
def test_the_task_builds_resets_and_steps(task, monkeypatch):
    import gymnasium as gym
    from isaaclab_arena.evaluation.arena_experiment_config_loader import load_arena_experiment_from_config_file
    from isaaclab_arena.evaluation.run_execution import build_arena_builder_from_run_cfg

    import dexbench_arena.environments  # noqa: F401 -- registers the environment types the experiments name
    from dexbench_arena.policy.hold_pose_policy import HoldPosePolicy, HoldPosePolicyCfg
    from dexbench_arena.tasks.predicates.hold import _holds

    monkeypatch.chdir(REPO_ROOT)
    experiment = load_arena_experiment_from_config_file(str(TASKS[task]), device="cuda:0")
    run = next(iter(experiment.runs.values()))
    if missing := _missing_inputs(run.environment):
        pytest.skip(f"{task} builds from {missing}; fetch them with {FETCH}")
    run = dataclasses.replace(
        run,
        environment=dataclasses.replace(
            run.environment, enable_cameras=False, wrist_cameras=False, benchmark_replay=False
        ),
        environment_builder=dataclasses.replace(run.environment_builder, recorded_variation_samples_path=None),
        recorded_variation_samples_paths=None,
        num_rebuilds=1,
        variations={},
    )
    try:
        name, cfg, kwargs = build_arena_builder_from_run_cfg(run).build_registered()
    except FileNotFoundError as error:
        if not _under_outputs(str(error.filename or "")):
            raise
        pytest.skip(f"{task} builds from {error.filename}; fetch it with {FETCH}")
    cfg.recorders = cfg.metrics = None
    env = gym.make(name, cfg=cfg, **kwargs).unwrapped
    policy_cfg = run.policy if isinstance(run.policy, HoldPosePolicyCfg) else HoldPosePolicyCfg()
    policy = HoldPosePolicy(policy_cfg)
    manager = env.termination_manager
    hold_lengths = _record_hold_lengths(monkeypatch)
    try:
        obs, _ = env.reset()
        _assert_finite(obs, "after reset")
        for step in range(1, STEPS + 1):
            with torch.inference_mode():
                obs, _, _, _, _ = env.step(policy.get_action(env, obs))
            _assert_finite(obs, f"at step {step}")
            if "success" in manager.active_terms:
                assert not bool(manager.get_term("success").any()), f"success at step {step} with the robot holding still"
        too_long = {key: hold_lengths[hold] for key, hold in _holds(env).items() if hold_lengths.get(hold, 0) >= STEPS}
        assert not too_long, f"holds of {too_long} control steps are not shorter than STEPS ({STEPS}); raise STEPS"
    finally:
        env.close()
