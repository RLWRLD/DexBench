# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Each EA task's episode cap reaches the composed env as a truncation, and every evaluation and
sensitivity experiment states its cap (Kit tier): a benchmark file the cap its manifest gives the task
(dexbench_arena.benchmark.spec's rule), a hand-written one the task's own default.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("simulation_app")

CONFIGS = Path(__file__).resolve().parents[3] / "scripts" / "evaluation" / "experiment_configs"
EXPERIMENT_FILES = sorted(
    [
        *CONFIGS.rglob("db_*_eval_*.yaml"),
        *(path for path in CONFIGS.rglob("db_*_sensitivity*.yaml") if not path.stem.endswith("_conditions")),
    ]
)


_MANIFEST = json.loads((CONFIGS / "benchmark" / "benchmark_manifest.json").read_text())
BENCHMARK_CAPS_S = {
    name: entry["cap_s"]
    for entry in _MANIFEST["tasks"].values()
    for name in [*(cell["eval"] for cell in entry["cells"].values()), entry["sensitivity"]["eval"]]
}

E1 = ["dexbench_retail_v1_environment", "--scene_layout", "fc_260821"]
R2 = E1 + ["--task", "retrieve_beverage_from_clutter", "--anchor", "Fridge", "--embodiment", "dexbench_rby1_wuji2"]

R3 = E1 + ["--task", "place_box_in_slot", "--embodiment", "dexbench_rby1_wuji2", "--arm_mode", "right"]

L2 = ["dexbench_logistics_center_e2_environment", "--task", "lift_to_conveyor", "--anchor", "A1_CAGE",
      "--embodiment", "dexbench_rby1_wuji2"]

H1 = ["dexbench_hotel_fb_e4_environment", "--task", "stack_plates", "--embodiment", "dexbench_rby1_wuji2_joint",
      "--arm_mode", "dual_arm"]

L1 = ["dexbench_logistics_e2_l1_environment", "--task", "pick_polybag_from_conveyor", "--embodiment",
      "dexbench_rby1_wuji2_joint"]

CAPS = {
    "r2-collection": (R2, 30.0),
    "r2-strict": (R2 + ["--success_tier", "strict"], 36.0),
    "r3-collection": (R3, 60.0),
    "l2": (L2, 60.0),
    "h1": (H1, 60.0),
    "l1": (L1, 600.0),
}


@pytest.mark.parametrize("argv, cap_s", CAPS.values(), ids=CAPS.keys())
def test_the_env_runs_to_the_task_cap_and_reports_it_as_a_truncation(argv, cap_s):
    from dexbench_arena.cli import get_arena_builder_from_cli, get_dexbench_arena_cli_parser
    from dexbench_arena.evaluation.env_server import episode_cap_steps

    env_cfg = get_arena_builder_from_cli(get_dexbench_arena_cli_parser().parse_args(argv)).compose_manager_cfg()[0]
    step_dt = env_cfg.sim.dt * env_cfg.decimation
    assert env_cfg.episode_length_s == cap_s
    assert math.ceil(env_cfg.episode_length_s / step_dt) == round(cap_s * 30)
    assert env_cfg.terminations.time_out.time_out is True
    assert episode_cap_steps(env_cfg) == round(cap_s * 30)


@pytest.mark.parametrize("path", EXPERIMENT_FILES, ids=lambda path: path.name)
def test_each_stated_experiment_cap_matches_its_task_and_composed_step_budget(path):
    import dexbench_arena.environments  # noqa: F401 -- registers the environment types the experiments name
    from isaaclab_arena.evaluation.arena_experiment_config_loader import load_arena_experiment_from_config_file
    from isaaclab_arena.evaluation.run_execution import build_arena_builder_from_run_cfg

    from dexbench_arena.evaluation.env_server import episode_cap_steps

    experiment = load_arena_experiment_from_config_file(str(path), device="cpu")
    seen = set()
    for name, run in experiment.runs.items():
        cap_s = run.environment.episode_length_s
        assert cap_s is not None, name
        key = (
            type(run.environment), run.environment.task, getattr(run.environment, "success_tier", None),
            getattr(run.environment, "strict_hold_s", None), cap_s,
        )
        if key in seen:
            continue
        seen.add(key)
        written = dataclasses.replace(
            run,
            environment=dataclasses.replace(run.environment, enable_cameras=False, wrist_cameras=False,
                                            benchmark_replay=False),
            environment_builder=dataclasses.replace(run.environment_builder, recorded_variation_samples_path=None),
            recorded_variation_samples_paths=None,
            num_rebuilds=1,
            variations={},
        )
        inherited = dataclasses.replace(
            written, environment=dataclasses.replace(written.environment, episode_length_s=None)
        )
        task_cfg = build_arena_builder_from_run_cfg(inherited).compose_manager_cfg()[0]
        if path.name in BENCHMARK_CAPS_S:
            assert cap_s == BENCHMARK_CAPS_S[path.name], name
        else:
            assert task_cfg.episode_length_s == cap_s, name

        env_cfg = build_arena_builder_from_run_cfg(written).compose_manager_cfg()[0]
        assert env_cfg.episode_length_s == cap_s, name
        step_dt = env_cfg.sim.dt * env_cfg.decimation
        assert math.ceil(env_cfg.episode_length_s / step_dt) == round(cap_s * 30) == episode_cap_steps(env_cfg), name
        assert env_cfg.terminations.time_out.time_out is True, name
