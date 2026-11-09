# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""A typed experiment's Run builder, read before Kit starts: its presets as upstream composes them,
a Mimic Run refused, and a Run as the command line the record/teleop wrappers take."""

from __future__ import annotations

import dataclasses

import pytest

pytest.importorskip("isaaclab_arena.hydra.typed_experiment_loader")

from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg  # noqa: E402
from isaaclab_arena.hydra.typed_experiment_loader import load_arena_experiment_from_yaml  # noqa: E402
from isaaclab_arena.policy.policy_base import PolicyCfg  # noqa: E402

from dexbench_arena.evaluation.experiment_backend import (  # noqa: E402
    experiment_run_presets,
    refuse_experiment_mimic_runs,
)

YAML = """
shared:
  environment:
    type: dexbench_hotel_fb_e4_environment
  policy:
    type: zero_action
  environment_builder:
    presets: {shared}
runs:
  first: {{}}
  second:
    environment_builder:
      num_envs: 1
"""


@pytest.fixture
def backend_variable_unset(monkeypatch):
    monkeypatch.delenv("DEXBENCH_TEST_BACKEND", raising=False)


def _experiment(tmp_path, shared="NEWTON", name="experiment.yaml"):
    path = tmp_path / name
    path.write_text(YAML.format(shared=shared))
    return str(path)


@dataclasses.dataclass
class _Environment(ArenaEnvironmentCfg):
    pass


@dataclasses.dataclass
class _Policy(PolicyCfg):
    pass


@pytest.mark.parametrize(
    "backend_env, overrides",
    [
        (None, []),
        ("NEWTON", []),
        (None, ["shared.environment_builder.presets=null"]),
        (None, ["runs.first.environment_builder={presets:null}"]),
        ("NEWTON", ["runs.first={environment_builder:{presets:null,num_envs:2}}"]),
        (None, ["runs.second.environment_builder.presets=null", "runs.first.num_rebuilds=2"]),
        (None, ["~runs.first"]),
    ],
)
def test_the_presets_are_the_ones_upstream_composes(tmp_path, backend_variable_unset, monkeypatch, backend_env, overrides):
    if backend_env is not None:
        monkeypatch.setenv("DEXBENCH_TEST_BACKEND", backend_env)
    path = _experiment(tmp_path, "${oc.env:DEXBENCH_TEST_BACKEND,NEWTON}")
    upstream = load_arena_experiment_from_yaml(
        path,
        environment_cfg_types={"dexbench_hotel_fb_e4_environment": _Environment},
        policy_cfg_type_resolver=lambda _: _Policy,
        overrides=list(overrides),
    )
    expected = {name: run.environment_builder.presets for name, run in upstream.runs.items()}
    assert experiment_run_presets(path, overrides) == expected


@pytest.mark.parametrize(
    "overrides, refused",
    [
        ([], False),
        (["runs.second.environment_builder.mimic=true"], True),
        (["runs.second.environment_builder={mimic:true}"], True),
        (["shared.environment_builder.presets=NEWTON"], False),
        (["runs.second.environment_builder.mimic=true", "runs.second.environment.mimic=true"], True),
        (["runs.second.environment_builder.mimic=true", "~runs.second"], False),
    ],
    ids=["no-mimic", "run-override", "mapping-override", "unrelated", "both-sides", "deleted-run"],
)
def test_a_run_with_builder_mimic_is_refused(tmp_path, overrides, refused):
    argv = ["--experiment_config", _experiment(tmp_path), *overrides]
    if refused:
        with pytest.raises(SystemExit, match=r"\['second'\].*environment_builder\.mimic.*a Mimic env does not have"):
            refuse_experiment_mimic_runs(argv)
    else:
        refuse_experiment_mimic_runs(argv)


def test_a_legacy_json_experiment_is_left_alone(tmp_path):
    path = tmp_path / "jobs.json"
    path.write_text('{"jobs": []}')
    refuse_experiment_mimic_runs(["--experiment_config", str(path)])


def test_a_collection_takes_its_environment_from_the_experiment_run():
    """``record_demos.py ... --experiment db_r2_collect.yaml --run tc1`` builds the Run's environment,
    the launcher keeping the flags before the pair and the IK twin given after it replacing the Run's."""
    import runpy
    from pathlib import Path

    helper = runpy.run_path(str(Path(__file__).resolve().parents[3] / "scripts/imitation_learning/_dexbench_arena_script.py"))
    argv = ["--viz", "kit", "--dataset_file", "d.hdf5", "--experiment", "db_r2_collect.yaml", "--run", "tc1",
            "--embodiment", "dexbench_rby1_wuji2", "--teleop_device", "openxr"]
    assert helper["expand_experiment_run"](argv) == [
        "--viz", "kit", "--dataset_file", "d.hdf5", "--presets", "newton",
        "dexbench_retail_v1_environment", "--arm_mode", "dual_arm", "--task", "retrieve_beverage_from_clutter",
        "--scene_layout", "fc_260821", "--anchor", "Fridge", "--start_pose", "teleop_home",
        "--rby1_control_hz", "30", "--with_full_items", "--fridge_shelf3_raise_m", "0.065",
        "--r2_target_lane", "4", "--r2_stock_rule", "unique_target", "--r2_stock_fillers", "TC2,TC1",
        "--r2_target_sku", "TC1",
        "dexbench_rby1_wuji2.base_pose.enabled=true", "dexbench_rby1_wuji2.base_pose.sampler_cfg.low=[-0.02,-0.02,-0.0523599]",
        "dexbench_rby1_wuji2.base_pose.sampler_cfg.high=[0.02,0.02,0.0523599]",
        "--embodiment", "dexbench_rby1_wuji2", "--teleop_device", "openxr",
    ]
    joined = [token for token in argv if token not in ("--experiment", "db_r2_collect.yaml")]
    joined.insert(4, "--experiment=db_r2_collect.yaml")
    assert helper["expand_experiment_run"](joined) == helper["expand_experiment_run"](argv)
    with pytest.raises(SystemExit, match="followed by --run <name>"):
        helper["expand_experiment_run"](["--experiment=db_r2_collect.yaml", "--run=tc1"])
    lab_argv, _, env_name, tail = helper["_compute_lab_argv"]("record_demos.py", helper["expand_experiment_run"](argv))
    assert env_name == "dexbench_retail_v1_environment" and "--no-reset_sim_buffer_each_episode" in lab_argv
    assert tail[-4:] == ["--embodiment", "dexbench_rby1_wuji2", "--teleop_device", "openxr"]
    l2 = helper["expand_experiment_run"](["--experiment", "db_l2_collect.yaml", "--run", "train_1",
                                          "cage_l2_package.pose.enabled=false"])
    assert l2[l2.index("--cage_box_alt_scales") + 1] == "0.85,1.15"
    assert l2[-4:] == ["cage_l2_package.cage_stack.enabled=true", "cage_l2_package.cage_stack.sampler_cfg.low=[0.0,0.0,-0.01]",
                       "cage_l2_package.cage_stack.sampler_cfg.high=[1.0,0.08,0.01]", "cage_l2_package.pose.enabled=false"]


def test_a_false_field_is_the_negative_flag_and_a_mapping_is_refused(tmp_path):
    """``delivery_render_settings: false`` reaches the E2 parser as ``--no-delivery_render_settings``; a mapping has no
    command-line form."""
    from dexbench_arena.evaluation.experiment_backend import experiment_run_command_line

    path = tmp_path / "e.yaml"
    path.write_text("shared:\n  environment:\n    type: dexbench_logistics_center_e2_environment\n"
                    "    delivery_render_settings: false\n    record_joint_torque: true\nruns:\n  only: {}\n")
    argv = experiment_run_command_line(path, "only")
    assert "--no-delivery_render_settings" in argv and "--record_joint_torque" in argv
    path.write_text("shared:\n  environment:\n    type: dexbench_retail_v1_environment\n"
                    "    with_full_items: {a: 1}\nruns:\n  only: {}\n")
    with pytest.raises(ValueError, match="with_full_items"):
        experiment_run_command_line(path, "only")


def test_a_collection_run_names_the_rows_it_is_collected_from():
    """DB-R3's collect Runs replay their carton's batch-0 dump (environment_builder.recorded_variation_samples_path):
    the wrapper passes it as --collection_rows; a file on the command line (another batch's) replaces it."""
    import runpy
    from pathlib import Path

    from dexbench_arena.evaluation.experiment_backend import experiment_run_collection_rows

    repo = Path(__file__).resolve().parents[3]
    helper = runpy.run_path(str(repo / "scripts/imitation_learning/_dexbench_arena_script.py"))
    rows = "outputs/db_conditions/db_r3_collect/batch_0/dexbench_retail_e1_fc_r3t2.jsonl"
    configs = repo / "scripts/evaluation/experiment_configs"
    assert experiment_run_collection_rows(configs / "db_r3_collect.yaml", "r3t2") == rows
    assert experiment_run_collection_rows(configs / "db_r2_collect.yaml", "tc1") is None
    argv = ["--viz", "kit", "--experiment", "db_r3_collect.yaml", "--run", "r3t2", "--embodiment", "dexbench_rby1_wuji2"]
    expanded = helper["expand_experiment_run"](argv)
    assert expanded[:4] == ["--collection_rows", rows, "--viz", "kit"]
    batch_1 = rows.replace("batch_0", "batch_1")
    replaced = helper["expand_experiment_run"](["--collection_rows", batch_1, *argv])
    assert replaced.count("--collection_rows") == 1 and replaced[replaced.index("--collection_rows") + 1] == batch_1
