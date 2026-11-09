# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Physics changes must remain visible across planning, generation and replay."""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import warnings

import h5py
import numpy as np
import pytest

from dexbench_arena.evaluation.evaluation_signature import NEWTON_SHAPE_OVERRIDES_ATTRIBUTE


SCRIPTS = Path(__file__).resolve().parents[3] / "scripts/imitation_learning"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


from dexbench_arena.recording import physics_runtime as physics  # noqa: E402
recording = importlib.import_module("dexbench_arena.recording.rerecord_metadata")


class NewtonCfg(SimpleNamespace):
    pass


class MJWarpSolverCfg(SimpleNamespace):
    pass


def config(substeps=2, dt=1 / 60, decimation=2):
    return SimpleNamespace(
        sim=SimpleNamespace(dt=dt, gravity=(0.0, 0.0, -9.81), use_newton_actuators=False,
                            physics=NewtonCfg(
                                num_substeps=substeps, collision_decimation=0, use_cuda_graph=True,
                                deterministic_mode="not_guaranteed",
                                solver_cfg=MJWarpSolverCfg(
                                    solver_type="mujoco_warp", solver="newton", integrator="implicitfast",
                                    update_data_interval=2, iterations=100, ls_iterations=15, impratio=10.0,
                                    cone="elliptic", tolerance=1e-6, ccd_iterations=15000,
                                    nconmax=16384, njmax=65536, use_mujoco_cpu=False, use_mujoco_contacts=False,
                                    disable_contacts=False, disable_sensors=False, default_actuator_gear=None,
                                ),
                                default_shape_cfg=SimpleNamespace(margin=0.0, gap=0.002, ke=2500.0, kd=100.0, mu=1.0),
                            )),
        decimation=decimation,
    )


def runtime(cfg, stiffness=400.0):
    """physics_runtime_metadata plus the robot_drive a made environment adds."""
    return {**physics.physics_runtime_metadata(cfg), "robot_drive": {"joints": {"stiffness": [stiffness]}}}


def test_replay_restores_recorded_clock_without_changing_solver_settings():
    cfg = config(substeps=8)
    solver = copy.deepcopy(cfg.sim.physics)
    physics.apply_recorded_timing(cfg, {"sim_args": {"dt": 1 / 60, "decimation": 1}})
    assert cfg.sim.dt * cfg.decimation == pytest.approx(1 / 60)
    assert cfg.sim.physics == solver
    physics.apply_recorded_timing(cfg, {"sim_args": {"dt": 1 / 60, "decimation": 2}})
    assert cfg.sim.dt * cfg.decimation == pytest.approx(1 / 30)
    with pytest.raises(ValueError, match="sim_args"):
        physics.apply_recorded_timing(cfg, {})


def test_newton_settings_are_json_numbers_booleans_and_null_with_explicit_determinism():
    cfg = config(substeps=np.int32(8), dt=np.float64(1 / 60))
    runtime = physics.physics_runtime_metadata(cfg)
    decoded = json.loads(json.dumps(runtime, allow_nan=False))
    assert decoded == runtime
    assert decoded["control_dt"] == pytest.approx(1 / 30)
    assert decoded["physics_backend"] == "NewtonCfg"
    settings = decoded["physics_config"]
    assert settings["num_substeps"] == 8 and type(settings["num_substeps"]) is int
    assert settings["dt"] == pytest.approx(1 / 60) and settings["decimation"] == 2
    assert settings["deterministic_mode"] == "not_guaranteed"
    assert settings["solver"]["disable_sensors"] is False
    assert settings["solver"]["default_actuator_gear"] is None
    assert settings["solver"]["update_data_interval"] == 2
    assert settings["solver"]["nconmax"] == 16384
    assert settings["solver"]["njmax"] == 65536
    assert settings["default_shape"]["gap"] == 0.002


@pytest.mark.parametrize("changed, expected", [
    (config(substeps=8), {"num_substeps"}),
    (config(dt=1 / 120, decimation=4), {"dt", "decimation"}),
])
def test_generation_detects_physics_changes_even_when_control_dt_is_unchanged(changed, expected):
    source = {"dexbench_recording": physics.physics_runtime_metadata(config())}
    before = copy.deepcopy(source)
    current = physics.physics_runtime_metadata(changed)
    with pytest.warns(RuntimeWarning, match="Generation: physics settings differ"):
        result = physics.compare_source_physics(source, current, context="Generation")
    assert result["status"] == "mismatch"
    assert set(result["differences"]) == expected
    assert source == before


def test_replay_compares_latest_execution_instead_of_the_older_snapshot():
    current = runtime(config(substeps=8))
    metadata = {"dexbench_recording": physics.physics_runtime_metadata(config(substeps=2)),
                "dexbench_rerecord": {"runtime": copy.deepcopy(current)}}
    with warnings.catch_warnings(record=True) as captured:
        result = physics.compare_source_physics(metadata, current, context="Replay")
    assert not captured
    assert result == {"status": "match", "source": "dexbench_rerecord.runtime", "differences": {}}
    current["physics_config"]["solver"]["impratio"] = 1.0
    with pytest.warns(RuntimeWarning, match="solver.impratio"):
        result = physics.compare_source_physics(metadata, current, context="Replay")
    assert result["differences"] == {"solver.impratio": {"source": 10.0, "current": 1.0}}


def test_replay_detects_a_change_in_native_contact_generation():
    cfg = config()
    cfg.sim.physics.solver_cfg.enable_multiccd = False
    source = {"dexbench_rerecord": {"runtime": physics.physics_runtime_metadata(cfg)}}
    cfg.sim.physics.solver_cfg.enable_multiccd = True
    with pytest.warns(RuntimeWarning, match="solver.enable_multiccd"):
        result = physics.compare_source_physics(source, physics.physics_runtime_metadata(cfg), context="Replay")
    assert result["differences"] == {"solver.enable_multiccd": {"source": False, "current": True}}


@pytest.mark.parametrize("metadata", [
    {},
    {"dexbench_recording": {"physics_backend": "NewtonCfg", "control_dt": 1 / 30}},
    {"dexbench_recording": physics.physics_runtime_metadata(config()),
     "dexbench_rerecord": {"runtime": {"physics_backend": "NewtonCfg"}}},
])
def test_legacy_input_is_allowed_but_explicitly_unverified(metadata):
    with pytest.warns(RuntimeWarning, match="allowing the input.*unverified"):
        result = physics.compare_source_physics(metadata, physics.physics_runtime_metadata(config()), context="Replay")
    assert result["status"] == "unverified"
    assert result["reason"] == "missing_physics_config"


def test_robot_gains_are_compared_and_a_source_without_them_is_unverified():
    source = {"dexbench_recording": runtime(config(), stiffness=400.0)}
    with pytest.warns(RuntimeWarning, match="robot_drive.joints.stiffness"):
        result = physics.compare_source_physics(source, runtime(config(), stiffness=40.0), context="Replay")
    assert result["status"] == "mismatch"
    assert result["differences"] == {"robot_drive.joints.stiffness": {"source": [400.0], "current": [40.0]}}
    del source["dexbench_recording"]["robot_drive"]
    with pytest.warns(RuntimeWarning, match="no robot_drive.*unverified"):
        result = physics.compare_source_physics(source, runtime(config()), context="Replay")
    assert result["status"] == "unverified" and result["reason"] == "missing_robot_drive"


def test_determinism_and_contact_sensor_settings_are_compared():
    current = physics.physics_runtime_metadata(config())
    source = {"dexbench_recording": copy.deepcopy(current)}
    current["physics_config"]["deterministic_mode"] = "run_to_run"
    current["physics_config"]["solver"]["disable_sensors"] = True
    with pytest.warns(RuntimeWarning, match="deterministic_mode"):
        result = physics.compare_source_physics(source, current, context="Replay")
    assert set(result["differences"]) == {"deterministic_mode", "solver.disable_sensors"}


def test_mpm_solver_sequence_survives_export_and_detects_order_changes():
    cfg = config()
    cfg.sim.physics.solver_cfg = SimpleNamespace(solver_type="mpm", solver=("cr", "gs"))
    current = runtime(cfg)
    saved = json.loads(json.dumps({"dexbench_recording": current}, allow_nan=False))
    assert saved["dexbench_recording"]["physics_config"]["solver"]["solver"] == ["cr", "gs"]
    assert physics.compare_source_physics(saved, current, context="Replay")["status"] == "match"
    cfg.sim.physics.solver_cfg.solver = ("gs", "cr")
    with pytest.warns(RuntimeWarning, match="solver.solver"):
        comparison = physics.compare_source_physics(saved, physics.physics_runtime_metadata(cfg), context="Replay")
    assert comparison["differences"] == {"solver.solver": {"source": ["cr", "gs"], "current": ["gs", "cr"]}}


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_nonfinite_physics_values_are_rejected_before_export(invalid):
    cfg = config()
    cfg.sim.physics.solver_cfg.impratio = invalid
    with pytest.raises(ValueError):
        physics.physics_runtime_metadata(cfg)


def test_native_metadata_export_preserves_runtime_and_comparison(tmp_path):
    source, output = tmp_path / "source.hdf5", tmp_path / "replayed.hdf5"
    source_runtime = physics.physics_runtime_metadata(config(substeps=2))
    source_metadata = {"dexbench_recording": source_runtime,
                       "dexbench_rerecord": {"runtime": source_runtime}}
    for path in (source, output):
        with h5py.File(path, "w") as handle:
            data = handle.create_group("data")
            data.attrs["env_args"] = json.dumps(source_metadata if path == source else {})
            data.create_group("demo_0")
    current = physics.physics_runtime_metadata(config(substeps=8))
    with pytest.warns(RuntimeWarning, match="num_substeps"):
        current["source_physics_check"] = physics.compare_source_physics(source_metadata, current, context="Replay")
    recording.preserve_rerecord_metadata(
        source, output, source_metadata=source_metadata, calibration={}, episodes={"demo_0": {}}, git_sha="test",
        runtime=current,
    )
    saved = recording.read_source_metadata(output)
    assert saved["dexbench_recording"]["physics_config"]["num_substeps"] == 2
    assert saved["dexbench_rerecord"]["runtime"]["physics_config"]["num_substeps"] == 8
    assert saved["dexbench_rerecord"]["runtime"]["source_physics_check"]["status"] == "mismatch"


def test_per_shape_contact_an_environment_installed_is_recorded_and_compared():
    """DB-R2's basket and drink contact is set on the model builder, not in sim.physics: it is recorded from the
    cfg attribute the environment sets, and a recording made without it replays as a difference."""
    before = physics.physics_runtime_metadata(config())
    assert "shape_overrides" not in before["physics_config"]
    cfg = config()
    setattr(cfg, NEWTON_SHAPE_OVERRIDES_ATTRIBUTE, {"shape_contact": {"r2_basket_contact": {"solref": (0.005, 2.0),
                                                                                          "solmix": 1000.0}}})
    after = physics.physics_runtime_metadata(cfg)
    assert after["physics_config"]["shape_overrides"]["shape_contact"]["r2_basket_contact"]["solref"] == [0.005, 2.0]
    with pytest.warns(RuntimeWarning, match="shape_overrides"):
        result = physics.compare_source_physics({"dexbench_rerecord": {"runtime": before}}, after, context="Replay")
    assert "shape_overrides.shape_contact.r2_basket_contact.solmix" in result["differences"]
