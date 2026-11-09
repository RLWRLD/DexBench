# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""The variation scenarios are Arena experiments: the three YAMLs load through
Arena's own loader, so a key the environment config does not declare, or a
value of the wrong shape, fails here and not in the GUI."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytest.importorskip("isaaclab_arena.assets.register", reason="needs IsaacLab-Arena importable")
loader = pytest.importorskip(
    "isaaclab_arena.evaluation.arena_experiment_config_loader", reason="needs IsaacLab-Arena on the path"
)

import dexbench_arena.environments  # noqa: E402, F401  -- registers the DexBench environment types the YAMLs name
from isaaclab_arena.assets.registries import EnvironmentRegistry  # noqa: E402
from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg  # noqa: E402
from isaaclab_arena.utils.physics_backend import PhysicsBackend  # noqa: E402

from dexbench_arena.environments.manipulation.e3_assembly_line_environment import (  # noqa: E402
    DexBenchManufacturingE3M1Environment,
)
from dexbench_arena.environments.manipulation.e4_hotel_fb_environment import DexBenchHotelFBE4Environment  # noqa: E402
from dexbench_arena.environments.manipulation.scene_environment_base import (  # noqa: E402
    check_backend_supported,
    check_builder_agrees,
    DexBenchEnvironmentCfg,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS = REPO_ROOT / "scripts" / "evaluation" / "experiment_configs"
EXPECTED = {
    "e1_retail_variations.yaml": ("dexbench_retail_v1_environment", 9),
    "e2_logistics_variations.yaml": ("dexbench_logistics_center_e2_environment", 9),
    "e4_hotel_fb_variations.yaml": ("dexbench_hotel_fb_e4_environment", 7),
}


@pytest.mark.parametrize("filename", sorted(EXPECTED))
def test_the_variation_experiments_load_through_arenas_loader(filename):
    env_name, run_count = EXPECTED[filename]
    experiment = loader.load_arena_experiment_from_config_file(str(CONFIGS / filename), device="cpu")
    assert len(experiment.runs) == run_count
    env_cls = EnvironmentRegistry().get_component_by_name(env_name)
    expected_cfg_type = EnvironmentRegistry().get_environment_cfg_type(env_cls)
    for name, run in experiment.runs.items():
        assert type(run.environment) is expected_cfg_type, name
        assert run.environment_builder.num_envs >= 1, name


def _field_paths(node, prefix: str = ""):
    """Every key path in a YAML node; a list's items share their list's path."""
    if isinstance(node, dict):
        for key, value in node.items():
            path = f"{prefix}.{key}"
            yield path
            yield from _field_paths(value, path)
    elif isinstance(node, list):
        for value in node:
            yield from _field_paths(value, f"{prefix}[]")


def _generated_runs() -> list[tuple[Path, str]]:
    """Runs of the generated benchmark files that together carry each field path of every one of them, per
    environment type: the largest Run first, then each Run that adds a path no chosen Run has."""
    candidates = []
    for path in sorted((CONFIGS / "benchmark").glob("*.yaml")):
        experiment = yaml.safe_load(path.read_text())
        shared = experiment.get("shared") or {}
        for name, run in experiment["runs"].items():
            kind = ((run or {}).get("environment") or {}).get("type") or shared["environment"]["type"]
            fields = {f"{kind}:{field}" for field in _field_paths({"shared": shared, "run": run})}
            candidates.append((path, name, fields))
    covered, chosen = set(), []
    for path, name, fields in sorted(candidates, key=lambda candidate: -len(candidate[2])):
        if fields - covered:
            chosen.append((path, name))
            covered |= fields
    return chosen


HAND_WRITTEN = sorted(path for path in CONFIGS.rglob("*.yaml") if path.parent.name != "benchmark")
GENERATED_RUNS = _generated_runs()


def _check_runs(experiment, label: str) -> None:
    for name, run in experiment.runs.items():
        factory = EnvironmentRegistry().get_factory_type_for_cfg(run.environment)
        try:
            check_builder_agrees(run.environment, run.environment_builder)
            check_backend_supported(factory, run.environment_builder.presets or PhysicsBackend.NEWTON)
        except ValueError as error:
            pytest.fail(f"{label} Run {name!r}: {error}")


@pytest.mark.parametrize("path, run", GENERATED_RUNS, ids=[f"{path.stem}:{run}" for path, run in GENERATED_RUNS])
def test_every_field_of_the_generated_runs_loads_through_arenas_loader(path, run, tmp_path):
    """One Run of a generated file, alone, through Arena's typed loader: an undeclared field or a value of the
    wrong type fails here. The build's agreement checks run on it too."""
    experiment = yaml.safe_load(path.read_text())
    experiment["runs"] = {run: experiment["runs"][run]}
    alone = tmp_path / path.name
    alone.write_text(yaml.safe_dump(experiment, sort_keys=False))
    _check_runs(loader.load_arena_experiment_from_config_file(str(alone), device="cpu"), path.name)


@pytest.mark.parametrize("path", HAND_WRITTEN, ids=lambda path: path.name)
def test_every_shipped_run_sets_num_envs_and_mimic_on_both_sides(path):
    """The build refuses a Run whose environment and builder disagree, or whose backend is not Newton
    (the environments' default, when the Run names none); no shipped one does. This is where the hand-written
    files load through Arena's typed loader."""
    _check_runs(loader.load_arena_experiment_from_config_file(str(path), device="cpu"), path.name)


def test_a_builder_that_disagrees_with_the_environment_is_refused():
    check_builder_agrees(DexBenchEnvironmentCfg(num_envs=4), ArenaEnvBuilderCfg(num_envs=4))
    with pytest.raises(ValueError, match="num_envs"):
        check_builder_agrees(DexBenchEnvironmentCfg(), ArenaEnvBuilderCfg(num_envs=2))
    with pytest.raises(ValueError, match="mimic"):
        check_builder_agrees(DexBenchEnvironmentCfg(mimic=True), ArenaEnvBuilderCfg())


@pytest.mark.parametrize("environment", [DexBenchManufacturingE3M1Environment, DexBenchHotelFBE4Environment],
                         ids=lambda cls: cls.name)
def test_every_environment_refuses_a_build_off_newton(environment):
    """Every DexBench scene runs on Newton only."""
    check_backend_supported(environment, PhysicsBackend.NEWTON)
    check_backend_supported(environment, "newton")
    with pytest.raises(ValueError, match="runs on Newton only, and this build resolved physx"):
        check_backend_supported(environment, PhysicsBackend.PHYSX)


def _object_axis(environment):
    """The object axis of the Run's task, from its environment class; None for a task without one."""
    factory = EnvironmentRegistry().get_factory_type_for_cfg(environment)
    return factory.object_axes.get(getattr(environment, "task", None))


CELL_OBJECTS = {"id": "train"}
TASK_EXPERIMENTS = sorted(CONFIGS.rglob("db_*.yaml"))


def _kind(path: Path) -> tuple[str, str | None]:
    """``(kind, cell)`` of ``db_<task>_<kind>[_<cell>].yaml``; a sensitivity file is ``("sensitivity", None)``."""
    rest = path.stem.split("_", 2)[2]
    kind, _, cell = rest.partition("_")
    return kind, (cell or ("id" if kind == "collect" else None))


def _load(path: Path):
    return loader.load_arena_experiment_from_config_file(str(path), device="cpu")


def _plain_environments(path: Path) -> dict:
    """Each Run's environment config built from the YAML's plain fields (shared, then the Run's own), without
    Arena's loader, which takes about a minute for a whole generated file; their fields load through it in
    test_every_field_of_the_generated_runs_loads_through_arenas_loader."""
    experiment = yaml.safe_load(path.read_text())
    shared = (experiment.get("shared") or {}).get("environment") or {}
    environments = {}
    for name, run in experiment["runs"].items():
        fields = {**shared, **((run or {}).get("environment") or {})}
        factory = EnvironmentRegistry().get_component_by_name(fields.pop("type"))
        environments[name] = EnvironmentRegistry().get_environment_cfg_type(factory)(**fields)
    return environments


@pytest.mark.parametrize("path", TASK_EXPERIMENTS, ids=lambda path: str(path.relative_to(CONFIGS)))
def test_every_task_experiment_run_is_an_object_of_its_axis(path):
    """Each Run names a listed object that fits, at the split's fixture profile; a hand-written cell runs exactly its objects."""
    kind, cell = _kind(path)
    if path.parent == CONFIGS:
        environments = {name: run.environment for name, run in _load(path).runs.items()}
    else:
        environments, cell = _plain_environments(path), None
    keys = []
    for name, environment in environments.items():
        axis = _object_axis(environment)
        assert axis is not None, f"{path.name} Run {name!r}: its task has no object axis"
        task_object, _fit = axis.resolve(environment)
        keys.append(task_object.key)
        if kind != "sensitivity":
            assert axis.profile_differences(environment) == {}, f"{path.name} Run {name!r}"
    if cell is not None:
        objects = CELL_OBJECTS[cell]
        expected = [axis.default] * len(keys) if objects == "default" else list(axis.keys(objects))
        assert keys == expected, path.name

