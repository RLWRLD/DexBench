# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""A typed experiment's Run builder fields, read before Kit starts."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from hydra import compose
from hydra.core.config_store import ConfigStore
from hydra.core.override_parser.overrides_parser import OverridesParser
from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
from isaaclab_arena.hydra.typed_experiment_loader import (
    _get_new_hydra_context_if_none_exists,
    load_experiment_run_definitions_from_yaml,
    split_shared_run_default_overrides,
)
from isaaclab_arena.utils.physics_backend import PhysicsBackend
from omegaconf import OmegaConf

from dexbench_arena.variations.collection_rows import variations_for_embodiment
from dexbench_arena.variations.variation_draws import dotted_items


def _reaches(override_key: str, field: str) -> bool:
    """Whether a Hydra override at ``override_key`` can change a Run's ``environment_builder.<field>``."""
    key = override_key.split(".")
    path = ("runs", None, "environment_builder", field)
    return len(key) <= len(path) and all(part == expected for part, expected in zip(key, path) if expected is not None)


def experiment_run_builder_values(yaml_path: str | Path, overrides: Sequence[str], field: str) -> dict[str, object]:
    """Each Run's ``environment_builder.<field>`` as upstream composes it, keyed by Run name."""
    shared_overrides, run_overrides = split_shared_run_default_overrides(list(overrides))
    runs = load_experiment_run_definitions_from_yaml(yaml_path, shared_overrides)
    experiment = {"runs": {}}
    for name, values in runs.items():
        run = OmegaConf.merge({"environment_builder": OmegaConf.structured(ArenaEnvBuilderCfg)}, values)
        value = OmegaConf.select(run, f"environment_builder.{field}", throw_on_resolution_failure=True)
        experiment["runs"][name] = {
            "environment_builder": OmegaConf.merge(OmegaConf.structured(ArenaEnvBuilderCfg), {field: value})
        }
    field_overrides = [
        override.input_line
        for override in OverridesParser.create().parse_overrides(run_overrides)
        if _reaches(override.key_or_group, field)
    ]
    config_name = "dexbench_experiment_backend_preflight"
    with _get_new_hydra_context_if_none_exists():
        ConfigStore.instance().store(name=config_name, node=OmegaConf.create(experiment))
        composed = compose(config_name=config_name, overrides=field_overrides)
    return {name: composed.runs[name].environment_builder[field] for name in runs if name in composed.runs}


def experiment_run_presets(yaml_path: str | Path, overrides: Sequence[str] = ()) -> dict[str, PhysicsBackend | None]:
    """Each Run's ``environment_builder.presets`` (its physics backend) as upstream composes it."""
    return experiment_run_builder_values(yaml_path, overrides, "presets")


def _typed_experiment(argv: Sequence[str]) -> tuple[str, list[str]] | None:
    """The YAML experiment and its Hydra overrides on the runner's ``argv``; None for a JSON one."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--experiment_config", "--eval_jobs_config", dest="experiment_config", default=None)
    args, rest = parser.parse_known_args(list(argv))
    if args.experiment_config is None or Path(args.experiment_config).suffix.lower() not in {".yaml", ".yml"}:
        return None
    return args.experiment_config, [
        token for token in rest if not token.startswith("-") and ("=" in token or token.startswith("~"))
    ]


def refuse_experiment_mimic_runs(argv: Sequence[str]) -> None:
    """Refuse a typed YAML experiment on ``argv`` with a Run whose builder sets ``mimic``."""
    typed = _typed_experiment(argv)
    if typed is None:
        return
    on_mimic = [name for name, mimic in experiment_run_builder_values(*typed, "mimic").items() if mimic]
    if on_mimic:
        raise SystemExit(
            f"Runs {on_mimic} of {typed[0]} set environment_builder.mimic. Arena's experiment runner "
            "evaluates on its RL env (episode_recorder, metrics), which a Mimic env does not have; "
            "drop mimic from these Runs, and run Mimic workflows through annotate_demos.py / "
            "generate_dataset.py with --mimic."
        )


def _hydra_value(value) -> str:
    """``value`` as a Hydra override writes it: lowercase booleans, lists in brackets."""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, list):
        return "[" + ",".join(_hydra_value(item) for item in value) + "]"
    return str(value)


def experiment_run_command_line(yaml_path: str | Path, run: str, embodiment: str | None = None) -> list[str]:
    """One Run of a typed experiment as the command line the record/teleop wrappers take."""
    runs = load_experiment_run_definitions_from_yaml(yaml_path)
    if run not in runs:
        raise ValueError(f"{yaml_path} has no Run {run!r}; its Runs are {sorted(runs)}")
    values = OmegaConf.to_container(OmegaConf.create(runs[run]), resolve=True)
    environment, builder = values["environment"], values.get("environment_builder") or {}
    unsupported = sorted(set(builder) - {"presets", "recorded_variation_samples_path"})
    if unsupported:
        raise ValueError(f"Run {run!r} of {yaml_path} sets environment_builder {unsupported}; "
                         "a teleop or recording session takes only presets from the experiment")
    presets = experiment_run_presets(yaml_path)[run]
    argv = ["--presets", str(presets)] if presets is not None else []
    argv.append(environment.pop("type"))
    for key, value in environment.items():
        if value is True:
            argv.append(f"--{key}")
        elif value is False:
            argv.append(f"--no-{key}")
        elif value is None:
            argv.extend((f"--{key}", "none"))
        elif isinstance(value, dict):
            raise ValueError(f"Run {run!r} of {yaml_path}: environment.{key}={value!r} has no command-line form; "
                             "set it in the experiment so the wrapper can pass it as a flag")
        else:
            text = ",".join(str(item) for item in value) if isinstance(value, list) else str(value)
            argv.extend((f"--{key}={text}",) if text.startswith("-") else (f"--{key}", text))
    variations = dict(dotted_items(values.get("variations") or {}))
    if embodiment is not None and environment.get("embodiment") not in (None, embodiment):
        variations = variations_for_embodiment(variations, environment["embodiment"], embodiment)
    argv += [f"{key}={_hydra_value(value)}" for key, value in variations.items()]
    return argv


def experiment_run_collection_rows(yaml_path: str | Path, run: str) -> str | None:
    """The rows file a collection Run is collected from (its ``environment_builder.recorded_variation_samples_path``,
    as upstream composes it), or None for a Run that draws its start states live."""
    rows = experiment_run_builder_values(yaml_path, (), "recorded_variation_samples_path")
    if run not in rows:
        raise ValueError(f"{yaml_path} has no Run {run!r}; its Runs are {sorted(rows)}")
    return rows[run]
