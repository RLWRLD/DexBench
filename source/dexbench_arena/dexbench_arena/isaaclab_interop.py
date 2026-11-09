# SPDX-License-Identifier: Apache-2.0
"""Register a DexBench environment from inside an Isaac Lab script."""

from __future__ import annotations

import argparse
from pathlib import Path

HEAD_ARGV: list[str] | None = None
"""Everything before the environment name (arena/app flags plus Lab-only flags),
with any tail flags the wrapper forwarded to the Lab side appended so both sides
agree on them."""

ENVIRONMENT_NAME: str | None = None
"""The positional DexBench environment name the wrapper split the argv at."""

TAIL_ARGV: list[str] | None = None
"""Everything after the environment name: that environment's own flags."""

IM_SCRIPT_NAME: str | None = None
GENERATION = None
"""The Mimic generation hook generate_dataset.py sets; None for every other script. The callback takes the
source demos' origin from it before the build and has it configure the built cfg."""
COLLECTION = None
"""The ``CollectionRows`` a session collects from, a recorded start-state dump (the wrapper's
``--collection_rows``; ``variations.collection_rows``); None otherwise."""

BUILDER = None
"""The builder the callback built the environment with, once it has run: what the
wrapper's recording provenance reads the task and the embodiment from."""


class _LiveKwarg:
    """Carry one runtime env kwarg through gym.make's kwargs deepcopy uncopied."""

    def __init__(self, value):
        self.value = value

    def __deepcopy__(self, memo):
        return self.value


def _split_variation_overrides(tail_remaining: list[str]) -> tuple[list[str], list[str]]:
    """Split the environment tail's leftovers into Arena variation overrides and the rest."""
    overrides, passthrough = [], []
    for token in tail_remaining:
        (overrides if "=" in token and not token.startswith("--") else passthrough).append(token)
    return overrides, passthrough


def environment_registration_callback() -> list[str]:
    """Build and gym-register the DexBench environment the wrapper named."""
    assert ENVIRONMENT_NAME is not None, (
        "dexbench_arena.isaaclab_interop was invoked without the DexBench wrapper: "
        "HEAD_ARGV/ENVIRONMENT_NAME/TAIL_ARGV are unset. Launch through "
        "scripts/imitation_learning/<name>.py rather than the Isaac Lab script directly."
    )

    from isaaclab_arena.assets.registries import EnvironmentRegistry

    from dexbench_arena.cli import ensure_dexbench_environments_registered, get_arena_builder_from_cli

    from isaaclab_arena.environments.isaaclab_interop import _purge_leaked_isaaclab_assets_presets

    _purge_leaked_isaaclab_assets_presets()

    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser

    head_parser = get_isaaclab_arena_cli_parser()
    head_args, head_remaining = head_parser.parse_known_args(HEAD_ARGV or [])

    ensure_dexbench_environments_registered()
    factory_type = EnvironmentRegistry().get_component_by_name(ENVIRONMENT_NAME)
    tail_parser = argparse.ArgumentParser(prog=ENVIRONMENT_NAME, add_help=False)
    factory_type.add_cli_args(tail_parser)
    tail_args, tail_remaining = tail_parser.parse_known_args(TAIL_ARGV or [])

    merged = argparse.Namespace(**{**vars(head_args), **vars(tail_args)})
    merged.example_environment = ENVIRONMENT_NAME
    if GENERATION is not None:
        merged.source_demo_origin = GENERATION.source_demo_origin(source_dataset())

    from isaaclab_arena.utils.hydra_overrides import assert_hydra_overrides

    variation_overrides, tail_passthrough = _split_variation_overrides(tail_remaining)
    assert_hydra_overrides(variation_overrides, tail_parser)
    remaining = head_remaining + tail_passthrough
    global BUILDER
    builder = BUILDER = get_arena_builder_from_cli(merged, hydra_overrides=variation_overrides)
    if COLLECTION is not None:
        _replay_collection_rows(builder)
    validate = getattr(builder.arena_env.task, "validate_imitation_dataset", None)
    if validate is not None:
        validate(IM_SCRIPT_NAME, HEAD_ARGV or [], TAIL_ARGV or [])
    _check_source_dataset(builder.arena_env)
    name, cfg, env_kwargs = builder.build_registered()
    if COLLECTION is not None:
        _hold_collection_rows(builder, env_kwargs)
    if GENERATION is not None:
        GENERATION.configure(cfg, merged, builder.arena_env.embodiment)
    _forward_env_kwargs(name, env_kwargs)
    return remaining


def _replay_collection_rows(builder) -> None:
    """Point ``builder`` at the collection's rows and record each take's row."""
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTermCfg

    from dexbench_arena.evaluation.replay_guards import ReplayGuardError
    from dexbench_arena.variations.collection_rows import EPISODE_RECORD_KEY, embodiment_mismatches, record_collection_row

    rows = COLLECTION
    settings = getattr(builder.arena_env, "dexbench_settings", None)
    differences = rows.build_differences(settings.cfg) if settings is not None else ["not a DexBench environment"]
    embodiments = [asset.name for asset in AssetRegistry().get_assets_by_tag("embodiment")]
    differences += embodiment_mismatches(rows.path, builder.arena_env.embodiment.name, embodiments)
    if differences:
        raise ReplayGuardError(f"[DexBench collection rows] {rows.path} is not this build's:\n  - " + "\n  - ".join(differences))
    builder.cfg.recorded_variation_samples_path = rows.path
    builder.arena_env.episode_recorder_terms[EPISODE_RECORD_KEY] = EpisodeRecorderTermCfg(func=record_collection_row)


def _hold_collection_rows(builder, env_kwargs: dict) -> None:
    """Make the build's replay hold one row until a demo of it is kept, after refusing rows that would leave a draw live."""
    from dexbench_arena.evaluation.replay_guards import (
        ReplayGuardError,
        benchmark_replay_problems,
        enabled_variation_keys,
        read_episode_records,
        variation_ranges,
    )
    from dexbench_arena.variations.collection_rows import hold_rows

    rows = COLLECTION
    replayed = builder.cfg.recorded_variation_samples_path
    variations = builder.get_all_variations()
    problems = benchmark_replay_problems(builder.arena_env.dexbench_settings.object_identity, read_episode_records(replayed),
                                         enabled_variation_keys(variations), variation_ranges(variations))
    if problems:
        raise ReplayGuardError(f"[DexBench collection rows] {replayed}:\n  - " + "\n  - ".join(problems))
    hold_rows(env_kwargs.get("variation_replay_scheduler"), rows)


SOURCE_DATASET_FLAGS = {
    "replay_demos.py": ("--dataset_file", "Replaying demos"),
    "annotate_demos.py": ("--input_file", "Annotating Mimic source demos"),
    "generate_dataset.py": ("--input_file", "Generating Mimic data"),
}


def source_dataset() -> Path | None:
    """The script's source dataset, if it names one that exists; the last of repeated flags, as argparse reads it."""
    flag, _ = SOURCE_DATASET_FLAGS.get(IM_SCRIPT_NAME, (None, None))
    if flag is None:
        return None
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument(flag, dest="path")
    path = parser.parse_known_args(HEAD_ARGV or [])[0].path
    return Path(path) if path and Path(path).is_file() else None


def _check_source_dataset(arena_env) -> None:
    """Check replay compatibility and the selected Mimic class's generation-source contract."""
    _, what = SOURCE_DATASET_FLAGS.get(IM_SCRIPT_NAME, (None, None))
    settings = getattr(arena_env, "dexbench_settings", None)
    path = source_dataset()
    if settings is not None and path is not None:
        settings.check_replay_source(str(path), what=what)
    embodiment = getattr(arena_env, "embodiment", None)
    if IM_SCRIPT_NAME == "generate_dataset.py" and path is not None and embodiment is not None:
        validate = getattr(embodiment.get_mimic_env(), "validate_mimic_source_dataset", None)
        if validate is not None:
            validate(path)


def _forward_env_kwargs(name: str, env_kwargs: dict) -> None:
    """Put ``build_registered``'s runtime env kwargs into gym's registered kwargs."""
    import gymnasium as gym

    registered = gym.spec(name).kwargs
    for key, value in env_kwargs.items():
        if value is not None:
            registered[key] = _LiveKwarg(value)
