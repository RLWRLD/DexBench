# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Each environment's typed config is its command line: the same names, the same defaults."""

from __future__ import annotations

import argparse
import dataclasses

import pytest

pytest.importorskip("isaaclab_arena.assets.register", reason="needs IsaacLab-Arena importable")

from isaaclab_arena.assets.registries import EnvironmentRegistry  # noqa: E402
from isaaclab_arena.cli.dataclass_cli import dataclass_from_cli  # noqa: E402

import dexbench_arena.environments  # noqa: E402, F401  -- registers the DexBench environments

SHARED_PARSER_FIELDS = {"enable_cameras", "mimic", "num_envs", "record_dataset", "record_dataset_images"}
LAUNCHER_FIELDS = {"source_demo_origin"}

FACTORIES = [
    EnvironmentRegistry().get_component_by_name(name)
    for name in sorted(EnvironmentRegistry().get_all_keys())
    if name.startswith("dexbench_")
]


@pytest.mark.parametrize("factory", FACTORIES, ids=lambda factory: factory.name)
def test_config_fields_are_the_environment_flags_with_their_defaults(factory):
    parser = argparse.ArgumentParser()
    factory.add_cli_args(parser)
    defaults = parser.parse_args([])
    cfg_type = factory._legacy_argparse_cfg_type
    fields = {field.name for field in dataclasses.fields(cfg_type)}

    assert EnvironmentRegistry().get_environment_cfg_type(factory) is cfg_type
    assert sorted(set(vars(defaults)) - fields) == [], "flags the config does not carry"
    assert sorted(fields - set(vars(defaults)) - SHARED_PARSER_FIELDS - LAUNCHER_FIELDS) == [], "config fields without a flag"
    assert dataclass_from_cli(cfg_type, defaults) == cfg_type()


RETAIL_IDS = ("dexbench_retail_v1_environment", "dexbench_retail_store_e1_environment")


def test_the_retail_store_resolves_under_its_new_and_its_old_id():
    """dexbench_retail_v1_environment, and its old id as an alias with the same flags."""
    new, old = (EnvironmentRegistry().get_component_by_name(name) for name in RETAIL_IDS)
    assert issubclass(old, new) and old is not new
    argv = ["--task", "retrieve_beverage_from_clutter", "--anchor", "Fridge", "--with_full_items"]
    parsed = []
    for factory in (new, old):
        parser = argparse.ArgumentParser()
        factory.add_cli_args(parser)
        parsed.append(dataclasses.asdict(dataclass_from_cli(factory._legacy_argparse_cfg_type, parser.parse_args(argv))))
    assert parsed[0] == parsed[1]


@pytest.mark.kit
def test_both_retail_ids_compose_the_same_env_cfg(simulation_app):
    from dexbench_arena.cli import get_arena_builder_from_cli, get_dexbench_arena_cli_parser

    def compose(name):
        argv = ["--presets", "newton", name, "--task", "retrieve_beverage_from_clutter", "--anchor", "Fridge",
                "--embodiment", "dexbench_rby1_wuji2_joint", "--with_full_items"]
        env_cfg = get_arena_builder_from_cli(get_dexbench_arena_cli_parser().parse_args(argv)).compose_manager_cfg()[0]
        return {
            "scene": sorted(vars(env_cfg.scene)),
            "dt": env_cfg.sim.dt,
            "decimation": env_cfg.decimation,
            "episode_length_s": env_cfg.episode_length_s,
            "physics": type(env_cfg.sim.physics).__name__,
            "terminations": sorted(vars(env_cfg.terminations)),
            "observations": sorted(vars(env_cfg.observations.policy)),
        }

    new, old = (compose(name) for name in RETAIL_IDS)
    assert new == old and new["scene"]


@pytest.mark.parametrize("argv,enabled", [
    ([], None), (["--wrist-cameras"], True), (["--no-wrist-cameras"], False),
])
def test_wrist_cli_reaches_typed_environment_cfg(argv, enabled):
    factory = EnvironmentRegistry().get_component_by_name("dexbench_hotel_fb_e4_environment")
    parser = argparse.ArgumentParser()
    factory.add_cli_args(parser)
    cfg = dataclass_from_cli(factory._legacy_argparse_cfg_type, parser.parse_args(argv))
    assert cfg.wrist_cameras is enabled
