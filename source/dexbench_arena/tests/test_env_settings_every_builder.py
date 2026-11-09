# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Every way of building an environment gets DexBench's settings and Newton, and PhysX is refused (Kit tier)."""

from __future__ import annotations

import dataclasses

import pytest

pytestmark = pytest.mark.usefixtures("simulation_app")

ENVIRONMENT = "dexbench_hotel_fb_e4_environment"
SETTINGS = {"task": "stack_plates", "embodiment": "dexbench_rby1_wuji2_joint", "arm_mode": "dual_arm"}


def _settings(env_cfg) -> dict:
    physics = env_cfg.sim.physics
    solver = getattr(physics, "solver_cfg", None)
    return {
        "dt": env_cfg.sim.dt,
        "decimation": env_cfg.decimation,
        "render_interval": env_cfg.sim.render_interval,
        "replicate_physics": env_cfg.scene.replicate_physics,
        "physics": type(physics).__name__,
        "substeps": physics.num_substeps,
        "nconmax": getattr(solver, "nconmax", None),
        "processed_actions_recorder": type(env_cfg.demo_recorder_config.record_post_step_processed_actions).__name__,
    }


def _physics_choices(env_cfg) -> tuple[str, dict[str, tuple[dict, str]]]:
    """Physics manager, and per spawned entity its variant selection and spawner."""
    spawns = {}
    for name, entity in vars(env_cfg.scene).items():
        spawn = getattr(entity, "spawn", None)
        if spawn is not None:
            spawns[name] = (dict(getattr(spawn, "variants", None) or {}), spawn.func.__name__)
    return type(env_cfg.sim.physics).__name__, spawns


def _compose(builder):
    composed = builder.compose_manager_cfg()
    return composed[0] if isinstance(composed, tuple) else composed


@pytest.fixture
def restore_builder(monkeypatch):
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    monkeypatch.setattr(ArenaEnvBuilder, "compose_manager_cfg", ArenaEnvBuilder.compose_manager_cfg)
    monkeypatch.setattr(ArenaEnvBuilder, "make_registered", ArenaEnvBuilder.make_registered)


def _argv(presets):
    argv = (["--presets", presets] if presets else []) + [ENVIRONMENT]
    return argv + [token for key, value in SETTINGS.items() for token in (f"--{key}", value)]


@pytest.mark.parametrize("presets", [None, "newton"], ids=["default", "presets-newton"])
def test_plain_and_typed_builders_compose_what_the_cli_builder_does(presets):
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    from dexbench_arena.cli import get_arena_builder_from_cli, get_dexbench_arena_cli_parser
    from dexbench_arena.environments.manipulation.e4_hotel_fb_environment import (
        DexBenchHotelFBE4Environment,
        DexBenchHotelFBE4EnvironmentCfg,
    )

    def parse():
        argv = (["--presets", presets] if presets else []) + [ENVIRONMENT]
        argv += [token for key, value in SETTINGS.items() for token in (f"--{key}", value)]
        return get_dexbench_arena_cli_parser().parse_args(argv)

    cli = get_arena_builder_from_cli(parse()).compose_manager_cfg()[0]
    args = parse()
    plain = ArenaEnvBuilder(DexBenchHotelFBE4Environment().get_env(args), arena_env_builder_cfg_from_argparse(args))
    direct = plain.compose_manager_cfg()[0]
    typed_env = DexBenchHotelFBE4Environment().build(DexBenchHotelFBE4EnvironmentCfg(**SETTINGS))
    typed = ArenaEnvBuilder(typed_env, arena_env_builder_cfg_from_argparse(parse())).compose_manager_cfg()[0]

    assert _settings(direct) == _settings(cli)
    assert _settings(typed) == _settings(cli)
    assert direct.sim.dt * direct.decimation == pytest.approx(1 / 30)
    assert (_settings(direct)["substeps"], _settings(direct)["nconmax"]) == (8, 16384)
    assert _settings(direct)["processed_actions_recorder"] == "Wuji2JointTargetsRecorderCfg"
    assert _settings(direct)["physics"] == "NewtonCfg"


@pytest.mark.parametrize("presets", [None, "newton"], ids=["default", "presets-newton"])
def test_typed_build_matches_the_cli_build_on_newton(restore_builder, presets):
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg

    from dexbench_arena.cli import get_arena_builder_from_cli, get_dexbench_arena_cli_parser
    from dexbench_arena.environments.manipulation.e4_hotel_fb_environment import (
        DexBenchHotelFBE4Environment,
        DexBenchHotelFBE4EnvironmentCfg,
    )

    cli = _physics_choices(_compose(get_arena_builder_from_cli(get_dexbench_arena_cli_parser().parse_args(_argv(presets)))))

    environment = DexBenchHotelFBE4Environment().build(DexBenchHotelFBE4EnvironmentCfg(**SETTINGS))
    typed = _physics_choices(_compose(ArenaEnvBuilder(environment, ArenaEnvBuilderCfg(presets=presets))))

    assert typed == cli
    assert typed[0] == "NewtonCfg"
    assert any(variants.get("Newton") == "Enabled" for variants, _ in typed[1].values())
    assert not any("PhysX" in variants for variants, _ in typed[1].values())
    assert any(func == "_spawn_then_compensate_gravity" for _, func in typed[1].values())


def test_physx_is_refused_on_the_cli_and_the_typed_path(restore_builder):
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg

    from dexbench_arena.cli import get_arena_builder_from_cli, get_dexbench_arena_cli_parser
    from dexbench_arena.environments.manipulation.e4_hotel_fb_environment import (
        DexBenchHotelFBE4Environment,
        DexBenchHotelFBE4EnvironmentCfg,
    )

    with pytest.raises(ValueError, match="runs on Newton only, and this build resolved physx"):
        get_arena_builder_from_cli(get_dexbench_arena_cli_parser().parse_args(_argv("physx")))
    environment = DexBenchHotelFBE4Environment().build(DexBenchHotelFBE4EnvironmentCfg(**SETTINGS))
    with pytest.raises(ValueError, match="runs on Newton only, and this build resolved physx"):
        _compose(ArenaEnvBuilder(environment, ArenaEnvBuilderCfg(presets="physx")))


def test_run_episode_length_reaches_the_env_cfg_through_the_task():
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    from dexbench_arena.cli import get_arena_builder_from_cli, get_dexbench_arena_cli_parser
    from dexbench_arena.environments.manipulation.e4_hotel_fb_environment import (
        DexBenchHotelFBE4Environment,
        DexBenchHotelFBE4EnvironmentCfg,
    )

    argv = [ENVIRONMENT] + [token for key, value in SETTINGS.items() for token in (f"--{key}", value)]
    builder_cfg = arena_env_builder_cfg_from_argparse(get_dexbench_arena_cli_parser().parse_args(argv))

    def typed_episode_length(**overrides) -> float:
        env = DexBenchHotelFBE4Environment().build(DexBenchHotelFBE4EnvironmentCfg(**SETTINGS, **overrides))
        return ArenaEnvBuilder(env, builder_cfg).compose_manager_cfg()[0].episode_length_s

    assert typed_episode_length() == 60.0
    assert typed_episode_length(episode_length_s=10.0) == 10.0
    cli_args = get_dexbench_arena_cli_parser().parse_args(argv + ["--episode_length_s", "12.5"])
    assert get_arena_builder_from_cli(cli_args).compose_manager_cfg()[0].episode_length_s == 12.5
    with pytest.raises(ValueError, match="episode_length_s must be a positive"):
        typed_episode_length(episode_length_s=0.0)


def test_every_episode_row_carries_the_environment_cfg_and_the_task_object():
    import json

    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    from dexbench_arena.cli import get_dexbench_arena_cli_parser
    from dexbench_arena.environments.manipulation.e4_hotel_fb_environment import (
        DexBenchHotelFBE4Environment,
        DexBenchHotelFBE4EnvironmentCfg,
    )

    argv = [ENVIRONMENT] + [token for key, value in SETTINGS.items() for token in (f"--{key}", value)]
    builder_cfg = arena_env_builder_cfg_from_argparse(get_dexbench_arena_cli_parser().parse_args(argv))
    cfg = DexBenchHotelFBE4EnvironmentCfg(**SETTINGS, base_offset_x=0.02, mimic_boundaries="/data/b.json")
    env_cfg = ArenaEnvBuilder(DexBenchHotelFBE4Environment().build(cfg), builder_cfg).compose_manager_cfg()[0]

    term = env_cfg.episode_recorders.environment_cfg
    fields = term.func(None, 0, **term.params)
    assert list(fields) == ["environment_cfg"]
    recorded = json.loads(json.dumps(fields))["environment_cfg"]
    assert recorded == {field: json.loads(json.dumps(getattr(cfg, field), default=str)) for field in recorded}
    assert recorded["task"] == "stack_plates" and recorded["base_offset_x"] == 0.02
    assert set(recorded) == {field.name for field in dataclasses.fields(cfg)}
    assert env_cfg.episode_recorders.exported_demo.func(None, 0) == {}

    term = env_cfg.episode_recorders.task_object
    task_object = term.func(None, 0, **term.params)["task_object"]
    assert task_object["axis"] == "plate_variant" and task_object["key"] == "harmynta_deep_plate_22"
    assert task_object["split"] == "train" and task_object["asset_version"]
    assert task_object["fit"]["scale"] == pytest.approx(1.0) and task_object["fit"]["nest_pitch_m"] > 0
    event = env_cfg.events.dexbench_task_object
    recorder = type("Recorder", (), {"get_ep_meta": lambda self: {"sim_args": {}}})()
    event.func(type("Env", (), {"recorder_manager": recorder})(), None, **event.params)
    assert recorder.get_ep_meta() == {"sim_args": {}, "task_object": task_object}


@pytest.mark.parametrize(
    "environment, settings",
    [
        ("dexbench_retail_v1_environment", {"task": "retrieve_beverage_from_clutter", "anchor": "Fridge"}),
        ("dexbench_logistics_center_e2_environment", {"task": "lift_to_conveyor", "cage_box_alt_scales": (0.85, 1.15)}),
        ("dexbench_hotel_fb_e4_environment", {"task": "stack_plates", "stack_jitter_xy": 0.015}),
    ],
    ids=["r2", "l2", "h1"],
)
def test_every_reset_event_of_a_task_replays_exactly(environment, settings):
    """A replay restores the recorded variation draws only, so every other reset event must draw nothing."""
    from isaaclab_arena.assets.registries import EnvironmentRegistry
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg

    from dexbench_arena.evaluation.replay_guards import undeclared_reset_events

    factory = EnvironmentRegistry().get_component_by_name(environment)
    cfg_type = EnvironmentRegistry().get_environment_cfg_type(factory)
    cfg = cfg_type(embodiment="dexbench_rby1_wuji2_joint", arm_mode="dual_arm", base_jitter_radius=0.02, **settings)
    env_cfg = ArenaEnvBuilder(factory().build(cfg), ArenaEnvBuilderCfg()).compose_manager_cfg()[0]
    assert undeclared_reset_events(env_cfg.events) == []

