# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""The hand contact material through the CLI and typed factories, on the spawned stage, and in the recorded metadata;
and the variation draws on every recording path."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from dexbench_arena.contact_materials import HandContactMaterialCfg, WUJI_GRASP_CONTACT_MATERIAL
from dexbench_arena.environments.hand_contact import configure_hand_contact_material


@pytest.fixture(autouse=True)
def clean_legacy_friction_environment(monkeypatch):
    monkeypatch.delenv("DEXBENCH_HAND_STATIC_FRICTION", raising=False)
    monkeypatch.delenv("DEXBENCH_HAND_DYNAMIC_FRICTION", raising=False)


@pytest.fixture
def e1():
    from dexbench_arena.environments.manipulation.e1_retail_environment import DexBenchRetailStoreE1Environment

    parser = argparse.ArgumentParser()
    DexBenchRetailStoreE1Environment.add_cli_args(parser)
    return DexBenchRetailStoreE1Environment(), parser


@pytest.mark.parametrize("task", ["place_box_in_slot", "retrieve_beverage_from_clutter"])
def test_factory_assigns_material_to_the_grasping_tasks(e1, task):
    factory, parser = e1
    args = parser.parse_args(["--task", task, "--embodiment", "dexbench_rby1_wuji2_joint"])
    assert factory._build_task(args).hand_contact_material == WUJI_GRASP_CONTACT_MATERIAL
    args.embodiment = "dexbench_some_other_hand"
    assert factory._build_task(args).hand_contact_material == WUJI_GRASP_CONTACT_MATERIAL


def test_cli_and_typed_builds_resolve_identically_without_cross_task_leakage(e1):
    from dexbench_arena.environments.manipulation.e1_retail_environment import DexBenchRetailStoreE1EnvironmentCfg
    from dexbench_arena.embodiments.rby1_wuji2.rby1_wuji2 import DexBenchRby1Wuji2JointEmbodiment

    factory, parser = e1
    name = "dexbench_rby1_wuji2_joint"
    args = parser.parse_args(
        [
            "--task",
            "retrieve_beverage_from_clutter",
            "--embodiment",
            name,
            "--hand_static_friction",
            "0.5",
            "--hand_dynamic_friction",
            "0.4",
            "--hand_friction_combine_mode",
            "average",
            "--disable_hand_contact_material",
        ]
    )
    typed = DexBenchRetailStoreE1EnvironmentCfg(
        task="retrieve_beverage_from_clutter",
        embodiment=name,
        hand_static_friction=0.5,
        hand_dynamic_friction=0.4,
        hand_friction_combine_mode="average",
        hand_contact_material_enabled=False,
    )
    first, second, baseline = (DexBenchRby1Wuji2JointEmbodiment() for _ in range(3))
    configure_hand_contact_material(args, factory._build_task(args), first)
    configure_hand_contact_material(typed, factory._build_task(typed), second)
    assert first.event_config.hand_contact_material_check.params == second.event_config.hand_contact_material_check.params
    assert first.hand_contact_material == HandContactMaterialCfg(0.5, 0.4, "average", False)
    assert baseline.event_config.hand_contact_material_check.params["static_friction"] == 6.0
    default_args = parser.parse_args(["--task", "retrieve_beverage_from_clutter", "--embodiment", name])
    task = factory._build_task(default_args)
    task.hand_contact_material = HandContactMaterialCfg(0.7, 0.5)
    configure_hand_contact_material(default_args, task, baseline)
    assert baseline.event_config.hand_contact_material_check.params["static_friction"] == 0.7
    assert baseline.event_config.hand_contact_material_check.params["source"] == "task"
    configure_hand_contact_material(default_args, SimpleNamespace(), first)
    assert first.hand_contact_material == WUJI_GRASP_CONTACT_MATERIAL


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("typed", [False, True])
def test_the_hand_material_is_spawn_data_and_leaves_replication_to_the_preset(e1, monkeypatch, typed, enabled, simulation_app):
    """No prestartup event: Isaac Lab refuses one under physics replication, and a USD
    write after the scene clones would not reach a Newton model built at clone time."""
    from dexbench_arena.cli import get_arena_builder_from_cli
    from dexbench_arena.environments.manipulation.e1_retail_environment import DexBenchRetailStoreE1EnvironmentCfg
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    factory, parser = e1
    args = parser.parse_args(["--embodiment", "dexbench_rby1_wuji2_joint", "--task", "retrieve_beverage_from_clutter"])
    args.hand_contact_material_enabled = enabled
    args.example_environment = factory.name
    args.presets = "newton"
    args.device = "cuda:0"
    monkeypatch.setattr(ArenaEnvBuilder, "compose_manager_cfg", ArenaEnvBuilder.compose_manager_cfg)
    if typed:
        arena_env = factory.build(
            DexBenchRetailStoreE1EnvironmentCfg(
                embodiment=args.embodiment,
                task=args.task,
                hand_contact_material_enabled=enabled,
            )
        )
        builder = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args))
    else:
        builder = get_arena_builder_from_cli(args)
    cfg, _ = builder.compose_manager_cfg()
    assert "newton" in type(cfg.sim.physics).__module__
    assert cfg.events.hand_contact_material_check.mode == "startup"
    assert not [name for name, term in vars(cfg.events).items() if getattr(term, "mode", None) == "prestartup"]
    spawn = cfg.scene.robot.spawn
    assert spawn.hand_static_friction == (WUJI_GRASP_CONTACT_MATERIAL.static_friction if enabled else 1.0)




@pytest.fixture
def isolate_factory_builds(monkeypatch, simulation_app):
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    monkeypatch.setattr(ArenaEnvBuilder, "compose_manager_cfg", ArenaEnvBuilder.compose_manager_cfg)


@pytest.fixture
def factory_case():
    """The E1 factory, its typed cfg and the task every case builds."""
    from dexbench_arena.embodiments.rby1_wuji2 import rby1_wuji2 as _rby1_wuji2  # noqa: F401
    from dexbench_arena.environments.manipulation.e1_retail_environment import (
        DexBenchRetailStoreE1Environment,
        DexBenchRetailStoreE1EnvironmentCfg,
    )

    return DexBenchRetailStoreE1Environment(), DexBenchRetailStoreE1EnvironmentCfg, {"task": "retrieve_beverage_from_clutter"}


def assert_material(environment, expected, source):
    params = environment.embodiment.event_config.hand_contact_material_check.params
    assert {key: params[key] for key in asdict(expected)} == asdict(expected)
    assert params["source"] == source
    assert bool(params["overrides"]) == (source == "override")
    assert environment.embodiment.hand_contact_material == expected
    spawn = environment.embodiment.scene_config.robot.spawn
    applied = expected if expected.enabled else environment.embodiment.model_hand_contact_material
    assert (spawn.hand_static_friction, spawn.hand_dynamic_friction, spawn.hand_friction_combine_mode) == (
        applied.static_friction, applied.dynamic_friction, applied.friction_combine_mode
    )
    term = environment.episode_recorder_terms["hand_contact_material"]
    metadata = {**asdict(expected), "source": source}
    runtime = SimpleNamespace(cfg=SimpleNamespace(dexbench_hand_contact_material=metadata))
    assert term.func(runtime, 0) == {"hand_contact_material": metadata}


@pytest.mark.usefixtures("isolate_factory_builds")
@pytest.mark.parametrize(
    "material",
    [
        HandContactMaterialCfg(10.0, 1.0),
        HandContactMaterialCfg(0.5, 0.4, "average", False),
        HandContactMaterialCfg(0.0, 0.0),
    ],
    ids=["override", "disabled", "zero"],
)
def test_typed_build_applies_overrides_without_leaking_to_next_build(factory_case, material):
    factory, cfg_type, defaults = factory_case
    embodiment = "dexbench_rby1_wuji2_joint"
    environment = factory.build(
        cfg_type(
            **defaults,
            embodiment=embodiment,
            hand_static_friction=material.static_friction,
            hand_dynamic_friction=material.dynamic_friction,
            hand_friction_combine_mode=material.friction_combine_mode,
            hand_contact_material_enabled=material.enabled,
        )
    )
    assert_material(environment, material, "override")
    baseline = factory.build(cfg_type(**defaults, embodiment=embodiment))
    source = "task" if getattr(baseline.task, "hand_contact_material", None) is not None else "embodiment"
    assert_material(baseline, WUJI_GRASP_CONTACT_MATERIAL, source)
    assert_material(environment, material, "override")


@pytest.mark.usefixtures("isolate_factory_builds")
def test_cli_and_typed_builds_apply_partial_overrides_identically(factory_case):
    factory, cfg_type, defaults = factory_case
    embodiment = "dexbench_rby1_wuji2_joint"
    parser = argparse.ArgumentParser()
    factory.add_cli_args(parser)
    parser.set_defaults(enable_cameras=False, **defaults)
    args = parser.parse_args(
        ["--embodiment", embodiment, "--hand_dynamic_friction", "1", "--disable_hand_contact_material"]
    )
    environment = factory.get_env(args)
    typed = factory.build(
        cfg_type(**defaults, embodiment=embodiment, hand_dynamic_friction=1.0, hand_contact_material_enabled=False)
    )
    assert environment.embodiment.event_config.hand_contact_material_check.params["overrides"] == {
        "dynamic_friction": 1.0, "enabled": False,
    }
    expected = HandContactMaterialCfg(6.0, 1.0, enabled=False)
    assert_material(environment, expected, "override")
    assert_material(typed, expected, "override")
    assert environment.embodiment.event_config.hand_contact_material_check.params == (
        typed.embodiment.event_config.hand_contact_material_check.params
    )
    args = parser.parse_args(
        ["--embodiment", embodiment, "--hand_dynamic_friction", "1", "--enable_hand_contact_material"]
    )
    assert_material(factory.get_env(args), HandContactMaterialCfg(6.0, 1.0), "override")


UNSUPPORTED_EMBODIMENT = "franka_joint_pos"


@pytest.mark.usefixtures("isolate_factory_builds")
@pytest.mark.parametrize(
    "field,value",
    [
        ("hand_static_friction", 0.5),
        ("hand_dynamic_friction", 0.4),
        ("hand_friction_combine_mode", "max"),
        ("hand_contact_material_enabled", False),
    ],
)
def test_unsupported_embodiment_rejects_each_explicit_setting(factory_case, field, value):
    factory, cfg_type, defaults = factory_case
    cfg = cfg_type(**defaults, embodiment=UNSUPPORTED_EMBODIMENT, **{field: value})
    with pytest.raises(ValueError, match="supported embodiment"):
        factory.build(cfg)
    with pytest.raises(ValueError, match="supported embodiment"):
        factory.get_env(argparse.Namespace(**asdict(cfg)))


@pytest.mark.usefixtures("isolate_factory_builds")
@pytest.mark.parametrize("name", ["DEXBENCH_HAND_STATIC_FRICTION", "DEXBENCH_HAND_DYNAMIC_FRICTION"])
def test_all_entry_points_reject_retired_environment_variables(factory_case, monkeypatch, name):
    factory, cfg_type, defaults = factory_case
    monkeypatch.setenv(name, "0")
    cfg = cfg_type(**defaults, embodiment="dexbench_rby1_wuji2_joint")
    with pytest.raises(ValueError, match="Unset them"):
        factory.build(cfg)
    with pytest.raises(ValueError, match="Unset them"):
        factory.get_env(argparse.Namespace(**asdict(cfg)))


@pytest.mark.usefixtures("isolate_factory_builds")
def test_unsupported_embodiment_keeps_authored_defaults_when_no_override_is_requested(factory_case):
    factory, cfg_type, defaults = factory_case
    environment = factory.build(cfg_type(**defaults, embodiment=UNSUPPORTED_EMBODIMENT))
    assert "hand_contact_material" not in environment.episode_recorder_terms
    assert getattr(environment.embodiment.event_config, "hand_contact_material_check", None) is None


@pytest.mark.usefixtures("isolate_factory_builds")
def test_h1_typed_build_proposes_its_task_material_without_turning_it_into_an_override():
    """DB-H1's 1.2/1.0 is the task's proposal. A typed build that sets nothing must
    record it as the task's, the same as the CLI, and a build without the task must
    fall back to the embodiment instead of inheriting 1.2/1.0 as a run override."""
    from dexbench_arena.environments.manipulation.e4_hotel_fb_environment import (
        H1_HAND_DYNAMIC_FRICTION,
        H1_HAND_STATIC_FRICTION,
        DexBenchHotelFBE4Environment,
        DexBenchHotelFBE4EnvironmentCfg,
    )

    factory = DexBenchHotelFBE4Environment()
    embodiment = "dexbench_rby1_wuji2_joint"
    expected = HandContactMaterialCfg(H1_HAND_STATIC_FRICTION, H1_HAND_DYNAMIC_FRICTION, "max")
    typed = factory.build(DexBenchHotelFBE4EnvironmentCfg(embodiment=embodiment, task="stack_plates"))
    assert_material(typed, expected, "task")
    parser = argparse.ArgumentParser()
    factory.add_cli_args(parser)
    parser.set_defaults(enable_cameras=False)
    cli = factory.get_env(parser.parse_args(["--embodiment", embodiment, "--task", "stack_plates"]))
    assert_material(cli, expected, "task")
    scene_only = factory.build(DexBenchHotelFBE4EnvironmentCfg(embodiment=embodiment))
    assert_material(scene_only, WUJI_GRASP_CONTACT_MATERIAL, "embodiment")




@pytest.fixture
def stage_env():
    """Two envs, each with a robot whose finger binds a hand_contact material and whose arm
    binds its own, and a second robot with a hand_contact material of its own."""
    from isaaclab.sim.utils import use_stage
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    stage = Usd.Stage.CreateInMemory()
    for index in range(2):
        for body, material in (("Robot/finger", "hand_contact"), ("Robot/arm", "friction_0"), ("RobotOther/finger", "hand_contact")):
            root = f"/World/envs/env_{index}/{body}/collisions"
            mat = UsdShade.Material.Define(stage, f"{root}/physics_materials/{material}")
            api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
            api.CreateStaticFrictionAttr(6.0 if material == "hand_contact" else 1.0)
            api.CreateDynamicFrictionAttr(5.0 if material == "hand_contact" else 1.0)
            mat.GetPrim().CreateAttribute("physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token).Set("max")
            UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, f"/World/envs/env_{index}/{body}").GetPrim())
            prim = UsdGeom.Cube.Define(stage, f"{root}/geom_0").GetPrim()
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")
    env = SimpleNamespace(
        sim=SimpleNamespace(stage=stage),
        scene=SimpleNamespace(env_prim_paths=["/World/envs/env_0", "/World/envs/env_1"]),
        cfg=SimpleNamespace(
            sim=SimpleNamespace(dt=0.01, render_interval=2, physics=SimpleNamespace()),
            decimation=3,
            scene=SimpleNamespace(num_envs=2),
        ),
    )
    with use_stage(stage):
        yield env


HAND = ("finger",)


def _coefficients(stage, path):
    from pxr import UsdPhysics

    prim = stage.GetPrimAtPath(path)
    api = UsdPhysics.MaterialAPI(prim)
    return (api.GetStaticFrictionAttr().Get(), api.GetDynamicFrictionAttr().Get(),
            prim.GetAttribute("physxMaterial:frictionCombineMode").Get())


@pytest.mark.parametrize("static,dynamic,mode", [(6.0, 5.0, "max"), (0.5, 0.4, "average"), (0.0, 0.0, "min")])
def test_spawned_coefficients_include_zero_and_only_the_spawned_robots_hand(stage_env, static, dynamic, mode):
    from dexbench_arena.embodiments.hand_contact import check_hand_contact_material
    from dexbench_arena.embodiments.rby1_wuji2.spawn import set_hand_contact_material

    stage = stage_env.sim.stage
    for index in range(2):
        assert set_hand_contact_material(stage.GetPrimAtPath(f"/World/envs/env_{index}/Robot"), static, dynamic, mode) == 1
    for index in range(2):
        env = f"/World/envs/env_{index}"
        got_static, got_dynamic, got_mode = _coefficients(stage, f"{env}/Robot/finger/collisions/physics_materials/hand_contact")
        assert (got_static, got_dynamic, got_mode) == (pytest.approx(static), pytest.approx(dynamic), mode)
        assert _coefficients(stage, f"{env}/Robot/arm/collisions/physics_materials/friction_0") == (1.0, 1.0, "max")
        assert _coefficients(stage, f"{env}/RobotOther/finger/collisions/physics_materials/hand_contact") == (6.0, 5.0, "max")
    check_hand_contact_material(stage_env, None, HAND, static_friction=static, dynamic_friction=dynamic, friction_combine_mode=mode)
    metadata = stage_env.cfg.dexbench_hand_contact_material
    assert metadata["bound_collision_count"] == 2
    assert (metadata["static_friction"], metadata["dynamic_friction"]) == (static, dynamic)
    assert not hasattr(stage_env.cfg, "get_ep_meta")


def test_the_check_refuses_coefficients_the_spawn_did_not_write(stage_env):
    from dexbench_arena.embodiments.hand_contact import check_hand_contact_material

    with pytest.raises(RuntimeError, match="do not carry the selected"):
        check_hand_contact_material(stage_env, None, HAND, static_friction=1.2, dynamic_friction=1.0)
    assert not hasattr(stage_env.cfg, "dexbench_hand_contact_material")


def test_a_disabled_material_puts_the_declared_model_material_on_the_hand_and_records_it(stage_env):
    """The opt-out is the MJCF's 1.0 / 1.0, max, as the embodiment declares it: not the grasp
    material, and not whatever the robot's other colliders carry."""
    from pxr import UsdPhysics

    from dexbench_arena.embodiments.hand_contact import HandContactMaterialMixin, check_hand_contact_material
    from dexbench_arena.embodiments.rby1_wuji2.rby1_wuji2 import Rby1Wuji2EmbodimentBase
    from dexbench_arena.embodiments.rby1_wuji2.spawn import set_hand_contact_material

    class Hand(HandContactMaterialMixin):
        model_hand_contact_material = Rby1Wuji2EmbodimentBase.model_hand_contact_material

        def __init__(self):
            self.event_config = SimpleNamespace(hand_contact_material_check=SimpleNamespace(params={}))
            self.scene_config = SimpleNamespace(robot=SimpleNamespace(spawn=SimpleNamespace()))

    hand = Hand()
    hand.configure_hand_contact_material(HandContactMaterialCfg(0.5, 0.4, enabled=False), source="override")
    spawn = hand.scene_config.robot.spawn
    stage = stage_env.sim.stage
    for index in range(2):
        robot = f"/World/envs/env_{index}/Robot"
        arm = UsdPhysics.MaterialAPI(stage.GetPrimAtPath(f"{robot}/arm/collisions/physics_materials/friction_0"))
        arm.GetStaticFrictionAttr().Set(3.0)
        arm.GetDynamicFrictionAttr().Set(3.0)
        set_hand_contact_material(
            stage.GetPrimAtPath(robot), spawn.hand_static_friction, spawn.hand_dynamic_friction,
            spawn.hand_friction_combine_mode,
        )
    check_hand_contact_material(stage_env, None, HAND, static_friction=0.5, dynamic_friction=0.4, enabled=False)
    metadata = stage_env.cfg.dexbench_hand_contact_material
    assert metadata["enabled"] is False
    assert (metadata["static_friction"], metadata["dynamic_friction"], metadata["friction_combine_mode"]) == (1.0, 1.0, "max")


def test_a_disabled_check_refuses_hand_materials_that_disagree(stage_env):
    """One record per run: with two hand materials apart, no value describes what ran."""
    from pxr import UsdPhysics

    from dexbench_arena.embodiments.hand_contact import check_hand_contact_material

    material = stage_env.sim.stage.GetPrimAtPath("/World/envs/env_1/Robot/finger/collisions/physics_materials/hand_contact")
    UsdPhysics.MaterialAPI(material).GetDynamicFrictionAttr().Set(4.0)
    with pytest.raises(RuntimeError, match="different coefficients"):
        check_hand_contact_material(stage_env, None, HAND, static_friction=6.0, dynamic_friction=5.0, enabled=False)
    assert not hasattr(stage_env.cfg, "dexbench_hand_contact_material")


def test_a_hand_collider_without_the_hand_material_fails_the_check(stage_env):
    """A body jointed below a hand body is hand, whatever its name: a build whose material rule
    missed it fails, instead of passing on the colliders that do bind the material."""
    from pxr import UsdGeom, UsdPhysics, UsdShade

    from dexbench_arena.embodiments.hand_contact import check_hand_contact_material

    stage = stage_env.sim.stage
    for index in range(2):
        robot = f"/World/envs/env_{index}/Robot"
        UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, f"{robot}/pad").GetPrim())
        joint = UsdPhysics.FixedJoint.Define(stage, f"{robot}/joints/pad_fixed")
        joint.CreateBody0Rel().SetTargets([f"{robot}/finger"])
        joint.CreateBody1Rel().SetTargets([f"{robot}/pad"])
        pad = UsdGeom.Cube.Define(stage, f"{robot}/pad/collisions/geom_0").GetPrim()
        UsdPhysics.CollisionAPI.Apply(pad)
        arm = UsdShade.Material(stage.GetPrimAtPath(f"{robot}/arm/collisions/physics_materials/friction_0"))
        UsdShade.MaterialBindingAPI.Apply(pad).Bind(arm, UsdShade.Tokens.weakerThanDescendants, "physics")
    with pytest.raises(RuntimeError, match="1 of 2 hand colliders.*pad/collisions/geom_0"):
        check_hand_contact_material(stage_env, None, HAND, static_friction=6.0, dynamic_friction=5.0)
    assert not hasattr(stage_env.cfg, "dexbench_hand_contact_material")


def test_a_hand_binding_without_the_binding_api_fails_the_check(stage_env):
    """The physics parser Newton reads ignores a collider's material relationship unless the
    collider has MaterialBindingAPI, so the check does too."""
    from pxr import UsdShade

    from dexbench_arena.embodiments.hand_contact import check_hand_contact_material

    collider = stage_env.sim.stage.GetPrimAtPath("/World/envs/env_1/Robot/finger/collisions/geom_0")
    collider.RemoveAPI(UsdShade.MaterialBindingAPI)
    assert UsdShade.MaterialBindingAPI(collider).GetDirectBinding("physics").GetMaterial()
    with pytest.raises(RuntimeError, match="1 of 1 hand colliders under /World/envs/env_1/Robot"):
        check_hand_contact_material(stage_env, None, HAND, static_friction=6.0, dynamic_friction=5.0)
    assert not hasattr(stage_env.cfg, "dexbench_hand_contact_material")


def test_an_object_jointed_to_the_hand_is_not_hand(stage_env):
    """A joint from a hand body to a body outside the robot (a grasped object attached to a
    finger) does not make that body's colliders hand colliders."""
    from pxr import UsdGeom, UsdPhysics

    from dexbench_arena.embodiments.hand_contact import check_hand_contact_material

    stage = stage_env.sim.stage
    for index in range(2):
        env = f"/World/envs/env_{index}"
        UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, f"{env}/TaskObject").GetPrim())
        UsdPhysics.CollisionAPI.Apply(UsdGeom.Cube.Define(stage, f"{env}/TaskObject/geom_0").GetPrim())
        joint = UsdPhysics.FixedJoint.Define(stage, f"{env}/Robot/joints/attach_object")
        joint.CreateBody0Rel().SetTargets([f"{env}/Robot/finger"])
        joint.CreateBody1Rel().SetTargets([f"{env}/TaskObject"])
    check_hand_contact_material(stage_env, None, HAND, static_friction=6.0, dynamic_friction=5.0)
    metadata = stage_env.cfg.dexbench_hand_contact_material
    assert metadata["bound_collision_count"] == 2
    assert metadata["material_path"] == "*/collisions/physics_materials/hand_contact"


def test_a_robot_without_the_hand_material_fails_at_spawn_and_at_the_check(stage_env):
    from dexbench_arena.embodiments.hand_contact import check_hand_contact_material
    from dexbench_arena.embodiments.rby1_wuji2.spawn import set_hand_contact_material

    stage = stage_env.sim.stage
    stage.RemovePrim("/World/envs/env_1/Robot/finger")
    with pytest.raises(RuntimeError, match="predates"):
        set_hand_contact_material(stage.GetPrimAtPath("/World/envs/env_1/Robot"), 6.0, 5.0, "max")
    with pytest.raises(RuntimeError, match="env_1/Robot"):
        check_hand_contact_material(stage_env, None, HAND, static_friction=6.0, dynamic_friction=5.0)
    assert not hasattr(stage_env.cfg, "dexbench_hand_contact_material")


@pytest.mark.parametrize("custom", [False, True])
def test_real_recorder_metadata_and_config_serialization(stage_env, custom):
    from isaaclab.managers import RecorderManager
    from isaaclab.utils.dict import class_to_dict
    from isaaclab.utils.string import callable_to_string, string_to_callable
    from dexbench_arena.embodiments.hand_contact import install_hand_contact_recording_metadata
    from dexbench_arena.embodiments.rby1_wuji2.rby1_wuji2 import Rby1Wuji2EventCfg

    env = stage_env
    original_callback = None
    if custom:
        original_callback = lambda: {"custom": {"preserved": True}}
        env.cfg.get_ep_meta = original_callback
    recorder = RecorderManager(None, env)
    recorder._env = env
    env.recorder_manager = recorder
    upstream = recorder.get_ep_meta()
    env.cfg.dexbench_hand_contact_material = {"overrides": {"enabled": False}}
    install_hand_contact_recording_metadata(env)
    assert recorder.get_ep_meta() == {**upstream, "hand_contact_material": env.cfg.dexbench_hand_contact_material}
    assert getattr(env.cfg, "get_ep_meta", None) is original_callback
    if not custom:
        assert "get_ep_meta" not in class_to_dict(env.cfg)
        assert "sim_args" in recorder.get_ep_meta()
    for cfg_type in (Rby1Wuji2EventCfg,):
        cfg = cfg_type()
        event_dict = class_to_dict(cfg)
        func_path = event_dict["hand_contact_recording_metadata"]["func"]
        assert string_to_callable(func_path) is install_hand_contact_recording_metadata
        assert callable_to_string(install_hand_contact_recording_metadata) == func_path


def test_cli_setup_rejects_legacy_environment_before_launcher(monkeypatch):
    from dexbench_arena.environments.hand_contact import add_hand_contact_material_cli_args

    monkeypatch.setenv("DEXBENCH_HAND_STATIC_FRICTION", "")
    with pytest.raises(ValueError, match="Unset them"):
        add_hand_contact_material_cli_args(argparse.ArgumentParser())


@pytest.mark.parametrize("recorders_cfg", [None, {}])
def test_startup_supports_completely_disabled_recording(stage_env, recorders_cfg):
    from isaaclab.managers import RecorderManager
    from dexbench_arena.embodiments.hand_contact import install_hand_contact_recording_metadata

    stage_env.recorder_manager = RecorderManager(recorders_cfg, stage_env)
    install_hand_contact_recording_metadata(stage_env)
    stage_env.recorder_manager.export_episodes()
    assert stage_env.recorder_manager.active_terms == []


@pytest.mark.parametrize("mimic", [False, True])
@pytest.mark.parametrize("typed", [False, True])
def test_every_recording_path_records_the_variation_draws(e1, monkeypatch, typed, mimic, simulation_app):
    """Isaac Lab's record_demos.py loads the registered demo recorder and
    generate_dataset.py takes the env's Mimic recorder, so the variation term must
    reach both on both build paths."""
    from dexbench_arena.cli import get_arena_builder_from_cli
    from dexbench_arena.environments.manipulation.e1_retail_environment import DexBenchRetailStoreE1EnvironmentCfg
    from dexbench_arena.variations.variation_samples_recorder import VariationSamplesRecorderCfg
    from isaaclab_arena.cli.isaaclab_arena_cli import arena_env_builder_cfg_from_argparse
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

    factory, parser = e1
    args = parser.parse_args(["--embodiment", "dexbench_rby1_wuji2_joint", "--task", "retrieve_beverage_from_clutter"])
    args.example_environment = factory.name
    args.mimic = mimic
    monkeypatch.setattr(ArenaEnvBuilder, "compose_manager_cfg", ArenaEnvBuilder.compose_manager_cfg)
    if typed:
        arena_env = factory.build(
            DexBenchRetailStoreE1EnvironmentCfg(embodiment=args.embodiment, task=args.task, mimic=mimic)
        )
        builder = ArenaEnvBuilder(arena_env, arena_env_builder_cfg_from_argparse(args))
    else:
        builder = get_arena_builder_from_cli(args)
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    name, cfg, _ = builder.build_registered()
    recorders = [load_cfg_from_registry(name, "demo_recorder_cfg_entry_point")]
    if mimic:
        recorders.append(cfg.mimic_recorder_config)
    for recorder in recorders:
        assert isinstance(recorder.record_variation_samples, VariationSamplesRecorderCfg)


def test_variation_callback_keeps_chaining_and_puts_one_term_on_a_shared_recorder(simulation_app):
    """A previous callback that edits in place returns None, and Arena gives demo and
    Mimic the same recorder when there is no Mimic cfg: neither may cost the term."""
    from dexbench_arena.environments.manipulation.scene_environment_base import _with_variation_samples
    from dexbench_arena.variations.variation_samples_recorder import VariationSamplesRecorderCfg
    from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg

    shared = ActionStateRecorderManagerCfg()
    env_cfg = SimpleNamespace(demo_recorder_config=shared, mimic_recorder_config=shared)
    seen = []
    assert _with_variation_samples(seen.append)(env_cfg) is env_cfg
    assert seen == [env_cfg]
    terms = [term for term in vars(shared).values() if isinstance(term, VariationSamplesRecorderCfg)]
    assert len(terms) == 1
