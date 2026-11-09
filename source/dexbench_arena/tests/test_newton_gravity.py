# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""On Newton, bodies PhysX excludes from gravity get their spawner wrapped for mjc:gravcomp, once."""

from __future__ import annotations

from types import SimpleNamespace

from dexbench_arena.assets.newton_gravity import (
    _spawn_then_compensate_gravity,
    apply_newton_gravity_compensation,
    newton_gravity_compensation_enabled,
)


def spawn(disable_gravity):
    rigid_props = None if disable_gravity is None else SimpleNamespace(disable_gravity=disable_gravity)
    return SimpleNamespace(func=lambda *a, **k: None, rigid_props=rigid_props)


def asset(object_type, cfg):
    return SimpleNamespace(object_type=SimpleNamespace(name=object_type), object_cfg=SimpleNamespace(spawn=cfg))


def test_only_disable_gravity_bodies_are_wrapped_robot_included():
    floating, falling, visual, robot = spawn(True), spawn(False), spawn(True), spawn(True)
    original = floating.func
    scene = SimpleNamespace(assets={
        "task_frame": asset("RIGID", floating), "can": asset("RIGID", falling),
        "shelf": asset("BASE", visual), "plain": asset("RIGID", spawn(None)),
    })
    embodiment = SimpleNamespace(name="dexbench_rby1_wuji2", scene_config=SimpleNamespace(robot=SimpleNamespace(spawn=robot)))
    assert apply_newton_gravity_compensation(scene, embodiment) == ["task_frame", "dexbench_rby1_wuji2"]
    assert floating.func is _spawn_then_compensate_gravity and floating.__dict__["_dexbench_spawn_func_before_gravcomp"] is original
    assert falling.func is not _spawn_then_compensate_gravity and visual.func is not _spawn_then_compensate_gravity
    assert apply_newton_gravity_compensation(scene, embodiment) == []


def test_the_run_flag_switches_it_off():
    assert newton_gravity_compensation_enabled(SimpleNamespace())
    assert not newton_gravity_compensation_enabled(SimpleNamespace(newton_gravity_compensation=False))


def test_the_spawner_reaches_every_link_body_and_a_bare_prim(simulation_app):
    """An articulation's link bodies under the spawned prim get mjc:gravcomp, and so does a
    spawn with no rigid body in it (on the prim itself), whatever writer semantics the pin has."""
    import isaaclab.sim as sim_utils
    from pxr import UsdGeom, UsdPhysics

    stage = sim_utils.create_new_stage()
    UsdGeom.Xform.Define(stage, "/World/envs/env_0/Robot")
    for link in ("base", "torso_link", "arm_link"):
        UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, f"/World/envs/env_0/Robot/{link}").GetPrim())
    UsdGeom.Xform.Define(stage, "/World/envs/env_0/Frame")
    for path in ("/World/envs/env_0/Robot", "/World/envs/env_0/Frame"):
        cfg = SimpleNamespace(func=None)
        cfg.__dict__["_dexbench_spawn_func_before_gravcomp"] = lambda prim_path, *a, **k: stage.GetPrimAtPath(prim_path)
        _spawn_then_compensate_gravity(path, cfg)

    authored = sorted(
        str(prim.GetPath()) for prim in stage.Traverse()
        if prim.GetAttribute("mjc:gravcomp") and prim.GetAttribute("mjc:gravcomp").Get() == 1.0
    )
    assert authored == [
        "/World/envs/env_0/Frame",
        "/World/envs/env_0/Robot/arm_link",
        "/World/envs/env_0/Robot/base",
        "/World/envs/env_0/Robot/torso_link",
    ]
    assert not stage.GetPrimAtPath("/World/envs/env_0/Robot").HasAPI(UsdPhysics.RigidBodyAPI)


def test_a_writer_that_skips_a_body_does_not_make_the_root_a_body(simulation_app, monkeypatch):
    """The writer also reports False when it skipped an instanced body or a fragment failed;
    the spawn root of a subtree that has bodies must not become a body nested above them."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim import schemas
    from pxr import UsdGeom, UsdPhysics

    stage = sim_utils.create_new_stage()
    UsdGeom.Xform.Define(stage, "/World/envs/env_0/Robot")
    UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, "/World/envs/env_0/Robot/base").GetPrim())
    writer = schemas.apply_rigid_body_properties
    monkeypatch.setattr(schemas, "apply_rigid_body_properties", lambda *a, **k: writer(*a, **k) and False)
    cfg = SimpleNamespace(func=None)
    cfg.__dict__["_dexbench_spawn_func_before_gravcomp"] = lambda prim_path, *a, **k: stage.GetPrimAtPath(prim_path)
    _spawn_then_compensate_gravity("/World/envs/env_0/Robot", cfg)

    assert not stage.GetPrimAtPath("/World/envs/env_0/Robot").HasAPI(UsdPhysics.RigidBodyAPI)
    assert stage.GetPrimAtPath("/World/envs/env_0/Robot/base").GetAttribute("mjc:gravcomp").Get() == 1.0
