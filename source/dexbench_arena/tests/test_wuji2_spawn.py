# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""The Wuji2 spawner and the USD the embodiments load."""
import xml.etree.ElementTree as ET
from dataclasses import astuple

import pytest


def test_soft_contact_is_limited_to_all_ten_sensor_pads():
    from pxr import Usd, UsdGeom, UsdPhysics
    from dexbench_arena.embodiments.rby1_wuji2.constants import RBY1_WUJI2_FINGERTIP_PAD_BODY_NAMES
    from dexbench_arena.embodiments.rby1_wuji2.spawn import set_fingertip_contact_dim

    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/Robot").GetPrim()
    paths = []
    for name in (*RBY1_WUJI2_FINGERTIP_PAD_BODY_NAMES, "right_index_finger_distal"):
        path = f"/Robot/{name}/collisions/hull"
        prim = UsdGeom.Cube.Define(stage, path).GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        paths.append(path)
    assert set(set_fingertip_contact_dim(root, 4)) == set(paths[:-1])
    for path in paths[:-1]:
        assert stage.GetPrimAtPath(path).GetAttribute("mjc:condim").Get() == 4
    assert not stage.GetPrimAtPath(paths[-1]).GetAttribute("mjc:condim")
    with pytest.raises(ValueError, match="Invalid fingertip"):
        set_fingertip_contact_dim(root, 2)
    stage.RemovePrim(paths[0])
    with pytest.raises(ValueError, match="No sensor-pad collider"):
        set_fingertip_contact_dim(root, 4)


@pytest.fixture
def asset(tmp_path):
    """A SimReady-shaped stand-in: the Isaac robot schema on the default prim, the
    articulation API on the world-fixed joint, visual and collision groups per body."""
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics
    path = tmp_path / 'revised.usda'
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, '/Robot').GetPrim()
    root.CreateAttribute('isaac:robotType', Sdf.ValueTypeNames.Token).Set('Manipulator')
    stage.SetDefaultPrim(root)
    for name in ('base', 'link_left_arm_0', 'wheel_fl_link'):
        body = UsdGeom.Xform.Define(stage, f'/Robot/{name}').GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body)
        for group in ('visuals', 'collisions'):
            UsdGeom.Xform.Define(stage, f'/Robot/{name}/{group}')
            shape = UsdGeom.Cube.Define(stage, f'/Robot/{name}/{group}/shape').GetPrim()
            if group == 'collisions':
                UsdPhysics.CollisionAPI.Apply(shape)
    joint = UsdPhysics.FixedJoint.Define(stage, '/Robot/joints/base_fixed').GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(joint)
    stage.GetRootLayer().Save()
    return path, stage


def test_normalization_moves_the_root_and_switches_off_only_the_wheel_colliders(asset):
    from pxr import Usd, UsdPhysics
    from dexbench_arena.embodiments.rby1_wuji2.spawn import _normalize_wuji2_layer
    path, source = asset
    stage = Usd.Stage.Open(_normalize_wuji2_layer(str(path)))
    assert stage.GetPrimAtPath('/Robot/base/collisions').IsActive()
    assert stage.GetPrimAtPath('/Robot/link_left_arm_0/collisions').IsActive()
    assert not stage.GetPrimAtPath('/Robot/wheel_fl_link/collisions').IsActive()
    assert stage.GetDefaultPrim().HasAPI(UsdPhysics.ArticulationRootAPI)
    assert not stage.GetPrimAtPath('/Robot/joints/base_fixed').HasAPI(UsdPhysics.ArticulationRootAPI)
    assert source.GetPrimAtPath('/Robot/wheel_fl_link/collisions').IsActive()


def test_a_body_without_collision_shapes_fails(asset):
    from dexbench_arena.embodiments.rby1_wuji2.spawn import _normalize_wuji2_layer
    file, stage = asset
    stage.RemovePrim('/Robot/link_left_arm_0/collisions/shape')
    stage.GetRootLayer().Save()
    with pytest.raises(RuntimeError, match='no collision shapes'):
        _normalize_wuji2_layer(str(file))


def test_an_export_without_the_robot_schema_is_refused(asset):
    """The DEFT export the SimReady build was made from (no Isaac robot schema) is not loaded."""
    from dexbench_arena.embodiments.rby1_wuji2.spawn import _normalize_wuji2_layer
    file, stage = asset
    stage.GetDefaultPrim().RemoveProperty('isaac:robotType')
    stage.GetRootLayer().Save()
    with pytest.raises(RuntimeError, match='Not the SimReady'):
        _normalize_wuji2_layer(str(file))


def test_the_usd_moves_exactly_the_joints_the_embodiment_names():
    """66 movable joints: the 62 action joints and the 4 wheels, so no mecanum roller."""
    from pxr import Usd, UsdPhysics
    from dexbench_arena.embodiments.rby1_wuji2.constants import (
        RBY1_WUJI2_JOINT_ACTION_NAMES, RBY1_WUJI2_USD_PATH, RBY1_WUJI2_WHEEL_JOINT_NAMES)
    if not RBY1_WUJI2_USD_PATH.exists():
        pytest.skip(f"Wuji v2 USD is unavailable: {RBY1_WUJI2_USD_PATH}")
    stage = Usd.Stage.Open(str(RBY1_WUJI2_USD_PATH))
    joints = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint) and not p.IsA(UsdPhysics.FixedJoint)]
    assert len(joints) == 66
    assert sorted(p.GetName() for p in joints) == sorted(
        (*RBY1_WUJI2_JOINT_ACTION_NAMES, *RBY1_WUJI2_WHEEL_JOINT_NAMES))
    lower = {p.GetName(): p.GetAttribute('physics:lowerLimit').Get() for p in joints}
    assert not {name: value for name, value in lower.items() if value > 0.0}


def test_spawner_defers_articulation_props_until_after_the_reference_swap():
    """One articulation root: the props must not be applied while the source is composed."""

    pytest.importorskip("isaaclab.sim", reason="needs the Isaac Sim runtime")
    from dexbench_arena.embodiments.rby1_wuji2 import spawn as wuji2_spawn

    seen = {}

    class _References:
        def ClearReferences(self):
            seen["cleared"] = True

        def AddReference(self, identifier):
            seen["referenced"] = identifier

    class _Prim:
        def GetReferences(self):
            return _References()

    class _ArticulationProps:
        fix_root_link = None

    class _Cfg:
        usd_path = "/tmp/does-not-matter.usd"
        articulation_props = _ArticulationProps()
        hand_static_friction, hand_dynamic_friction, hand_friction_combine_mode = 1.0, 1.0, "max"

        def replace(self, **kwargs):
            clone = _Cfg()
            clone.articulation_props = kwargs.get("articulation_props", self.articulation_props)
            return clone

    cfg = _Cfg()
    layer = type("_Layer", (), {"identifier": "anon:normalized"})()
    order = []

    def _fake_normalize(usd_path):
        return layer

    def _fake_spawn(prim_path, spawn_cfg, translation, orientation, **kwargs):
        order.append("spawn")
        seen["spawn_articulation_props"] = spawn_cfg.articulation_props
        return _Prim()

    def _fake_modify(prim_path, props):
        order.append("modify")
        seen["modify_props"] = props

    def _fake_set_material(prim, static, dynamic, mode):
        order.append("material")
        seen["material"] = (static, dynamic, mode)

    original = (wuji2_spawn._normalize_wuji2_layer, wuji2_spawn.spawn_from_usd,
                wuji2_spawn.schemas.modify_articulation_root_properties, wuji2_spawn.set_hand_contact_material)
    wuji2_spawn._normalize_wuji2_layer = _fake_normalize
    wuji2_spawn.spawn_from_usd = _fake_spawn
    wuji2_spawn.schemas.modify_articulation_root_properties = _fake_modify
    wuji2_spawn.set_hand_contact_material = _fake_set_material
    try:
        wuji2_spawn.spawn_rby1_wuji2_usd.__wrapped__("/World/envs/env_0/Robot", cfg)
    finally:
        (wuji2_spawn._normalize_wuji2_layer, wuji2_spawn.spawn_from_usd,
         wuji2_spawn.schemas.modify_articulation_root_properties, wuji2_spawn.set_hand_contact_material) = original

    assert seen["spawn_articulation_props"] is None, (
        "articulation props were handed to the file spawner, which applies them nested onto the "
        "source asset's articulation root and leaves a second root behind"
    )
    assert seen["cleared"] and seen["referenced"] == "anon:normalized"
    assert seen["modify_props"] is cfg.articulation_props
    assert order == ["spawn", "modify", "material"]
    assert seen["material"] == (1.0, 1.0, "max")


def test_simready_build_normalizes_to_the_same_tree_with_editable_hulls():
    """The SimReady build keeps every offline hull, gets the articulation API moved from its
    world-fixed joint to the default prim, and its instanceable geometry groups become
    editable, so the spawner can re-value the hand material every hand hull binds itself."""
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
    from dexbench_arena.embodiments.rby1_wuji2.constants import RBY1_WUJI2_USD_PATH
    from dexbench_arena.embodiments.rby1_wuji2.spawn import (
        _normalize_wuji2_layer,
        hand_contact_materials,
        is_simready_build,
        set_hand_contact_material,
    )
    import pytest as _pytest

    if not RBY1_WUJI2_USD_PATH.exists():
        _pytest.skip(f"SimReady Wuji v2 build is unavailable: {RBY1_WUJI2_USD_PATH}")
    source = Usd.Stage.Open(str(RBY1_WUJI2_USD_PATH))
    assert is_simready_build(source.GetDefaultPrim())
    stage = Usd.Stage.Open(_normalize_wuji2_layer(str(RBY1_WUJI2_USD_PATH)))
    root = stage.GetDefaultPrim()
    assert root.HasAPI(UsdPhysics.ArticulationRootAPI)
    assert not stage.GetPrimAtPath(root.GetPath().AppendPath("joints/base_fixed")).HasAPI(UsdPhysics.ArticulationRootAPI)
    bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    assert len(bodies) == 81
    colliders = [p for p in Usd.PrimRange(root) if p.IsA(UsdGeom.Gprim) and p.HasAPI(UsdPhysics.CollisionAPI)]
    attachment_colliders = [p for p in colliders if "/wrist_camera/" in str(p.GetPath())]
    assert len(attachment_colliders) == 20
    assert len(colliders) - len(attachment_colliders) == 572
    assert not any(p.IsInstanceProxy() for p in colliders)
    assert not any("/wheel_" in str(p.GetPath()) for p in colliders)
    assert all(str(p.GetAttribute("physics:approximation").Get()) == "convexHull" for p in colliders)
    thumb = stage.GetPrimAtPath(root.GetPath().AppendPath("right_thumb_proximal/collisions/geom_0"))
    assert thumb.HasAPI(UsdPhysics.CollisionAPI) and not thumb.IsInstanceProxy()
    hand = {str(p.GetPath()) for p in colliders
            if UsdShade.MaterialBindingAPI(p).GetDirectBinding("physics").GetMaterial().GetPrim().GetName() == "hand_contact"}
    assert len(hand) == 56 and sum("tip_sensor_frame" in p for p in hand) == 10
    assert all(UsdShade.MaterialBindingAPI(p).GetDirectBinding("physics").GetMaterial() for p in colliders)
    assert len(hand_contact_materials(root)) == 52
    from dexbench_arena.embodiments.rby1_wuji2.constants import (
        RBY1_WUJI2_MJCF_HAND_CONTACT_MATERIAL,
        RBY1_WUJI2_MJCF_PATH,
    )

    assert not any("friction" in e.attrib for e in ET.parse(RBY1_WUJI2_MJCF_PATH).iter())
    body = set()
    for prim in colliders:
        if str(prim.GetPath()) in hand or "/wrist_camera/" in str(prim.GetPath()):
            continue
        material = UsdShade.MaterialBindingAPI(prim).GetDirectBinding("physics").GetMaterial().GetPrim()
        api = UsdPhysics.MaterialAPI(material)
        body.add((api.GetStaticFrictionAttr().Get(), api.GetDynamicFrictionAttr().Get(),
                  material.GetAttribute("physxMaterial:frictionCombineMode").Get()))
    assert body == {astuple(RBY1_WUJI2_MJCF_HAND_CONTACT_MATERIAL)[:3]}
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        assert set_hand_contact_material(root, 1.2, 1.0, "max") == 52
    assert {UsdPhysics.MaterialAPI(m).GetDynamicFrictionAttr().Get() for m in hand_contact_materials(root)} == {1.0}


def test_collection_collision_filter_keeps_opposing_limbs_and_excludes_own_chain():
    from pxr import Usd, UsdGeom, UsdPhysics
    from dexbench_arena.embodiments.rby1_wuji2.spawn import configure_bimanual_collision_pairs

    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/Robot").GetPrim()
    names = ("left_thumb_distal", "left_index_finger_distal", "left_wrist",
             "right_thumb_distal", "link_right_arm_0", "torso")
    bodies = {}
    for name in names:
        body = UsdGeom.Xform.Define(stage, f"/Robot/{name}").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body)
        bodies[name] = body
    external = UsdGeom.Xform.Define(stage, "/Utensil").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(external)
    report = configure_bimanual_collision_pairs(root)
    assert report['opposing_limb_pairs'] == 6
    for i, name in enumerate(names):
        targets = UsdPhysics.FilteredPairsAPI(bodies[name]).GetFilteredPairsRel().GetTargets()
        for other in names[i + 1:]:
            opposite = (name.startswith('left_') and other.startswith(('right_', 'link_right_arm_')))
            assert (bodies[other].GetPath() in targets) == (not opposite)
        assert external.GetPath() not in targets
    assert not external.HasAPI(UsdPhysics.FilteredPairsAPI)
