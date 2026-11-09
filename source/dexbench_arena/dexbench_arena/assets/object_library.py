"""DexBench-native procedural object assets."""

from __future__ import annotations


import dataclasses
from collections.abc import Callable
from dataclasses import MISSING
import json
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab_arena.relations.physics_settle_params import PhysicsSettleParams
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.utils import bind_physics_material as _bind_physics_material
from isaaclab.sim.utils import clone as _clone_spawner
from isaaclab.utils.configclass import configclass
from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.assets.object_library import LibraryObject, LightBase
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.utils.pose import Pose
from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics

from dexbench_arena import ASSETS_DIR
from dexbench_arena.assets.fixture_looks import fixture_look_spawner
from dexbench_arena.assets.physics_variants import NEWTON_VARIANT, make_static
from dexbench_arena.assets.polybag.sort_variants import BOX_VARIANTS, box_variant_usd_relpath
from dexbench_arena.assets.retail_sku_packages import RETAIL_E1_FC_SKU_PACKAGES
from dexbench_arena.assets.usd_file_spawn import spawn_usd_file
from dexbench_arena.assets.scene_bundles import runtime_usd
from dexbench_arena.evaluation.replay_guards import deterministic_reset
from dexbench_arena.tasks.r1_clutter import (
    R1_FRUIT_KINDS,
    VARIANT_INDICES as R1_VARIANT_INDICES,
    variant_asset_name as r1_variant_asset_name,
)
from dexbench_arena.assets.e2_logistics_cell import (
    E2_L4_BOX_FRICTION,
    E2_L4_PARCELS,
    E2_CAGE_BOX_ASSETS,
    E2_CAGE_CARTON_COLLIDERS,
    E2_HIGH_BAY_LIGHT_ASSET,
    E2_PENDANT_LIGHT_ASSET,
    E2_REFLECT_CARD_ASSET,
    E2_STRIP_LIGHT_ASSET,
    E2_ITEM_ASSET_USDS,
    E2_LIFT_OBJECT_BALLAST_KG,
    E2_LIFT_OBJECTS,
    E2_PARCEL_ASSET,
    E2LiftObject,
    e2_cage_box_asset,
    e2_item_usd,
)
from dexbench_arena.layouts.db_r2_layout import (
    FC_260821_FRIDGE_SHELF3_PART_NAMES,
    check_fridge_shelf3_raise,
)


POOL_DIR = ASSETS_DIR / "object_library"


def pool_usd(name: str, ext: str = "usd") -> str:
    """Path of a pooled asset's main file, by package name."""
    return runtime_usd(POOL_DIR / name / "usd" / f"{name}.{ext}")


RETAIL_E1_ITEM_PHYSICS_MATERIAL_NAME = "physicsMaterial"
RETAIL_E1_ITEM_PHYSICS_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    static_friction=0.6,
    dynamic_friction=0.5,
    restitution=0.0,
)
RETAIL_E1_FC_R3_PHYSICS_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    static_friction=1.0,
    dynamic_friction=0.8,
    restitution=0.0,
)
RETAIL_E1_ITEM_CONTACT_OFFSET_M = 0.002
RETAIL_E1_ITEM_REST_OFFSET_M = 0.0

_FC_VISUAL_ONLY_LAYER_CACHE: dict[str, object] = {}


def _spawn_retail_e1_item_with_collision_impl(
    prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    *,
    collision_approximation: str,
    **kwargs,
):
    """Spawn a delivered E1 retail item with guaranteed collision + friction."""

    from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics

    props = getattr(cfg, "collision_props", None)
    spawn_cfg = cfg
    if getattr(props, "collision_enabled", None) is not None:
        spawn_cfg = cfg.replace(collision_props=props.replace(collision_enabled=None))
    prim = spawn_usd_file(prim_path, cfg.usd_path, spawn_cfg, translation, orientation, **kwargs)
    rigid_props = getattr(cfg, "rigid_props", None)
    sleep_threshold = getattr(rigid_props, "sleep_threshold", None)
    stabilization_threshold = getattr(rigid_props, "stabilization_threshold", None)
    if sleep_threshold is not None and getattr(cfg, "activate_contact_sensors", False):
        cfg.activate_contact_sensors = False
    if sleep_threshold is not None or stabilization_threshold is not None:
        for child_prim in Usd.PrimRange(prim):
            if not child_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(child_prim)
            if sleep_threshold is not None:
                body_api.CreateSleepThresholdAttr(float(sleep_threshold))
            if stabilization_threshold is not None:
                body_api.CreateStabilizationThresholdAttr(float(stabilization_threshold))
    for child_prim in Usd.PrimRange(prim):
        if not child_prim.IsA(UsdGeom.Mesh):
            continue
        if "visonly" in child_prim.GetName().lower():
            continue
        if child_prim.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(child_prim).GetCollisionEnabledAttr().Get() is False:
            continue
        collision_api = UsdPhysics.CollisionAPI.Apply(child_prim)
        collision_api.CreateCollisionEnabledAttr(True)
        mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(child_prim)
        mesh_collision_api.CreateApproximationAttr(collision_approximation)
        physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(child_prim)
        props = getattr(cfg, "collision_props", None)
        contact = getattr(props, "contact_offset", None); rest = getattr(props, "rest_offset", None)
        physx_collision_api.CreateContactOffsetAttr(RETAIL_E1_ITEM_CONTACT_OFFSET_M if contact is None else float(contact))
        physx_collision_api.CreateRestOffsetAttr(RETAIL_E1_ITEM_REST_OFFSET_M if rest is None else float(rest))

    material_cfg = cfg.physics_material or RETAIL_E1_ITEM_PHYSICS_MATERIAL
    material_path = f"{prim_path.rstrip('/')}/{RETAIL_E1_ITEM_PHYSICS_MATERIAL_NAME}"
    material_cfg.func(material_path, material_cfg)
    _bind_physics_material(prim_path, material_path)
    return prim


@_clone_spawner
def _spawn_retail_e1_item_with_collision(
    prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    return _spawn_retail_e1_item_with_collision_impl(
        prim_path,
        cfg,
        translation,
        orientation,
        collision_approximation="convexHull",
        **kwargs,
    )


@_clone_spawner
def _spawn_retail_e1_r3_carton_with_collision(
    prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    return _spawn_retail_e1_item_with_collision_impl(
        prim_path,
        cfg,
        translation,
        orientation,
        collision_approximation="boundingCube",
        **kwargs,
    )


@_clone_spawner
def _spawn_fc_tote_with_collision_proxies(
    prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn an FC tote with fixed box colliders matching its authored proxies."""

    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    prim = spawn_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation, **kwargs)
    authored_proxies = []
    for child_prim in Usd.PrimRange(prim):
        if not child_prim.IsA(UsdGeom.Mesh) or not child_prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        UsdPhysics.CollisionAPI(child_prim).CreateCollisionEnabledAttr(False)
        authored_proxies.append(child_prim)
    assert len(authored_proxies) == 5, (
        f"FC tote expected 5 authored collision proxies, found {len(authored_proxies)} at {prim_path}"
    )
    authored_materials = {
        str(UsdShade.MaterialBindingAPI(proxy).GetDirectBinding("physics").GetMaterialPath()) for proxy in authored_proxies
    }
    assert len(authored_materials) == 1 and "" not in authored_materials, (
        f"FC tote proxies at {prim_path} bind {sorted(authored_materials)}, expected one physics material"
    )
    (proxy_material_path,) = authored_materials

    proxy_specs = {
        "floor": ((0.60, 0.40, 0.005), (0.0, 0.0, 0.0025)),
        "wall_e": ((0.025, 0.40, 0.21), (0.2875, 0.0, 0.105)),
        "wall_w": ((0.025, 0.40, 0.21), (-0.2875, 0.0, 0.105)),
        "wall_n": ((0.60, 0.025, 0.21), (0.0, 0.1875, 0.105)),
        "wall_s": ((0.60, 0.025, 0.21), (0.0, -0.1875, 0.105)),
    }
    for name, (size, center) in proxy_specs.items():
        proxy_cfg = sim_utils.CuboidCfg(
            size=size,
            visible=False,
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=RETAIL_E1_ITEM_CONTACT_OFFSET_M,
                rest_offset=RETAIL_E1_ITEM_REST_OFFSET_M,
            ),
        )
        proxy_cfg.func(
            f"{prim_path}/ArenaCollision/{name}",
            proxy_cfg,
            translation=center,
        )
        _bind_physics_material(f"{prim_path}/ArenaCollision/{name}/geometry/mesh", proxy_material_path)
    return prim


def _spawn_fc_fridge_with_review_state(
    prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None,
    orientation: tuple[float, float, float, float] | None,
    *,
    door_angle_deg: float,
    **kwargs,
):
    """Spawn one FC fridge and author review-only state on the composed stage."""

    from pxr import Gf, UsdGeom

    prim = spawn_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation, **kwargs)
    door = prim.GetChild("Door")
    assert door, f"FC fridge spawned without Door at {prim.GetPath()}"
    door_rotation = door.GetAttribute("xformOp:rotateXYZ")
    assert door_rotation, f"FC fridge Door has no xformOp:rotateXYZ at {door.GetPath()}"
    door_rotation.Set(Gf.Vec3f(0.0, 0.0, door_angle_deg))

    geom = prim.GetChild("Geom")
    assert geom, f"FC fridge spawned without Geom at {prim.GetPath()}"
    stage = prim.GetStage()
    tier_heights_m = {
        "Shelf1": 0.682,
        "Shelf3": 1.160,
        "Shelf4": 1.398,
    }
    composed = () if geom.GetChild("FrgFC_MiniFitS1A") else tier_heights_m.items()
    for shelf_name, height_m in composed:
        for module_name, x_m in (("A", -0.125), ("B", 0.125)):
            source = geom.GetChild(f"FrgFC_MiniFit{module_name}")
            assert source, f"FC fridge missing MiniFit{module_name} prototype at {geom.GetPath()}"
            instance = stage.DefinePrim(
                geom.GetPath().AppendChild(f"FrgFC_MiniFit{module_name}_{shelf_name}"),
                "Xform",
            )
            instance.GetInherits().AddInherit(source.GetPath())
            xform = UsdGeom.Xformable(instance)
            xform.ClearXformOpOrder()
            xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
                Gf.Vec3d(x_m, -0.0325, height_m)
            )
            xform.AddRotateXYZOp(precision=UsdGeom.XformOp.PrecisionFloat).Set(Gf.Vec3f(0.0, 0.0, 0.0))
            xform.AddScaleOp(precision=UsdGeom.XformOp.PrecisionFloat).Set(Gf.Vec3f(1.0, 1.0, 1.0))
    raise_m = getattr(cfg, "shelf3_raise_m", 0.0)
    if raise_m:
        parts = FC_260821_FRIDGE_SHELF3_PART_NAMES
        if not geom.GetChild(parts[1]):
            parts = (parts[0], "FrgFC_MiniFitA_Shelf3", "FrgFC_MiniFitB_Shelf3")
        for name in parts:
            part = geom.GetChild(name)
            assert part, f"FC fridge has no {name} to raise at {geom.GetPath()}"
            api = UsdGeom.XformCommonAPI(part)
            position, _rotation, _scale, _pivot, _order = api.GetXformVectors(0)
            api.SetTranslate(Gf.Vec3d(position[0], position[1], position[2] + raise_m))
    return prim


@configclass
class FcFridgeUsdFileCfg(sim_utils.UsdFileCfg):
    """The FC fridge's spawn cfg with the R2 fixture profile's Shelf3 raise [m]; 0 is the delivered fridge."""

    shelf3_raise_m: float = 0.0


@_clone_spawner
def _spawn_fc_fridge_open_180(
    prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    return _spawn_fc_fridge_with_review_state(
        prim_path,
        cfg,
        translation,
        orientation,
        door_angle_deg=180.0,
        **kwargs,
    )


@_clone_spawner
def _spawn_fc_fridge_closed(
    prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    return _spawn_fc_fridge_with_review_state(
        prim_path,
        cfg,
        translation,
        orientation,
        door_angle_deg=0.0,
        **kwargs,
    )


def _fc_visual_only_layer(usd_path: str):
    """A cached anonymous layer referencing ``usd_path`` with every USD/PhysX physics API removed."""
    from pxr import Sdf, Usd

    resolved_path = str(Path(usd_path).resolve())
    cached = _FC_VISUAL_ONLY_LAYER_CACHE.get(resolved_path)
    if cached is not None:
        return cached

    source_stage = Usd.Stage.Open(resolved_path)
    assert source_stage, f"Unable to open FC stock asset {resolved_path}"
    source_root = source_stage.GetDefaultPrim()
    assert source_root, f"FC stock asset has no default prim: {resolved_path}"

    layer = Sdf.Layer.CreateAnonymous(f"{source_root.GetName()}_visual_only.usda")
    normalized_stage = Usd.Stage.Open(layer)
    normalized_root = normalized_stage.DefinePrim(source_root.GetPath(), source_root.GetTypeName())
    normalized_root.GetReferences().AddReference(resolved_path, source_root.GetPath())
    normalized_stage.SetDefaultPrim(normalized_root)
    for source_prim in Usd.PrimRange(normalized_root):
        for schema_name in list(source_prim.GetAppliedSchemas()):
            if schema_name.startswith(("Physics", "Physx")):
                source_prim.RemoveAppliedSchema(schema_name)
    remaining_physics = [
        str(source_prim.GetPath())
        for source_prim in Usd.PrimRange(normalized_root)
        if any(schema_name.startswith(("Physics", "Physx")) for schema_name in source_prim.GetAppliedSchemas())
    ]
    assert not remaining_physics, f"Visual-only FC stock still has physics schemas: {resolved_path}: {remaining_physics}"
    _FC_VISUAL_ONLY_LAYER_CACHE[resolved_path] = layer
    return layer


@_clone_spawner
def _spawn_fc_visual_stock_item(
    prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn one FC display-stock item: a physics-free reference to its package."""
    from isaaclab.sim.utils import create_prim

    layer = _fc_visual_only_layer(cfg.usd_path)
    prim = create_prim(prim_path, prim_type="Xform", translation=translation, orientation=orientation, scale=cfg.scale)
    assert prim.GetReferences().AddReference(layer.identifier), f"could not reference {layer.identifier} at {prim_path}"
    return prim


def _env_prim_path(prim_path: str) -> str:
    if prim_path.startswith("{ENV_REGEX_NS}"):
        return prim_path
    return f"{{ENV_REGEX_NS}}/{prim_path.lstrip('/')}"


_IDENTITY_XYZW: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)


class _ProceduralRigidAsset(Asset):
    """Small Arena asset wrapper for procedural rigid objects."""

    tags = ["object"]

    def __init__(self, name: str, object_cfg: RigidObjectCfg):
        self.object_type = ObjectType.RIGID
        self.object_cfg = object_cfg
        self.event_cfg = None
        super().__init__(name=name, tags=self.tags)

    def get_object_cfg(self) -> tuple[str, RigidObjectCfg]:
        return self.name, self.object_cfg

    def get_event_cfg(self) -> tuple[str, None]:
        return self.name, self.event_cfg


@register_asset
class DexBenchProceduralTable(_ProceduralRigidAsset):
    """Procedural workspace table for native DexBench smoke scenes."""

    name = "dexbench_table"
    tags = ["object", "workspace"]

    def __init__(
        self,
        prim_path: str = "Table",
        size: tuple[float, float, float] = (1.2, 0.7, 0.05),
        pos: tuple[float, float, float] = (0.0, 0.0, 0.72),
        rot_xyzw: tuple[float, float, float, float] = _IDENTITY_XYZW,
        color: tuple[float, float, float] = (0.6, 0.3, 0.1),
        metallic: float = 0.0,
        kinematic: bool = True,
        initial_pose: Pose | None = None,
        instance_name: str | None = None,
    ):
        """``instance_name`` is the scene-entity name, for a scene with more than one table (default ``dexbench_table``)."""
        asset_name = instance_name if instance_name is not None else self.name
        if initial_pose is not None:
            pos = initial_pose.position_xyz
            rot_xyzw = initial_pose.rotation_xyzw

        object_cfg = RigidObjectCfg(
            prim_path=_env_prim_path(prim_path),
            spawn=sim_utils.CuboidCfg(
                size=size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=kinematic),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=color,
                    metallic=metallic,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot_xyzw),
        )
        object_cfg.name = asset_name
        super().__init__(name=asset_name, object_cfg=object_cfg)


@register_asset
class DexBenchProceduralBlock(_ProceduralRigidAsset):
    """Procedural rigid block used while concrete DexBench object USDs are added."""

    name = "dexbench_block"
    tags = ["object"]

    def __init__(
        self,
        instance_name: str | None = None,
        prim_path: str | None = None,
        size: tuple[float, float, float] = (0.045, 0.045, 0.045),
        color: tuple[float, float, float] = (1.0, 0.0, 0.0),
        pos: tuple[float, float, float] = (0.0, 0.0, 0.77),
        rot_xyzw: tuple[float, float, float, float] = _IDENTITY_XYZW,
        initial_pose: Pose | None = None,
    ):
        asset_name = instance_name if instance_name is not None else self.name
        if prim_path is None:
            prim_path = asset_name
        if initial_pose is not None:
            pos = initial_pose.position_xyz
            rot_xyzw = initial_pose.rotation_xyzw

        object_cfg = RigidObjectCfg(
            prim_path=_env_prim_path(prim_path),
            spawn=sim_utils.CuboidCfg(
                size=size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.7),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot_xyzw),
        )
        object_cfg.name = asset_name
        super().__init__(name=asset_name, object_cfg=object_cfg)


@register_asset
class DexBenchTaskFrame(_ProceduralRigidAsset):
    """Invisible, collider-free task frame recorded as a rigid object."""

    name = "dexbench_task_frame"
    tags = ["object", "task-frame"]

    def __init__(
        self,
        instance_name: str | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
    ):
        asset_name = instance_name if instance_name is not None else self.name
        if prim_path is None:
            prim_path = asset_name
        pos = initial_pose.position_xyz if initial_pose is not None else (0.0, 0.0, 0.0)
        rot_xyzw = initial_pose.rotation_xyzw if initial_pose is not None else _IDENTITY_XYZW
        object_cfg = RigidObjectCfg(
            prim_path=_env_prim_path(prim_path),
            spawn=sim_utils.CuboidCfg(
                size=(0.001, 0.001, 0.001),
                visible=False,
                mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=False,
                    disable_gravity=True,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot_xyzw),
        )
        object_cfg.name = asset_name
        super().__init__(name=asset_name, object_cfg=object_cfg)


class _RetailE1FcStaticFixture(LibraryObject):
    """Static fixture instance from the split FC 260821 asset package."""

    tags = ["object", "retail_store_e1", "fixture-composability"]
    object_type = ObjectType.BASE

    def __init__(
        self,
        instance_name: str | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        scale: tuple[float, float, float] | None = None,
        *,
        look=None,
    ):
        self.look = look
        super().__init__(instance_name=instance_name, prim_path=prim_path, initial_pose=initial_pose, scale=scale)

    def _get_spawn_cfg(self, activate_contact_sensors: bool = False):
        cfg = super()._get_spawn_cfg(activate_contact_sensors)
        if self.look is not None:
            cfg.func = fixture_look_spawner(cfg.func, self.look.usd_path)
        return cfg


@register_asset
class DexBenchRetailE1FcCeilingLights(_RetailE1FcStaticFixture):
    """Ceiling fixture geometry; illumination remains an Arena light asset."""

    name = "dexbench_retail_e1_fc_ceiling_lights"
    usd_path = runtime_usd(ASSETS_DIR / "env" / "e1_retail" / "fc" / "shell" / "ceiling_lights.usdc")


@register_asset
class DexBenchRetailE1FcGondola(_RetailE1FcStaticFixture):
    name = "dexbench_retail_e1_fc_gondola"
    usd_path = str(ASSETS_DIR / "object_library" / "gondola_fc" / "usd" / "gondola_fc.usd")


@register_asset
class DexBenchRetailE1FcProduceDisplay(_RetailE1FcStaticFixture):
    name = "dexbench_retail_e1_fc_produce_display"
    usd_path = str(ASSETS_DIR / "object_library" / "produce_display_fc" / "usd" / "produce_display_fc.usd")


@register_asset
class DexBenchRetailE1FcTote(_RetailE1FcStaticFixture):
    """One fixed tote; the scene composes four instances into the reviewed stack."""

    name = "dexbench_retail_e1_fc_tote"
    usd_path = str(ASSETS_DIR / "object_library" / "tote_fc" / "usd" / "tote_fc.usd")
    spawn_cfg_addon = {"func": _spawn_fc_tote_with_collision_proxies}


@register_asset
class DexBenchRetailE1FcShoppingCart(_RetailE1FcStaticFixture):
    """PSL 260914 shopping cart: W600 x D440 x H920, cradle plane at z 0.600 that
    carries a detachable shopping basket. A support fixture, never a task object."""

    name = "dexbench_retail_e1_fc_shopping_cart"
    usd_path = str(ASSETS_DIR / "object_library" / "shopping_cart_fc" / "usd" / "shopping_cart_fc.usd")


@register_asset
class DexBenchRetailE1FcShoppingBasket(_RetailE1FcStaticFixture):
    """PSL 260914 detachable shopping basket, the DB-R1/DB-R2 destination when it
    sits on a cart's cradle. Collision proxies: floor top at local z 0.003,
    interior x +/-0.1775, y +/-0.1175, wall top 0.230."""

    name = "dexbench_retail_e1_fc_shopping_basket"
    usd_path = str(ASSETS_DIR / "object_library" / "shopping_basket_fc" / "usd" / "shopping_basket_fc.usd")


class _DexBenchRetailE1FcFridge(_RetailE1FcStaticFixture):
    """Static FC fridge composed with all-tier MiniFit modules at spawn."""

    tags = [*_RetailE1FcStaticFixture.tags, "refrigerator", "minifit-all-tiers"]
    usd_path = str(ASSETS_DIR / "object_library" / "fridge_minifit_fc" / "usd" / "fridge_minifit_fc.usd")

    def __init__(
        self,
        instance_name: str | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        scale: tuple[float, float, float] | None = None,
        *,
        shelf3_raise_m: float = 0.0,
        look=None,
    ):
        self.shelf3_raise_m = check_fridge_shelf3_raise(shelf3_raise_m)
        super().__init__(
            instance_name=instance_name, prim_path=prim_path, initial_pose=initial_pose, scale=scale, look=look
        )

    def _get_spawn_cfg(self, activate_contact_sensors: bool = False):
        cfg = super()._get_spawn_cfg(activate_contact_sensors)
        if not self.shelf3_raise_m:
            return cfg
        values = {field.name: getattr(cfg, field.name) for field in dataclasses.fields(cfg)}
        return FcFridgeUsdFileCfg(**values, shelf3_raise_m=self.shelf3_raise_m)


@register_asset
class DexBenchRetailE1FcFridgeOpen180(_DexBenchRetailE1FcFridge):
    name = "dexbench_retail_e1_fc_fridge_open_180"
    spawn_cfg_addon = {"func": _spawn_fc_fridge_open_180}


@register_asset
class DexBenchRetailE1FcFridgeClosed(_DexBenchRetailE1FcFridge):
    name = "dexbench_retail_e1_fc_fridge_closed"
    spawn_cfg_addon = {"func": _spawn_fc_fridge_closed}


class _RetailE1BoxItem(LibraryObject):
    """Dynamic rigid boxed-goods item from the PhysicsSim Lab E1 delivery."""

    tags = ["object", "retail_store_e1"]
    object_type = ObjectType.RIGID
    spawn_cfg_addon = {
        "func": _spawn_retail_e1_item_with_collision,
        "rigid_props": sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=False,
            disable_gravity=False,
            max_depenetration_velocity=5.0,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=4,
        ),
        "collision_props": sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=RETAIL_E1_ITEM_CONTACT_OFFSET_M,
            rest_offset=RETAIL_E1_ITEM_REST_OFFSET_M,
        ),
    }

    def __init__(
        self,
        instance_name: str | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        scale: tuple[float, float, float] | None = None,
    ):
        super().__init__(instance_name=instance_name, prim_path=prim_path, initial_pose=initial_pose, scale=scale)


class _RetailE1FcDrinkItem(_RetailE1BoxItem):
    """Dynamic Shelf2 drink from the reviewed FC 260821 asset package."""

    tags = ["object", "retail_store_e1", "fixture-composability", "fridge_shelf2"]


RETAIL_E1_PRODUCE_CONTACT_OFFSET_M = 0.003
RETAIL_E1_PRODUCE_REST_OFFSET_M = 0.001
RETAIL_E1_PRODUCE_MAX_DEPENETRATION_VELOCITY_M_S = 0.1
RETAIL_E1_PRODUCE_LINEAR_DAMPING = 0.05
RETAIL_E1_PRODUCE_REST_GATES = (0.1, 6.0)
RETAIL_E1_PRODUCE_SETTLE_PARAMS = PhysicsSettleParams(num_steps=30, lin_vel_thresh=RETAIL_E1_PRODUCE_REST_GATES[0], ang_vel_thresh=RETAIL_E1_PRODUCE_REST_GATES[1])
RETAIL_E1_PRODUCE_ANGULAR_DAMPING = 2.0
RETAIL_E1_PRODUCE_SLEEP_THRESHOLD = 0.005
RETAIL_E1_PRODUCE_STABILIZATION_THRESHOLD = 0.02


class _RetailE1FcProduceItem(_RetailE1BoxItem):
    """Dynamic E1 produce: DB-R1's pile variants, the SpaceAI scans and the delivery's own produce proxies."""

    tags = ["object", "retail_store_e1", "fixture-composability", "db-r1"]
    spawn_cfg_addon = {
        **_RetailE1BoxItem.spawn_cfg_addon,
        "rigid_props": sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=False,
            disable_gravity=False,
            max_depenetration_velocity=RETAIL_E1_PRODUCE_MAX_DEPENETRATION_VELOCITY_M_S,
            linear_damping=RETAIL_E1_PRODUCE_LINEAR_DAMPING,
            angular_damping=RETAIL_E1_PRODUCE_ANGULAR_DAMPING,
            sleep_threshold=RETAIL_E1_PRODUCE_SLEEP_THRESHOLD,
            stabilization_threshold=RETAIL_E1_PRODUCE_STABILIZATION_THRESHOLD,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=4,
        ),
        "collision_props": sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=RETAIL_E1_PRODUCE_CONTACT_OFFSET_M,
            rest_offset=RETAIL_E1_PRODUCE_REST_OFFSET_M,
        ),
    }


class _RetailE1FcR3Carton(_RetailE1BoxItem):
    """Canonical DB-R3 carton proxy from the reviewed FC 260821 package."""

    tags = ["object", "retail_store_e1", "fixture-composability", "db-r3"]
    spawn_cfg_addon = {
        **_RetailE1BoxItem.spawn_cfg_addon,
        "func": _spawn_retail_e1_r3_carton_with_collision,
        "physics_material": RETAIL_E1_FC_R3_PHYSICS_MATERIAL,
    }


def retail_e1_fc_sku_usd(sku: str) -> str:
    """Main USD of the package an E1 FC SKU code resolves to."""
    return pool_usd(RETAIL_E1_FC_SKU_PACKAGES[sku])


@register_asset
class DexBenchRetailE1FcR3T1(_RetailE1FcR3Carton):
    name = "dexbench_retail_e1_fc_r3t1"
    usd_path = retail_e1_fc_sku_usd("R3T1")


@register_asset
class DexBenchRetailE1FcR3T2(_RetailE1FcR3Carton):
    name = "dexbench_retail_e1_fc_r3t2"
    usd_path = retail_e1_fc_sku_usd("R3T2")


@register_asset
class DexBenchRetailE1FcR3T3(_RetailE1FcR3Carton):
    name = "dexbench_retail_e1_fc_r3t3"
    usd_path = retail_e1_fc_sku_usd("R3T3")


@register_asset
class DexBenchRetailE1FcR3T4(_RetailE1FcR3Carton):
    name = "dexbench_retail_e1_fc_r3t4"
    usd_path = retail_e1_fc_sku_usd("R3T4")


@register_asset
class DexBenchRetailE1FcR3E1(_RetailE1FcR3Carton):
    name = "dexbench_retail_e1_fc_r3e1"
    usd_path = retail_e1_fc_sku_usd("R3E1")


@register_asset
class DexBenchRetailE1FcR3E2(_RetailE1FcR3Carton):
    name = "dexbench_retail_e1_fc_r3e2"
    usd_path = retail_e1_fc_sku_usd("R3E2")


@register_asset
class DexBenchRetailE1FcR3E4(_RetailE1FcR3Carton):
    name = "dexbench_retail_e1_fc_r3e4"
    usd_path = retail_e1_fc_sku_usd("R3E4")


@register_asset
class DexBenchRetailE1FcR3E5(_RetailE1FcR3Carton):
    name = "dexbench_retail_e1_fc_r3e5"
    usd_path = retail_e1_fc_sku_usd("R3E5")


@register_asset
class DexBenchRetailE1FcR3E6(_RetailE1FcR3Carton):
    name = "dexbench_retail_e1_fc_r3e6"
    usd_path = retail_e1_fc_sku_usd("R3E6")


@register_asset
class DexBenchRetailE1FcTc2(_RetailE1FcDrinkItem):
    name = "dexbench_retail_e1_fc_tc2"
    usd_path = retail_e1_fc_sku_usd("TC2")


@register_asset
class DexBenchRetailE1FcEc1(_RetailE1FcDrinkItem):
    name = "dexbench_retail_e1_fc_ec1"
    usd_path = retail_e1_fc_sku_usd("EC1")


@register_asset
class DexBenchRetailE1FcTc1(_RetailE1FcDrinkItem):
    name = "dexbench_retail_e1_fc_tc1"
    usd_path = retail_e1_fc_sku_usd("TC1")


@register_asset
class DexBenchRetailE1FcEc2(_RetailE1FcDrinkItem):
    name = "dexbench_retail_e1_fc_ec2"
    usd_path = retail_e1_fc_sku_usd("EC2")


@register_asset
class DexBenchRetailE1FcTp1(_RetailE1FcDrinkItem):
    name = "dexbench_retail_e1_fc_tp1"
    usd_path = retail_e1_fc_sku_usd("TP1")


@register_asset
class DexBenchRetailE1FcEp2(_RetailE1FcDrinkItem):
    name = "dexbench_retail_e1_fc_ep2"
    usd_path = retail_e1_fc_sku_usd("EP2")


@register_asset
class DexBenchRetailE1FcEp3(_RetailE1FcDrinkItem):
    name = "dexbench_retail_e1_fc_ep3"
    usd_path = retail_e1_fc_sku_usd("EP3")


def _register_library_item(
    class_name: str, base: type, asset_name: str, usd_path: str, doc: str, extra: dict | None = None
) -> type:
    """Register a LibraryObject subclass that differs from ``base`` by name, usd, docstring and ``extra`` attributes."""

    cls = type(class_name, (base,), {"__doc__": doc, "name": asset_name, "usd_path": usd_path, **(extra or {})})
    register_asset(cls)
    return cls


for _fruit_sku in (
    "APPLE_SCAN", "PLUM_SCAN", "STRAWBERRY_SCAN", "BANANA_SCAN", "R1_PEACH",
    "YELLOW_LEMON_FC", "RIPE_TOMATO_FC", "FRESH_ORANGE_FC", "RIPE_MANGO_FC",
):
    _register_library_item(
        f"DexBenchRetailE1Fc{_fruit_sku.title().replace('_', '')}",
        _RetailE1FcProduceItem,
        f"dexbench_retail_e1_fc_{_fruit_sku.lower()}",
        retail_e1_fc_sku_usd(_fruit_sku),
        f"E1 FC produce {_fruit_sku}: display/bake prop; DB-R1 piles use per-kind variants.",
    )

def _package_physics_material(package: str) -> sim_utils.RigidBodyMaterialCfg:
    """The friction/restitution a package's sidecar records. A missing sidecar raises: the spawn's default
    friction in its place would run the fruit on values nobody delivered."""
    sidecar = POOL_DIR / package / "usd" / f"{package}.meta.json"
    if not sidecar.is_file():
        raise FileNotFoundError(f"{package}: no sidecar at {sidecar}, so no delivered friction")
    material = json.loads(sidecar.read_text())["asset"]["physics_material"]
    return sim_utils.RigidBodyMaterialCfg(
        static_friction=material["static_friction"],
        dynamic_friction=material["dynamic_friction"],
        restitution=material["restitution"],
    )


for _r1_variant in (pkg for kind in R1_FRUIT_KINDS for pkg in (f"{kind}_v{i:02d}" for i in R1_VARIANT_INDICES)):
    _material = _package_physics_material(_r1_variant)
    _register_library_item(
        f"DexBenchRetailE1Fc{_r1_variant.title().replace('_', '')}",
        _RetailE1FcProduceItem,
        r1_variant_asset_name(_r1_variant),
        pool_usd(_r1_variant),
        f"DB-R1 fruit variant {_r1_variant}: one kind's pile member, dynamic.",
        {"spawn_cfg_addon": {**_RetailE1FcProduceItem.spawn_cfg_addon, "physics_material": _material}},
    )




class _RetailE1FcVisualStock(LibraryObject):
    """One SKU of FC display stock, rendered at a Python-owned pose with no physics."""

    tags = ["object", "retail_store_e1", "fixture-composability", "visual-stock"]
    object_type = ObjectType.BASE
    spawn_cfg_addon = {
        "func": _spawn_fc_visual_stock_item,
    }

    def __init__(
        self,
        instance_name: str | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        scale: tuple[float, float, float] | None = None,
    ):
        super().__init__(instance_name=instance_name, prim_path=prim_path, initial_pose=initial_pose, scale=scale)


RETAIL_E1_FC_STOCK_ASSET_PREFIX = "dexbench_retail_e1_fc_stock_"


def retail_e1_fc_stock_asset_name(sku: str) -> str:
    """Registry name of the visual-stock asset for an E1 FC SKU code."""
    return RETAIL_E1_FC_STOCK_ASSET_PREFIX + sku.lower()


for _sku in RETAIL_E1_FC_SKU_PACKAGES:
    register_asset(
        type(
            "DexBenchRetailE1FcStock" + _sku.title().replace("_", ""),
            (_RetailE1FcVisualStock,),
            {
                "name": retail_e1_fc_stock_asset_name(_sku),
                "usd_path": retail_e1_fc_sku_usd(_sku),
                "__doc__": f"Visual-only FC display stock for SKU {_sku} ({RETAIL_E1_FC_SKU_PACKAGES[_sku]}).",
                "__module__": __name__,
            },
        )
    )




@deterministic_reset
def set_kinematic_object_pose(env, env_ids, asset_cfg, pose: Pose) -> None:
    """Reset event for a kinematic body: the pose, and nothing else."""

    if env_ids is None:
        return
    asset = env.scene[asset_cfg.name]
    root_pose = pose.to_tensor(device=env.device).repeat(len(env_ids), 1)
    root_pose[:, :3] += env.scene.env_origins[env_ids]
    asset.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)


class KinematicObjectMixin:
    """What a kinematic rigid object needs on top of its LibraryObject family."""

    object_type = ObjectType.RIGID

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.variations.pop("mass", None)

    def _build_reset_event(self):
        from isaaclab.managers import EventTermCfg, SceneEntityCfg

        if not self._requires_reset_pose_event():
            return None
        initial_pose = self.get_initial_pose()
        assert isinstance(initial_pose, Pose), (
            f"{type(self).__name__} is kinematic and takes a single Pose as initial_pose; every other "
            "kind of placement resets through a velocity write PhysX rejects for kinematic bodies."
        )
        return EventTermCfg(
            func=set_kinematic_object_pose,
            mode="reset",
            params={"asset_cfg": SceneEntityCfg(self.name), "pose": initial_pose},
        )


@register_asset
class LogisticsPostOfficePackage(LibraryObject):
    """Local post-office parcel (box_5): 0.272 (X) x 0.188 (Y) x 0.160 m."""

    name = E2_PARCEL_ASSET
    tags = ["object", "logistics_center"]
    usd_path = pool_usd("box_5")
    object_type = ObjectType.RIGID
    spawn_cfg_addon = {
        "rigid_props": sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=False,
            disable_gravity=False,
            max_depenetration_velocity=5.0,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=4,
        ),
    }




@_clone_spawner
def _spawn_e2_static_prop(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """A pool package as a static prop on every engine: spawned from its usd, every prim made static."""

    prim = spawn_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation, **kwargs)
    for part in Usd.PrimRange(prim):
        make_static(part)
    return prim


class _LogisticsE2StaticItem(LibraryObject):
    """An E2 prop as a static collider, or drawn only when its package has no collider."""

    tags = ["object", "logistics_center_e2"]
    object_type = ObjectType.BASE
    spawn_cfg_addon = {"func": _spawn_e2_static_prop}

    def __init__(
        self,
        instance_name: str | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        scale: tuple[float, float, float] | None = None,
    ):
        super().__init__(instance_name=instance_name, prim_path=prim_path, initial_pose=initial_pose, scale=scale)


def _register_logistics_e2_items() -> None:
    """One registered static asset per entry of E2_ITEM_ASSET_USDS; the table is the list."""

    for asset_name, usd_name in E2_ITEM_ASSET_USDS.items():
        _register_library_item(
            "DexBenchLogisticsE2Item_" + asset_name.removeprefix("dexbench_"),
            _LogisticsE2StaticItem,
            asset_name,
            pool_usd(usd_name),
            f"PhysicsSim Lab E2 belt item, the pool package {usd_name}, as a static prop.",
        )


_register_logistics_e2_items()


class _LogisticsE2KinematicItem(KinematicObjectMixin, LibraryObject):
    """A delivered E2 prop as a kinematic rigid body: solid like a static collider, movable at reset."""

    tags = ["object", "logistics_center_e2"]
    spawn_cfg_addon = {
        "rigid_props": sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    }

    def __init__(
        self,
        instance_name: str | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        scale: tuple[float, float, float] | None = None,
    ):
        super().__init__(instance_name=instance_name, prim_path=prim_path, initial_pose=initial_pose, scale=scale)


def _register_logistics_e2_kinematic_items() -> None:
    """A kinematic twin, named <asset>_kinematic, per asset the cage stack is built from."""

    for asset_name in E2_CAGE_BOX_ASSETS:
        _register_library_item(
            "DexBenchLogisticsE2KinematicItem_" + asset_name.removeprefix("dexbench_"),
            _LogisticsE2KinematicItem,
            e2_cage_box_asset(asset_name),
            str(e2_item_usd(asset_name)),
            f"E2 cage box: {E2_CAGE_CARTON_COLLIDERS[asset_name]}'s body dressed as PSL 261014's carton, "
            "as a kinematic rigid body.",
        )


_register_logistics_e2_kinematic_items()


class _LogisticsE2LiftBox(LibraryObject):
    """A box for the E2 lift_to_conveyor task: a corrugated-carton package at a spec size."""

    tags = ["object", "logistics_center_e2"]
    object_type = ObjectType.RIGID
    spawn_cfg_addon = {
        "rigid_props": LogisticsPostOfficePackage.spawn_cfg_addon["rigid_props"],
        "mass_props": sim_utils.MassPropertiesCfg(mass=E2_LIFT_OBJECT_BALLAST_KG),
    }


def _register_logistics_e2_lift_boxes() -> None:
    """One asset per box of E2_LIFT_OBJECTS; the parcel stand-in is LogisticsPostOfficePackage itself."""

    for lift_object in E2_LIFT_OBJECTS:
        if lift_object.package is None:
            continue
        size_mm = " x ".join(str(int(round(extent * 1000))) for extent in lift_object.size_xyz_m)
        _register_library_item(
            "DexBenchLogisticsE2LiftBox_" + lift_object.name,
            _LogisticsE2LiftBox,
            lift_object.asset,
            pool_usd(lift_object.package),
            f"E2 lift box {size_mm} mm ({lift_object.group} set): object_library/{lift_object.package}.",
        )


_register_logistics_e2_lift_boxes()




class _L1StreamBox(LibraryObject):
    """A DB-L1 stream box: a labelled Korea Post box, dynamic, colliding as one box of its size (its Newton runtime)."""

    tags = ["object", "logistics_center_e2"]
    object_type = ObjectType.RIGID
    spawn_cfg_addon = {"rigid_props": LogisticsPostOfficePackage.spawn_cfg_addon["rigid_props"]}

    def _get_spawn_cfg(self, activate_contact_sensors: bool = False):
        return sim_utils.UsdFileCfg(
            usd_path=self.usd_path,
            scale=self.scale,
            activate_contact_sensors=activate_contact_sensors,
            variants={NEWTON_VARIANT: "Enabled"},
            **self.spawn_cfg_addon,
        )


def _register_l1_stream_boxes() -> None:
    for variant in BOX_VARIANTS:
        size_mm = " x ".join(str(int(round(extent * 1000))) for extent in variant.size_m)
        _register_library_item(
            "DexBenchL1StreamBox_" + variant.key,
            _L1StreamBox,
            variant.asset,
            str(POOL_DIR / box_variant_usd_relpath(variant.key)),
            f"Korea Post box No.{variant.number} with a {variant.sort_code} shipping label, one {size_mm} mm box "
            "collider.",
        )


_register_l1_stream_boxes()




@_clone_spawner
def _spawn_parcel_with_box_collider(
    prim_path: str,
    cfg: L4BoxParcelSpawnCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    prim = spawn_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation, **kwargs)
    for descendant in Usd.PrimRange(prim):
        if descendant.IsInstanceable():
            descendant.SetInstanceable(False)
    bodies = []
    for descendant in Usd.PrimRange(prim):
        if descendant.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(descendant).CreateCollisionEnabledAttr(False)
        if descendant.HasAPI(UsdPhysics.RigidBodyAPI):
            bodies.append(descendant)
    if len(bodies) != 1:
        raise RuntimeError(f"{cfg.usd_path}: expected one rigid body under {prim_path}, found {len(bodies)}")
    sx, sy, sz = cfg.collision_size_xyz_m
    cube = UsdGeom.Cube.Define(prim.GetStage(), bodies[0].GetPath().AppendChild("collision_box"))
    cube.CreateSizeAttr(1.0)
    cube.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, sz / 2.0))
    cube.AddScaleOp().Set(Gf.Vec3f(sx, sy, sz))
    cube.CreatePurposeAttr(UsdGeom.Tokens.guide)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True)
    material_path = f"{prim_path.rstrip('/')}/collision_box_material"
    material = sim_utils.RigidBodyMaterialCfg(static_friction=cfg.collision_friction,
                                              dynamic_friction=cfg.collision_friction)
    material.func(material_path, material)
    _bind_physics_material(str(cube.GetPath()), material_path)
    return prim


@configclass
class L4BoxParcelSpawnCfg(sim_utils.UsdFileCfg):
    """``UsdFileCfg`` for a DB-L4 box parcel, carrying the size of the one box it collides as."""

    func: Callable = _spawn_parcel_with_box_collider
    collision_size_xyz_m: tuple[float, float, float] = MISSING
    collision_friction: float = E2_L4_BOX_FRICTION
    """The box collider's static and dynamic friction."""


class _LogisticsL4BoxParcel(LibraryObject):
    """A DB-L4 box parcel: a pool package, dynamic, with one box collider (E2_L4_PARCELS in the cell module)."""

    tags = ["object", "logistics_center_e2"]
    object_type = ObjectType.RIGID
    spawn_cfg_addon = {"rigid_props": LogisticsPostOfficePackage.spawn_cfg_addon["rigid_props"]}
    collision_size_xyz_m: tuple[float, float, float]
    nominal_mass_kg: float

    def _get_spawn_cfg(self, activate_contact_sensors: bool = False):
        return L4BoxParcelSpawnCfg(
            usd_path=self.usd_path,
            scale=self.scale,
            activate_contact_sensors=activate_contact_sensors,
            collision_size_xyz_m=self.collision_size_xyz_m,
            mass_props=sim_utils.MassPropertiesCfg(mass=self.nominal_mass_kg),
            **self.spawn_cfg_addon,
        )


def _register_logistics_l4_box_parcels() -> None:
    for parcel in E2_L4_PARCELS:
        if parcel.is_polybag:
            continue
        usd = next(path for path in (pool_usd(parcel.package), pool_usd(parcel.package, "usda")) if Path(path).is_file())
        size_mm = " x ".join(str(int(round(extent * 1000))) for extent in parcel.collider_size_xyz_m)
        _register_library_item(
            "DexBenchLogisticsL4BoxParcel_" + parcel.name,
            _LogisticsL4BoxParcel,
            parcel.asset,
            usd,
            f"DB-L4 parcel {parcel.name}: object_library/{parcel.package}, one {size_mm} mm box collider.",
            extra={"collision_size_xyz_m": parcel.collider_size_xyz_m, "nominal_mass_kg": parcel.nominal_mass_kg},
        )


_register_logistics_l4_box_parcels()


E2_LIFT_BOX_SET_MEMBER_PRIM = "lift_box"


def e2_lift_box_set_member_usd(lift_object: E2LiftObject, cache_dir: Path | None = None) -> str:
    """A self-standing usda for ``lift_object`` with its rigid body on the default prim, for a multi-asset spawner."""

    from isaaclab_arena.assets.asset_cache import get_arena_asset_cache_dir
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    from dexbench_arena.assets.physics_variants import physics_variants_of

    assert lift_object.package is not None, f"{lift_object.name} is not a box package; spawn it as its own asset"
    cache_dir = Path(cache_dir) if cache_dir is not None else get_arena_asset_cache_dir()
    package_usd = pool_usd(lift_object.package)
    member = cache_dir / f"{lift_object.asset}.usda"
    sources = [(member, package_usd)]

    for path, source in sources:
        source_stage = Usd.Stage.Open(source)
        source_prim = source_stage.GetDefaultPrim()
        source_root = source_prim.GetPath()
        bodies = [
            prim for prim in source_prim.GetChildren()
            if prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ]
        assert len(bodies) == 1, f"{source}: expected one rigid body directly under {source_root}"
        body_schemas = [
            schema
            for schema in bodies[0].GetAppliedSchemas()
            if "RigidBody" in schema or "Mass" in schema
        ]
        body_schemas += [schema for schema in ("PhysxRigidBodyAPI",) if schema not in body_schemas]

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
        root = UsdGeom.Xform.Define(stage, Sdf.Path.absoluteRootPath.AppendChild(E2_LIFT_BOX_SET_MEMBER_PRIM))
        root.GetPrim().GetReferences().AddReference(source, source_root)
        UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
        UsdPhysics.MassAPI.Apply(root.GetPrim())
        stage.SetDefaultPrim(root.GetPrim())
        inner = Sdf.CreatePrimInLayer(stage.GetRootLayer(), root.GetPath().AppendChild(bodies[0].GetName()))
        inner.specifier = Sdf.SpecifierOver
        inner.SetInfo("apiSchemas", Sdf.TokenListOp.Create(deletedItems=body_schemas))
        text = stage.GetRootLayer().ExportToString()
        path.parent.mkdir(parents=True, exist_ok=True)
        if not (path.is_file() and path.read_text() == text):
            path.write_text(text)
        physics = physics_variants_of(source)
        if physics:
            sidecar = path.with_name(f"{path.stem}.meta.json")
            text = json.dumps({"asset": {"variants": {"physics": physics}}}, indent=2) + "\n"
            if not (sidecar.is_file() and sidecar.read_text() == text):
                sidecar.write_text(text)
    return str(member)


@register_asset
class AssemblyWorkbench(LibraryObject):
    """Shipped wood-laminate workbench with steel legs and leveling feet."""

    name = "assembly_workbench"
    tags = ["object", "manufacturing", "workspace"]
    usd_path = pool_usd("assembly_workbench_fc")
    object_type = ObjectType.BASE

    def __init__(self, instance_name: str | None = None, prim_path: str | None = None,
                 initial_pose: Pose | None = None, scale: tuple[float, float, float] | None = None):
        super().__init__(instance_name=instance_name, prim_path=prim_path, initial_pose=initial_pose, scale=scale)


class _AssemblyLineE3Prop(LibraryObject):
    """Base class for DexBench E3 assembly-line item meshes (kit box, loose parts)."""

    tags = ["object", "manufacturing"]
    object_type = ObjectType.RIGID

    def __init__(
        self,
        instance_name: str | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        scale: tuple[float, float, float] | None = None,
    ):
        super().__init__(instance_name=instance_name, prim_path=prim_path, initial_pose=initial_pose, scale=scale)


@register_asset
class AssemblyLineE3KitBox(_AssemblyLineE3Prop):
    """DexBench E3 M5 moving kit box (0.370 x 0.275 x 0.083 m, 2x2 compartments, blue PPC)."""

    name = "dexbench_assembly_line_e3_kit_box"
    usd_path = pool_usd("kit_box_fc")


@register_asset
class AssemblyLineE3PartBoltM24(_AssemblyLineE3Prop):
    """DexBench E3 M5 kitting part -- an M24 bolt, one of the delivery's loose-part types."""

    name = "dexbench_assembly_line_e3_part_bolt_m24"
    usd_path = pool_usd("part_bolt_m24_fc")


@register_asset
class AssemblyLineE3PartPomRoller(_AssemblyLineE3Prop):
    """DexBench E3 M5 kitting part -- a POM conveyor roller, axis along local +X, 0.35 kg."""

    name = "dexbench_assembly_line_e3_part_pom_roller"
    usd_path = pool_usd("part_pom_roller_fc")


@register_asset
class AssemblyLineE3PartElbow50A(_AssemblyLineE3Prop):
    """DexBench E3 M5 kitting part -- a 50A stainless pipe elbow, 0.55 kg."""

    name = "dexbench_assembly_line_e3_part_elbow_50a"
    usd_path = pool_usd("part_elbow_50a_fc")


@register_asset
class AssemblyLineE3PartBearing6206(_AssemblyLineE3Prop):
    """DexBench E3 M5 kitting part -- a 6206 ball bearing lying flat, 0.2 kg."""

    name = "dexbench_assembly_line_e3_part_bearing_6206"
    usd_path = pool_usd("part_bearing_6206_fc")


@register_asset
class AssemblyLineE3PartNema17(_AssemblyLineE3Prop):
    """DexBench E3 M5 kitting part -- a NEMA 17 stepper motor standing on its end (shaft +Z), 0.35 kg."""

    name = "dexbench_assembly_line_e3_part_nema17"
    usd_path = pool_usd("part_nema17_fc")


M5_POOL_PART_PACKAGES = (
    "automotive_exhaust_90_bend_od63",
    "boltkorea_iso4014_class_8_8_hex_bolt_m36x160",
    "deco_ad02_2_station_ng6cetop3_manifold",
    "gy1_rigid_flange_coupling_half_flange",
    "jl_sus304_welded_tee_50a",
    "norgren_rt_57232_m_50_roundline_cylinder",
    "rs_pro_stepper_motor_535_0502_nema23_class",
    "two_piece_split_shaft_collar_od45_bore2_w12",
    "tyb_ucp205_pillow_block_bearing_unit",
    "v_belt_pulley_1a_70_12_4",
)
for _package in M5_POOL_PART_PACKAGES:
    _register_library_item(
        "AssemblyLineE3Part" + _package.title().replace("_", ""),
        _AssemblyLineE3Prop,
        f"dexbench_assembly_line_e3_part_{_package}",
        pool_usd(_package),
        f"DexBench E3 M5 kitting part from the object-library pool: {_package}.",
    )




@register_asset
class LgFh25eaeWasher(KinematicObjectMixin, LibraryObject):
    """LG FH25EAE washer, the DB-M1 fixture: one kinematic body, 112 kg, with the drawer slot."""

    name = "lg_fh25eae_washer"
    tags = ["object", "manufacturing"]
    usd_path = pool_usd("lg_fh25eae_washer")
    spawn_cfg_addon = {
        "rigid_props": sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    }

    def __init__(
        self,
        instance_name: str | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        scale: tuple[float, float, float] | None = None,
    ):
        super().__init__(instance_name=instance_name, prim_path=prim_path, initial_pose=initial_pose, scale=scale)


@register_asset
class LgFh25eaeDetergentDrawer(LibraryObject):
    """LG FH25EAE detergent drawer, the DB-M1 handled part: one dynamic body, 1.21 kg (authored)."""

    name = "lg_fh25eae_detergent_drawer"
    tags = ["object", "manufacturing"]
    usd_path = pool_usd("lg_fh25eae_detergent_drawer")
    object_type = ObjectType.RIGID




@register_asset
class LgMnext3LinerConnectorTask(LibraryObject):
    """LG M-Next3 refrigerator liner (Liner_2, pre-foam) with harness connectors S1 and S2, the DB-M2 fixture."""

    name = "lg_mnext3_liner_connector_task"
    tags = ["object", "manufacturing"]
    usd_path = pool_usd("lg_mnext3_liner_connector_task")
    object_type = ObjectType.BASE
    physics_variant_on_base = True




@register_asset
class LgFx25BaseCabinet(KinematicObjectMixin, LibraryObject):
    """LG FX25 base cabinet, upside down, the DB-M4 workpiece: one kinematic body."""

    name = "lg_fx25_base_cabinet"
    tags = ["object", "manufacturing"]
    usd_path = pool_usd("lg_fx25_base_cabinet")
    spawn_cfg_addon = {
        "rigid_props": sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    }

    def __init__(
        self,
        instance_name: str | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        scale: tuple[float, float, float] | None = None,
    ):
        super().__init__(instance_name=instance_name, prim_path=prim_path, initial_pose=initial_pose, scale=scale)


@register_asset
class LgFx25LevelingLeg(LibraryObject):
    """LG FX25 leveling leg, the DB-M4 handled part: one dynamic body, 0.0696 kg (N.Light's estimate)."""

    name = "lg_fx25_leveling_leg"
    tags = ["object", "manufacturing"]
    usd_path = pool_usd("lg_fx25_leveling_leg")
    object_type = ObjectType.RIGID


class DexBenchInteriorLightBase(LightBase):
    """Base for lights that sit inside enclosed rooms, carrying the variations."""

    default_prim_path = None

    def __init__(
        self,
        instance_name: str | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        spawner_cfg: sim_utils.LightCfg | None = None,
        position_range_m: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None,
    ):
        """
        Args:
            position_range_m: ``(low, high)`` of the position variation's ``[dx, dy, dz]`` offset
                [m]. None keeps LightPositionVariationCfg's default, sized for the E1 room; a light
                near a wall or inside a fixture passes the room it has.
        """
        from isaaclab_arena.variations.light_color_temperature_variation import LightColorTemperatureVariation
        from isaaclab_arena.variations.light_color_variation import LightColorVariation
        from isaaclab_arena.variations.light_intensity_variation import (
            LightIntensityVariation,
            LightIntensityVariationCfg,
        )
        from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg

        super().__init__(
            instance_name=instance_name,
            prim_path=prim_path,
            initial_pose=initial_pose,
            spawner_cfg=spawner_cfg if spawner_cfg is not None else self.default_spawner_cfg,
        )
        intensity_cfg = LightIntensityVariationCfg(
            sampler_cfg=UniformSamplerCfg(
                low=[self.default_intensity * 0.25], high=[self.default_intensity * 1.5]
            )
        )
        self.add_variation(LightIntensityVariation(self, cfg=intensity_cfg))
        self.add_variation(LightColorVariation(self))
        self.add_variation(LightColorTemperatureVariation(self))
        from dexbench_arena.variations.light_position_variation import LightPositionVariation, LightPositionVariationCfg

        position_cfg = None
        if position_range_m is not None:
            low, high = (tuple(float(v) for v in bound) for bound in position_range_m)
            if len(low) != 3 or len(high) != 3 or any(a > b for a, b in zip(low, high)):
                raise ValueError(f"position_range_m must be ((dx, dy, dz) low, (dx, dy, dz) high), low <= high; got {position_range_m}")
            position_cfg = LightPositionVariationCfg(sampler_cfg=UniformSamplerCfg(low=list(low), high=list(high)))
        self.add_variation(LightPositionVariation(self, cfg=position_cfg))


@register_asset
class DexBenchTubeLight(DexBenchInteriorLightBase):
    """A cylinder light -- a fluorescent tube, for industrial and retail rooms."""

    name = "dexbench_tube_light"
    default_intensity = 60000.0
    default_spawner_cfg = sim_utils.CylinderLightCfg(
        color=(1.0, 1.0, 1.0), intensity=default_intensity, radius=0.04, length=1.2
    )

    spawner_cfg: sim_utils.CylinderLightCfg
    """Narrows the base-class spawner cfg type to ``CylinderLightCfg``."""


@configclass
class RectLightCfg(sim_utils.LightCfg):
    """A ``UsdLux.RectLight``, which Isaac Lab ships no cfg for. Emits along local -Z."""

    prim_type = "RectLight"

    width: float = 1.0
    """Width of the emitting rectangle [m]."""

    height: float = 1.0
    """Height of the emitting rectangle [m]."""

    diffuse: float = 1.0
    """Multiplier on the light's diffuse contribution (``inputs:diffuse``); 0 leaves only its reflections."""


@_clone_spawner
def _spawn_shaped_rect_light(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """A RectLight with ``UsdLux.ShapingAPI``'s cone, which Isaac Lab's light spawner has no fields for."""

    fields = cfg.to_dict()
    for name in ("func", "cone_angle", "cone_softness"):
        fields.pop(name)
    prim = sim_utils.spawn_light.__wrapped__(prim_path, RectLightCfg(**fields), translation, orientation, **kwargs)
    shaping = UsdLux.ShapingAPI.Apply(prim)
    shaping.CreateShapingConeAngleAttr(cfg.cone_angle)
    shaping.CreateShapingConeSoftnessAttr(cfg.cone_softness)
    return prim


@configclass
class ShapedRectLightCfg(RectLightCfg):
    """A ``RectLightCfg`` whose emission is narrowed to a cone (``UsdLux.ShapingAPI``)."""

    func: Callable = _spawn_shaped_rect_light

    cone_angle: float = 90.0
    """Half-angle of the emission cone [deg]; 90 is the unshaped hemisphere."""

    cone_softness: float = 0.0
    """How far inside the cone's edge the falloff starts, 0 (hard) to 1."""




@register_asset
class DexBenchHighBayLight(DexBenchInteriorLightBase):
    """The E2 cell's ceiling high-bay: a 0.1 m square RectLight, normalized, pointing down."""

    name = E2_HIGH_BAY_LIGHT_ASSET
    default_intensity = 6302.5356
    default_spawner_cfg = RectLightCfg(
        color=(1.0, 0.97, 0.92), intensity=default_intensity, normalize=True, width=0.1, height=0.1
    )

    spawner_cfg: RectLightCfg
    """Narrows the base-class spawner cfg type to ``RectLightCfg``."""


@register_asset
class DexBenchPendantLight(DexBenchInteriorLightBase):
    """The E2 cell's pendant over a workstation: a 0.1 m square RectLight in a 42 degree cone, pointing down."""

    name = E2_PENDANT_LIGHT_ASSET
    default_intensity = 26000.0
    default_spawner_cfg = ShapedRectLightCfg(
        color=(1.0, 0.95, 0.86), intensity=default_intensity, normalize=True, width=0.1, height=0.1,
        cone_angle=42.0, cone_softness=0.6,
    )

    spawner_cfg: ShapedRectLightCfg
    """Narrows the base-class spawner cfg type to ``ShapedRectLightCfg``."""


@register_asset
class DexBenchStripLight(DexBenchInteriorLightBase):
    """The E2 cell's strip under the packing bench's top shelf: a 1.0 x 0.04 m RectLight, normalized."""

    name = E2_STRIP_LIGHT_ASSET
    default_intensity = 1900.0
    default_spawner_cfg = RectLightCfg(
        color=(1.0, 0.95, 0.86), intensity=default_intensity, normalize=True, width=1.0, height=0.04
    )

    spawner_cfg: RectLightCfg
    """Narrows the base-class spawner cfg type to ``RectLightCfg``."""


@register_asset
class DexBenchReflectCard(DexBenchInteriorLightBase):
    """The E2 cell's reflector card: a white RectLight seen only in reflections (diffuse 0)."""

    name = E2_REFLECT_CARD_ASSET
    default_intensity = 200.0
    default_spawner_cfg = RectLightCfg(
        color=(1.0, 1.0, 1.0), intensity=default_intensity, normalize=False, width=1.1, height=1.0, diffuse=0.0
    )

    spawner_cfg: RectLightCfg
    """Narrows the base-class spawner cfg type to ``RectLightCfg``."""
