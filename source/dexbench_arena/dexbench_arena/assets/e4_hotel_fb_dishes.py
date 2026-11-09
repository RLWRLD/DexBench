# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""The DB-H1 dishes: the NdotLight tableware set the E4 hall's plate task stacks."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.sim.utils import clone as _clone_spawner
from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.assets.object_library import LibraryObject
from isaaclab_arena.assets.register import register_asset

from dexbench_arena.assets.e4_hotel_fb_dish_specs import (
    DISH_COLLIDER_PARAMS,
    DISH_CONTACT_OFFSET_M,
    DISH_VARIANTS,
)
from dexbench_arena.assets.physics_variants import NEWTON_VARIANT
from dexbench_arena.assets.usd_file_spawn import spawn_usd_file
from dexbench_arena.tasks.task_termination import composed_success_term

__all__ = [
    "DISH_CONTACT_OFFSET_M",
    "DISH_STACK_RIGID_PROPS",
    "DISH_COLLIDER_PARAMS",
    "DexBenchHotelFbE4GodmiddagDeepPlate23",
    "DexBenchHotelFbE4HarmyntaDeepPlate22",
    "DexBenchHotelFbE4GodmiddagBowl16",
    "DexBenchHotelFbE4Ikea365Bowl16",
    "DexBenchHotelFbE4Ikea365Bowl19",
    "DexBenchHotelFbE4Ikea365Bowl22",
]


DISH_STACK_RIGID_PROPS = sim_utils.RigidBodyPropertiesCfg(
    max_depenetration_velocity=1.0,
    solver_position_iteration_count=32,
    solver_velocity_iteration_count=4,
)


def configure_h1_newton_physics(env_cfg) -> None:
    """Apply the validated H1 contact settings on every environment build path."""
    physics = getattr(env_cfg.sim, "physics", None)
    solver = getattr(physics, "solver_cfg", None)
    if getattr(solver, "solver_type", None) != "mujoco_warp":
        return
    solver.nconmax = max(solver.nconmax or 0, 16384)
    solver.njmax = max(solver.njmax or 0, 65536)
    solver.iterations = 100
    physics.default_shape_cfg.ke = 10000.0
    physics.default_shape_cfg.kd = 200.0
    physics.num_substeps = max(physics.num_substeps, 8)
    from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialBaseCfg

    for name in composed_success_term(env_cfg).params["plate_names"]:
        getattr(env_cfg.scene, name).spawn.physics_material = RigidBodyMaterialBaseCfg(
            static_friction=0.6, dynamic_friction=0.6
        )
    print(
        f"[dexbench] H1 Newton: substeps={physics.num_substeps}, dish friction=0.6, "
        f"contacts={solver.nconmax}, constraints={solver.njmax}, "
        f"MuJoCo collision detection={solver.use_mujoco_contacts}, "
        f"iterations={solver.iterations}, CCD iterations={solver.ccd_iterations}, "
        f"multiccd={solver.enable_multiccd}, "
        f"contact stiffness/damping={physics.default_shape_cfg.ke}/{physics.default_shape_cfg.kd}"
    )


@_clone_spawner
def _spawn_dish_with_nesting_collider(
    prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn the dish USD with a collider that nests."""

    prim = spawn_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation, **kwargs)
    variants = getattr(cfg, "variants", None) or {}
    if variants.get(NEWTON_VARIANT) == "Enabled":
        return prim
    from isaaclab.sim.schemas import define_mesh_collision_properties
    from isaaclab_physx.sim.schemas import schemas_cfg as physx_cfg
    from pxr import Usd, UsdGeom, UsdPhysics

    mesh_cfg = physx_cfg.ConvexDecompositionPropertiesCfg(
        mesh_approximation_name="convexDecomposition", **DISH_COLLIDER_PARAMS
    )
    for child_prim in Usd.PrimRange(prim):
        if not child_prim.IsA(UsdGeom.Mesh) or not child_prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        define_mesh_collision_properties(str(child_prim.GetPath()), mesh_cfg)
        sim_utils.modify_collision_properties(
            str(child_prim.GetPath()),
            sim_utils.CollisionPropertiesCfg(contact_offset=DISH_CONTACT_OFFSET_M, rest_offset=0.0),
        )
    return prim


class _HotelFbE4Dish(LibraryObject):
    """A DB-H1 dish: rigid, authored mass, nesting collider from its physics runtime variant."""

    tags = ["object", "hotel_fb_e4", "tableware", "db-h1"]
    object_type = ObjectType.RIGID
    scale = (1.0, 1.0, 1.0)
    spawn_cfg_addon = {
        "func": _spawn_dish_with_nesting_collider,
        "rigid_props": DISH_STACK_RIGID_PROPS,
    }


def _usd(variant: str) -> str:
    return str(DISH_VARIANTS[variant].usd_path)


@register_asset
class DexBenchHotelFbE4GodmiddagDeepPlate23(_HotelFbE4Dish):
    name = "dexbench_hotel_fb_e4_godmiddag_deep_plate_23"
    usd_path = _usd("godmiddag_deep_plate_23")


@register_asset
class DexBenchHotelFbE4HarmyntaDeepPlate22(_HotelFbE4Dish):
    name = "dexbench_hotel_fb_e4_harmynta_deep_plate_22"
    usd_path = _usd("harmynta_deep_plate_22")


@register_asset
class DexBenchHotelFbE4GodmiddagBowl16(_HotelFbE4Dish):
    name = "dexbench_hotel_fb_e4_godmiddag_bowl_16"
    usd_path = _usd("godmiddag_bowl_16")


@register_asset
class DexBenchHotelFbE4Ikea365Bowl16(_HotelFbE4Dish):
    name = "dexbench_hotel_fb_e4_ikea_365_bowl_16"
    usd_path = _usd("ikea_365_bowl_16")


@register_asset
class DexBenchHotelFbE4Ikea365Bowl19(_HotelFbE4Dish):
    name = "dexbench_hotel_fb_e4_ikea_365_bowl_19"
    usd_path = _usd("ikea_365_bowl_19")


@register_asset
class DexBenchHotelFbE4Ikea365Bowl22(_HotelFbE4Dish):
    name = "dexbench_hotel_fb_e4_ikea_365_bowl_22"
    usd_path = _usd("ikea_365_bowl_22")
