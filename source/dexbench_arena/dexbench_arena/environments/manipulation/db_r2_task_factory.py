# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""E1 factory for ``--task retrieve_beverage_from_clutter`` (DB-R2)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dexbench_arena.contact_materials import WUJI_GRASP_CONTACT_MATERIAL
from dexbench_arena.evaluation.evaluation_signature import NEWTON_SHAPE_OVERRIDES_ATTRIBUTE
from dexbench_arena.mimic_plugin import mimic_task_class
from dexbench_arena.layouts.db_r2_layout import (
    FC_260821_R2_BASKET_INSTANCE_NAME,
    FC_260821_R2_ANCHOR,
    FC_260821_R2_BASKET_CENTER_HEIGHT_RANGE_M,
    FC_260821_R2_BASKET_FLOOR_TOP_Z_M,
    FC_260821_R2_BASKET_INTERIOR_HALF_EXTENTS_XY_M,
    FC_260821_R2_BASKET_ORIGIN_XYZ_M,
    FC_260821_R2_DESTINATION_FRAME_INSTANCE_NAME,
    FC_260821_R2_EPISODE_LENGTH_S,
    FC_260821_R2_EXTRACTION_CLEARANCE_M,
    FC_260821_R2_FRIDGE_FRONT_PLANE_Y_M,
    FC_260821_R2_NEIGHBOR_DISPLACEMENT_COLLECTION_M,
    FC_260821_R2_NEIGHBOR_DISPLACEMENT_STRICT_M,
    FC_260821_R2_NEIGHBOR_KNOCK_LATERAL_M,
    FC_260821_R2_NEIGHBOR_KNOCK_LIFT_M,
    FC_260821_R2_NEIGHBOR_TOPPLE_MAX_UP_ALIGNMENT,
    FC_260821_R2_OBJECT_AXIS,
    FC_260821_R2_OBJECT_DROPPED_MIN_Z_M,
    FC_260821_R2_OPENING_FRAME_INSTANCE_NAME,
    FC_260821_R2_PROGRESS_LIFT_HEIGHT_GAIN_M,
    FC_260821_R2_PROGRESS_OVER_BASKET_XY_MARGIN_M,
    FC_260821_R2_SKU_ASSET_NAMES,
    FC_260821_R2_STRICT_HOLD_S_DEFAULT,
    FC_260821_R2_STRICT_HOLD_SHARE,
    FC_260821_R2_TARGET_ASSET_NAME,
    FC_260821_R2_TARGET_INSTANCE_NAME,
    FC_260821_R2_VELOCITY_THRESHOLD_M_S,
    FC_260821_R2_WORKSPACE_XY_BOUNDS,
    FC_260821_R2_WRONG_PICK_LIFT_M,
    fc_260821_r2_stocking_for,
    fc_260821_r2_task_description,
)
from dexbench_arena.environments.manipulation.newton_rolling_friction import (
    ROLLING_CONDIM,
    install_rolling_friction,
)
from dexbench_arena.environments.manipulation.newton_shape_contact import (
    ShapeContactRequest,
    install_shape_contact,
)
from dexbench_arena.environments.manipulation.newton_solref import solref_to_newton_ke_kd

if TYPE_CHECKING:
    from dexbench_arena.environments.manipulation.e1_retail_environment import DexBenchRetailStoreE1EnvironmentCfg

RETRIEVE_BEVERAGE_TASK_NAME = "retrieve_beverage_from_clutter"

R2_BASKET_SHAPE_PATTERN = rf"/{FC_260821_R2_BASKET_INSTANCE_NAME}/"
R2_TARGET_SHAPE_PATTERN = rf"/{FC_260821_R2_TARGET_INSTANCE_NAME}/"
R2_NEIGHBOR_SHAPE_PATTERN = r"/r2_neighbor_\d+/"
R2_BASKET_SOLREF = (0.005, 2.0)
R2_BASKET_SOLMIX = 1000.0
R2_DRINK_SOLREF = (0.01, 1.0)
R2_DRINK_SOLMIX = 100.0
R2_ROLLING_FRICTION_M = 1.0e-3
"""Rolling-resistance lever arm on the drinks and the basket [m]."""
_R2_SHAPE_CONTACTS = (
    ("r2_basket_contact", R2_BASKET_SHAPE_PATTERN, R2_BASKET_SOLREF, R2_BASKET_SOLMIX),
    ("r2_target_contact", R2_TARGET_SHAPE_PATTERN, R2_DRINK_SOLREF, R2_DRINK_SOLMIX),
    ("r2_neighbor_contact", R2_NEIGHBOR_SHAPE_PATTERN, R2_DRINK_SOLREF, R2_DRINK_SOLMIX),
)
_R2_ROLLING_FRICTION = (
    ("r2_target_roll", R2_TARGET_SHAPE_PATTERN),
    ("r2_neighbor_roll", R2_NEIGHBOR_SHAPE_PATTERN),
    ("r2_basket_roll", R2_BASKET_SHAPE_PATTERN),
)


def apply_r2_newton_contact(env_cfg, task: str) -> None:
    """Install DB-R2's basket and drink contact for a ``task`` build; any other E1 task installs none."""

    if task != RETRIEVE_BEVERAGE_TASK_NAME:
        if getattr(env_cfg, NEWTON_SHAPE_OVERRIDES_ATTRIBUTE, None) is not None:
            delattr(env_cfg, NEWTON_SHAPE_OVERRIDES_ATTRIBUTE)
        return
    recorded = {"shape_contact": {}, "rolling_friction": {}}
    for name, pattern, solref, solmix in _R2_SHAPE_CONTACTS:
        ke, kd = solref_to_newton_ke_kd(*solref)
        install_shape_contact(ShapeContactRequest(name, pattern, ke, kd, solmix))
        recorded["shape_contact"][name] = {"pattern": pattern, "solref": list(solref), "ke": ke, "kd": kd,
                                           "solmix": solmix}
    for name, pattern in _R2_ROLLING_FRICTION:
        install_rolling_friction(name, pattern, R2_ROLLING_FRICTION_M)
        recorded["rolling_friction"][name] = {"pattern": pattern, "mu_rolling": R2_ROLLING_FRICTION_M,
                                              "condim": ROLLING_CONDIM}
    setattr(env_cfg, NEWTON_SHAPE_OVERRIDES_ATTRIBUTE, recorded)


def target_center_offset(asset_registry, asset_name: str = FC_260821_R2_TARGET_ASSET_NAME):
    """Local bbox center of the target SKU from the asset registry."""
    probe = asset_registry.get_asset_by_name(asset_name)(instance_name="_fc_r2_target_probe")
    return tuple(float(v) for v in probe.get_bounding_box().center[0].tolist())


def make_retrieve_beverage_from_clutter_task(environment, cfg: DexBenchRetailStoreE1EnvironmentCfg):
    """Build the DB-R2 task from the parsed CLI: ``cfg.r2_target_sku`` in ``cfg.r2_target_lane``, its neighbors from the lane."""
    from dexbench_arena.assets import object_library as _object_library  # noqa: F401
    from dexbench_arena.tasks.retrieve_from_clutter import DexBenchRetrieveFromClutterTask

    if cfg.scene_layout != "fc_260821":
        raise ValueError(
            f"--task {RETRIEVE_BEVERAGE_TASK_NAME} requires --scene_layout fc_260821 "
            "(the layout with the refrigerator station and the R2 basket)."
        )
    if cfg.with_interaction_items:
        raise ValueError(
            f"--task {RETRIEVE_BEVERAGE_TASK_NAME} spawns its own target/neighbor set; "
            "drop --with_interaction_items (the 32-drink feasibility set would duplicate them)."
        )
    anchor = cfg.anchor
    if anchor and anchor != FC_260821_R2_ANCHOR:
        print(
            f"[dexbench] WARNING: --task {RETRIEVE_BEVERAGE_TASK_NAME} is reviewed at "
            f"--anchor {FC_260821_R2_ANCHOR}; got {anchor!r}."
        )

    sku = cfg.r2_target_sku
    object_center_offset_xyz = target_center_offset(environment.asset_registry, FC_260821_R2_SKU_ASSET_NAMES[sku])
    basket_center_halfsize_xy = FC_260821_R2_BASKET_INTERIOR_HALF_EXTENTS_XY_M
    basket_center_xy = (FC_260821_R2_BASKET_ORIGIN_XYZ_M[0], FC_260821_R2_BASKET_ORIGIN_XYZ_M[1])
    stocking = fc_260821_r2_stocking_for(cfg)
    neighbor_names = list(stocking.neighbor_instance_names)

    success_override = None
    episode_length_s = FC_260821_R2_EPISODE_LENGTH_S
    max_neighbor_displacement_m = FC_260821_R2_NEIGHBOR_DISPLACEMENT_COLLECTION_M
    if cfg.success_tier == "strict":
        from isaaclab.managers import SceneEntityCfg, TerminationTermCfg

        from dexbench_arena.tasks.predicates.e1_retail import db_r2_target_placed

        hold_s = cfg.strict_hold_s
        if hold_s is None:
            hold_s = FC_260821_R2_STRICT_HOLD_S_DEFAULT
        hold_s = float(hold_s)
        max_neighbor_displacement_m = FC_260821_R2_NEIGHBOR_DISPLACEMENT_STRICT_M
        success_override = TerminationTermCfg(
            func=db_r2_target_placed,
            params={
                "object_cfg": SceneEntityCfg(FC_260821_R2_TARGET_INSTANCE_NAME),
                "object_center_offset_xyz": object_center_offset_xyz,
                "basket_center_xy": basket_center_xy,
                "basket_center_halfsize_xy": basket_center_halfsize_xy,
                "basket_floor_z_m": FC_260821_R2_BASKET_FLOOR_TOP_Z_M,
                "center_height_range_m": FC_260821_R2_BASKET_CENTER_HEIGHT_RANGE_M,
                "neighbor_names": neighbor_names,
                "max_neighbor_displacement_m": None,
                "neighbor_knock": {
                    "max_up_alignment": FC_260821_R2_NEIGHBOR_TOPPLE_MAX_UP_ALIGNMENT,
                    "min_lift_m": FC_260821_R2_NEIGHBOR_KNOCK_LIFT_M,
                    "plane_y_m": FC_260821_R2_FRIDGE_FRONT_PLANE_Y_M,
                    "max_lateral_m": FC_260821_R2_NEIGHBOR_KNOCK_LATERAL_M,
                },
                "velocity_threshold_m_s": FC_260821_R2_VELOCITY_THRESHOLD_M_S,
                "hold_time_s": hold_s,
                "hold_share": FC_260821_R2_STRICT_HOLD_SHARE,
            },
        )
        episode_length_s = FC_260821_R2_EPISODE_LENGTH_S + hold_s + 5.0

    return mimic_task_class(DexBenchRetrieveFromClutterTask)(
        hand_contact_material=WUJI_GRASP_CONTACT_MATERIAL,
        object_name=FC_260821_R2_TARGET_INSTANCE_NAME,
        object_center_offset_xyz=object_center_offset_xyz,
        opening_plane_y_m=FC_260821_R2_FRIDGE_FRONT_PLANE_Y_M,
        extraction_clearance_m=FC_260821_R2_EXTRACTION_CLEARANCE_M,
        basket_center_xy=basket_center_xy,
        basket_center_halfsize_xy=basket_center_halfsize_xy,
        basket_floor_z_m=FC_260821_R2_BASKET_FLOOR_TOP_Z_M,
        center_height_range_m=FC_260821_R2_BASKET_CENTER_HEIGHT_RANGE_M,
        neighbor_names=neighbor_names,
        max_neighbor_displacement_m=max_neighbor_displacement_m,
        workspace_xy_bounds=FC_260821_R2_WORKSPACE_XY_BOUNDS,
        object_dropped_min_z_m=FC_260821_R2_OBJECT_DROPPED_MIN_Z_M,
        wrong_pick_lift_m=FC_260821_R2_WRONG_PICK_LIFT_M,
        neighbor_topple_max_up_alignment=FC_260821_R2_NEIGHBOR_TOPPLE_MAX_UP_ALIGNMENT,
        velocity_threshold_m_s=FC_260821_R2_VELOCITY_THRESHOLD_M_S,
        progress_lift_height_gain_m=FC_260821_R2_PROGRESS_LIFT_HEIGHT_GAIN_M,
        progress_over_basket_xy_margin_m=FC_260821_R2_PROGRESS_OVER_BASKET_XY_MARGIN_M,
        episode_length_s=episode_length_s,
        task_description=fc_260821_r2_task_description(
            FC_260821_R2_OBJECT_AXIS.get(sku).label, stocking.lanes[0] if len(stocking.lanes) == 1 else None
        ),
        robot_joint_pos_overrides=environment._rby1_manipulation_gaze(cfg),
        success_override=success_override,
        opening_ref_name=FC_260821_R2_OPENING_FRAME_INSTANCE_NAME,
        destination_ref_name=FC_260821_R2_DESTINATION_FRAME_INSTANCE_NAME,
    )
