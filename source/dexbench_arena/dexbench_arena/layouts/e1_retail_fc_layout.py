"""Arena-native fixture layout for the reviewed 260821 E1 retail scene."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path

from dexbench_arena.embodiments.rby1m.constants import RBY1M_BASE_MODEL
from dexbench_arena.tasks.object_axis import ObjectAxis, TaskObject
from dexbench_arena.assets.retail_sku_packages import RETAIL_E1_FC_SKU_PACKAGES
from dexbench_arena.layouts.e1_retail_fc_produce_clutter import (
    PRODUCE_CLUTTER,
    PRODUCE_CLUTTER_SET_IDS,
    PRODUCE_CLUTTER_TARGET_SKUS,
    ProducePlacement,
)
from dexbench_arena.layouts.e1_retail_fc_stock_layout import (
    FC_260821_STOCK_LAYOUT,
    FC_260821_STOCK_SKUS,
    RetailStockPlacement,
)


@dataclass(frozen=True)
class RetailFixturePlacement:
    """One fixed fixture instance in the reviewed FC layout."""

    asset_name: str
    instance_name: str
    position_xyz: tuple[float, float, float]
    yaw_deg: float = 0.0
    scale_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0)
    rotation_override_xyzw: tuple[float, float, float, float] | None = None

    @property
    def rotation_xyzw(self) -> tuple[float, float, float, float]:
        if self.rotation_override_xyzw is not None:
            return self.rotation_override_xyzw
        from isaaclab_arena.utils.yaw import rotate_quat_by_yaw

        return rotate_quat_by_yaw((0.0, 0.0, 0.0, 1.0), math.radians(self.yaw_deg))


@dataclass(frozen=True)
class RetailR3VariationProfile:
    """One axis-isolated FC DB-R3 source/base variation profile."""

    source_xy_m: float = 0.0
    source_yaw_deg: float = 0.0
    base_xy_m: float = 0.0


FC_260821_R3_SUPPLY_CENTER_XY_M = (1.43, 1.28)
FC_260821_CART_CRADLE_Z_M = 0.60
FC_260821_R3_TOTE_COUNT = 6
FC_260821_R3_TOTE_SOURCE_HEIGHT_M = 0.21
FC_260821_R3_TOTE_HEIGHT_M = 0.18
FC_260821_R3_TOTE_SCALE_XYZ = (
    1.0,
    1.0,
    FC_260821_R3_TOTE_HEIGHT_M / FC_260821_R3_TOTE_SOURCE_HEIGHT_M,
)
FC_260821_R3_TOTE_PLACEMENTS = tuple(
    RetailFixturePlacement(
        "dexbench_retail_e1_fc_tote",
        f"retail_e1_fc_tote_{index + 1:02d}",
        (*FC_260821_R3_SUPPLY_CENTER_XY_M, index * FC_260821_R3_TOTE_HEIGHT_M),
        scale_xyz=FC_260821_R3_TOTE_SCALE_XYZ,
    )
    for index in range(FC_260821_R3_TOTE_COUNT)
)


FC_260821_FIXTURE_LAYOUT: tuple[RetailFixturePlacement, ...] = (
    RetailFixturePlacement(
        "dexbench_retail_e1_fc_ceiling_lights",
        "retail_e1_fc_ceiling_lights",
        (0.0, 0.0, 0.0),
    ),
    RetailFixturePlacement(
        "dexbench_retail_e1_fc_gondola",
        "retail_e1_fc_gondola",
        (0.785, 0.70, 0.0),
        90.0,
    ),
    RetailFixturePlacement(
        "dexbench_retail_e1_fc_produce_display",
        "retail_e1_fc_produce_display",
        (3.32, 0.70, 0.0),
        -90.0,
    ),
    RetailFixturePlacement(
        "dexbench_retail_e1_fc_fridge_open_180",
        "retail_e1_fc_fridge_01",
        (2.63, 2.705, 0.0),
    ),
    RetailFixturePlacement(
        "dexbench_retail_e1_fc_fridge_closed",
        "retail_e1_fc_fridge_02",
        (1.88, 2.705, 0.0),
    ),
    *FC_260821_R3_TOTE_PLACEMENTS,
    RetailFixturePlacement(
        "dexbench_retail_e1_fc_shopping_cart",
        "retail_e1_fc_cart_r1",
        (2.60, 0.25, 0.0),
        0.0,
    ),
    RetailFixturePlacement(
        "dexbench_retail_e1_fc_shopping_basket",
        "retail_e1_fc_fruit_basket",
        (2.60, 0.25, FC_260821_CART_CRADLE_Z_M),
        0.0,
    ),
    RetailFixturePlacement(
        "dexbench_retail_e1_fc_shopping_cart",
        "retail_e1_fc_cart_r2",
        (3.20, 1.90, 0.0),
        90.0,
    ),
    RetailFixturePlacement(
        "dexbench_retail_e1_fc_shopping_basket",
        "retail_e1_fc_r2_basket",
        (3.20, 1.90, FC_260821_CART_CRADLE_Z_M),
        90.0,
    ),
)


FC_260821_FIXTURE_INSTANCE_NAMES = tuple(placement.instance_name for placement in FC_260821_FIXTURE_LAYOUT)


FC_260821_R3_SHELF_TOP_Z_M = 1.0
FC_260821_R3_TARGET_ASSET_NAME = "dexbench_retail_e1_fc_r3t2"
FC_260821_R3_TARGET_SKU_HALF_EXTENTS_XY_M = {
    "dexbench_retail_e1_fc_r3t1": (0.0115, 0.0439),
    "dexbench_retail_e1_fc_r3t2": (0.0250, 0.0700),
    "dexbench_retail_e1_fc_r3t3": (0.0189, 0.0875),
    "dexbench_retail_e1_fc_r3t4": (0.0305, 0.1225),
    "dexbench_retail_e1_fc_r3e1": (0.0173, 0.0430),
    "dexbench_retail_e1_fc_r3e2": (0.0173, 0.1055),
    "dexbench_retail_e1_fc_r3e4": (0.0305, 0.1210),
    "dexbench_retail_e1_fc_r3e5": (0.0190, 0.1325),
    "dexbench_retail_e1_fc_r3e6": (0.0318, 0.1160),
}
_SQRT_HALF = math.sqrt(0.5)
FC_260821_R3_TARGET_SKU_ROTATION_XYZW = {
    "dexbench_retail_e1_fc_r3t1": (0.0, 0.0, _SQRT_HALF, _SQRT_HALF),
    "dexbench_retail_e1_fc_r3t2": (0.0, 0.0, _SQRT_HALF, _SQRT_HALF),
    "dexbench_retail_e1_fc_r3t3": (0.5, 0.5, 0.5, 0.5),
    "dexbench_retail_e1_fc_r3t4": (0.5, 0.5, 0.5, 0.5),
    "dexbench_retail_e1_fc_r3e2": (0.5, 0.5, 0.5, 0.5),
    "dexbench_retail_e1_fc_r3e4": (0.5, 0.5, 0.5, 0.5),
    "dexbench_retail_e1_fc_r3e5": (0.5, 0.5, 0.5, 0.5),
    "dexbench_retail_e1_fc_r3e6": (0.5, 0.5, 0.5, 0.5),
    "dexbench_retail_e1_fc_r3e1": (-_SQRT_HALF, 0.0, _SQRT_HALF, 0.0),
}
FC_260821_R3_TARGET_SKU_LOCAL_UP_XYZ = {
    "dexbench_retail_e1_fc_r3t1": (0.0, 0.0, 1.0),
    "dexbench_retail_e1_fc_r3t2": (0.0, 0.0, 1.0),
    "dexbench_retail_e1_fc_r3t3": (0.0, 1.0, 0.0),
    "dexbench_retail_e1_fc_r3t4": (0.0, 1.0, 0.0),
    "dexbench_retail_e1_fc_r3e1": (-1.0, 0.0, 0.0),
    "dexbench_retail_e1_fc_r3e2": (0.0, 1.0, 0.0),
    "dexbench_retail_e1_fc_r3e4": (0.0, 1.0, 0.0),
    "dexbench_retail_e1_fc_r3e5": (0.0, 1.0, 0.0),
    "dexbench_retail_e1_fc_r3e6": (0.0, 1.0, 0.0),
}
FC_260821_R3_TARGET_SKU_HALF_HEIGHT_M = {
    "dexbench_retail_e1_fc_r3t1": 0.0815,
    "dexbench_retail_e1_fc_r3t2": 0.0250,
    "dexbench_retail_e1_fc_r3t3": 0.0680,
    "dexbench_retail_e1_fc_r3t4": 0.1050,
    "dexbench_retail_e1_fc_r3e1": 0.0633,
    "dexbench_retail_e1_fc_r3e2": 0.1005,
    "dexbench_retail_e1_fc_r3e4": 0.0515,
    "dexbench_retail_e1_fc_r3e5": 0.0950,
    "dexbench_retail_e1_fc_r3e6": 0.1030,
}
FC_260821_R3_TARGET_SKU_ORIGIN_ABOVE_BOTTOM_M = {
    "dexbench_retail_e1_fc_r3t1": 0.0,
    "dexbench_retail_e1_fc_r3t2": 0.0,
    "dexbench_retail_e1_fc_r3t3": 0.0680,
    "dexbench_retail_e1_fc_r3t4": 0.1050,
    "dexbench_retail_e1_fc_r3e1": 0.0633,
    "dexbench_retail_e1_fc_r3e2": 0.1005,
    "dexbench_retail_e1_fc_r3e4": 0.0515,
    "dexbench_retail_e1_fc_r3e5": 0.0950,
    "dexbench_retail_e1_fc_r3e6": 0.1030,
}
FC_260821_R3_TRAIN_TARGET_ASSET_NAMES = (
    "dexbench_retail_e1_fc_r3t1",
    "dexbench_retail_e1_fc_r3t2",
    "dexbench_retail_e1_fc_r3t3",
    "dexbench_retail_e1_fc_r3t4",
)
FC_260821_R3_EVAL_TARGET_ASSET_NAMES = (
    "dexbench_retail_e1_fc_r3e1",
    "dexbench_retail_e1_fc_r3e2",
    "dexbench_retail_e1_fc_r3e4",
    "dexbench_retail_e1_fc_r3e5",
    "dexbench_retail_e1_fc_r3e6",
)
FC_260821_R3_TARGET_ASSET_NAMES = FC_260821_R3_TRAIN_TARGET_ASSET_NAMES + FC_260821_R3_EVAL_TARGET_ASSET_NAMES
@dataclass(frozen=True)
class RetailE1LayoutSpec:
    """What an E1 scene layout owns besides its USD: the station frames the CLI anchors name."""

    robot_base_anchors_xyz: dict[str, tuple[float, float, float]]
    """Where each CLI anchor puts the robot base, metres, Z-up, room origin."""
    anchor_facing_targets_xy: dict[str, tuple[float, float]]
    """The point the base faces from each anchor (embodiments face +X at identity)."""
    anchor_overrides_by_base: dict[str, dict[str, tuple[float, float, float]]]
    """Per base model (an embodiment's ``base_model``), the base positions it uses instead of the anchors above,
    where the nominal frame leaves it short of reach; a base model not listed stands at the anchors."""


FC_260821_LAYOUT_SPEC = RetailE1LayoutSpec(
    robot_base_anchors_xyz={
        "Fridge": (2.63, 1.86, 0.0),
        "Fruit": (2.47, 0.80, 0.0),
        "Gondola": (1.63, 0.70, 0.0),
    },
    anchor_facing_targets_xy={
        "Fridge": (2.63, 2.86),
        "Fruit": (3.47, 0.80),
        "Gondola": (0.63, 0.70),
    },
    anchor_overrides_by_base={
        RBY1M_BASE_MODEL: {
            "Gondola": (1.53, 0.70, 0.0),
            "Fruit": (2.55, 0.80, 0.0),
            "Fridge": (2.53, 1.90, 0.0),
        },
    },
)


FC_260821_R3_TARGET_INSTANCE_NAME = "target_carton"
FC_260821_R3_SLOT_FRAME_ASSET_NAME = "dexbench_task_frame"
FC_260821_R3_SLOT_FRAME_INSTANCE_NAME = "fc_r3_slot_frame"
FC_260821_R3_SLOT_CENTER_XY_M = (0.95550001, 0.7365)
FC_260821_R3_SLOT_OUTWARD_XY = (1.0, 0.0)
FC_260821_R3_DEFAULT_FACING_WIDTH_M = 0.180
FC_260821_R3_RELEASE_DEPTH_WINDOW_M = 0.020
FC_260821_R3_TARGET_HALF_EXTENTS_XY_M = (0.025, 0.070)
FC_260821_R3_TOTE_TOP_ORIGIN_Z_M = (FC_260821_R3_TOTE_COUNT - 1) * FC_260821_R3_TOTE_HEIGHT_M
FC_260821_R3_TOTE_RIM_Z_M = FC_260821_R3_TOTE_COUNT * FC_260821_R3_TOTE_HEIGHT_M
FC_260821_R3_TOTE_FLOOR_TOP_Z_M = FC_260821_R3_TOTE_TOP_ORIGIN_Z_M + (
    0.005 * FC_260821_R3_TOTE_SCALE_XYZ[2]
)
FC_260821_R3_TARGET_BOTTOM_Z_M = FC_260821_R3_TOTE_FLOOR_TOP_Z_M + 0.001
FC_260821_R3_TOTE_INTERIOR_HALF_EXTENTS_XY_M = (0.275, 0.175)

FC_260821_R3_VARIATION_PROFILES: dict[str, RetailR3VariationProfile] = {
    "collection_p0": RetailR3VariationProfile(),
    "train_xy_p1": RetailR3VariationProfile(source_xy_m=0.02),
    "train_yaw_p2": RetailR3VariationProfile(source_yaw_deg=5.0),
    "eval_xy_ood": RetailR3VariationProfile(source_xy_m=0.05),
    "eval_yaw_ood": RetailR3VariationProfile(source_yaw_deg=15.0),
    "eval_base_ood": RetailR3VariationProfile(base_xy_m=0.05),
}

_FC_260821_R3_E2_STANDING_ROTATION_XYZW = (0.5, 0.5, 0.5, 0.5)
_FC_260821_R3_E2_ORIGIN_ABOVE_BOTTOM_M = 0.1005
_FC_260821_R3_SOURCE_NEIGHBOR_LAYOUT: tuple[RetailFixturePlacement, ...] = (
    RetailFixturePlacement(
        "dexbench_retail_e1_fc_r3e2",
        "fc_r3_neighbor_r3e2_0_0",
        (0.95200002, 0.932, FC_260821_R3_SHELF_TOP_Z_M + _FC_260821_R3_E2_ORIGIN_ABOVE_BOTTOM_M),
        90.0,
        rotation_override_xyzw=_FC_260821_R3_E2_STANDING_ROTATION_XYZW,
    ),
    RetailFixturePlacement(
        "dexbench_retail_e1_fc_r3e2",
        "fc_r3_neighbor_r3e2_0_1",
        (0.88999999, 0.932, FC_260821_R3_SHELF_TOP_Z_M + _FC_260821_R3_E2_ORIGIN_ABOVE_BOTTOM_M),
        90.0,
        rotation_override_xyzw=_FC_260821_R3_E2_STANDING_ROTATION_XYZW,
    ),
    RetailFixturePlacement(
        "dexbench_retail_e1_fc_r3t2",
        "fc_r3_neighbor_r3t2_0_0",
        (0.95550001, 0.4325, FC_260821_R3_SHELF_TOP_Z_M),
        90.0,
    ),
    RetailFixturePlacement(
        "dexbench_retail_e1_fc_r3t2",
        "fc_r3_neighbor_r3t2_0_1",
        (0.9005, 0.4325, FC_260821_R3_SHELF_TOP_Z_M),
        90.0,
    ),
    RetailFixturePlacement(
        "dexbench_retail_e1_fc_r3t2",
        "fc_r3_neighbor_r3t2_1_0",
        (0.95550001, 0.5765, FC_260821_R3_SHELF_TOP_Z_M),
        90.0,
    ),
    RetailFixturePlacement(
        "dexbench_retail_e1_fc_r3t2",
        "fc_r3_neighbor_r3t2_1_1",
        (0.9005, 0.5765, FC_260821_R3_SHELF_TOP_Z_M),
        90.0,
    ),
)


FC_260821_R3_NEIGHBOR_START_OFFSET_XYZ_M = {
    "dexbench_retail_e1_fc_r3t2": (-0.015, 0.0, 0.0005),
    "dexbench_retail_e1_fc_r3e2": (-0.015, 0.0, 0.0),
}


def compute_fc_260821_r3_layout(
    facing_width_m: float = FC_260821_R3_DEFAULT_FACING_WIDTH_M,
    *,
    target_asset_name: str = FC_260821_R3_TARGET_ASSET_NAME,
) -> dict:
    """Repack the neighbor banks and task geometry for one target SKU."""

    facing_width_m = float(facing_width_m)
    if target_asset_name not in FC_260821_R3_TARGET_ASSET_NAMES:
        raise ValueError(
            f"Unknown FC DB-R3 target {target_asset_name!r}; choose one of "
            f"{FC_260821_R3_TARGET_ASSET_NAMES}"
        )
    target_half_extents_xy = FC_260821_R3_TARGET_SKU_HALF_EXTENTS_XY_M[target_asset_name]
    target_width_m = 2.0 * target_half_extents_xy[1]
    if not math.isfinite(facing_width_m) or facing_width_m < target_width_m:
        raise ValueError(
            "FC DB-R3 facing width must be finite and at least the target width "
            f"({target_width_m * 1000.0:.1f} mm), got {facing_width_m * 1000.0:.1f} mm"
        )

    half_delta = 0.5 * (facing_width_m - FC_260821_R3_DEFAULT_FACING_WIDTH_M)
    neighbor_layout = []
    for placement in _FC_260821_R3_SOURCE_NEIGHBOR_LAYOUT:
        dx, dy, dz = FC_260821_R3_NEIGHBOR_START_OFFSET_XYZ_M[placement.asset_name]
        x, y, z = (a + b for a, b in zip(placement.position_xyz, (dx, dy, dz)))
        direction = -1.0 if y < FC_260821_R3_SLOT_CENTER_XY_M[1] else 1.0
        neighbor_layout.append(
            RetailFixturePlacement(
                placement.asset_name,
                placement.instance_name,
                (x, y + direction * half_delta, z),
                placement.yaw_deg,
                placement.scale_xyz,
                placement.rotation_override_xyzw,
            )
        )

    return {
        "facing_width_m": facing_width_m,
        "target_asset_name": target_asset_name,
        "target_half_extents_xy": target_half_extents_xy,
        "target_half_height_m": FC_260821_R3_TARGET_SKU_HALF_HEIGHT_M[target_asset_name],
        "target_local_up_xyz": FC_260821_R3_TARGET_SKU_LOCAL_UP_XYZ[target_asset_name],
        "target_rotation_xyzw": FC_260821_R3_TARGET_SKU_ROTATION_XYZW[target_asset_name],
        "target_origin_above_bottom_m": FC_260821_R3_TARGET_SKU_ORIGIN_ABOVE_BOTTOM_M[
            target_asset_name
        ],
        "neighbor_layout": tuple(neighbor_layout),
        "slot_center_xy": FC_260821_R3_SLOT_CENTER_XY_M,
        "slot_halfsize_xy": (target_half_extents_xy[0] + FC_260821_R3_RELEASE_DEPTH_WINDOW_M, 0.5 * facing_width_m),
        "outward_xy": FC_260821_R3_SLOT_OUTWARD_XY,
        "front_plane_xy": tuple(
            center + target_half_extents_xy[0] * outward
            for center, outward in zip(FC_260821_R3_SLOT_CENTER_XY_M, FC_260821_R3_SLOT_OUTWARD_XY)
        ),
    }


FC_260821_R3_FACING_CLEARANCE_M = 0.040


def fc_260821_r3_facing_width_m(target_asset_name: str, facing_width_mm: float | None) -> float:
    """The empty facing [m]: ``facing_width_mm`` when given (a sensitivity run), else the carton's width + 40 mm."""
    if facing_width_mm is not None:
        return float(facing_width_mm) / 1000.0
    if target_asset_name not in FC_260821_R3_TARGET_SKU_HALF_EXTENTS_XY_M:
        raise ValueError(f"Unknown FC DB-R3 target {target_asset_name!r}; choose one of {FC_260821_R3_TARGET_ASSET_NAMES}")
    return 2.0 * FC_260821_R3_TARGET_SKU_HALF_EXTENTS_XY_M[target_asset_name][1] + FC_260821_R3_FACING_CLEARANCE_M


def fc_260821_r3_fit(task_object: TaskObject, cfg) -> dict:
    """What DB-R3 derives from its carton: the empty facing it is placed in (refused when narrower than the carton)."""
    facing_width_m = fc_260821_r3_facing_width_m(task_object.key, getattr(cfg, "facing_width_mm", None))
    layout = compute_fc_260821_r3_layout(facing_width_m, target_asset_name=task_object.key)
    return {"facing_width_m": round(layout["facing_width_m"], 6)}


FC_260821_R3_TARGET_LABELS: dict[str, str] = {
    "dexbench_retail_e1_fc_r3t1": "red chocolate-stick box",
    "dexbench_retail_e1_fc_r3t2": "purple cracker pack",
    "dexbench_retail_e1_fc_r3t3": "red wafer box",
    "dexbench_retail_e1_fc_r3t4": "red chocolate-pie box",
    "dexbench_retail_e1_fc_r3e1": "yellow snack box",
    "dexbench_retail_e1_fc_r3e2": "yellow cookie box",
    "dexbench_retail_e1_fc_r3e4": "red berry-cake box",
    "dexbench_retail_e1_fc_r3e5": "red wafer box",
    "dexbench_retail_e1_fc_r3e6": "green custard-cake box",
}


def _fc_260821_r3_object(asset_name: str, split: str) -> TaskObject:
    half_depth, half_width = FC_260821_R3_TARGET_SKU_HALF_EXTENTS_XY_M[asset_name]
    return TaskObject(
        key=asset_name,
        asset=asset_name,
        split=split,
        label=FC_260821_R3_TARGET_LABELS[asset_name],
        size_m=(2.0 * half_width, 2.0 * half_depth, 2.0 * FC_260821_R3_TARGET_SKU_HALF_HEIGHT_M[asset_name]),
        benchmark_asset_major_renames=((1, 2,
            f"assets/object_library/{RETAIL_E1_FC_SKU_PACKAGES[asset_name.removeprefix('dexbench_retail_e1_fc_').upper()]}/CHANGELOG.md"),),
    )


FC_260821_R3_OBJECT_AXIS = ObjectAxis(
    field="r3_target_asset",
    default=FC_260821_R3_TARGET_ASSET_NAME,
    objects=(
        *(_fc_260821_r3_object(name, "train") for name in FC_260821_R3_TRAIN_TARGET_ASSET_NAMES),
        *(_fc_260821_r3_object(name, "eval") for name in FC_260821_R3_EVAL_TARGET_ASSET_NAMES),
    ),
    size_axes="width along the shelf, depth, height (standing)",
    fit=fc_260821_r3_fit,
)
"""DB-R3's object axis: the carton, Train R3-T1..T4 and held-out R3-E1, E2, E4, E5, E6 (E3 has no rigid model)."""


FC_260821_R3_NEIGHBOR_LAYOUT = compute_fc_260821_r3_layout()["neighbor_layout"]
FC_260821_R3_NEIGHBOR_INSTANCE_NAMES = tuple(
    placement.instance_name for placement in FC_260821_R3_NEIGHBOR_LAYOUT
)
FC_260821_R3_NEIGHBOR_LOCAL_UP_XYZ = tuple(
    FC_260821_R3_TARGET_SKU_LOCAL_UP_XYZ[placement.asset_name] for placement in FC_260821_R3_NEIGHBOR_LAYOUT
)
"""Each neighbour's local up axis, in the order of FC_260821_R3_NEIGHBOR_INSTANCE_NAMES (the Choco Chip pair
stands on its local +Y)."""

FC_260821_R3_NEIGHBOR_TOPPLE_MAX_UP_ALIGNMENT = 0.70
FC_260821_R3_NEIGHBOR_MAX_DROP_M = 0.030
FC_260821_R3_NEIGHBOR_MAX_LIFT_M = 0.020
FC_260821_R3_NEIGHBOR_MAX_ALONG_M = 0.050
FC_260821_R3_NEIGHBOR_MAX_DEPTH_M = 0.050
if FC_260821_R3_SLOT_OUTWARD_XY[0] * FC_260821_R3_SLOT_OUTWARD_XY[1] != 0.0:
    raise ValueError(f"the shelf's outward normal {FC_260821_R3_SLOT_OUTWARD_XY} is not along a world axis")
FC_260821_R3_NEIGHBOR_DEPTH_AXIS = 0 if FC_260821_R3_SLOT_OUTWARD_XY[0] != 0.0 else 1
FC_260821_R3_NEIGHBOR_ALONG_AXIS = 1 - FC_260821_R3_NEIGHBOR_DEPTH_AXIS


RETAIL_E1_FC_STOCK_ASSET_PREFIX = "dexbench_retail_e1_fc_stock_"
FC_260821_STOCK_SKU_SUBSTITUTIONS: dict[str, str] = {
    "R3E3": "R3E2", "EP1": "TC1", "R1_PLUM": "PLUM_SCAN", "R1_STRAWBERRY": "STRAWBERRY_SCAN",
    "FRESH_RED_APPLE_FC": "APPLE_SCAN", "GREEN_BANANA_FC": "BANANA_SCAN",
}


def fc_260821_resolved_sku(sku: str) -> str:
    """The SKU code that stands in for ``sku`` on the shelf (itself unless substituted)."""
    return FC_260821_STOCK_SKU_SUBSTITUTIONS.get(sku, sku)
FC_260821_STOCK_SKU_ASSETS: dict[str, str] = {
    sku: RETAIL_E1_FC_STOCK_ASSET_PREFIX + sku.lower()
    for sku in (*FC_260821_STOCK_SKUS, *FC_260821_STOCK_SKU_SUBSTITUTIONS.values(), "APPLE_SCAN", "PLUM_SCAN", "STRAWBERRY_SCAN", "BANANA_SCAN")
    if sku not in FC_260821_STOCK_SKU_SUBSTITUTIONS
}
_QX90 = (0.70711, 0.0, 0.0, 0.70711)
FC_260821_STOCK_SKU_STANDING: dict[str, tuple[tuple[float, float, float, float], float]] = {
    "R3T3": (_QX90, 0.068),
    "R3E1": ((-0.5, 0.5, 0.5, 0.5), 0.0633),
    "R3E2": (_QX90, 0.1005),
    "R3E5": (_QX90, 0.095),
}
FC_260821_STOCK_SHELF2_GROUP = "MiniFit"
FC_260821_STOCK_R3_TASK_SHELF_PREFIX = "G3_"
FC_260821_STOCK_PRODUCE_GROUP = "Produce"

FC_260821_PRODUCE_CLUTTER_PACKAGE_SKUS: dict[str, str] = {
    "apple_scan": "APPLE_SCAN",
    "plum_scan": "PLUM_SCAN",
    "strawberry_scan": "STRAWBERRY_SCAN",
    "banana_scan": "BANANA_SCAN",
    "ripe_peach": "R1_PEACH",
    "yellow_lemon_fc": "YELLOW_LEMON_FC",
    "ripe_tomato_fc": "RIPE_TOMATO_FC",
    "fresh_orange_fc": "FRESH_ORANGE_FC",
    "ripe_mango_fc": "RIPE_MANGO_FC",
}
FC_260821_PRODUCE_CLUTTER_TARGET_SKUS = PRODUCE_CLUTTER_TARGET_SKUS
FC_260821_PRODUCE_CLUTTER_SET_IDS = PRODUCE_CLUTTER_SET_IDS
FC_260821_PRODUCE_CLUTTER_DEFAULT = (FC_260821_PRODUCE_CLUTTER_SET_IDS[0], "plum")


def fc_260821_produce_clutter_prim_name(p: ProducePlacement, index: int) -> str:
    return f"clutter_s{p.set_id}_{p.target_sku}_t{p.tray}b{p.bin}_{index:03d}"


def fc_260821_produce_clutter(set_id: int, target_sku: str) -> tuple[ProducePlacement, ...]:
    if target_sku not in FC_260821_PRODUCE_CLUTTER_TARGET_SKUS or set_id not in FC_260821_PRODUCE_CLUTTER_SET_IDS:
        raise ValueError(f"no produce clutter set {set_id} for target {target_sku!r}; have sets {FC_260821_PRODUCE_CLUTTER_SET_IDS} x {FC_260821_PRODUCE_CLUTTER_TARGET_SKUS}")
    return tuple(p for p in PRODUCE_CLUTTER if p.set_id == set_id and p.target_sku == target_sku)


def fc_260821_produce_clutter_placements(
    set_id: int, target_sku: str, *, omit_bin: tuple[int, int] | None = None
) -> tuple[RetailStockPlacement, ...]:
    """The clutter set as display-stock placements (``omit_bin``'s (tray, bin) left out when a task spawns fruit
    there: the DB-R1 pile's)."""
    out = []
    for i, p in enumerate(fc_260821_produce_clutter(set_id, target_sku)):
        if (p.tray, p.bin) == omit_bin:
            continue
        out.append(RetailStockPlacement(FC_260821_STOCK_PRODUCE_GROUP, fc_260821_produce_clutter_prim_name(p, i),
                                        FC_260821_PRODUCE_CLUTTER_PACKAGE_SKUS[p.package], tuple(p.position_xyz), tuple(p.rotation_xyzw)))
    return tuple(out)


@dataclass(frozen=True)
class ProduceBakeRow:
    """One fruit of a clutter set spawned as a rigid body for the Arena bake."""

    instance_name: str
    tray: int
    bin: int
    package: str
    position_xyz: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]


def fc_260821_produce_bake_rows(path, set_id: int, target_sku: str) -> tuple[ProduceBakeRow, ...]:
    """The fruit of one clutter set from a produce-clutter JSON (authored or settled), with bake instance names."""
    import json

    data = json.loads(Path(path).read_text())
    for entry in data["sets"]:
        if int(entry["set"]) == int(set_id) and entry["target_sku"] == target_sku:
            return tuple(
                ProduceBakeRow(
                    f"bake_t{f['tray']}b{f['bin']}_{i:03d}", int(f["tray"]), int(f["bin"]), f["package"],
                    tuple(float(v) for v in f["position_xyz"]), tuple(float(v) for v in f["rotation_xyzw"]),
                )
                for i, f in enumerate(entry["fruit"])
            )
    raise ValueError(f"{path} has no clutter set {set_id} for target {target_sku!r}")


def fc_260821_produce_rigid_asset_name(package: str) -> str:
    """Registry name of the dynamic (rigid) class that spawns a produce package."""
    return f"dexbench_retail_e1_fc_{FC_260821_PRODUCE_CLUTTER_PACKAGE_SKUS[package].lower()}"


def build_fc_260821_produce_bake_assets(asset_registry, path, set_id: int, target_sku: str) -> list:
    """Every fruit of one clutter set as a rigid Arena object at its authored pose (bake mode)."""
    from isaaclab_arena.utils.pose import Pose

    return [
        asset_registry.get_asset_by_name(fc_260821_produce_rigid_asset_name(row.package))(
            instance_name=row.instance_name,
            initial_pose=Pose(position_xyz=row.position_xyz, rotation_xyzw=row.rotation_xyzw),
        )
        for row in fc_260821_produce_bake_rows(path, set_id, target_sku)
    ]


def stock_item_pose(placement: RetailStockPlacement) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """World pose to spawn a stock item at: the transcribed pose composed with its SKU's standing correction."""
    from isaaclab_arena.utils.pose import Pose, compose_poses

    correction = FC_260821_STOCK_SKU_STANDING.get(placement.sku)
    if correction is None:
        return placement.position_xyz, placement.rotation_xyzw
    rotation, lift = correction
    pose = compose_poses(
        Pose(position_xyz=placement.position_xyz, rotation_xyzw=placement.rotation_xyzw),
        Pose(position_xyz=(0.0, 0.0, lift), rotation_xyzw=rotation),
    )
    return pose.position_xyz, pose.rotation_xyzw


def fc_260821_stock_placements(
    *,
    omit_shelf2: bool = False,
    omit_source_prim_names: tuple[str, ...] = (),
    omit_r3_task_shelf: bool = False,
    produce_clutter: tuple[int, str] | None = FC_260821_PRODUCE_CLUTTER_DEFAULT,
    omit_produce_bin: tuple[int, int] | None = None,
    extra_placements: tuple[RetailStockPlacement, ...] = (),
) -> tuple[RetailStockPlacement, ...]:
    """The display stock to spawn, minus whatever a task spawns as dynamic entities."""
    omitted_names = set(omit_source_prim_names)
    kept, matched_names, matched_g3 = [], set(), 0
    for placement in FC_260821_STOCK_LAYOUT:
        if placement.sku in FC_260821_STOCK_SKU_SUBSTITUTIONS:
            placement = dataclasses.replace(placement, sku=fc_260821_resolved_sku(placement.sku))
        if omit_shelf2 and placement.group == FC_260821_STOCK_SHELF2_GROUP:
            continue
        if produce_clutter is not None and placement.group == FC_260821_STOCK_PRODUCE_GROUP:
            continue
        if placement.source_prim_name in omitted_names:
            matched_names.add(placement.source_prim_name)
            continue
        if omit_r3_task_shelf and placement.group == "Gondola" and placement.source_prim_name.startswith(
            FC_260821_STOCK_R3_TASK_SHELF_PREFIX
        ):
            matched_g3 += 1
            continue
        kept.append(placement)
    if produce_clutter is not None:
        kept.extend(fc_260821_produce_clutter_placements(*produce_clutter, omit_bin=omit_produce_bin))
    kept.extend(extra_placements)
    if matched_names != omitted_names:
        raise ValueError(f"FC stock: dynamic prims not in the layout: {sorted(omitted_names - matched_names)}")
    if omit_r3_task_shelf and matched_g3 != len(FC_260821_R3_NEIGHBOR_INSTANCE_NAMES):
        raise ValueError(
            f"FC stock: expected {len(FC_260821_R3_NEIGHBOR_INSTANCE_NAMES)} G3_ shelf items to hand to DB-R3, matched {matched_g3}"
        )
    return tuple(kept)


def build_fc_260821_stock_assets(asset_registry, **omissions) -> list:
    """Instantiate the display stock as physics-free Arena objects (see ``fc_260821_stock_placements``)."""
    from isaaclab_arena.utils.pose import Pose

    assets = []
    for placement in fc_260821_stock_placements(**omissions):
        position, rotation = stock_item_pose(placement)
        assets.append(
            asset_registry.get_asset_by_name(FC_260821_STOCK_SKU_ASSETS[placement.sku])(
                instance_name=placement.instance_name,
                initial_pose=Pose(position_xyz=position, rotation_xyzw=rotation),
            )
        )
    return assets


def build_fc_260821_fixture_assets(
    asset_registry, *, supply_offset_xy=(0.0, 0.0), fixture_kwargs=None, fixture_looks=None, instances=None
) -> list:
    """Instantiate the fixed FC fixtures through the Arena asset registry."""
    from isaaclab_arena.utils.pose import Pose

    fixture_kwargs = dict(fixture_kwargs or {})
    unknown = set(fixture_kwargs) - set(FC_260821_FIXTURE_INSTANCE_NAMES)
    if unknown:
        raise ValueError(f"No FC fixture named {sorted(unknown)}; the fixtures are {list(FC_260821_FIXTURE_INSTANCE_NAMES)}")
    looks = fixture_looks or {}
    unknown = set(instances or ()) - set(FC_260821_FIXTURE_INSTANCE_NAMES)
    if unknown:
        raise ValueError(f"No FC fixture named {sorted(unknown)}; the fixtures are {list(FC_260821_FIXTURE_INSTANCE_NAMES)}")
    return [
        asset_registry.get_asset_by_name(placement.asset_name)(
            instance_name=placement.instance_name,
            **fixture_kwargs.get(placement.instance_name, {}),
            **({"look": looks[placement.asset_name]} if placement.asset_name in looks else {}),
            scale=placement.scale_xyz,
            initial_pose=Pose(
                position_xyz=(
                    placement.position_xyz[0] + supply_offset_xy[0],
                    placement.position_xyz[1] + supply_offset_xy[1],
                    placement.position_xyz[2],
                ) if placement in FC_260821_R3_TOTE_PLACEMENTS else placement.position_xyz,
                rotation_xyzw=placement.rotation_xyzw,
            ),
        )
        for placement in FC_260821_FIXTURE_LAYOUT
        if instances is None or placement.instance_name in instances
    ]


def build_fc_260821_r3_task_assets(
    asset_registry,
    *,
    facing_width_m: float | None = None,
    target_asset_name: str = FC_260821_R3_TARGET_ASSET_NAME,
    box_jitter_xy_m: float = 0.0,
    box_jitter_yaw_deg: float = 0.0,
    box_offset_xy_m=(0.0, 0.0),
    supply_offset_xy=(0.0, 0.0),
) -> list:
    """Instantiate the FC DB-R3 neighbors, slot frame, and tote supply."""
    from isaaclab_arena.utils.pose import Pose

    from dexbench_arena.variations.object_pose_variation import PlanarPoseDomain, make_planar_pose_variation

    if facing_width_m is None:
        facing_width_m = fc_260821_r3_facing_width_m(target_asset_name, None)
    layout = compute_fc_260821_r3_layout(
        facing_width_m, target_asset_name=target_asset_name
    )
    assets = [
        asset_registry.get_asset_by_name(placement.asset_name)(
            instance_name=placement.instance_name,
            initial_pose=Pose(
                position_xyz=placement.position_xyz,
                rotation_xyzw=placement.rotation_xyzw,
            ),
        )
        for placement in layout["neighbor_layout"]
    ]

    if box_jitter_xy_m < 0.0 or box_jitter_yaw_deg < 0.0:
        raise ValueError("FC DB-R3 supply jitter values must be non-negative")
    tote_domain = PlanarPoseDomain(
        max_abs_xy_m=(
            FC_260821_R3_TOTE_INTERIOR_HALF_EXTENTS_XY_M[0] - layout["target_half_extents_xy"][0],
            FC_260821_R3_TOTE_INTERIOR_HALF_EXTENTS_XY_M[1] - layout["target_half_extents_xy"][1],
        ),
        footprint_half_extents_xy_m=layout["target_half_extents_xy"],
        reason="the box would leave the top tote",
    )

    supply_x, supply_y = FC_260821_R3_SUPPLY_CENTER_XY_M
    supply_pose = Pose(
        position_xyz=(
            supply_x + supply_offset_xy[0],
            supply_y + supply_offset_xy[1],
            FC_260821_R3_TARGET_BOTTOM_Z_M + layout["target_origin_above_bottom_m"],
        ),
        rotation_xyzw=layout["target_rotation_xyzw"],
    )

    supply = asset_registry.get_asset_by_name(target_asset_name)(
        instance_name=FC_260821_R3_TARGET_INSTANCE_NAME,
        initial_pose=supply_pose,
    )
    supply.add_variation(
        make_planar_pose_variation(
            supply.name,
            xy_half_extent_m=box_jitter_xy_m,
            yaw_half_extent_rad=math.radians(box_jitter_yaw_deg),
            xy_offset_m=box_offset_xy_m,
            domain=tote_domain,
        )
    )
    assets.append(supply)
    assets.append(
        asset_registry.get_asset_by_name(FC_260821_R3_SLOT_FRAME_ASSET_NAME)(
            instance_name=FC_260821_R3_SLOT_FRAME_INSTANCE_NAME,
            initial_pose=Pose(
                position_xyz=(*layout["slot_center_xy"], FC_260821_R3_SHELF_TOP_Z_M),
                rotation_xyzw=(0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)),
            ),
        )
    )
    return assets
