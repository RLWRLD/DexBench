"""The retail_v1 look: what the FC_260821 store is drawn with, separate from what it is."""

from __future__ import annotations

import math
from dataclasses import dataclass

from dexbench_arena import ASSETS_DIR
from dexbench_arena.assets.scene_bundles import runtime_usd
from dexbench_arena.layouts.e1_retail_fc_layout import (
    FC_260821_FIXTURE_LAYOUT,
    RETAIL_E1_FC_STOCK_ASSET_PREFIX,
)
from dexbench_arena.layouts.retail_v1_dressing import (
    RETAIL_V1_DRESSING_SKU_HEIGHT_M,
    RETAIL_V1_FRIDGE_DRESSING,
    RETAIL_V1_FRIDGE_TIER_CEILING_Z_M,
)

RETAIL_V1_DIR = ASSETS_DIR / "env" / "e1_retail" / "fc" / "retail_v1"

RETAIL_V1_RENDER_SETTINGS: dict[str, int | float] = {
    "rtx:pathtracing:maxBounces": 8,
    "rtx:pathtracing:maxSpecularAndTransmissionBounces": 48,
    "rtx:post:tonemap:exposureTime": 0.02,
    "rtx:post:tonemap:fNumber": 5.0,
    "rtx:post:tonemap:filmIso": 430.0,
    "rtx:post:tonemap:irayReinhard:burnHighlights": 0.2,
    "rtx:post:tonemap:irayReinhard:crushBlacks": 0.0,
    "rtx:post:tonemap:irayReinhard:saturation": 0.8,
    "rtx:post:tonemap:op": 7,
    "rtx:rtpt:maxSpecularAndTransmissionBounces": 48,
    "rtx:translucency:maxRefractionBounces": 48,
}
RETAIL_V1_LIGHT_INTENSITY_SCALE_RANGE = (0.25, 1.5)
RETAIL_V1_LIGHT_COLOR_TEMPERATURE_RANGE_K = (2700.0, 6500.0)
RETAIL_V1_LIGHTING_ASSET = "dexbench_retail_v1_lighting"


@dataclass(frozen=True)
class FixtureLook:
    """What a fixture draws instead of its own meshes: a layer referenced under it, or nothing."""

    usd_path: str | None
    """Layer in the fixture's frame; None only hides the fixture's meshes."""


def _look(name: str) -> FixtureLook:
    return FixtureLook(runtime_usd(RETAIL_V1_DIR / "fixtures" / f"{name}.usdc"))


RETAIL_V1_ROOM_LOOK = FixtureLook(runtime_usd(RETAIL_V1_DIR / "room.usdc"))
RETAIL_V1_FIXTURE_LOOKS: dict[str, FixtureLook] = {
    "dexbench_retail_e1_fc_ceiling_lights": FixtureLook(None),
    "dexbench_retail_e1_fc_gondola": _look("gondola"),
    "dexbench_retail_e1_fc_produce_display": _look("produce_display"),
    "dexbench_retail_e1_fc_fridge_open_180": _look("fridge"),
    "dexbench_retail_e1_fc_fridge_closed": _look("fridge"),
    "dexbench_retail_e1_fc_tote": _look("tote"),
    "dexbench_retail_e1_fc_shopping_cart": _look("shopping_cart"),
    "dexbench_retail_e1_fc_shopping_basket": _look("shopping_basket"),
}


RETAIL_V1_DRESSING_CLEARANCE_M = 0.003
RETAIL_V1_DRESSING_TILT_ALLOWANCE_M = 0.004
RETAIL_V1_DRESSING_FALLBACK_SKUS = ("TC1", "EC1", "TC2")
RETAIL_V1_RAISABLE_SHELF3_FRIDGES = ("retail_e1_fc_fridge_01",)
RETAIL_V1_DRESSING_SKIPPED_TIERS = {"retail_e1_fc_fridge_01": (2,)}


@dataclass(frozen=True)
class RetailV1DressingSlot:
    """One item of the look's fridge stock in its fridge's frame: what it shows, where it stands."""

    instance_name: str
    fridge: str
    """The FC fixture instance the item dresses."""
    tier: int
    sku: str
    """What the item shows: the delivery's SKU, or the fallback that fits under a raised Shelf3."""
    local_xyz: tuple[float, float, float]
    """In the fridge's frame, with the Shelf3 raise applied."""
    rotation_xyzw: tuple[float, float, float, float]


@dataclass(frozen=True)
class RetailV1Dressing:
    """The look's fridge stock of one build: what spawns and what a hidden SKU kept out."""

    slots: tuple[RetailV1DressingSlot, ...]
    hidden: tuple[str, ...]
    """Instance names not spawned because they would have shown a hidden SKU (the delivery's
    item, or the only fallback that fit)."""


@dataclass(frozen=True)
class RetailV1DressingPlacement:
    """One visual-only fridge item of the retail_v1 look, at its world pose."""

    instance_name: str
    sku: str
    position_xyz: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]


def retail_v1_dressing_sku(
    sku: str, top_of_base_z_m: float, ceiling_z_m: float, hidden_skus: tuple[str, ...] = ()
) -> str | None:
    """``sku`` if it stands clear of ``ceiling_z_m`` from ``top_of_base_z_m``, else the tallest
    fallback that does, else None (the slot stays empty). A SKU in ``hidden_skus`` is never
    shown: an item that fits as itself and is hidden stays out (it is the delivery's, and is
    not swapped for another), and a fallback passes over a hidden one."""

    def fits(candidate: str) -> bool:
        top = top_of_base_z_m + RETAIL_V1_DRESSING_SKU_HEIGHT_M[candidate] + RETAIL_V1_DRESSING_TILT_ALLOWANCE_M
        return top <= ceiling_z_m - RETAIL_V1_DRESSING_CLEARANCE_M

    if fits(sku):
        return None if sku in hidden_skus else sku
    return next((c for c in RETAIL_V1_DRESSING_FALLBACK_SKUS if c not in hidden_skus and fits(c)), None)


def fc_260821_retail_v1_dressing(fridge_shelf3_raise_m: float = 0.0, hidden_skus: tuple[str, ...] = ()) -> RetailV1Dressing:
    """The look's fridge stock, fridge-local: plain data, no pose math."""
    if isinstance(fridge_shelf3_raise_m, bool) or not math.isfinite(fridge_shelf3_raise_m) or fridge_shelf3_raise_m < 0:
        raise ValueError("fridge_shelf3_raise_m must be a finite non-negative distance")
    unknown = set(hidden_skus) - set(RETAIL_V1_DRESSING_SKU_HEIGHT_M)
    if unknown:
        raise ValueError(f"hidden_skus {sorted(unknown)}: the dressing's SKUs are {sorted(RETAIL_V1_DRESSING_SKU_HEIGHT_M)}")
    slots, hidden = [], []
    for instance, tier, lane, depth, sku, position, rotation in RETAIL_V1_FRIDGE_DRESSING:
        if tier in RETAIL_V1_DRESSING_SKIPPED_TIERS.get(instance, ()):
            continue
        name = f"retail_v1_dressing_{instance.rsplit('_', 1)[-1]}_t{tier}_l{lane}_d{depth}"
        raise_m = fridge_shelf3_raise_m if instance in RETAIL_V1_RAISABLE_SHELF3_FRIDGES else 0.0
        base_z = position[2] + (raise_m if tier == 3 else 0.0)
        ceiling = RETAIL_V1_FRIDGE_TIER_CEILING_Z_M[tier] + (raise_m if tier == 2 else 0.0)
        chosen = retail_v1_dressing_sku(sku, base_z, ceiling, tuple(hidden_skus))
        if chosen is None:
            if retail_v1_dressing_sku(sku, base_z, ceiling) is not None:
                hidden.append(name)
            continue
        slots.append(RetailV1DressingSlot(name, instance, tier, chosen, (position[0], position[1], base_z), rotation))
    return RetailV1Dressing(slots=tuple(slots), hidden=tuple(hidden))


def fc_260821_retail_v1_fridge_dressing(
    fridge_shelf3_raise_m: float = 0.0, hidden_skus: tuple[str, ...] = ()
) -> tuple[RetailV1DressingPlacement, ...]:
    """The look's fridge stock (``fc_260821_retail_v1_dressing``) at the FC fridges' poses."""
    from isaaclab_arena.utils.pose import Pose, compose_poses

    fridges = {p.instance_name: p for p in FC_260821_FIXTURE_LAYOUT}
    out = []
    for slot in fc_260821_retail_v1_dressing(fridge_shelf3_raise_m, hidden_skus).slots:
        fridge = fridges[slot.fridge]
        world = compose_poses(
            Pose(position_xyz=fridge.position_xyz, rotation_xyzw=fridge.rotation_xyzw),
            Pose(position_xyz=slot.local_xyz, rotation_xyzw=slot.rotation_xyzw),
        )
        out.append(
            RetailV1DressingPlacement(
                instance_name=slot.instance_name,
                sku=slot.sku,
                position_xyz=tuple(float(v) for v in world.position_xyz),
                rotation_xyzw=tuple(float(v) for v in world.rotation_xyzw),
            )
        )
    return tuple(out)


def build_fc_260821_retail_v1_dressing_assets(
    asset_registry, fridge_shelf3_raise_m: float = 0.0, hidden_skus: tuple[str, ...] = ()
) -> list:
    """The look's fridge dressing as physics-free Arena objects (the FC visual-stock assets)."""
    from isaaclab_arena.utils.pose import Pose

    return [
        asset_registry.get_asset_by_name(RETAIL_E1_FC_STOCK_ASSET_PREFIX + p.sku.lower())(
            instance_name=p.instance_name,
            initial_pose=Pose(position_xyz=p.position_xyz, rotation_xyzw=p.rotation_xyzw),
        )
        for p in fc_260821_retail_v1_fridge_dressing(fridge_shelf3_raise_m, hidden_skus)
    ]
