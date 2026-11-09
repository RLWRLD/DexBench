"""DB-R2 task layout on the reviewed 260821 E1 retail scene (refrigerator station)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

from dexbench_arena.tasks.object_axis import ObjectAxis, TaskObject
from dexbench_arena.assets.retail_sku_packages import RETAIL_E1_FC_SKU_PACKAGES
from dexbench_arena.layouts.e1_retail_fc_layout import (
    FC_260821_LAYOUT_SPEC,
    FC_260821_STOCK_LAYOUT,
    FC_260821_STOCK_SHELF2_GROUP,
    RetailFixturePlacement,
    fc_260821_resolved_sku,
    fc_260821_stock_placements,
)
from dexbench_arena.layouts.e1_retail_fc_stock_layout import RetailStockPlacement
from dexbench_arena.layouts.retail_v1_look import fc_260821_retail_v1_dressing


@dataclass(frozen=True)
class RetailR2ItemPlacement:
    """One dynamic DB-R2 drink: a task role bound to an authored source pose."""

    asset_name: str
    instance_name: str
    source_prim_name: str
    position_xyz: tuple[float, float, float]

    @property
    def rotation_xyzw(self) -> tuple[float, float, float, float]:
        return FC_260821_R2_SHELF2_ITEM_ROT_XYZW



FC_260821_R2_FRIDGE_INSTANCE_NAME = "retail_e1_fc_fridge_01"
FC_260821_R2_BASKET_INSTANCE_NAME = "retail_e1_fc_r2_basket"
FC_260821_R2_ANCHOR = "Fridge"

FC_260821_R2_FRIDGE_FRONT_PLANE_Y_M = 2.41
FC_260821_R2_SHELF2_TOP_Z_M = 0.925
FC_260821_R2_MINIFIT_RAIL_TOP_Z_M = 1.035
FC_260821_R2_SHELF2_ITEM_PITCH_DEG = 5.0
FC_260821_R2_SHELF2_ITEM_ROT_XYZW = (
    math.sin(math.radians(FC_260821_R2_SHELF2_ITEM_PITCH_DEG) * 0.5),
    0.0,
    0.0,
    math.cos(math.radians(FC_260821_R2_SHELF2_ITEM_PITCH_DEG) * 0.5),
)

FC_260821_R2_TARGET_ASSET_NAME = "dexbench_retail_e1_fc_tc2"
FC_260821_R2_TARGET_INSTANCE_NAME = "r2_target_drink"
FC_260821_R2_TARGET_SOURCE_PRIM_NAME = "MF6_TC2_0"
FC_260821_R2_TARGET_SKU = "TC2"

FC_260821_R2_TARGET_PLACEMENT = RetailR2ItemPlacement(
    FC_260821_R2_TARGET_ASSET_NAME,
    FC_260821_R2_TARGET_INSTANCE_NAME,
    FC_260821_R2_TARGET_SOURCE_PRIM_NAME,
    (2.83818, 2.52630, 0.93739),
)

FC_260821_R2_NEIGHBOR_LAYOUT: tuple[RetailR2ItemPlacement, ...] = (
    RetailR2ItemPlacement(
        "dexbench_retail_e1_fc_ec2",
        "r2_neighbor_0",
        "MF5_EC2_0",
        (2.75500, 2.52635, 0.93739),
    ),
    RetailR2ItemPlacement(
        "dexbench_retail_e1_fc_tc2",
        "r2_neighbor_1",
        "MF6_TC2_1",
        (2.83818, 2.59404, 0.94331),
    ),
    RetailR2ItemPlacement(
        "dexbench_retail_e1_fc_ec2",
        "r2_neighbor_2",
        "MF5_EC2_1",
        (2.75500, 2.59419, 0.94333),
    ),
)

FC_260821_R2_START_CLEARANCE_XYZ_M = (0.0, 0.002, 0.001)


def _start_clear(placement: RetailR2ItemPlacement) -> RetailR2ItemPlacement:
    position = tuple(a + b for a, b in zip(placement.position_xyz, FC_260821_R2_START_CLEARANCE_XYZ_M))
    return replace(placement, position_xyz=position)


FC_260821_R2_NEIGHBOR_INSTANCE_NAMES = tuple(p.instance_name for p in FC_260821_R2_NEIGHBOR_LAYOUT)
FC_260821_R2_DYNAMIC_PLACEMENTS = (FC_260821_R2_TARGET_PLACEMENT, *FC_260821_R2_NEIGHBOR_LAYOUT)
FC_260821_R2_DYNAMIC_SOURCE_PRIM_NAMES = tuple(p.source_prim_name for p in FC_260821_R2_DYNAMIC_PLACEMENTS)

FC_260821_R2_TASK_FRAME_ASSET_NAME = "dexbench_task_frame"

FC_260821_R2_OPENING_FRAME_INSTANCE_NAME = "r2_fridge_opening_frame"
FC_260821_R2_OPENING_FRAME_XYZ_M = (
    FC_260821_R2_TARGET_PLACEMENT.position_xyz[0],
    FC_260821_R2_FRIDGE_FRONT_PLANE_Y_M,
    0.98,
)

FC_260821_R2_DESTINATION_FRAME_INSTANCE_NAME = "r2_destination_frame"
FC_260821_R2_BASKET_ORIGIN_XYZ_M = (3.20, 1.90, 0.60)
FC_260821_R2_BASKET_INTERIOR_HALF_EXTENTS_XY_M = (0.1175, 0.1775)
FC_260821_R2_BASKET_FLOOR_TOP_Z_M = 0.603
FC_260821_R2_BASKET_WALL_TOP_Z_M = 0.830
FC_260821_R2_DESTINATION_FRAME_XYZ_M = (
    FC_260821_R2_BASKET_ORIGIN_XYZ_M[0],
    FC_260821_R2_BASKET_ORIGIN_XYZ_M[1],
    FC_260821_R2_BASKET_FLOOR_TOP_Z_M,
)


FC_260821_R2_EXTRACTION_CLEARANCE_M = 0.04
FC_260821_R2_WRONG_PICK_LIFT_M = 0.03
FC_260821_R2_NEIGHBOR_DISPLACEMENT_COLLECTION_M = 0.030
FC_260821_R2_NEIGHBOR_DISPLACEMENT_STRICT_M = 0.010
"""A displacement gate for the contract check and for rescoring runs scored with it; the strict tier gates
knocked neighbours (``FC_260821_R2_NEIGHBOR_KNOCK_*``)."""
FC_260821_R2_NEIGHBOR_KNOCK_LIFT_M = 0.02
FC_260821_R2_NEIGHBOR_TOPPLE_MAX_UP_ALIGNMENT = 0.70
FC_260821_R2_VELOCITY_THRESHOLD_M_S = 0.10
FC_260821_R2_BASKET_CENTER_HEIGHT_RANGE_M = (0.015, FC_260821_R2_BASKET_WALL_TOP_Z_M - FC_260821_R2_BASKET_FLOOR_TOP_Z_M)
FC_260821_R2_WORKSPACE_XY_BOUNDS = ((1.90, 1.20), (3.60, 3.10))
FC_260821_R2_OBJECT_DROPPED_MIN_Z_M = 0.45
FC_260821_R2_EPISODE_LENGTH_S = 30.0
FC_260821_R2_STRICT_HOLD_S_DEFAULT = 1.0
FC_260821_R2_STRICT_HOLD_SHARE = 0.9

FC_260821_R2_PROGRESS_LIFT_HEIGHT_GAIN_M = 0.02
FC_260821_R2_PROGRESS_OVER_BASKET_XY_MARGIN_M = 0.05


def fc_260821_r2_source_row_gaps_m() -> dict[str, float]:
    """Authored center-to-center gaps between the target and each neighbor (XY only)."""

    tx, ty, _ = FC_260821_R2_TARGET_PLACEMENT.position_xyz
    return {
        p.instance_name: math.hypot(p.position_xyz[0] - tx, p.position_xyz[1] - ty)
        for p in FC_260821_R2_NEIGHBOR_LAYOUT
    }


FC_260821_R2_LANE_DRAW_OMIT_DEPTHS = 3


def _minifit_slot(source_prim_name: str) -> tuple[int, str, int]:
    """``(lane, authored SKU, depth)`` of an authored ``MF<lane>_<SKU>_<depth>`` prim."""
    lane, sku, depth = source_prim_name.split("_")
    return int(lane.removeprefix("MF")), sku, int(depth)


FC_260821_R2_SHELF2_SLOTS: tuple[tuple[int, str, int, str, tuple[float, float, float]], ...] = tuple(
    (*_minifit_slot(p.source_prim_name), p.source_prim_name, p.position_xyz)
    for p in FC_260821_STOCK_LAYOUT
    if p.group == FC_260821_STOCK_SHELF2_GROUP
)
"""Every authored Shelf2 slot as ``(lane, authored SKU, depth, prim name, position)``."""
FC_260821_R2_SHELF2_LANE_X_M: dict[int, float] = {lane: xyz[0] for lane, _, _, _, xyz in FC_260821_R2_SHELF2_SLOTS}
FC_260821_R2_SHELF2_LANES: tuple[int, ...] = tuple(sorted(FC_260821_R2_SHELF2_LANE_X_M))
FC_260821_R2_NEIGHBOR_KNOCK_LATERAL_M = 0.5 * min(
    abs(FC_260821_R2_SHELF2_LANE_X_M[b] - FC_260821_R2_SHELF2_LANE_X_M[a])
    for a, b in zip(FC_260821_R2_SHELF2_LANES, FC_260821_R2_SHELF2_LANES[1:])
)
"""Half the narrowest lane pitch on Shelf2: a neighbour pushed this far sideways has left its lane."""


FC_260821_R2_SKU_ASSET_NAMES: dict[str, str] = {
    "TC1": "dexbench_retail_e1_fc_tc1",
    "TC2": "dexbench_retail_e1_fc_tc2",
    "TP1": "dexbench_retail_e1_fc_tp1",
    "EC1": "dexbench_retail_e1_fc_ec1",
    "EC2": "dexbench_retail_e1_fc_ec2",
    "EP2": "dexbench_retail_e1_fc_ep2",
    "EP3": "dexbench_retail_e1_fc_ep3",
}
FC_260821_R2_SKU_FOOTPRINT_M: dict[str, tuple[float, float]] = {
    "TC1": (0.0572, 0.1461),
    "TC2": (0.0660, 0.1242),
    "TP1": (0.0659, 0.2215),
    "EC1": (0.0534, 0.1348),
    "EC2": (0.0660, 0.1686),
    "EP2": (0.0674, 0.2350),
    "EP3": (0.0750, 0.2250),
}
FC_260821_R2_SHELF2_FRONT_STOP_Y_M = 2.49330
FC_260821_R2_SHELF2_PACK_CLEARANCE_M = 0.001754
FC_260821_R2_SHELF2_RAIL_SLOPE = math.tan(math.radians(FC_260821_R2_SHELF2_ITEM_PITCH_DEG))
FC_260821_R2_SHELF2_BOTTOM_Z_AT_STOP_M = 0.93739 - 0.033 * FC_260821_R2_SHELF2_RAIL_SLOPE
FC_260821_R2_SHELF2_CEILING_Z_M = 1.152
FC_260821_R2_SHELF2_STOPPER_TOP_Z_M = 0.959
FC_260821_R2_DEFAULT_TARGET_SKU = FC_260821_R2_TARGET_SKU
FC_260821_R2_DEFAULT_TARGET_LANE = 6
"""Where the reviewed set stands (``MF6_TC2_0``) and the published recordings were made."""
FC_260821_R2_STOCK_RULES = ("authored", "unique_target")
FC_260821_R2_DEFAULT_STOCK_RULE = "authored"
FC_260821_R2_DEFAULT_STOCK_FILLERS = "TC2,TC1"
"""TC2 first: the train SKU the delivery stocks most (MF1, MF6), so a held-out run gains no new
SKU and a train run no held-out one; TC1 (a train SKU too) stands in when TC2 is the target.
Both are cans, which fit under every Shelf3 profile."""

FC_260821_FRIDGE_SHELF3_PART_NAMES = ("FrgFC_Shelf3", "FrgFC_MiniFitS3A", "FrgFC_MiniFitS3B")
FC_260821_FRIDGE_SHELF3_MODULE_TOP_Z_M = 1.2711
FC_260821_FRIDGE_SHELF4_UNDERSIDE_Z_M = 1.390
FC_260821_FRIDGE_SHELF3_MAX_RAISE_M = FC_260821_FRIDGE_SHELF4_UNDERSIDE_Z_M - FC_260821_FRIDGE_SHELF3_MODULE_TOP_Z_M


def check_fridge_shelf3_raise(raise_m: float) -> float:
    """``raise_m`` as a float when the task fridge's Shelf3 can be raised that far; a ValueError otherwise."""
    if isinstance(raise_m, bool) or not math.isfinite(float(raise_m)) or float(raise_m) < 0.0:
        raise ValueError(f"fridge_shelf3_raise_m must be a finite non-negative distance in metres, got {raise_m!r}")
    if float(raise_m) >= FC_260821_FRIDGE_SHELF3_MAX_RAISE_M:
        raise ValueError(
            f"fridge_shelf3_raise_m {raise_m} m puts Shelf3's MiniFit rails into Shelf4: their top is at "
            f"{FC_260821_FRIDGE_SHELF3_MODULE_TOP_Z_M} m and Shelf4's underside at {FC_260821_FRIDGE_SHELF4_UNDERSIDE_Z_M} m, "
            f"so the raise must stay below {FC_260821_FRIDGE_SHELF3_MAX_RAISE_M * 1000:.1f} mm."
        )
    return float(raise_m)


def fc_260821_r2_sku_headroom_m(sku: str, shelf3_raise_m: float = 0.0) -> float:
    """Vertical room left above ``sku`` [m] once it is lifted over the front stopper; negative does not fit."""
    diameter, height = FC_260821_R2_SKU_FOOTPRINT_M[sku]
    pitch = math.radians(FC_260821_R2_SHELF2_ITEM_PITCH_DEG)
    top = height * math.cos(pitch) + 0.5 * diameter * math.sin(pitch)
    lift = FC_260821_R2_SHELF2_STOPPER_TOP_Z_M - FC_260821_R2_SHELF2_BOTTOM_Z_AT_STOP_M
    ceiling = FC_260821_R2_SHELF2_CEILING_Z_M + check_fridge_shelf3_raise(shelf3_raise_m)
    return ceiling - FC_260821_R2_SHELF2_BOTTOM_Z_AT_STOP_M - lift - top


FC_260821_R2_TRAIN_SKUS = ("TC1", "TC2", "TP1")
FC_260821_R2_EVAL_SKUS = ("EC1", "EC2", "EP2", "EP3")
FC_260821_R2_TARGET_SKUS = (*FC_260821_R2_TRAIN_SKUS, *FC_260821_R2_EVAL_SKUS)


def check_r2_target_sku(sku: str, shelf3_raise_m: float = 0.0) -> str:
    """``sku`` when Shelf2 can hold it as the target under the Shelf3 raise; a ValueError naming why not otherwise."""
    if sku not in FC_260821_R2_TARGET_SKUS:
        raise ValueError(f"--r2_target_sku {sku!r} is not a DB-R2 SKU; choose from {list(FC_260821_R2_TARGET_SKUS)}.")
    headroom = fc_260821_r2_sku_headroom_m(sku, shelf3_raise_m)
    if headroom <= 0.0:
        fitting = [other for other in FC_260821_R2_TARGET_SKUS if fc_260821_r2_sku_headroom_m(other, shelf3_raise_m) > 0.0]
        raise ValueError(
            f"--r2_target_sku {sku}: {FC_260821_R2_SKU_FOOTPRINT_M[sku][1] * 1000:.0f} mm tall, it does not fit "
            f"under Shelf3 raised {float(shelf3_raise_m) * 1000:.0f} mm ({-headroom * 1000:.0f} mm short of the room "
            f"to lift it over the stopper); this fixture profile holds {fitting}. Raise Shelf3 for the whole split "
            "(fridge_shelf3_raise_m) to fit it."
        )
    return sku


FC_260821_R2_SPLIT_SHELF3_RAISE_M = 0.065


def fc_260821_r2_fit(task_object: TaskObject, cfg) -> dict:
    """What DB-R2 derives from its target under ``cfg``: the Shelf3 fit and the Shelf2 row it builds."""
    sku = task_object.key
    raise_m = float(getattr(cfg, "fridge_shelf3_raise_m", 0.0) or 0.0)
    check_r2_target_sku(sku, raise_m)
    diameter, _height = FC_260821_R2_SKU_FOOTPRINT_M[sku]
    stocking = fc_260821_r2_stocking_for(cfg, sku)
    fillers = getattr(cfg, "r2_stock_fillers", None) or FC_260821_R2_DEFAULT_STOCK_FILLERS
    return {
        "fridge_shelf3_raise_m": raise_m,
        "headroom_m": round(fc_260821_r2_sku_headroom_m(sku, raise_m), 6),
        "lane_pitch_m": round(diameter + FC_260821_R2_SHELF2_PACK_CLEARANCE_M, 6),
        "r2_stock_fillers": ",".join(parse_r2_stock_fillers(fillers, raise_m)),
        **fc_260821_r2_stocking_record(stocking),
    }


FC_260821_R2_SKU_LABELS: dict[str, str] = {
    "TC1": "can",
    "TC2": "can",
    "TP1": "bottle",
    "EC1": "can",
    "EC2": "can",
    "EP2": "bottle",
    "EP3": "bottle",
}
FC_260821_R2_SPLIT_STOCK_RULE = "unique_target"


FC_260821_R2_OBJECT_AXIS = ObjectAxis(
    field="r2_target_sku",
    default=FC_260821_R2_DEFAULT_TARGET_SKU,
    objects=tuple(
        TaskObject(
            key=sku,
            asset=FC_260821_R2_SKU_ASSET_NAMES[sku],
            split="train" if sku in FC_260821_R2_TRAIN_SKUS else "eval",
            label=FC_260821_R2_SKU_LABELS[sku],
            size_m=FC_260821_R2_SKU_FOOTPRINT_M[sku],
            benchmark_asset_major={"EP2": 3}.get(sku),
            benchmark_asset_major_renames=((1, 2,
                f"assets/object_library/{RETAIL_E1_FC_SKU_PACKAGES[sku]}/CHANGELOG.md"),),
        )
        for sku in FC_260821_R2_TARGET_SKUS
    ),
    size_axes="diameter, height",
    fixture_profile={
        "fridge_shelf3_raise_m": FC_260821_R2_SPLIT_SHELF3_RAISE_M,
        "r2_stock_rule": FC_260821_R2_SPLIT_STOCK_RULE,
        "r2_stock_fillers": FC_260821_R2_DEFAULT_STOCK_FILLERS,
    },
    fit=fc_260821_r2_fit,
)
"""DB-R2's object axis: the target drink, Train TC1/TC2/TP1 and held-out EC1/EC2/EP2/EP3."""


def fc_260821_r2_pack_slot(lane: int, sku: str, depth: int) -> tuple[float, float, float]:
    """Where ``sku``'s ``depth``-th drink stands when ``lane`` is packed with it (the three rules above)."""
    diameter, _ = FC_260821_R2_SKU_FOOTPRINT_M[sku]
    offset = 0.5 * diameter + depth * (diameter + FC_260821_R2_SHELF2_PACK_CLEARANCE_M)
    return (
        FC_260821_R2_SHELF2_LANE_X_M[lane],
        FC_260821_R2_SHELF2_FRONT_STOP_Y_M + offset,
        FC_260821_R2_SHELF2_BOTTOM_Z_AT_STOP_M + offset * FC_260821_R2_SHELF2_RAIL_SLOPE,
    )


FC_260821_R2_ORDINALS = ("first", "second", "third", "fourth", "fifth", "sixth")


def fc_260821_r2_lane_ordinal(lane: int) -> str:
    """Where ``lane`` is in Shelf2's row counted from the left as the robot at the R2 anchor sees it ("third")."""
    base_x, base_y, _ = FC_260821_LAYOUT_SPEC.robot_base_anchors_xyz[FC_260821_R2_ANCHOR]
    target_x, target_y = FC_260821_LAYOUT_SPEC.anchor_facing_targets_xy[FC_260821_R2_ANCHOR]
    left_x, left_y = -(target_y - base_y), target_x - base_x
    front = {slot_lane: xyz for slot_lane, _, depth, _, xyz in FC_260821_R2_SHELF2_SLOTS if depth == 0}
    from_left = sorted(front, key=lambda each: -(left_x * front[each][0] + left_y * front[each][1]))
    return FC_260821_R2_ORDINALS[from_left.index(lane)]


def fc_260821_r2_task_description(label: str, lane: int | None) -> str:
    """The instruction: the drink by its kind, its lane from the left, and Shelf2 as the EA text named it."""
    target = label if lane is None else f"{label} in the {fc_260821_r2_lane_ordinal(lane)} lane from the left"
    return (
        f"Take the {target} out of the open refrigerator's second shelf "
        "and put it in the basket without disturbing the drinks next to it."
    )


def fc_260821_r2_lane_label(lane: int) -> str:
    """The name a lane goes by in the delivery and in a recorded draw (``MF6``)."""
    return f"MF{lane}"


def fc_260821_r2_lateral_lane(lane: int) -> int:
    """The lane that supplies the same-depth lateral and rear-diagonal neighbors."""
    return lane - 1 if lane > FC_260821_R2_SHELF2_LANES[0] else lane + 1


def fc_260821_r2_other_lateral_lane(lane: int) -> int | None:
    """The lane on the target's other side, which only a middle lane has (MF2-MF5); None at a wall."""
    return lane + 1 if FC_260821_R2_SHELF2_LANES[0] < lane < FC_260821_R2_SHELF2_LANES[-1] else None


FC_260821_R2_SHELF2_BOARD_END_Y_M = 2.91
_R2_ROLE_NAMES = ("r2_neighbor_0", "r2_neighbor_1", "r2_neighbor_2", "r2_neighbor_3", "r2_neighbor_4")


def parse_r2_stock_fillers(value: str | Sequence[str], shelf3_raise_m: float = 0.0) -> tuple[str, ...]:
    """The filler SKUs of ``--r2_stock_fillers`` (``"TC2,TC1"``): at least two distinct DB-R2 SKUs that fit."""
    tokens = value.replace(",", " ").split() if isinstance(value, str) else [str(v) for v in value]
    fillers = tuple(dict.fromkeys(token.strip().upper() for token in tokens))
    if len(fillers) < 2:
        raise ValueError(f"--r2_stock_fillers needs two distinct SKUs (one is never the target); got {value!r}")
    for sku in fillers:
        check_r2_target_sku(sku, shelf3_raise_m)
    return fillers


@dataclass(frozen=True)
class RetailR2LaneStock:
    """What one Shelf2 lane holds, front to back, before the dynamic set takes its slots."""

    lane: int
    sku: str
    """The SKU the lane shows."""
    authored_sku: str
    """The delivery's code for the lane (EP1 for MF3, which shows TC1)."""
    placements: tuple[RetailStockPlacement, ...]
    packed: bool
    """Re-packed at ``sku``'s pitch (``fc_260821_r2_pack_slot``); otherwise the authored poses."""


@dataclass(frozen=True)
class RetailR2Stocking:
    """The DB-R2 row of one build: the dynamic set per candidate lane and the backdrop around it."""

    sku: str
    target_lane: int
    """The nominal lane: where the set spawns, and the lane a build without a draw keeps."""
    lanes: tuple[int, ...]
    """The lanes the target can be in: ``(target_lane,)``, or a draw's lanes."""
    draw: bool
    stock_rule: str
    filler_sku: str | None
    """What a lane that would repeat the target is restocked with (``unique_target`` only)."""
    rows: tuple[RetailR2LaneStock, ...]
    seats: tuple[tuple[RetailR2ItemPlacement, ...], ...]
    """Per lane of ``lanes``, the target then its neighbors in role order."""
    opening_frames_xyz: tuple[tuple[float, float, float], ...]
    """Per lane of ``lanes``, the opening frame on the fridge front plane in front of that lane."""
    omitted_source_prim_names: tuple[str, ...]
    """Authored Shelf2 prims the visual stock drops (the dynamic slots, every re-packed lane)."""
    backdrop_stock: tuple[RetailStockPlacement, ...]
    """Visual stock the re-packed lanes add in their place."""
    restocked: tuple[tuple[int, str, str], ...]
    """``(lane, SKU it would show, filler)`` for each lane the stock rule restocked."""
    shelf3_raise_m: float = 0.0
    """The task fridge's Shelf3 raise the row was built for (the dressing above it follows it)."""
    hidden_source_prim_names: tuple[str, ...] = ()
    """Authored FC stock prims outside the row the visual stock drops because they show the
    target SKU (``unique_target``: the tote stock's)."""
    hidden_dressing: tuple[str, ...] = ()
    """The look's fridge-dressing items left out because they would show the target SKU."""

    @property
    def hidden_skus(self) -> tuple[str, ...]:
        """SKUs no visual item outside the row may show: the target under ``unique_target``."""
        return (self.sku,) if self.stock_rule == "unique_target" else ()

    @property
    def stock_omissions(self) -> tuple[str, ...]:
        """Every authored FC stock prim the visual stock drops: the row's and the hidden ones."""
        return (*self.omitted_source_prim_names, *self.hidden_source_prim_names)

    @property
    def hidden_visual_stock(self) -> tuple[str, ...]:
        """Instance names of every visual item this build leaves out for showing the target SKU."""
        by_prim = {p.source_prim_name: p.instance_name for p in FC_260821_STOCK_LAYOUT}
        return (*(by_prim[name] for name in self.hidden_source_prim_names), *self.hidden_dressing)

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(fc_260821_r2_lane_label(lane) for lane in self.lanes)

    @property
    def spawn_index(self) -> int:
        return self.lanes.index(self.target_lane)

    @property
    def placements(self) -> tuple[RetailR2ItemPlacement, ...]:
        """The dynamic set where it spawns (the nominal lane)."""
        return self.seats[self.spawn_index]

    @property
    def neighbor_instance_names(self) -> tuple[str, ...]:
        return tuple(p.instance_name for p in self.seats[0][1:])

    def row(self, lane: int) -> RetailR2LaneStock:
        return next(row for row in self.rows if row.lane == lane)


def _seat_yz(sku: str) -> dict[int, tuple[float, float]]:
    """Front and second slot of ``sku`` (y, z), the same in every lane; TC2's are the authored lane-6 pair."""
    if sku == FC_260821_R2_TARGET_SKU:
        rear = FC_260821_R2_NEIGHBOR_LAYOUT[1]
        return {0: FC_260821_R2_TARGET_PLACEMENT.position_xyz[1:], 1: rear.position_xyz[1:]}
    return {depth: fc_260821_r2_pack_slot(FC_260821_R2_SHELF2_LANES[0], sku, depth)[1:] for depth in (0, 1)}


def fc_260821_r2_stocking(
    sku: str = FC_260821_R2_DEFAULT_TARGET_SKU,
    *,
    target_lane: int = FC_260821_R2_DEFAULT_TARGET_LANE,
    lane_draw: Sequence[int] = (),
    stock_rule: str = FC_260821_R2_DEFAULT_STOCK_RULE,
    stock_fillers: str | Sequence[str] = FC_260821_R2_DEFAULT_STOCK_FILLERS,
    shelf3_raise_m: float = 0.0,
) -> RetailR2Stocking:
    """The Shelf2 row for ``sku`` as the target in ``target_lane`` (or a draw over ``lane_draw``)."""
    check_r2_target_sku(sku, shelf3_raise_m)
    if stock_rule not in FC_260821_R2_STOCK_RULES:
        raise ValueError(f"--r2_stock_rule {stock_rule!r}: choose one of {list(FC_260821_R2_STOCK_RULES)}")
    fillers = parse_r2_stock_fillers(stock_fillers, shelf3_raise_m)
    target_lane = int(target_lane)
    if target_lane not in FC_260821_R2_SHELF2_LANE_X_M:
        raise ValueError(f"--r2_target_lane {target_lane}: Shelf2's lanes are {list(FC_260821_R2_SHELF2_LANES)}")
    draw = tuple(sorted({int(lane) for lane in lane_draw}))
    unknown = [lane for lane in draw if lane not in FC_260821_R2_SHELF2_LANE_X_M]
    if unknown:
        raise ValueError(f"Shelf2 has no lane(s) {unknown}; its lanes are {list(FC_260821_R2_SHELF2_LANES)}.")
    if draw and target_lane not in draw:
        raise ValueError(
            f"--r2_lane_draw {','.join(map(str, draw))} leaves out the nominal --r2_target_lane {target_lane}, "
            "where the set spawns and a build without the draw keeps it."
        )
    lanes = draw or (target_lane,)
    filler = next(f for f in fillers if f != sku) if stock_rule == "unique_target" else None

    both_sides = all(fc_260821_r2_other_lateral_lane(lane) is not None for lane in lanes)

    def laterals(lane: int) -> tuple[int, ...]:
        other = fc_260821_r2_other_lateral_lane(lane)
        return (fc_260821_r2_lateral_lane(lane), *((other,) if both_sides else ()))

    dynamic_lanes = set(lanes) | {side for lane in lanes for side in laterals(lane)}
    cleared_depths = range(FC_260821_R2_LANE_DRAW_OMIT_DEPTHS) if draw else range(2)

    authored: dict[int, list] = {}
    for lane, authored_sku, depth, name, xyz in FC_260821_R2_SHELF2_SLOTS:
        authored.setdefault(lane, []).append((depth, authored_sku, name, xyz))
    authored_rotation = {p.source_prim_name: p.rotation_xyzw for p in FC_260821_STOCK_LAYOUT}

    rows, restocked = [], []
    for lane in FC_260821_R2_SHELF2_LANES:
        slots = sorted(authored[lane])
        raw = slots[0][1]
        shown = fc_260821_resolved_sku(raw)
        if not draw and lane == target_lane:
            shown = sku
        elif filler is not None and shown == sku:
            restocked.append((lane, shown, filler))
            shown = filler
        packed = shown != raw and (lane in dynamic_lanes or shown != fc_260821_resolved_sku(raw))
        rotation = authored_rotation[slots[0][2]]
        if packed:
            count = len(slots)
            while fc_260821_r2_pack_slot(lane, shown, count - 1)[1] + 0.5 * FC_260821_R2_SKU_FOOTPRINT_M[shown][0] > FC_260821_R2_SHELF2_BOARD_END_Y_M:
                count -= 1
            placements = tuple(
                RetailStockPlacement(
                    FC_260821_STOCK_SHELF2_GROUP, f"MF{lane}_{shown}_{depth}", shown,
                    fc_260821_r2_pack_slot(lane, shown, depth), rotation,
                )
                for depth in range(count)
            )
        else:
            placements = tuple(
                RetailStockPlacement(FC_260821_STOCK_SHELF2_GROUP, name, shown, xyz, authored_rotation[name])
                for _depth, _sku, name, xyz in slots
            )
        rows.append(RetailR2LaneStock(lane, shown, raw, placements, packed))
    row = {r.lane: r for r in rows}

    def seat(name: str, lane: int, depth: int, role_sku: str | None) -> RetailR2ItemPlacement:
        if draw:
            seat_sku = role_sku
            prim = f"{fc_260821_r2_lane_label(lane)}_{seat_sku}_{depth}"
            xyz = (FC_260821_R2_SHELF2_LANE_X_M[lane], *_seat_yz(seat_sku)[depth])
        else:
            stock = row[lane]
            seat_sku, prim, xyz = stock.sku, stock.placements[depth].source_prim_name, stock.placements[depth].position_xyz
        return _start_clear(RetailR2ItemPlacement(FC_260821_R2_SKU_ASSET_NAMES[seat_sku], name, prim, xyz))

    lateral_sku = (filler or sku) if draw else None
    seats, frames = [], []
    for lane in lanes:
        sides = laterals(lane)
        roles = [(FC_260821_R2_TARGET_INSTANCE_NAME, lane, 0, sku), (_R2_ROLE_NAMES[0], sides[0], 0, lateral_sku),
                 (_R2_ROLE_NAMES[1], lane, 1, sku), (_R2_ROLE_NAMES[2], sides[0], 1, lateral_sku)]
        if len(sides) == 2:
            roles += [(_R2_ROLE_NAMES[3], sides[1], 0, lateral_sku), (_R2_ROLE_NAMES[4], sides[1], 1, lateral_sku)]
        seats.append(tuple(seat(*role) for role in roles))
        frames.append((FC_260821_R2_SHELF2_LANE_X_M[lane], *FC_260821_R2_OPENING_FRAME_XYZ_M[1:]))

    omitted: list[str] = []
    for lane in sorted(dynamic_lanes) if draw else ():
        if not row[lane].packed:
            omitted += [p.source_prim_name for depth, p in enumerate(row[lane].placements) if depth in cleared_depths]
    if not draw:
        omitted += [p.source_prim_name for p in seats[0] if not row[_lane_of(p.position_xyz[0])].packed]
    backdrop: list[RetailStockPlacement] = []
    for stock in rows:
        if stock.packed:
            omitted += [name for _depth, _sku, name, _xyz in sorted(authored[stock.lane])]
            keep = [depth not in cleared_depths or stock.lane not in dynamic_lanes for depth in range(len(stock.placements))]
            backdrop += [p for p, kept in zip(stock.placements, keep) if kept]

    hidden_skus = (sku,) if stock_rule == "unique_target" else ()
    stocking = RetailR2Stocking(
        sku=sku,
        target_lane=target_lane,
        lanes=lanes,
        draw=bool(draw),
        stock_rule=stock_rule,
        filler_sku=filler,
        rows=tuple(rows),
        seats=tuple(seats),
        opening_frames_xyz=tuple(frames),
        omitted_source_prim_names=tuple(omitted),
        backdrop_stock=tuple(backdrop),
        restocked=tuple(restocked),
        shelf3_raise_m=float(shelf3_raise_m),
        hidden_source_prim_names=tuple(
            p.source_prim_name
            for p in FC_260821_STOCK_LAYOUT
            if p.group != FC_260821_STOCK_SHELF2_GROUP and fc_260821_resolved_sku(p.sku) in hidden_skus
        ),
        hidden_dressing=fc_260821_retail_v1_dressing(shelf3_raise_m, hidden_skus).hidden,
    )
    check_r2_stocking(stocking)
    return stocking


def _lane_of(x: float) -> int:
    lane = min(FC_260821_R2_SHELF2_LANE_X_M, key=lambda k: abs(FC_260821_R2_SHELF2_LANE_X_M[k] - x))
    if abs(FC_260821_R2_SHELF2_LANE_X_M[lane] - x) > 1e-3:
        raise ValueError(f"x {x} is on no Shelf2 lane")
    return lane


_R2_ASSET_SKUS = {asset: sku for sku, asset in FC_260821_R2_SKU_ASSET_NAMES.items()}


def fc_260821_r2_row_view(stocking: RetailR2Stocking, lane: int) -> dict[int, list[tuple[str, float, str]]]:
    """Shelf2 as the camera sees it with the target in ``lane``: per lane, ``(SKU, y, prim)`` front to back."""
    visual = [
        p for p in fc_260821_stock_placements(
            omit_source_prim_names=stocking.omitted_source_prim_names,
            extra_placements=stocking.backdrop_stock,
            produce_clutter=None,
        )
        if p.group == FC_260821_STOCK_SHELF2_GROUP
    ]
    items = [(p.sku, p.position_xyz, p.source_prim_name) for p in visual]
    items += [(_R2_ASSET_SKUS[p.asset_name], p.position_xyz, p.instance_name) for p in stocking.seats[stocking.lanes.index(lane)]]
    view: dict[int, list[tuple[str, float, str]]] = {k: [] for k in FC_260821_R2_SHELF2_LANES}
    for item_sku, xyz, name in items:
        view[_lane_of(xyz[0])].append((item_sku, xyz[1], name))
    return {k: sorted(v, key=lambda item: item[1]) for k, v in view.items()}


def r2_stocking_violations(stocking: RetailR2Stocking) -> tuple[list[str], list[str]]:
    """``(uniqueness, geometry)`` problems of the row, for every lane the target can be in."""
    unique, geometry = [], []
    tolerance = FC_260821_R2_START_CLEARANCE_XYZ_M[1] + 1e-4
    for lane in stocking.lanes:
        label = fc_260821_r2_lane_label(lane)
        view = fc_260821_r2_row_view(stocking, lane)
        for other, items in view.items():
            if other != lane and any(item_sku == stocking.sku for item_sku, _, _ in items):
                names = [name for item_sku, _, name in items if item_sku == stocking.sku]
                unique.append(f"target in {label}: {fc_260821_r2_lane_label(other)} also shows {stocking.sku} ({', '.join(names)})")
            for (sku_a, y_a, name_a), (sku_b, y_b, name_b) in zip(items, items[1:]):
                need = 0.5 * (FC_260821_R2_SKU_FOOTPRINT_M[sku_a][0] + FC_260821_R2_SKU_FOOTPRINT_M[sku_b][0])
                if y_b - y_a < need - tolerance:
                    geometry.append(f"target in {label}: {name_a} and {name_b} overlap by {(need - (y_b - y_a)) * 1000:.1f} mm")
            if items and items[-1][1] + 0.5 * FC_260821_R2_SKU_FOOTPRINT_M[items[-1][0]][0] > FC_260821_R2_SHELF2_BOARD_END_Y_M:
                geometry.append(f"target in {label}: {items[-1][2]} stands past the MiniFit board's end")
        front = view[lane][:2]
        expected = [FC_260821_R2_TARGET_INSTANCE_NAME, _R2_ROLE_NAMES[1]]
        if [name for _, _, name in front] != expected or any(item_sku != stocking.sku for item_sku, _, _ in front):
            unique.append(f"target in {label}: its front two are {front}, not the target and its rear neighbor as {stocking.sku}")
    return unique, geometry


def r2_scene_violations(stocking: RetailR2Stocking) -> list[str]:
    """Visual items outside the Shelf2 row that show the target SKU, as the scene spawns them."""
    stock = fc_260821_stock_placements(
        omit_source_prim_names=stocking.stock_omissions,
        extra_placements=stocking.backdrop_stock,
        produce_clutter=None,
    )
    shown = [p.instance_name for p in stock if p.group != FC_260821_STOCK_SHELF2_GROUP and p.sku == stocking.sku]
    dressing = fc_260821_retail_v1_dressing(stocking.shelf3_raise_m, stocking.hidden_skus)
    shown += [slot.instance_name for slot in dressing.slots if slot.sku == stocking.sku]
    return [f"{name} shows {stocking.sku} outside the target lane" for name in shown]


def check_r2_stocking(stocking: RetailR2Stocking) -> RetailR2Stocking:
    """``stocking`` when its row holds; a ValueError naming what does not otherwise."""
    unique, geometry = r2_stocking_violations(stocking)
    problems = geometry + (unique + r2_scene_violations(stocking) if stocking.stock_rule == "unique_target" else [])
    if problems:
        raise ValueError(f"DB-R2 Shelf2 row for {stocking.sku} ({stocking.stock_rule}) refused:\n  - " + "\n  - ".join(problems))
    return stocking


def fc_260821_r2_stocking_for(cfg, sku: str | None = None) -> RetailR2Stocking:
    """The row an environment config builds (its ``r2_*`` fields and the Shelf3 raise); ``sku`` overrides the target."""
    return fc_260821_r2_stocking(
        sku or getattr(cfg, "r2_target_sku", None) or FC_260821_R2_DEFAULT_TARGET_SKU,
        target_lane=int(getattr(cfg, "r2_target_lane", None) or FC_260821_R2_DEFAULT_TARGET_LANE),
        lane_draw=parse_r2_lane_draw(getattr(cfg, "r2_lane_draw", None)),
        stock_rule=getattr(cfg, "r2_stock_rule", None) or FC_260821_R2_DEFAULT_STOCK_RULE,
        stock_fillers=getattr(cfg, "r2_stock_fillers", None) or FC_260821_R2_DEFAULT_STOCK_FILLERS,
        shelf3_raise_m=float(getattr(cfg, "fridge_shelf3_raise_m", 0.0) or 0.0),
    )


def fc_260821_r2_stocking_record(stocking: RetailR2Stocking) -> dict:
    """The row as JSON values, for the ``task_object`` fit record: what each lane shows, front to back."""
    unique, _geometry = r2_stocking_violations(stocking)
    elsewhere = r2_scene_violations(stocking)
    hidden = stocking.hidden_visual_stock
    return {
        "r2_target_lane": stocking.target_lane,
        "r2_stock_rule": stocking.stock_rule,
        "lanes": list(stocking.labels),
        "filler_sku": stocking.filler_sku,
        "restocked": {fc_260821_r2_lane_label(lane): f"{shown}->{filler}" for lane, shown, filler in stocking.restocked},
        "row": {
            fc_260821_r2_lane_label(lane): [item_sku for item_sku, _, _ in items]
            for lane, items in fc_260821_r2_row_view(stocking, stocking.target_lane).items()
        },
        "neighbors": list(stocking.neighbor_instance_names),
        "target_sku_only_in_target_lane": not unique,
        "target_sku_visible_only_in_target_lane": not unique and not elsewhere,
        "hidden_visual_stock": {"count": len(hidden), "items": list(hidden)},
    }


def parse_r2_lane_draw(value: str | Sequence[int] | None) -> tuple[int, ...]:
    """The lanes of a ``--r2_lane_draw`` value: ``"4,5,6"``, ``"4 5 6"`` or a sequence; ``()`` when unset."""
    if value is None:
        return ()
    tokens = value.replace(",", " ").split() if isinstance(value, str) else list(value)
    try:
        lanes = tuple(sorted({int(token) for token in tokens}))
    except ValueError:
        raise ValueError(f"--r2_lane_draw takes MiniFit lane numbers like 4,5,6; got {value!r}") from None
    unknown = [lane for lane in lanes if lane not in FC_260821_R2_SHELF2_LANE_X_M]
    if unknown:
        raise ValueError(
            f"--r2_lane_draw: Shelf2 has no lane(s) {unknown}; its lanes are {list(FC_260821_R2_SHELF2_LANES)}."
        )
    return lanes


def build_fc_260821_r2_task_assets(asset_registry, stocking: RetailR2Stocking | None = None) -> list:
    """Instantiate the DB-R2 target, watched neighbors and the two task frames."""

    from isaaclab_arena.utils.pose import Pose

    stocking = stocking if stocking is not None else fc_260821_r2_stocking()
    assets = [
        asset_registry.get_asset_by_name(placement.asset_name)(
            instance_name=placement.instance_name,
            initial_pose=Pose(
                position_xyz=placement.position_xyz,
                rotation_xyzw=placement.rotation_xyzw,
            ),
        )
        for placement in stocking.placements
    ]
    if stocking.draw:
        from dexbench_arena.variations.db_r2_lane_variation import R2LaneVariation

        assets[0].add_variation(R2LaneVariation(stocking, FC_260821_R2_TARGET_INSTANCE_NAME))
    frame_asset = asset_registry.get_asset_by_name(FC_260821_R2_TASK_FRAME_ASSET_NAME)
    identity = (0.0, 0.0, 0.0, 1.0)
    assets.append(
        frame_asset(
            instance_name=FC_260821_R2_OPENING_FRAME_INSTANCE_NAME,
            initial_pose=Pose(position_xyz=stocking.opening_frames_xyz[stocking.spawn_index], rotation_xyzw=identity),
        )
    )
    assets.append(
        frame_asset(
            instance_name=FC_260821_R2_DESTINATION_FRAME_INSTANCE_NAME,
            initial_pose=Pose(position_xyz=FC_260821_R2_DESTINATION_FRAME_XYZ_M, rotation_xyzw=identity),
        )
    )
    return assets


__all__ = [
    "RetailFixturePlacement",
    "RetailR2ItemPlacement",
    "RetailR2LaneStock",
    "RetailR2Stocking",
    "FC_260821_R2_OBJECT_AXIS",
    "build_fc_260821_r2_task_assets",
    "check_fridge_shelf3_raise",
    "check_r2_target_sku",
    "check_r2_stocking",
    "r2_scene_violations",
    "fc_260821_r2_stocking",
    "fc_260821_r2_stocking_for",
    "fc_260821_r2_stocking_record",
    "fc_260821_r2_pack_slot",
    "fc_260821_r2_source_row_gaps_m",
    "parse_r2_lane_draw",
]
