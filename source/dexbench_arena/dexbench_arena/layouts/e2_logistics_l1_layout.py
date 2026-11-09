# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Where DB-L1's belt, robot, tote and item stream go in the E2 cell, and the numbers its stream and judge use."""

from __future__ import annotations

from dexbench_arena.assets.e2_logistics_cell import (
    E2_SORT_TOTE_SIZE_XYZ_M,
    E2_SORT_TOTE_SLOT_XYZ_M,
    E2_SORT_TOTE_WALL_M,
)
from dexbench_arena.assets.polybag.sort_variants import SORT_VARIANTS_BY_KEY, VARIANT_PACKAGE
from dexbench_arena.embodiments.rby1m.constants import rby1m_torso_z_fold
from dexbench_arena.tasks.conveyor_placement_config import (
    CONVEYOR_PLACE_XY_RECTS_M,
    CONVEYOR_SET_DOWN_TOP_Z_M,
    CONVEYOR_SET_DOWN_XY_RECT_M,
)
from dexbench_arena.tasks.object_axis import ObjectAxis, TaskObject


L1_BELTS = ("HB2",)
L1_BELT_X_RANGE_M = (CONVEYOR_PLACE_XY_RECTS_M[0][0][0], CONVEYOR_PLACE_XY_RECTS_M[0][1][0])
L1_BELT_CLEAR_Y_RANGE_M = (CONVEYOR_SET_DOWN_XY_RECT_M[0][1], CONVEYOR_SET_DOWN_XY_RECT_M[1][1])
L1_BELT_TOP_Z_M = CONVEYOR_SET_DOWN_TOP_Z_M
L1_BELT_SPEED_M_S = 0.2
L1_BELT_UV_PER_METRE = 10.0 / 3.0


L1_ROBOT_BASE_ANCHORS_XYZ = {"A_L1": (2.70, 1.60, 0.0)}
L1_ANCHOR_YAWS_DEG = {"A_L1": 180.0}

L1_TORSO_PITCH_RAD = 0.60
L1_VIEW_EYE_LOOKAT = ((2.2, 0.15, 2.0), (2.6, 1.5, 0.85))
L1_RBY1_START_JOINT_POS = {
    "torso_0": 0.0,
    **rby1m_torso_z_fold(L1_TORSO_PITCH_RAD),
    "torso_4": 0.0,
    "torso_5": 0.0,
    "left_arm_0": -0.6834,
    "left_arm_1": 0.0825,
    "left_arm_2": 0.2122,
    "left_arm_3": -1.2333,
    "left_arm_4": -0.0071,
    "left_arm_5": 0.3165,
    "left_arm_6": 0.0129,
    "right_arm_0": -0.6834,
    "right_arm_1": -0.0825,
    "right_arm_2": -0.2122,
    "right_arm_3": -1.2333,
    "right_arm_4": 0.0071,
    "right_arm_5": 0.3165,
    "right_arm_6": -0.0129,
    "head_0": 0.8,
    "head_1": 0.6,
}


L1_TOTE_XY_M = E2_SORT_TOTE_SLOT_XYZ_M[:2]
L1_TOTE_CATCH_DEPTH_M = 0.12


def l1_tote_footprint() -> tuple[tuple[float, float], tuple[float, float]]:
    """((min x, min y), (max x, max y)) of the tote's outside, long side along x."""

    (x, y, _), (sx, sy, _) = E2_SORT_TOTE_SLOT_XYZ_M, E2_SORT_TOTE_SIZE_XYZ_M
    return (x - sx / 2.0, y - sy / 2.0), (x + sx / 2.0, y + sy / 2.0)


def l1_tote_inner_box() -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """World (min, max) corners of the tote's clear inside, floor top to rim."""

    (x0, y0), (x1, y1) = l1_tote_footprint()
    z, wall, rim = E2_SORT_TOTE_SLOT_XYZ_M[2], E2_SORT_TOTE_WALL_M, E2_SORT_TOTE_SIZE_XYZ_M[2]
    return (x0 + wall, y0 + wall, z + wall), (x1 - wall, y1 - wall, z + rim)


def l1_tote_catch_box() -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Centre and world-axis size of the invisible catch box under the tote's floor: the tote's footprint, its top
    flush with the floor's."""

    (x0, y0), (x1, y1) = l1_tote_footprint()
    top = l1_tote_inner_box()[0][2]
    centre = ((x0 + x1) / 2.0, (y0 + y1) / 2.0, top - L1_TOTE_CATCH_DEPTH_M / 2.0)
    return centre, (x1 - x0, y1 - y0, L1_TOTE_CATCH_DEPTH_M)


L1_STREAM_BAGS = 3
L1_STREAM_BOXES = 2
L1_STREAM_MAX_BAGS = 4
L1_STREAM_MAX_BOXES = 4
L1_STREAM_LEADS = ("bag", "box")
L1_STREAM_MODES = ("episode", "continuous")
L1_ENTRY_EDGE_X_M = 1.09
L1_POLYBAG_SPAWN_CLEARANCE_M = 0.006
L1_BOX_SPAWN_CLEARANCE_M = 0.005
L1_ITEM_Y_RANGE_M = (0.80, 0.95)
L1_BELT_EDGE_MARGIN_M = 0.01
L1_SPACING_RANGE_M = (0.72, 0.88)
L1_SPAWN_CLEARANCE_M = 0.15
L1_YAW_OFF_BELT_DEG = {"train": (0.0, 30.0), "ga_s": (60.0, 90.0)}
L1_BOX_YAW_OFF_BELT_DEG = L1_YAW_OFF_BELT_DEG["train"]
L1_PICK_ZONE_X_M = 2.20
L1_MISS_LINE_X_M = 2.90
L1_PLACED_MAX_SPEED_M_S = 0.10
L1_PLACED_HOLD_S = 0.5
L1_PLACED_MIN_FRACTION = 0.5
L1_DROPPED_Z_M = 0.45
L1_MISSED_ABOVE_BELT_M = 0.05
L1_BOX_DISTURBED_ACROSS_M = 0.02
L1_BOX_DISTURBED_TURNED_DEG = 5.0
L1_BOX_DISTURBED_LIFTED_M = 0.01
L1_EPISODE_LENGTH_S = 600.0
L1_CONTINUOUS_EPISODE_LENGTH_S = 3600.0
L1_GRASP_STYLES = ("palm_down_squeeze", "edge_pinch", "side_corner")


L1_BAG_SETS = {
    "train_a": ("silver_30x40__1a", "grey_25x35__1a", "black_25x35__2b", "pink_20x30__3c"),
    "train_b": ("silver_30x40__2b", "black_25x35__1a", "white_25x35__2b", "grey_20x30__3c"),
    "train_c": ("black_30x40__3c", "grey_25x35__2b", "white_25x35__1a", "pink_20x30__1a"),
    "train_d": ("black_30x40__1a", "grey_25x35__3c", "white_25x35__3c", "grey_20x30__2b"),
    "eval_a": ("grey_35x45__1a", "black_20x40__3c", "grey_35x45__2b", "black_20x40__1a"),
    "eval_b": ("grey_35x45__3c", "black_20x40__2b", "grey_35x45__1a", "black_20x40__3c"),
    "eval_c": ("purple_25x35__1a", "skyblue_20x30__2b", "purple_25x35__2b", "skyblue_20x30__1a"),
    "eval_d": ("purple_25x35__3c", "skyblue_20x30__3c", "purple_25x35__1a", "skyblue_20x30__2b"),
}
L1_GA_O_HELD_OUT = {
    "colour": ("eval_c", "eval_d"),
    "size": ("eval_a", "eval_b"),
    "profile": (),
}
L1_DEFAULT_BAG_SET = "train_a"
L1_STREAM_BOX_VARIANTS = ("krpost1_1__1a", "krpost2_1__2b", "krpost3_1__3c", "krpost2_2__1a")
_BODY_SPLIT_TO_SET_SPLIT = {"train": "train", "eval_colour": "eval", "eval_shape": "eval"}


def l1_bag_set_split(key: str) -> str:
    """``train`` or ``eval``: the split every bag of set ``key`` is of; ValueError for an unknown or a mixed set."""

    if key not in L1_BAG_SETS:
        raise ValueError(f"l1_bag_set {key!r}: one of {tuple(L1_BAG_SETS)}")
    splits = {_BODY_SPLIT_TO_SET_SPLIT[SORT_VARIANTS_BY_KEY[v].split] for v in L1_BAG_SETS[key]}
    if len(splits) != 1:
        raise ValueError(f"l1_bag_set {key}: mixes training and held-out bags")
    return splits.pop()


def _bag_set_size(key: str) -> tuple[float, float, float]:
    variants = [SORT_VARIANTS_BY_KEY[v] for v in L1_BAG_SETS[key]]
    return (max(v.body.footprint_m[0] for v in variants), max(v.body.footprint_m[1] for v in variants),
            max(v.thickness_m for v in variants))


def _bag_set_fit(task_object: TaskObject, cfg) -> dict:
    """The set's bags, in the order the stream's bag entities take them (``l1_bag_<i>``)."""

    del cfg
    return {"bag_variants": list(L1_BAG_SETS[task_object.key])}


L1_BAG_SET_AXIS = ObjectAxis(
    field="l1_bag_set",
    default=L1_DEFAULT_BAG_SET,
    objects=tuple(
        TaskObject(key=key, asset=VARIANT_PACKAGE, split=l1_bag_set_split(key), label="polybags",
                   size_m=_bag_set_size(key))
        for key in L1_BAG_SETS
    ),
    size_axes="the set's widest and longest bag footprint, its thickest bag",
    fit=_bag_set_fit,
    scene_layout="261014",
    unlabelled_scene_layout="260827",
)



L1_POLYBAG_STEPPINGS = {"a": (8, 5, 8), "b": (16, 5, 16)}


L1_PARK_SLAB_TOP_Z_M = -0.60
L1_PARK_SLOT_X_M = (1.70, 2.20, 2.70, 3.20)
L1_PARK_ROW_Y_M = (1.60, 2.15)
L1_PARK_SLAB_SIZE_M = (2.10, 1.10, 0.10)
L1_PARK_SLAB_CENTER_M = (
    (L1_PARK_SLOT_X_M[0] + L1_PARK_SLOT_X_M[-1]) / 2.0,
    (L1_PARK_ROW_Y_M[0] + L1_PARK_ROW_Y_M[1]) / 2.0,
    L1_PARK_SLAB_TOP_Z_M - L1_PARK_SLAB_SIZE_M[2] / 2.0,
)
L1_PARK_SLOTS = len(L1_PARK_SLOT_X_M) * len(L1_PARK_ROW_Y_M)


def l1_park_xy(slot: int) -> tuple[float, float]:
    """Where item ``slot`` (0-7) of an episode waits: slots 0-3 on the row nearer the belt, 4-7 on the other."""

    if not 0 <= slot < L1_PARK_SLOTS:
        raise ValueError(f"park slot {slot} is not one of 0..{L1_PARK_SLOTS - 1}")
    row, column = divmod(slot, len(L1_PARK_SLOT_X_M))
    return (L1_PARK_SLOT_X_M[column], L1_PARK_ROW_Y_M[row])
