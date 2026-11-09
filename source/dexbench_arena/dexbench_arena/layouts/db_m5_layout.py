"""The DB-M5 layout and splits: where the robot, the belt, the kit box and the parts bin stand, the compartments
the task judges, and the parts, compartments and piles a Run draws from. The environment
(``e3_manufacturing_db_m5_environment``), its task and its tools share them."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from dexbench_arena.assets.assembly_line_e3_m5_station import SLOT_PARTS_BIN_XYZ, STAGING_SHELF_BIN_YAW_DEG
from dexbench_arena.layouts.db_m5_parts import DEFAULT_PART, M5_PARTS, get_m5_part
from dexbench_arena.tasks.object_axis import ObjectAxis, TaskObject


M5_ANCHOR = "A_m5"
ROBOT_BASE_ANCHORS_XYZ = {
    M5_ANCHOR: (3.2, 2.374, 0.0),
}
A_M5_BASE_YAW_DEG = 90.0
ANCHOR_FACING_TARGETS_XY = {
    M5_ANCHOR: (ROBOT_BASE_ANCHORS_XYZ[M5_ANCHOR][0], ROBOT_BASE_ANCHORS_XYZ[M5_ANCHOR][1] + 1.0),
}

ANCHOR_MOUNT_Z_M = 0.0

RBY1_A_M5_BASE_XYZ = (3.5, 2.65, 0.0)



CONVEYOR_BELT_END_X_M = 4.3
CONVEYOR_TRAVEL_DIRECTION_W = (1.0, 0.0, 0.0)
CONVEYOR_TOP_Z_M = 0.750
CONVEYOR_FRAME_S_BELT_FACE_Y_M = 3.047
CONVEYOR_FRAME_N_BELT_FACE_Y_M = 3.847


KIT_BOX_NAME = "belt_kit_box"
"""The kit box's scene-entity name: the task's ``kit_box_name``, and the asset a recorded draw is filed under
(``variations/belt_kit_box/empty_compartment``)."""
KIT_BOX_ON_BELT_XYZ = (1.80000001, 3.20449996, CONVEYOR_TOP_Z_M)

KIT_BOX_COMPARTMENT_XY_OFFSET = (0.087, 0.065)
KIT_BOX_COMPARTMENT_FLOOR_TOP_Z = 0.008

KIT_BOX_COMPARTMENTS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
DEFAULT_EMPTY_COMPARTMENT = 3



def kit_box_part_poses(
    part: str = DEFAULT_PART,
    empty_compartment: int = DEFAULT_EMPTY_COMPARTMENT,
) -> tuple[tuple[tuple[float, float, float], tuple[float, float, float, float]], ...]:
    """World ``(position_xyz, rotation_xyzw)`` of each loaded compartment's part, resting on its floor."""

    if empty_compartment not in range(len(KIT_BOX_COMPARTMENTS)):
        raise ValueError(f"kit-box compartment {empty_compartment!r}; the box has 0..{len(KIT_BOX_COMPARTMENTS) - 1}")
    staged = get_m5_part(part)
    poses = []
    for index, (sign_x, sign_y) in enumerate(KIT_BOX_COMPARTMENTS):
        if index == empty_compartment:
            continue
        dx, dy = sign_x * KIT_BOX_COMPARTMENT_XY_OFFSET[0], sign_y * KIT_BOX_COMPARTMENT_XY_OFFSET[1]
        (ox, oy, oz), rotation_xyzw = staged.kit_box_pose_local(dx)
        position_xyz = (
            KIT_BOX_ON_BELT_XYZ[0] + dx + ox,
            KIT_BOX_ON_BELT_XYZ[1] + dy + oy,
            KIT_BOX_ON_BELT_XYZ[2] + KIT_BOX_COMPARTMENT_FLOOR_TOP_Z + oz,
        )
        poses.append((position_xyz, rotation_xyzw))
    return tuple(poses)


PARTS_BIN_FLOOR_TOP_Z_LOCAL = 0.01
PARTS_BIN_INTERIOR_HALF_XY = (0.215, 0.125)

PARTS_BIN_CLUTTER_POSES_BIN_LOCAL: tuple[tuple[float, float, float, float, float, float, float], ...] = (
    (0.10200639814138412, -0.023917214944958687, 0.06737133860588074, 0.693011462688446, -0.1386236995458603, 0.6935340166091919, 0.1397472620010376),
    (0.13986189663410187, 0.02876129373908043, 0.010269210673868656, -0.27137890458106995, 0.17640306055545807, -0.3980494737625122, 0.8583658933639526),
    (-0.042774613946676254, -0.08197099715471268, 0.024028610438108444, -0.5793405771255493, 0.31643274426460266, -0.300991415977478, 0.6882144212722778),
    (-0.13966573774814606, -0.049993399530649185, 0.007571875117719173, -0.010271323844790459, 0.1994185894727707, 0.9653708934783936, 0.16788645088672638),
    (-0.08199410885572433, 0.08603838831186295, 0.007594105787575245, 0.06111055240035057, -0.19075734913349152, -0.9760741591453552, 0.08459629863500595),
    (0.03795907273888588, -0.005109655670821667, 0.03928597271442413, 0.43019360303878784, 0.7858003377914429, -0.41074785590171814, -0.1695217788219452),
    (0.07618482410907745, 0.0577932745218277, 0.0075879753567278385, -0.10217112302780151, 0.1723063588142395, -0.7111797332763672, 0.673865795135498),
    (-0.02854122593998909, 0.0048958128318190575, 0.0472383014857769, -0.5347503423690796, -0.8414244651794434, -0.030971059575676918, -0.07132826745510101),
)


def _turned_about_z(xyz, yaw_rad: float) -> tuple[float, float, float]:
    """``xyz`` turned by ``yaw_rad`` about +z, in float64."""

    import torch
    from isaaclab_arena.utils.yaw import rotate_points_by_yaw

    ((x, y, z),) = rotate_points_by_yaw(torch.tensor([[float(v) for v in xyz]], dtype=torch.float64), yaw_rad).tolist()
    return x, y, z


def parts_bin_local_to_world(xyz) -> tuple[float, float, float]:
    """A point in the bin's own frame (parts_bin_fc's pivot), in world."""

    turned = _turned_about_z(xyz, math.radians(STAGING_SHELF_BIN_YAW_DEG))
    return tuple(o + v for o, v in zip(SLOT_PARTS_BIN_XYZ, turned))


def world_to_parts_bin_local(xyz) -> tuple[float, float, float]:
    """Inverse of ``parts_bin_local_to_world``."""

    offset = tuple(float(v) - o for v, o in zip(xyz, SLOT_PARTS_BIN_XYZ))
    return _turned_about_z(offset, -math.radians(STAGING_SHELF_BIN_YAW_DEG))


def parts_bin_part_poses_world(
    poses_bin_local=PARTS_BIN_CLUTTER_POSES_BIN_LOCAL,
) -> tuple[tuple[tuple[float, float, float], tuple[float, float, float, float]], ...]:
    """World ``(position_xyz, rotation_xyzw)`` of each bin part, from its ``(px, py, pz, qx, qy, qz, qw)`` in the bin."""

    from isaaclab_arena.utils.yaw import rotate_quat_by_yaw

    yaw_rad = math.radians(STAGING_SHELF_BIN_YAW_DEG)
    return tuple(
        (parts_bin_local_to_world(pose[:3]), rotate_quat_by_yaw(tuple(pose[3:]), yaw_rad)) for pose in poses_bin_local
    )


KIT_BOX_INTERIOR_HALF_XY = (0.172, 0.125)
KIT_BOX_DIVIDER_HALF_XY = (0.002, 0.005)


def kit_box_compartment_rectangle_local(index: int) -> tuple[tuple[float, float], tuple[float, float]]:
    """``(min_xy, max_xy)`` of ``KIT_BOX_COMPARTMENTS[index]`` in the kit box's frame."""

    def span(sign: int, interior: float, divider: float) -> tuple[float, float]:
        return (divider, interior) if sign > 0 else (-interior, -divider)

    sign_x, sign_y = KIT_BOX_COMPARTMENTS[index]
    (x0, x1), (y0, y1) = (
        span(sign_x, KIT_BOX_INTERIOR_HALF_XY[0], KIT_BOX_DIVIDER_HALF_XY[0]),
        span(sign_y, KIT_BOX_INTERIOR_HALF_XY[1], KIT_BOX_DIVIDER_HALF_XY[1]),
    )
    return (x0, y0), (x1, y1)


KIT_BOX_COMPARTMENTS_MIN_XY_LOCAL, KIT_BOX_COMPARTMENTS_MAX_XY_LOCAL = (
    tuple(corners)
    for corners in zip(*(kit_box_compartment_rectangle_local(index) for index in range(len(KIT_BOX_COMPARTMENTS))))
)
KIT_BOX_EMPTY_COMPARTMENT_MIN_XY_LOCAL, KIT_BOX_EMPTY_COMPARTMENT_MAX_XY_LOCAL = kit_box_compartment_rectangle_local(
    DEFAULT_EMPTY_COMPARTMENT
)
KIT_BOX_LEADING_EDGE_LOCAL_X = 0.185
KIT_BOX_OUTER_HALF_XY = (0.185, 0.1375)
KIT_BOX_MAX_BELT_GAP_M = 0.01



M5_PART_SPLIT: dict[str, str] = {
    "bolt_m24": "train",
    "pom_roller": "train",
    "elbow_50a": "train",
    "bearing_6206": "train",
    "nema17": "eval",
    "boltkorea_iso4014_class_8_8_hex_bolt_m36x160": "eval",
    "tyb_ucp205_pillow_block_bearing_unit": "eval",
    "rs_pro_stepper_motor_535_0502_nema23_class": "train",
    "automotive_exhaust_90_bend_od63": "eval",
    "jl_sus304_welded_tee_50a": "eval",
    "v_belt_pulley_1a_70_12_4": "eval",
    "gy1_rigid_flange_coupling_half_flange": "eval",
    "two_piece_split_shaft_collar_od45_bore2_w12": "eval",
    "deco_ad02_2_station_ng6cetop3_manifold": "eval",
    "norgren_rt_57232_m_50_roundline_cylinder": "eval",
}
"""``train`` or ``eval`` per ``db_m5_parts`` key; every part exactly once."""

M5_PART_AXIS = ObjectAxis(
    field="part",
    default=DEFAULT_PART,
    objects=tuple(
        TaskObject(
            key=part.key,
            asset=part.asset_name,
            split=M5_PART_SPLIT[part.key],
            label=part.label,
            size_m=part.extents_local,
        )
        for part in M5_PARTS.values()
    ),
    size_axes="collider box along the root frame's x, y, z",
)


M5_SPATIAL_RANGES: dict[str, dict[str, list]] = {
    "training": {"kit_box_empty_compartments": [0, 3]},
    "beyond": {"kit_box_empty_compartments": [1, 2]},
}
"""The environment field a Run sets for each spatial range: ``training`` for the demos, Train and GA-O,
``beyond`` for GA-S and GA-OS. ``kit_box_empty_compartments`` indexes ``KIT_BOX_COMPARTMENTS`` above."""

M5_PILE_SEED_QUANTILES: list[float] = [0.0, 0.6]
"""The ``parts_bin_seed_quantiles`` every Run sets, the demos and every cell alike: the first 24 of a part's 40
seeds by rank, one pile per demo of a training pair (240 demos over 5 parts and 2 compartments). The last 16
are drawn by nothing."""


def seeds_in_quantiles(available: Iterable[int], quantiles: Sequence[float]) -> tuple[int, ...]:
    """The seeds of ``available`` (one part's dbclutter seeds) whose rank lies in ``quantiles``, sorted."""

    bounds = tuple(float(value) for value in quantiles)
    if len(bounds) != 2 or not 0.0 <= bounds[0] < bounds[1] <= 1.0:
        raise ValueError(f"parts_bin_seed_quantiles {list(quantiles)}: give [low, high) with 0 <= low < high <= 1")
    ordered = sorted(int(seed) for seed in available)
    low, high = (round(bound * len(ordered)) for bound in bounds)
    return tuple(ordered[low:high])
