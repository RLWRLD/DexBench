# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Where DB-L4's belt, robot and parcels go in the E2 cell, and how a parcel can be put down."""

from __future__ import annotations

import math

from dexbench_arena.assets.e2_logistics_cell import E2_L4_PARCELS_BY_NAME


L4_PLATE_X_RANGE_M = (3.0045, 3.8325)
L4_PLATE_Y_RANGE_M = (0.35, 1.25)
L4_PLATE_TOP_Z_M = 0.821
L4_BELT_X_RANGE_M = (3.3165, 4.1925)
L4_BELT_Y_RANGE_M = (0.425, 1.175)
L4_BELT_TOP_Z_M = 0.800
L4_BELT_UV_PER_METRE = 10.0 / 3.0
L4_HB2_BELT_X_RANGE_M = (1.0745, 2.9545)
L4_HB2_BELT_Y_RANGE_M = (0.425, 1.175)
L4_HB2_BELT_TOP_Z_M = 0.812

L4_PLATE_SPOT_XY_M = (3.35, 1.00)
L4_SPAWN_XY_M = (2.195, 0.925)
L4_SPAWN_SURFACE_Z_M = L4_HB2_BELT_TOP_Z_M
L4_BOX_SPAWN_CLEARANCE_M = 0.005
L4_POLYBAG_SPAWN_CLEARANCE_M = 0.006
L4_JUDGE_LINE_X_M = 4.10

L4_ROBOT_BASE_ANCHORS_XYZ = {"A_L4": (3.191, 1.640, 0.0)}
L4_ANCHOR_YAWS_DEG = {"A_L4": -90.0}
L4_RBY1_START_JOINT_POS = {
    "torso_0": 0.0,
    "torso_1": 1.431877,
    "torso_2": -2.096101,
    "torso_3": 0.664224,
    "torso_4": 0.0,
    "torso_5": 0.0,
    "right_arm_0": -0.87099,
    "right_arm_1": -1.780335,
    "right_arm_2": 1.48381,
    "right_arm_3": -2.464465,
    "right_arm_4": 0.630876,
    "right_arm_5": 1.093394,
    "right_arm_6": -2.327197,
    "left_arm_0": -0.87099,
    "left_arm_1": 1.780335,
    "left_arm_2": -1.48381,
    "left_arm_3": -2.464465,
    "left_arm_4": -0.630876,
    "left_arm_5": 1.093394,
    "left_arm_6": 2.327197,
    "head_0": 0.0,
    "head_1": 0.40,
}

L4_PARK_SLAB_TOP_Z_M = -0.60
L4_PARK_SLAB_SIZE_M = (2.80, 0.80, 0.10)
L4_PARK_SLAB_CENTER_M = (
    (L4_PLATE_X_RANGE_M[0] + L4_BELT_X_RANGE_M[1]) / 2.0,
    (L4_BELT_Y_RANGE_M[0] + L4_BELT_Y_RANGE_M[1]) / 2.0,
    L4_PARK_SLAB_TOP_Z_M - 0.05,
)
L4_PARK_SPACING_M = 0.40
L4_PARCELS_PER_EPISODE = 5


def l4_parcel_entity_name(slot: int) -> str:
    """Scene entity of the parcel in ``slot``: by slot, since a sequence can name a parcel twice."""

    return f"l4_parcel_{slot:02d}"


def l4_park_xy(slot: int) -> tuple[float, float]:
    """Where parcel ``slot`` of an episode waits, centred over the slab."""

    if not 0 <= slot < L4_PARCELS_PER_EPISODE:
        raise ValueError(f"park slot {slot} is not one of 0..{L4_PARCELS_PER_EPISODE - 1}")
    offset = (slot - (L4_PARCELS_PER_EPISODE - 1) / 2.0) * L4_PARK_SPACING_M
    return (L4_PARK_SLAB_CENTER_M[0] + offset, L4_PARK_SLAB_CENTER_M[1])


SEQUENCE_INTERVAL_S = 1.0
PARCEL_TIMEOUT_S = 20.0
MAX_BARCODE_ANGLE_DEG = 60.0
SETTLE_EPS_M = 0.010
JUDGE_DWELL_S = 0.1
CARRY_FRACTION = 0.5
FALL_CLEARANCE_M = 0.10
EPISODE_LENGTH_S = 105.0



STATION_AXIS_BY_BARCODE_ORIENTATION = {
    "up": (0, 0, 1),
    "down": (0, 0, -1),
    "left": (0, 1, 0),
    "right": (0, -1, 0),
    "far": (1, 0, 0),
    "near": (-1, 0, 0),
}
AXIS_BY_NAME = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0), "+z": (0, 0, 1), "-z": (0, 0, -1)}
STATION_TO_WORLD = ((0, 1, 0), (-1, 0, 0), (0, 0, 1))

Matrix = tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def mat_vec(matrix, vector) -> tuple:
    """Row-major 3x3 matrix times a 3-vector."""

    return tuple(sum(matrix[i][j] * vector[j] for j in range(3)) for i in range(3))


def _mat_mul(a, b) -> Matrix:
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)) for i in range(3))


def _axis_aligned_rotations() -> tuple[Matrix, ...]:
    """The 24 proper axis-aligned rotations, in a fixed order."""

    axes = tuple(AXIS_BY_NAME[key] for key in ("+x", "-x", "+y", "-y", "+z", "-z"))
    rotations = []
    for c0 in axes:
        for c1 in axes:
            if sum(c0[i] * c1[i] for i in range(3)) != 0:
                continue
            c2 = _cross(c0, c1)
            rotations.append(tuple((c0[i], c1[i], c2[i]) for i in range(3)))
    return tuple(rotations)


AXIS_ALIGNED_ROTATIONS = _axis_aligned_rotations()


def _lowest_two(size_xyz_m: tuple[float, float, float], candidates: list[Matrix]) -> list[Matrix]:
    """The two of ``candidates`` that leave a box standing on its origin lowest (the source's rule)."""

    centre = (0.0, 0.0, size_xyz_m[2] / 2.0)
    half = tuple(extent / 2.0 for extent in size_xyz_m)

    def height(rotation: Matrix) -> float:
        return sum(abs(rotation[2][j]) * half[j] for j in range(3)) - mat_vec(rotation, centre)[2]

    return sorted(candidates, key=height)[:2]


def put_down_candidates(name: str, orientation: str) -> list[Matrix]:
    """The axis-aligned rotations (station frame) parcel ``name`` can be put down at with its barcode facing
    ``orientation``, in a fixed order."""

    parcel = E2_L4_PARCELS_BY_NAME[name]
    barcode_axis = AXIS_BY_NAME[parcel.barcode_axis]
    station_axis = STATION_AXIS_BY_BARCODE_ORIENTATION[orientation]
    candidates = [r for r in AXIS_ALIGNED_ROTATIONS if mat_vec(r, barcode_axis) == station_axis]
    if parcel.is_polybag or orientation in ("up", "down"):
        return candidates
    return _lowest_two(parcel.collider_size_xyz_m, candidates)


OOD_YAW_DEG = 45.0
ROTATION_DOMAINS = ("id", "ood")
MASS_MULTIPLIERS = {
    "id": tuple(2.0 ** (k / 2.0) for k in (-2, -1, 0, 1, 2)),
    "ood": tuple(2.0 ** (k / 2.0) for k in (-4, -3, 3, 4)),
}
_DIAGONAL = math.sqrt(0.5)
DIRECTIONS = {
    **{name: tuple(float(v) for v in axis) for name, axis in STATION_AXIS_BY_BARCODE_ORIENTATION.items()},
    "diag_45": (_DIAGONAL, _DIAGONAL, 0.0),
    "diag_135": (-_DIAGONAL, _DIAGONAL, 0.0),
    "diag_225": (-_DIAGONAL, -_DIAGONAL, 0.0),
    "diag_315": (_DIAGONAL, -_DIAGONAL, 0.0),
}
_TURNED_OOD = {"far": "diag_45", "left": "diag_135", "near": "diag_225", "right": "diag_315", "up": "up", "down": "down"}
ROTATION_LABELS = tuple(
    f"{barcode}/{x}"
    for barcode, b in DIRECTIONS.items()
    for x, v in DIRECTIONS.items()
    if abs(sum(p * q for p, q in zip(b, v))) < 1e-9
)


def _direction_name(vector) -> str:
    for name, axis in DIRECTIONS.items():
        if all(abs(float(p) - q) < 1e-9 for p, q in zip(vector, axis)):
            return name
    raise ValueError(f"{tuple(vector)} is none of DIRECTIONS")


def rotation_label(name: str, rotation) -> str:
    """The label of parcel ``name`` put down at ``rotation`` (station frame, row-major)."""

    barcode = mat_vec(rotation, AXIS_BY_NAME[E2_L4_PARCELS_BY_NAME[name].barcode_axis])
    return f"{_direction_name(barcode)}/{_direction_name(mat_vec(rotation, (1, 0, 0)))}"


def rotation_labels(name: str, domain: str = "id") -> tuple[str, ...]:
    """Parcel ``name``'s put-down rotations in ``domain`` (``id`` or ``ood``), in a fixed order."""

    if domain not in ROTATION_DOMAINS:
        raise ValueError(f"rotation domain {domain!r} is not one of {ROTATION_DOMAINS}")
    faces = ("up", "down") if E2_L4_PARCELS_BY_NAME[name].is_polybag else tuple(STATION_AXIS_BY_BARCODE_ORIENTATION)
    labels = tuple(rotation_label(name, r) for orientation in faces for r in put_down_candidates(name, orientation))
    if domain == "ood":
        labels = tuple("/".join(_TURNED_OOD[part] for part in label.split("/")) for label in labels)
    return labels


def rotation_domain(name: str, label: str) -> str:
    """Which domain ``label`` is in for parcel ``name``; ValueError for a rotation it is never put down at."""

    for domain in ROTATION_DOMAINS:
        if label in rotation_labels(name, domain):
            return domain
    raise ValueError(f"{label!r} is not a put-down rotation of {name}; its in-domain ones are {rotation_labels(name)}")


def station_rotation(name: str, label: str) -> tuple[tuple[float, float, float], ...]:
    """The rotation (station frame, row-major) that puts parcel ``name`` down as ``label`` says."""

    barcode_name, x_name = label.split("/")
    barcode, x = DIRECTIONS[barcode_name], DIRECTIONS[x_name]
    if abs(sum(p * q for p, q in zip(barcode, x))) > 1e-9:
        raise ValueError(f"{label!r}: the barcode and +x directions are not perpendicular")
    sign = AXIS_BY_NAME[E2_L4_PARCELS_BY_NAME[name].barcode_axis][2]
    z = tuple(sign * v for v in barcode)
    y = _cross(z, x)
    return tuple((x[i], y[i], z[i]) for i in range(3))


def world_rotation(name: str, label: str) -> tuple[tuple[float, float, float], ...]:
    """``station_rotation`` in the world frame."""

    return _mat_mul(STATION_TO_WORLD, station_rotation(name, label))


IDENTITY: Matrix = ((1, 0, 0), (0, 1, 0), (0, 0, 1))


def box_bounds(size_xyz_m: tuple[float, float, float]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """A box parcel's bounds in its own frame: centred in XY on its origin, standing on it."""

    sx, sy, sz = size_xyz_m
    return (-sx / 2.0, -sy / 2.0, 0.0), (sx / 2.0, sy / 2.0, sz)


def resting_position(
    rotation: Matrix, local_min, local_max, xy: tuple[float, float], surface_z: float, clearance: float
) -> tuple[float, float, float]:
    """Where a body's origin goes so that its bounds, turned by ``rotation``, are centred over ``xy`` and
    start ``clearance`` above ``surface_z``. ``local_min``/``local_max`` are its bounds in its own frame."""

    centre = tuple((local_min[i] + local_max[i]) / 2.0 for i in range(3))
    half = tuple((local_max[i] - local_min[i]) / 2.0 for i in range(3))
    rotated_centre = mat_vec(rotation, centre)
    rotated_half_z = sum(abs(rotation[2][j]) * half[j] for j in range(3))
    return (
        xy[0] - rotated_centre[0],
        xy[1] - rotated_centre[1],
        surface_z + clearance + rotated_half_z - rotated_centre[2],
    )

