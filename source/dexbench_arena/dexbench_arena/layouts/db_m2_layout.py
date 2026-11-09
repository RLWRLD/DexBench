# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""The DB-M2 layout and connector geometry, shared by the environment, its task and its tools."""

from __future__ import annotations

import math
from dataclasses import dataclass

from dexbench_arena.tasks.object_axis import ObjectAxis, TaskObject


LINE_TOP_Z_M = 0.20

LINER_ROOT_XYZ = (0.0, -0.02, LINE_TOP_Z_M)
"""Where the package root goes: the liner's underside rests on the belt top."""

LINE_SIZE_M = (2.4, 0.91, LINE_TOP_Z_M)
LINE_CENTER_XYZ = (LINER_ROOT_XYZ[0], LINER_ROOT_XYZ[1], LINE_TOP_Z_M / 2)
LINE_NEAR_EDGE_Y_M = LINE_CENTER_XYZ[1] - LINE_SIZE_M[1] / 2



@dataclass(frozen=True)
class ConnectorSite:
    """One connector of the package: its plug prim and its two task frames, in the package frame."""

    key: str
    plug_prim: str
    """The plug's articulation root, relative to the package root."""
    seated_xyz: tuple[float, float, float]
    """``Frames/<key>_Seated``: the plug origin mated (where the package file has it)."""
    unplugged_xyz: tuple[float, float, float]
    """``Frames/<key>_Unplugged``: the plug origin pulled out by the CAD mating gap."""
    seated_quat_xyzw: tuple[float, float, float, float]
    """The plug body's orientation mated, in the package frame."""
    label: str

    @property
    def stroke_m(self) -> float:
        """The mating gap: seated to unplugged."""
        return math.dist(self.seated_xyz, self.unplugged_xyz)

    @property
    def pull_out_axis(self) -> tuple[float, float, float]:
        """Unit vector from seated toward unplugged; the insertion runs against it."""
        stroke = self.stroke_m
        return tuple((u - s) / stroke for s, u in zip(self.seated_xyz, self.unplugged_xyz))


CONNECTOR_SITES = {
    "S1": ConnectorSite(
        key="S1",
        plug_prim="Task/Connector_S1",
        seated_xyz=(0.263107, -0.316413, 0.801550),
        unplugged_xyz=(0.278391, -0.318945, 0.812375),
        seated_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
        label="KET MG: harness connector into its mate (fridge fan motor)",
    ),
    "S2": ConnectorSite(
        key="S2",
        plug_prim="Task/Connector_S2",
        seated_xyz=(0.251798, -0.288963, 0.798261),
        unplugged_xyz=(0.259882, -0.289158, 0.806437),
        seated_quat_xyzw=(0.0, 0.0, 0.0, 1.0),
        label="JST YL 16-pin: harness connector into its mate (fridge fan / sensor harness)",
    ),
}
DEFAULT_CONNECTOR = "S2"


def other_connector(key: str) -> str:
    """The connector that is not the task's; it stays seated."""
    (other,) = (name for name in CONNECTOR_SITES if name != key)
    return other


LINER_ASSET = "lg_mnext3_liner_connector_task"

M2_CONNECTOR_OBJECT_AXIS = ObjectAxis(
    field="m2_connector",
    default=DEFAULT_CONNECTOR,
    objects=(
        TaskObject(key="S2", asset=LINER_ASSET, split="train", label="harness connector", size_m=(0.021, 0.018)),
        TaskObject(key="S1", asset=LINER_ASSET, split="unfit", label="harness connector", size_m=(0.037, 0.031),
                   unfit_reason="DB-M2 inserts S2 only: S1 has no tape spot, demos or collector qualification, and no "
                                "S1 work is planned; it stays mated as S2's neighbour"),
    ),
    size_axes="housing width, height",
)


def site_in_env(xyz) -> tuple[float, float, float]:
    """A package-frame point in the env frame (the package is placed without rotation)."""
    return tuple(float(v) + o for v, o in zip(xyz, LINER_ROOT_XYZ))



START_PULL_OUT_M = {"S2": CONNECTOR_SITES["S2"].stroke_m}


def start_offset(site: ConnectorSite, pull_out_m: float | None = None) -> tuple[float, float, float]:
    """Where the plug is released at reset, from its seat (package frame): ``pull_out_m`` (default the site's) along the axis."""
    distance = START_PULL_OUT_M[site.key] if pull_out_m is None else pull_out_m
    return tuple(distance * a for a in site.pull_out_axis)


START_MODES = ("tape", "entry")
DEFAULT_START = "tape"
TAPE_RELEASE = {
    "S2": ((-0.062, -0.016, 0.0347), (-0.00918979, -0.38002490, 0.0, 0.92493060)),
}
"""Per connector: the release point from its seat (package frame, m) and the plug's orientation there (xyzw)."""
TAPE_REST_FROM_SEAT_M = {"S2": (-0.0618, -0.0154, 0.0327)}
"""Where the released plug comes to rest, from its seat (measured; the progress chain's lift height reads it)."""
TAPE_SPOT_VARIATION = "tape_spot"
"""The tape start's per-episode draw, recorded as ``plug.tape_spot``: ``[dx m, dy m, dyaw rad]`` added to the
release (env x along the line, y across it, a turn about the vertical, which is the liner face's normal), as
every :class:`~dexbench_arena.variations.object_pose_variation.ObjectPoseVariation` draws. Experiments set and
override its range through Arena's variation channel (``plug.tape_spot.sampler_cfg.low`` / ``high``)."""
TAPE_SPOT_RANGE = {"S2": ((-0.008, 0.0, -0.349), (0.0, 0.008, 0.349))}
"""The training range of ``plug.tape_spot`` per connector (low, high): up to 8 mm further along the line (-x)
and 8 mm away from the robot, turned up to 20 deg either way."""
TAPE_SPOT_LIMIT = {"S2": ((-0.016, 0.0, -0.524), (0.0, 0.016, 0.524))}
"""The swept region every draw of which comes to rest standing where it was put (16 mm, 16 mm, 30 deg); a range
past it is refused. Toward the pocket (+x), toward the robot (-y) and further
along the line than 16 mm the plug slides off, topples or drops."""


@dataclass(frozen=True)
class BoxPoseDomain:
    """The scene's limit on a ``[dx, dy, dyaw]`` pose draw: a box, checked against a sampler range."""

    low: tuple[float, float, float]
    high: tuple[float, float, float]
    reason: str = ""

    def violation(self, low, high) -> str | None:
        """Why the range ``[low, high]`` can leave the box, or None when every draw stays in it."""
        for k, name in enumerate(("dx", "dy", "dyaw")):
            if float(low[k]) < self.low[k] - 1e-12 or float(high[k]) > self.high[k] + 1e-12:
                return (f"{name} [{float(low[k]):g}, {float(high[k]):g}] leaves the swept "
                        f"[{self.low[k]:g}, {self.high[k]:g}]")
        return None


PROGRESS_LIFT_M = 0.012
"""Partial credit on the tape start: the plug lifted this far above its resting height."""
PROGRESS_PREINSERT_DEPTH_M = 0.045
PROGRESS_PREINSERT_LATERAL_M = 0.002
PROGRESS_PREINSERT_TILT_DEG = 5.0
"""Partial credit on the tape start: the plug at most this far out along its axis, on it and square (pre-insertion)."""


MATE_COLLIDERS = {
    "S1": ("Liner/Collision/Mate_S1_Front", "Liner/Collision/Mate_S1_Rear"),
    "S2": ("Liner/Collision/Mate_S2_Front", "Liner/Collision/Mate_S2_Rear"),
}
PLUG_COLLIDERS = {
    key: tuple(f"Task/Connector_{key}/Collision/{name}" for name in ("Wall_e1neg", "Wall_e1pos", "Wall_e2neg", "Wall_e2pos", "Back"))
    for key in ("S1", "S2")
}
"""Each plug's colliders in the package (convex meshes, not instanced)."""
MATE_FRICTION = 0.6
"""Static and dynamic friction bound on both connectors' mates and S2's guide by default, on the ``common_s16``
Newton profile with S2's chamfered collision hulls (``PLUG_COLLISION_VARIANTS``)."""
PACKAGE_MATE_FRICTION = 2.0
"""The mate friction of the ``package`` Newton profile (8 substeps, the package's own contact; the task doc has the
sweep). Well past the 45.3 deg incline's friction angle: a plug placed square a few mm out stays out, and a push
seats it and it stays seated."""
PLUG_COLLISION_VARIANTS = {"S2": "s2_side_clip_front_chamfer_v1"}
"""The collision hulls a plug spawns with by default (``assets/connector_collision_variants``): S2's five colliders
clipped 0.4 / 0.1 mm on the side walls and chamfered at the front, the visual meshes untouched."""
_HOLDER = "Liner/Collision/P_Holder_Bracket_Mesh_8535"
GUIDE_COLLIDERS = {
    "S2": ("Liner/Collision/P_Case_Inner_001_Mesh_8655/piece_055",
           *(f"{_HOLDER}/piece_{n:03d}" for n in (1, 13, 18, 38, 55, 62, 63, 64, 65, 66, 81, 86, 97, 98, 104, 125, 135, 136, 138, 140, 141, 142, 143, 144, 146, 149, 150, 152, 153, 155, 156, 198, 201, 203, 207, 209, 212, 213, 219, 252, 254, 255, 256, 257, 258, 259, 262, 267, 268, 273, 276, 277, 278, 280, 281, 282, 288, 312, 314, 318, 332, 333, 338, 349, 353, 357, 359, 361, 366, 383, 405, 430, 442, 450, 457, 460, 461, 463, 464, 472, 473, 482, 486, 489, 490, 491, 492, 493, 495, 499, 502, 517, 526, 542, 560, 593, 594, 595, 596, 598, 600, 602, 613, 618, 663, 682, 685, 691, 695, 696, 697, 702, 703, 705))),
}


SEAT_MAX_DEPTH_M = 0.001
SEAT_MAX_LATERAL_M = 0.0005
SEAT_MAX_TILT_DEG = 2.0
SEAT_HOLD_S = 2.0

PROGRESS_ALIGN_MAX_LATERAL_ERROR_M = 0.0015
PROGRESS_ALIGN_MAX_TILT_DEG = 5.0

PLUG_DROP_M = 0.10
NEIGHBOR_MAX_DISPLACEMENT_M = 0.003

EPISODE_TIMEOUT_S = 60.0


BASE_STANDOFF_FROM_LINE_M = 0.036
_BASE_BOX_FRONT_M = 0.3493
ROBOT_BASE_ANCHORS_XYZ = {
    "A_m2": (0.0, round(LINE_NEAR_EDGE_Y_M - _BASE_BOX_FRONT_M - BASE_STANDOFF_FROM_LINE_M, 4), 0.0),
}
ANCHOR_FACING_TARGETS_XY = {"A_m2": (ROBOT_BASE_ANCHORS_XYZ["A_m2"][0], ROBOT_BASE_ANCHORS_XYZ["A_m2"][1] + 1.0)}


INTERIOR_LIGHT_XYZ = (0.0, -0.6, 2.6)

LINER_NAME = "liner"
PLUG_NAME = "plug"
NEIGHBOR_PLUG_NAME = "neighbor_plug"
