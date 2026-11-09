# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""The parts DB-M5 spawns and judges: five of the PSL E3 delivery's loose fc parts and ten pool parts."""

from __future__ import annotations

from dataclasses import dataclass

M5_PART_NEWTON_COLLIDERS = "primitives"
"""The ``NewtonColliders`` option DB-M5 selects on its parts: dbclutter's ``_prim5`` primitives, the colliders
every parts-bin pile was settled with, so a pile loads at rest. The packages default to their convex pieces."""


@dataclass(frozen=True)
class M5Part:
    key: str
    """dbclutter's part key, also the token in every spawned instance name."""

    label: str
    """What the task description calls it."""

    extents_local: tuple[float, float, float]
    """Collider bounding-box size along the root frame's x, y, z [m]."""

    axis_local: tuple[float, float, float]
    """The part's long axis in its root frame (dbclutter's ``axis_local``)."""

    center_local: tuple[float, float, float]
    """The bounding-box centre in the root frame."""

    staging_rotation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    staging_offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)

    kit_box_rotation_xyzw: tuple[float, float, float, float] | None = None
    """The part's pose in a +X kit-box compartment, for a part that does not fit it staged; None uses the staging pose."""

    kit_box_offset_xyz: tuple[float, float, float] | None = None
    """Where the root origin goes with ``kit_box_rotation_xyzw``, relative to the compartment's centre on its floor."""

    @property
    def asset_name(self) -> str:
        return f"dexbench_assembly_line_e3_part_{self.key}"

    def kit_box_pose_local(
        self, sign_x: float
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        """``(offset_xyz, rotation_xyzw)`` of the root in a kit-box compartment on the box's ``sign_x`` side."""

        if self.kit_box_rotation_xyzw is None:
            return self.staging_offset_xyz, self.staging_rotation_xyzw
        (ox, oy, oz), (x, y, z, w) = self.kit_box_offset_xyz, self.kit_box_rotation_xyzw
        if sign_x > 0:
            return (ox, oy, oz), (x, y, z, w)
        return (-ox, -oy, oz), (-y, x, w, -z)

    @property
    def end_points_local(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """The centre moved half the part's length both ways along its axis."""

        half = sum(abs(a) * e for a, e in zip(self.axis_local, self.extents_local)) / 2
        cx, cy, cz = self.center_local
        ax, ay, az = self.axis_local
        return (
            (cx - half * ax, cy - half * ay, cz - half * az),
            (cx + half * ax, cy + half * ay, cz + half * az),
        )


M5_PARTS: dict[str, M5Part] = {
    part.key: part
    for part in (
        M5Part(
            "bolt_m24",
            "M24 bolt",
            (0.115008400124, 0.048008799553, 0.041576839983),
            (1.0, 0.0, 0.0),
            center_local=(4.199938e-06, 0.0, 0.02079),
            staging_offset_xyz=(-4.199938e-06, 0.0, -1.580008e-06),
        ),
        M5Part(
            "pom_roller",
            "POM conveyor roller",
            (0.14998537302, 0.05033783034, 0.05033783034),
            (1.0, 0.0, 0.0),
            center_local=(1.877546e-06, 2.4521785e-05, 0.024802937729),
            staging_offset_xyz=(-1.877546e-06, -2.4521785e-05, 0.000365977441),
        ),
        M5Part(
            "elbow_50a",
            "50A pipe elbow",
            (0.0778798, 0.077755627667, 0.062086643994),
            (1.0, 0.0, 0.0),
            center_local=(3.18e-05, -9.1729109e-05, 0.030931392212),
            staging_offset_xyz=(-3.18e-05, 9.1729109e-05, 0.000111929785),
        ),
        M5Part(
            "bearing_6206",
            "6206 ball bearing",
            (0.062019728249, 0.062019728249, 0.015973634352),
            (1.0, 0.0, 0.0),
            center_local=(-2.301562e-06, 7.160269e-06, 0.008003220295),
            staging_offset_xyz=(2.301562e-06, -7.160269e-06, -1.6403119e-05),
        ),
        M5Part(
            "nema17",
            "NEMA 17 stepper motor",
            (0.0421676, 0.0417353, 0.0352378),
            (1.0, 0.0, 0.0),
            center_local=(1.84e-05, 0.00019955, 0.0176477),
            staging_offset_xyz=(-1.84e-05, -0.00019955, -2.88e-05),
        ),
        M5Part(
            "automotive_exhaust_90_bend_od63",
            "90-degree exhaust bend",
            (0.105654681593, 0.105654683176, 0.063775655847),
            (0.0, 1.0, 0.0),
            center_local=(-0.004399372977, -0.004399371711, 0.031500005359),
            staging_offset_xyz=(0.004399372977, 0.004399371711, 0.000387822564),
        ),
        M5Part(
            "deco_ad02_2_station_ng6cetop3_manifold",
            "2-station NG6 hydraulic manifold",
            (0.120000004785, 0.070000028385, 0.07000002957),
            (1.0, 0.0, 0.0),
            center_local=(0.0, 1.124e-09, 0.035000013377),
            staging_rotation_xyzw=(0.707106781187, 0.0, 0.0, 0.707106781187),
            staging_offset_xyz=(0.0, 0.035000013377, 0.035000013068),
        ),
        M5Part(
            "gy1_rigid_flange_coupling_half_flange",
            "rigid flange coupling half",
            (0.07998678884, 0.07998678884, 0.063994),
            (1.0, 0.0, 0.0),
            center_local=(0.0, 0.0, 0.0320024),
            staging_offset_xyz=(0.0, 0.0, -5.4e-06),
        ),
        M5Part(
            "jl_sus304_welded_tee_50a",
            "50A welded stainless tee",
            (0.1269334, 0.0605, 0.0937704),
            (1.0, 0.0, 0.0),
            center_local=(-2.41e-05, 0.0, 0.0468852),
            staging_rotation_xyzw=(0.707106781187, 0.0, 0.0, 0.707106781187),
            staging_offset_xyz=(2.41e-05, 0.0468852, 0.03025),
        ),
        M5Part(
            "rs_pro_stepper_motor_535_0502_nema23_class",
            "NEMA 23 stepper motor",
            (0.056400001049, 0.058285947889, 0.075589577831),
            (0.0, 0.0, 1.0),
            center_local=(0.0, 5.7031259e-05, 0.037794788916),
            staging_rotation_xyzw=(0.0, -0.707106781187, 0.0, 0.707106781187),
            staging_offset_xyz=(0.037794788916, -5.7031259e-05, 0.028200000525),
        ),
        M5Part(
            "two_piece_split_shaft_collar_od45_bore2_w12",
            "split shaft collar",
            (0.045704997738, 0.045704997738, 0.0119992),
            (0.0, 1.0, 0.0),
            center_local=(0.0, 0.0, 0.0060016),
            staging_offset_xyz=(0.0, 0.0, -2e-06),
        ),
        M5Part(
            "tyb_ucp205_pillow_block_bearing_unit",
            "UCP205 pillow block bearing",
            (0.1399758, 0.0387902, 0.0755908),
            (1.0, 0.0, 0.0),
            center_local=(0.0, -1.34e-05, 0.0342537),
            staging_rotation_xyzw=(0.707106781187, 0.0, 0.0, 0.707106781187),
            staging_offset_xyz=(0.0, 0.0342537, 0.0194085),
        ),
        M5Part(
            "v_belt_pulley_1a_70_12_4",
            "V-belt pulley",
            (0.066497985545, 0.066497985545, 0.0299966),
            (0.0, 1.0, 0.0),
            center_local=(0.0, 0.0, 0.0150017),
            staging_offset_xyz=(0.0, 0.0, -3.4e-06),
        ),
        M5Part(
            "boltkorea_iso4014_class_8_8_hex_bolt_m36x160",
            "M36 hex bolt",
            (0.063575401869, 0.055057913223, 0.182478700447),
            (0.0, 0.0, 1.0),
            center_local=(0.0, 0.0, 0.091260650224),
            staging_rotation_xyzw=(0.707106781187, 0.0, 0.0, 0.707106781187),
            staging_offset_xyz=(0.0, 0.091260650224, 0.027528956612),
            kit_box_rotation_xyzw=(0.604210067128, -0.604210067128, -0.367328456263, 0.367328456263),
            kit_box_offset_xyz=(0.089689799486, 0.0, 0.108940455583),
        ),
        M5Part(
            "norgren_rt_57232_m_50_roundline_cylinder",
            "Norgren pneumatic cylinder",
            (0.0336, 0.0336, 0.1730151),
            (0.0, 0.0, 1.0),
            center_local=(1e-07, 2e-07, 0.08650755),
            staging_rotation_xyzw=(0.0, -0.707106781187, 0.0, 0.707106781187),
            staging_offset_xyz=(0.08650755, -2e-07, 0.0167999),
            kit_box_rotation_xyzw=(0.375944681762, 0.375944681762, 0.598886964506, 0.598886964506),
            kit_box_offset_xyz=(-0.077697584999, -1e-07, 0.015629790297),
        ),
    )
}
DEFAULT_PART = "bolt_m24"


def get_m5_part(key: str) -> M5Part:
    try:
        return M5_PARTS[key]
    except KeyError:
        raise ValueError(f"unknown DB-M5 part {key!r}; known: {sorted(M5_PARTS)}") from None


def bin_part_names(key: str, count: int) -> tuple[str, ...]:
    """``bin_part_<key>_NN``: the parts-bin pile's instance names."""

    return tuple(f"bin_part_{key}_{index:02d}" for index in range(count))


def kit_box_part_names(key: str, count: int) -> tuple[str, ...]:
    """``kit_box_<key>_NN``: the kit box's pre-loaded instance names."""

    return tuple(f"kit_box_{key}_{index:02d}" for index in range(count))
