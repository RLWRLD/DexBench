# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Recorded independent drawer/washer placement for tabletop M1 development."""

from __future__ import annotations

import math

from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg

from dexbench_arena.layouts.db_m1_layout import (
    TABLETOP_DRAWER_LOCAL_BOUNDS_XY_M,
    TABLETOP_SPATIAL_PILOT_BOUNDS,
    TABLETOP_SUPPORT_BOUNDS_XY_M,
    TABLETOP_WASHER_LOCAL_BOUNDS_XY_M,
    WASHER_LINE_CENTER_XYZ,
    WASHER_LINE_SIZE_M,
    WASHER_ROOT_XYZ,
)
from dexbench_arena.variations.object_pose_variation import (
    ObjectPoseVariation,
    ObjectPoseVariationCfg,
    SupportedPlanarPoseDomain,
)


def tabletop_pose_variations(drawer_xyz, drawer_yaw_deg) -> tuple[ObjectPoseVariation, ObjectPoseVariation]:
    """Construct the two asset variations for the scene's configured nominal pose."""
    support_bounds = tuple(
        (center - size / 2, center + size / 2)
        for center, size in zip(WASHER_LINE_CENTER_XYZ[:2], WASHER_LINE_SIZE_M[:2])
    )
    domains = (
        SupportedPlanarPoseDomain(
            default_xy_m=tuple(drawer_xyz[:2]),
            default_yaw_rad=math.radians(drawer_yaw_deg),
            local_bounds_xy_m=TABLETOP_DRAWER_LOCAL_BOUNDS_XY_M,
            support_bounds_xy_m=TABLETOP_SUPPORT_BOUNDS_XY_M,
            reason="the flat drawer must start completely on the worktop",
        ),
        SupportedPlanarPoseDomain(
            default_xy_m=WASHER_ROOT_XYZ[:2],
            default_yaw_rad=0.0,
            local_bounds_xy_m=TABLETOP_WASHER_LOCAL_BOUNDS_XY_M,
            support_bounds_xy_m=support_bounds,
            reason="the washer footprint must stay on the fixed pallet",
        ),
    )
    return tuple(
        ObjectPoseVariation(
            name,
            domain=domain,
            cfg=ObjectPoseVariationCfg(
                enabled=False,
                sampler_cfg=UniformSamplerCfg(
                    low=list(TABLETOP_SPATIAL_PILOT_BOUNDS[name][0]),
                    high=list(TABLETOP_SPATIAL_PILOT_BOUNDS[name][1]),
                ),
            ),
        )
        for name, domain in zip(("drawer", "washer"), domains)
    )
