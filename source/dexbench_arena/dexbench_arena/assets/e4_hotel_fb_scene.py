# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Check the authored H1 support fixture before constructing the scene."""

from __future__ import annotations

import math


def require_h1_destination(usd_path: str, expected_xyz: tuple[float, float, float]) -> None:
    """Require the authored cart slot's world pose, before environment placement."""
    from pxr import Gf, Tf, Usd, UsdGeom

    path = "/World/FixtureSlots/Slot_StackDestination"
    guidance = (
        f"H1 needs an active {path} at {expected_xyz} with identity rotation and unit scale; "
        f"check {usd_path}. Use an assets revision containing the H1 stacking cart "
        "before collecting episodes."
    )
    try:
        stage = Usd.Stage.Open(str(usd_path), load=Usd.Stage.LoadNone)
    except Tf.ErrorException as error:
        raise ValueError(
            f"Cannot open the H1 scene. Ensure the pinned assets and Git LFS files are available. {guidance}"
        ) from error
    prim = stage.GetPrimAtPath(path) if stage else None
    transformable = UsdGeom.Xformable(prim) if prim and prim.IsActive() else None
    if not transformable:
        raise ValueError(guidance)
    actual = transformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    expected = Gf.Matrix4d(1.0).SetTranslate(Gf.Vec3d(*expected_xyz))
    if any(
        not math.isclose(actual[row][col], expected[row][col], rel_tol=0.0, abs_tol=1e-6)
        for row in range(4)
        for col in range(4)
    ):
        raise ValueError(guidance)
