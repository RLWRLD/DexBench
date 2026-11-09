# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Edits a DB-M task authors over the E3 assembly line at spawn."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pxr import Usd


def edit_e3_cell(
    stage: Usd.Stage,
    cell_root: str,
    *,
    slot_poses: Mapping[str, tuple[Sequence[float], float | None]],
    deactivated: Sequence[str],
) -> None:
    """Move each slot of ``slot_poses`` to its ``(xyz, yaw_deg)`` (the yaw about Z written only when not None)
    and deactivate ``deactivated``, over the assembly line composed under ``cell_root``."""
    from pxr import Gf, UsdGeom

    def prim_at(relative_path: str):
        prim = stage.GetPrimAtPath(f"{cell_root}/{relative_path}")
        if not prim.IsValid():
            raise RuntimeError(f"E3 assembly line has no {cell_root}/{relative_path}; the layer changed shape.")
        return prim

    def op(xformable, op_type, slot: str, what: str):
        found = next((o for o in xformable.GetOrderedXformOps() if o.GetOpType() == op_type), None)
        if found is None:
            raise RuntimeError(f"{slot} has no {what} op to set.")
        return found

    for slot, (xyz, yaw_deg) in slot_poses.items():
        xformable = UsdGeom.Xformable(prim_at(slot))
        op(xformable, UsdGeom.XformOp.TypeTranslate, slot, "translate").Set(Gf.Vec3d(*xyz))
        if yaw_deg is not None:
            op(xformable, UsdGeom.XformOp.TypeRotateXYZ, slot, "rotateXYZ").Set(Gf.Vec3f(0.0, 0.0, float(yaw_deg)))
    for relative_path in deactivated:
        prim_at(relative_path).SetActive(False)
