# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""DexBench's edits to the M5 corner of the E3 assembly line, authored at spawn."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pxr import Usd


M5_CONVEYOR_GEOM_RELATIVE_PATH = "FixtureSlots/Slot_Conveyor/Asset/Mesh/Geom/conveyor_fc"
M5_STAGING_SHELF_RELATIVE_PATH = "FixtureSlots/Slot_StagingShelf"
M5_PARTS_BIN_RELATIVE_PATH = "FixtureSlots/Slot_PartsBin"

CONVEYOR_FRAME_BELT_FACE_Y = {"ConvFC_Frame_S": (3.047, -1.0), "ConvFC_Frame_N": (3.847, 1.0)}
CONVEYOR_LEGS_OUTWARD_Y = {f"ConvFC_Leg_{i}{side}": sign for i in (1, 2, 3, 4) for side, sign in (("S", -1.0), ("N", 1.0))}
CONVEYOR_FRAME_MODULE_THICKNESS_M = 0.123
CONVEYOR_FRAME_THICKNESS_M = 0.03
CONVEYOR_LEGS_SHIFT_Y_M = CONVEYOR_FRAME_MODULE_THICKNESS_M - CONVEYOR_FRAME_THICKNESS_M

STAGING_SHELF_TOP_Z_M = 0.7

STAGING_SHELF_BIN_YAW_DEG = 90.0
SLOT_STAGING_SHELF_XYZ = (3.0, 2.667, 0.0)
SLOT_PARTS_BIN_XYZ = (3.0, 2.761, STAGING_SHELF_TOP_Z_M)

PARTS_BIN_GEOM_RELATIVE_PATH = f"{M5_PARTS_BIN_RELATIVE_PATH}/Asset/Geom/parts_bin_fc"
PARTS_BIN_FLOOR = "BinFC_Floor"
PARTS_BIN_WALLS = {
    "BinFC_SideWall_E": (0, 1.0),
    "BinFC_SideWall_W": (0, -1.0),
    "BinFC_FrontBar": (1, -1.0),
    "BinFC_BackWall": (1, 1.0),
}
PARTS_BIN_FLOOR_COLLIDER_THICKNESS_M = 0.15
PARTS_BIN_MJC_PRIORITY = 1
PARTS_BIN_MJC_SOLIMP = (0.99, 0.999, 0.001, 0.5, 2.0)


def _require_prim(stage: Usd.Stage, path: str) -> Usd.Prim:
    """The layer's shape is this module's contract; fail loudly when it moves."""

    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"E3 assembly line has no prim at {path}; assembly_line.usda changed shape.")
    return prim


def _set_op(prim: Usd.Prim, op_type, value) -> None:
    """Set the prim's one ``op_type`` xform op, adding it when the prim has none."""

    from pxr import UsdGeom

    xformable = UsdGeom.Xformable(prim)
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == op_type and not op.IsInverseOp():
            op.Set(value)
            return
    xformable.AddXformOp(op_type, UsdGeom.XformOp.PrecisionDouble).Set(value)


def _require_slot(stage: Usd.Stage, root: str, relative_path: str, xyz, yaw_deg: float) -> None:
    """Refuse a layer whose slot is not where the bin pile is placed from."""

    slot = _require_prim(stage, f"{root}/{relative_path}")
    for name, want in (("xformOp:translate", xyz), ("xformOp:rotateXYZ", (0.0, 0.0, yaw_deg))):
        current = tuple(slot.GetAttribute(name).Get())
        if max(abs(c - v) for c, v in zip(current, want)) > 1e-6:
            raise RuntimeError(f"{relative_path} {name} is {current}, expected {want}; assembly_line.usda changed.")


def _require_thin_conveyor_frames(stage: Usd.Stage, root: str) -> None:
    """Refuse a layer whose side frames are not ``CONVEYOR_FRAME_THICKNESS_M`` thick on their belt-side faces."""

    from pxr import Usd, UsdGeom

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    root_prim = _require_prim(stage, root)
    for name, (belt_face_y, outward) in CONVEYOR_FRAME_BELT_FACE_Y.items():
        frame = _require_prim(stage, f"{root}/{M5_CONVEYOR_GEOM_RELATIVE_PATH}/{name}")
        box = cache.ComputeRelativeBound(frame, root_prim).ComputeAlignedRange()
        span = (box.GetMin()[1], box.GetMax()[1])
        want = tuple(sorted((belt_face_y, belt_face_y + outward * CONVEYOR_FRAME_THICKNESS_M)))
        if max(abs(c - v) for c, v in zip(span, want)) > 1e-4:
            raise RuntimeError(f"{name} spans y {span}, expected {want}; assembly_line.usda changed.")


def _pull_in_conveyor_legs(stage: Usd.Stage, root: str, shift: float) -> None:
    from pxr import Gf, UsdGeom

    for name, outward in CONVEYOR_LEGS_OUTWARD_Y.items():
        leg = _require_prim(stage, f"{root}/{M5_CONVEYOR_GEOM_RELATIVE_PATH}/{name}")
        _set_op(leg, UsdGeom.XformOp.TypeTranslate, Gf.Vec3d(0.0, -outward * shift, 0.0))


def _collider_like(stage: Usd.Stage, path, material) -> Usd.Prim:
    """Make the prim at ``path`` an invisible, stiff collider bound to the physics ``material``."""

    from pxr import Sdf, UsdGeom, UsdPhysics, UsdShade, Vt

    prim = stage.GetPrimAtPath(path)
    UsdGeom.Imageable(prim).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    UsdGeom.Imageable(prim).CreatePurposeAttr(UsdGeom.Tokens.guide)
    UsdPhysics.CollisionAPI.Apply(prim)
    prim.CreateAttribute("mjc:priority", Sdf.ValueTypeNames.Int).Set(PARTS_BIN_MJC_PRIORITY)
    prim.CreateAttribute("mjc:solimp", Sdf.ValueTypeNames.DoubleArray).Set(Vt.DoubleArray(PARTS_BIN_MJC_SOLIMP))
    if material:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, materialPurpose="physics"
        )
    return prim


def _define_box(stage: Usd.Stage, path, lower, upper):
    """A ``Cube`` spanning ``lower``..``upper`` in its parent's frame."""

    from pxr import Gf, UsdGeom

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    _set_op(cube.GetPrim(), UsdGeom.XformOp.TypeTranslate, Gf.Vec3d(*((lo + hi) / 2 for lo, hi in zip(lower, upper))))
    _set_op(cube.GetPrim(), UsdGeom.XformOp.TypeScale, Gf.Vec3d(*(hi - lo for lo, hi in zip(lower, upper))))
    return cube


def _replace_parts_bin_colliders(stage: Usd.Stage, root: str) -> None:
    """The bin's floor and walls collide as boxes (see ``PARTS_BIN_FLOOR_COLLIDER_THICKNESS_M``)."""

    from pxr import Gf, UsdGeom, UsdPhysics, UsdShade, Vt

    geom = f"{root}/{PARTS_BIN_GEOM_RELATIVE_PATH}"

    def authored(name):
        mesh = _require_prim(stage, f"{geom}/{name}/{name}")
        if not mesh.HasAPI(UsdPhysics.CollisionAPI):
            raise RuntimeError(f"parts_bin_fc's {name} has no collider; the fixture changed shape.")
        points = [tuple(p) for p in UsdGeom.Mesh(mesh).GetPointsAttr().Get()]
        material, _ = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial(materialPurpose="physics")
        UsdPhysics.CollisionAPI(mesh).CreateCollisionEnabledAttr(False)
        return points, material

    points, material = authored(PARTS_BIN_FLOOR)
    lower = [min(p[i] for p in points) for i in range(3)]
    upper = [max(p[i] for p in points) for i in range(3)]
    floor_top = upper[2]
    bottom = floor_top - PARTS_BIN_FLOOR_COLLIDER_THICKNESS_M
    path = f"{geom}/{PARTS_BIN_FLOOR}/{PARTS_BIN_FLOOR}_box"
    _define_box(stage, path, (lower[0], lower[1], bottom), upper)
    _collider_like(stage, path, material)

    for name, (axis, outward) in PARTS_BIN_WALLS.items():
        points, material = authored(name)
        coords = [p[axis] for p in points]
        outer = max(coords) if outward > 0 else min(coords)
        inner = min((p[axis] for p in points if abs(p[2] - floor_top) < 1e-6), key=lambda c: outward * c)
        low, high = sorted((inner, outer))
        lower = [min(p[i] for p in points) for i in range(3)]
        upper = [max(p[i] for p in points) for i in range(3)]
        lower[axis], upper[axis] = low, high
        lower[2] = bottom
        along = 1 - axis
        top = [p for p in points if abs(p[axis] - outer) < 1e-6 and p[2] > floor_top + 1e-6]
        top_low = min(top, key=lambda p: p[2])
        top_high = max(top, key=lambda p: p[2])
        path = f"{geom}/{name}/{name}_box"
        _define_box(stage, path, lower, (upper[0], upper[1], top_low[2]))
        _collider_like(stage, path, material)
        if top_high[2] - top_low[2] < 1e-6:
            continue

        def corner(across, end, z):
            p = [0.0, 0.0, z]
            p[axis], p[along] = across, end
            return Gf.Vec3f(*p)

        wedge = UsdGeom.Mesh.Define(stage, f"{geom}/{name}/{name}_wedge")
        wedge.CreatePointsAttr(
            Vt.Vec3fArray(
                [corner(c, top_low[along], top_low[2]) for c in (low, high)]
                + [corner(c, top_high[along], top_low[2]) for c in (low, high)]
                + [corner(c, top_high[along], top_high[2]) for c in (low, high)]
            )
        )
        wedge.CreateFaceVertexCountsAttr([3, 3, 4, 4, 4])
        wedge.CreateFaceVertexIndicesAttr([0, 2, 4, 1, 5, 3, 0, 1, 3, 2, 2, 3, 5, 4, 0, 4, 5, 1])
        UsdPhysics.MeshCollisionAPI.Apply(wedge.GetPrim()).CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
        _collider_like(stage, wedge.GetPath(), material)


def apply_m5_station_edits(stage: Usd.Stage, root: str) -> None:
    """Check r01's M5 layout and author the station edits over the assembly line composed under ``root``."""

    _require_prim(stage, root)
    _require_slot(stage, root, M5_STAGING_SHELF_RELATIVE_PATH, SLOT_STAGING_SHELF_XYZ, STAGING_SHELF_BIN_YAW_DEG)
    _require_slot(stage, root, M5_PARTS_BIN_RELATIVE_PATH, SLOT_PARTS_BIN_XYZ, STAGING_SHELF_BIN_YAW_DEG)
    _require_thin_conveyor_frames(stage, root)
    _pull_in_conveyor_legs(stage, root, CONVEYOR_LEGS_SHIFT_Y_M)
    _replace_parts_bin_colliders(stage, root)
