"""DexBench's E2 logistics cell: the PhysicsSim Lab delivery, wired the way E1 is."""

from __future__ import annotations

import math
from pathlib import Path

from dataclasses import dataclass
from typing import TYPE_CHECKING

from dexbench_arena import ASSETS_DIR
from dexbench_arena.assets.scene_bundles import runtime_usd
from dexbench_arena.assets.logistics_conveyor import (
    LOGISTICS_CONVEYOR_BODIES,
    freeze_logistics_conveyor,
    logistics_conveyor_frozen_bodies,
)
from dexbench_arena.tasks.object_axis import ObjectAxis, TaskObject

if TYPE_CHECKING:
    from pxr import Usd


E2_ENV_DIR = ASSETS_DIR / "env" / "e2_logistics" / "cell"

E2_CANONICAL_LAYER = Path(runtime_usd(E2_ENV_DIR / "cell.usda"))



E2_PARCEL_CAGE_SLOT = "FixtureSlots/Slot_ParcelCage"
E2_PARCEL_CAGE_SLOT_XY_M = (1.01, 1.6922)
E2_PARCEL_CAGE_SLOT_YAW_DEG = 90.0

E2_COLLISION_PROXY_SUFFIX = "_colproxy"

E2_CONVEYOR_ASSET = "FixtureSlots/Slot_Conveyor/Asset"
E2_FROZEN_BODIES = tuple(f"{E2_CONVEYOR_ASSET}/{body}" for body in LOGISTICS_CONVEYOR_BODIES)
E2_FIXTURE_SLOTS = "FixtureSlots"

E2_SORT_TOTE_SLOT = f"{E2_FIXTURE_SLOTS}/Slot_SortTote"
E2_SORT_TOTE_SLOT_XYZ_M = (2.47, 2.28, 0.75)
E2_SORT_TOTE_SIZE_XYZ_M = (0.80, 0.56, 0.26)
E2_SORT_TOTE_WALL_M = 0.010
_E2_SORT_TOTE_COLLIDERS = f"{E2_SORT_TOTE_SLOT}/Asset/Geom/sort_tote_single_fc/Colliders"
E2_SORT_TOTE_FLOOR_COLLIDER = f"{_E2_SORT_TOTE_COLLIDERS}/StoteFC_Floor/StoteFC_Floor"
E2_SORT_TOTE_WALL_COLLIDERS = tuple(
    f"{_E2_SORT_TOTE_COLLIDERS}/StoteFC_Wall_{side}/StoteFC_Wall_{side}" for side in "NSEW"
)


def _require_prim(stage: Usd.Stage, path: str) -> Usd.Prim:
    """The delivery's shape is this module's contract; fail loudly when it moves."""

    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"E2 scene has no prim at {path}; expected the canonical cell.usda composition (loaded from {E2_CANONICAL_LAYER}).")
    return prim


def _hide_cage_collision_proxies(stage: Usd.Stage, cell_root: str) -> None:
    from pxr import Usd, UsdGeom

    asset = _require_prim(stage, f"{cell_root}/{E2_PARCEL_CAGE_SLOT}/Asset")
    hidden = 0
    for prim in Usd.PrimRange(asset):
        if prim.GetName().endswith(E2_COLLISION_PROXY_SUFFIX):
            UsdGeom.Imageable(prim).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
            hidden += 1
    if hidden == 0:
        raise RuntimeError(f"E2 parcel cage has no *{E2_COLLISION_PROXY_SUFFIX} prims; the fixture changed shape.")


def e2_frozen_bodies(moving_belts: tuple[str, ...] = ()) -> tuple[str, ...]:
    """E2_FROZEN_BODIES less the top runs of ``moving_belts`` (logistics_conveyor.logistics_conveyor_frozen_bodies)."""

    return tuple(f"{E2_CONVEYOR_ASSET}/{body}" for body in logistics_conveyor_frozen_bodies(moving_belts))


def apply_e2_cell_edits(stage: Usd.Stage, cell_root: str, moving_belts: tuple[str, ...] = ()) -> None:
    """Turn the canonical layer composed under ``cell_root`` into DexBench's cell."""

    _require_prim(stage, cell_root)
    _hide_cage_collision_proxies(stage, cell_root)
    freeze_logistics_conveyor(stage, f"{cell_root}/{E2_CONVEYOR_ASSET}", moving_belts)


E2_BENCH_STATION_DISTRACTORS = ("Distractors/Dist_Tote_02", "Distractors/Dist_Bench_WCBox")
E2_OUTBOUND_BOX_BODY = "FixtureSlots/Slot_OutboundBox/Asset"
E2_OUTBOUND_BOX_WALL_MESH = "Geom/outbound_box_fc/OutboxFC_Wall_S/OutboxFC_Wall_S_e2_001"
E2_OUTBOUND_BOX_FLOOR_MESH = "Geom/outbound_box_fc/OutboxFC_Floor/OutboxFC_Floor"
E2_OUTBOUND_BOX_FLAPS = {
    side: f"Geom/outbound_box_fc/BG_E2_Outbox_Flap_{side}_visonly" for side in "NESW"
}
"""Flap side -> its body (the Xform over the mesh of the same name), under the box body."""
E2_OUTBOUND_BOX_FLAPS_ROOT = "Geom/outbound_box_fc"
"""The flaps' common parent, under the box body: the articulation root."""
E2_OUTBOUND_BOX_FLAP_HINGE = "FlapHinge"
"""The hinge joint prim, under each flap body."""
E2_OUTBOUND_BOX_WELD = "BoxWeld"
"""The fixed joint, under the box body, that holds it in the world."""
E2_OUTBOUND_BOX_FLAP_MASS_KG = 0.03
E2_OUTBOUND_BOX_FLAP_HINGE_STIFFNESS_NM_PER_RAD = 0.2
E2_OUTBOUND_BOX_FLAP_HINGE_DAMPING_NMS_PER_RAD = 0.02
E2_OUTBOUND_BOX_FLAP_FOLD_IN_MARGIN_DEG = 5.0
E2_OUTBOUND_BOX_FLAP_FOLD_OUT_DEG = 60.0
E2_PACKING_BENCH_BODY = "FixtureSlots/Slot_PackingBench/Asset"
E2_PACKING_BENCH_DRESSING = "FixtureSlots/Slot_PackingBench/Dressing"
"""The props on the packing bench's upper shelf: static colliders, not part of the bench body."""
E2_PACKING_BENCH_OVERLAP_PREFIXES = ("BenchFC_Leg_", "BenchFC_Foot_")
"""The bench colliders that reach the floor (``_bench_station_overlap_colliders``)."""
E2_BENCH_STATION_REFERENCES = {"packing_bench": E2_PACKING_BENCH_BODY, "outbound_box": E2_OUTBOUND_BOX_BODY}
"""The bench station's movable bodies as the environment's object references: scene entity name ->
path under the cell. Their variations are ``packing_bench.height`` and ``outbound_box.pose``."""
E2_STATION_SIZE_OP = "dexbench_station_size"
"""Suffix of the ops the box's size scale is authored with."""
E2_PACK_INVOICE_BOX_SCALE_XYZ = (0.30 / 0.394, 1.0, 0.165 / 0.25)
"""The pack-invoice task's outbound box over the delivery's, along the box's x, y and height: its
25 cm walls become 16.5 cm (16 cm deep, 5 mm floor), so the box's far rim and a hand reaching into it stay in the
320x180 ego frames the policy trains on, and its 39.4 cm along x become 30 cm, which leaves the box room
to move on the worktop. Its y stays 29.4 cm inside: the sheet (20.2 x 27.6 cm at its default
scale) goes in with its long side along it, so there is less than 2 cm to give."""
E2_OUTBOUND_BOX_SHELL = ("OutboxFC_Wall_E", "OutboxFC_Wall_W", "OutboxFC_Wall_N", "OutboxFC_Wall_S", "OutboxFC_Fill")
"""The box's parts, under ``E2_OUTBOUND_BOX_FLAPS_ROOT``, that its size scale scales on every axis; the
floor keeps its thickness and the flaps keep their angle (``_resize_outbound_box_flap``)."""


def clear_e2_bench_station(
    stage: Usd.Stage,
    cell_root: str,
    outbound_box_scale_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> None:
    """Free the floor in front of the packing bench and make its bench and box movable bodies."""

    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

    for rel in E2_BENCH_STATION_DISTRACTORS:
        _require_prim(stage, f"{cell_root}/{rel}").SetActive(False)
    box_path = f"{cell_root}/{E2_OUTBOUND_BOX_BODY}"
    box = _require_prim(stage, box_path)
    bench = _require_prim(stage, f"{cell_root}/{E2_PACKING_BENCH_BODY}")
    floor = _require_prim(stage, f"{box_path}/{E2_OUTBOUND_BOX_FLOOR_MESH}")
    scale = tuple(float(v) for v in outbound_box_scale_xyz)
    if len(scale) != 3 or not all(0.0 < v <= 2.0 for v in scale):
        raise ValueError(f"outbound_box_scale_xyz needs three factors in (0, 2], got {outbound_box_scale_xyz!r}")
    if scale != (1.0, 1.0, 1.0):
        _resize_outbound_box(stage, box_path, floor, scale)
    UsdPhysics.RigidBodyAPI.Apply(bench).CreateKinematicEnabledAttr(True)
    for path in _bench_station_overlap_colliders(stage, cell_root):
        UsdPhysics.CollisionAPI(stage.GetPrimAtPath(path)).CreateCollisionEnabledAttr(False)
    if not box.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError(f"{box_path} is no rigid body; the outbound box changed shape.")
    UsdPhysics.ArticulationRootAPI.Apply(box)
    box_to_world = UsdGeom.Xformable(box).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    weld = UsdPhysics.FixedJoint.Define(stage, f"{box_path}/{E2_OUTBOUND_BOX_WELD}")
    weld.CreateBody1Rel().SetTargets([box.GetPath()])
    weld.CreateLocalPos0Attr(Gf.Vec3f(box_to_world.ExtractTranslation()))
    weld.CreateLocalRot0Attr(Gf.Quatf(box_to_world.ExtractRotationQuat()))
    weld.CreateLocalPos1Attr(Gf.Vec3f(0.0))
    weld.CreateLocalRot1Attr(Gf.Quatf(1.0))

    wall = _require_prim(stage, f"{box_path}/{E2_OUTBOUND_BOX_WALL_MESH}")
    if not wall.HasAPI(UsdPhysics.CollisionAPI):
        raise RuntimeError(f"{wall.GetPath()} has no collider to copy; the outbound box changed shape.")
    collider = {
        "schemas": [s for s in wall.GetAppliedSchemas() if "Collision" in s],
        "attrs": [a for a in wall.GetAttributes() if a.GetName().startswith(("physics:", "newton:")) and a.HasAuthoredValue()],
        "material": UsdShade.MaterialBindingAPI(wall).GetDirectBinding("physics").GetMaterial(),
    }
    flap_paths = {side: f"{box_path}/{rel}" for side, rel in E2_OUTBOUND_BOX_FLAPS.items()}
    box_colliders = [
        str(p.GetPath()) for p in Usd.PrimRange(box)
        if p.HasAPI(UsdPhysics.CollisionAPI) and not any(p.GetPath().HasPrefix(f) for f in flap_paths.values())
    ]
    for side, flap_path in flap_paths.items():
        others = [p for s, p in flap_paths.items() if s != side]
        _hinge_outbound_box_flap(
            stage, side, flap_path, box_path, f"{box_path}/{E2_OUTBOUND_BOX_FLAPS_ROOT}", floor, collider,
            filtered=[*box_colliders, *others],
        )


def _bench_station_overlap_colliders(stage, cell_root: str) -> list:
    """The colliders a moved packing bench overlaps: its legs and feet, and the props on its shelf."""

    from pxr import Usd, UsdPhysics

    bench = _require_prim(stage, f"{cell_root}/{E2_PACKING_BENCH_BODY}")
    dressing = _require_prim(stage, f"{cell_root}/{E2_PACKING_BENCH_DRESSING}")
    legs = [p.GetPath() for p in Usd.PrimRange(bench)
            if p.HasAPI(UsdPhysics.CollisionAPI) and p.GetName().startswith(E2_PACKING_BENCH_OVERLAP_PREFIXES)]
    props = [p.GetPath() for p in Usd.PrimRange(dressing) if p.HasAPI(UsdPhysics.CollisionAPI)]
    if not legs or not props:
        raise RuntimeError(f"packing bench overlap colliders not found (legs {len(legs)}, shelf props {len(props)}); "
                           "the packing bench changed shape.")
    return [*legs, *props]


def _prepend_ops(prim, name: str, ops) -> None:
    """Author ``ops`` ([(kind, value)], kind ``translate`` / ``scale`` / ``rotateZ``) ahead of the
    prim's own, as ``xformOp:<kind>:<name>_<index>``."""

    from pxr import UsdGeom

    xformable = UsdGeom.Xformable(prim)
    order = xformable.GetOrderedXformOps()
    add = {
        "translate": xformable.AddTranslateOp, "scale": xformable.AddScaleOp, "rotateZ": xformable.AddRotateZOp,
    }
    added = []
    for index, (kind, value) in enumerate(ops):
        op = add[kind](UsdGeom.XformOp.PrecisionDouble, f"{name}_{index}")
        op.Set(value)
        added.append(op)
    xformable.SetXformOpOrder([*added, *order], xformable.GetResetXformStack())


def _in_parent(prim, world_point):
    """``world_point`` in the frame of ``prim``'s parent."""

    from pxr import Gf, Usd, UsdGeom

    parent = UsdGeom.Xformable(prim.GetParent()).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return parent.GetInverse().Transform(Gf.Vec3d(*world_point))


def _floor_bottom_center(floor):
    """The middle of the box floor's underside, in the world."""

    from pxr import Gf, Usd, UsdGeom

    bound = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(floor)
    local = bound.GetRange()
    mid = local.GetMidpoint()
    return bound.GetMatrix().Transform(Gf.Vec3d(mid[0], mid[1], local.GetMin()[2]))


def _resize_outbound_box(stage, box_path: str, floor, scale) -> None:
    """Scale the box's walls, fill and floor about its floor's underside and move its flaps along."""

    from pxr import Gf

    bottom = _floor_bottom_center(floor)
    floor_xform = floor.GetParent()
    for part in [_require_prim(stage, f"{box_path}/{E2_OUTBOUND_BOX_FLAPS_ROOT}/{n}") for n in E2_OUTBOUND_BOX_SHELL] + [floor_xform]:
        factors = Gf.Vec3d(scale[0], scale[1], 1.0 if part == floor_xform else scale[2])
        pivot = _in_parent(part, bottom)
        _prepend_ops(part, E2_STATION_SIZE_OP, [("translate", pivot), ("scale", factors), ("translate", -pivot)])
    for side, rel in E2_OUTBOUND_BOX_FLAPS.items():
        flap = _require_prim(stage, f"{box_path}/{rel}")
        _resize_outbound_box_flap(stage, side, _require_prim(stage, f"{box_path}/{rel}/{flap.GetName()}"), bottom, scale)


def _flap_hinge_and_tip(mesh, box_center, across: int, world_to_frame):
    """A cuboid flap mesh's hinge (the edge nearest the box) and tip midpoints, in a frame."""

    from pxr import Gf, Usd, UsdGeom

    to_frame = UsdGeom.Xformable(mesh).ComputeLocalToWorldTransform(Usd.TimeCode.Default()) * world_to_frame
    corners = [to_frame.Transform(Gf.Vec3d(p)) for p in UsdGeom.Mesh(mesh).GetPointsAttr().Get()]
    if len(corners) != 8:
        raise RuntimeError(f"{mesh.GetPath()}: {len(corners)} points, not a cuboid; the outbound box changed shape.")
    outward = 1.0 if sum(c[across] for c in corners) / 8 > box_center[across] else -1.0
    corners.sort(key=lambda c: outward * c[across])
    return sum(corners[:4], Gf.Vec3d(0.0)) / 4, sum(corners[4:], Gf.Vec3d(0.0)) / 4


def _resize_outbound_box_flap(stage, side, mesh, bottom, scale) -> None:
    """Move a flap with its wall's top edge and resize it to match, at its angle."""

    from pxr import Gf

    along, across = (0, 1) if side in "NS" else (1, 0)
    hinge, _ = _flap_hinge_and_tip(mesh, bottom, across, Gf.Matrix4d(1.0))
    moved = Gf.Vec3d(*(bottom[i] + (hinge[i] - bottom[i]) * scale[i] for i in range(3)))
    factors = Gf.Vec3d(scale[2], scale[2], scale[2])
    factors[along] = scale[along]
    _prepend_ops(mesh, E2_STATION_SIZE_OP, [
        ("translate", _in_parent(mesh, moved)), ("scale", factors), ("translate", -_in_parent(mesh, hinge)),
    ])


def _hinge_outbound_box_flap(stage, side, flap_path, box_path, root_path, floor, collider, filtered) -> None:
    """Make one flap a body with the wall's collider, hinged on the box body."""


    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    flap = _require_prim(stage, flap_path)
    mesh = _require_prim(stage, f"{flap_path}/{flap.GetName()}")
    for schema in collider["schemas"]:
        mesh.AddAppliedSchema(schema)
    for attr in collider["attrs"]:
        mesh.CreateAttribute(attr.GetName(), attr.GetTypeName(), custom=attr.IsCustom()).Set(attr.Get())
    if collider["material"]:
        UsdShade.MaterialBindingAPI.Apply(mesh).Bind(collider["material"], UsdShade.Tokens.weakerThanDescendants, "physics")
    UsdPhysics.RigidBodyAPI.Apply(flap)
    UsdPhysics.MassAPI.Apply(flap).CreateMassAttr(E2_OUTBOUND_BOX_FLAP_MASS_KG)
    UsdPhysics.FilteredPairsAPI.Apply(flap).CreateFilteredPairsRel().SetTargets([Sdf.Path(p) for p in filtered])

    t = Usd.TimeCode.Default()
    root_to_world = UsdGeom.Xformable(_require_prim(stage, root_path)).ComputeLocalToWorldTransform(t)
    world_to_root = root_to_world.GetInverse()
    floor_box = UsdGeom.BBoxCache(t, ["default", "render"]).ComputeRelativeBound(
        floor, _require_prim(stage, root_path)
    ).ComputeAlignedRange()
    along, across = (0, 1) if side in "NS" else (1, 0)
    hinge, tip = _flap_hinge_and_tip(mesh, floor_box.GetMidpoint(), across, world_to_root)
    reach = tip - hinge
    elevation_deg = math.degrees(math.atan2(reach[2], abs(reach[across])))
    axis = Gf.Vec3d(0.0)
    axis[along] = 1.0
    fold_in_sign = 1.0 if Gf.Cross(axis, reach)[2] > 0.0 else -1.0
    fold_in = 180.0 - elevation_deg - E2_OUTBOUND_BOX_FLAP_FOLD_IN_MARGIN_DEG
    fold_out = E2_OUTBOUND_BOX_FLAP_FOLD_OUT_DEG
    lower, upper = sorted((fold_in_sign * fold_in, -fold_in_sign * fold_out))

    base_to_world = Gf.Matrix4d().SetTranslate(hinge) * root_to_world
    hinge_w = base_to_world.ExtractTranslation()
    world_to_box = UsdGeom.Xformable(_require_prim(stage, box_path)).ComputeLocalToWorldTransform(t).GetInverse()

    body_to_world = UsdGeom.Xformable(flap).ComputeLocalToWorldTransform(t)
    world_to_body = body_to_world.GetInverse()
    joint = UsdPhysics.RevoluteJoint.Define(stage, f"{flap_path}/{E2_OUTBOUND_BOX_FLAP_HINGE}")
    joint.CreateBody0Rel().SetTargets([Sdf.Path(box_path)])
    joint.CreateBody1Rel().SetTargets([flap.GetPath()])
    joint.CreateAxisAttr("X" if along == 0 else "Y")
    joint.CreateLocalPos0Attr(Gf.Vec3f(world_to_box.Transform(hinge_w)))
    joint.CreateLocalRot0Attr(Gf.Quatf((base_to_world * world_to_box).ExtractRotationQuat()))
    joint.CreateLocalPos1Attr(Gf.Vec3f(world_to_body.Transform(hinge_w)))
    joint.CreateLocalRot1Attr(Gf.Quatf((base_to_world * world_to_body).ExtractRotationQuat()))
    joint.CreateLowerLimitAttr(lower)
    joint.CreateUpperLimitAttr(upper)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateTargetPositionAttr(0.0)
    drive.CreateStiffnessAttr(E2_OUTBOUND_BOX_FLAP_HINGE_STIFFNESS_NM_PER_RAD * math.pi / 180.0)
    drive.CreateDampingAttr(E2_OUTBOUND_BOX_FLAP_HINGE_DAMPING_NMS_PER_RAD * math.pi / 180.0)


E2_ROOM_WALLS = tuple(f"Structure/RoomShell/Wall_{side}" for side in "NSEW")
E2_WALL_DRESSING = (
    *(f"Structure/RoomShell/ISA_DadoEdge_{i}" for i in range(8)),
    *(f"Structure/RoomShell/BG_E2_R03_Wall_Base_{side}" for side in ("N_a", "N_b", "S_a", "E_a", "W_a")),
    *(f"Structure/RoomShell/BG_E2_R03_Wall_Corner_{c}" for c in ("NE_E", "NE_N", "SE_S", "SW_S", "SW_W")),
    "Structure/Shutter",
    "Background/BG_N/BG_E2_N_Sign_Extinguisher",
    "Background/BG_N/BG_E2_R03_Wall_Scuff_N",
    "Background/BG_S/BG_E2_S_KPIBoard",
    "Background/BG_S/BG_E2_S_FireHose",
    "Background/BG_S/BG_E2_S_Clock",
    "Background/BG_S/BG_E2_R03_Wall_Scuff_S",
    "Background/BG_S/BG_E2_S_Sign_FireHose",
    "Background/BG_S/BG_E2_S_Decal_Caution",
    "Background/BG_E/BG_E2_E_FireAlarm",
    "Background/BG_E/BG_E2_R03_Wall_Scuff_E",
    "Background/BG_W/BG_E2_W_Sign_HighVoltage",
    "Background/BG_W/BG_E2_W_Switchboard",
    "Background/BG_W/BG_E2_W_JunctionBox",
    "Background/BG_W/BG_E2_R03_Wall_Scuff_W",
)


def hide_e2_room_walls(stage: Usd.Stage, cell_root: str) -> list[str]:
    """Stop drawing the room's walls and the dressing on them; returns the names hidden."""

    from pxr import UsdGeom

    prims = [_require_prim(stage, f"{cell_root}/{path}") for path in E2_ROOM_WALLS + E2_WALL_DRESSING]
    for prim in prims:
        UsdGeom.Imageable(prim).MakeInvisible()
    return [prim.GetName() for prim in prims]



E2_ITEM_ASSET_USDS: dict[str, str] = {
    "dexbench_e2_belt_box_400x300x250": "box_400x300x250",
    "dexbench_e2_belt_box_410x310x280": "box_410x310x280",
    "dexbench_e2_belt_box_450x350x300": "box_450x350x300",
    "dexbench_e2_belt_bag_cotton": "polybag_bubble_cotton_deformable",
    "dexbench_e2_belt_bag_empty": "polybag_bubble_empty_deformable",
}


def e2_package_usd(package: str) -> Path:
    """Main USD of a pool package (object_library/<name>/usd/<name>.usd)."""

    return Path(runtime_usd(ASSETS_DIR / "object_library" / package / "usd" / f"{package}.usd"))


E2_CAGE_STACK_DIR = E2_ENV_DIR / "shell" / "cage_stack"
E2_CAGE_CARTON_COLLIDERS = {
    **{f"dexbench_e2_cage_carton_{box}": "box_krpost_fc_05" for box in ("l1b1", "l1b2", "l1f1", "l1f2")},
    **{f"dexbench_e2_cage_carton_{box}": "box_krpost_fc_04" for box in ("l2b1", "l2b2", "l2f1", "l2f2")},
}


def e2_item_usd(asset_name: str) -> Path:
    """Main USD of an E2 item asset: its pool package, or for a cage carton its shell layer."""

    if asset_name in E2_CAGE_CARTON_COLLIDERS:
        return Path(runtime_usd(E2_CAGE_STACK_DIR / f"{asset_name.removeprefix('dexbench_e2_')}.usda"))
    return e2_package_usd(E2_ITEM_ASSET_USDS[asset_name])


E2_CAGE_INNER_HALF_WIDTH_M = 0.566
E2_CAGE_INNER_DEPTH_M = 0.966
E2_CAGE_FLOOR_TOP_Z_M = 0.250
E2_CAGE_INNER_TOP_Z_M = 1.670

E2_CAGE_BOX_PACKAGE_SIZE_XYZ_M = {
    "box_krpost_fc_04": (0.4112, 0.310, 0.281),
    "box_krpost_fc_05": (0.4812, 0.360, 0.341),
}
E2_ITEM_SIZE_XYZ_M = {carton: E2_CAGE_BOX_PACKAGE_SIZE_XYZ_M[box] for carton, box in E2_CAGE_CARTON_COLLIDERS.items()}


E2_PARCEL_ASSET = "logistics_post_office_package"
E2_PARCEL_SIZE_XYZ_M = (0.272, 0.188, 0.160)
E2_LIFT_OBJECT_BALLAST_KG = 3.0
E2_LIFT_BOX_ASSET_PREFIX = "dexbench_e2_lift_box_"


@dataclass(frozen=True)
class E2LiftObject:
    """One object the lift_to_conveyor task can be run with: a box package of ``size_xyz_m``, or the parcel."""

    name: str
    """CLI name (``--object_box``) and the key of E2_LIFT_OBJECTS_BY_NAME."""
    size_xyz_m: tuple[float, float, float]
    """Long side, depth, height."""
    group: str
    """``train``, ``eval`` or ``stand-in``."""
    package: str | None = None
    """object_library package the box spawns from; None for the parcel stand-in."""

    @property
    def asset(self) -> str:
        """Registered Arena asset: the parcel's own, else the box object_library registers from this row."""
        return E2_PARCEL_ASSET if self.package is None else E2_LIFT_BOX_ASSET_PREFIX + self.name

    @property
    def footprint_xy_m(self) -> tuple[float, float]:
        return self.size_xyz_m[:2]


E2_LIFT_OBJECTS = (
    E2LiftObject("train_0", (0.410, 0.310, 0.280), "train", "box_410x310x280"),
    E2LiftObject("train_1", (0.480, 0.380, 0.340), "train", "box_480x380x340"),
    E2LiftObject("eval_0", (0.400, 0.300, 0.250), "eval", "box_400x300x250"),
    E2LiftObject("eval_1", (0.450, 0.350, 0.300), "eval", "box_450x350x300"),
    E2LiftObject("eval_2", (0.500, 0.400, 0.250), "eval", "box_500x400x250"),
    E2LiftObject("eval_3", (0.600, 0.400, 0.300), "eval", "box_600x400x300"),
    E2LiftObject("parcel", E2_PARCEL_SIZE_XYZ_M, "stand-in"),
)
E2_LIFT_OBJECTS_BY_NAME = {lift_object.name: lift_object for lift_object in E2_LIFT_OBJECTS}
E2_LIFT_OBJECT_SETS = {
    group: tuple(lift_object.name for lift_object in E2_LIFT_OBJECTS if lift_object.group == group)
    for group in ("train", "eval")
}
E2_DEFAULT_LIFT_OBJECT = "train_0"
E2_LIFT_OBJECT_AXIS = ObjectAxis(
    field="object_box",
    default=E2_DEFAULT_LIFT_OBJECT,
    objects=tuple(
        TaskObject(
            key=lift_object.name,
            asset=lift_object.asset,
            split=lift_object.group,
            label="parcel" if lift_object.package is None else "box",
            size_m=lift_object.size_xyz_m,
        )
        for lift_object in E2_LIFT_OBJECTS
    ),
    size_axes="long side, depth, height",
    scene_layout="261014",
    unlabelled_scene_layout="260827",
)

E2_INVOICE_OBJECT_AXIS = ObjectAxis(
    field="invoice_sheet",
    default="invoice_202x276",
    objects=(
        TaskObject(key="invoice_202x276", asset="PaperStackVBD", split="train", label="invoice sheet",
                   size_m=(0.202, 0.276)),
    ),
    size_axes="width, length",
)


E2_L4_PARCEL_ASSET_PREFIX = "dexbench_l4_parcel_"


@dataclass(frozen=True)
class E2L4Parcel:
    """A parcel of the DB-L4 episodes: a box package with a box collider, or the tet bag in a colour."""

    name: str
    barcode_axis: str
    """+x, -x, +y, -y, +z or -z, in the parcel's own frame."""
    package: str | None = None
    """Box: the object_library package. None for a polybag."""
    collider_size_xyz_m: tuple[float, float, float] | None = None
    """Box: its one collider, centred in XY on the package origin and standing on it."""
    tint_rgb: tuple[int, int, int] | None = None
    """Polybag: its film's colour, 0-255 sRGB."""
    nominal_mass_kg: float = 0.0
    """What it weighs unscaled [kg] (see above); every row sets it."""
    size_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Polybag: its (x, y, z) size over the tet bag's (see above)."""

    @property
    def is_polybag(self) -> bool:
        return self.package is None

    @property
    def asset(self) -> str:
        """Box: the Arena asset object_library registers from this row."""
        assert not self.is_polybag, f"{self.name} is a polybag; it is not an Arena asset"
        return E2_L4_PARCEL_ASSET_PREFIX + self.name

    @property
    def waybill_albedo(self) -> str:
        """Polybag: its albedo, relative to the tet bag package's ``usd/`` directory."""
        assert self.is_polybag, f"{self.name} is a box; its waybill is in its package"
        return f"textures/green_albedo_waybill_{self.name}.jpg"


E2_L4_BOX_FRICTION = 0.4

E2_L4_PARCELS = (
    E2L4Parcel("box_1", "-z", package="box_1", collider_size_xyz_m=(0.271, 0.264, 0.222), nominal_mass_kg=1.0),
    E2L4Parcel("box_2", "-z", package="box_2", collider_size_xyz_m=(0.339, 0.271, 0.176),
               nominal_mass_kg=1.0),
    E2L4Parcel("box_3", "-z", package="box_3", collider_size_xyz_m=(0.324, 0.265, 0.197),
               nominal_mass_kg=1.0),
    E2L4Parcel("box_4", "+z", package="box_4", collider_size_xyz_m=(0.215, 0.195, 0.095),
               nominal_mass_kg=1.0),
    E2L4Parcel("box_5", "+z", package="box_5", collider_size_xyz_m=(0.272, 0.188, 0.160),
               nominal_mass_kg=1.0),
    E2L4Parcel("polybag_1", "+z", tint_rgb=(56, 56, 58), nominal_mass_kg=0.140),
    E2L4Parcel("polybag_2", "+z", tint_rgb=(85, 60, 59), nominal_mass_kg=0.100,
               size_scale=(0.9, 0.9, 0.5)),
    E2L4Parcel("polybag_3", "+z", tint_rgb=(18, 60, 93), nominal_mass_kg=0.120),
    E2L4Parcel("polybag_4", "+z", tint_rgb=(38, 81, 83), nominal_mass_kg=0.080,
               size_scale=(0.9, 0.9, 0.5)),
    E2L4Parcel("polybag_5", "+z", tint_rgb=(71, 17, 36), nominal_mass_kg=0.300),
)
E2_L4_PARCELS_BY_NAME = {parcel.name: parcel for parcel in E2_L4_PARCELS}


def polybag_size_scale(name: str) -> tuple[float, float, float] | None:
    """A bag's ``PolybagObjectCfg.tet_size_scale``: its row's ``size_scale``, None for the asset's own, so the bags
    of one size share one template."""
    scale = E2_L4_PARCELS_BY_NAME[name].size_scale
    return None if scale == (1.0, 1.0, 1.0) else scale


E2_CAGE_BOX_MOVABLE_ASSET_SUFFIX = "_kinematic"
E2_CAGE_BOX_ASSETS = tuple(sorted(E2_CAGE_CARTON_COLLIDERS))


def e2_cage_box_asset(asset: str) -> str:
    """The kinematic twin object_library registers for a cage-box asset."""

    return asset + E2_CAGE_BOX_MOVABLE_ASSET_SUFFIX


E2_HIGH_BAY_LIGHT_ASSET = "dexbench_high_bay_light"
E2_PENDANT_LIGHT_ASSET = "dexbench_pendant_light"
E2_STRIP_LIGHT_ASSET = "dexbench_strip_light"
E2_REFLECT_CARD_ASSET = "dexbench_reflect_card"
