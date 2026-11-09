"""Where the E2 logistics cell's props, box stack and lights go, and how the stack varies."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from dexbench_arena.assets.e2_logistics_cell import (
    E2_CAGE_FLOOR_TOP_Z_M,
    E2_CAGE_INNER_DEPTH_M,
    E2_CAGE_INNER_HALF_WIDTH_M,
    E2_CAGE_INNER_TOP_Z_M,
    E2_HIGH_BAY_LIGHT_ASSET,
    E2_ITEM_SIZE_XYZ_M,
    E2_PARCEL_CAGE_SLOT_XY_M,
    E2_PARCEL_CAGE_SLOT_YAW_DEG,
    E2_PENDANT_LIGHT_ASSET,
    E2_REFLECT_CARD_ASSET,
    E2_STRIP_LIGHT_ASSET,
)


E2_RENDER_SETTINGS: dict[str, bool | int | float] = {
    "rtx:directLighting:sampledLighting:maxRayUnexposedIntensity": 27519.998046875,
    "rtx:indirectDiffuse:maxRayUnexposedIntensity": 27519.998046875,
    "rtx:pathtracing:fireflyFilter:maxPerEmissiveUnexposedIntensity": 13759.9990234375,
    "rtx:pathtracing:fireflyFilter:maxUnexposedIntensityPerSample": 13759.9990234375,
    "rtx:pathtracing:fireflyFilter:maxUnexposedIntensityPerSampleDiffuse": 13759.9990234375,
    "rtx:pathtracing:maxBounces": 8,
    "rtx:pathtracing:maxSpecularAndTransmissionBounces": 16,
    "rtx:post:tonemap:exposureTime": 0.02,
    "rtx:post:tonemap:fNumber": 5.0,
    "rtx:post:tonemap:filmIso": 430.0,
    "rtx:post:tonemap:irayReinhard:burnHighlights": 0.2,
    "rtx:post:tonemap:irayReinhard:crushBlacks": 0.0,
    "rtx:post:tonemap:irayReinhard:saturation": 0.9,
    "rtx:post:tonemap:op": 7,
    "rtx:raytracing:fractionalCutoutOpacity": True,
    "rtx:raytracing:inscattering:maxRTSampleUnexposedIntensity": 880639.9375,
    "rtx:rtpt:maxSpecularAndTransmissionBounces": 16,
    "rtx:translucency:maxRayUnexposedIntensity": 82559.9921875,
    "rtx:translucency:maxRefractionBounces": 16,
}




@dataclass(frozen=True)
class E2ItemPlacement:
    name: str
    """Scene entity and prim name."""
    asset: str
    """Registered Arena asset, a key of E2_ITEM_ASSET_USDS or, for a cage box, of E2_CAGE_CARTON_COLLIDERS."""
    position_xyz: tuple[float, float, float]
    yaw_deg: float = 0.0
    scale: float = 1.0
    """Uniform scale applied to the usdc; 1.0 is as delivered."""

    @property
    def rotation_xyzw(self) -> tuple[float, float, float, float]:
        from isaaclab_arena.utils.yaw import rotate_quat_by_yaw

        return rotate_quat_by_yaw((0.0, 0.0, 0.0, 1.0), math.radians(self.yaw_deg))




def e2_footprint_half_extents_xy(yaw_rad: float, half_long: float, half_short: float) -> tuple[float, float]:
    """World-axis half-extents of a ``2 half_long x 2 half_short`` footprint turned to ``yaw_rad``."""

    return (
        abs(half_long * math.cos(yaw_rad)) + abs(half_short * math.sin(yaw_rad)),
        abs(half_long * math.sin(yaw_rad)) + abs(half_short * math.cos(yaw_rad)),
    )


E2_CAGE_BOX_DELIVERED_LOCAL_PLACEMENTS = (
    E2ItemPlacement("cage_box_l1b1", "dexbench_e2_cage_carton_l1b1", (-0.240, 0.590, 0.250)),
    E2ItemPlacement("cage_box_l1b2", "dexbench_e2_cage_carton_l1b2", (0.240, 0.590, 0.250)),
    E2ItemPlacement("cage_box_l1f1", "dexbench_e2_cage_carton_l1f1", (-0.240, 0.230, 0.250)),
    E2ItemPlacement("cage_box_l1f2", "dexbench_e2_cage_carton_l1f2", (0.240, 0.230, 0.250)),
    E2ItemPlacement("cage_box_l2b1", "dexbench_e2_cage_carton_l2b1", (-0.205, 0.515, 0.590)),
    E2ItemPlacement("cage_box_l2b2", "dexbench_e2_cage_carton_l2b2", (0.205, 0.515, 0.590)),
    E2ItemPlacement("cage_box_l2f1", "dexbench_e2_cage_carton_l2f1", (-0.205, 0.205, 0.590)),
    E2ItemPlacement("cage_box_l2f2", "dexbench_e2_cage_carton_l2f2", (0.205, 0.205, 0.590)),
)

E2_CAGE_STACK_ANCHOR_LOCAL_XYZ_M = (0.0, 0.050, E2_CAGE_FLOOR_TOP_Z_M)
E2_OBJECT_SUPPORT_BOX_NAMES = ("cage_box_l2f1", "cage_box_l2f2", "cage_box_l2b1", "cage_box_l2b2")
E2_OBJECT_SUPPORT_MARGIN_M = 0.010


@dataclass(frozen=True)
class E2CageLayoutParams:
    """What may vary about the box stack. The defaults reproduce the delivery."""

    box_scale: float = 1.0
    """Uniform scale of every cage box; the arrangement scales with it about the stack anchor."""
    stack_offset_depth_m: float = 0.0
    """Slide the whole stack toward the back wall (cage-local +Y; world -X as the cage stands)."""
    stack_offset_lateral_m: float = 0.0
    """Slide the whole stack across the cage (cage-local +X; world +Y as the cage stands)."""


@dataclass(frozen=True)
class E2CageLayout:
    """A box stack, in world coordinates, and where an object rests on it."""

    params: E2CageLayoutParams
    placements: tuple[E2ItemPlacement, ...]
    """The eight cage boxes, scaled and placed."""
    support_top_z_m: float
    """Height of those top faces; an object's origin (bottom-face centre) goes here."""
    support_local_x: tuple[float, float]
    """The tier's top in the cage's frame: across the width ..."""
    support_local_y: tuple[float, float]
    """... and along the depth, opening side first."""
    front_row_local_y: float
    """Depth of the front pair's centre line -- the seam a small object lies across."""

    @property
    def support_xy_bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """World AABB of the upper tier's top faces, ((min_x, min_y), (max_x, max_y))."""

        corners = [
            _cage_local_to_world((corner_x, corner_y, self.support_top_z_m))
            for corner_x in self.support_local_x
            for corner_y in self.support_local_y
        ]
        return (
            _rounded((min(c[0] for c in corners), min(c[1] for c in corners))),
            _rounded((max(c[0] for c in corners), max(c[1] for c in corners))),
        )

    def object_pose(
        self, footprint_xy_m: tuple[float, float], front_clearance_m: float = 0.0
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        """Where an object of ``footprint_xy_m`` (long side, depth) rests on this stack."""

        long_side, depth = (float(value) for value in footprint_xy_m)
        margin = E2_OBJECT_SUPPORT_MARGIN_M
        clearance = max(margin, float(front_clearance_m))
        (x_min, x_max), (y_min, y_max) = self.support_local_x, self.support_local_y
        tolerance = 1e-6
        if long_side + 2.0 * margin > (x_max - x_min) + tolerance:
            raise ValueError(
                f"E2 cage layout {self.params}: an object {long_side:.3f} m long (plus {margin} m margin) does not "
                f"fit the {x_max - x_min:.3f} m wide upper tier."
            )
        centre_y = max(self.front_row_local_y, y_min + clearance + depth / 2.0)
        if centre_y + depth / 2.0 + margin > y_max + tolerance:
            raise ValueError(
                f"E2 cage layout {self.params}: an object {depth:.3f} m deep, set {clearance:.3f} m behind the "
                f"tier's front edge, reaches {centre_y + depth / 2.0:.3f} m; the tier ends at {y_max:.3f} m "
                f"(margin {margin} m)."
            )
        position = _rounded(_cage_local_to_world(((x_min + x_max) / 2.0, centre_y, self.support_top_z_m)))
        return position, _rounded(self.placements[0].rotation_xyzw, 8)


def _cage_local_to_world(local_xyz: tuple[float, float, float]) -> tuple[float, float, float]:
    import torch
    from isaaclab_arena.utils.yaw import rotate_points_by_yaw

    cage_x, cage_y = E2_PARCEL_CAGE_SLOT_XY_M
    point = rotate_points_by_yaw(torch.tensor([local_xyz], dtype=torch.float64), math.radians(E2_PARCEL_CAGE_SLOT_YAW_DEG))[0]
    return (cage_x + float(point[0]), cage_y + float(point[1]), float(point[2]))


def _rounded(values, digits: int = 6) -> tuple:
    """Strip the 1e-17 dust a 90 deg rotation leaves so layouts compare as the numbers they mean."""

    return tuple(round(float(value), digits) for value in values)


def e2_cage_layout(params: E2CageLayoutParams | None = None) -> E2CageLayout:
    """Lay the cage boxes out for ``params``."""

    params = params if params is not None else E2CageLayoutParams()
    scale = float(params.box_scale)
    if not scale > 0.0:
        raise ValueError(f"E2 cage layout: box_scale must be positive, got {params.box_scale!r}.")
    anchor_x, anchor_y, anchor_z = E2_CAGE_STACK_ANCHOR_LOCAL_XYZ_M
    shift_x, shift_y = float(params.stack_offset_lateral_m), float(params.stack_offset_depth_m)

    local: dict[str, tuple[float, float, float]] = {}
    sizes: dict[str, tuple[float, float, float]] = {}
    for box in E2_CAGE_BOX_DELIVERED_LOCAL_PLACEMENTS:
        px, py, pz = box.position_xyz
        local[box.name] = (
            anchor_x + shift_x + (px - anchor_x) * scale,
            anchor_y + shift_y + (py - anchor_y) * scale,
            anchor_z + (pz - anchor_z) * scale,
        )
        sizes[box.name] = tuple(extent * scale for extent in E2_ITEM_SIZE_XYZ_M[box.asset])

    tolerance = 1e-6
    for name, (x, y, z) in local.items():
        size_x, size_y, size_z = sizes[name]
        if abs(x) + size_x / 2.0 > E2_CAGE_INNER_HALF_WIDTH_M + tolerance:
            raise ValueError(
                f"E2 cage layout {params}: {name} reaches {abs(x) + size_x / 2.0:.3f} m from the cage's "
                f"centre line; the side walls are {E2_CAGE_INNER_HALF_WIDTH_M} m out."
            )
        if y - size_y / 2.0 < -tolerance:
            raise ValueError(
                f"E2 cage layout {params}: {name}'s front face is {size_y / 2.0 - y:.3f} m past the opening."
            )
        if y + size_y / 2.0 > E2_CAGE_INNER_DEPTH_M + tolerance:
            raise ValueError(
                f"E2 cage layout {params}: {name}'s back face is {y + size_y / 2.0:.3f} m in; the back wall "
                f"is at {E2_CAGE_INNER_DEPTH_M} m."
            )
        if z + size_z > E2_CAGE_INNER_TOP_Z_M + tolerance:
            raise ValueError(
                f"E2 cage layout {params}: {name}'s top is at {z + size_z:.3f} m; the cage's top is at "
                f"{E2_CAGE_INNER_TOP_Z_M} m."
            )

    tier = E2_OBJECT_SUPPORT_BOX_NAMES
    size_x, size_y, size_z = sizes[tier[0]]
    top_z = local[tier[0]][2] + size_z
    for name in tier:
        if sizes[name] != sizes[tier[0]] or abs(local[name][2] + size_z - top_z) > tolerance:
            raise ValueError(f"E2 cage layout {params}: {name} does not make a level top with the rest of the tier.")
    xs = sorted(local[name][0] for name in tier)
    ys = sorted(local[name][1] for name in tier)
    if abs((xs[-1] - xs[0]) - size_x) > 0.002 or abs((ys[-1] - ys[0]) - size_y) > 0.002:
        raise ValueError(f"E2 cage layout {params}: the upper tier's boxes do not touch in a 2 x 2.")
    support_local_x = (xs[0] - size_x / 2.0, xs[-1] + size_x / 2.0)
    support_local_y = (ys[0] - size_y / 2.0, ys[-1] + size_y / 2.0)

    placements = tuple(
        replace(
            box,
            position_xyz=_rounded(_cage_local_to_world(local[box.name])),
            yaw_deg=box.yaw_deg + E2_PARCEL_CAGE_SLOT_YAW_DEG,
            scale=scale,
        )
        for box in E2_CAGE_BOX_DELIVERED_LOCAL_PLACEMENTS
    )
    return E2CageLayout(
        params=params,
        placements=placements,
        support_top_z_m=round(top_z, 6),
        support_local_x=_rounded(support_local_x),
        support_local_y=_rounded(support_local_y),
        front_row_local_y=round(ys[0], 6),
    )

E2_CAGE_BOX_ALT_SET_SUFFIX = "_alt{index}"
E2_CAGE_BOX_PARK_OFFSET_XYZ_M = (-30.0, -30.0, 0.0)
E2_CAGE_BOX_PARK_SET_SPACING_M = 3.0


def e2_cage_box_entity_name(box_name: str, set_index: int) -> str:
    """Scene entity name of ``box_name`` in box set ``set_index``; set 0 keeps the plain name."""

    return box_name if set_index == 0 else box_name + E2_CAGE_BOX_ALT_SET_SUFFIX.format(index=set_index)


def e2_cage_box_park_position(
    position_xyz: tuple[float, float, float], set_index: int
) -> tuple[float, float, float]:
    """Where a box of set ``set_index`` waits while another set stands in the cage."""

    dx, dy, dz = E2_CAGE_BOX_PARK_OFFSET_XYZ_M
    return _rounded((
        position_xyz[0] + dx - E2_CAGE_BOX_PARK_SET_SPACING_M * set_index,
        position_xyz[1] + dy,
        position_xyz[2] + dz,
    ))


def e2_cage_stack_scales(nominal: E2CageLayoutParams, alt_scales) -> tuple[float, ...]:
    """The box sets' scales: the nominal first, then the alternates in the order given."""

    scales = [float(nominal.box_scale)]
    for scale in alt_scales:
        scale = float(scale)
        if all(abs(scale - have) > 1e-9 for have in scales):
            scales.append(scale)
    return tuple(scales)


def e2_cage_stack_params_for_draw(
    nominal: E2CageLayoutParams, scales: tuple[float, ...], draw
) -> tuple[int, E2CageLayoutParams]:
    """Map one ``[set_pick, ddepth, dlateral]`` draw to the set that stands in the cage and its layout."""

    set_pick, ddepth, dlateral = (float(value) for value in draw)
    index = min(max(int(math.floor(set_pick * len(scales))), 0), len(scales) - 1)
    params = replace(
        nominal,
        box_scale=scales[index],
        stack_offset_depth_m=nominal.stack_offset_depth_m + ddepth,
        stack_offset_lateral_m=nominal.stack_offset_lateral_m + dlateral,
    )
    return index, params


E2_BELT_ITEM_PLACEMENTS = (
    E2ItemPlacement("belt_box_01", "dexbench_e2_belt_box_400x300x250", (4.0125, 0.8, 0.800), 86.0),
    E2ItemPlacement("belt_box_02", "dexbench_e2_belt_box_410x310x280", (2.325, 0.8, 0.812), 92.0),
    E2ItemPlacement("belt_box_03", "dexbench_e2_belt_box_450x350x300", (2.72, 0.8, 0.812), 95.0),
    E2ItemPlacement("belt_bag_t1", "dexbench_e2_belt_bag_cotton", (2.6833, 0.828019, 0.706325), 0.0),
    E2ItemPlacement("belt_bag_t2", "dexbench_e2_belt_bag_empty", (2.9568, 0.939273, 0.701), 0.0),
)


def e2_item_placements(cage_layout: E2CageLayout | None = None) -> tuple[E2ItemPlacement, ...]:
    """Every prop the environment spawns, in world coordinates: the cage's box stack, then the belt items."""

    if cage_layout is None:
        cage_layout = e2_cage_layout()
    return (*cage_layout.placements, *E2_BELT_ITEM_PLACEMENTS)


E2_HIGH_BAY_Z_M = 3.32
E2_HIGH_BAY_XY_M = {
    "high_bay_light_01": (1.0, 1.0),
    "high_bay_light_02": (3.0, 1.0),
    "high_bay_light_03": (5.0, 1.0),
    "high_bay_light_04": (1.0, 3.0),
    "high_bay_light_05": (3.0, 3.0),
    "high_bay_light_06": (5.0, 3.0),
}
E2_HIGH_BAY_POSITION_RANGE_M = ((-0.4, -0.4, -0.25), (0.4, 0.4, 0.0))
E2_PENDANT_POSITION_RANGE_M = ((-0.4, -0.4, -0.25), (0.4, 0.4, 0.15))


@dataclass(frozen=True)
class E2LightPlacement:
    """A light the environment spawns: its scene name, Arena asset and pose."""

    name: str
    asset: str
    """Registered Arena asset, one of the cell module's E2_*_LIGHT_ASSET / E2_REFLECT_CARD_ASSET."""
    position_xyz: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    """(x, y, z, w). Every one is a RectLight, which emits along its local -Z."""
    size_m: tuple[float, float] | None = None
    """(width, height) where it differs from the asset's."""
    position_range_m: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
    """(low, high) of the position variation's [dx, dy, dz] offset [m]; see E2_HIGH_BAY_POSITION_RANGE_M."""


E2_TASK_LIGHTS = (
    E2LightPlacement(
        "task_light_pend_sort", E2_PENDANT_LIGHT_ASSET, (3.45, 2.28, 2.45), position_range_m=E2_PENDANT_POSITION_RANGE_M
    ),
    E2LightPlacement(
        "task_light_pend_belt", E2_PENDANT_LIGHT_ASSET, (3.0, 0.8, 2.45), position_range_m=E2_PENDANT_POSITION_RANGE_M
    ),
    E2LightPlacement(
        "task_light_strip_bench", E2_STRIP_LIGHT_ASSET, (5.0, 3.72, 1.615), (-0.32556815, 0.0, 0.0, 0.94551858),
        position_range_m=((-0.1, -0.03, -0.02), (0.1, 0.03, 0.02)),
    ),
    E2LightPlacement(
        "reflect_card_n", E2_REFLECT_CARD_ASSET, (5.45, 3.93, 1.55), (-0.70710678, 0.0, 0.0, 0.70710678),
        position_range_m=((-0.3, -0.1, -0.15), (0.3, 0.05, 0.15)),
    ),
    E2LightPlacement(
        "reflect_card_e", E2_REFLECT_CARD_ASSET, (5.93, 3.45, 1.55), (0.5, 0.5, 0.5, 0.5),
        position_range_m=((-0.1, -0.3, -0.15), (0.05, 0.3, 0.15)),
    ),
    E2LightPlacement(
        "reflect_card_top", E2_REFLECT_CARD_ASSET, (5.6, 3.5, 2.6), size_m=(1.2, 1.2),
        position_range_m=((-0.3, -0.3, -0.15), (0.3, 0.3, 0.15)),
    ),
)
E2_LIGHTS = (
    *(
        E2LightPlacement(
            name, E2_HIGH_BAY_LIGHT_ASSET, (x, y, E2_HIGH_BAY_Z_M), position_range_m=E2_HIGH_BAY_POSITION_RANGE_M
        )
        for name, (x, y) in E2_HIGH_BAY_XY_M.items()
    ),
    *E2_TASK_LIGHTS,
)
