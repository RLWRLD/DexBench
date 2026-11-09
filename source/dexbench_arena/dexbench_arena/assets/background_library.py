# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""DexBench-native background assets."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.utils import clone as _clone_spawner
from isaaclab.utils import configclass
from isaaclab_arena.assets.background_library import LibraryBackground
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.utils.yaw import rotate_quat_by_yaw

from dexbench_arena import ASSETS_DIR
from dexbench_arena.assets.scene_bundles import runtime_usd
from dexbench_arena.assets.fixture_looks import fixture_look_spawner
from dexbench_arena.assets.e4_hotel_fb_lighting import (
    E4_LIGHTING_ASSET,
    E4_LIGHTING_LAYER,
    apply_light_edits,
    check_light_edits,
)
from dexbench_arena.assets.assembly_line_e3_m5_station import apply_m5_station_edits
from dexbench_arena.assets.db_m3_station import apply_m3_station_edits
from dexbench_arena.assets.db_m1_station import apply_m1_station_edits
from dexbench_arena.assets.db_m4_station import apply_m4_station_edits
from dexbench_arena.assets.usd_file_spawn import spawn_usd_file
from dexbench_arena.assets.e2_logistics_cell import (
    E2_CANONICAL_LAYER,
    E2_OUTBOUND_BOX_BODY,
    E2_PACKING_BENCH_BODY,
    E2_ROOM_WALLS,
    E2_WALL_DRESSING,
    apply_e2_cell_edits,
    clear_e2_bench_station,
    e2_frozen_bodies,
    hide_e2_room_walls,
)

from dexbench_arena.layouts.db_m1_layout import E3_CELL_XYZ, E3_CELL_YAW_DEG
from dexbench_arena.layouts.db_m4_layout import E3_CELL_XYZ as E3_M4_CELL_XYZ
from dexbench_arena.layouts.db_m4_layout import E3_CELL_YAW_DEG as E3_M4_CELL_YAW_DEG
from dexbench_arena.layouts.e1_retail_fc_layout import FC_260821_LAYOUT_SPEC
from dexbench_arena.layouts.retail_v1_look import (
    RETAIL_V1_DIR,
    RETAIL_V1_LIGHT_COLOR_TEMPERATURE_RANGE_K,
    RETAIL_V1_LIGHT_INTENSITY_SCALE_RANGE,
    RETAIL_V1_LIGHTING_ASSET,
)

if TYPE_CHECKING:
    from pxr import Usd


E2_WALL_FILL_PRIM_PATH = "/World/E2WallFill"
E2_WALL_FILL_INTENSITY = 40.0


def _spawn_e2_cell_impl(
    prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn the delivered canonical layer and turn it into DexBench's cell."""

    from isaaclab_arena.assets.background import _deinstance_nested_physics

    prim = spawn_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation, **kwargs)
    _deinstance_nested_physics(prim)
    apply_e2_cell_edits(prim.GetStage(), str(prim.GetPath()), tuple(getattr(cfg, "moving_belts", ())))
    return prim


_spawn_e2_cell = _clone_spawner(_spawn_e2_cell_impl)


@configclass
class E2CellLayerCfg(UsdFileCfg):
    """``UsdFileCfg`` for the E2 cell, carrying the cell edits a task chooses: which belts it moves
    and, at the pack-invoice bench station, the outbound box's size."""

    func: Callable = _spawn_e2_cell
    moving_belts: tuple[str, ...] = ()
    """The conveyor's belts whose top run stays a kinematic rigid body (``apply_e2_cell_edits``)."""
    outbound_box_scale_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """The bench station's outbound box size (``clear_e2_bench_station``), which the bench-station spawners read."""


def _clear_bench_station(prim, cfg) -> None:
    clear_e2_bench_station(prim.GetStage(), str(prim.GetPath()), cfg.outbound_box_scale_xyz)


@_clone_spawner
def _spawn_e2_cell_bench_station(
    prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn the E2 cell for the pack-invoice station: floor in front of the bench cleared, box held."""

    prim = _spawn_e2_cell_impl(prim_path, cfg, translation, orientation, **kwargs)
    _clear_bench_station(prim, cfg)
    return prim


@_clone_spawner
def _spawn_e2_cell_bench_station_hiding_room_shell(
    prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """``_spawn_e2_cell_bench_station`` with the room's walls hidden for a teleop session."""

    prim = _spawn_e2_cell_impl(prim_path, cfg, translation, orientation, **kwargs)
    _clear_bench_station(prim, cfg)
    _hide_e2_room_shell(prim)
    return prim


def _hide_room_shell_walls(prim) -> list[str]:
    """Stop drawing the ``Structure/RoomShell/Geom/Wall_*`` prims under ``prim``; the names hidden."""

    from pxr import UsdGeom

    geom = prim.GetStage().GetPrimAtPath(prim.GetPath().AppendPath("Structure/RoomShell/Geom"))
    hidden = []
    if geom.IsValid():
        for wall in geom.GetChildren():
            if wall.GetName().startswith("Wall_"):
                UsdGeom.Imageable(wall).MakeInvisible()
                hidden.append(wall.GetName())
    return hidden


@_clone_spawner
def _spawn_e2_cell_hiding_room_shell(
    prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn the E2 cell with its walls hidden and their light put back."""

    prim = _spawn_e2_cell_impl(prim_path, cfg, translation, orientation, **kwargs)
    _hide_e2_room_shell(prim)
    return prim


def _hide_e2_room_shell(prim):
    """The walls-hidden half of ``_spawn_e2_cell_hiding_room_shell``: hide them, add the fill."""

    from pxr import Gf, UsdLux

    stage = prim.GetStage()
    hide_e2_room_walls(stage, str(prim.GetPath()))

    if not stage.GetPrimAtPath(E2_WALL_FILL_PRIM_PATH).IsValid():
        fill = UsdLux.DomeLight.Define(stage, E2_WALL_FILL_PRIM_PATH)
        fill.CreateIntensityAttr(E2_WALL_FILL_INTENSITY)
        fill.CreateColorAttr(Gf.Vec3f(1.0, 0.97, 0.92))
        print(
            f"E2 cell: hid the {len(E2_ROOM_WALLS)} walls and {len(E2_WALL_DRESSING)} pieces of wall dressing "
            "(colliders kept, floor still drawn); "
            f"dome fill {E2_WALL_FILL_INTENSITY:g} at {E2_WALL_FILL_PRIM_PATH} for the lost bounce."
        )
    return prim


def _spawn_e3_cell_impl(
    prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None,
    orientation: tuple[float, float, float, float] | None,
    edits: Callable[[Usd.Stage, str], None],
    **kwargs,
):
    """Spawn the E3 assembly line, then author a station's ``edits(stage, cell_root)`` on the live stage."""

    from isaaclab_arena.assets.background import _deinstance_nested_physics

    prim = spawn_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation, **kwargs)
    _deinstance_nested_physics(prim)
    edits(prim.GetStage(), str(prim.GetPath()))
    return prim


@_clone_spawner
def _spawn_assembly_line_e3_m1(
    prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn the E3 assembly line with the DB-M1 cell edits (:func:`_spawn_e3_cell_impl`)."""

    return _spawn_e3_cell_impl(prim_path, cfg, translation, orientation, apply_m1_station_edits, **kwargs)


@_clone_spawner
def _spawn_assembly_line_e3_m4(
    prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn the E3 assembly line with the DB-M4 station edits (:func:`_spawn_e3_cell_impl`)."""

    return _spawn_e3_cell_impl(prim_path, cfg, translation, orientation, apply_m4_station_edits, **kwargs)


@_clone_spawner
def _spawn_assembly_line_e3_m3(prim_path, cfg, translation=None, orientation=None, **kwargs):
    return _spawn_e3_cell_impl(prim_path, cfg, translation, orientation, apply_m3_station_edits, **kwargs)


class _AssemblyLineE3StationBackground(LibraryBackground):
    """The E3 assembly line edited at spawn for one DB-M station; a subclass names its pose and spawner."""

    tags = ["background", "manufacturing"]
    usd_path = runtime_usd(ASSETS_DIR / "env" / "e3_manufacturing" / "assembly_line" / "assembly_line.usda")
    object_min_z = -0.05

    def __init__(self):
        super().__init__()

    def _get_spawn_cfg(self, activate_contact_sensors: bool = False):
        """Keep the editing spawner, which ``Background`` would replace with its own."""

        cfg = super()._get_spawn_cfg(activate_contact_sensors)
        cfg.func = type(self).spawn_cfg_addon["func"]
        return cfg


@register_asset
class DexBenchAssemblyLineE3M1Background(_AssemblyLineE3StationBackground):
    """The E3 assembly line as the DB-M1 washer cell (``db_m1_station``)."""

    name = "dexbench_assembly_line_e3_m1"
    initial_pose = Pose(
        position_xyz=E3_CELL_XYZ,
        rotation_xyzw=rotate_quat_by_yaw((0.0, 0.0, 0.0, 1.0), math.radians(E3_CELL_YAW_DEG)),
    )
    spawn_cfg_addon = {"func": _spawn_assembly_line_e3_m1}


@register_asset
class DexBenchAssemblyLineE3M3Background(_AssemblyLineE3StationBackground):
    """M3 replaces the washer bay in M1's room frame, with its own slot edits."""

    name = "dexbench_assembly_line_e3_m3"
    initial_pose = DexBenchAssemblyLineE3M1Background.initial_pose
    spawn_cfg_addon = {"func": _spawn_assembly_line_e3_m3}


@register_asset
class DexBenchAssemblyLineE3M4Background(_AssemblyLineE3StationBackground):
    """The E3 assembly line as the DB-M4 leveling-leg station (``db_m4_station``)."""

    name = "dexbench_assembly_line_e3_m4"
    initial_pose = Pose(
        position_xyz=E3_M4_CELL_XYZ,
        rotation_xyzw=rotate_quat_by_yaw((0.0, 0.0, 0.0, 1.0), math.radians(E3_M4_CELL_YAW_DEG)),
    )
    spawn_cfg_addon = {"func": _spawn_assembly_line_e3_m4}


@register_asset
class DexBenchRetailStoreE1Fc260821Background(LibraryBackground):
    """Reviewed 260821 E1 room shell."""

    name = "dexbench_retail_store_e1_fc_260821"
    tags = ["background", "retail", "fixture-composability"]
    layout_spec = FC_260821_LAYOUT_SPEC
    usd_path = runtime_usd(
        ASSETS_DIR
        / "env"
        / "e1_retail"
        / "fc"
        / "shell"
        / "room_shell.usdc"
    )
    initial_pose = Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
    object_min_z = -0.05

    def __init__(self, look=None):
        """
        Args:
            look: The room look (retail_v1_look.FixtureLook): the
                shell's walls, floor and ceiling stop being drawn and keep colliding, and the
                look is drawn over them. None draws the shell as it is.
        """
        self.look = look
        super().__init__()

    def _get_spawn_cfg(self, activate_contact_sensors: bool = False):
        cfg = super()._get_spawn_cfg(activate_contact_sensors)
        if self.look is not None:
            cfg.func = fixture_look_spawner(cfg.func, self.look.usd_path)
        return cfg


@register_asset
class DexBenchLogisticsCenterE2CellBackground(LibraryBackground):
    """PhysicsSim Lab E2 logistics centre, delivery 261014, as DexBench's working cell."""

    name = "dexbench_logistics_center_e2_cell"
    tags = ["background"]
    usd_path = str(E2_CANONICAL_LAYER)
    initial_pose = Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
    object_min_z = -0.05
    spawn_cfg_addon = {"func": _spawn_e2_cell}

    def __init__(
        self,
        hide_room_shell: bool = False,
        moving_belts: tuple[str, ...] = (),
        bench_station: bool = False,
        outbound_box_scale_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ):
        """
        Args:
            moving_belts: The conveyor's belts ("HB2", "FB1") whose top run
                stays a kinematic rigid body, for a task that moves the belt
                itself; DB-L1 moves HB2. Empty, the default, stands the whole
                conveyor still.
            hide_room_shell: Stop drawing the room's four walls, and add a dome
                fill to replace the light they were bouncing. The cell is small
                and its walls sit between a viewpoint inside it and the
                workspace, so a session driven from in there -- teleop -- spends
                most of its time looking through them. Rendering only: the walls
                keep colliding, and the dressing on them is hidden with them. The floor
                is always kept, walls or not; the
                spawner's docstring has the reasons behind both halves. Off by
                default, because for anything that reads the scene from outside
                the walls belong in the picture and the delivered lighting is
                what an evaluation run should see. ``--teleop_session`` asks for
                it, and a teleop session records no camera frames by default,
                so this normally only changes what the operator sees; a
                recording run that does add --enable_cameras
                would capture the cell without its walls and with the fill.
            bench_station: The scene is the pack-invoice station at the packing bench: the
                two floor distractors where the robot stands there are taken out, the
                packing bench becomes a kinematic body and the outbound box a fixed-base
                articulation with hinged flaps, both moved by the station's object
                references (``clear_e2_bench_station``).
            outbound_box_scale_xyz: Resize the station's outbound box along its x, y and
                height (``clear_e2_bench_station``); ones are the delivered box. Only with
                ``bench_station``.
        """

        self.bench_station = bool(bench_station)
        self.outbound_box_scale_xyz = tuple(float(v) for v in outbound_box_scale_xyz)
        if not self.bench_station and self.outbound_box_scale_xyz != (1.0, 1.0, 1.0):
            raise ValueError("outbound_box_scale_xyz sizes the bench station's box; it needs bench_station=True")

        self.moving_belts = tuple(moving_belts)
        e2_frozen_bodies(self.moving_belts)
        spawner = {
            (False, False): None,
            (True, False): _spawn_e2_cell_hiding_room_shell,
            (False, True): _spawn_e2_cell_bench_station,
            (True, True): _spawn_e2_cell_bench_station_hiding_room_shell,
        }[(bool(hide_room_shell), bool(bench_station))]
        if spawner is not None:
            self.spawn_cfg_addon = {**self.spawn_cfg_addon, "func": spawner}
        super().__init__()

    def _get_spawn_cfg(self, activate_contact_sensors: bool = False):
        """Keep this background's spawner, which ``Background`` would replace."""

        cfg = super()._get_spawn_cfg(activate_contact_sensors)
        fields = {name: getattr(cfg, name) for name in cfg.__dataclass_fields__ if name != "func"}
        return E2CellLayerCfg(
            **fields,
            func=self.spawn_cfg_addon["func"],
            moving_belts=self.moving_belts,
            outbound_box_scale_xyz=self.outbound_box_scale_xyz,
        )

    def get_nested_physics_prim_paths(self, referenced_prim_paths=None):
        """Arena's nested physics roots, less the conveyor bodies the spawner removes,
        and at the bench station with the bodies its spawner makes."""

        station = {f"{self.prim_path}/{rel}" for rel in (E2_PACKING_BENCH_BODY, E2_OUTBOUND_BOX_BODY)}
        referenced = {p: kind for p, kind in (referenced_prim_paths or {}).items() if p not in station}
        roots = super().get_nested_physics_prim_paths(referenced)
        frozen = {f"{self.prim_path}/{path}" for path in e2_frozen_bodies(self.moving_belts)}
        roots = {path: kind for path, kind in roots.items() if path not in frozen}
        if self.bench_station:
            missing = station - set(referenced_prim_paths or {})
            if missing:
                raise ValueError(f"the bench station's bodies {sorted(missing)} are reset by their object references, "
                                 "which the scene lacks")
            roots = {path: kind for path, kind in roots.items() if path not in station}
        return roots


@register_asset
class DexBenchHotelFBE4260831Background(LibraryBackground):
    """E4 banquet hall with an additional serving cart for dish stacking."""

    name = "dexbench_hotel_fb_e4_260831"
    tags = ["background", "hotel", "fixture-composability"]
    usd_path = runtime_usd(ASSETS_DIR / "env" / "e4_hotel_fb" / "banquet" / "banquet.usda")
    initial_pose = Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
    object_min_z = -0.05

    def __init__(self):
        super().__init__()


def _spawn_lighting_layer_impl(
    prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn a delivered light layer, then apply this build's brightness factor and colour temperature."""

    prim = spawn_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation, **kwargs)
    count = apply_light_edits(prim, cfg.intensity_scale, cfg.color_temperature_k)
    temperature = "as delivered" if cfg.color_temperature_k is None else f"{cfg.color_temperature_k:.0f} K"
    print(
        f"[dexbench] {cfg.label}: {count} delivered lights, intensity x{cfg.intensity_scale:.3f}, "
        f"colour temperature {temperature}"
    )
    return prim


_spawn_lighting_layer = _clone_spawner(_spawn_lighting_layer_impl)


@configclass
class LightingLayerCfg(UsdFileCfg):
    """``UsdFileCfg`` for a delivered light layer, carrying the two values its variations draw."""

    func: Callable = _spawn_lighting_layer
    label: str = "lighting"
    """Name in the spawn log line."""
    intensity_scale: float = 1.0
    """Factor on every light's authored intensity."""
    color_temperature_k: float | None = None
    """Colour temperature set on every light; None keeps the delivered tints with colour temperature off."""


class _LightingLayer(LibraryBackground):
    """A delivery's own lights, as one layer spawned as a variable asset."""

    tags = ["background", "lighting"]
    initial_pose = Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
    object_min_z = -0.05
    label = "lighting"

    @classmethod
    def variation_ranges(cls) -> tuple[tuple[float, float], tuple[float, float]]:
        """(intensity scale range, colour temperature range [K])."""
        raise NotImplementedError

    def __init__(self):
        from isaaclab_arena.variations.light_color_temperature_variation import (
            LightColorTemperatureVariation,
            LightColorTemperatureVariationCfg,
        )
        from isaaclab_arena.variations.light_intensity_variation import (
            LightIntensityVariation,
            LightIntensityVariationCfg,
        )
        from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg

        intensity_scale_range, color_temperature_range_k = self.variation_ranges()
        self.intensity_scale = 1.0
        self.color_temperature_k: float | None = None
        super().__init__(reset_nested_physics=False)
        def uniform(low_high: tuple[float, float]) -> UniformSamplerCfg:
            return UniformSamplerCfg(low=[low_high[0]], high=[low_high[1]])

        self.add_variation(
            LightIntensityVariation(
                self,
                LightIntensityVariationCfg(sampler_cfg=uniform(intensity_scale_range)),
                name="intensity_scale",
            )
        )
        self.add_variation(
            LightColorTemperatureVariation(
                self, LightColorTemperatureVariationCfg(sampler_cfg=uniform(color_temperature_range_k))
            )
        )

    def _get_spawn_cfg(self, activate_contact_sensors: bool = False):
        return LightingLayerCfg(
            usd_path=self.usd_path,
            scale=self.scale,
            activate_contact_sensors=activate_contact_sensors,
            label=self.label,
            intensity_scale=self.intensity_scale,
            color_temperature_k=self.color_temperature_k,
        )

    def set_intensity_scale(self, intensity_scale: float) -> None:
        """Scale every delivered light's intensity by ``intensity_scale``."""
        check_light_edits(intensity_scale=intensity_scale)
        self.intensity_scale = float(intensity_scale)
        self.object_cfg = self._init_object_cfg()

    def set_intensity(self, intensity: float) -> None:
        """Arena's ``LightIntensityVariation`` setter, read as a factor: :meth:`set_intensity_scale`."""
        self.set_intensity_scale(intensity)

    def set_color_temperature(self, color_temperature: float) -> None:
        """Set one colour temperature [K] on every delivered light, within Isaac Lab's [1000, 10000] K."""
        check_light_edits(color_temperature_k=color_temperature)
        self.color_temperature_k = float(color_temperature)
        self.object_cfg = self._init_object_cfg()


def _spawn_assembly_line_e3_edited(
    prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None,
    orientation: tuple[float, float, float, float] | None,
    *,
    m5_station_edits: bool,
    hide_room_shell: bool = False,
    **kwargs,
):
    """Spawn the E3 assembly line, then author DexBench's edits on the live stage."""

    from isaaclab_arena.assets.background import _deinstance_nested_physics

    prim = spawn_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation, **kwargs)
    _deinstance_nested_physics(prim)
    if m5_station_edits:
        apply_m5_station_edits(prim.GetStage(), str(prim.GetPath()))
    if hide_room_shell:
        hidden = _hide_room_shell_walls(prim)
        if not hidden:
            raise RuntimeError(
                f"E3 assembly line has no Structure/RoomShell/Geom/Wall_* under {prim.GetPath()}; the layer changed shape."
            )
        print(
            f"[dexbench] E3 assembly line: hid {', '.join(sorted(hidden))} for the operator "
            "(colliders kept; floor and ceiling still drawn)"
        )
    return prim


@_clone_spawner
def _spawn_assembly_line_e3_m5_station(prim_path, cfg, translation=None, orientation=None, **kwargs):
    return _spawn_assembly_line_e3_edited(prim_path, cfg, translation, orientation, m5_station_edits=True, **kwargs)


@_clone_spawner
def _spawn_assembly_line_e3_m5_station_hiding_room_shell(prim_path, cfg, translation=None, orientation=None, **kwargs):
    return _spawn_assembly_line_e3_edited(
        prim_path, cfg, translation, orientation, m5_station_edits=True, hide_room_shell=True, **kwargs
    )


_ASSEMBLY_LINE_E3_SPAWNERS = {
    (True, False): _spawn_assembly_line_e3_m5_station,
    (True, True): _spawn_assembly_line_e3_m5_station_hiding_room_shell,
}


@register_asset
class DexBenchAssemblyLineE3Background(LibraryBackground):
    """DexBench E3 manufacturing assembly line -- fixture slots, structure, anchors only."""

    name = "dexbench_assembly_line_e3"
    tags = ["background"]
    usd_path = runtime_usd(ASSETS_DIR / "env" / "e3_manufacturing" / "assembly_line" / "assembly_line.usda")
    initial_pose = Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
    object_min_z = -0.05

    def __init__(self, m5_station_edits: bool = False, hide_room_shell: bool = False):
        """
        Args:
            m5_station_edits: Apply ``apply_m5_station_edits`` at spawn (checks
                r01's M5 layout, conveyor legs under the frames, bin box colliders).
            hide_room_shell: Stop drawing the room's four walls, for a session
                driven from inside the cell (teleop): the 5 x 4 m room puts a
                wall between any viewpoint outside it and the M5 corner, and
                the Kit viewport starts outside. Rendering only: the walls keep
                colliding, and the floor and ceiling stay drawn. Wired with
                ``m5_station_edits`` only, the E2 cell's ``hide_room_shell``
                for this scene.
        """

        if hide_room_shell and not m5_station_edits:
            raise ValueError("hide_room_shell is wired for the M5 station only; pass m5_station_edits=True with it")
        spawner = _ASSEMBLY_LINE_E3_SPAWNERS.get((m5_station_edits, hide_room_shell))
        if spawner is not None:
            self.spawn_cfg_addon = {**self.spawn_cfg_addon, "func": spawner}
        super().__init__()

    def _get_spawn_cfg(self, activate_contact_sensors: bool = False):
        """Keep the editing spawner, which ``Background`` would replace."""

        cfg = super()._get_spawn_cfg(activate_contact_sensors)
        func = self.spawn_cfg_addon.get("func")
        if func in _ASSEMBLY_LINE_E3_SPAWNERS.values():
            cfg.func = func
        return cfg


@register_asset
class DexBenchHotelFBE4Lighting(_LightingLayer):
    """The E4 banquet hall's own 40 lights, from PSL's r01 delivery, as a variable asset."""

    name = E4_LIGHTING_ASSET
    tags = ["background", "hotel", "lighting"]
    usd_path = str(E4_LIGHTING_LAYER)
    label = "E4 lighting"

    @classmethod
    def variation_ranges(cls):
        from dexbench_arena.layouts.db_h1_layout import (
            LIGHT_COLOR_TEMPERATURE_RANGE_K,
            LIGHT_INTENSITY_SCALE_RANGE,
        )

        return LIGHT_INTENSITY_SCALE_RANGE, LIGHT_COLOR_TEMPERATURE_RANGE_K


@register_asset
class DexBenchRetailV1Lighting(_LightingLayer):
    """The retail_v1 look's 19 lights and dome, in FC_260821 coordinates, as a variable asset."""

    name = RETAIL_V1_LIGHTING_ASSET
    tags = ["background", "retail", "lighting"]
    usd_path = runtime_usd(RETAIL_V1_DIR / "lighting.usda")
    label = "retail_v1 lighting"

    @classmethod
    def variation_ranges(cls):
        return RETAIL_V1_LIGHT_INTENSITY_SCALE_RANGE, RETAIL_V1_LIGHT_COLOR_TEMPERATURE_RANGE_K
