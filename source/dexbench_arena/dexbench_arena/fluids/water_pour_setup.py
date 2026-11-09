# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Newton implicit-MPM water sidecar for the hotel pour scene."""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import newton
import numpy as np
import torch
import warp as wp
from isaaclab.utils.math import matrix_from_quat, quat_slerp
from newton import Axis, ModelBuilder
from newton.solvers import SolverImplicitMPM
from pxr import Gf, Usd, UsdGeom, Vt

from dexbench_arena import ASSETS_DIR

from .goblet_collider import GobletFluidCollider
from .lathe_cavity import LatheCavity
from .mpm_sidecar import ImplicitMPMSidecarCfg, StandaloneMPMHost
from .pour_jugs import GOBLET_PROFILE
from .volume_seed import add_seed_particles, seed_profile
from .water_surface import WaterSurfaceRenderer, cavity_pose

WATER_DENSITY = 1000.0

SERVING_COUNTER_TOP_Z = 0.90
"""Catch-surface top [m] when the cfg names none: the hotel counter. The E4 serving trolley is 0.80
(``WaterPourSidecarCfg.counter_top_z``; the teleop water takes it from DEXBENCH_WATER_COUNTER_Z)."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


GOBLET_PROFILE_NPZ = ASSETS_DIR / GOBLET_PROFILE
"""``pour_goblet``'s bowl cavity, the profile its package ships beside the USD (``pour_jugs.GOBLET_PROFILE``). The
teleop water's goblet collider when ``DEXBENCH_WATER_CUP_ASSET`` names the pour goblet; ``DEXBENCH_WATER_CUP_PROFILE``
overrides it."""


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class WaterPourSidecarCfg:
    """Layout + solver knobs for the hotel water sidecar [m]."""

    vessel_asset: str = "serving_jug"
    cup_asset: str = "serving_cup"

    profile_npz: str | None = None
    """Interior lathe profile (z, r_inner, floor_z, rim_z). Defaults to the
    measured IDEALISK profile in the pinned assets submodule."""

    axis_xy_local: tuple[float, float] = (-0.022, 0.0)
    """Cavity axis offset in the vessel's LOCAL frame. The IDEALISK cavity sits
    ~2.2 cm off the asset origin in -X; the shipped profile npz carries no
    axis, so the offset is declared here and rotates with the vessel."""

    render_radius_vox: float = 0.45
    """Radius of the drawn particle spheres in voxels (render only). The contact band holds
    water 4 to 9 mm inside a wall; a full voxel (8 mm) draws it through the tall jug's 0.6 mm
    guard plate, which reads as water leaking out round the pour hole. 0.45 (3.6 mm) stays
    inside, with neighbouring spheres (about 7.5 mm apart) still touching."""

    fill: float = 0.6
    target_ml: float | None = None
    """If set, override ``fill`` so the seeded water volume is ~this many ml
    (컬럼 부피 = ml — 시드 박스 단면 × 높이로 환산)."""
    voxel_size: float = 0.008
    """8 mm. 바닥 캡 두께 0.016이 정확히 2복셀이어야 하며,
    더 가는 복셀에서는 캡이 SDF 밴드 밖 데드 인테리어가 되어 물이 통과한다."""
    substeps: int | None = None
    counter_friction: float = 0.4
    counter_top_z: float | None = None
    """Top of the catch shell under the goblet [m, world]: the surface the goblet stands on. None takes
    ``SERVING_COUNTER_TOP_Z`` (0.90, the hotel counter; the E4 serving trolley is 0.80). The shell is rebuilt
    at every re-seed, so a caller sets this field rather than the module global."""
    cup_profile_npz: str | None = None
    """Optional measured bowl profile for an open, watertight MPM shell."""
    cup_margin: float = 0.006
    cup_wall_height: float = 0.12
    cup_inner_radius: float = 0.05
    perf_every: int = 0

    stride: int = 1
    """Advance the fluid every N env steps (dt scales by N). 2 halves the
    average cost with no visible change at teleop speeds."""

    particles_per_cell: float = 1.2
    """MPM seed density. Realtime teleop needs fewer particles than a dense
    offline stream (cost is linear in count)."""

    max_iterations: int = 30
    """Implicit solver iterations."""


    grid_type: str = "fixed"
    """MPM grid: fixed (deterministic allocation), sparse or dense (DEXBENCH_WATER_GRID_TYPE)."""
    grid_padding: int = 64
    """Fixed-grid radius around the seeded water [voxels]. A smaller grid lets water splashed out of the pour reach
    the edge, where it is culled; a larger one costs more every step (DEXBENCH_WATER_GRID_PAD)."""
    max_active_cells_log2: int = 13
    """The fixed grid's active-cell budget, 2**n: it costs every step whatever the particle count
    (DEXBENCH_WATER_CELLS_POW)."""
    collider_basis: str = "S2"
    """Collider basis (DEXBENCH_WATER_COLLIDER_BASIS). Volume holds only with dt <= 1/180 s, 2 particles per cell
    axis, P1d and S2 together."""
    strain_basis: str = "P1d"
    """Strain basis (DEXBENCH_WATER_STRAIN_BASIS); see ``collider_basis``."""
    max_dt: float = 1.0 / 180.0
    """Largest per-substep dt [s] (CFL: at 8 mm voxels 1/180 s holds to about 1.44 m/s); 0 falls back to 4 substeps
    (DEXBENCH_WATER_MAX_DT). ``substeps`` set wins over it."""
    cull: bool = True
    """Park particles that leave the fixed grid or go non-finite (one such particle turns the whole solve to NaN on
    Newton main 646cece4); they count as culled (DEXBENCH_WATER_CULL=0 turns it off)."""
    cull_margin_vox: float = 4.0
    """How far inside the grid's edge a particle is culled [voxels] (DEXBENCH_WATER_CULL_MARGIN_VOX)."""
    vessel_margin_vox: float | None = None
    """The vessel collider's contact margin [voxels]; None takes the baked shell's, else 1
    (DEXBENCH_WATER_VESSEL_MARGIN_VOX)."""
    cup_box_proxy: bool = False
    """Catch in a box proxy instead of the cup's mesh when no cup profile is given (DEXBENCH_WATER_CUP=box)."""


class ProfileVesselCollider:
    """Watertight vessel lathe driven by the full measured pose."""

    def __init__(self, profile_npz: str, axis_xy_local: tuple[float, float],
                 n_segments: int = 24, n_rings: int = 12, thickness: float = 0.016):
        profile = np.load(profile_npz)
        z_all = np.asarray(profile["z"], dtype=np.float64)
        r_all = np.asarray(profile["r_inner"], dtype=np.float64)
        floor_z = float(profile["floor_z"])
        keep = z_all >= floor_z
        z_kept = np.concatenate(([floor_z], z_all[keep]))
        r_kept = np.concatenate(([np.interp(floor_z, z_all, r_all)], r_all[keep]))
        idx = np.linspace(0, len(z_kept) - 1, n_rings).astype(int)
        z_prof, r_in = z_kept[idx], r_kept[idx]
        r_out = r_in + thickness

        ax, ay = float(axis_xy_local[0]), float(axis_xy_local[1])
        prof_rz = [(float(r), float(z)) for r, z in zip(r_in, z_prof)] + [
            (float(r), float(z)) for r, z in zip(r_out[::-1], z_prof[::-1])
        ]
        theta = np.linspace(0.0, 2.0 * np.pi, n_segments, endpoint=False)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        verts = [(r * cos_t[s] + ax, r * sin_t[s] + ay, z) for r, z in prof_rz for s in range(n_segments)]
        tris: list[int] = []
        n_prof = len(prof_rz)
        for k in range(n_prof - 1):
            for s in range(n_segments):
                a = k * n_segments + s
                b = k * n_segments + (s + 1) % n_segments
                c = (k + 1) * n_segments + s
                d = (k + 1) * n_segments + (s + 1) % n_segments
                tris.extend([a, b, c, b, d, c])
        center_in = len(verts)
        verts.append((ax, ay, float(z_prof[0])))
        center_out = len(verts)
        verts.append((ax, ay, float(z_prof[0]) - 0.016))
        outer_ring0 = (n_prof - 1) * n_segments
        for s in range(n_segments):
            a, b = s, (s + 1) % n_segments
            tris.extend([center_in, b, a])
            a2, b2 = outer_ring0 + s, outer_ring0 + (s + 1) % n_segments
            tris.extend([center_out, a2, b2])

        self._local = np.asarray(verts, dtype=np.float64)
        indices = np.asarray(tris, dtype=np.int32).reshape(-1, 3)[:, ::-1].reshape(-1).copy()
        world0 = self._local.astype(np.float32)
        self._prev_world = world0
        points = wp.array(world0, dtype=wp.vec3)
        self.mesh = wp.Mesh(points, wp.array(indices, dtype=wp.int32), wp.zeros_like(points))
        self.rim_z = float(profile["rim_z"])
        self.floor_z = floor_z
        self.r_inner_min = float(r_all[keep].min()) if keep.any() else float(r_all.min())
        print(f"[water] vessel lathe: {len(self._local)} verts, {len(indices) // 3} tris", flush=True)

    def update(self, pos: np.ndarray, quat_xyzw: np.ndarray, dt: float | None = None) -> None:
        R = _rotation(quat_xyzw)
        world = (self._local @ R.T + np.asarray(pos, dtype=np.float64)).astype(np.float32)
        self.mesh.points.assign(wp.array(world, dtype=wp.vec3))
        if dt is not None:
            self.mesh.velocities.assign(wp.array((world - self._prev_world) / float(dt), dtype=wp.vec3))
        self._prev_world = world
        self.mesh.refit()


class BakedVesselCollider(ProfileVesselCollider):
    """The vessel's own thin shell, baked offline by a maintainers' tool."""

    def __init__(self, shell_npz: str, profile_npz: str | None):
        with np.load(shell_npz) as shell:
            self._local = np.asarray(shell["verts"], dtype=np.float64)
            tris = np.asarray(shell["tris"], dtype=np.int32)
            self.margin_vox = float(shell["margin_vox"])
        world0 = self._local.astype(np.float32)
        self._prev_world = world0
        points = wp.array(world0, dtype=wp.vec3)
        self.mesh = wp.Mesh(points, wp.array(tris.reshape(-1), dtype=wp.int32), wp.zeros_like(points))
        self.rim_z = self.floor_z = self.r_inner_min = float("nan")
        if profile_npz:
            with np.load(profile_npz) as profile:
                z_all = np.asarray(profile["z"], dtype=np.float64)
                r_all = np.asarray(profile["r_inner"], dtype=np.float64)
                self.floor_z, self.rim_z = float(profile["floor_z"]), float(profile["rim_z"])
                keep = z_all >= self.floor_z
                self.r_inner_min = float(r_all[keep].min()) if keep.any() else float(r_all.min())
        print(f"[water] baked vessel collider {Path(shell_npz).name}: {len(self._local)} verts, {len(tris)} tris, "
              f"margin {self.margin_vox:g} vox", flush=True)


def _lerp(a, b, t: float) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return a + (np.asarray(b, dtype=np.float64) - a) * float(t)


def _slerp(a, b, t: float) -> np.ndarray:
    """The xyzw quaternion ``t`` of the way from ``a`` to ``b`` (copies: quat_slerp flips ``b`` in place)."""
    return quat_slerp(torch.tensor(a, dtype=torch.float64), torch.tensor(b, dtype=torch.float64), float(t)).numpy()


def _rotation(quat_xyzw) -> np.ndarray:
    """The rotation matrix of an xyzw quaternion."""
    return matrix_from_quat(torch.as_tensor(quat_xyzw, dtype=torch.float64)).numpy()


def _wp_box_mesh(center, half_extents):
    box = newton.Mesh.create_box(
        float(half_extents[0]), float(half_extents[1]), float(half_extents[2]),
        duplicate_vertices=False, compute_normals=False, compute_uvs=False, compute_inertia=False)
    vertices = np.asarray(box.vertices, dtype=np.float32)[:, :3] + np.asarray(center, dtype=np.float32)
    indices = np.asarray(box.indices, dtype=np.int32).reshape(-1)
    return wp.Mesh(points=wp.array(vertices, dtype=wp.vec3),
                   indices=wp.array(indices, dtype=wp.int32),
                   velocities=wp.zeros(len(vertices), dtype=wp.vec3))


def _mug_collider_mesh(stage, measured_root_pos: np.ndarray | None, cup_prim_name: str = "serving_cup"):
    """Static wp.Mesh from the visible serving_cup, shifted to its measured pose."""
    roots = [p for p in stage.Traverse() if p.GetName() == cup_prim_name and p.IsActive()]
    roots.sort(key=lambda p: "/envs/" not in str(p.GetPath()))
    verts_list, tris_list, base = [], [], 0
    root_stage_pos = None
    for root in roots[:1]:
        xf_root = np.array(UsdGeom.Xformable(root).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        root_stage_pos = xf_root[3, :3].astype(np.float64)
        for prim in Usd.PrimRange(root):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            if "colproxy" in str(prim.GetPath()).lower():
                continue
            mesh = UsdGeom.Mesh(prim)
            pts = mesh.GetPointsAttr().Get()
            counts = mesh.GetFaceVertexCountsAttr().Get()
            idx = mesh.GetFaceVertexIndicesAttr().Get()
            if not pts or not counts or not idx:
                continue
            xf = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
            world = np.asarray(pts, dtype=np.float64) @ xf[:3, :3] + xf[3, :3]
            idx = np.asarray(idx, dtype=np.int64)
            tris = []
            off = 0
            for c in counts:
                for k in range(1, int(c) - 1):
                    tris.extend((idx[off], idx[off + k], idx[off + k + 1]))
                off += int(c)
            verts_list.append(world)
            tris_list.append(np.asarray(tris, dtype=np.int32) + base)
            base += len(world)
    if not verts_list:
        return None
    points = np.vstack(verts_list)
    if measured_root_pos is not None and root_stage_pos is not None:
        points = points + (np.asarray(measured_root_pos, dtype=np.float64) - root_stage_pos)
    points = points.astype(np.float32)
    indices = np.concatenate(tris_list)
    return wp.Mesh(points=wp.array(points, dtype=wp.vec3),
                   indices=wp.array(indices, dtype=wp.int32),
                   velocities=wp.zeros(len(points), dtype=wp.vec3))


def _build_mpm_cfg(cfg: WaterPourSidecarCfg, box_min, box_max):
    return ImplicitMPMSidecarCfg(
        voxel_size=cfg.voxel_size,
        grid_type=cfg.grid_type,
        grid_padding=int(cfg.grid_padding),
        max_active_cell_count=1 << int(cfg.max_active_cells_log2),
        max_iterations=cfg.max_iterations,
        viscosity=0.001,
        tensile_yield_ratio=0.0,
        friction=0.0,
        particles_per_cell=cfg.particles_per_cell,
        collider_basis=cfg.collider_basis,
        strain_basis=cfg.strain_basis,
        particle_box_min=tuple(box_min),
        particle_box_max=tuple(box_max),
    )


def _init_standalone_mpm(mpm_cfg, device: str, seed=None, positions=None, cfg: WaterPourSidecarCfg | None = None):
    builder = ModelBuilder()
    builder.up_axis = Axis.from_string("Z")
    SolverImplicitMPM.register_custom_attributes(builder)
    if seed is None:
        StandaloneMPMHost._emit_mpm_particle_box(builder, mpm_cfg)
    else:
        add_seed_particles(builder, seed, positions)
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -9.81))
    StandaloneMPMHost._set_mpm_material_parameters(model, mpm_cfg)

    options = SolverImplicitMPM.Config()
    for name in ("max_iterations", "tolerance", "solver", "voxel_size", "grid_type",
                 "grid_padding", "max_active_cell_count", "transfer_scheme",
                 "integration_scheme", "strain_basis", "collider_basis", "velocity_basis"):
        if not hasattr(options, name):
            raise RuntimeError(f"[water] newton's SolverImplicitMPM.Config has no {name}; the water is written "
                               "against newton 1.8's")
        setattr(options, name, getattr(mpm_cfg, name))
    print(f"[water] solver options: max_iterations={getattr(options, 'max_iterations', None)} "
          f"tolerance={getattr(options, 'tolerance', None)} solver={getattr(options, 'solver', None)} "
          f"strain_basis={getattr(options, 'strain_basis', None)} "
          f"collider_basis={getattr(options, 'collider_basis', None)} "
          f"velocity_basis={getattr(options, 'velocity_basis', None)}", flush=True)

    class _StandaloneMPMHost(StandaloneMPMHost):
        """This sidecar's fluid, on a class of its own so no other sidecar shares its state. It is
        stepped by the sidecar itself (``step``), never inside the robot's physics substeps."""

    _StandaloneMPMHost._mpm_model = model
    _StandaloneMPMHost._mpm_solver = SolverImplicitMPM(model, options)
    _StandaloneMPMHost._mpm_state_0 = model.state()
    _StandaloneMPMHost._mpm_enabled = True
    return _StandaloneMPMHost


class _PointsRenderer:
    """Author fluid particles as a UsdGeom.Points cloud (teleop-visible)."""

    def __init__(self, stage, radius: float):
        self._gf = Gf
        self.points = UsdGeom.Points.Define(stage, "/World/dexbench_water_particles")
        self.points.CreateDisplayColorAttr().Set([Gf.Vec3f(0.10, 0.40, 0.90)])
        self.points.CreateDisplayOpacityAttr().Set([0.7])
        self._radius = radius

    def update(self, positions: np.ndarray) -> None:
        self.points.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in positions]))
        self.points.GetWidthsAttr().Set(Vt.FloatArray([self._radius * 2.0] * len(positions)))


class WaterPourSidecar:
    def __init__(self, env, cfg: WaterPourSidecarCfg):
        self._env = getattr(env, "unwrapped", env)
        self.cfg = cfg
        self._vessel = self._scene_object(cfg.vessel_asset)
        self._cup = self._scene_object(cfg.cup_asset)
        self._perf = {"n": 0, "s": 0.0}
        self._renderer = None
        self._render_cavities_cache = None

        v_p, _ = self._measure(self._vessel)
        self._origin = np.asarray(v_p, dtype=np.float64)
        self._cup_xy = tuple(self._measure(self._cup)[0][:2]) if self._cup is not None \
            else (self._origin[0] + 0.13, self._origin[1])

        print(f"[water] newton {getattr(newton, '__version__', '?')} at {newton.__file__}", flush=True)

        self._build()
        _decim = 1
        try:
            _decim = max(1, int(round(float(getattr(self._env, "step_dt", 0.0)) / float(self._env.cfg.sim.dt))))
        except Exception:  # noqa: BLE001
            pass
        _max_dt = float(cfg.max_dt)
        if cfg.substeps:
            self._substeps = int(cfg.substeps)
        elif _max_dt > 0.0:
            try:
                _span = float(self._env.step_dt) * max(1, int(cfg.stride))
            except Exception:  # noqa: BLE001
                _span = (1.0 / 30.0) * max(1, int(cfg.stride))
            self._substeps = max(_decim, 4, int(math.ceil(_span / _max_dt * (1.0 - 1e-3))))
            print(f"[water] CFL target dt <= {_max_dt*1000:.1f} ms -> substeps {self._substeps}", flush=True)
        else:
            self._substeps = max(_decim, 4)
        self._prev_vessel_pose = None
        self._prev_cup_pose = None
        self._cull_nonfinite = bool(cfg.cull)
        self._cull_kernel = None
        self._reset_cull()
        if not self._cull_nonfinite:
            print("[water] out-of-grid / non-finite particle culling OFF", flush=True)
        if self._substeps > 1:
            print(f"[water] fluid substeps = {self._substeps} (per-substep dt = env step_dt / {self._substeps})", flush=True)
        _mass_kg = float(self._nm._mpm_model.particle_mass.numpy().astype(np.float64).sum())
        print(f"[water] seeded {_mass_kg / WATER_DENSITY * 1e6:.1f} ml ({self.particle_count} particles)", flush=True)
        print(f"[water] sidecar ready: {self.particle_count} particles, vessel at "
              f"{np.round(self._origin, 3)}, cup_xy={tuple(round(v, 3) for v in self._cup_xy)}, "
              f"substeps={self._substeps}", flush=True)

    def _scene_object(self, name: str):
        try:
            return self._env.scene[name]
        except KeyError:
            print(f"[water] scene has no rigid object '{name}'", flush=True)
            return None

    @staticmethod
    def _measure(obj):
        if obj is None:
            return np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0])
        pose = obj.data.root_pose_w[0].detach().cpu().numpy().astype(np.float64)
        return pose[:3], pose[3:7]

    _IDEALISK_PROFILE = ("assets", "object_library", "steel_jug_1l", "usd", "idealisk_profile.npz")

    def _profile_path(self) -> str:
        """cfg.profile_npz, else the ``*_profile.npz`` beside the vessel's spawn USD, else IDEALISK's."""
        if self.cfg.profile_npz:
            return self.cfg.profile_npz
        usd = getattr(getattr(getattr(self._vessel, "cfg", None), "spawn", None), "usd_path", None)
        if usd:
            found = sorted(Path(usd).parent.glob("*_profile.npz"))
            if found:
                return str(found[0])
        return str(_repo_root().joinpath(*self._IDEALISK_PROFILE))

    def _axis_xy(self, profile_npz: str) -> tuple[float, float]:
        """Cavity axis in the vessel frame: the profile's axis_xy, else cfg.axis_xy_local."""
        with np.load(profile_npz) as profile:
            if "axis_xy" in profile.files:
                axis = np.asarray(profile["axis_xy"], dtype=np.float64).reshape(2)
                return (float(axis[0]), float(axis[1]))
        return (float(self.cfg.axis_xy_local[0]), float(self.cfg.axis_xy_local[1]))

    def _baked_path(self, suffix: str) -> str | None:
        """The ``*_water_<suffix>.npz`` beside the vessel's spawn USD, if the package ships one."""
        usd = getattr(getattr(getattr(self._vessel, "cfg", None), "spawn", None), "usd_path", None)
        if usd:
            found = sorted(Path(usd).parent.glob(f"*_water_{suffix}.npz"))
            if found:
                return str(found[0])
        return None

    def _build(self) -> None:
        cfg = self.cfg
        profile_npz = self._profile_path()
        self.axis_xy_local = self._axis_xy(profile_npz)
        print(f"[water] vessel profile {profile_npz} axis_xy={self.axis_xy_local}", flush=True)
        shell_npz = self._baked_path("shell")
        if shell_npz:
            self._collider = BakedVesselCollider(shell_npz, profile_npz)
        else:
            self._collider = ProfileVesselCollider(profile_npz, self.axis_xy_local)

        v_p, v_q = self._measure(self._vessel)
        R = _rotation(v_q)
        axis_w = R @ np.array([self.axis_xy_local[0], self.axis_xy_local[1], 0.0])
        cx, cy = float(v_p[0] + axis_w[0]), float(v_p[1] + axis_w[1])
        half = max(0.004, self._collider.r_inner_min - 2.0 * cfg.voxel_size)
        z0 = float(v_p[2]) + self._collider.floor_z + 4.0 * cfg.voxel_size
        positions = seed = None
        if cfg.target_ml is not None:
            _s = float(cfg.voxel_size) / float(cfg.particles_per_cell)
            seed = seed_profile(self._profile_path(), volume_ml=float(cfg.target_ml), spacing_m=_s,
                                density_kg_m3=WATER_DENSITY,
                                wall_clearance_m=2.0 * cfg.voxel_size, floor_clearance_m=4.0 * cfg.voxel_size)
            positions = seed.positions @ R.T + np.asarray(v_p, dtype=np.float64) + axis_w
            z1 = float(positions[:, 2].max()) + _s
            half = max(half, float(np.max(np.hypot(positions[:, 0] - cx, positions[:, 1] - cy))) + _s)
            print(f"[water] target {cfg.target_ml:.0f} ml -> {seed.count} particles of "
                  f"{seed.particle_volume_m3 * 1e6:.4f} ml, surface {seed.surface_z_m * 100:.1f} cm above the jug base "
                  f"(cavity {seed.capacity_ml:.0f} ml)", flush=True)
        else:
            z1 = z0 + max(cfg.voxel_size,
                          (self._collider.rim_z - self._collider.floor_z - 0.03)
                          * max(0.05, min(1.0, cfg.fill)))
        mpm_cfg = _build_mpm_cfg(cfg, (cx - half, cy - half, z0), (cx + half, cy + half, z1))
        self._mpm_cfg = mpm_cfg
        self._nm = _init_standalone_mpm(mpm_cfg, str(self._env.device), seed=seed, positions=positions, cfg=cfg)
        self._collider.update(v_p, v_q, dt=None)

        meshes = [self._collider.mesh]
        margin_vox = float(getattr(self._collider, "margin_vox", 1.0) if cfg.vessel_margin_vox is None
                           else cfg.vessel_margin_vox)
        margins = [float(cfg.voxel_size) * margin_vox]
        frictions = [0.0]
        self._lid_collider = None
        lid_npz = self._baked_path("lid")
        if lid_npz:
            self._lid_collider = BakedVesselCollider(lid_npz, None)
            self._lid_collider.update(v_p, v_q, dt=None)
            meshes.append(self._lid_collider.mesh)
            margins.append(float(cfg.voxel_size) * self._lid_collider.margin_vox)
            frictions.append(0.0)
        cup_mesh = None
        self._cup_collider = None
        if cfg.cup_profile_npz is not None:
            if self._cup is None:
                raise ValueError("A measured cup pose is required for a profile collider")
            cup_p, cup_q = self._measure(self._cup)
            self._cup_collider = GobletFluidCollider(
                cfg.cup_profile_npz, cup_p, _rotation(cup_q), cfg.voxel_size)
            cup_mesh = self._cup_collider.mesh
        elif cfg.cup_box_proxy:
            pass
        elif self._cup is not None:
            try:
                cup_mesh = _mug_collider_mesh(
                    self._env.scene.stage, self._measure(self._cup)[0], cfg.cup_asset)
            except Exception as exc:  # noqa: BLE001
                print(f"[water] mug collider unavailable ({exc}); box proxy instead", flush=True)
        ccx, ccy = self._cup_xy
        cz = self.counter_top_z
        if cup_mesh is not None:
            meshes.append(cup_mesh)
            margins.append(cfg.cup_margin)
            frictions.append(0.0)
        else:
            h = cfg.cup_wall_height
            for m in (_wp_box_mesh((ccx, ccy, cz + 0.005), (0.055, 0.055, 0.005)),
                      _wp_box_mesh((ccx + 0.055, ccy, cz + h / 2), (0.005, 0.055, h / 2)),
                      _wp_box_mesh((ccx - 0.055, ccy, cz + h / 2), (0.005, 0.055, h / 2)),
                      _wp_box_mesh((ccx, ccy + 0.055, cz + h / 2), (0.055, 0.005, h / 2)),
                      _wp_box_mesh((ccx, ccy - 0.055, cz + h / 2), (0.055, 0.005, h / 2))):
                meshes.append(m)
                margins.append(0.0)
                frictions.append(0.0)
        meshes.append(_wp_box_mesh((ccx, ccy, cz - 0.008), (0.45, 0.55, 0.008)))
        margins.append(0.0)
        frictions.append(float(cfg.counter_friction))
        self._nm.setup_mpm_colliders(meshes, collider_margins=margins, collider_friction=frictions)
        print(f"[water] registered {len(meshes)} MPM colliders", flush=True)

    def step(self, dt: float | None = None) -> None:
        self._tick = getattr(self, "_tick", 0) + 1
        if self.cfg.stride > 1 and (self._tick % self.cfg.stride):
            return
        eff = self.cfg.stride
        t0 = time.perf_counter()
        sim_dt = (float(self._env.cfg.sim.dt) if dt is None else float(dt) / self._substeps) * eff
        v_p, v_q = self._measure(self._vessel)
        c_p = c_q = None
        if self._cup_collider is not None and self._cup is not None:
            c_p, c_q = self._measure(self._cup)
        t1 = time.perf_counter()

        prev_p, prev_q = self._prev_vessel_pose or (v_p, v_q)
        n_sub = self._substeps
        sub_dt = sim_dt

        prev_cup = self._prev_cup_pose or (c_p, c_q)
        for k in range(1, n_sub + 1):
            a = k / n_sub
            self._collider.update(_lerp(prev_p, v_p, a), _slerp(prev_q, v_q, a), dt=sub_dt)
            if self._lid_collider is not None:
                self._lid_collider.update(_lerp(prev_p, v_p, a), _slerp(prev_q, v_q, a), dt=sub_dt)
            if self._cup_collider is not None and c_p is not None:
                self._cup_collider.update(
                    _lerp(prev_cup[0], c_p, a), _rotation(_slerp(prev_cup[1], c_q, a)), dt=sub_dt)
            if k == 1:
                t2 = time.perf_counter()
            self._nm._step_mpm_sidecar(None, sub_dt)
            if self._cull_nonfinite:
                self._cull_nonfinite_particles()
        self._prev_vessel_pose = (np.asarray(v_p).copy(), np.asarray(v_q).copy())
        self._prev_cup_pose = (np.asarray(c_p).copy(), np.asarray(c_q).copy()) if c_p is not None else None
        if self.cfg.perf_every:
            wp.synchronize()
            t3 = time.perf_counter()
            self._perf.setdefault("meas", 0.0)
            self._perf.setdefault("coll", 0.0)
            self._perf.setdefault("solve", 0.0)
            self._perf["meas"] += t1 - t0
            self._perf["coll"] += t2 - t1
            self._perf["solve"] += t3 - t2
        if self._renderer is not None:
            pos = self.positions()
            if pos is not None:
                self._renderer.update(pos)
        if self.cfg.perf_every:
            self._perf["n"] += 1
            self._perf["s"] += time.perf_counter() - t0
            if self._perf["n"] >= self.cfg.perf_every:
                n = self._perf["n"]
                print(f"[water] mpm {self._perf['s'] / n * 1000:.1f} ms/step "
                      f"(measure {self._perf.get('meas', 0) / n * 1000:.1f} | collider "
                      f"{self._perf.get('coll', 0) / n * 1000:.1f} | solve "
                      f"{self._perf.get('solve', 0) / n * 1000:.1f})", flush=True)
                for k in ("n", "s", "meas", "coll", "solve"):
                    self._perf[k] = 0 if k == "n" else 0.0

    def enable_render(self, mode: str = "surface") -> None:
        """Draw the water: ``surface`` (``water_surface``, held inside the jug and the goblet) or ``points``."""
        try:
            if mode == "points":
                self._renderer = _PointsRenderer(self._env.scene.stage,
                                                 float(self.cfg.render_radius_vox) * float(self.cfg.voxel_size))
                print("[water] particle renderer attached", flush=True)
            else:
                self._renderer = WaterSurfaceRenderer(self._env.scene.stage, self.cfg.voxel_size, self._render_cavities)
                print("[water] surface renderer attached", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[water] renderer unavailable: {exc}", flush=True)

    def _render_cavities(self) -> list:
        """The jug's and the target goblet's cavities at their current poses, for the surface renderer."""
        if self._render_cavities_cache is None:
            profile = self._profile_path()
            cavities = [(self._vessel, LatheCavity.from_npz(profile), self._axis_xy(profile))]
            if self._cup is not None and self.cfg.cup_profile_npz:
                cavities.append((self._cup, LatheCavity.from_npz(self.cfg.cup_profile_npz), (0.0, 0.0)))
            self._render_cavities_cache = cavities
        out = []
        for body, cavity, axis_xy in self._render_cavities_cache:
            if body is None:
                continue
            centre, rotation = cavity_pose(*self._measure(body), axis_xy)
            out.append((centre, rotation, cavity))
        return out

    @property
    def counter_top_z(self) -> float:
        """Top of the catch shell under the goblet [m]: the cfg's value, else the module default."""
        return float(SERVING_COUNTER_TOP_Z if self.cfg.counter_top_z is None else self.cfg.counter_top_z)

    def reset(self) -> None:
        v_p, _ = self._measure(self._vessel)
        self._origin = np.asarray(v_p, dtype=np.float64)
        if self._cup is not None:
            self._cup_xy = tuple(self._measure(self._cup)[0][:2])
        self._build()
        self._reset_cull()
        self._prev_vessel_pose = None
        self._prev_cup_pose = None
        print("[water] sidecar re-seeded", flush=True)

    @property
    def particle_count(self) -> int:
        model = getattr(self._nm, "_mpm_model", None)
        return int(getattr(model, "particle_count", 0)) if model is not None else 0

    def positions(self) -> np.ndarray | None:
        state = getattr(self._nm, "_mpm_state_0", None)
        if state is None:
            return None
        return state.particle_q.numpy()

    _CULL_PARK_Z = -50.0

    def _reset_cull(self) -> None:
        """Forget the cull bounds and counts: the next cull measures the bounds around the water as it stands then (the
        grid a re-seed lays out). The kernel holds no episode's data and is kept."""
        self._cull_bounds = None
        self._cull_count = None
        self._culled = 0

    def _cull_nonfinite_particles(self) -> None:
        model = self._nm._mpm_model
        if self._cull_bounds is None:
            q0 = wp.to_torch(self._nm.get_mpm_state_0().particle_q).cpu().numpy()
            pad = float(self.cfg.voxel_size) * float(getattr(self._mpm_cfg, "grid_padding", 48))
            inner = float(self.cfg.voxel_size) * float(self.cfg.cull_margin_vox)
            lo = q0.min(axis=0) - pad + inner
            hi = q0.max(axis=0) + pad - inner
            self._cull_bounds = (lo, hi)
            self._cull_count = wp.zeros(1, dtype=wp.int32, device=model.particle_flags.device)
            print(f"[water] cull bounds x[{lo[0]:.2f},{hi[0]:.2f}] y[{lo[1]:.2f},{hi[1]:.2f}] z[{lo[2]:.2f},{hi[2]:.2f}]", flush=True)
        if self._cull_kernel is None:
            active = int(newton.ParticleFlags.ACTIVE)
            park_z = float(self._CULL_PARK_Z)

            @wp.kernel
            def _cull(q: wp.array(dtype=wp.vec3), qd: wp.array(dtype=wp.vec3),
                      flags: wp.array(dtype=wp.int32), lo: wp.vec3, hi: wp.vec3,
                      count: wp.array(dtype=wp.int32)):
                i = wp.tid()
                p = q[i]
                v = qd[i]
                ok = (wp.isfinite(p[0]) and wp.isfinite(p[1]) and wp.isfinite(p[2]) and
                      wp.isfinite(v[0]) and wp.isfinite(v[1]) and wp.isfinite(v[2]))
                if ok and (flags[i] & active):
                    ok = (p[0] > lo[0] and p[0] < hi[0] and p[1] > lo[1] and p[1] < hi[1]
                          and p[2] > lo[2] and p[2] < hi[2])
                if not ok:
                    if flags[i] & active:
                        wp.atomic_add(count, 0, 1)
                    flags[i] = flags[i] & ~active
                    q[i] = wp.vec3(0.0, 0.0, park_z)
                    qd[i] = wp.vec3(0.0, 0.0, 0.0)

            self._cull_kernel = _cull
        lo, hi = self._cull_bounds
        for st in (self._nm.get_mpm_state_0(),):
            wp.launch(self._cull_kernel, dim=model.particle_count,
                      inputs=[st.particle_q, st.particle_qd, model.particle_flags,
                              wp.vec3(*[float(x) for x in lo]), wp.vec3(*[float(x) for x in hi]), self._cull_count],
                      device=model.particle_flags.device)
        n = int(self._cull_count.numpy()[0])
        if n > self._culled:
            print(f"[water] culled {n - self._culled} particles that left the grid or went non-finite (total {n})", flush=True)
            self._culled = n

    def metrics(self) -> dict:
        positions = self.positions()
        if positions is None or not len(positions):
            return dict(total=0, caught=0, spilled=0, caught_frac=0.0, accuracy=0.0)
        parked = positions[:, 2] < self._CULL_PARK_Z + 1.0
        positions = positions[~parked]
        if not len(positions):
            return dict(total=int(parked.sum()), caught=0, spilled=0, in_vessel=0, culled=int(parked.sum()),
                        caught_frac=0.0, accuracy=0.0)
        cx, cy = self._cup_xy
        cfg = self.cfg
        cup_r = np.hypot(positions[:, 0] - cx, positions[:, 1] - cy)
        cup_z = positions[:, 2] - self.counter_top_z
        caught = (cup_r < cfg.cup_inner_radius) & (cup_z > 0.0) & (cup_z < cfg.cup_wall_height)
        v_p, v_q = self._measure(self._vessel)
        axis_w = _rotation(v_q) @ np.array([self.axis_xy_local[0], self.axis_xy_local[1], 0.0])
        ves_r = np.hypot(positions[:, 0] - (v_p[0] + axis_w[0]),
                         positions[:, 1] - (v_p[1] + axis_w[1]))
        in_vessel = (ves_r < self._collider.r_inner_min + 0.02) & ~caught
        landed = (cup_z < 0.02) & ~caught & ~in_vessel
        n_caught, n_spill = int(caught.sum()), int(landed.sum())
        delivered = n_caught + n_spill
        total = len(positions) + int(parked.sum())
        return dict(total=total, caught=n_caught, spilled=n_spill,
                    in_vessel=int(in_vessel.sum()), culled=int(parked.sum()),
                    caught_frac=n_caught / total,
                    accuracy=(n_caught / delivered) if delivered else 0.0)


_VESSEL_SCENE_NAMES = ("pour_jug", "idealisk_jug", "serving_jug")
"""The scene names a pour vessel spawns under, in the order the sidecar looks for them when
DEXBENCH_WATER_VESSEL is unset: E4's ``--pour_jug`` puts a Trinix jug at ``pour_jug`` and
IDEALISK at ``idealisk_jug``, and only one of the two is spawned."""


def _default_vessel_asset(env) -> str:
    """DEXBENCH_WATER_VESSEL, else the first of ``_VESSEL_SCENE_NAMES`` the scene holds, so
    switching the jug with ``--pour_jug`` needs no second setting."""
    named = os.environ.get("DEXBENCH_WATER_VESSEL")
    if named:
        return named
    keys = set(env.scene.keys())
    return next((name for name in _VESSEL_SCENE_NAMES if name in keys), _VESSEL_SCENE_NAMES[-1])


def _env_optional(name: str, parse):
    value = os.environ.get(name)
    return parse(value) if value else None


def water_cfg_from_environ(env) -> WaterPourSidecarCfg:
    """The teleop water's cfg: ``WaterPourSidecarCfg``'s defaults, each overridden by its DEXBENCH_WATER_* variable."""
    cfg = WaterPourSidecarCfg(
        vessel_asset=_default_vessel_asset(env),
        cup_asset=os.environ.get("DEXBENCH_WATER_CUP_ASSET", "serving_cup"),
        cup_profile_npz=(os.environ.get("DEXBENCH_WATER_CUP_PROFILE")
                         or (str(GOBLET_PROFILE_NPZ)
                             if os.environ.get("DEXBENCH_WATER_CUP_ASSET", "").startswith("pour_goblet")
                             else None)),
        target_ml=_env_optional("DEXBENCH_WATER_ML", float),
        fill=_env_float("DEXBENCH_WATER_FILL", 0.6),
        voxel_size=_env_float("DEXBENCH_WATER_VOXEL", 0.008),
        perf_every=int(_env_float("DEXBENCH_WATER_PERF_EVERY", 0)),
        particles_per_cell=_env_float("DEXBENCH_WATER_PPC", 2.0),
        stride=int(_env_float("DEXBENCH_WATER_STRIDE", 1)),
        max_iterations=int(_env_float("DEXBENCH_WATER_ITERS", 30)),
    )
    default = WaterPourSidecarCfg()
    cfg.substeps = _env_optional("DEXBENCH_WATER_SUBSTEPS", int)
    cfg.profile_npz = os.environ.get("DEXBENCH_WATER_PROFILE") or None
    cfg.render_radius_vox = _env_float("DEXBENCH_WATER_RENDER_RADIUS_VOX", default.render_radius_vox)
    cfg.counter_top_z = _env_optional("DEXBENCH_WATER_COUNTER_Z", float)
    cfg.counter_friction = _env_float("DEXBENCH_WATER_COUNTER_FRICTION", default.counter_friction)
    cfg.grid_type = os.environ.get("DEXBENCH_WATER_GRID_TYPE", default.grid_type)
    cfg.grid_padding = int(_env_float("DEXBENCH_WATER_GRID_PAD", default.grid_padding))
    cfg.max_active_cells_log2 = int(_env_float("DEXBENCH_WATER_CELLS_POW", default.max_active_cells_log2))
    cfg.collider_basis = os.environ.get("DEXBENCH_WATER_COLLIDER_BASIS") or default.collider_basis
    cfg.strain_basis = os.environ.get("DEXBENCH_WATER_STRAIN_BASIS") or default.strain_basis
    cfg.max_dt = _env_float("DEXBENCH_WATER_MAX_DT", default.max_dt)
    cfg.cull = os.environ.get("DEXBENCH_WATER_CULL", "1") != "0"
    cfg.cull_margin_vox = _env_float("DEXBENCH_WATER_CULL_MARGIN_VOX", default.cull_margin_vox)
    cfg.vessel_margin_vox = _env_optional("DEXBENCH_WATER_VESSEL_MARGIN_VOX", float)
    cfg.cup_box_proxy = os.environ.get("DEXBENCH_WATER_CUP", "mug") == "box"
    return cfg


def water_environment_variables() -> list[str]:
    """The DEXBENCH_WATER variables set in this process's environment (the teleop water's switch and its settings)."""
    return sorted(name for name in os.environ if name == "DEXBENCH_WATER" or name.startswith("DEXBENCH_WATER_"))


def refuse_water_environment(what: str) -> None:
    """Refuse to build ``what``'s water while DEXBENCH_WATER variables are set."""
    names = water_environment_variables()
    if names:
        raise RuntimeError(f"{what} builds its water from its pour config and reads no environment variable; "
                           f"unset {', '.join(names)} (they set the teleop water, DEXBENCH_WATER=1, only)")


def build_water_pour_sidecar(env, cfg: WaterPourSidecarCfg | None = None) -> WaterPourSidecar:
    """The teleop water: ``cfg``, else the environment's (``water_cfg_from_environ``), drawn as DEXBENCH_WATER_RENDER
    says (surface, the default; points; 0 for none)."""
    if cfg is None:
        cfg = water_cfg_from_environ(env)
    sidecar = WaterPourSidecar(env, cfg)
    render = os.environ.get("DEXBENCH_WATER_RENDER", "surface").lower()
    if render not in ("0", "false", "no", "off"):
        sidecar.enable_render("points" if render in ("points", "beads") else "surface")
    return sidecar
