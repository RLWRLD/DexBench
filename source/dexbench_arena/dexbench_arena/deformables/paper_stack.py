# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""VBD paper-stack sidecar for DexBench scene environments."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field, replace

import numpy as np

import newton
import warp as wp
from isaaclab_arena.utils.yaw import yaw_from_quat_xyzw

_ACTIVE: PaperStackVBD | None = None

def set_active_paper_stack(stack: PaperStackVBD | None) -> None:
    """Register the paper stack read by task termination predicates."""

    global _ACTIVE
    _ACTIVE = stack


def get_active_paper_stack() -> PaperStackVBD | None:
    """Return the active paper stack, or None when no sidecar is running."""

    return _ACTIVE


@dataclass
class StaticBoxCollider:
    """A static box collider in the sidecar world (world frame) [m]."""

    center_xyz: tuple[float, float, float]
    half_extents_xyz: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    """World orientation of the box's axes (a tilted box flap); axis-aligned by default."""


@dataclass
class OpenBoxCfg:
    """Open-top box: interior success volume + 5 thin mirror-collider walls [m]."""

    center_xy: tuple[float, float]
    floor_z: float
    """World z of the interior floor (top face of the bottom slab)."""
    interior_half_x: float
    interior_half_y: float
    interior_depth: float
    """Interior wall height; the rim is at ``floor_z + interior_depth``."""
    wall_thickness: float = 0.004
    yaw_rad: float = 0.0
    """Yaw of the box about +Z, about ``center_xy``: the interior half extents are along
    the box's own axes, and the walls and the success volume turn with it."""

    @property
    def rim_z(self) -> float:
        return self.floor_z + self.interior_depth

    def _rotation(self) -> np.ndarray:
        """The box axes in the world, as a 3x3 matrix (columns)."""

        q = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), self.yaw_rad)
        return np.array(wp.quat_to_matrix(q), dtype=float).reshape(3, 3)

    def moved(self, pos_xyz, quat_xyzw) -> OpenBoxCfg:
        """This box carried by a world transform that is a turn about +Z and a shift (``pos_xyz``,
        ``quat_xyzw``): the box a bench or box pose draw puts where this one was."""

        turn = wp.quat(*(float(v) for v in quat_xyzw))
        center = wp.quat_rotate(turn, wp.vec3(self.center_xy[0], self.center_xy[1], self.floor_z))
        yaw = yaw_from_quat_xyzw(tuple(float(v) for v in quat_xyzw))
        return replace(
            self,
            center_xy=(float(center[0] + pos_xyz[0]), float(center[1] + pos_xyz[1])),
            floor_z=float(center[2] + pos_xyz[2]),
            yaw_rad=self.yaw_rad + yaw,
        )

    def to_box_frame(self, points: np.ndarray) -> np.ndarray:
        """World points (N, 3) in the box frame: origin on ``center_xy``, axes the box's, z kept."""

        rel = np.asarray(points, dtype=float) - np.array([*self.center_xy, 0.0])
        return rel @ self._rotation()

    def collider_boxes(self) -> list[StaticBoxCollider]:
        """Bottom slab + 4 thin walls as static mirror colliders, turned with the box."""

        t = self.wall_thickness
        hx, hy, h = self.interior_half_x, self.interior_half_y, self.interior_depth
        wall_cz = self.floor_z + h / 2
        local = [
            ((0.0, 0.0, self.floor_z - t / 2), (hx + t, hy + t, t / 2)),
            ((-(hx + t / 2), 0.0, wall_cz), (t / 2, hy + t, h / 2)),
            ((hx + t / 2, 0.0, wall_cz), (t / 2, hy + t, h / 2)),
            ((0.0, -(hy + t / 2), wall_cz), (hx, t / 2, h / 2)),
            ((0.0, hy + t / 2, wall_cz), (hx, t / 2, h / 2)),
        ]
        rot = self._rotation()
        quat = tuple(float(v) for v in wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), self.yaw_rad))
        boxes = []
        for center, half in local:
            world = rot @ np.array(center) + np.array([*self.center_xy, 0.0])
            boxes.append(StaticBoxCollider(tuple(float(v) for v in world), half, quat))
        return boxes


@dataclass
class StackSiteCfg:
    """One pile of sheets (e.g. an invoice stack or a flyer stack) [m]."""

    center_xy: tuple[float, float]
    base_z: float
    """World z of the surface the pile rests on."""
    num_sheets: int = 8
    sheet_size_xy: tuple[float, float] = (0.148, 0.210)
    """Sheet extents (local x, y) [m]; default A5."""
    yaw_rad: float = 0.0
    color_rgb: tuple[float, float, float] = (0.92, 0.92, 0.88)
    """Render displayColor of this pile's sheets."""
    texture_path: str | None = None
    """Image stretched over each sheet (u along local x, v along local y); display only.
    ``None`` renders the flat ``color_rgb``."""


@dataclass
class PaperStackVBDCfg:
    """Paper stack + VBD solver parameters (poker-cards recipe, meters/kg)."""

    sites: tuple[StackSiteCfg, ...] = ()
    """Sheet piles, at least one."""

    resolution: tuple[int, int] = (4, 6)
    """Cloth grid cells along (x, y), shared by all sites. 4x6 matches the
    poker-cards example; the per-sheet particle count must be uniform."""
    sheet_mass_kg: float = 0.0025
    """Mass per sheet when ``areal_density_kg_m2`` is None."""
    areal_density_kg_m2: float | None = 0.180
    """Paper weight (180 gsm, the teleop stock). Sheet mass = this times the
    sheet area, so it follows the measured sheet size; None uses ``sheet_mass_kg``."""
    sheet_spacing_m: float = 0.0015
    drop_gap_m: float = 0.009
    """Gap between the bench top and the lowest sheet at seeding (above
    ``particle_radius`` so the pile does not start inside the slab contact)."""
    jitter_m: float = 0.003
    seed: int = 7

    tri_ke: float = 1.0e4
    tri_ka: float = 1.0e4
    tri_kd: float = 1.0e-4
    edge_ke: float = 5.0
    edge_kd: float = 1
    particle_radius: float = 0.005
    """The sheet's apparent half thickness for contact -- the paper's thickness
    as the fingers feel it. Static colliders are sunk by this much
    (``collider_sink_m``) so the cloth still rests on the visible surface."""
    collider_sink_m: float | None = None
    """How far the static colliders sit below the surface they copy, so the
    cloth midplane (which floats ``particle_radius`` above a collider) is on the
    visible surface. None = ``particle_radius``."""

    contact_ke: float = 1.0e5
    contact_kd: float = 1.0e-4
    contact_mu: float = 0.35
    collider_mu: float = 0.5
    kinematic_mu: float | None = None
    """Friction for the kinematic mirror bodies (hand colliders). ``None`` reuses
    ``collider_mu``, which is tuned for the bench and carton the sheets rest on.
    A physical pinch has to hold a sheet against gravity on friction alone, so
    the hand wants a higher value than the surfaces the paper slides across."""
    soft_contact_margin: float = 0.012
    enable_rigid_soft_full_surface_contact: bool = True
    """Collide the sheet's edges and triangle interiors with the rigid colliders,
    not just its particles (newton#3756, Newton >= 1.5). The per-particle
    path lets a rigid feature narrower than a cloth cell -- a bench corner, a
    fingertip -- slip between vertices; the full-surface pass catches it. Only
    VBD consumes these contacts. The stack raises on a Newton without the flag
    rather than falling back to vertex-only contact."""

    substeps: int = 20
    iterations: int = 20
    particle_self_contact: bool = True
    """Paper-paper contact. False: sheets pass through each other and through
    themselves, and only colliders and hand mirrors push the cloth."""
    self_contact_radius: float = 0.001
    self_contact_margin: float = 0.0015

    kinematic_hulls: dict[str, list] = field(default_factory=dict)
    """Named kinematic mirror bodies made of convex hulls: name -> list of
    (vertices (N,3), triangles (M,3)) in that body's frame, posed by the proxy
    pass (``proxy_mirrors``); with full-surface contact each hull gets an SDF."""
    hull_sdf_voxel_m: float = 0.001
    """SDF voxel size for the hull mirrors (full-surface contact only). A
    fingertip hull is ~20 mm across, so 1 mm keeps its curvature."""
    hull_sdf_band_m: float = 0.05
    """Outer narrow band and AABB padding of the hull SDFs. Newton 1.5.2 reads
    0 for a grid cell outside the band but inside the padded box, so with a
    narrow band points well off a finger read as touching it; the face
    search over a 5 cm cloth triangle finds those cells and pushes the sheet
    before the hand reaches it. With band = padding the empty cells are left
    only in the box corners, more than this far from the hull (outside the box
    the sampler extrapolates correctly). Keep it well above
    ``soft_contact_margin`` + ``particle_radius``."""

    proxy_mirrors: bool = False
    """Build the hand mirrors (``kinematic_hulls``) as dynamic
    bodies flagged ``BodyFlags.PROXY`` for the proxy pass in
    ``paper_coupling.py``, which poses them every substep from the main solver; hand mirrors
    need it. Needs ``enable_rigid_soft_full_surface_contact``."""
    body_contact_buffer_size: int | None = 1024
    """VBD's per-body particle contact buffer (edge and face records included);
    None keeps Newton's default (256), which a palm over a few cells with
    full-surface contact overflows, dropping the excess."""


class PaperStackVBD:
    """Sidecar world: N cloth sheets + static mirror colliders + hand mirrors."""

    def __init__(
        self,
        cfg: PaperStackVBDCfg,
        box: OpenBoxCfg,
        extra_colliders: tuple[StaticBoxCollider, ...] = (),
        device: str | None = None,
    ):
        self.cfg = cfg
        self.box = box
        self.device = device

        if cfg.enable_rigid_soft_full_surface_contact and (
            "enable_rigid_soft_full_surface_contact" not in inspect.signature(newton.CollisionPipeline).parameters
        ):
            raise RuntimeError(
                f"Newton {getattr(newton, '__version__', '?')} lacks full-surface rigid-soft contacts "
                "(newton#3756, Newton >= 1.5.0). Run inside the DexBench container; the stack does not "
                "fall back to vertex-only contact."
            )

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.8))

        static_cfg = newton.ModelBuilder.ShapeConfig()
        static_cfg.density = 0.0
        static_cfg.ke = cfg.contact_ke
        static_cfg.kd = cfg.contact_kd
        static_cfg.mu = cfg.collider_mu
        static_cfg.has_shape_collision = False
        if cfg.proxy_mirrors and not cfg.enable_rigid_soft_full_surface_contact:
            raise ValueError("proxy_mirrors needs enable_rigid_soft_full_surface_contact (the coupled pass is built on it)")
        if cfg.kinematic_hulls and not cfg.proxy_mirrors:
            raise ValueError("the hand mirrors are proxies of the main solver's bodies; set proxy_mirrors")

        sink = cfg.particle_radius if cfg.collider_sink_m is None else cfg.collider_sink_m
        self._static_body_ids: dict[str, int] = {}
        for group, colliders in (("box", box.collider_boxes()), ("bench", extra_colliders)):
            statics = builder.add_body(xform=wp.transform(), label=f"static_colliders_{group}")
            self._static_body_ids[group] = statics
            for coll in colliders:
                cx, cy, cz = coll.center_xyz
                builder.add_shape_box(
                    statics,
                    xform=wp.transform(p=wp.vec3(cx, cy, cz - sink), q=wp.quat(*coll.rotation_xyzw)),
                    hx=coll.half_extents_xyz[0],
                    hy=coll.half_extents_xyz[1],
                    hz=coll.half_extents_xyz[2],
                    cfg=static_cfg,
                )
        self.built_box = box
        """The box as built; ``set_static_pose("box", ...)`` moves ``box`` from it."""

        kinematic_cfg = newton.ModelBuilder.ShapeConfig()
        kinematic_cfg.density = 0.0
        kinematic_cfg.ke = cfg.contact_ke
        kinematic_cfg.kd = cfg.contact_kd
        kinematic_cfg.mu = cfg.collider_mu if cfg.kinematic_mu is None else cfg.kinematic_mu

        mirror_body_kwargs = (
            dict(mass=1.0, inertia=wp.mat33(1e-4, 0.0, 0.0, 0.0, 1e-4, 0.0, 0.0, 0.0, 1e-4))
            if cfg.proxy_mirrors else {}
        )
        mirror_ids: dict[str, int] = {}
        if cfg.kinematic_hulls:
            hull_cfg = newton.ModelBuilder.ShapeConfig()
            hull_cfg.density = 0.0
            hull_cfg.ke = kinematic_cfg.ke
            hull_cfg.kd = kinematic_cfg.kd
            hull_cfg.mu = kinematic_cfg.mu
            n_hulls = 0
            for k, (name, hulls) in enumerate(cfg.kinematic_hulls.items()):
                body = builder.add_body(xform=wp.transform(), label=name, **mirror_body_kwargs)
                link_cfg = newton.ModelBuilder.ShapeConfig()
                link_cfg.density, link_cfg.ke, link_cfg.kd, link_cfg.mu = (
                    hull_cfg.density, hull_cfg.ke, hull_cfg.kd, hull_cfg.mu
                )
                link_cfg.collision_group = 1000 + k
                for verts, tris in hulls:
                    mesh = newton.Mesh(
                        np.asarray(verts, dtype=np.float32),
                        np.asarray(tris, dtype=np.int32).reshape(-1),
                        compute_inertia=False,
                    )
                    if cfg.enable_rigid_soft_full_surface_contact:
                        mesh.build_sdf(
                            device=device,
                            target_voxel_size=cfg.hull_sdf_voxel_m,
                            narrow_band_range=(-0.01, cfg.hull_sdf_band_m),
                            margin=cfg.hull_sdf_band_m,
                        )
                    builder.add_shape_convex_hull(body, mesh=mesh, cfg=link_cfg)
                    n_hulls += 1
                mirror_ids[name] = body
            print(f"[paper_stack] hand mirrors: {len(cfg.kinematic_hulls)} bodies, {n_hulls} convex hulls"
                  f"{' (SDF %.1f mm)' % (cfg.hull_sdf_voxel_m * 1000) if cfg.enable_rigid_soft_full_surface_contact else ''}"
                  f"{', proxies' if cfg.proxy_mirrors else ''}")
        self._mirror_body_ids: dict[str, int] = mirror_ids

        if not cfg.sites:
            raise ValueError("PaperStackVBDCfg.sites is empty: the paper needs at least one pile")
        self.sites: tuple[StackSiteCfg, ...] = cfg.sites

        dim_x, dim_y = cfg.resolution
        self.particles_per_sheet = (dim_x + 1) * (dim_y + 1)
        self._sheet_site: list[int] = []
        rng = np.random.default_rng(cfg.seed)
        yaw_axis = wp.vec3(0.0, 0.0, 1.0)
        for site_index, site in enumerate(self.sites):
            sheet_w, sheet_h = site.sheet_size_xy
            sheet_mass = (
                cfg.areal_density_kg_m2 * sheet_w * sheet_h
                if cfg.areal_density_kg_m2 is not None
                else cfg.sheet_mass_kg
            )
            print(
                f"[paper_stack] site {site_index}: {site.num_sheets} sheets "
                f"{sheet_w * 1e3:.0f}x{sheet_h * 1e3:.0f} mm, {sheet_mass * 1e3:.1f} g each"
                + (f" ({cfg.areal_density_kg_m2 * 1e3:.0f} gsm)" if cfg.areal_density_kg_m2 is not None else "")
            )
            for i in range(site.num_sheets):
                jx, jy = rng.uniform(-cfg.jitter_m, cfg.jitter_m, size=2)
                builder.add_cloth_grid(
                    pos=wp.vec3(
                        site.center_xy[0] - sheet_w / 2 + jx,
                        site.center_xy[1] - sheet_h / 2 + jy,
                        site.base_z + cfg.drop_gap_m + i * cfg.sheet_spacing_m,
                    ),
                    rot=wp.quat_from_axis_angle(yaw_axis, site.yaw_rad + float(rng.uniform(-0.05, 0.05))),
                    vel=wp.vec3(0.0, 0.0, 0.0),
                    dim_x=dim_x,
                    dim_y=dim_y,
                    cell_x=sheet_w / dim_x,
                    cell_y=sheet_h / dim_y,
                    mass=sheet_mass / self.particles_per_sheet,
                    tri_ke=cfg.tri_ke,
                    tri_ka=cfg.tri_ka,
                    tri_kd=cfg.tri_kd,
                    edge_ke=cfg.edge_ke,
                    edge_kd=cfg.edge_kd,
                    particle_radius=cfg.particle_radius,
                )
                self._sheet_site.append(site_index)

        builder.color(include_bending=True)
        self.model = builder.finalize(device=device)
        self.model.soft_contact_ke = cfg.contact_ke
        self.model.soft_contact_kd = cfg.contact_kd
        self.model.soft_contact_mu = cfg.contact_mu
        if self._mirror_body_ids:
            flags = self.model.body_flags.numpy()
            flags[list(self._mirror_body_ids.values())] |= int(newton.BodyFlags.PROXY)
            self.model.body_flags.assign(flags)

        solver_kwargs = {}
        if cfg.body_contact_buffer_size is not None:
            solver_kwargs["rigid_body_particle_contact_buffer_size"] = cfg.body_contact_buffer_size
        pipeline_kwargs: dict[str, object] = {
            "broad_phase": "nxn",
            "soft_contact_gap": cfg.soft_contact_margin,
        }
        if cfg.enable_rigid_soft_full_surface_contact:
            pipeline_kwargs["enable_rigid_soft_full_surface_contact"] = True
        self.collision_pipeline = newton.CollisionPipeline(self.model, **pipeline_kwargs)
        self.solver = newton.solvers.SolverVBD(
            model=self.model,
            iterations=cfg.iterations,
            **solver_kwargs,
            particle_enable_self_contact=cfg.particle_self_contact,
            particle_self_contact_margin=cfg.self_contact_radius,
            particle_self_contact_gap=cfg.self_contact_margin - cfg.self_contact_radius,
            particle_topological_contact_filter_threshold=2,
            particle_rest_shape_contact_exclusion_radius=0.0,
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.collision_pipeline.contacts()

        self._initial_q = self.state_0.particle_q.numpy().copy()
        self.reset_offset_xy: tuple[float, float] = (0.0, 0.0)
        """Added to the piles' rest pose by ``reset`` (the paper-pile variation's offset) [m]."""
        self.reset_yaw_rad: float = 0.0
        """Turn of the piles' rest pose about the first pile's centre, applied by ``reset`` before
        ``reset_offset_xy`` (the paper-pile variation's yaw) [rad]."""
        self.reset_offset_z: float = 0.0
        """Added to the piles' rest height by ``reset`` (the bench-height variation's draw) [m]."""

        self._body_q_np: np.ndarray | None = None
        self._body_qd_np: np.ndarray | None = None

        self._render_meshes: list = []


    def set_static_pose(self, group: str, pos_xyz, quat_xyzw) -> None:
        """Carry a group of static colliders (``"box"`` or ``"bench"``) by a world transform from
        where they were built; the box group also moves the success volume (``box``)."""

        if self._body_q_np is None:
            self._body_q_np = self.state_0.body_q.numpy().copy()
            self._body_qd_np = self.state_0.body_qd.numpy().copy()
        body = self._static_body_ids[group]
        pose = (*(float(v) for v in pos_xyz), *(float(v) for v in quat_xyzw))
        self._body_q_np[body] = pose
        self.state_0.body_q.assign(self._body_q_np)
        self.state_1.body_q.assign(self._body_q_np)
        model_q = self.model.body_q.numpy()
        model_q[body] = pose
        self.model.body_q.assign(model_q)
        notify = getattr(self.solver, "coupling_notify_input_state_update", None)
        if notify is not None:
            notify(self.state_0, newton.StateFlags.BODY_Q, dt=0.0)
        if group == "box":
            self.box = self.built_box.moved(pos_xyz, quat_xyzw)

    def move_sheet_to(self, sheet_index: int, center_xyz: tuple[float, float, float]) -> None:
        """Teleport one sheet so its centroid sits at ``center_xyz``, at rest."""

        sheet = self.sheet_slice(sheet_index)
        for state in (self.state_0, self.state_1):
            q = state.particle_q.numpy()
            q[sheet] += np.asarray(center_xyz, dtype=q.dtype) - q[sheet].mean(axis=0)
            state.particle_q.assign(q)
            qd = state.particle_qd.numpy()
            qd[sheet] = 0.0
            state.particle_qd.assign(qd)

    def reset(self) -> None:
        """Re-seed all sheets to their initial stacked poses, turned by ``reset_yaw_rad`` and moved by
        ``reset_offset_xy`` and ``reset_offset_z`` (env reset)."""

        q = self._initial_q
        if self.reset_yaw_rad:
            pivot = np.array([*self.sites[0].center_xy, 0.0])
            turn = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(self.reset_yaw_rad))
            rot = np.array(wp.quat_to_matrix(turn), dtype=float).reshape(3, 3)
            q = ((q - pivot) @ rot.T + pivot).astype(q.dtype)
        if any(self.reset_offset_xy) or self.reset_offset_z:
            q = q.copy()
            q[:, 0] += self.reset_offset_xy[0]
            q[:, 1] += self.reset_offset_xy[1]
            q[:, 2] += self.reset_offset_z
        self.state_0.particle_q.assign(q)
        self.state_0.particle_qd.zero_()
        self.state_1.particle_q.assign(q)
        self.state_1.particle_qd.zero_()


    @property
    def num_sheets(self) -> int:
        return len(self._sheet_site)

    def sheet_site(self, sheet_index: int) -> StackSiteCfg:
        return self.sites[self._sheet_site[sheet_index]]

    def sheets_of_site(self, site_index: int) -> list[int]:
        return [i for i, s in enumerate(self._sheet_site) if s == site_index]

    def sheet_slice(self, sheet_index: int) -> slice:
        n = self.particles_per_sheet
        return slice(sheet_index * n, (sheet_index + 1) * n)

    def sheet_positions(self, sheet_index: int) -> np.ndarray:
        return self.state_0.particle_q.numpy()[self.sheet_slice(sheet_index)]

    def sheet_centroid(self, sheet_index: int) -> np.ndarray:
        return self.sheet_positions(sheet_index).mean(axis=0)

    def sheet_max_speed(self, sheet_index: int) -> float:
        qd = self.state_0.particle_qd.numpy()[self.sheet_slice(sheet_index)]
        return float(np.linalg.norm(qd, axis=1).max())

    def sheet_in_box(
        self,
        sheet_index: int,
        xy_margin_m: float = 0.01,
        min_inside_frac: float = 0.9,
        settle_speed_m_s: float = 0.30,
    ) -> bool:
        """True when the sheet rests inside the open box's interior volume."""

        box = self.box
        q = box.to_box_frame(self.sheet_positions(sheet_index))
        in_x = np.abs(q[:, 0]) < box.interior_half_x + xy_margin_m
        in_y = np.abs(q[:, 1]) < box.interior_half_y + xy_margin_m
        in_z = (q[:, 2] > box.floor_z - 0.005) & (q[:, 2] < box.rim_z + 0.01)
        inside_frac = float(np.mean(in_x & in_y & in_z))

        centroid = q.mean(axis=0)
        centroid_in = (
            abs(centroid[0]) < box.interior_half_x
            and abs(centroid[1]) < box.interior_half_y
            and box.floor_z - 0.005 < centroid[2] < box.rim_z
        )
        settled = self.sheet_max_speed(sheet_index) < settle_speed_m_s
        return bool(centroid_in and inside_frac >= min_inside_frac and settled)

    def any_sheet_in_box(self, **kwargs) -> bool:
        return any(self.sheet_in_box(i, **kwargs) for i in range(self.num_sheets))

    def any_sheet_dropped(self, drop_height_m: float) -> bool:
        """True when any sheet fell ``drop_height_m`` below its own site's base."""

        return any(
            float(self.sheet_centroid(i)[2]) < self.sheet_site(i).base_z + self.reset_offset_z - drop_height_m
            for i in range(self.num_sheets)
        )


    def create_usd_meshes(self, stage, root_path: str = "/World/dexbench_paper_stack") -> None:
        """Author one doubleSided UsdGeom.Mesh per sheet (points updated per frame)."""

        from pxr import Gf, UsdGeom

        tri_indices = self.model.tri_indices.numpy()
        UsdGeom.Xform.Define(stage, root_path)
        q = self.state_0.particle_q.numpy()
        self._render_meshes = []
        for i in range(self.num_sheets):
            sheet = self.sheet_slice(i)
            tris = tri_indices[(tri_indices[:, 0] >= sheet.start) & (tri_indices[:, 0] < sheet.stop)]
            local = tris - sheet.start
            mesh = UsdGeom.Mesh.Define(stage, f"{root_path}/sheet_{i:02d}")
            mesh.CreateFaceVertexCountsAttr().Set([3] * len(local))
            mesh.CreateFaceVertexIndicesAttr().Set([int(v) for v in local.flatten()])
            mesh.CreateDoubleSidedAttr().Set(True)
            mesh.CreateDisplayColorAttr().Set([Gf.Vec3f(*[float(c) for c in self.sheet_site(i).color_rgb])])
            mesh.CreatePointsAttr().Set([Gf.Vec3f(*[float(v) for v in p]) for p in q[sheet]])
            texture = self.sheet_site(i).texture_path
            if texture:
                self._bind_sheet_texture(stage, mesh, texture)
            self._render_meshes.append(mesh)

    def _bind_sheet_texture(self, stage, mesh, texture: str) -> None:
        """Give ``mesh`` per-particle UVs from its cloth grid and a UsdPreviewSurface reading ``texture``."""

        from pxr import Sdf, UsdGeom, UsdShade, Vt

        dim_x, dim_y = self.cfg.resolution
        u, v = np.meshgrid(np.linspace(0.0, 1.0, dim_x + 1), np.linspace(0.0, 1.0, dim_y + 1))
        st = np.stack([u.ravel(), v.ravel()], axis=1).astype(np.float32)
        primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                         UsdGeom.Tokens.vertex)
        primvar.Set(Vt.Vec2fArray.FromNumpy(st))

        mat_path = Sdf.Path(f"{mesh.GetPath()}_material")
        material = UsdShade.Material.Define(stage, mat_path)
        surface = UsdShade.Shader.Define(stage, mat_path.AppendChild("surface"))
        surface.CreateIdAttr("UsdPreviewSurface")
        surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
        surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        reader = UsdShade.Shader.Define(stage, mat_path.AppendChild("st_reader"))
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        image = UsdShade.Shader.Define(stage, mat_path.AppendChild("image"))
        image.CreateIdAttr("UsdUVTexture")
        image.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture)
        image.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        image.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            reader.ConnectableAPI(), "result")
        surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            image.ConnectableAPI(), "rgb")
        material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)

    def update_usd_meshes(self) -> None:
        from pxr import Gf, Vt

        if not self._render_meshes:
            return
        q = self.state_0.particle_q.numpy()
        for i, mesh in enumerate(self._render_meshes):
            sheet = q[self.sheet_slice(i)].astype(np.float32)
            try:
                points = Vt.Vec3fArray.FromNumpy(sheet)
            except AttributeError:
                points = Vt.Vec3fArray([Gf.Vec3f(*[float(v) for v in p]) for p in sheet])
            mesh.GetPointsAttr().Set(points)
