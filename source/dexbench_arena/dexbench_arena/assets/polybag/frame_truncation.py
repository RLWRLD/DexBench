"""Self-contact truncation in each bag's own moving frame: the cap on a bag's *deformation*, not on
its *travel*."""

from __future__ import annotations

import inspect

import numpy as np
import warp as wp
from newton import ParticleFlags

MAX_GROUPS = 16
BLOCK = 256
OVER_SHARE = 0.01
OVER_MAX = 3.0
N_ACC = 19


@wp.kernel
def _moments(chunk_start: wp.array(dtype=wp.int32), chunk_group: wp.array(dtype=wp.int32),
             group_start: wp.array(dtype=wp.int32), group_end: wp.array(dtype=wp.int32), with_boxes: int,
             pos: wp.array(dtype=wp.vec3), disp: wp.array(dtype=wp.vec3),
             inv_mass: wp.array(dtype=float), flags: wp.array(dtype=wp.int32), active_flag: wp.int32,
             acc: wp.array2d(dtype=float), lo: wp.array2d(dtype=float), hi: wp.array2d(dtype=float)):
    """Block reduction of each group's displacement moments, positions taken relative to the group's
    first particle (float32 sums of metre-scale positions would cancel what the fit needs)."""
    c, lane = wp.tid()
    g = chunk_group[c]
    i = chunk_start[c] + lane
    w = float(0.0)
    q = wp.vec3(0.0)
    d = wp.vec3(0.0)
    p_lo = wp.vec3(1.0e30)
    p_hi = wp.vec3(-1.0e30)
    if i < group_end[g]:
        if inv_mass[i] > 0.0 and (flags[i] & active_flag) != 0:
            w = 1.0
            q = pos[i] - pos[group_start[g]]
            d = disp[i]
        p_lo = pos[i]
        p_hi = pos[i]
    sn = wp.tile_reduce(wp.add, wp.tile(w))[0]
    sq = wp.tile_reduce(wp.add, wp.tile(q, preserve_type=True))[0]
    sd = wp.tile_reduce(wp.add, wp.tile(d, preserve_type=True))[0]
    sqd = wp.tile_reduce(wp.add, wp.tile(wp.cross(q, d), preserve_type=True))[0]
    sqq = wp.tile_reduce(wp.add, wp.tile(wp.outer(q, q), preserve_type=True))[0]
    if lane == 0:
        wp.atomic_add(acc, g, 0, sn)
        for k in range(3):
            wp.atomic_add(acc, g, 1 + k, sq[k])
            wp.atomic_add(acc, g, 4 + k, sd[k])
            wp.atomic_add(acc, g, 7 + k, sqd[k])
            for m in range(3):
                wp.atomic_add(acc, g, 10 + 3 * k + m, sqq[k, m])
    if with_boxes != 0:
        b_lo = wp.tile_reduce(wp.min, wp.tile(p_lo, preserve_type=True))[0]
        b_hi = wp.tile_reduce(wp.max, wp.tile(p_hi, preserve_type=True))[0]
        if lane == 0:
            for k in range(3):
                wp.atomic_min(lo, g, k, b_lo[k])
                wp.atomic_max(hi, g, k, b_hi[k])


@wp.kernel
def _fit(n_groups: int, group_start: wp.array(dtype=wp.int32), pos: wp.array(dtype=wp.vec3),
         acc: wp.array2d(dtype=float), shift: wp.array(dtype=wp.vec3), rot: wp.array(dtype=wp.quat),
         centre: wp.array(dtype=wp.vec3)):
    """One thread: each group's rigid fit (translation = mean; rotation from the least-squares angular
    displacement about the centroid, I w = sum r x d); then clear the sums for the next call."""
    for g in range(n_groups):
        n = acc[g, 0]
        T = wp.vec3(0.0)
        c = wp.vec3(0.0)
        r = wp.quat_identity()
        if n > 0.0:
            sq = wp.vec3(acc[g, 1], acc[g, 2], acc[g, 3])
            sd = wp.vec3(acc[g, 4], acc[g, 5], acc[g, 6])
            sqd = wp.vec3(acc[g, 7], acc[g, 8], acc[g, 9])
            sqq = wp.mat33(acc[g, 10], acc[g, 11], acc[g, 12], acc[g, 13], acc[g, 14], acc[g, 15],
                           acc[g, 16], acc[g, 17], acc[g, 18])
            cq = sq / n
            T = sd / n
            L = sqd - n * wp.cross(cq, T)
            C = sqq - n * wp.outer(cq, cq)
            tr = C[0, 0] + C[1, 1] + C[2, 2]
            if tr > 0.0:
                inertia = (1.0 + 1.0e-6) * tr * wp.identity(n=3, dtype=float) - C
                omega = wp.inverse(inertia) * L
                angle = wp.length(omega)
                if angle > 1.0e-12:
                    r = wp.quat_from_axis_angle(omega / angle, angle)
            c = cq + pos[group_start[g]]
        shift[g] = T
        rot[g] = r
        centre[g] = c
        for k in range(N_ACC):
            acc[g, k] = 0.0


@wp.func
def _rigid(p: wp.vec3, T: wp.vec3, r: wp.quat, c: wp.vec3):
    return T + wp.quat_rotate(r, p - c) - (p - c)


@wp.kernel
def _spread(chunk_start: wp.array(dtype=wp.int32), chunk_group: wp.array(dtype=wp.int32),
            group_end: wp.array(dtype=wp.int32), pos: wp.array(dtype=wp.vec3), disp: wp.array(dtype=wp.vec3),
            shift: wp.array(dtype=wp.vec3), rot: wp.array(dtype=wp.quat), centre: wp.array(dtype=wp.vec3),
            max_disp: float, spread: wp.array(dtype=float), over: wp.array(dtype=float),
            reach: wp.array(dtype=float)):
    """Each group's largest remainder |d - M|, how many particles have one over D, and its largest
    distance from its centroid."""
    c, lane = wp.tid()
    g = chunk_group[c]
    i = chunk_start[c] + lane
    e = float(0.0)
    a = float(0.0)
    o = float(0.0)
    if i < group_end[g]:
        e = wp.length(disp[i] - _rigid(pos[i], shift[g], rot[g], centre[g]))
        a = wp.length(pos[i] - centre[g])
        if e > max_disp:
            o = 1.0
    se = wp.tile_reduce(wp.max, wp.tile(e))[0]
    sa = wp.tile_reduce(wp.max, wp.tile(a))[0]
    so = wp.tile_reduce(wp.add, wp.tile(o))[0]
    if lane == 0:
        wp.atomic_max(spread, g, se)
        wp.atomic_max(reach, g, sa)
        wp.atomic_add(over, g, so)


@wp.func
def _box_gap(lo: wp.array2d(dtype=float), hi: wp.array2d(dtype=float), a: int, b: int):
    s = float(0.0)
    for k in range(3):
        x = wp.max(wp.max(lo[b, k] - hi[a, k], lo[a, k] - hi[b, k]), 0.0)
        s += x * x
    return wp.sqrt(s)


@wp.func
def _angle(r: wp.quat):
    return 2.0 * wp.asin(wp.min(wp.length(wp.vec3(r[0], r[1], r[2])), 1.0))


@wp.kernel
def _decide(n_groups: int, max_disp: float, query_radius: float, over_share: float, over_max: float,
            group_start: wp.array(dtype=wp.int32), group_end: wp.array(dtype=wp.int32),
            shift: wp.array(dtype=wp.vec3), rot: wp.array(dtype=wp.quat), spread: wp.array(dtype=float),
            over: wp.array(dtype=float), reach: wp.array(dtype=float),
            lo: wp.array2d(dtype=float), hi: wp.array2d(dtype=float), use: wp.array(dtype=wp.int32),
            last_spread: wp.array(dtype=float)):
    """One thread: a group moves in its own frame only if it moves rigidly (few particles, and none by
    much, would be clamped there) and no other bag can reach it within this window; then clear the
    reductions (the largest remainder is kept in `last_spread`, for diagnostics)."""
    for g in range(n_groups):
        n = float(group_end[g] - group_start[g])
        rigid = spread[g] <= over_max * max_disp and over[g] <= over_share * n
        use[g] = wp.where(rigid, 1, 0)
    changed = int(1)
    rounds = int(0)
    while changed != 0 and rounds <= n_groups:
        changed = int(0)
        rounds += 1
        for a in range(n_groups):
            for b in range(a + 1, n_groups):
                if use[a] != 0 or use[b] != 0:
                    ta = wp.vec3(0.0)
                    tb = wp.vec3(0.0)
                    ra = float(0.0)
                    rb = float(0.0)
                    if use[a] != 0:
                        ta = shift[a]
                        ra = _angle(rot[a]) * reach[a]
                    if use[b] != 0:
                        tb = shift[b]
                        rb = _angle(rot[b]) * reach[b]
                    if _box_gap(lo, hi, a, b) <= query_radius + 2.0 * max_disp + wp.length(ta - tb) + ra + rb:
                        use[a] = 0
                        use[b] = 0
                        changed = int(1)
    for g in range(n_groups):
        last_spread[g] = spread[g]
        spread[g] = 0.0
        over[g] = 0.0
        reach[g] = 0.0
        for k in range(3):
            lo[g, k] = 1.0e30
            hi[g, k] = -1.0e30


@wp.kernel
def _relative(group: wp.array(dtype=wp.int32), use: wp.array(dtype=wp.int32), pos: wp.array(dtype=wp.vec3),
              disp: wp.array(dtype=wp.vec3), shift: wp.array(dtype=wp.vec3), rot: wp.array(dtype=wp.quat),
              centre: wp.array(dtype=wp.vec3), rel: wp.array(dtype=wp.vec3)):
    """The remainder in the bag's frame (rotated back), or the displacement itself."""
    i = wp.tid()
    g = group[i]
    d = disp[i]
    if g >= 0:
        if use[g] != 0:
            d = wp.quat_rotate_inv(rot[g], d - _rigid(pos[i], shift[g], rot[g], centre[g]))
    rel[i] = d


@wp.kernel
def _apply(group: wp.array(dtype=wp.int32), use: wp.array(dtype=wp.int32), pos: wp.array(dtype=wp.vec3),
           rel: wp.array(dtype=wp.vec3), ts: wp.array(dtype=float), max_disp: float,
           shift: wp.array(dtype=wp.vec3), rot: wp.array(dtype=wp.quat), centre: wp.array(dtype=wp.vec3),
           disp_out: wp.array(dtype=wp.vec3), pos_out: wp.array(dtype=wp.vec3)):
    """``apply_truncation_ts`` on the bag-frame remainder, then the bag's rigid motion back on."""
    i = wp.tid()
    g = group[i]
    d = rel[i] * ts[i]
    n = wp.length(d)
    if n > max_disp:
        d = d * max_disp / n
    if g >= 0:
        if use[g] != 0:
            d = _rigid(pos[i], shift[g], rot[g], centre[g]) + wp.quat_rotate(rot[g], d)
    disp_out[i] = d
    if pos_out:
        pos_out[i] = pos[i] + d


class FrameTruncation:
    """Replaces one SolverVBD instance's ``_penetration_free_truncation``. Graph-safe: fixed arrays,
    no host readback."""

    def __init__(self, solver, groups):
        from newton._src.solvers.vbd.particle_vbd_kernels import apply_planar_truncation_parallel_by_collision

        if not getattr(solver, "particle_enable_self_contact", False):
            raise ValueError("self-contact is off; there is no truncation to move into the bag frame")
        groups = [np.sort(np.asarray(g, dtype=np.int64)) for g in groups if len(g)]
        if not 0 < len(groups) <= MAX_GROUPS:
            raise ValueError(f"{len(groups)} groups; 1..{MAX_GROUPS} supported")
        model = solver.model
        label = np.full(model.particle_count, -1, dtype=np.int32)
        starts, owner, g_start, g_end = [], [], [], []
        for k, g in enumerate(groups):
            if g[-1] - g[0] + 1 != len(g):
                raise ValueError("a group must be a contiguous run of particle indices (a bag's slice)")
            if (label[g] >= 0).any():
                raise ValueError("a particle is in two groups")
            label[g] = k
            g_start.append(int(g[0]))
            g_end.append(int(g[-1]) + 1)
            for a in range(int(g[0]), int(g[-1]) + 1, BLOCK):
                starts.append(a)
                owner.append(k)
        d = solver.device

        def i32(v):
            return wp.array(np.asarray(v, dtype=np.int32), dtype=wp.int32, device=d)

        self.solver, self.model, self.n = solver, model, len(groups)
        self.planar = apply_planar_truncation_parallel_by_collision
        self.planar_strided = "stride" in [a.label for a in self.planar.adj.args]
        self.active_flag = int(ParticleFlags.ACTIVE)
        self.group = i32(label)
        self.chunk_start, self.chunk_group, self.group_start, self.group_end = (
            i32(starts), i32(owner), i32(g_start), i32(g_end))
        self.n_chunks = len(starts)
        self.rel = wp.zeros(model.particle_count, dtype=wp.vec3, device=d)
        self.acc = wp.zeros((self.n, N_ACC), dtype=float, device=d)
        self.lo = wp.full((self.n, 3), 1.0e30, dtype=float, device=d)
        self.hi = wp.full((self.n, 3), -1.0e30, dtype=float, device=d)
        self.spread = wp.zeros(self.n, dtype=float, device=d)
        self.over = wp.zeros(self.n, dtype=float, device=d)
        self.reach = wp.zeros(self.n, dtype=float, device=d)
        self.shift = wp.zeros(self.n, dtype=wp.vec3, device=d)
        self.rot = wp.array([wp.quat_identity()] * self.n, dtype=wp.quat, device=d)
        self.centre = wp.zeros(self.n, dtype=wp.vec3, device=d)
        self.use = wp.zeros(self.n, dtype=wp.int32, device=d)
        self.last_spread = wp.zeros(self.n, dtype=float, device=d)
        self.api = _api(solver)
        if getattr(solver, "rigid_soft_enable_dat", False):
            raise SystemExit("rigid-soft DAT is on; frame_truncation.py handles the self-contact truncation only")
        self.query_radius = float(getattr(solver, "_self_contact_query_radius", None)
                                  or solver.particle_self_contact_margin)
        relax = getattr(solver, "dat_conservative_bound_relaxation", None)
        self.relaxation = float(relax if relax is not None else solver.particle_conservative_bound_relaxation)
        if self.api == "1.5":
            eps = getattr(solver, "_self_contact_edge_edge_parallel_epsilon", None)
            self.parallel_eps = float(eps if eps is not None else
                                      solver.trimesh_collision_detector.edge_edge_parallel_epsilon)
        self.max_disp = self.query_radius * self.relaxation * 0.5

    def __call__(self, *args):
        """1.5.x / 1.6.1: (particle_q_out=None). main: (state, contacts) -- writes state.particle_q."""
        particle_q_out = (args[0] if args else None) if self.api == "1.5" else args[0].particle_q
        s, m, d = self.solver, self.model, self.solver.device
        ref, disp = s.pos_prev_collision_detection, s.particle_displacements
        wp.launch_tiled(_moments, dim=[self.n_chunks], block_dim=BLOCK,
                        inputs=[self.chunk_start, self.chunk_group, self.group_start, self.group_end, int(self.n > 1),
                                ref, disp, m.particle_inv_mass, m.particle_flags, self.active_flag,
                                self.acc, self.lo, self.hi], device=d)
        wp.launch(_fit, dim=1, inputs=[self.n, self.group_start, ref, self.acc, self.shift, self.rot, self.centre],
                  device=d)
        wp.launch_tiled(_spread, dim=[self.n_chunks], block_dim=BLOCK,
                        inputs=[self.chunk_start, self.chunk_group, self.group_end, ref, disp, self.shift, self.rot,
                                self.centre, self.max_disp, self.spread, self.over, self.reach], device=d)
        wp.launch(_decide, dim=1, inputs=[self.n, self.max_disp, self.query_radius, OVER_SHARE, OVER_MAX,
                                          self.group_start, self.group_end, self.shift, self.rot, self.spread,
                                          self.over, self.reach, self.lo, self.hi, self.use, self.last_spread],
                  device=d)
        wp.launch(_relative, dim=m.particle_count,
                  inputs=[self.group, self.use, ref, disp, self.shift, self.rot, self.centre, self.rel], device=d)
        s.truncation_ts.fill_(1.0)
        n = s.particle_self_contact_evaluation_kernel_launch_size
        if self.api == "1.5":
            planar_args = [ref, self.rel, m.tri_indices, m.edge_indices, s.trimesh_collision_info, self.parallel_eps,
                           self.relaxation]
        else:
            planar_args = [ref, self.rel, m.tri_indices, m.edge_indices, s.trimesh_collision_info, self.relaxation]
            if self.planar_strided:
                planar_args.append(n)
        wp.launch(self.planar, dim=n, inputs=planar_args, outputs=[s.truncation_ts], device=d)
        wp.launch(_apply, dim=m.particle_count,
                  inputs=[self.group, self.use, ref, self.rel, s.truncation_ts, self.max_disp, self.shift, self.rot,
                          self.centre],
                  outputs=[disp, particle_q_out], device=d)


SCOPES = ("init", "all")


def _api(solver) -> str:
    """Which SolverVBD truncation this is: "1.5" (1.5.x and 1.6.1: _penetration_free_truncation(particle_q_out))
    or "main" (Newton main: (state, contacts), the planar cut strided over stored pairs)."""
    params = list(inspect.signature(type(solver)._penetration_free_truncation).parameters)
    if params == ["self", "particle_q_out"]:
        return "1.5"
    if params == ["self", "state", "contacts"]:
        return "main"
    raise SystemExit(f"this SolverVBD's _penetration_free_truncation{tuple(params[1:])} is not one "
                     "frame_truncation.py knows (1.5.x / 1.6.1: particle_q_out; main: state, contacts)")


def install(solver, groups, scope: str = "init") -> FrameTruncation:
    """Swap `solver`'s truncation for the bag-frame one; `groups`: one particle index range per bag."""
    if scope not in SCOPES:
        raise ValueError(f"scope {scope!r}; one of {SCOPES}")
    ft = FrameTruncation(solver, groups)
    if scope == "all":
        solver._penetration_free_truncation = ft
        return ft
    initialize = solver._initialize_particles

    def _initialize_particles(*a, **kw):
        solver._penetration_free_truncation = ft
        try:
            return initialize(*a, **kw)
        finally:
            del solver._penetration_free_truncation

    solver._initialize_particles = _initialize_particles
    return ft
