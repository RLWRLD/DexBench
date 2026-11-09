# SPDX-License-Identifier: Apache-2.0
"""Newton physics for a scene with a polybag: the robot on MJWarp, the bag on VBD."""

from __future__ import annotations

import contextlib
import inspect
import io
import logging
import math
import re
import time
from typing import Literal

import newton
import numpy as np
import warp as wp
from newton import CollisionPipeline
from newton.solvers import SolverVBD
from pxr import Gf

from isaaclab.physics import PhysicsManager
from isaaclab.utils.configclass import configclass
from isaaclab_contrib.custom_coupling.coupled_mjwarp_vbd_manager import NewtonCoupledMJWarpVBDManager
from isaaclab_contrib.custom_coupling.newton_manager_cfg import CoupledMJWarpVBDSolverCfg
from isaaclab_newton.physics import NewtonManager, VBDSolverCfg
from isaaclab_newton.physics.newton_collision_cfg import NewtonCollisionPipelineCfg
from isaaclab_newton.physics.newton_manager_cfg import NewtonCfg
from isaaclab_newton.physics.vbd_manager import NewtonVBDManager

from dexbench_arena.assets.newton_conveyor import BeltTreadmill

from . import frame_truncation
from .canonical import loader
from .parking import stop_particle_collision
from .rigid_substeps import interpolate_body_q, rigid_arrays
from .surface_drive import SoftContactDrive, drive_surfaces
from .two_way_reaction import body_soft_contact_reaction, contact_filter_data, previous_particle_positions
from .polybag_object import registered_polybags, sync_polybag_renders
from .polybag_object_cfg import PolybagObjectCfg
from .template import load_cfg_template

logger = logging.getLogger(__name__)


@configclass
class PolybagVBDSolverCfg(VBDSolverCfg):
    """VBD for the polybag. Self-contact radius and margin, contact buffers and exclusions are the
    asset's (read by the canonical loader); the fields here are the scene's choices."""

    class_type: type | str = "{DIR}.physics:NewtonPolybagVBDManager"

    iterations: int = 6
    """VBD iterations per substep. The asset was verified at 128 substeps x 80 iterations; the cotton bag
    bursts below 16 substeps."""

    integrate_with_external_rigid_solver: bool = True
    """The robot is integrated by MJWarp; VBD reads its bodies as kinematic colliders."""

    particle_collision_detection_interval: int = 0
    """Self-contact detection before and right after each substep's initialization (Newton's default)."""

    frame_truncation: Literal["init", "all"] | None = "init"
    """Self-contact truncation in each bag's own frame instead of SolverVBD's, which caps a bag's travel
    at ``margin x 0.85 x 0.5 x 60 x substeps`` (0.21 m/s at 8 substeps). "init": only the truncation of
    each substep's initial guess; "all": every call; None: SolverVBD's own."""

    initial_guess: Literal["inertia", "contact"] = "inertia"
    """Where each substep's iterate starts (``vbd_init.py``). inertia: SolverVBD's own x + (v + g dt) dt, which a
    supported bag never climbs back from in a few iterations (its contact forces then exceed its weight: the belt holds
    a resting bag up with more than its weight, a held bag pulls the hand down); contact: a bag the colliders press
    with a quarter of its weight or more, or one lying on another bag, starts at x + v dt, any other as inertia. The
    solve's target and every force law are unchanged."""

    film_hessian: Literal["newton", "psd"] = "newton"
    """The film springs' Hessian in VBD's per-vertex step (``film_hessian.py``). newton: Newton's own, indefinite while
    a spring is compressed; psd: the same force, its transverse term clamped at >= 0."""

    tet_poisson: float | None = None
    """Every bag tet's Poisson ratio, set at its own shear modulus (lambda = 2 mu nu / (1 - 2 nu)); None: the asset's."""

    tet_damping: float | None = None
    """Every bag tet's damping [Pa s]; None: the asset's."""

    dat_relaxation: float | None = None
    """SolverVBD's conservative-bound relaxation (gamma in D = query radius x gamma / 2); None keeps 0.85."""

    rigid_soft_log_barrier: bool = False
    """SolverVBD's ``rigid_soft_contact_use_log_barrier`` (PolybagObjectCfg.contact_log_barrier); the two-way reaction
    reads the same switch off the solver."""


class NewtonPolybagVBDManager(NewtonVBDManager):
    """Creates the polybag scene's SolverVBD; used as the coupled manager's soft solver."""

    @classmethod
    def _create_solver(cls, model, solver_cfg: PolybagVBDSolverCfg):
        entries = registered_polybags()
        if not entries:
            raise RuntimeError("PolybagVBDSolverCfg is set but the scene has no PolybagObject")
        templates = {id(e.template): e.template for e in entries}
        if len({tuple(sorted(t.solver_kwargs.items())) for t in templates.values()}) != 1:
            raise ValueError("the scene's polybags ask for different SolverVBD settings")
        if len({(tuple(sorted(t.contact.items())), tuple(sorted(t.fixture.items())), t.fixture_on_bodies,
                 t.collider_mu_by_path) for t in templates.values()}) != 1:
            raise ValueError("the scene's polybags ask for different contact numbers")
        template = entries[0].template
        apply_contact_material(model, template)
        poisson, damping = getattr(solver_cfg, "tet_poisson", None), getattr(solver_cfg, "tet_damping", None)
        if poisson is not None or damping is not None:
            count = apply_tet_material(model, entries, poisson, damping)
            print(f"[dexbench] polybag: {count} bag tets at Poisson ratio "
                  f"{'as authored' if poisson is None else f'{poisson:g}'}, damping "
                  f"{'as authored' if damping is None else f'{damping:g} Pa s'}", flush=True)
        reaches = [e.cfg.collider_reach_m for e in entries]
        if all(r is not None for r in reaches):
            limit_particle_colliders(model, entries, max(reaches))
        kwargs = dict(template.solver_kwargs)
        vt, ee = {}, {}
        for entry in entries:
            a, b = entry.exclusion_maps()
            vt.update(a)
            ee.update(b)
        if vt:
            kwargs["particle_external_vertex_contact_filtering_map"] = vt
        if ee:
            kwargs["particle_external_edge_contact_filtering_map"] = ee
        kwargs.update(iterations=int(solver_cfg.iterations),
                      integrate_with_external_rigid_solver=bool(solver_cfg.integrate_with_external_rigid_solver),
                      particle_collision_detection_interval=int(solver_cfg.particle_collision_detection_interval))
        if solver_cfg.rigid_soft_log_barrier:
            if "rigid_soft_contact_use_log_barrier" not in inspect.signature(SolverVBD.__init__).parameters:
                raise ValueError("contact_log_barrier: this Newton's SolverVBD has no rigid_soft_contact_use_log_barrier")
            kwargs["rigid_soft_contact_use_log_barrier"] = True
        if solver_cfg.dat_relaxation is not None:
            kwargs["particle_conservative_bound_relaxation"] = float(solver_cfg.dat_relaxation)
        solver_class = SolverVBD
        initial_guess = getattr(solver_cfg, "initial_guess", "inertia")
        if initial_guess != "inertia":
            from . import vbd_init

            if initial_guess not in vbd_init.INITIAL_GUESSES:
                raise ValueError(f"initial_guess {initial_guess!r}: one of {vbd_init.INITIAL_GUESSES}")
            solver_class = vbd_init.ContactInitSolverVBD
        contact_init = solver_class is not SolverVBD
        film = getattr(solver_cfg, "film_hessian", "newton")
        if film != "newton":
            from . import film_hessian

            solver_class = film_hessian.with_film_hessian(solver_class, film)
        solver = solver_class(model, **kwargs)
        if contact_init:
            solver.set_bag_groups([r for e in entries for r in e.particle_ranges()])
            print(f"[dexbench] polybag: SolverVBD initial guess {initial_guess} (vbd_init.py)", flush=True)
        if film != "newton":
            film_hessian.load(solver.device)
            print(f"[dexbench] polybag: SolverVBD film springs' Hessian {film} (film_hessian.py)", flush=True)
        if solver_cfg.frame_truncation and kwargs.get("particle_enable_self_contact"):
            groups = [r for e in entries for r in e.particle_ranges()]
            frame_truncation.install(solver, groups, scope=solver_cfg.frame_truncation)
        logger.info("polybag SolverVBD: %d bag(s), %s", sum(len(e.particle_offsets) for e in entries),
                    {k: v for k, v in kwargs.items() if "filtering_map" not in k})
        return solver


def apply_contact_material(model, template) -> None:
    """The asset's contact numbers on the model, and its fixture material on every scene collider."""
    for name, value in template.contact.items():
        setattr(model, name, value)
    if model.shape_count == 0:
        return
    body = model.shape_body.numpy()
    for name, key in (("shape_material_ke", "shape_material_ke"), ("shape_material_kd", "shape_material_kd")):
        array = getattr(model, name)
        values = array.numpy()
        values[:] = template.fixture[key]
        array.assign(values)
    mu = model.shape_material_mu.numpy()
    if getattr(template, "fixture_on_bodies", False):
        mu[:] = template.fixture["shape_material_mu"]
    else:
        mu[body < 0] = template.fixture["shape_material_mu"]
    by_path = getattr(template, "collider_mu_by_path", ())
    if by_path:
        labels = [str(label) for label in model.shape_label]
        unmatched = set(range(len(by_path)))
        for shape, label in enumerate(labels):
            for index, (pattern, value) in enumerate(by_path):
                if re.search(pattern, label):
                    mu[shape] = value
                    unmatched.discard(index)
                    break
        if unmatched:
            raise ValueError(f"collider_mu_by_path: {[by_path[i][0] for i in sorted(unmatched)]} match no collider")
    model.shape_material_mu.assign(mu)


def apply_tet_material(model, entries, poisson: float | None, damping: float | None) -> int:
    """Every bag tet's Poisson ratio (at its own shear modulus: lambda = 2 mu nu / (1 - 2 nu)) and damping [Pa s] in
    ``model.tet_materials`` (mu, lambda, damping per tet), in place; None leaves that column as the asset has it.
    -> the number of bag tets. ValueError for a ratio outside [0, 0.5) or a negative damping."""
    if poisson is not None and not 0.0 <= poisson < 0.5:
        raise ValueError(f"tet Poisson ratio {poisson!r}: 0 <= nu < 0.5")
    if damping is not None and not damping >= 0.0:
        raise ValueError(f"tet damping {damping!r}: >= 0 Pa s")
    if not model.tet_count:
        return 0
    bag = np.zeros(model.particle_count, dtype=bool)
    for entry in entries:
        for r in entry.particle_ranges():
            bag[np.asarray(r, dtype=np.int64)] = True
    tets = bag[model.tet_indices.numpy().reshape(-1, 4)[:, 0]]
    materials = model.tet_materials.numpy()
    if poisson is not None:
        materials[tets, 1] = 2.0 * materials[tets, 0] * poisson / (1.0 - 2.0 * poisson)
    if damping is not None:
        materials[tets, 2] = damping
    model.tet_materials.assign(materials)
    return int(tets.sum())


def check_two_way_contact(template, coupling_mode: str) -> None:
    """ValueError for two-way coupling with the bag's and the colliders' contact stiffness or damping unequal."""
    if coupling_mode != "two_way":
        return
    for side in ("ke", "kd"):
        bag, collider = template.contact[f"soft_contact_{side}"], template.fixture[f"shape_material_{side}"]
        if bag != collider:
            raise ValueError(f"two-way coupling needs equal contact {side} on both sides: the bag's {bag:g}, the "
                             f"colliders' {collider:g} ({template.usd_path.rsplit('/', 1)[-1]})")


def apply_polybag_physics(cfg) -> bool:
    """Put a composed env cfg that holds a polybag on MJWarp + VBD at the bag's stepping; -> whether it did."""

    bags = [v for v in vars(cfg.scene).values() if isinstance(v, PolybagObjectCfg)]
    physics = cfg.sim.physics
    if not bags or not isinstance(physics, NewtonCfg):
        return False
    first = bags[0]
    keys = ("vbd_substeps", "mjwarp_substeps", "vbd_iterations", "frame_truncation", "full_surface_contact", "sdf_voxel_m",
            "coupling_mode", "contact_log_barrier")
    if any(tuple(getattr(b, k) for k in keys) != tuple(getattr(first, k) for k in keys) for b in bags):
        raise ValueError("the scene's polybags ask for different VBD stepping")
    if isinstance(physics.solver_cfg, CoupledMJWarpVBDSolverCfg):
        rigid = physics.solver_cfg.rigid_solver_cfg
    else:
        rigid = physics.solver_cfg
    if not getattr(rigid, "use_mujoco_contacts", False):
        raise ValueError("the polybag's coupled solver needs MJWarp's own contacts (use_mujoco_contacts=True)")
    if first.vbd_substeps % first.mjwarp_substeps:
        raise ValueError(f"mjwarp_substeps {first.mjwarp_substeps} must divide vbd_substeps {first.vbd_substeps}")
    templates = [load_cfg_template(b) for b in bags]
    for template in templates:
        check_two_way_contact(template, first.coupling_mode)
    margin = max(t.contact_margin(first.vbd_substeps, 1.0 / cfg.sim.dt, b.fall_height_m) for t, b in zip(templates, bags))
    solver = PolybagCoupledSolverCfg(
        rigid_solver_cfg=rigid,
        soft_solver_cfg=PolybagVBDSolverCfg(iterations=int(first.vbd_iterations), frame_truncation=first.frame_truncation,
                                            rigid_soft_log_barrier=bool(first.contact_log_barrier)),
        rigid_every=int(first.vbd_substeps // first.mjwarp_substeps),
        coupling_mode=first.coupling_mode,
    )
    num_envs = int(getattr(cfg.scene, "num_envs", 1) or 1)
    full_surface = bool(first.full_surface_contact)
    soft_contact_max = sum(soft_contact_elements(t, full_surface) * int(b.soft_contacts_per_particle)
                           for t, b in zip(templates, bags)) * num_envs
    cfg.sim.physics = physics.replace(class_type=None, solver_cfg=solver, num_substeps=int(first.vbd_substeps),
                                      collision_cfg=NewtonCollisionPipelineCfg(
                                          soft_contact_margin=margin, soft_contact_max=soft_contact_max,
                                          enable_rigid_soft_full_surface_contact=full_surface))
    print(f"[dexbench] polybag: MJWarp + VBD ({first.coupling_mode.replace('_', '-')}), {first.vbd_substeps} substeps x {first.vbd_iterations} "
          f"VBD iterations, MJWarp {first.mjwarp_substeps} substeps, soft-contact band {margin * 1e3:.1f} mm, "
          f"bag-frame truncation {first.frame_truncation or 'off'}, full-surface contact "
          f"{f'on (collider SDF voxel {first.sdf_voxel_m * 1e3:.1f} mm)' if full_surface else 'off'}, "
          f"soft-contact buffer {soft_contact_max}", flush=True)
    for t in {id(t): t for t in templates}.values():
        size = "" if t.size_scale == (1.0, 1.0, 1.0) else " at " + " x ".join(f"{s:g}" for s in t.size_scale) + " (x, y, z)"
        print(f"[dexbench] polybag: {t.usd_path.rsplit('/', 1)[-1]}{size}: material "
              f"{ {k.rsplit('.', 1)[-1]: v for k, v in t.material.items()} or 'as authored'}, "
              f"bag contact {t.contact}, collider contact {t.fixture}"
              + (f", collider friction by path {dict(t.collider_mu_by_path)}" if t.collider_mu_by_path else ""), flush=True)
    return True


def soft_contact_elements(template, full_surface: bool) -> int:
    """What can touch a collider: the bag's particles, and with full-surface contact its surface edges and faces."""
    n = template.particle_count
    if full_surface:
        n += template.builder.tri_count + len(template.builder.edge_indices)
    return n


def bounding_sphere_centre(transform, scale, kind: int, source) -> np.ndarray:
    """World centre of the sphere of radius ``shape_collision_radius`` that bounds a shape."""
    if kind not in (int(newton.GeoType.MESH), int(newton.GeoType.CONVEX_MESH)) or source is None:
        return np.asarray(wp.transform_get_translation(transform), dtype=np.float64)
    vertices = np.asarray(source.vertices, dtype=np.float64) * np.asarray(scale, dtype=np.float64)
    middle = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    return np.asarray(wp.transform_point(transform, wp.vec3(*middle)), dtype=np.float64)


def prepare_full_surface_contact(builder, entries) -> tuple[int, int]:
    """Before ``finalize``: the colliders full-surface contact samples, each with an SDF; -> (cleared, provisioned)."""
    reaches = [e.cfg.collider_reach_m for e in entries]
    reach_m = max(reaches) if all(r is not None for r in reaches) else None
    voxel = min(float(e.cfg.sdf_voxel_m) for e in entries)
    q = np.asarray(builder.particle_q, dtype=np.float64)
    boxes = [(q[r].min(axis=0), q[r].max(axis=0)) for e in entries for r in e.particle_ranges()]
    collide = int(newton.ShapeFlags.COLLIDE_PARTICLES)
    shapes_flag = int(newton.ShapeFlags.COLLIDE_SHAPES)
    meshes = (int(newton.GeoType.MESH), int(newton.GeoType.CONVEX_MESH))
    cleared = provisioned = 0
    no_shape_collision = []
    for s in range(builder.shape_count):
        flags = int(builder.shape_flags[s])
        if not flags & collide:
            continue
        kind = int(builder.shape_type[s])
        if builder.shape_body[s] < 0 and reach_m is not None and kind != int(newton.GeoType.PLANE):
            centre = bounding_sphere_centre(builder.shape_transform[s], builder.shape_scale[s], kind,
                                            builder.shape_source[s])
            gap = min(float(np.linalg.norm(np.maximum(0.0, np.maximum(lo - centre, centre - hi)))) for lo, hi in boxes)
            if gap - float(builder.shape_collision_radius[s]) > reach_m:
                builder.shape_flags[s] = flags & ~collide
                cleared += 1
                continue
        if kind in meshes:
            if not flags & shapes_flag:
                no_shape_collision.append(str(builder.shape_label[s]))
            builder.shape_sdf_target_voxel_size[s] = voxel
            builder.shape_sdf_max_resolution[s] = None
            provisioned += 1
    if no_shape_collision:
        raise ValueError(f"full-surface contact: these particle colliders do not collide with shapes and get no SDF: "
                         f"{no_shape_collision[:8]}")
    print(f"[dexbench] polybag: full-surface contact: {cleared} static collider(s) out of reach stop colliding with "
          f"the bag; {provisioned} mesh collider(s) get a {voxel * 1e3:.1f} mm SDF", flush=True)
    return cleared, provisioned


def limit_particle_colliders(model, entries, reach_m: float) -> int:
    """Clear particle collision on the static colliders no bag can reach; -> how many were cleared."""
    q = model.particle_q.numpy()
    boxes = [(q[r].min(axis=0), q[r].max(axis=0)) for e in entries for r in e.particle_ranges()]
    flags = model.shape_flags.numpy()
    body = model.shape_body.numpy()
    transforms = model.shape_transform.numpy()
    scales = model.shape_scale.numpy()
    radius = model.shape_collision_radius.numpy()
    kind = model.shape_type.numpy()
    collide = int(newton.ShapeFlags.COLLIDE_PARTICLES)
    candidates = np.flatnonzero((body < 0) & ((flags & collide) != 0))
    cleared = 0
    for s in candidates:
        if int(kind[s]) == int(newton.GeoType.PLANE):
            continue
        transform = wp.transform(wp.vec3(*transforms[s, :3]), wp.quat(*transforms[s, 3:7]))
        centre = bounding_sphere_centre(transform, scales[s], int(kind[s]), model.shape_source[s])
        gap = min(float(np.linalg.norm(np.maximum(0.0, np.maximum(lo - centre, centre - hi)))) for lo, hi in boxes)
        if gap - float(radius[s]) > reach_m:
            flags[s] &= ~collide
            cleared += 1
    if cleared:
        model.shape_flags.assign(flags)
    print(f"[dexbench] polybag: {cleared} of {len(candidates)} static colliders are farther than {reach_m:.2f} m "
          f"from the bag and do not collide with its particles", flush=True)
    return cleared


def _build_belt_treadmill(model, state, velocities: dict[str, tuple[float, float, float]]):
    """The treadmill for ``rigid_belt_velocities``: every shape colliding with shapes whose label matches a key,
    at that velocity; None when there are none. Before graph capture (it allocates)."""
    if not velocities:
        return None
    labels = list(model.shape_label)
    flags = model.shape_flags.numpy()
    collide = int(newton.ShapeFlags.COLLIDE_SHAPES)
    treadmill = BeltTreadmill(model)
    driven = []
    for pattern, velocity in velocities.items():
        shapes = [s for s, label in enumerate(labels) if re.search(pattern, str(label)) and int(flags[s]) & collide]
        if not shapes:
            raise ValueError(f"rigid belt velocity: no shape colliding with shapes matches {pattern!r}")
        for s in shapes:
            treadmill.add_belt(s, state, velocity=wp.vec3(*(float(v) for v in velocity)))
            driven.append(str(labels[s]))
    treadmill.finalize()
    print(f"[dexbench] polybag: belt treadmill {velocities} on {driven}, for MJWarp's rigid bodies", flush=True)
    return treadmill


@configclass
class PolybagCoupledSolverCfg(CoupledMJWarpVBDSolverCfg):
    """MJWarp (robot, its own contacts) + VBD (polybag), one-way by default: the hand drives the bag, the bag
    does not push back on the hand (``coupling_mode``, from the bags' ``PolybagObjectCfg``). MJWarp's options
    are the environment's."""

    class_type: type | str = "{DIR}.physics:NewtonPolybagCoupledManager"

    soft_solver_cfg: PolybagVBDSolverCfg = PolybagVBDSolverCfg()

    coupling_mode: Literal["one_way", "two_way"] = "one_way"

    rigid_every: int = 1
    """MJWarp steps once every this many substeps, over that many substeps' time; the bag's VBD steps in
    every substep and reads the robot's body poses interpolated in between. Must divide the substep
    count; one-way coupling only."""

    surface_velocities: dict[str, tuple[float, float, float]] = {}
    """Surface velocity [m/s, world frame] of the mesh colliders whose shape label matches each key (a regex),
    for the bag's particles: the collider stays put and its surface carries the bag by friction, as a
    conveyor belt does (``surface_drive.py``). MJWarp's rigid bodies do not see it."""

    rigid_belt_velocities: dict[str, tuple[float, float, float]] = {}
    """Velocity [m/s, world frame] of the belts whose MJWarp collider's shape label matches each key (a regex),
    for the rigid bodies: each such collider's kinematic body has its pose pinned and this velocity prescribed
    at every MJWarp step (``dexbench_arena.assets.newton_conveyor.BeltTreadmill``, Newton's
    ``example_basic_conveyor`` made a treadmill), so MuJoCo's contact friction carries what rests on it. Give it
    the speed of the same belt's ``surface_velocities``: the bag still rides those, and MJWarp leaves a
    kinematic body's ``body_q`` where the treadmill put it, so the bag does not see the belt move twice."""

    texture_scrolls: dict[str, float] = {}
    """``UsdTransform2d`` shaders whose prim path matches each key (a regex) slide their texture along u at the
    value [UV units per second of simulated time], written every rendered frame: the look of a moving belt.
    The delivered conveyor scrolls its texture from an OmniGraph on the timeline's playback tick, which
    never fires under Newton (its manager does not play the timeline)."""

    particle_collision_off: tuple[str, ...] = ()
    """Shapes whose label matches any of these regexes stop colliding with the bags' particles, edges and faces;
    MJWarp's rigid contacts keep them (``parking.stop_particle_collision``). For the surface a bag waits on while it
    is out of the simulation: an inactive bag is not moved, so it needs no support, and full-surface contact
    would otherwise sample its edges and faces against that surface every substep."""

    rtf_report_interval_s: float | None = 10.0
    """Print the real-time factor (simulated over wall-clock time, everything in the loop included: physics,
    rendering, teleop) every this many wall-clock seconds, with the most soft contacts a collision pass found in
    that time against the buffer's capacity; None: never. Time spent paused counts as wall time."""


@wp.kernel
def soft_contact_peak(count: wp.array(dtype=wp.int32), peak: wp.array(dtype=wp.int32)):
    """``peak`` = the most soft contacts a collision pass has reported since it was zeroed. Newton reports the
    count it needed when the buffer overflows, so a peak over ``soft_contact_max`` is an overflow."""
    wp.atomic_max(peak, 0, count[0])


class NewtonPolybagCoupledManager(NewtonCoupledMJWarpVBDManager):
    """The coupled manager, with the polybag's colouring, a soft-contact-only collision pipeline, and
    MJWarp stepped every ``rigid_every`` substeps."""

    _rigid_every: int = 1
    _substep: int = 0
    _rigid_start: wp.array | None = None
    _rigid_end: dict | None = None
    _texture_scroll_rates: dict[str, float] = {}
    _texture_scrolls: list | None = None
    _rtf_interval_s: float | None = None
    _rtf_mark: tuple[float, float, int] | None = None
    _contact_peak: wp.array | None = None
    _contact_drive: SoftContactDrive | None = None
    _belt_treadmill = None
    _particle_q_prev = None

    @classmethod
    def _build_solver(cls, model, solver_cfg: PolybagCoupledSolverCfg) -> None:
        super()._build_solver(model, solver_cfg)
        every = int(getattr(solver_cfg, "rigid_every", 1))
        if every < 1 or cls._num_substeps % every:
            raise ValueError(f"rigid_every {every} must divide the {cls._num_substeps} substeps")
        if every > 1 and cls._coupling_mode != "one_way":
            raise ValueError("rigid_every > 1 is implemented for one-way coupling only")
        NewtonPolybagCoupledManager._rigid_every = every
        NewtonPolybagCoupledManager._substep = 0
        NewtonPolybagCoupledManager._rtf_interval_s = getattr(solver_cfg, "rtf_report_interval_s", None)
        NewtonPolybagCoupledManager._rtf_mark = None
        NewtonPolybagCoupledManager._contact_peak = wp.zeros(1, dtype=wp.int32) if cls._rtf_interval_s else None
        NewtonPolybagCoupledManager._texture_scroll_rates = dict(getattr(solver_cfg, "texture_scrolls", None) or {})
        NewtonPolybagCoupledManager._particle_q_prev = (
            wp.zeros_like(model.particle_q) if cls._coupling_mode == "two_way" and model.particle_count else None)
        NewtonPolybagCoupledManager._texture_scrolls = None
        surfaces = dict(getattr(solver_cfg, "surface_velocities", None) or {})
        NewtonPolybagCoupledManager._contact_drive = None
        if surfaces:
            body_frame: dict[int, np.ndarray] = {}
            driven = drive_surfaces(model, cls._state_0.body_q.numpy(), surfaces, body_frame)
            NewtonPolybagCoupledManager._contact_drive = SoftContactDrive(model, body_frame)
            print(f"[dexbench] polybag: surface velocity {surfaces} on {driven}, written into each of their contacts",
                  flush=True)
        NewtonPolybagCoupledManager._belt_treadmill = _build_belt_treadmill(
            model, cls._state_0, dict(getattr(solver_cfg, "rigid_belt_velocities", None) or {}))
        if every > 1:
            NewtonPolybagCoupledManager._rigid_start = wp.empty_like(cls._state_0.body_q)
            NewtonPolybagCoupledManager._rigid_end = {k: wp.empty_like(v) for k, v in rigid_arrays(cls._state_0).items()}
            print(f"[dexbench] polybag: MJWarp every {every} substeps, carrying {sorted(cls._rigid_end)} between",
                  flush=True)

    @classmethod
    def _solver_specific_clear(cls) -> None:
        super()._solver_specific_clear()
        NewtonPolybagCoupledManager._rigid_every = 1
        NewtonPolybagCoupledManager._substep = 0
        NewtonPolybagCoupledManager._rigid_start = None
        NewtonPolybagCoupledManager._rigid_end = None
        NewtonPolybagCoupledManager._texture_scroll_rates = {}
        NewtonPolybagCoupledManager._texture_scrolls = None
        NewtonPolybagCoupledManager._rtf_interval_s = None
        NewtonPolybagCoupledManager._rtf_mark = None
        NewtonPolybagCoupledManager._contact_peak = None
        NewtonPolybagCoupledManager._contact_drive = None
        NewtonPolybagCoupledManager._belt_treadmill = None
        NewtonPolybagCoupledManager._particle_q_prev = None

    @classmethod
    def _rigid_step(cls, state_in, state_out, control, dt: float) -> None:
        """MJWarp's step, against belts already pinned and moving (``rigid_belt_velocities``)."""
        if cls._belt_treadmill is not None:
            cls._belt_treadmill.apply(state_in)
        super()._rigid_step(state_in, state_out, control, dt)

    @classmethod
    def step(cls) -> None:
        """A physics step, and every ``rtf_report_interval_s`` of wall-clock time a line with the real-time factor."""
        super().step()
        interval = cls._rtf_interval_s
        if not interval:
            return
        now, sim_t = time.perf_counter(), cls.get_simulation_time()
        mark = cls._rtf_mark
        if mark is None or sim_t < mark[1]:
            NewtonPolybagCoupledManager._rtf_mark = (now, sim_t, 0)
            return
        steps = mark[2] + 1
        if now - mark[0] < interval:
            NewtonPolybagCoupledManager._rtf_mark = (mark[0], mark[1], steps)
            return
        wall = now - mark[0]
        contacts = ""
        if cls._contact_peak is not None and cls._contacts is not None:
            peak, capacity = int(cls._contact_peak.numpy()[0]), int(cls._contacts.soft_contact_max)
            cls._contact_peak.zero_()
            contacts = f"; soft contacts at most {peak} of {capacity}{' (OVERFLOW)' if peak > capacity else ''}"
        print(f"[dexbench] polybag: RTF {(sim_t - mark[1]) / wall:.3f} over the last {wall:.1f} s "
              f"({steps} physics steps, {wall / steps * 1e3:.1f} ms each, everything included){contacts}", flush=True)
        NewtonPolybagCoupledManager._rtf_mark = (now, sim_t, 0)

    @classmethod
    def pre_render(cls) -> None:
        """Before each rendered frame: draw the tet bags' textured meshes when their particles moved, and slide
        the ``texture_scrolls`` shaders to where simulated time puts them."""
        moved = cls._particles_dirty
        super().pre_render()
        if moved:
            sync_polybag_renders()
        if not cls._texture_scroll_rates:
            return
        if cls._texture_scrolls is None:
            NewtonPolybagCoupledManager._texture_scrolls = _resolve_texture_scrolls(cls._texture_scroll_rates)
        t = cls.get_simulation_time()
        for attr, rate, v in cls._texture_scrolls:
            attr.Set(Gf.Vec2f(math.fmod(rate * t, 1.0), v))

    @classmethod
    def _step_one_way(cls, state_in, state_out, control, dt: float) -> None:
        """One substep; MJWarp runs in the first of every ``rigid_every``, over all of their time."""
        every = cls._rigid_every
        state_out.clear_forces()
        cls._collide(state_in)
        if every == 1:
            cls._rigid_step(state_in, state_out, control, dt)
            cls._soft_solver.step(state_in, state_out, control, cls._contacts, dt)
            return
        phase = cls._substep % every
        NewtonPolybagCoupledManager._substep = cls._substep + 1
        end = cls._rigid_end
        if phase == 0:
            wp.copy(cls._rigid_start, state_in.body_q)
            cls._rigid_step(state_in, state_out, control, dt * every)
            for name, array in rigid_arrays(state_out).items():
                wp.copy(end[name], array)
        else:
            for name, array in rigid_arrays(state_out).items():
                wp.copy(array, end[name])
        if phase + 1 < every:
            wp.launch(interpolate_body_q, dim=cls._rigid_start.shape[0],
                      inputs=[cls._rigid_start, end["body_q"], float(phase + 1) / every], outputs=[state_out.body_q])
        cls._soft_solver.step(state_in, state_out, control, cls._contacts, dt)

    @classmethod
    def _step_two_way(cls, state_in, state_out, control, dt: float) -> None:
        """One substep with the bag's contact reactions on the bodies: the upstream two-way step, its
        contacts detected through ``_collide`` so the driven belt's speed is in every one of them, as in the
        one-way step, and its reactions from ``two_way_reaction``, which reads the edge and face contacts of
        full-surface contact as well as the particles'. The upstream step calls the collision pipeline
        directly and its reaction kernel reads every contact as a particle's."""
        state_out.clear_forces()
        cls._collide(state_in)
        if state_in.body_f is not None:
            cls._apply_soft_contact_reactions(state_in, state_out, dt)
        cls._rigid_step(state_in, state_out, control, dt)
        cls._soft_solver.step(state_in, state_out, control, cls._contacts, dt)

    @classmethod
    def _apply_soft_contact_reactions(cls, state, state_prev, dt: float) -> None:
        """The reaction of every soft contact VBD applies a force for into ``state.body_f``; ``state_prev`` holds
        the bodies' poses a substep ago, for the friction's relative motion (the upstream ``_apply_reactions``'s
        arguments)."""
        model, contacts = cls._model, cls._contacts
        if contacts is None or cls._particle_q_prev is None or int(contacts.soft_contact_indices.shape[0]) == 0:
            return
        wp.launch(previous_particle_positions, dim=model.particle_count,
                  inputs=[state.particle_q, state.particle_qd, float(dt)], outputs=[cls._particle_q_prev])
        wp.launch(body_soft_contact_reaction, dim=int(contacts.soft_contact_indices.shape[0]),
                  inputs=[contacts.soft_contact_count, contacts.soft_contact_indices, contacts.soft_contact_barycentric,
                          contacts.soft_contact_shape, contacts.soft_contact_body_pos, contacts.soft_contact_body_vel,
                          contacts.soft_contact_normal, state.particle_q, cls._particle_q_prev, model.particle_radius,
                          state.body_q, state_prev.body_q, state.body_qd, model.body_com, model.shape_body,
                          model.shape_material_mu, model.shape_margin, float(model.soft_contact_ke),
                          float(model.soft_contact_kd), float(model.soft_contact_mu),
                          float(cls._soft_solver.friction_epsilon), float(dt),
                          bool(getattr(cls._soft_solver, "rigid_soft_contact_use_log_barrier", False)),
                          contact_filter_data(model, contacts, state.particle_q, state.body_q)],
                  outputs=[state.body_f])

    @classmethod
    def _collide(cls, state_in) -> None:
        """The bag's contacts at the substep's start, the driven surfaces' speed on every one of theirs."""
        cls._collision_pipeline.collide(state_in, cls._contacts)
        if cls._contact_peak is not None:
            wp.launch(soft_contact_peak, dim=1, inputs=[cls._contacts.soft_contact_count, cls._contact_peak])
        if cls._contact_drive is not None:
            cls._contact_drive(cls._contacts)

    @classmethod
    def _initialize_contacts(cls) -> None:
        """Build Newton's collision pipeline with no rigid shape pairs: only the bag's particle contacts."""
        if cls._needs_collision_pipeline and cls._collision_pipeline is None:
            args = cls._collision_cfg.to_pipeline_args() if cls._collision_cfg is not None else {}
            args.update(broad_phase="explicit",
                        shape_pairs_filtered=wp.zeros(0, dtype=wp.vec2i, device=PhysicsManager._device),
                        deterministic=cls._deterministic_mode != wp.DeterministicMode.NOT_GUARANTEED)
            NewtonManager._collision_pipeline = CollisionPipeline(cls._model, **args)
        super()._initialize_contacts()

    @classmethod
    def instantiate_builder_from_stage(cls) -> None:
        super(NewtonVBDManager, cls).instantiate_builder_from_stage()
        if cls._builder is None:
            raise RuntimeError("Newton stage import did not create a builder.")
        colour_with_seals(cls._builder)

    @classmethod
    def start_simulation(cls) -> None:
        if cls._builder is not None:
            colour_with_seals(cls._builder)
        super(NewtonVBDManager, cls).start_simulation()

    @classmethod
    def _prepare_builder_for_finalize(cls, builder) -> None:
        """Right before ``finalize``, the bags in it: ``particle_collision_off`` stop colliding with them, and the
        colliders full-surface contact samples get their SDFs."""
        super()._prepare_builder_for_finalize(builder)
        off = tuple(getattr(PhysicsManager._cfg.solver_cfg, "particle_collision_off", None) or ())
        if off:
            changed = stop_particle_collision(builder, off)
            print(f"[dexbench] polybag: {changed} stop colliding with the bags (particle_collision_off)", flush=True)
        entries = registered_polybags()
        if entries and any(e.cfg.full_surface_contact for e in entries):
            prepare_full_surface_contact(builder, entries)


def _resolve_texture_scrolls(rates: dict[str, float]) -> list:
    """``(inputs:translation attribute, rate, its v)`` for every stage prim whose path matches a key of ``rates``."""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    found = []
    for pattern, rate in rates.items():
        matched = 0
        for prim in stage.Traverse():
            if not re.search(pattern, str(prim.GetPath())):
                continue
            attr = prim.GetAttribute("inputs:translation")
            if not attr:
                continue
            current = attr.Get()
            found.append((attr, float(rate), float(current[1]) if current is not None else 0.0))
            matched += 1
        if not matched:
            raise ValueError(f"texture scroll: no shader with inputs:translation matches {pattern!r}")
        print(f"[dexbench] polybag: texture scroll {pattern!r} at {rate:+.3f} UV/s on {matched} shader(s)", flush=True)
    return found


def colour_with_seals(builder) -> None:
    """VBD colouring with the bending edges and every polybag's seal springs in the graph."""
    springs = [e.seal_springs() for e in registered_polybags()]
    springs = np.concatenate(springs) if springs else np.zeros((0, 2), dtype=np.int64)
    if not len(springs) and not any(ke or kd for ke, kd in builder.edge_bending_properties):
        builder.color()
        return
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        loader.colour_for_vbd(builder, springs)
    logger.info("polybag colouring: %s", log.getvalue().strip())
