# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""What a polybag meets and how the contact acts: a collider's friction against the bag, the static colliders it can
reach, a box-shaped mesh met as an analytic box, a parked bag out of the simulation, a conveyor belt's surface
velocity, and the reaction two-way coupling gives a body from the bag's contacts."""

import math
import types

import numpy as np
import pytest

wp = pytest.importorskip("warp")
newton = pytest.importorskip("newton")

from dexbench_arena import ASSETS_DIR  # noqa: E402
from dexbench_arena.assets.polybag.parking import set_particles_active, stop_particle_collision  # noqa: E402
from dexbench_arena.assets.polybag.particle_boxes import add_particle_boxes  # noqa: E402
from dexbench_arena.assets.polybag.physics import apply_contact_material, limit_particle_colliders  # noqa: E402
from dexbench_arena.assets.polybag.surface_drive import drive_surfaces  # noqa: E402
from dexbench_arena.assets.polybag.template import load_template  # noqa: E402
from dexbench_arena.assets.polybag.two_way_reaction import (  # noqa: E402
    HAS_CONTACT_FILTER,
    _SoftContactFilterData,
    body_soft_contact_reaction,
    contact_filter_data,
)



def _model():
    builder = newton.ModelBuilder()
    builder.add_cloth_grid(pos=wp.vec3(0.0, 0.0, 1.0), rot=wp.quat_identity(), vel=wp.vec3(0.0, 0.0, 0.0),
                           dim_x=2, dim_y=2, cell_x=0.1, cell_y=0.1, mass=0.01)
    builder.add_shape_box(-1, hx=0.5, hy=0.5, hz=0.01, label="/World/envs/env_0/cell/Plate/WorkPlate/Piece_00")
    builder.add_shape_box(-1, hx=0.5, hy=0.1, hz=0.01, label="/World/envs/env_0/cell/Belts/HB2_Belt/TopRun")
    hand = builder.add_body(label="/World/envs/env_0/Robot/left_wrist")
    builder.add_shape_sphere(hand, radius=0.02, label="/World/envs/env_0/Robot/left_wrist/collisions/geom_0")
    return builder.finalize(device="cpu")


def _template(by_path):
    return types.SimpleNamespace(
        contact={"soft_contact_mu": 0.4},
        fixture={"shape_material_ke": 3000.0, "shape_material_kd": 1.0, "shape_material_mu": 0.8},
        fixture_on_bodies=True, collider_mu_by_path=by_path)


def test_a_collider_matching_a_pattern_takes_its_friction_and_the_rest_the_fixtures():
    model = _model()
    apply_contact_material(model, _template(((r"/Robot/", 1.2), (r"/Plate/WorkPlate/", 0.25), (r"/cell/", 0.6))))
    mu = dict(zip(model.shape_label, model.shape_material_mu.numpy().tolist()))
    assert mu["/World/envs/env_0/cell/Plate/WorkPlate/Piece_00"] == pytest.approx(0.25)
    assert mu["/World/envs/env_0/cell/Belts/HB2_Belt/TopRun"] == pytest.approx(0.6)
    assert mu["/World/envs/env_0/Robot/left_wrist/collisions/geom_0"] == pytest.approx(1.2)
    assert model.soft_contact_mu == pytest.approx(0.4)

    model = _model()
    apply_contact_material(model, _template(()))
    np.testing.assert_allclose(model.shape_material_mu.numpy(), 0.8)


def test_a_pattern_that_matches_no_collider_is_refused():
    with pytest.raises(ValueError, match="WorkPlates"):
        apply_contact_material(_model(), _template(((r"/Robot/", 1.2), (r"/WorkPlates/", 0.25))))



COLLIDE_PARTICLES = int(newton.ShapeFlags.COLLIDE_PARTICLES)


class _Bag:
    """The one thing the reach check reads of a registered bag: its particle range."""

    def __init__(self, count: int):
        self._count = count

    def particle_ranges(self):
        return [range(self._count)]


def _box_mesh(centre) -> newton.Mesh:
    """A 0.2 m box mesh whose vertices sit around ``centre`` in its own frame."""
    corners = np.array([(x, y, z) for x in (-0.1, 0.1) for y in (-0.1, 0.1) for z in (-0.1, 0.1)]) + np.asarray(centre)
    faces = [0, 1, 3, 0, 3, 2, 4, 6, 7, 4, 7, 5, 0, 4, 5, 0, 5, 1, 2, 3, 7, 2, 7, 6, 0, 2, 6, 0, 6, 4, 1, 5, 7, 1, 7, 3]
    return newton.Mesh(corners.astype(np.float32), np.array(faces, dtype=np.int32))


def test_a_mesh_far_from_its_origin_is_kept_where_its_geometry_is_near_the_bag():
    builder = newton.ModelBuilder()
    builder.add_cloth_grid(pos=wp.vec3(0.0, 0.0, 1.0), rot=wp.quat_identity(), vel=wp.vec3(0.0, 0.0, 0.0),
                           dim_x=2, dim_y=2, cell_x=0.1, cell_y=0.1, mass=0.01)
    origin = wp.transform(wp.vec3(-2.5, 0.0, 1.0), wp.quat_identity())
    builder.add_shape_mesh(-1, xform=origin, mesh=_box_mesh((3.0, 0.0, 0.0)), label="near_part_far_origin")
    builder.add_shape_mesh(-1, xform=origin, mesh=_box_mesh((12.0, 0.0, 0.0)), label="far_part")
    model = builder.finalize(device="cpu")

    assert limit_particle_colliders(model, [_Bag(model.particle_count)], 1.5) == 1
    flags = dict(zip(model.shape_label, (int(f) for f in model.shape_flags.numpy())))
    assert flags["near_part_far_origin"] & COLLIDE_PARTICLES
    assert not flags["far_part"] & COLLIDE_PARTICLES



BOX = np.array([[x, y, z] for z in (0.0, 0.05) for y in (-0.4, 0.4) for x in (-2.0, 2.0)], dtype=np.float32)
FACES = np.array([0, 2, 1, 1, 2, 3, 4, 5, 6, 5, 7, 6, 0, 1, 4, 1, 5, 4, 2, 6, 3, 3, 6, 7,
                  0, 4, 2, 2, 4, 6, 1, 3, 5, 3, 7, 5], dtype=np.int32)


def _conveyor():
    builder = newton.ModelBuilder()
    body = builder.add_body(xform=wp.transform(wp.vec3(3.0, 0.8, 0.0), wp.quat_identity()), mass=10.0, label="cell/Conveyor")
    belt = builder.add_shape_convex_hull(body, xform=wp.transform(wp.vec3(0.0, 0.0, 0.7), wp.quat_identity()),
                                         mesh=newton.Mesh(BOX, FACES), label="cell/Conveyor/Belt")
    return builder, body, belt


def test_the_belt_meets_particles_as_a_box_and_the_robot_as_its_mesh():
    builder, body, belt = _conveyor()
    mass = builder.body_mass[body]
    (box,) = add_particle_boxes(builder, [r"/Belt$"])
    collide_particles, collide_shapes = int(newton.ShapeFlags.COLLIDE_PARTICLES), int(newton.ShapeFlags.COLLIDE_SHAPES)
    assert builder.shape_type[box] == newton.GeoType.BOX and builder.shape_body[box] == body
    assert np.allclose(builder.shape_scale[box], [2.0, 0.4, 0.025])
    assert np.allclose(wp.transform_get_translation(builder.shape_transform[box]), [0.0, 0.0, 0.725])
    assert builder.shape_flags[box] & collide_particles and not builder.shape_flags[box] & collide_shapes
    assert not builder.shape_flags[belt] & collide_particles and builder.shape_flags[belt] & collide_shapes
    assert builder.shape_label[box] == "cell/Conveyor/Belt_particles" and builder.body_mass[body] == mass


def test_only_a_box_shaped_mesh_is_replaced():
    builder = newton.ModelBuilder()
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    builder.add_shape_convex_hull(-1, mesh=newton.Mesh(v, np.array([0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3], dtype=np.int32)),
                                  label="cell/Wedge")
    with pytest.raises(ValueError, match="not a box"):
        add_particle_boxes(builder, ["Wedge"])



ACTIVE = int(newton.ParticleFlags.ACTIVE)


def _cloth_builder():
    """Two 2x2-cell cloth patches, one per environment, 1 m up and nothing under them."""
    builder = newton.ModelBuilder()
    for x in (0.0, 5.0):
        builder.add_cloth_grid(pos=wp.vec3(x, 0.0, 1.0), rot=wp.quat_identity(), vel=wp.vec3(0.0, 0.0, 0.0),
                               dim_x=2, dim_y=2, cell_x=0.1, cell_y=0.1, mass=0.01)
    builder.color()
    return builder


def test_an_inactive_bag_stays_put_without_support():
    model = _cloth_builder().finalize(device="cpu")
    per_object = model.particle_count // 2
    offsets = wp.array([0, per_object], dtype=wp.int32, device="cpu")
    set_particles_active(model.particle_flags, offsets, wp.array([1], dtype=wp.int32, device="cpu"), per_object, False)
    flags = model.particle_flags.numpy()
    assert (flags[:per_object] & ACTIVE).all() and not (flags[per_object:] & ACTIVE).any()

    solver = newton.solvers.SolverVBD(model, iterations=2)
    state_0, state_1 = model.state(), model.state()
    start = state_0.particle_q.numpy().copy()
    for _ in range(10):
        solver.step(state_0, state_1, None, None, 1.0 / 60.0)
        state_0, state_1 = state_1, state_0
    q = state_0.particle_q.numpy()
    assert np.allclose(q[per_object:], start[per_object:])
    assert (q[:per_object, 2] < start[:per_object, 2] - 0.01).all()

    set_particles_active(model.particle_flags, offsets, wp.array([1], dtype=wp.int32, device="cpu"), per_object, True)
    assert (model.particle_flags.numpy() & ACTIVE).all()


def test_only_the_named_colliders_stop_colliding_with_particles():
    builder = newton.ModelBuilder()
    builder.add_shape_box(-1, xform=wp.transform(wp.vec3(0.0, 0.0, -0.6), wp.quat_identity()), hx=1.0, hy=0.4, hz=0.05,
                          label="/World/envs/env_0/l4_park_slab/geometry/mesh")
    builder.add_shape_box(-1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.7), wp.quat_identity()), hx=0.3, hy=0.3, hz=0.05,
                          label="/World/envs/env_0/l4_station_cube/geometry/mesh")
    particles, shapes = int(newton.ShapeFlags.COLLIDE_PARTICLES), int(newton.ShapeFlags.COLLIDE_SHAPES)
    assert stop_particle_collision(builder, (r"/l4_park_slab/",)) == ["/World/envs/env_0/l4_park_slab/geometry/mesh"]
    slab, cube = (int(f) for f in builder.shape_flags)
    assert not slab & particles and slab & shapes
    assert cube & particles
    with pytest.raises(ValueError, match="no particle-colliding shape"):
        stop_particle_collision(builder, (r"/l4_park_slab/",))




def _belt_model(yaw_rad: float, scale=(2.0, 1.0, 1.0)):
    vertices = np.array([[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0],
                         [-0.5, -0.5, -0.1], [0.5, -0.5, -0.1], [0.5, 0.5, -0.1], [-0.5, 0.5, -0.1]], dtype=np.float32)
    faces = np.array([0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 0, 4, 5, 0, 5, 1,
                      1, 5, 6, 1, 6, 2, 2, 6, 7, 2, 7, 3, 3, 7, 4, 3, 4, 0], dtype=np.int32)
    builder = newton.ModelBuilder()
    rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw_rad)
    builder.add_shape_mesh(-1, xform=wp.transform(wp.vec3(1.0, 2.0, 0.75), rot), mesh=newton.Mesh(vertices, faces),
                           scale=wp.vec3(*scale), label="cell/Conveyor/Belt")
    builder.add_shape_box(-1, xform=wp.transform(wp.vec3(5.0, 0.0, 0.0), wp.quat_identity()), hx=0.1, hy=0.1, hz=0.1,
                          label="cell/Conveyor/Leg")
    return builder.finalize(device="cpu")


def test_world_velocity_is_stored_in_the_mesh_frame():
    model = _belt_model(math.pi / 2)
    assert drive_surfaces(model, None, {r"/Belt$": (0.4, 0.0, 0.0)}) == ["cell/Conveyor/Belt"]
    v = model.shape_source[0].mesh.velocities.numpy()
    assert np.allclose(v, [0.0, -0.4, 0.0], atol=1e-6)

    model = _belt_model(0.0)
    drive_surfaces(model, None, {r"/Belt$": (0.4, 0.0, 0.0)})
    assert np.allclose(model.shape_source[0].mesh.velocities.numpy(), [0.2, 0.0, 0.0], atol=1e-6)


def test_a_pattern_that_matches_nothing_or_a_primitive_is_refused():
    model = _belt_model(0.0)
    with pytest.raises(ValueError, match="no particle-colliding shape"):
        drive_surfaces(model, None, {"Roller": (0.4, 0.0, 0.0)})
    with pytest.raises(ValueError, match="only MESH and CONVEX_MESH"):
        drive_surfaces(model, None, {"Leg": (0.4, 0.0, 0.0)})


def test_a_box_is_driven_through_its_contacts():
    from types import SimpleNamespace

    from dexbench_arena.assets.polybag.surface_drive import SoftContactDrive

    model = _belt_model(0.0)
    body_frame = {}
    assert drive_surfaces(model, None, {"Leg": (0.4, 0.0, 0.0)}, body_frame) == ["cell/Conveyor/Leg"]
    assert set(body_frame) == {1} and np.allclose(body_frame[1], [0.4, 0.0, 0.0])
    contacts = SimpleNamespace(
        soft_contact_count=wp.array([3], dtype=wp.int32, device="cpu"),
        soft_contact_shape=wp.array([1, 0, 1, 1], dtype=wp.int32, device="cpu"),
        soft_contact_body_vel=wp.zeros(4, dtype=wp.vec3, device="cpu"))
    SoftContactDrive(model, body_frame)(contacts)
    assert np.allclose(contacts.soft_contact_body_vel.numpy(), [[0.4, 0, 0], [0, 0, 0], [0.4, 0, 0], [0, 0, 0]])



USD = ASSETS_DIR / "object_library" / "polybag_cotton_tet_deformable" / "usd" / "polybag_cotton_tet_deformable.usd"

KE = 1000.0
RADIUS = 0.004
DEPTH = 0.003


def test_face_and_particle_contacts_push_the_body_by_their_penetration():
    height = RADIUS - DEPTH
    q = np.array([(0.0, 0.0, height), (0.02, 0.0, height), (0.0, 0.02, height), (0.05, 0.0, height)], np.float32)
    records = [((0, 1, 2), (1 / 3, 1 / 3, 1 / 3)), ((3, -1, -1), (1.0, 0.0, 0.0))]
    contact_pos = [(0.02 / 3, 0.02 / 3, 0.0), (0.05, 0.0, 0.0)]

    def arr(values, dtype):
        return wp.array(values, dtype=dtype, device="cpu")

    body_f = wp.zeros(1, dtype=wp.spatial_vector, device="cpu")
    particle_q = arr(q, wp.vec3)
    wp.launch(body_soft_contact_reaction, dim=2, device="cpu", inputs=[
        arr([2], wp.int32), arr([r[0] for r in records], wp.vec3i), arr([r[1] for r in records], wp.vec3),
        arr([0, 0], wp.int32), arr(contact_pos, wp.vec3), arr([(0, 0, 0)] * 2, wp.vec3), arr([(0, 0, 1)] * 2, wp.vec3),
        particle_q, particle_q, arr([RADIUS] * 4, wp.float32),
        arr([wp.transform_identity()], wp.transform), arr([wp.transform_identity()], wp.transform),
        arr([wp.spatial_vector()], wp.spatial_vector), arr([(0, 0, 0)], wp.vec3),
        arr([0], wp.int32), arr([0.8], wp.float32), arr([0.0], wp.float32),
        KE, 1.0, 0.5, 1.0e-3, 1.0 / 480.0, False, _SoftContactFilterData(),
    ], outputs=[body_f])

    wrench = body_f.numpy()[0]
    assert wrench[:3] == pytest.approx((0.0, 0.0, -2 * KE * DEPTH), abs=1e-4)
    assert np.isfinite(wrench).all()


def _pad_reaction(mesh: bool, use_filter: bool) -> float:
    """A 6 x 4 cm pad, a triangle mesh (with the SDF full-surface contact gives a robot's mesh) or an analytic box,
    pressed 8 mm into the DB-L4 small bag's top with full-surface contact: the upward reaction on it [N]."""
    t = load_template(str(USD), tet_size_scale=(0.9, 0.9, 0.5))
    builder = newton.ModelBuilder()
    lift = -float(t.rest[:, 2].min()) + t.radius
    builder.add_builder(t.builder, xform=wp.transform(wp.vec3(0.0, 0.0, lift), wp.quat_identity()))
    top = float(t.rest[:, 2].max()) + lift + t.radius
    centre = 0.5 * (t.rest[:, :2].min(axis=0) + t.rest[:, :2].max(axis=0))
    half = (0.03, 0.02, 0.01)
    body = builder.add_body(xform=wp.transform(wp.vec3(float(centre[0]), float(centre[1]), top + half[2] - 0.008),
                                               wp.quat_identity()))
    cfg = newton.ModelBuilder.ShapeConfig(density=1000.0)
    if mesh:
        corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], np.float32) * half
        faces = [0, 2, 1, 1, 2, 3, 4, 5, 6, 5, 7, 6, 0, 1, 4, 1, 5, 4, 2, 6, 3, 3, 6, 7, 0, 4, 2, 2, 4, 6, 1, 3, 5, 3, 7, 5]
        shape = builder.add_shape_mesh(body, mesh=newton.Mesh(corners, np.array(faces, np.int32)), cfg=cfg)
        builder.shape_sdf_target_voxel_size[shape] = 0.002
        builder.shape_sdf_max_resolution[shape] = None
    else:
        builder.add_shape_box(body, hx=half[0], hy=half[1], hz=half[2], cfg=cfg)
    builder.color()
    model = builder.finalize()
    for name, value in t.contact.items():
        setattr(model, name, value)
    substeps = t.stepping[0]
    pipeline = newton.CollisionPipeline(
        model, broad_phase="nxn", soft_contact_margin=t.contact_margin(substeps, 60.0, 0.02),
        soft_contact_max=8 * (t.particle_count + t.builder.tri_count + len(t.builder.edge_indices)),
        enable_rigid_soft_full_surface_contact=True)
    contacts = pipeline.contacts()
    state = model.state()
    pipeline.collide(state, contacts)
    body_f = wp.zeros(model.body_count, dtype=wp.spatial_vector)
    contact_filter = contact_filter_data(model, contacts, state.particle_q, state.body_q) if use_filter \
        else _SoftContactFilterData()
    wp.launch(body_soft_contact_reaction, dim=int(contacts.soft_contact_indices.shape[0]), inputs=[
        contacts.soft_contact_count, contacts.soft_contact_indices, contacts.soft_contact_barycentric,
        contacts.soft_contact_shape, contacts.soft_contact_body_pos, contacts.soft_contact_body_vel,
        contacts.soft_contact_normal, state.particle_q, state.particle_q, model.particle_radius, state.body_q,
        state.body_q, state.body_qd, model.body_com, model.shape_body, model.shape_material_mu, model.shape_margin,
        float(model.soft_contact_ke), float(model.soft_contact_kd), float(model.soft_contact_mu), 1.0e-4,
        1.0 / (60.0 * substeps), False, contact_filter,
    ], outputs=[body_f])
    return float(body_f.numpy()[body][2])


@pytest.mark.skipif(not USD.exists() or USD.stat().st_size < 4096,
                    reason="tet polybag package payload not checked out (LFS pointer or missing)")
@pytest.mark.skipif(not HAS_CONTACT_FILTER, reason="this Newton predates newton#4227: every contact carries a force")
def test_a_mesh_pad_gets_the_reaction_of_the_contacts_vbd_applies_only():
    every, applied = _pad_reaction(mesh=True, use_filter=False), _pad_reaction(mesh=True, use_filter=True)
    assert applied > 0.0 and every > 1.5 * applied
    assert _pad_reaction(mesh=False, use_filter=True) == pytest.approx(_pad_reaction(mesh=False, use_filter=False))


@pytest.mark.skipif(not USD.exists() or USD.stat().st_size < 4096,
                    reason="needs the tet polybag package's LFS payload")
def test_two_way_coupling_refuses_unequal_contact_sides():
    """The reaction is the bag's side alone and VBD's force the sides' mean: two-way takes only equal sides."""
    from dexbench_arena.assets.polybag.physics import check_two_way_contact

    def template(contact_ke):
        return load_template(str(USD), tet_contact_ke=contact_ke, tet_contact_kd=1.0, tet_collider_ke=3000.0,
                             tet_collider_kd=1.0)

    with pytest.raises(ValueError, match="equal contact ke"):
        check_two_way_contact(template(5000.0), "two_way")
    check_two_way_contact(template(5000.0), "one_way")
    check_two_way_contact(template(3000.0), "two_way")
