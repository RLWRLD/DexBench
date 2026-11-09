# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""DB-L1's VBD solver choices for the bag: the psd film Hessian, the bag tets' material, and the contact initial guess."""

import numpy as np
import pytest
from packaging.version import Version

wp = pytest.importorskip("warp")
newton = pytest.importorskip("newton")

from dexbench_arena.teleop.newton_solver import NEWTON_SOLVER_MIN_NEWTON  # noqa: E402

if Version(newton.__version__) < Version(NEWTON_SOLVER_MIN_NEWTON):
    pytest.skip(f"Newton {newton.__version__}: DB-L1 runs on {NEWTON_SOLVER_MIN_NEWTON} or later",
                allow_module_level=True)

from newton._src.solvers.vbd.particle_vbd_kernels import accumulate_spring_force_and_hessian  # noqa: E402

from dexbench_arena.assets.polybag.film_hessian import accumulate_spring_force_and_psd_hessian  # noqa: E402
from dexbench_arena.assets.polybag.two_way_reaction import (  # noqa: E402
    body_soft_contact_reaction,
    contact_filter_data,
)
from dexbench_arena.assets.polybag.vbd_init import ContactInitSolverVBD  # noqa: E402


def _spring(kernel, length: float):
    """One spring of rest length 0.1 m, stiffness 2000 N/m, damping 0.1, along a skewed axis at ``length`` (its anchor
    at 0.1 m): -> (forces, hessians) of its two vertices."""
    device = "cpu"
    axis = np.array([0.6, 0.0, 0.8])
    pos = wp.array(np.array([axis * length, [0.0, 0.0, 0.0]]), dtype=wp.vec3, device=device)
    anchor = wp.array(np.array([axis * 0.1, [0.0, 0.0, 0.0]]), dtype=wp.vec3, device=device)
    forces = wp.zeros(2, dtype=wp.vec3, device=device)
    hessians = wp.zeros(2, dtype=wp.mat33, device=device)
    wp.launch(kernel, dim=1, device=device, inputs=[
        1.0 / 480.0, 0, anchor, pos, wp.array([0, 0], dtype=int, device=device), 1,
        wp.array([0, 1], dtype=int, device=device), wp.array([0.1], dtype=float, device=device),
        wp.array([2000.0], dtype=float, device=device), wp.array([0.1], dtype=float, device=device),
    ], outputs=[forces, hessians])
    return forces.numpy(), hessians.numpy()


def test_the_psd_film_hessian_keeps_newtons_force_and_drops_only_the_compressed_transverse_term():
    stretched_newton, stretched_psd = _spring(accumulate_spring_force_and_hessian, 0.12), _spring(
        accumulate_spring_force_and_psd_hessian, 0.12)
    for a, b in zip(stretched_newton, stretched_psd):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-4)
    (f_newton, h_newton), (f_psd, h_psd) = (_spring(k, 0.08) for k in (accumulate_spring_force_and_hessian,
                                                                       accumulate_spring_force_and_psd_hessian))
    np.testing.assert_allclose(f_newton, f_psd, rtol=1e-5, atol=1e-4)
    assert np.linalg.eigvalsh(h_newton[0]).min() < -100.0
    assert np.linalg.eigvalsh(h_psd[0]).min() > -1e-3
    np.testing.assert_allclose(h_psd[0], h_psd[1])


def test_apply_tet_material_sets_the_bag_tets_poisson_ratio_at_their_own_shear_modulus_and_their_damping():
    from dexbench_arena.assets.polybag.physics import apply_tet_material

    builder = newton.ModelBuilder()
    builder.add_soft_grid(pos=wp.vec3(0.0), rot=wp.quat_identity(), vel=wp.vec3(0.0), dim_x=2, dim_y=1, dim_z=1,
                          cell_x=0.01, cell_y=0.01, cell_z=0.01, density=100.0, k_mu=1000.0, k_lambda=50.0, k_damp=0.5)
    model = builder.finalize(device="cpu")

    class Bag:
        def particle_ranges(self):
            return [np.arange(model.particle_count)]

    before = model.tet_materials.numpy().copy()
    assert apply_tet_material(model, [Bag()], 0.45, None) == model.tet_count
    after = model.tet_materials.numpy()
    np.testing.assert_allclose(after[:, 0], before[:, 0])
    np.testing.assert_allclose(after[:, 1], 2.0 * 1000.0 * 0.45 / 0.1, rtol=1e-5)
    np.testing.assert_allclose(after[:, 2], before[:, 2])
    apply_tet_material(model, [Bag()], None, 10.0)
    np.testing.assert_allclose(model.tet_materials.numpy()[:, 2], 10.0)
    np.testing.assert_allclose(model.tet_materials.numpy()[:, 1], after[:, 1])
    with pytest.raises(ValueError):
        apply_tet_material(model, [Bag()], 0.5, None)


GRAVITY = 9.81
MASS = 0.25
KE = 3000.0
RADIUS = 0.004
SUBSTEPS = 8
ITERATIONS = 5
FILM_KE = 2000.0


def _support_of_a_resting_block(solver_class) -> float:
    """A 10 x 8 x 6 cm soft tet block of 0.25 kg in a stiff surface film, laid 1 mm above a kinematic slab and
    stepped 0.5 s at 8 substeps x 5 iterations (60 Hz): the slab's push on the block from the last final iterate's
    contacts, as the two-way reaction evaluates it at the next substep's start [N]. Without the film both starts
    give the same support: the iterations keep up."""
    device = "cpu"
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -GRAVITY))
    size, cell = (0.10, 0.08, 0.06), 0.015
    dims = [int(round(s / cell)) for s in size]
    e, nu = 2500.0, 0.05
    builder.add_soft_grid(
        pos=wp.vec3(-size[0] / 2, -size[1] / 2, RADIUS + 0.001), rot=wp.quat_identity(), vel=wp.vec3(0.0),
        dim_x=dims[0], dim_y=dims[1], dim_z=dims[2], cell_x=size[0] / dims[0], cell_y=size[1] / dims[1],
        cell_z=size[2] / dims[2], density=MASS / float(np.prod(size)), k_mu=e / (2 * (1 + nu)),
        k_lambda=e * nu / ((1 + nu) * (1 - 2 * nu)), k_damp=0.5, particle_radius=RADIUS)
    count = builder.particle_count
    mass = np.asarray(builder.particle_mass, dtype=float)
    builder.particle_mass = list(mass * (MASS / mass.sum()))
    tris = np.asarray(builder.tri_indices, dtype=np.int64).reshape(-1, 3)
    edges = np.unique(np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1), axis=0)
    for i, j in edges:
        builder.add_spring(int(i), int(j), FILM_KE, 0.1, 0)
    slab = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, -0.01), wp.quat_identity()), is_kinematic=True)
    builder.add_shape_box(slab, hx=0.2, hy=0.2, hz=0.01,
                          cfg=newton.ModelBuilder.ShapeConfig(density=0.0, ke=KE, kd=1.0, mu=1.0))
    builder.color(include_bending=False)
    model = builder.finalize(device=device)
    model.soft_contact_ke, model.soft_contact_kd, model.soft_contact_mu = KE, 1.0, 1.0
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", soft_contact_gap=0.008)
    contacts = pipeline.contacts()
    solver = solver_class(model, iterations=ITERATIONS, integrate_with_external_rigid_solver=True)
    if solver_class is ContactInitSolverVBD:
        solver.set_bag_groups([np.arange(count)])
    s0, s1, control = model.state(), model.state(), model.control()
    dt = 1.0 / (60.0 * SUBSTEPS)
    for _ in range(30 * SUBSTEPS):
        s0.clear_forces()
        pipeline.collide(s0, contacts)
        solver.step(s0, s1, control, contacts, dt)
        s0, s1 = s1, s0
    pipeline.collide(s0, contacts)
    body_f = wp.zeros(model.body_count, dtype=wp.spatial_vector, device=device)
    wp.launch(body_soft_contact_reaction, dim=int(contacts.soft_contact_indices.shape[0]), device=device, inputs=[
        contacts.soft_contact_count, contacts.soft_contact_indices, contacts.soft_contact_barycentric,
        contacts.soft_contact_shape, contacts.soft_contact_body_pos, contacts.soft_contact_body_vel,
        contacts.soft_contact_normal, s0.particle_q, solver.particle_q_prev, model.particle_radius, s0.body_q,
        s0.body_q, s0.body_qd, model.body_com, model.shape_body, model.shape_material_mu, model.shape_margin,
        float(model.soft_contact_ke), float(model.soft_contact_kd), float(model.soft_contact_mu),
        float(solver.friction_epsilon), dt, False, contact_filter_data(model, contacts, s0.particle_q, s0.body_q),
    ], outputs=[body_f])
    assert np.isfinite(s0.particle_q.numpy()).all()
    return -float(body_f.numpy()[slab][2])


def test_a_resting_block_pushes_its_slab_with_its_weight_only_when_its_iterate_starts_without_the_sink():
    weight = MASS * GRAVITY
    newtons_own = _support_of_a_resting_block(newton.solvers.SolverVBD)
    without_sink = _support_of_a_resting_block(ContactInitSolverVBD)
    assert without_sink == pytest.approx(weight, rel=0.02)
    assert newtons_own > 1.3 * weight
