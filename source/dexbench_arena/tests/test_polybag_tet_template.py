# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""The tet polybag template: the bag loads the way its delivery says, and its textured mesh rides on the tets."""

import numpy as np
import pytest

newton = pytest.importorskip("newton")
wp = pytest.importorskip("warp")
pytest.importorskip("pxr")
pytest.importorskip("isaaclab.scene_data.deformable_vis_remap")

from dexbench_arena import ASSETS_DIR  # noqa: E402
from dexbench_arena.assets.polybag.template import changed_layer, lame, load_template  # noqa: E402
from dexbench_arena.assets.polybag.tet_template import (  # noqa: E402
    FILM_KD_ATTR, FILM_KE_ATTR, FILM_MODE_ATTR, TET_COLLIDER_KE, TET_POISSON_RATIO_ATTR, TET_SELF_CONTACT_MARGIN_M,
    TET_SELF_CONTACT_RADIUS_M, TET_STEPPING, TET_YOUNGS_MODULUS_ATTR, is_tet_polybag,
)

USD = ASSETS_DIR / "object_library" / "polybag_cotton_tet_deformable" / "usd" / "polybag_cotton_tet_deformable.usd"
FILM_USD = ASSETS_DIR / "object_library" / "polybag_bubble_cotton_deformable" / "usd" / "polybag_bubble_cotton_deformable.usd"
pytestmark = pytest.mark.skipif(not USD.exists() or USD.stat().st_size < 4096,
                                reason="tet polybag package payload not checked out (LFS pointer or missing)")


def test_the_bag_loads_as_its_delivery_says():
    t = load_template(str(USD))
    assert is_tet_polybag(str(USD))
    assert t.particle_count == 536 and t.builder.tet_count == 1770
    assert np.isclose(np.sum(t.builder.particle_mass), 0.090, rtol=1e-3)
    assert np.allclose(t.builder.particle_radius, 0.004)
    assert np.allclose(np.asarray(t.builder.tet_materials, dtype=np.float64)[:, 2], 30.0)
    assert t.builder.shape_count == 0 and t.builder.body_count == 0
    assert t.film.stop == t.film.start and t.contents == slice(0, 536) and not t.seal.size
    assert t.stepping == TET_STEPPING and t.solver_kwargs == {
        "particle_enable_self_contact": True, "particle_self_contact_radius": TET_SELF_CONTACT_RADIUS_M,
        "particle_self_contact_margin": TET_SELF_CONTACT_MARGIN_M}
    b = newton.ModelBuilder()
    b.add_particle((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1.0)
    defaults = b.finalize(device="cpu")
    assert t.contact == {k: float(getattr(defaults, k)) for k in t.contact}
    assert t.fixture["shape_material_ke"] == TET_COLLIDER_KE and t.fixture_on_bodies


def test_textured_mesh_rides_on_the_tets():
    t = load_template(str(USD))
    r = t.render
    assert r.points.shape == (16506, 3) and r.tet_vertex_indices.shape == (16506, 4)
    assert np.allclose(r.bary_weights.sum(axis=1), 1.0, atol=1e-4)
    assert ((r.tet_vertex_indices >= 0) & (r.tet_vertex_indices < t.particle_count)).all()
    rest = t.rest.astype(np.float32)
    remapped = np.einsum("vk,vkd->vd", r.bary_weights, rest[r.tet_vertex_indices])
    assert np.abs(remapped - r.points).max() < 1e-4



def _per_tet_bag(path, values: dict) -> str:
    """The delivered bag with per-tet `values` ({attribute: array}) authored on its TetMesh, saved at `path`."""
    from pxr import Sdf, Usd, UsdGeom, Vt

    layer = changed_layer(str(USD), {})
    stage = Usd.Stage.Open(layer)
    tet = next(p for p in stage.Traverse() if p.IsA(UsdGeom.TetMesh))
    for name, array in values.items():
        tet.CreateAttribute(name, Sdf.ValueTypeNames.FloatArray).Set(Vt.FloatArray.FromNumpy(np.asarray(array, np.float32)))
    layer.Export(str(path))
    return str(path)


def test_a_bag_of_several_materials_keeps_them_and_their_ratios_under_a_modulus_override(tmp_path):
    from pxr import Usd

    from dexbench_arena.assets.polybag.tet_template import _parts

    n = load_template(str(USD)).builder.tet_count
    stiff = np.arange(n) % 2 == 1
    e, nu = np.where(stiff, 200_000.0, 10_000.0), np.where(stiff, 0.3, 0.05).astype(np.float32)
    usd = _per_tet_bag(tmp_path / "layered.usda", {TET_YOUNGS_MODULUS_ATTR: e, TET_POISSON_RATIO_ATTR: nu})
    lames = np.array([lame(a, b) for a, b in zip(e, nu)])
    tets = np.asarray(load_template(usd).builder.tet_materials, dtype=np.float64)
    assert np.allclose(tets[:, :2], lames, rtol=1e-6) and np.allclose(tets[:, 2], 30.0)
    asset = Usd.Stage.Open(str(USD))
    bound_e = float(_parts(asset)[1].GetAttribute("physics:youngsModulus").Get())
    scaled = np.asarray(load_template(usd, tet_youngs_modulus=3.0 * bound_e).builder.tet_materials, dtype=np.float64)
    assert np.allclose(scaled[:, :2], 3.0 * lames, rtol=1e-6)
    plain = np.asarray(load_template(str(USD)).builder.tet_materials, dtype=np.float64)
    assert np.allclose(plain, plain[0])
    with pytest.raises(ValueError, match="tets"):
        load_template(_per_tet_bag(tmp_path / "short.usda", {TET_YOUNGS_MODULUS_ATTR: e[:-1]}))


def _film_bag(path, mode: str, ke: float, kd: float) -> str:
    """The delivered bag with a surface film authored on its TetMesh, saved at `path`."""
    from pxr import Sdf, Usd, UsdGeom

    stage = Usd.Stage.Open(changed_layer(str(USD), {}))
    tet = next(p for p in stage.Traverse() if p.IsA(UsdGeom.TetMesh))
    tet.CreateAttribute(FILM_MODE_ATTR, Sdf.ValueTypeNames.String).Set(mode)
    tet.CreateAttribute(FILM_KE_ATTR, Sdf.ValueTypeNames.Float).Set(ke)
    tet.CreateAttribute(FILM_KD_ATTR, Sdf.ValueTypeNames.Float).Set(kd)
    stage.GetRootLayer().Export(str(path))
    return str(path)


def test_a_film_authored_on_the_bag_is_a_spring_on_every_surface_edge_at_its_length(tmp_path):
    plain = load_template(str(USD)).builder
    assert plain.spring_count == 0
    builder = load_template(_film_bag(tmp_path / "film.usda", "springs", 2000.0, 0.1)).builder
    tri = np.asarray(plain.tri_indices).reshape(-1, 3)
    edges = {tuple(sorted(e)) for t in tri for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))}
    pairs = np.asarray(builder.spring_indices).reshape(-1, 2)
    assert {tuple(sorted(p)) for p in pairs.tolist()} == edges and len(pairs) == len(edges)
    q = np.asarray(builder.particle_q, dtype=np.float64)
    assert np.allclose(builder.spring_rest_length, np.linalg.norm(q[pairs[:, 0]] - q[pairs[:, 1]], axis=1), rtol=1e-5)
    assert set(builder.spring_stiffness) == {2000.0} and np.allclose(builder.spring_damping, 0.1)
    assert not np.asarray(builder.tri_materials).any() and not np.asarray(builder.edge_bending_properties).any()
    for mode, ke in (("membrane", 2000.0), ("springs", 0.0)):
        with pytest.raises(ValueError, match="film"):
            load_template(_film_bag(tmp_path / f"bad_{mode}_{ke:g}.usda", mode, ke, 0.1))


def _rest_volume(t) -> float:
    """The tets' rest volume as the solver holds it (``tet_poses``, each its rest edge matrix inverted) [m^3]."""
    poses = np.asarray(t.builder.tet_poses, dtype=np.float64).reshape(-1, 3, 3)
    return float(np.sum(1.0 / np.abs(np.linalg.det(poses))) / 6.0)


SMALL = (0.9, 0.9, 0.5)


def test_a_smaller_bag_is_the_same_bag_with_its_rest_shapes_and_textured_mesh_scaled():
    t, small = load_template(str(USD)), load_template(str(USD), tet_size_scale=SMALL)
    assert small.size_scale == SMALL and t.size_scale == (1.0, 1.0, 1.0)
    assert small.particle_count == t.particle_count and small.builder.tet_count == t.builder.tet_count
    assert np.allclose(np.ptp(small.rest, axis=0), np.ptp(t.rest, axis=0) * SMALL)
    assert np.allclose(np.ptp(small.render.points, axis=0), np.ptp(t.render.points, axis=0) * SMALL, atol=1e-6)
    assert _rest_volume(small) == pytest.approx(float(np.prod(SMALL)) * _rest_volume(t), rel=1e-6)
    rest = small.rest.astype(np.float32)
    remapped = np.einsum("vk,vkd->vd", small.render.bary_weights, rest[small.render.tet_vertex_indices])
    assert np.abs(remapped - small.render.points).max() < 1e-4
    assert small.solver_kwargs == t.solver_kwargs and small.contact == t.contact and small.stepping == t.stepping
    with pytest.raises(ValueError, match="positive"):
        load_template(str(USD), tet_size_scale=(1.0, 1.0, 0.0))


def test_film_and_cotton_arguments_are_refused():
    with pytest.raises(ValueError, match="one tet soft body"):
        load_template(str(USD), cotton_youngs_modulus=10_000.0)
    if FILM_USD.exists() and FILM_USD.stat().st_size >= 4096:
        assert not is_tet_polybag(str(FILM_USD))
        with pytest.raises(ValueError, match="tet bag only"):
            load_template(str(FILM_USD), tet_youngs_modulus=30_000.0)
    with pytest.raises(ValueError, match="poisson_ratio"):
        load_template(str(USD), tet_poisson_ratio=0.5)
    with pytest.raises(ValueError, match="contact_ke"):
        load_template(str(USD), tet_contact_ke=0.0)


def _drop(frames: int = 60, **material) -> tuple[float, float]:
    """Drop the bag 2 cm onto a plane at its delivery's stepping and let it settle for `frames` / 60 s."""
    t = load_template(str(USD), **material)
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    builder.add_builder(t.builder, xform=wp.transform(wp.vec3(0.0, 0.0, 0.02 - float(t.rest[:, 2].min())),
                                                      wp.quat_identity()))
    builder.color()
    model = builder.finalize()
    for name, value in t.contact.items():
        setattr(model, name, value)
    for name, value in t.fixture.items():
        array = getattr(model, name)
        array.assign(np.full(array.shape, value, dtype=np.float32))
    substeps, iterations = t.stepping
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", soft_contact_margin=t.contact_margin(substeps, 60.0, 0.02))
    contacts = pipeline.contacts()
    solver = newton.solvers.SolverVBD(model, iterations=iterations, **t.solver_kwargs)
    state_0, state_1, control = model.state(), model.state(), model.control()
    for _ in range(frames):
        solver.rebuild_bvh(state_0)
        for _ in range(substeps):
            state_0.clear_forces()
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, 1.0 / (60.0 * substeps))
            state_0, state_1 = state_1, state_0
    q = state_0.particle_q.numpy()
    assert np.isfinite(q).all()
    return float(np.ptp(q[:, 2]) / np.ptp(t.rest[:, 2])), float(t.radius - q[:, 2].min())


def test_a_stiffer_bag_slumps_less_when_dropped():
    soft, delivered, stiff = (_drop(tet_youngs_modulus=e)[0] for e in (1_500.0, None, 150_000.0))
    assert soft < delivered - 0.02 and stiff > delivered + 0.005


def test_a_stiffer_contact_lets_the_bag_sink_less_into_the_floor():
    soft = _drop(tet_contact_ke=1e2, tet_collider_ke=1e3)[1]
    delivered = _drop()[1]
    stiff = _drop(tet_contact_ke=1e5, tet_collider_ke=1e5)[1]
    assert soft > delivered + 5e-4 and stiff < delivered - 5e-5


def test_a_smaller_bag_settles_at_its_own_thickness():
    height = _drop(tet_size_scale=SMALL)[0]
    assert 0.85 < height < 1.05
