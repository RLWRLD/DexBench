# SPDX-License-Identifier: Apache-2.0
"""A polybag delivered as one tet soft body (``object_library/polybag_cotton_tet_deformable``)."""

from __future__ import annotations

import contextlib
import io
import re
from dataclasses import dataclass
from functools import lru_cache

import newton
import numpy as np
from pxr import Usd, UsdGeom, UsdShade, Vt

from isaaclab.scene_data.deformable_vis_remap import build_volume_vis_barycentric_remap

from .template import FILM_SIM, PolybagTemplate, changed_layer, lame

TET_STEPPING = (8, 5)
TET_SELF_CONTACT_RADIUS_M = 1.0e-3
TET_SELF_CONTACT_MARGIN_M = 2.0e-3
TET_COLLIDER_KE = 1.0e4
TET_COLLIDER_KD = 10.0
TET_COLLIDER_MU = 0.8
TET_SOFT_CONTACT = {"soft_contact_ke": 1.0e3, "soft_contact_kd": 10.0, "soft_contact_mu": 0.5, "soft_contact_restitution": 0.0}


@dataclass
class PolybagRender:
    """The bag's textured mesh, as authored, and how its vertices ride on the tets."""

    mesh_path: str
    material_path: str
    points: np.ndarray
    tet_vertex_indices: np.ndarray
    bary_weights: np.ndarray


def is_tet_polybag(usd_path: str) -> bool:
    """Whether the asset is a polybag of one TetMesh (and no film): this module's kind."""
    stage = Usd.Stage.Open(str(usd_path))
    return not stage.GetPrimAtPath(FILM_SIM) and any(p.IsA(UsdGeom.TetMesh) for p in stage.Traverse())


def _parts(stage):
    """-> (tet prim, its physics material prim, the textured mesh prim, its material prim)."""
    tets = [p for p in stage.Traverse() if p.IsA(UsdGeom.TetMesh)]
    if len(tets) != 1:
        raise ValueError(f"a tet polybag has one TetMesh; this asset has {len(tets)}")
    tet = tets[0]
    targets = tet.GetRelationship("material:binding:physics").GetTargets()
    if not targets:
        raise ValueError(f"{tet.GetPath()} binds no physics material")
    material = stage.GetPrimAtPath(targets[0])
    meshes = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)
              and UsdGeom.Imageable(p).GetPurposeAttr().Get() in (None, UsdGeom.Tokens.default_)]
    if len(meshes) != 1:
        raise ValueError(f"a tet polybag has one textured mesh to draw; this asset has {len(meshes)}")
    mesh = meshes[0]
    look, _ = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()
    if not look:
        raise ValueError(f"{mesh.GetPath()} binds no material")
    return tet, material, mesh, look.GetPrim()


def material_values(stage, youngs_modulus=None, poisson_ratio=None, k_damp=None) -> dict:
    """-> {"prim.attr": value} on the bag's physics material for the changes asked, the Lame pair kept equal."""
    _, material, _, _ = _parts(stage)
    path = str(material.GetPath())
    out = {}
    if youngs_modulus is not None or poisson_ratio is not None:
        e = float(material.GetAttribute("physics:youngsModulus").Get() if youngs_modulus is None else youngs_modulus)
        nu = float(material.GetAttribute("physics:poissonsRatio").Get() if poisson_ratio is None else poisson_ratio)
        if not e > 0.0:
            raise ValueError(f"youngs_modulus {e} must be positive")
        if not 0.0 <= nu < 0.5:
            raise ValueError(f"poisson_ratio {nu} is outside [0, 0.5)")
        mu, lam = lame(e, nu)
        out.update({f"{path}.physics:youngsModulus": e, f"{path}.physics:poissonsRatio": nu,
                    f"{path}.newton:kMu": mu, f"{path}.newton:kLambda": lam})
    if k_damp is not None:
        if not float(k_damp) >= 0.0:
            raise ValueError(f"k_damp {k_damp} must not be negative")
        out[f"{path}.newton:kDamp"] = float(k_damp)
    return out


TET_YOUNGS_MODULUS_ATTR = "dexbench:tetYoungsModulus"
TET_POISSON_RATIO_ATTR = "dexbench:tetPoissonsRatio"
TET_K_DAMP_ATTR = "dexbench:tetKDamp"


def per_tet_materials(stage, tet_count: int, youngs_modulus=None, poisson_ratio=None, k_damp=None) -> np.ndarray | None:
    """-> (tet_count, 3) Newton tet materials (k_mu, k_lambda, k_damp) from the asset's per-tet attributes, None when
    its TetMesh authors no ``dexbench:tetYoungsModulus`` (the bound material then holds for every tet)."""
    tet, material, _, _ = _parts(stage)

    def authored(name: str):
        attr = tet.GetAttribute(name)
        return attr if attr and attr.HasAuthoredValue() else None

    if authored(TET_YOUNGS_MODULUS_ATTR) is None:
        return None

    def per_tet(name: str, fallback: float) -> np.ndarray:
        attr = authored(name)
        if attr is None:
            return np.full(tet_count, float(fallback))
        values = np.asarray(attr.Get(), dtype=np.float64)
        if values.shape != (tet_count,) or not np.isfinite(values).all():
            raise ValueError(f"{tet.GetPath()}.{name}: {values.size} value(s) for {tet_count} tets, or not finite")
        return values

    bound_e = float(material.GetAttribute("physics:youngsModulus").Get())
    e = per_tet(TET_YOUNGS_MODULUS_ATTR, bound_e)
    if youngs_modulus is not None:
        e = e * float(youngs_modulus) / bound_e
    nu = (np.full(tet_count, float(poisson_ratio)) if poisson_ratio is not None
          else per_tet(TET_POISSON_RATIO_ATTR, material.GetAttribute("physics:poissonsRatio").Get()))
    damp = (np.full(tet_count, float(k_damp)) if k_damp is not None
            else per_tet(TET_K_DAMP_ATTR, material.GetAttribute("newton:kDamp").Get()))
    if not (e > 0.0).all() or not ((nu >= 0.0) & (nu < 0.5)).all() or not (damp >= 0.0).all():
        raise ValueError(f"{tet.GetPath()}: per-tet E must be positive, nu in [0, 0.5) and damping not negative")
    mu, lam = np.array([lame(a, b) for a, b in zip(e, nu)]).T
    return np.column_stack([mu, lam, damp])


FILM_MODE_ATTR = "dexbench:surfaceFilmMode"
FILM_KE_ATTR = "dexbench:surfaceTriKe"
FILM_KD_ATTR = "dexbench:surfaceTriKd"
FILM_REGION_ATTR = "dexbench:surfaceFilmRegion"


def surface_film_springs(stage, builder) -> np.ndarray:
    """Add the film the asset's TetMesh authors to `builder` (its imported tet bag): -> the (n, 2) particle pairs the
    springs join, empty when the asset authors no film."""
    tet, _, _, _ = _parts(stage)

    def authored(name: str, fallback=None):
        attr = tet.GetAttribute(name)
        return attr.Get() if attr and attr.HasAuthoredValue() else fallback

    mode = authored(FILM_MODE_ATTR)
    if mode is None:
        return np.zeros((0, 2), dtype=np.int64)
    ke, kd = float(authored(FILM_KE_ATTR, 0.0)), float(authored(FILM_KD_ATTR, 0.0))
    if mode != "springs" or authored(FILM_REGION_ATTR, "all") != "all" or not ke > 0.0 or not kd >= 0.0:
        raise ValueError(f"{tet.GetPath()}: a film is {FILM_MODE_ATTR} 'springs' over {FILM_REGION_ATTR} 'all' with "
                         f"{FILM_KE_ATTR} > 0 and {FILM_KD_ATTR} >= 0")
    tri = np.asarray(builder.tri_indices, dtype=np.int64).reshape(-1, 3)
    edges = np.unique(np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1), axis=0)
    for i, j in edges:
        builder.add_spring(int(i), int(j), ke, kd, 0)
    return edges


def _contact_numbers(defaults: dict, keys: dict, given: dict) -> dict:
    """`defaults` with the values `given` for its `keys` ({argument: entry}) that are not None, checked."""
    out = dict(defaults)
    for argument, key in keys.items():
        value = given[argument]
        if value is None:
            continue
        if argument.endswith("_ke") and not float(value) > 0.0:
            raise ValueError(f"{argument} {value} must be positive")
        if not float(value) >= 0.0:
            raise ValueError(f"{argument} {value} must not be negative")
        out[key] = float(value)
    return out


def size_values(stage, size_scale: tuple[float, float, float]) -> dict:
    """-> {"prim.points": points} scaling the TetMesh's and the textured mesh's points by ``size_scale`` (x, y, z).
    ValueError for a scale that is not three positive numbers, or an asset whose thickness is not its z."""
    if len(size_scale) != 3 or not all(float(s) > 0.0 for s in size_scale):
        raise ValueError(f"size_scale {size_scale} must be three positive numbers (x, y, z)")
    tet, _, mesh, _ = _parts(stage)
    out = {}
    for prim in (tet, mesh):
        points = np.asarray(UsdGeom.PointBased(prim).GetPointsAttr().Get(), dtype=np.float64)
        extent = points.max(axis=0) - points.min(axis=0)
        if prim == tet and int(np.argmin(extent)) != 2:
            raise ValueError(f"{prim.GetPath()}: its thinnest extent is along axis {int(np.argmin(extent))}, not z")
        points *= np.asarray(size_scale, dtype=np.float64)
        out[f"{prim.GetPath()}.points"] = Vt.Vec3fArray.FromNumpy(points.astype(np.float32))
    return out


def _collider_mu_by_path(pairs) -> tuple[tuple[str, float], ...]:
    """`pairs` of (prim path pattern, friction), checked: ValueError for a pattern that does not compile or a negative
    friction."""
    out = []
    for pattern, mu in pairs or ():
        try:
            re.compile(pattern)
        except re.error as error:
            raise ValueError(f"collider_mu_by_path pattern {pattern!r}: {error}") from None
        if not float(mu) >= 0.0:
            raise ValueError(f"collider_mu_by_path {pattern!r}: friction {mu} must not be negative")
        out.append((str(pattern), float(mu)))
    return tuple(out)


@lru_cache(maxsize=8)
def load_tet_template(usd_path: str, *, youngs_modulus: float | None = None, poisson_ratio: float | None = None,
                      k_damp: float | None = None, contact_ke: float | None = None, contact_kd: float | None = None,
                      contact_mu: float | None = None, collider_ke: float | None = None,
                      collider_kd: float | None = None, collider_mu: float | None = None,
                      collider_mu_by_path: tuple[tuple[str, float], ...] | None = None,
                      size_scale: tuple[float, float, float] | None = None):
    """Build the tet polybag at `usd_path` once per material: -> ``template.PolybagTemplate`` with ``render`` set."""
    source = Usd.Stage.Open(str(usd_path))
    values = material_values(source, youngs_modulus, poisson_ratio, k_damp)
    contact = _contact_numbers(TET_SOFT_CONTACT, {"contact_ke": "soft_contact_ke", "contact_kd": "soft_contact_kd",
                                                  "contact_mu": "soft_contact_mu"},
                               {"contact_ke": contact_ke, "contact_kd": contact_kd, "contact_mu": contact_mu})
    fixture = _contact_numbers({"shape_material_ke": TET_COLLIDER_KE, "shape_material_kd": TET_COLLIDER_KD,
                                "shape_material_mu": TET_COLLIDER_MU},
                               {"collider_ke": "shape_material_ke", "collider_kd": "shape_material_kd",
                                "collider_mu": "shape_material_mu"},
                               {"collider_ke": collider_ke, "collider_kd": collider_kd, "collider_mu": collider_mu})
    by_path = _collider_mu_by_path(collider_mu_by_path)
    size_scale = (1.0, 1.0, 1.0) if size_scale is None else tuple(float(s) for s in size_scale)
    geometry = size_values(source, size_scale) if size_scale != (1.0, 1.0, 1.0) else {}
    layer = changed_layer(str(usd_path), {**values, **geometry}) if values or geometry else None
    asset = layer.identifier if layer is not None else str(usd_path)
    stage = Usd.Stage.Open(layer) if layer is not None else source
    tet, material, mesh, look = _parts(stage)
    radius = float(material.GetAttribute("newton:particleRadius").Get())
    tet_k_damp = float(material.GetAttribute("newton:kDamp").Get())
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        builder = newton.ModelBuilder()
        builder.default_particle_radius = radius
        builder.default_tet_k_damp = tet_k_damp
        builder.add_usd(asset, ignore_paths=[f"^{mesh.GetPath()}$"])
    if builder.shape_count or builder.body_count:
        raise ValueError(f"{usd_path}: the import added {builder.shape_count} shape(s) and {builder.body_count} "
                         "body(ies); a tet polybag is particles only")
    tet_materials = per_tet_materials(source, len(builder.tet_materials), youngs_modulus, poisson_ratio, k_damp)
    if tet_materials is not None:
        builder.tet_materials = [tuple(row) for row in tet_materials.tolist()]
    surface_film_springs(source, builder)
    rest =np.asarray(builder.particle_q, dtype=np.float64)
    tets = np.asarray(builder.tet_indices, dtype=np.int32).reshape(-1)
    points = np.asarray(UsdGeom.Mesh(mesh).GetPointsAttr().Get(), dtype=np.float32)
    remap = build_volume_vis_barycentric_remap(rest.astype(np.float32), tets, points, device="cpu")
    if remap is None:
        raise ValueError(f"{usd_path}: no tet holds the textured mesh's vertices")
    render = PolybagRender(mesh_path=str(mesh.GetPath()), material_path=str(look.GetPath()), points=points,
                           tet_vertex_indices=remap.tet_vertex_indices.numpy().astype(np.int32),
                           bary_weights=remap.bary_weights.numpy().astype(np.float32))
    n = len(rest)
    return PolybagTemplate(
        usd_path=str(usd_path), builder=builder, rest=rest, film=slice(0, 0), contents=slice(0, n),
        film_tris=np.zeros((0, 3), dtype=np.int64), seal=np.zeros((0, 2), dtype=np.int64),
        vertex_triangle_exclusions={}, edge_edge_exclusions={},
        contact=contact, fixture=fixture, fixture_on_bodies=True, material=values,
        solver_kwargs={"particle_enable_self_contact": True, "particle_self_contact_radius": TET_SELF_CONTACT_RADIUS_M,
                       "particle_self_contact_margin": TET_SELF_CONTACT_MARGIN_M},
        radius=radius, stepping=TET_STEPPING, log=log.getvalue(), render=render,
        size_scale=size_scale, collider_mu_by_path=by_path,
    )
