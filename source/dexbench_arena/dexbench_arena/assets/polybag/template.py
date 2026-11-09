# SPDX-License-Identifier: Apache-2.0
"""One polybag, built once the way its asset says, ready to copy into every Newton world."""

from __future__ import annotations

import contextlib
import copy
import inspect
import io
from dataclasses import dataclass, field
from functools import lru_cache

import newton
import numpy as np
from pxr import Sdf, Usd

from .canonical import loader, newton_scene, setups, usd_deformable

CONTENTS_MATERIAL = "/Polybag/Contents/PhysicsMaterial"
FILM_MATERIAL = "/Polybag/Film/PhysicsMaterial"
FILM_SIM = "/Polybag/Film/Simulation"
CONTENTS_SIM = "/Polybag/Contents/Simulation"
EXCLUSION_KWARGS = ("particle_external_vertex_contact_filtering_map", "particle_external_edge_contact_filtering_map")


def lame(youngs_modulus: float, poisson_ratio: float) -> tuple[float, float]:
    """Young's modulus [Pa] and Poisson ratio -> Lame (mu, lambda) [Pa]."""
    e, nu = float(youngs_modulus), float(poisson_ratio)
    return e / (2.0 * (1.0 + nu)), e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))


def material_values(stage, cotton_youngs_modulus=None, cotton_poisson_ratio=None, film_bend_scale=None) -> dict:
    """-> {"prim.attr": value} for the changes asked, with every value the asset restates kept equal."""
    out = {}
    if cotton_youngs_modulus is not None or cotton_poisson_ratio is not None:
        mat = stage.GetPrimAtPath(CONTENTS_MATERIAL)
        e = float(mat.GetAttribute("physics:youngsModulus").Get() if cotton_youngs_modulus is None else cotton_youngs_modulus)
        nu = float(mat.GetAttribute("physics:poissonsRatio").Get() if cotton_poisson_ratio is None else cotton_poisson_ratio)
        if not 0.0 <= nu < 0.5:
            raise ValueError(f"cotton_poisson_ratio {nu} is outside [0, 0.5)")
        mu, lam = lame(e, nu)
        out.update({f"{CONTENTS_MATERIAL}.physics:youngsModulus": e, f"{CONTENTS_MATERIAL}.physics:poissonsRatio": nu,
                    f"{CONTENTS_MATERIAL}.newton:kMu": mu, f"{CONTENTS_MATERIAL}.newton:kLambda": lam})
    if film_bend_scale is not None and float(film_bend_scale) != 1.0:
        mat = stage.GetPrimAtPath(FILM_MATERIAL)
        for name in ("newton:edgeKe", "physics:bendStiffness", "newton:edgeKd"):
            out[f"{FILM_MATERIAL}.{name}"] = float(mat.GetAttribute(name).Get()) * float(film_bend_scale)
    return out


def changed_layer(usd_path: str, values: dict, tokens: dict | None = None):
    """An anonymous layer that sublayers `usd_path` and sets `values` over it (kept alive by the caller)."""

    src = Usd.Stage.Open(usd_path)
    layer = Sdf.Layer.CreateAnonymous("polybag_material.usda")
    layer.subLayerPaths.append(usd_path)
    stage = Usd.Stage.Open(layer)
    for name in ("upAxis", "metersPerUnit", "kilogramsPerUnit"):
        if src.HasAuthoredMetadata(name):
            stage.SetMetadata(name, src.GetMetadata(name))
    layer.defaultPrim = src.GetDefaultPrim().GetName()
    for path, value in values.items():
        prim_path, name = path.split(".", 1)
        prim = stage.GetPrimAtPath(prim_path)
        attr = prim.GetAttribute(name) if prim else None
        if not attr or not attr.HasAuthoredValue():
            raise ValueError(f"{usd_path}: {path} is not authored; only values the asset states can be changed")
        attr.Set(type(attr.Get())(value))
    for path, value in (tokens or {}).items():
        prim_path, name = path.split(".", 1)
        stage.GetPrimAtPath(prim_path).CreateAttribute(name, Sdf.ValueTypeNames.Token).Set(str(value))
    return layer


@dataclass
class PolybagTemplate:
    """The built bag and what a scene needs to place it and to configure its solver."""

    usd_path: str
    builder: object
    rest: np.ndarray
    film: slice
    contents: slice
    film_tris: np.ndarray
    seal: np.ndarray
    vertex_triangle_exclusions: dict
    edge_edge_exclusions: dict
    contact: dict
    fixture: dict
    solver_kwargs: dict
    radius: float
    stepping: tuple[int, int]
    material: dict = field(default_factory=dict)
    log: str = ""
    render: object | None = None
    fixture_on_bodies: bool = False
    size_scale: tuple = (1.0, 1.0, 1.0)
    collider_mu_by_path: tuple = ()

    @property
    def particle_count(self) -> int:
        return len(self.rest)

    def contact_margin(self, substeps: int, fps: float, fall_height: float) -> float:
        """The soft-contact band [m] for this stepping: wide enough that a particle falling `fall_height`
        cannot cross a collider between two substeps (the loader's ``newton_scene.contact_margin``)."""

        return float(newton_scene.contact_margin(self.radius, substeps, fps, drop=fall_height))

    def world_maps(self, particle_offset: int, triangle_offset: int, edge_offset: int) -> tuple[dict, dict]:
        """The exclusion maps of a copy whose particles, triangles and bending edges start at these offsets."""
        vt = {int(k) + particle_offset: {int(v) + triangle_offset for v in vs} for k, vs in self.vertex_triangle_exclusions.items()}
        ee = {int(k) + edge_offset: {int(v) + edge_offset for v in vs} for k, vs in self.edge_edge_exclusions.items()}
        return vt, ee


@contextlib.contextmanager
def _recording_solver(sink: dict):
    """SolverVBD replaced by a recorder while the loader picks its arguments; its signature is kept,
    because the loader asks it whether the exclusion maps are taken."""
    real = newton.solvers.SolverVBD

    class _Recorder:
        def __init__(self, model, **kwargs):
            sink.update(kwargs)

    _Recorder.__init__.__signature__ = inspect.signature(real.__init__)
    newton.solvers.SolverVBD = _Recorder
    try:
        yield
    finally:
        newton.solvers.SolverVBD = real


FILM_AND_COTTON_MATERIAL_FIELDS = ("cotton_youngs_modulus", "cotton_poisson_ratio", "film_bend_scale", "film_rest_bend_angles")
TET_MATERIAL_FIELDS = ("tet_youngs_modulus", "tet_poisson_ratio", "tet_k_damp", "tet_contact_ke", "tet_contact_kd",
                       "tet_contact_mu", "tet_collider_ke", "tet_collider_kd", "tet_collider_mu", "tet_collider_mu_by_path",
                       "tet_size_scale")


def polybag_particle_masses(template: PolybagTemplate, mass_kg: float) -> list[float]:
    """The template's particle masses scaled by one factor so they sum to ``mass_kg``. ValueError for a mass
    that is not positive, or a template without mass to scale."""
    if not mass_kg > 0.0:
        raise ValueError(f"a polybag's mass_kg must be positive, got {mass_kg!r}")
    masses = [float(m) for m in template.builder.particle_mass]
    total = sum(masses)
    if not total > 0.0:
        raise ValueError("the polybag template's particles carry no mass to scale")
    return [m * mass_kg / total for m in masses]


def load_cfg_template(cfg) -> PolybagTemplate:
    """The template of a ``PolybagObjectCfg``: its asset with its material fields."""
    fields = {k: getattr(cfg, k) for k in (*FILM_AND_COTTON_MATERIAL_FIELDS, *TET_MATERIAL_FIELDS)}
    if fields["tet_collider_mu_by_path"] is not None:
        fields["tet_collider_mu_by_path"] = tuple((str(p), float(mu)) for p, mu in fields["tet_collider_mu_by_path"])
    return load_template(str(cfg.spawn.usd_path), **fields)


@lru_cache(maxsize=8)
def load_template(usd_path: str, *, cotton_youngs_modulus: float | None = None, cotton_poisson_ratio: float | None = None,
                  film_bend_scale: float | None = None, film_rest_bend_angles: str | None = None,
                  tet_youngs_modulus: float | None = None, tet_poisson_ratio: float | None = None,
                  tet_k_damp: float | None = None, tet_contact_ke: float | None = None,
                  tet_contact_kd: float | None = None, tet_contact_mu: float | None = None,
                  tet_collider_ke: float | None = None, tet_collider_kd: float | None = None,
                  tet_collider_mu: float | None = None,
                  tet_collider_mu_by_path: tuple[tuple[str, float], ...] | None = None,
                  tet_size_scale: tuple[float, float, float] | None = None) -> PolybagTemplate:
    """Build the polybag at `usd_path` once per material (None keeps the asset's value)."""
    from .tet_template import is_tet_polybag, load_tet_template

    arguments = locals()
    film_and_cotton = {k: arguments[k] for k in FILM_AND_COTTON_MATERIAL_FIELDS if arguments[k] is not None}
    tet = {k: arguments[k] for k in TET_MATERIAL_FIELDS if arguments[k] is not None}
    if is_tet_polybag(usd_path):
        if film_and_cotton:
            raise ValueError(f"{usd_path} is one tet soft body; {sorted(film_and_cotton)} apply to the film and cotton bag only")
        return load_tet_template(usd_path, **{k.removeprefix("tet_"): v for k, v in tet.items()})
    if tet:
        raise ValueError(f"{usd_path} is the film and cotton bag; {sorted(tet)} apply to the tet bag only")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        source = Usd.Stage.Open(usd_path)
        values = material_values(source, cotton_youngs_modulus, cotton_poisson_ratio, film_bend_scale)
        if film_rest_bend_angles not in (None, "restShape", "flat"):
            raise ValueError(f"film_rest_bend_angles {film_rest_bend_angles!r}: one of 'restShape', 'flat', None")
        tokens = {f"{FILM_SIM}.physics:restBendAnglesDefault": film_rest_bend_angles} if film_rest_bend_angles else {}
        layer = changed_layer(usd_path, values, tokens) if values or tokens else None
        asset = layer.identifier if layer is not None else usd_path
        setup, canon_substeps, canon_iterations = newton_scene.resolve_stepping(asset, setups.CANON, 60.0, tag="polybag")
        loaded = loader.load(asset, setup, tag="polybag")
        builder = loaded.builder
        stage = loaded.stage
        film_index = usd_deformable.builder_index(builder, stage.GetPrimAtPath(FILM_SIM))
        contents_index = usd_deformable.builder_index(builder, stage.GetPrimAtPath(CONTENTS_SIM))
        margin = 2.0 * loaded.radius
        ref = copy.deepcopy(builder)
        ground = ref.shape_count
        ref.add_ground_plane()
        loader.colour_for_vbd(ref, loaded.seal)
        model = ref.finalize()
        loader.configure(loaded, model, fixtures=[ground])
        kwargs: dict = {}
        with _recording_solver(kwargs):
            loader.vbd_solver(loaded, model, 1, margin)
    contact = {k: float(getattr(model, k)) for k in
               ("soft_contact_ke", "soft_contact_kd", "soft_contact_mu", "soft_contact_restitution")}
    fixture = {k: float(getattr(model, k).numpy()[ground]) for k in
               ("shape_material_ke", "shape_material_kd", "shape_material_mu")}
    film, contents = _contiguous(film_index, "film"), _contiguous(contents_index, "contents")
    tris = np.asarray(builder.tri_indices, dtype=np.int64).reshape(-1, 3)
    film_tris = tris[(tris >= film.start).all(axis=1) & (tris < film.stop).all(axis=1)] - film.start
    vt, ee = loaded.exclusions
    return PolybagTemplate(
        usd_path=usd_path, builder=builder, rest=np.asarray(builder.particle_q, dtype=np.float64),
        film=film, contents=contents, film_tris=film_tris, seal=np.asarray(loaded.seal, dtype=np.int64).reshape(-1, 2),
        vertex_triangle_exclusions=dict(vt), edge_edge_exclusions=dict(ee), contact=contact, fixture=fixture,
        solver_kwargs={k: v for k, v in kwargs.items() if k not in (*EXCLUSION_KWARGS, "iterations")},
        radius=float(loaded.radius), stepping=(int(canon_substeps), int(canon_iterations)),
        material=values, log=out.getvalue(),
    )


def _contiguous(index: np.ndarray, what: str) -> slice:
    """The particle range a body occupies; the render binding needs it in authored order."""
    start = int(index[0])
    if not np.array_equal(index, np.arange(start, start + len(index))):
        raise ValueError(f"the {what}'s particles are not one contiguous range in its authored order")
    return slice(start, start + len(index))
