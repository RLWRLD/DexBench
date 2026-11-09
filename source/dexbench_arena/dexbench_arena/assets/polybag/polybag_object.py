# SPDX-License-Identifier: Apache-2.0
"""The VBD polybag as an Isaac Lab asset on Newton."""

from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

from isaaclab.assets.deformable_object.base_deformable_object import BaseDeformableObject
from isaaclab.sim.utils import clone, create_prim
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils.math import combine_frame_transforms, quat_apply
from isaaclab_newton.assets.mpm_object.mpm_object import MPMObject
from isaaclab_newton.physics import NewtonManager as SimulationManager
from isaaclab_arena.assets.asset_cache import get_arena_asset_cache_dir

from dexbench_arena.environments.manipulation.newton_build_scope import register_build_cleanup

from .canonical import usd_deformable
from .parking import set_particles_active
from .particle_boxes import add_particle_boxes
from .template import FILM_SIM, PolybagTemplate, load_cfg_template, polybag_particle_masses
from .tet_template import _parts, is_tet_polybag

if TYPE_CHECKING:
    from .polybag_object_cfg import PolybagObjectCfg, PolybagVisualCfg

logger = logging.getLogger(__name__)

FILM_PRIM_NAME = "Film"
RENDER_PRIM_NAME = "Render"
LOOKS_PRIM_NAME = "Looks"


@dataclass(eq=False)
class PolybagRegistryEntry:
    """One polybag asset, as the builder hook and the solver set-up need it."""

    cfg: PolybagObjectCfg
    template: PolybagTemplate
    particle_offsets: list[int] = field(default_factory=list)
    triangle_offsets: list[int] = field(default_factory=list)
    edge_offsets: list[int] = field(default_factory=list)
    particles_per_object: int = 0

    def seal_springs(self) -> np.ndarray:
        """(k, 2) seal-spring particle pairs of every copy, model indices."""
        seal = self.template.seal
        return np.concatenate([seal + p0 for p0 in self.particle_offsets]) if self.particle_offsets else seal[:0]

    def exclusion_maps(self) -> tuple[dict, dict]:
        """SolverVBD's vertex-triangle and edge-edge exclusion maps over every copy, model indices."""
        vt, ee = {}, {}
        for p0, t0, e0 in zip(self.particle_offsets, self.triangle_offsets, self.edge_offsets):
            a, b = self.template.world_maps(p0, t0, e0)
            vt.update(a)
            ee.update(b)
        return vt, ee

    def particle_ranges(self) -> list[np.ndarray]:
        """Each copy's particle indices (one bag per range, for the bag-frame truncation)."""
        n = self.particles_per_object
        return [np.arange(p0, p0 + n) for p0 in self.particle_offsets]


def registered_polybags() -> list[PolybagRegistryEntry]:
    """The polybag entries of the scene being built (empty when there is none)."""
    return getattr(SimulationManager, "_polybag_registry", [])


def register_polybag(entry: PolybagRegistryEntry) -> None:
    """Add a polybag to the scene being built, and the builder hook that puts the scene's bags into the model."""
    if add_registered_polybags_to_builder not in SimulationManager._per_world_builder_hooks:
        SimulationManager._polybag_registry = []
        _render_bindings.clear()
        SimulationManager._per_world_builder_hooks.append(add_registered_polybags_to_builder)
    SimulationManager._polybag_registry.append(entry)
    register_build_cleanup(_forget_registered_polybags)


def _forget_registered_polybags() -> None:
    """Drop the polybags the previous build registered (newton_build_scope runs this when the next build begins)."""
    registered_polybags().clear()
    if add_registered_polybags_to_builder in SimulationManager._per_world_builder_hooks:
        SimulationManager._per_world_builder_hooks.remove(add_registered_polybags_to_builder)


def _placed_rest(entry: PolybagRegistryEntry, env_position, env_rotation) -> np.ndarray:
    """The template's authored particle positions at the asset's initial pose in one environment [m]."""
    rest = torch.as_tensor(entry.template.rest, dtype=torch.float64)
    env_t = torch.tensor([env_position], dtype=torch.float64)
    env_q = torch.tensor([env_rotation], dtype=torch.float64)
    init_t = torch.tensor([entry.cfg.init_state.pos], dtype=torch.float64)
    init_q = torch.tensor([entry.cfg.init_state.rot], dtype=torch.float64)
    t, q = combine_frame_transforms(env_t, env_q, init_t, init_q)
    return (quat_apply(q.expand(len(rest), 4), rest) + t).numpy()


def add_polybag_entry_to_builder(builder, entry: PolybagRegistryEntry, env_idx: int, env_position, env_rotation) -> None:
    """Copy the template bag into the builder's current world at the asset's initial pose."""
    if env_idx == 0:
        entry.particle_offsets.clear()
        entry.triangle_offsets.clear()
        entry.edge_offsets.clear()
        entry.particles_per_object = 0
    p0, t0, e0 = builder.particle_count, builder.tri_count, len(builder.edge_indices)
    builder.add_builder(entry.template.builder)
    placed = _placed_rest(entry, env_position, env_rotation)
    builder.particle_q[p0 : p0 + len(placed)] = [wp.vec3(*map(float, p)) for p in placed]
    if entry.cfg.mass_kg is not None:
        builder.particle_mass[p0 : p0 + len(placed)] = polybag_particle_masses(entry.template, entry.cfg.mass_kg)
    entry.particle_offsets.append(p0)
    entry.triangle_offsets.append(t0)
    entry.edge_offsets.append(e0)
    delta = builder.particle_count - p0
    if env_idx == 0:
        entry.particles_per_object = delta
    elif delta != entry.particles_per_object:
        raise RuntimeError(f"polybag '{entry.cfg.prim_path}' gave {delta} particles in env {env_idx}, "
                           f"{entry.particles_per_object} in env 0")


def add_registered_polybags_to_builder(builder, world_idx: int, env_position, env_rotation) -> None:
    """Per-world builder hook: every registered polybag into this world, and the world's ``box_colliders``."""
    entries = registered_polybags()
    for entry in entries:
        add_polybag_entry_to_builder(builder, entry, world_idx, env_position, env_rotation)
    add_particle_boxes(builder, sorted({p for e in entries for p in e.cfg.box_colliders}))


@clone
def spawn_polybag_visual(
    prim_path: str,
    cfg: PolybagVisualCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs: object,
) -> Usd.Prim:
    """An Xform at the bag's pose with the film's simulation mesh under it, authored in the bag's frame."""
    if is_tet_polybag(cfg.usd_path):
        return _spawn_tet_render(prim_path, cfg, translation, orientation)
    source = Usd.Stage.Open(str(cfg.usd_path))
    film_prim = source.GetPrimAtPath(FILM_SIM)
    points = usd_deformable._world_points(film_prim)
    indices = np.asarray(UsdGeom.Mesh(film_prim).GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    prim = create_prim(prim_path, prim_type="Xform", translation=translation, orientation=orientation)
    mesh = UsdGeom.Mesh.Define(prim.GetStage(), f"{prim_path}/{FILM_PRIM_NAME}")
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in points]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * (len(indices) // 3)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(indices.tolist()))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, cfg.color))]))
    return prim


def _spawn_tet_render(prim_path: str, cfg: PolybagVisualCfg, translation, orientation) -> Usd.Prim:
    """A tet bag's drawing: an Xform at the bag's pose with the asset's textured mesh and its material under it."""
    source = Usd.Stage.Open(str(cfg.usd_path))
    _, _, mesh, look = _parts(source)
    prim = create_prim(prim_path, prim_type="Xform", translation=translation, orientation=orientation)
    stage = prim.GetStage()
    looks = stage.DefinePrim(f"{prim_path}/{LOOKS_PRIM_NAME}")
    looks.GetReferences().AddReference(str(cfg.usd_path), look.GetPath())
    layer = stage.GetEditTarget().GetLayer()
    render_path = Sdf.Path(f"{prim_path}/{RENDER_PRIM_NAME}")
    if not Sdf.CopySpec(mesh.GetPrimStack()[0].layer, mesh.GetPath(), layer, render_path):
        raise RuntimeError(f"could not copy {mesh.GetPath()} of {cfg.usd_path} to {render_path}")
    spec = layer.GetPrimAtPath(render_path)
    for name in ("normals", "primvars:normals", "material:binding"):
        if name in spec.properties:
            spec.RemoveProperty(spec.properties[name])
    render = stage.GetPrimAtPath(render_path)
    UsdShade.MaterialBindingAPI.Apply(render).Bind(UsdShade.Material(looks))
    if cfg.diffuse_texture is not None:
        _albedo_input(looks, cfg.usd_path).Set(Sdf.AssetPath(str(Path(cfg.diffuse_texture).resolve(strict=True))))
    if cfg.diffuse_tint_rgb is not None:
        _tint_looks(stage, looks, cfg.usd_path, cfg.diffuse_tint_rgb)
    return prim


TINT_MIN_SATURATION = 0.12
TINT_MAX_HUE_DISTANCE = 0.08


def tint_albedo(rgb: np.ndarray, tint_rgb: tuple[int, int, int]) -> np.ndarray:
    """``rgb`` (h, w, 3 uint8 sRGB) with the film's pixels in ``tint_rgb``, each keeping its brightness relative
    to the film's mean, so the folds and the weave stay; every other pixel unchanged."""
    image = rgb.astype(np.float64) / 255.0
    high, low = image.max(axis=-1), image.min(axis=-1)
    saturated = high - low >= TINT_MIN_SATURATION * np.maximum(high, 1e-9)
    if not saturated.any():
        raise ValueError("the texture has no saturated pixels: no film to tint")
    r, g, b = image[..., 0], image[..., 1], image[..., 2]
    hue = np.arctan2(np.sqrt(3.0) * (g - b), 2.0 * r - g - b)
    film_hue = np.angle(np.exp(1j * hue[saturated]).mean())
    film = saturated & (np.abs(np.angle(np.exp(1j * (hue - film_hue)))) <= 2.0 * np.pi * TINT_MAX_HUE_DISTANCE)
    luminance = image @ np.array([0.2126, 0.7152, 0.0722])
    relative = luminance / max(float(luminance[film].mean()), 1e-9)
    tinted = np.clip(np.asarray(tint_rgb, dtype=np.float64) / 255.0 * relative[..., None], 0.0, 1.0)
    return np.round(np.where(film[..., None], tinted, image) * 255.0).astype(np.uint8)


def _albedo_input(looks, usd_path: str) -> UsdShade.Input:
    """The ``file`` input of the texture the referenced material's diffuseColor reads."""
    surface = UsdShade.Material(looks).ComputeSurfaceSource()[0]
    sources = surface.GetInput("diffuseColor").GetConnectedSources()[0] if surface else []
    if not sources:
        raise RuntimeError(f"{usd_path}: its material's diffuseColor reads no texture")
    return UsdShade.Shader(sources[0].source.GetPrim()).GetInput("file")


def _tint_looks(stage, looks, usd_path: str, tint_rgb: tuple[int, int, int]) -> None:
    """Point the referenced material's albedo texture at a recoloured copy in Arena's asset cache."""
    r, g, b = (int(v) for v in tint_rgb)
    if not all(0 <= v <= 255 for v in (r, g, b)):
        raise ValueError(f"diffuse_tint_rgb {tint_rgb} is not three 0-255 values")
    file_input = _albedo_input(looks, usd_path)
    source = Path(file_input.Get().resolvedPath)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    target = get_arena_asset_cache_dir() / "polybag_tint" / f"{source.stem}_{digest}_{r:02x}{g:02x}{b:02x}.png"
    if not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        tinted = tint_albedo(np.asarray(Image.open(source).convert("RGB")), (r, g, b))
        partial = target.with_name(f"{target.stem}.{os.getpid()}.tmp.png")
        Image.fromarray(tinted).save(partial)
        partial.replace(target)
    file_input.Set(Sdf.AssetPath(str(target)))


@wp.kernel(enable_backward=False)
def _remap_render_points(
    points: wp.fabricarrayarray(dtype=wp.vec3f),
    world_matrices: wp.fabricarray(dtype=wp.mat44d),
    slots: wp.fabricarray(dtype=wp.uint32),
    particle_offsets: wp.array(dtype=wp.int32),
    particle_q: wp.array(dtype=wp.vec3),
    tet_vertex_indices: wp.array2d(dtype=wp.int32),
    bary_weights: wp.array2d(dtype=wp.float32),
):
    """Each textured-mesh vertex at the barycentric point of its tet, in the mesh prim's local frame
    (Fabric stores the world matrix transposed, as Isaac Lab's ``_sync_particle_points`` reads it)."""
    i, j = wp.tid()
    o = particle_offsets[int(slots[i])]
    x = (bary_weights[j, 0] * particle_q[o + tet_vertex_indices[j, 0]] + bary_weights[j, 1] * particle_q[o + tet_vertex_indices[j, 1]]
         + bary_weights[j, 2] * particle_q[o + tet_vertex_indices[j, 2]] + bary_weights[j, 3] * particle_q[o + tet_vertex_indices[j, 3]])
    inv = wp.inverse(wp.transpose(wp.mat44f(world_matrices[i])))
    points[i][j] = wp.transform_point(inv, x)


@dataclass(eq=False)
class _RenderBinding:
    """The textured meshes of one tet polybag asset, tagged in Fabric with ``attr`` = their environment's slot."""

    attr: str
    count: int
    particle_offsets: wp.array
    tet_vertex_indices: wp.array
    bary_weights: wp.array


_render_bindings: list[_RenderBinding] = []


def sync_polybag_renders() -> None:
    """Write every tet polybag's textured mesh from its particles (before each rendered frame)."""
    if not _render_bindings or SimulationManager._usdrt_stage is None:
        return
    state = SimulationManager.get_state_0()
    if state is None or state.particle_q is None:
        return
    import usdrt

    for b in _render_bindings:
        selection = SimulationManager._usdrt_stage.SelectPrims(
            require_attrs=[(usdrt.Sdf.ValueTypeNames.Point3fArray, "points", usdrt.Usd.Access.ReadWrite),
                           (usdrt.Sdf.ValueTypeNames.UInt, b.attr, usdrt.Usd.Access.Read),
                           (usdrt.Sdf.ValueTypeNames.Matrix4d, "omni:fabric:worldMatrix", usdrt.Usd.Access.Read)],
            device=str(state.particle_q.device))
        if selection.GetCount() == 0:
            continue
        wp.launch(_remap_render_points, dim=(selection.GetCount(), b.count),
                  inputs=[wp.fabricarrayarray(data=selection, attrib="points", dtype=wp.vec3f),
                          wp.fabricarray(data=selection, attrib="omni:fabric:worldMatrix"),
                          wp.fabricarray(data=selection, attrib=b.attr), b.particle_offsets, state.particle_q,
                          b.tet_vertex_indices, b.bary_weights], device=state.particle_q.device)


class PolybagObject(MPMObject):
    """A Newton VBD polybag: film (cloth) and cotton (tet soft body), built by the canonical loader."""

    cfg: PolybagObjectCfg

    def __init__(self, cfg: PolybagObjectCfg):
        BaseDeformableObject.__init__(self, cfg)
        template = load_cfg_template(cfg)
        self._registry_entry = PolybagRegistryEntry(cfg=cfg, template=template)
        register_polybag(self._registry_entry)
        self._physics_ready_handle = None

    @property
    def template(self) -> PolybagTemplate:
        return self._registry_entry.template

    @property
    def particle_offsets(self) -> list[int]:
        """Where each environment's copy of this bag starts in the Newton model's particle arrays, by environment
        (model indices; ``particles_per_object`` long each). Filled while Newton replicates the scene."""
        return list(self._registry_entry.particle_offsets)

    def write_rest_pose_to_sim_index(self, pos_w: torch.Tensor, quat_w: torch.Tensor, env_ids: torch.Tensor) -> None:
        """Put the bags of ``env_ids`` back in their authored rest shape at a world pose, at rest."""
        rest = torch.as_tensor(self.template.rest, dtype=torch.float32, device=self.device)
        n, count = len(env_ids), len(rest)
        points = quat_apply(quat_w[:, None, :].expand(n, count, 4), rest[None].expand(n, count, 3)) + pos_w[:, None, :]
        state = torch.cat((points, torch.zeros_like(points)), dim=-1)
        self.write_nodal_state_to_sim_index(state, env_ids=env_ids)

    def write_active_to_sim_index(self, active: bool, env_ids: torch.Tensor | None = None) -> None:
        """Take the bags of ``env_ids`` (all when None) into or out of the simulation (``parking.py``)."""
        if env_ids is None:
            ids = self._ALL_INDICES
        else:
            ids = wp.from_torch(torch.as_tensor(env_ids, device=self.device).to(torch.int32).flatten(), dtype=wp.int32)
        set_particles_active(SimulationManager.get_model().particle_flags, self._particle_offsets, ids,
                             self.particles_per_object, active)

    def _create_particle_visualization(self) -> None:
        """Bind each environment's film mesh to that copy's film particles in Fabric, or tag its textured mesh
        for ``sync_polybag_renders`` (a tet bag)."""
        if not self.cfg.spawn.visible:
            return
        if self.template.render is not None:
            self._bind_render_meshes()
            return
        film = self.template.film
        paths = [f"{p}/{FILM_PRIM_NAME}" for p in _instance_paths(self.cfg.prim_path, self._num_instances)]
        bound = bind_meshes_to_particles(paths, [offset + film.start for offset in self.particle_offsets],
                                         film.stop - film.start)
        logger.info("Polybag film meshes bound to Newton particles: %d of %d", bound, len(paths))

    def _bind_render_meshes(self) -> None:
        """Tag each environment's textured mesh in Fabric with its slot and register the remap tables."""
        import usdrt

        if SimulationManager._usdrt_stage is None:
            SimulationManager._usdrt_stage = get_current_stage(fabric=True)
        fabric = SimulationManager._usdrt_stage
        if fabric is None:
            logger.warning("polybag: no Fabric stage; the bag is not drawn")
            return
        render = self.template.render
        attr = f"dexbench:polybagRender{id(self._registry_entry)}"
        paths = [f"{p}/{RENDER_PRIM_NAME}" for p in _instance_paths(self.cfg.prim_path, self._num_instances)]
        bound = 0
        for slot, path in enumerate(paths):
            prim = fabric.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                logger.warning("polybag: Fabric prim %s not found", path)
                continue
            prim.CreateAttribute(attr, usdrt.Sdf.ValueTypeNames.UInt, True).Set(slot)
            bound += 1
        device = SimulationManager.get_state_0().particle_q.device
        self._render_binding = _RenderBinding(
            attr=attr, count=len(render.points),
            particle_offsets=wp.array(np.asarray(self.particle_offsets, dtype=np.int32), dtype=wp.int32, device=device),
            tet_vertex_indices=wp.array(render.tet_vertex_indices, dtype=wp.int32, device=device),
            bary_weights=wp.array(render.bary_weights, dtype=wp.float32, device=device))
        _render_bindings.append(self._render_binding)
        sync_polybag_renders()
        logger.info("Polybag textured meshes bound to Newton particles: %d of %d", bound, len(paths))

    def _clear_callbacks(self) -> None:
        BaseDeformableObject._clear_callbacks(self)
        binding = getattr(self, "_render_binding", None)
        if binding is not None and binding in _render_bindings:
            _render_bindings.remove(binding)
        if self._physics_ready_handle is not None:
            self._physics_ready_handle.deregister()
            self._physics_ready_handle = None
        registry = registered_polybags()
        if self._registry_entry in registry:
            registry.remove(self._registry_entry)
        if not registry and add_registered_polybags_to_builder in SimulationManager._per_world_builder_hooks:
            SimulationManager._per_world_builder_hooks.remove(add_registered_polybags_to_builder)


def _instance_paths(prim_path: str, count: int) -> list[str]:
    """The asset's prim path in each environment (``env_.*`` -> ``env_0`` ...)."""
    paths = []

    for i in range(count):
        p = re.sub(r"(?<=[Ee]nv_)(?:\[\^/\][*+]|\.\*)", str(i), prim_path)
        paths.append(re.sub(r"\[\^/\][*+]|\.\*", str(i), p))
    return paths


def bind_meshes_to_particles(mesh_paths: list[str], offsets: list[int], count: int) -> int:
    """Tag Fabric mesh prims with the particle slice whose positions Newton writes into their points."""
    import usdrt

    if SimulationManager._usdrt_stage is None:
        SimulationManager._usdrt_stage = get_current_stage(fabric=True)
    fabric = SimulationManager._usdrt_stage
    if fabric is None:
        logger.warning("polybag: no Fabric stage; the film is not drawn")
        return 0
    bound = 0
    for path, offset in zip(mesh_paths, offsets):
        prim = fabric.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            logger.warning("polybag: Fabric prim %s not found", path)
            continue
        prim.CreateAttribute(SimulationManager._newton_particle_offset_attr, usdrt.Sdf.ValueTypeNames.UInt, True).Set(int(offset))
        prim.CreateAttribute(SimulationManager._newton_particle_count_attr, usdrt.Sdf.ValueTypeNames.UInt, True).Set(int(count))
        bound += 1
    if bound:
        SimulationManager._mark_particles_dirty()
        SimulationManager.sync_particles_to_usd()
    return bound
