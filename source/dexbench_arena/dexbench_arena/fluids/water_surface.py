# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Render the MPM water as one continuous surface instead of a cloud of beads."""

from __future__ import annotations

from typing import Callable

import isaaclab.sim as sim_utils
import numpy as np
import torch
import warp as wp
from isaaclab.utils.math import matrix_from_quat
from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

from .lathe_cavity import LatheCavity

GRID_CELLS = 128
"""Cells along each axis of the density grid, centred on the particles' median each frame."""
CELL_PER_VOXEL = 0.8
"""Grid step as a fraction of the sidecar's voxel (6.4 mm at the 8 mm voxel)."""
SPLAT_RADIUS_CELLS = 1.4
"""How far a particle's weight reaches, in cells."""
ISO_LEVEL = 0.45
"""Density the surface is drawn at."""
WATER_COLOR = (0.08, 0.35, 0.85)
SURFACE_PRIM = "/World/dexbench_water_surface"


@wp.kernel(enable_backward=False)
def _splat_density(points: wp.array(dtype=wp.vec3), origin: wp.vec3, inv_h: float, n: int,
                   field: wp.array3d(dtype=float), radius: float, reach: int):
    i = wp.tid()
    p = (points[i] - origin) * inv_h
    ci = int(wp.floor(p[0]))
    cj = int(wp.floor(p[1]))
    ck = int(wp.floor(p[2]))
    for di in range(-reach, reach + 1):
        for dj in range(-reach, reach + 1):
            for dk in range(-reach, reach + 1):
                ii = ci + di
                jj = cj + dj
                kk = ck + dk
                if ii >= 0 and ii < n and jj >= 0 and jj < n and kk >= 0 and kk < n:
                    d = wp.length(p - wp.vec3(float(ii) + 0.5, float(jj) + 0.5, float(kk) + 0.5))
                    w = wp.max(0.0, 1.0 - d / radius)
                    if w > 0.0:
                        wp.atomic_add(field, ii, jj, kk, w * w)


Cavity = tuple[np.ndarray, np.ndarray, LatheCavity]
"""(axis origin in the world, rotation container -> world, the cavity) for one container this frame."""


class WaterSurfaceRenderer:
    """A ``UsdGeom.Mesh`` of the water's surface, rebuilt from the particles every ``update``."""

    def __init__(self, stage, voxel_size: float, cavities: Callable[[], list[Cavity]] | None = None):
        self._cavities = cavities
        self.h = float(voxel_size) * CELL_PER_VOXEL
        n = GRID_CELLS
        self.field = wp.zeros((n, n, n), dtype=float)
        self.mc = wp.MarchingCubes(n, n, n, 400000, 800000)
        self.mesh = UsdGeom.Mesh.Define(stage, SURFACE_PRIM)
        self.mesh.CreateDoubleSidedAttr().Set(True)
        material_path = SURFACE_PRIM + "_material"
        material_cfg = sim_utils.MdlFileCfg(mdl_path="OmniPBR.mdl")
        material_cfg.func(material_path, material_cfg)
        shader = stage.GetPrimAtPath(material_path + "/Shader")
        shader.CreateAttribute("inputs:diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*WATER_COLOR))
        shader.CreateAttribute("inputs:reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(0.05)
        material = UsdShade.Material(stage.GetPrimAtPath(material_path))
        UsdShade.MaterialBindingAPI.Apply(self.mesh.GetPrim()).Bind(material)
        self.loose = UsdGeom.Points.Define(stage, SURFACE_PRIM + "_loose")
        UsdShade.MaterialBindingAPI.Apply(self.loose.GetPrim()).Bind(material)
        self._loose_width = 0.5 * float(voxel_size)

    def update(self, positions: np.ndarray) -> None:
        pos = np.asarray(positions, dtype=np.float32)
        pos = pos[np.isfinite(pos).all(axis=1) & (pos[:, 2] > -1.0)]
        n, h = GRID_CELLS, self.h
        if len(pos) == 0:
            self._set_mesh(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32))
            return
        origin = np.median(pos, axis=0) - 0.5 * n * h
        loose = pos[np.any((pos < origin) | (pos > origin + n * h), axis=1)]
        self.loose.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(loose))
        self.loose.GetWidthsAttr().Set(Vt.FloatArray([self._loose_width] * len(loose)))
        if len(loose):
            self.loose.GetExtentAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, loose.min(0) - self._loose_width)),
                                                          Gf.Vec3f(*map(float, loose.max(0) + self._loose_width))]))
        self.field.zero_()
        wp.launch(_splat_density, dim=len(pos),
                  inputs=[wp.array(pos, dtype=wp.vec3), wp.vec3(*origin.astype(np.float32)), 1.0 / h, n, self.field,
                          SPLAT_RADIUS_CELLS, int(np.ceil(SPLAT_RADIUS_CELLS))])
        self.mc.surface(self.field, ISO_LEVEL)
        tris = self.mc.indices.numpy().reshape(-1, 3)
        world = self.mc.verts.numpy().astype(np.float64) * h + origin
        if len(tris) and self._cavities is not None:
            world = self._hold(world, pos.astype(np.float64))
        self._set_mesh(world.astype(np.float32), tris)

    def _hold(self, world: np.ndarray, particles: np.ndarray) -> np.ndarray:
        """Each vertex goes to the container whose wall it overshoots least."""
        best = np.full(len(world), np.inf)
        target = world.copy()
        for centre, rotation, cavity in self._cavities():
            local = (world - centre) @ rotation
            held, moved = cavity.hold(local, (particles - centre) @ rotation)
            if not moved.any():
                continue
            dist = np.linalg.norm(held - local, axis=1)
            take = moved & (dist < best)
            target[take] = centre + held[take] @ rotation.T
            best[take] = dist[take]
        return target

    def _set_mesh(self, points: np.ndarray, tris: np.ndarray) -> None:
        self.mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(points)))
        self.mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(np.ascontiguousarray(tris.reshape(-1), dtype=np.int32)))
        self.mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(len(tris), 3, dtype=np.int32)))
        if len(points):
            self.mesh.GetExtentAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, points.min(0))),
                                                         Gf.Vec3f(*map(float, points.max(0)))]))


def cavity_pose(position, quat_xyzw, axis_xy=(0.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    """World origin of a container's cavity axis and its rotation, from the body pose (xyzw quaternion)."""
    rotation = matrix_from_quat(torch.as_tensor(quat_xyzw, dtype=torch.float64)).numpy()
    return np.asarray(position, np.float64) + rotation @ np.array([axis_xy[0], axis_xy[1], 0.0]), rotation
