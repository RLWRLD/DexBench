# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""The robot hand's own collision hulls, read off the live stage."""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull

from .hand_spec import PaperHandSpec

ROBOT_ROOT = "/World/envs/env_0/Robot"
HULL_CELL_M = 0.003
"""Vertex clustering before hulling. PhysX cooks at most 64 hull vertices, so a
hull of a coarsened cloud is closer to what the robot collides with than a hull
of all ~30k source points."""


def hand_hull_links(hand: PaperHandSpec, body_names: list[str], side: str) -> list[str]:
    """Hand links that carry collision geometry, as the embodiment's :class:`~.hand_spec.PaperHandSpec` names them."""

    return hand.hull_links(body_names, side)


def _triangulate(counts, indices) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int64)
    if counts.size and np.all(counts == 3):
        return indices.reshape(-1, 3)
    tris, start = [], 0
    for c in counts:
        for k in range(1, c - 1):
            tris.append((indices[start], indices[start + k], indices[start + k + 1]))
        start += c
    return np.asarray(tris, dtype=np.int64).reshape(-1, 3)


def _decimate(verts: np.ndarray, faces: np.ndarray, cell: float) -> np.ndarray:
    """Vertex clustering: snap to a grid, merge, keep the vertices surviving faces use."""

    if len(faces) == 0:
        return verts
    q = np.floor(verts / cell).astype(np.int64)
    _, inv, counts = np.unique(q, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    merged = np.zeros((counts.size, 3), dtype=np.float64)
    np.add.at(merged, inv, verts)
    merged /= counts[:, None]
    f = inv[faces]
    keep = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    return merged[np.unique(f[keep])].astype(np.float32).astype(np.float64)


def convex_hull(points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """(vertices float32 (N,3), triangles uint32 (M,3)) of the points' convex hull."""

    if len(points) < 4:
        return None
    try:
        h = ConvexHull(points)
    except Exception:
        return None
    used = np.unique(h.simplices)
    remap = np.full(len(points), -1, dtype=np.int64)
    remap[used] = np.arange(used.size)
    return points[used].astype(np.float32), remap[h.simplices].astype(np.uint32)


def link_collision_hulls(
    stage, body_names: list[str], robot_root: str = ROBOT_ROOT
) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """Link name -> convex hulls of its collision meshes, in that link's own frame."""

    from pxr import Usd, UsdGeom, UsdPhysics

    xc = UsdGeom.XformCache(Usd.TimeCode.Default())
    root = stage.GetPrimAtPath(robot_root)
    assert root.IsValid(), f"no robot at {robot_root}"
    wanted = set(body_names)
    bodies = {
        p.GetName(): p
        for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
        if p.HasAPI(UsdPhysics.RigidBodyAPI) and p.GetName() in wanted
    }
    out: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for name in body_names:
        body = bodies.get(name)
        if body is None:
            continue
        inv_body = np.linalg.inv(np.asarray(xc.GetLocalToWorldTransform(body), dtype=np.float64))
        hulls = []
        it = iter(Usd.PrimRange(body, Usd.TraverseInstanceProxies()))
        for p in it:
            if p != body and p.HasAPI(UsdPhysics.RigidBodyAPI):
                it.PruneChildren()
                continue
            if not (p.IsA(UsdGeom.Mesh) and p.HasAPI(UsdPhysics.CollisionAPI)):
                continue
            mesh = UsdGeom.Mesh(p)
            pts = mesh.GetPointsAttr().Get()
            if not pts:
                continue
            v = np.asarray(pts, dtype=np.float64)
            m = np.asarray(xc.GetLocalToWorldTransform(p), dtype=np.float64)
            local = (np.hstack([v, np.ones((len(v), 1))]) @ m @ inv_body)[:, :3]
            faces = _triangulate(mesh.GetFaceVertexCountsAttr().Get() or [], mesh.GetFaceVertexIndicesAttr().Get() or [])
            hull = convex_hull(_decimate(local, faces, HULL_CELL_M))
            if hull is not None:
                hulls.append(hull)
        if hulls:
            out[name] = hulls
    return out
