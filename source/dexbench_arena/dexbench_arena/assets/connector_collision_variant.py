# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Apply a validated CAD-preserving S2 collision hull variant to the live USD stage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from dexbench_arena import REPO_ROOT

VARIANT_DIR = Path(__file__).with_name("connector_collision_variants")
"""The variants this package ships, ``<name>.json``."""


def collision_variant_path(variant: str) -> Path:
    """A variant's JSON: a name in :data:`VARIANT_DIR`, or a ``.json`` path (absolute, or under the checkout)."""
    path = Path(variant)
    if path.suffix != ".json":
        path = VARIANT_DIR / f"{variant}.json"
    elif not path.is_absolute():
        path = REPO_ROOT / path if (REPO_ROOT / path).is_file() else VARIANT_DIR / path.name
    if not path.is_file():
        raise FileNotFoundError(f"No S2 collision variant {variant!r} ({path})")
    return path


def apply_connector_collision_variant(stage, root_path: str, variant_path: str) -> dict:
    from pxr import Gf, UsdGeom, Vt

    path = collision_variant_path(variant_path)
    payload = path.read_bytes()
    spec = json.loads(payload)
    prefix = "/DBM2_Liner2"
    relative_root = "/Task/Connector_S2"
    expected = {prefix + relative_root + "/Collision/" + name
                for name in ("Wall_e1neg", "Wall_e1pos", "Wall_e2neg", "Wall_e2pos", "Back")}
    assert {row["prim_path"] for row in spec["meshes"]} == expected
    assert len(spec["meshes"]) == 5
    cache = UsdGeom.XformCache()
    housing = stage.GetPrimAtPath(root_path + relative_root)
    assert housing.IsValid(), root_path + relative_root
    validated = []
    for row in spec["meshes"]:
        live_path = root_path + row["prim_path"][len(prefix):]
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(live_path))
        assert mesh, live_path
        current = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        original = np.asarray(row["expected_original_points_mesh_local_m"], dtype=float)
        assert current.shape == original.shape and np.allclose(current, original, atol=2e-8, rtol=0), live_path
        transform, _ = cache.ComputeRelativeTransform(mesh.GetPrim(), housing)
        assert np.allclose(np.asarray(transform), row["source_transform_mesh_to_housing"], atol=1e-9, rtol=0), live_path
        points = np.asarray(row["points_mesh_local_m"], dtype=float)
        counts = np.asarray(row["face_vertex_counts"], dtype=int)
        indices = np.asarray(row["face_vertex_indices"], dtype=int)
        assert points.ndim == 2 and points.shape[1] == 3 and np.isfinite(points).all()
        assert np.all(counts == 3) and len(indices) == sum(counts)
        assert np.all((indices >= 0) & (indices < len(points)))
        triangles = indices.reshape(-1, 3)
        edges = np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1)
        _, multiplicity = np.unique(edges, axis=0, return_counts=True)
        assert np.all(multiplicity == 2), live_path
        volume = np.einsum("ij,ij->i", points[triangles[:, 0]],
                           np.cross(points[triangles[:, 1]], points[triangles[:, 2]])).sum() / 6
        assert volume > 0, (live_path, volume)
        validated.append((mesh, points, counts, indices))
    for mesh, points, counts, indices in validated:
        mesh.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in points]))
        mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray(counts.tolist()))
        mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(indices.tolist()))
        mesh.GetNormalsAttr().Block()
        mesh.GetExtentAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, points.min(axis=0))),
                                              Gf.Vec3f(*map(float, points.max(axis=0)))]))
    audit = {"name": spec["name"], "sha256": hashlib.sha256(payload).hexdigest(),
             "mesh_paths": [str(m.GetPath()) for m, *_ in validated], "preconditions_passed": True,
             "inertia_note": "Authored mass attributes preserved; effective COM/inertia audited separately."}
    print("M2_COLLISION_VARIANT " + json.dumps(audit), flush=True)
    return audit
