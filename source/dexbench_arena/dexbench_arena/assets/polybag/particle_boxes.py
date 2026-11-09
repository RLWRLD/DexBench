# SPDX-License-Identifier: Apache-2.0
"""Box-shaped mesh colliders the bag meets as analytic boxes (``PolybagObjectCfg.box_colliders``)."""

from __future__ import annotations

import re

import newton
import numpy as np
import warp as wp


def add_particle_boxes(builder, patterns) -> list[int]:
    """Each mesh collider of the world being built whose label matches a pattern meets the particles as an analytic
    box instead; -> the boxes added."""
    if not patterns:
        return []
    collide = int(newton.ShapeFlags.COLLIDE_PARTICLES)
    meshes = (int(newton.GeoType.MESH), int(newton.GeoType.CONVEX_MESH))
    world = builder.current_world
    added = []
    for s in range(builder.shape_count):
        label = str(builder.shape_label[s])
        if (int(builder.shape_world[s]) != world or not any(re.search(p, label) for p in patterns)
                or not int(builder.shape_flags[s]) & collide):
            continue
        if int(builder.shape_type[s]) not in meshes:
            raise ValueError(f"box collider {label}: only a mesh collider can be replaced by a box "
                             f"({newton.GeoType(int(builder.shape_type[s])).name})")
        v = np.asarray(builder.shape_source[s].vertices, dtype=np.float64) * np.asarray(builder.shape_scale[s], dtype=np.float64)
        lo, hi = v.min(axis=0), v.max(axis=0)
        corners = np.isclose(v, lo, atol=1e-5) | np.isclose(v, hi, atol=1e-5)
        if len(v) != 8 or not corners.all():
            raise ValueError(f"box collider {label}: the mesh is not a box ({len(v)} vertices)")
        frame = builder.shape_transform[s]
        centre = wp.transform_point(frame, wp.vec3(*((lo + hi) / 2.0)))
        cfg = newton.ModelBuilder.ShapeConfig(ke=float(builder.shape_material_ke[s]), kd=float(builder.shape_material_kd[s]),
                                              mu=float(builder.shape_material_mu[s]), density=0.0,
                                              has_shape_collision=False, has_particle_collision=True)
        box = builder.add_shape_box(body=int(builder.shape_body[s]), xform=wp.transform(centre, wp.transform_get_rotation(frame)),
                                    hx=float(hi[0] - lo[0]) / 2.0, hy=float(hi[1] - lo[1]) / 2.0,
                                    hz=float(hi[2] - lo[2]) / 2.0, cfg=cfg, label=f"{label}_particles")
        builder.shape_flags[s] = int(builder.shape_flags[s]) & ~collide
        added.append(box)
        print(f"[dexbench] polybag: {label} meets the bag as a {np.round(hi - lo, 4).tolist()} m box", flush=True)
    return added
