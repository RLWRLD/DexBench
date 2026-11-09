# SPDX-License-Identifier: Apache-2.0
"""A collider's surface moving under the bag while its geometry stays put: a conveyor belt (Warp and Newton only)."""

from __future__ import annotations

import re

import newton
import numpy as np
import warp as wp


def _quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate ``v`` by the inverse of the xyzw quaternion ``q``."""
    x, y, z, w = q
    u = np.array([-x, -y, -z])
    return 2.0 * np.dot(u, v) * u + (w * w - np.dot(u, u)) * v + 2.0 * w * np.cross(u, v)


def drive_surfaces(model, body_q: np.ndarray | None, velocities: dict[str, tuple[float, float, float]],
                   body_frame: dict[int, np.ndarray] | None = None) -> list[str]:
    """Give every particle-colliding mesh shape whose label matches a key (a regex) the value as its surface
    velocity [m/s, world frame]; -> the labels driven. ``body_frame``, given, receives each driven shape's
    velocity in its body's frame (the world's for a static shape), the frame contacts store it in."""
    if not velocities:
        return []
    labels = list(model.shape_label)
    kinds = model.shape_type.numpy()
    flags = model.shape_flags.numpy()
    bodies = model.shape_body.numpy()
    xforms = model.shape_transform.numpy()
    scales = model.shape_scale.numpy()
    mesh_kinds = (int(newton.GeoType.MESH), int(newton.GeoType.CONVEX_MESH))
    collide = int(newton.ShapeFlags.COLLIDE_PARTICLES)
    driven = []
    for pattern, velocity in velocities.items():
        matched = False
        for s, label in enumerate(labels):
            if not re.search(pattern, str(label)) or not flags[s] & collide:
                continue
            if int(kinds[s]) not in mesh_kinds and body_frame is None:
                raise ValueError(f"surface velocity for {label}: a {newton.GeoType(int(kinds[s])).name} shape has no "
                                 "vertex velocities; only MESH and CONVEX_MESH colliders can be driven without "
                                 "SoftContactDrive")
            v = np.asarray(velocity, dtype=np.float64)
            if bodies[s] >= 0:
                v = _quat_rotate_inverse(np.asarray(body_q[bodies[s]][3:], dtype=np.float64), v)
            if body_frame is not None:
                body_frame[s] = v.copy()
            driven.append(str(label))
            matched = True
            if int(kinds[s]) not in mesh_kinds:
                continue
            v = _quat_rotate_inverse(np.asarray(xforms[s][3:], dtype=np.float64), v) / np.asarray(scales[s], dtype=np.float64)
            mesh = model.shape_source[s].mesh
            mesh.velocities.fill_(wp.vec3(*(float(c) for c in v)))
        if not matched:
            raise ValueError(f"surface velocity: no particle-colliding shape matches {pattern!r}")
    return driven


@wp.kernel(enable_backward=False)
def _drive_soft_contacts(
    soft_contact_count: wp.array(dtype=wp.int32),
    soft_contact_shape: wp.array(dtype=wp.int32),
    shape_velocity: wp.array(dtype=wp.vec3),
    shape_driven: wp.array(dtype=wp.int32),
    soft_contact_body_vel: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    if i >= soft_contact_count[0]:
        return
    s = soft_contact_shape[i]
    if s >= 0 and shape_driven[s] != 0:
        soft_contact_body_vel[i] = shape_velocity[s]


class SoftContactDrive:
    """The driven shapes' surface velocity on every soft contact of theirs, particle, edge or face alike."""

    def __init__(self, model, body_frame: dict[int, np.ndarray]):
        velocity = np.zeros((model.shape_count, 3), dtype=np.float32)
        driven = np.zeros(model.shape_count, dtype=np.int32)
        for s, v in body_frame.items():
            velocity[s] = v
            driven[s] = 1
        self.shape_velocity = wp.array(velocity, dtype=wp.vec3, device=model.device)
        self.shape_driven = wp.array(driven, dtype=wp.int32, device=model.device)

    def __call__(self, contacts) -> None:
        """After a collision pass (graph-capturable: the launch covers the buffer, the kernel reads the count)."""
        wp.launch(_drive_soft_contacts, dim=contacts.soft_contact_shape.shape[0],
                  inputs=[contacts.soft_contact_count, contacts.soft_contact_shape, self.shape_velocity, self.shape_driven],
                  outputs=[contacts.soft_contact_body_vel], device=contacts.soft_contact_shape.device)
