# SPDX-License-Identifier: Apache-2.0
"""Where SolverVBD starts each substep's iterate: Newton's inertial point, or (contact) the bag's own motion while
it touches something."""

from __future__ import annotations

import inspect

import newton
import numpy as np
import warp as wp
from newton import ParticleFlags

from newton._src.geometry.tri_mesh_collision import (
    TriMeshCollisionInfo,
    get_vertex_colliding_triangles,
    get_vertex_colliding_triangles_count,
)
from newton.solvers import SolverVBD

INITIAL_GUESSES = ("inertia", "contact")
LOAD_SHARE = 0.25
"""A bag counts as supported when its colliders press it with at least this share of its weight."""
_ACTIVE = wp.constant(int(ParticleFlags.ACTIVE))


@wp.kernel
def _bag_touch(contact_count: wp.array(dtype=wp.int32), contact_max: int, contact_indices: wp.array(dtype=wp.vec3i),
               contact_barycentric: wp.array(dtype=wp.vec3), contact_shape: wp.array(dtype=wp.int32),
               contact_body_pos: wp.array(dtype=wp.vec3), contact_normal: wp.array(dtype=wp.vec3),
               shape_body: wp.array(dtype=wp.int32), shape_margin: wp.array(dtype=float),
               body_q: wp.array(dtype=wp.transform), pos: wp.array(dtype=wp.vec3), radius: wp.array(dtype=float),
               particle_bag: wp.array(dtype=wp.int32), soft_ke: float, shape_ke: wp.array(dtype=float),
               load: wp.array(dtype=float)):
    """Each bag's colliders' elastic normal force at the substep's start, summed over its records (the geometry and
    stiffness of ``_eval_soft_ef_contact``: the barycentric point of the record's corners, the largest corner radius,
    the bag's and the shape's stiffness averaged)."""
    tid = wp.tid()
    if tid >= wp.min(contact_count[0], contact_max):
        return
    corners = contact_indices[tid]
    v0 = corners[0]
    if v0 < 0:
        return
    b = particle_bag[v0]
    if b < 0:
        return
    bary = contact_barycentric[tid]
    x = bary[0] * pos[v0]
    r = radius[v0]
    if corners[1] >= 0:
        x += bary[1] * pos[corners[1]]
        r = wp.max(r, radius[corners[1]])
    if corners[2] >= 0:
        x += bary[2] * pos[corners[2]]
        r = wp.max(r, radius[corners[2]])
    shape = contact_shape[tid]
    body = shape_body[shape]
    X = wp.transform_identity()
    if body >= 0:
        X = body_q[body]
    bx = wp.transform_point(X, contact_body_pos[tid])
    margin = float(0.0)
    if shape_margin.shape[0] > 0:
        margin = shape_margin[shape]
    pen = r + margin - wp.dot(contact_normal[tid], x - bx)
    if pen > 0.0:
        wp.atomic_add(load, b, 0.5 * (soft_ke + shape_ke[shape]) * pen)


@wp.kernel
def _bag_bag_touch(info_array: wp.array(dtype=TriMeshCollisionInfo), tri_indices: wp.array2d(dtype=wp.int32),
                   particle_bag: wp.array(dtype=wp.int32), touched: wp.array(dtype=wp.int32)):
    """Two bags touch when a vertex of one has a triangle of the other in its self-contact list (the detection at
    the substep's start, within its query radius): a bag lying on another is supported."""
    i = wp.tid()
    b = particle_bag[i]
    if b < 0:
        return
    info = info_array[0]
    if i >= info.vertex_colliding_triangles_offsets.shape[0]:
        return
    for k in range(get_vertex_colliding_triangles_count(info, i)):
        tri = get_vertex_colliding_triangles(info, i, k)
        if tri >= 0:
            bt = particle_bag[tri_indices[tri, 0]]
            if bt >= 0 and bt != b:
                touched[b] = 1
                touched[bt] = 1


@wp.kernel
def _bag_supported(load: wp.array(dtype=float), bag_weight: wp.array(dtype=float), share: float,
                   touched: wp.array(dtype=wp.int32)):
    """touched: 1 when the colliders press the bag with ``share`` of its weight; a bag on a bag is already 1."""
    b = wp.tid()
    if load[b] >= share * bag_weight[b]:
        touched[b] = 1


@wp.kernel
def _forward_step_contact(dt: float, gravity: wp.array(dtype=wp.vec3), particle_world: wp.array(dtype=wp.int32),
                          pos_prev: wp.array(dtype=wp.vec3), pos: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec3),
                          inv_mass: wp.array(dtype=float), external_force: wp.array(dtype=wp.vec3),
                          particle_flags: wp.array(dtype=wp.int32), particle_bag: wp.array(dtype=wp.int32),
                          touched: wp.array(dtype=wp.int32), inertia_out: wp.array(dtype=wp.vec3),
                          displacements_out: wp.array(dtype=wp.vec3)):
    """Newton's ``forward_step`` with the guess's gravity left out for a touching bag; the inertia target as Newton's."""
    i = wp.tid()
    pos_prev[i] = pos[i]
    if not particle_flags[i] & _ACTIVE or inv_mass[i] == 0.0:
        inertia_out[i] = pos_prev[i]
        displacements_out[i] = wp.vec3(0.0, 0.0, 0.0)
        return
    g = gravity[particle_world[i]]
    f_acc = external_force[i] * inv_mass[i]
    vel_new = vel[i] + (g + f_acc) * dt
    inertia_out[i] = pos[i] + vel_new * dt
    b = particle_bag[i]
    if b >= 0 and touched[b] != 0:
        displacements_out[i] = (vel[i] + f_acc * dt) * dt
    else:
        displacements_out[i] = vel_new * dt


class ContactInitSolverVBD(SolverVBD):
    """SolverVBD whose touching bags start each substep's iterate at x + v dt (module docstring). Call
    :meth:`set_bag_groups` before the first step; until then it steps as SolverVBD."""

    _bag_groups_set = False

    def __init__(self, *args, **kwargs):
        if "contacts" not in inspect.signature(SolverVBD._initialize_particles).parameters:
            raise RuntimeError(f"polybag_vbd_init 'contact' overrides Newton 1.8's SolverVBD._initialize_particles"
                               f"(state_in, state_out, contacts, dt); Newton {newton.__version__}'s takes no contacts")
        super().__init__(*args, **kwargs)

    def set_bag_groups(self, groups) -> None:
        """``groups``: one particle index array per bag (PolybagRegistryEntry.particle_ranges())."""
        model = self.model
        label = np.full(model.particle_count, -1, dtype=np.int32)
        for k, g in enumerate(groups):
            label[np.asarray(g, dtype=np.int64)] = k
        self._bag_count = max(len(groups), 1)
        self._particle_bag = wp.array(label, dtype=wp.int32, device=self.device)
        self._bag_touched = wp.zeros(self._bag_count, dtype=wp.int32, device=self.device)
        self._bag_load = wp.zeros(self._bag_count, dtype=float, device=self.device)
        mass = model.particle_mass.numpy()
        g = float(np.linalg.norm(np.asarray(model.gravity.numpy()).reshape(-1, 3)[0]))
        weight = [float(mass[np.asarray(r, dtype=np.int64)].sum()) * g for r in groups] or [0.0]
        self._bag_weight = wp.array(np.asarray(weight, dtype=np.float32), dtype=float, device=self.device)
        self._bag_groups_set = True
        wp.load_module(module=__name__, device=self.device)

    def _initialize_particles(self, state_in, state_out, contacts, dt):
        if not self._bag_groups_set:
            return super()._initialize_particles(state_in, state_out, contacts, dt)
        model = self.model
        if model.particle_count == 0:
            return
        if not self.rigid_soft_enable_dat and not self.particle_enable_self_contact:
            self.pos_prev_collision_detection.assign(state_in.particle_q)
            self.particle_displacements.zero_()
        self._bag_touched.zero_()
        self._bag_load.zero_()
        n_c = int(contacts.soft_contact_indices.shape[0]) if contacts is not None else 0
        if n_c:
            wp.launch(_bag_touch, dim=n_c,
                      inputs=[contacts.soft_contact_count, n_c, contacts.soft_contact_indices,
                              contacts.soft_contact_barycentric, contacts.soft_contact_shape,
                              contacts.soft_contact_body_pos, contacts.soft_contact_normal, model.shape_body,
                              model.shape_margin, state_in.body_q, state_in.particle_q, model.particle_radius,
                              self._particle_bag, float(model.soft_contact_ke), model.shape_material_ke],
                      outputs=[self._bag_load], device=self.device)
        if self._bag_count > 1 and self.particle_enable_self_contact:
            wp.launch(_bag_bag_touch, dim=model.particle_count,
                      inputs=[self.trimesh_collision_info, model.tri_indices, self._particle_bag],
                      outputs=[self._bag_touched], device=self.device)
        wp.launch(_bag_supported, dim=self._bag_count, inputs=[self._bag_load, self._bag_weight, LOAD_SHARE],
                  outputs=[self._bag_touched], device=self.device)
        wp.launch(_forward_step_contact, dim=model.particle_count,
                  inputs=[dt, model.gravity, model.particle_world, self.particle_q_prev, state_in.particle_q,
                          state_in.particle_qd, model.particle_inv_mass, state_in.particle_f, model.particle_flags,
                          self._particle_bag, self._bag_touched],
                  outputs=[self.inertia, self.particle_displacements], device=self.device)
        self._penetration_free_truncation(state_in, contacts)
