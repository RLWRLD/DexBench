# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
#
# Adapted from Isaac Lab 2d3ae03ede6be38d71af5698509fe8116b6b7216,
# source/isaaclab_contrib/isaaclab_contrib/custom_coupling/kernels.py,
# _kernel_body_particle_reaction (BSD-3-Clause). DexBench's additions retain
# Apache-2.0; the adapted kernel also retains the following BSD terms.
# Changes include edge/face contacts, Newton API compatibility and contact filtering.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""The bag's contact reactions on MJWarp's bodies, for two-way coupling, edge and face contacts included."""

from __future__ import annotations

import inspect

import warp as wp

from newton._src.solvers.vbd.rigid_vbd_kernels import _eval_soft_ef_contact

try:
    from newton._src.geometry.soft_contacts_mesh import (
        _make_soft_contact_filter_data,
        _soft_contact_force_enabled,
        _SoftContactFilterData,
    )

    HAS_CONTACT_FILTER = True
except ImportError:
    HAS_CONTACT_FILTER = False

    @wp.struct
    class _SoftContactFilterData:
        unused: int

    @wp.func
    def _soft_contact_force_enabled(contact_index: int, data: _SoftContactFilterData) -> bool:
        return True

    def _make_soft_contact_filter_data(model, contacts, particle_q, body_q) -> _SoftContactFilterData:
        return _SoftContactFilterData()

_TAKES_LOG_BARRIER = "use_log_barrier" in inspect.signature(_eval_soft_ef_contact.func).parameters

if _TAKES_LOG_BARRIER:

    @wp.func
    def _soft_contact(
        tid: int, corners: wp.vec3i, bary: wp.vec3, pos: wp.array(dtype=wp.vec3), pos_prev: wp.array(dtype=wp.vec3),
        radius: wp.array(dtype=wp.float32), ke: float, kd: float, mu: float, friction_epsilon: float,
        shape_body: wp.array(dtype=wp.int32), body_q: wp.array(dtype=wp.transform),
        body_q_prev: wp.array(dtype=wp.transform), body_qd: wp.array(dtype=wp.spatial_vector),
        body_com: wp.array(dtype=wp.vec3), contact_shape: wp.array(dtype=wp.int32),
        contact_body_pos: wp.array(dtype=wp.vec3), contact_body_vel: wp.array(dtype=wp.vec3),
        contact_normal: wp.array(dtype=wp.vec3), shape_margin: wp.array(dtype=wp.float32), dt: float,
        use_log_barrier: bool,
    ):
        return _eval_soft_ef_contact(tid, corners, bary, pos, pos_prev, radius, ke, kd, mu, friction_epsilon, shape_body,
                                     body_q, body_q_prev, body_qd, body_com, contact_shape, contact_body_pos,
                                     contact_body_vel, contact_normal, shape_margin, dt, use_log_barrier)

else:

    @wp.func
    def _soft_contact(
        tid: int, corners: wp.vec3i, bary: wp.vec3, pos: wp.array(dtype=wp.vec3), pos_prev: wp.array(dtype=wp.vec3),
        radius: wp.array(dtype=wp.float32), ke: float, kd: float, mu: float, friction_epsilon: float,
        shape_body: wp.array(dtype=wp.int32), body_q: wp.array(dtype=wp.transform),
        body_q_prev: wp.array(dtype=wp.transform), body_qd: wp.array(dtype=wp.spatial_vector),
        body_com: wp.array(dtype=wp.vec3), contact_shape: wp.array(dtype=wp.int32),
        contact_body_pos: wp.array(dtype=wp.vec3), contact_body_vel: wp.array(dtype=wp.vec3),
        contact_normal: wp.array(dtype=wp.vec3), shape_margin: wp.array(dtype=wp.float32), dt: float,
        use_log_barrier: bool,
    ):
        return _eval_soft_ef_contact(tid, corners, bary, pos, pos_prev, radius, ke, kd, mu, friction_epsilon, shape_body,
                                     body_q, body_q_prev, body_qd, body_com, contact_shape, contact_body_pos,
                                     contact_body_vel, contact_normal, shape_margin, dt)


@wp.kernel
def previous_particle_positions(
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
    dt: float,
    particle_q_prev: wp.array(dtype=wp.vec3),
):
    """Where each particle was a substep ago: VBD moves ``particle_q`` in place, so it is rebuilt from the
    velocity, as the upstream kernel does per contact."""
    i = wp.tid()
    particle_q_prev[i] = particle_q[i] - particle_qd[i] * dt


@wp.kernel
def body_soft_contact_reaction(
    contact_count: wp.array(dtype=wp.int32),
    contact_indices: wp.array(dtype=wp.vec3i),
    contact_barycentric: wp.array(dtype=wp.vec3),
    contact_shape: wp.array(dtype=wp.int32),
    contact_body_pos: wp.array(dtype=wp.vec3),
    contact_body_vel: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    particle_q: wp.array(dtype=wp.vec3),
    particle_q_prev: wp.array(dtype=wp.vec3),
    particle_radius: wp.array(dtype=wp.float32),
    body_q: wp.array(dtype=wp.transform),
    body_q_prev: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    shape_body: wp.array(dtype=wp.int32),
    shape_material_mu: wp.array(dtype=wp.float32),
    shape_margin: wp.array(dtype=wp.float32),
    soft_contact_ke: float,
    soft_contact_kd: float,
    soft_contact_mu: float,
    friction_epsilon: float,
    dt: float,
    use_log_barrier: bool,
    contact_filter: _SoftContactFilterData,
    body_f: wp.array(dtype=wp.spatial_vector),
):
    """Add each soft contact's reaction, as a wrench about the body's centre of mass, to the body it touches."""
    tid = wp.tid()
    if tid >= contact_count[0]:
        return
    if not _soft_contact_force_enabled(tid, contact_filter):
        return
    corners = contact_indices[tid]
    if corners[0] < 0:
        return
    shape = contact_shape[tid]
    body = shape_body[shape]
    if body < 0:
        return
    mu = wp.sqrt(soft_contact_mu * shape_material_mu[shape])
    force, _hessian, contact_pos = _soft_contact(
        tid, corners, contact_barycentric[tid], particle_q, particle_q_prev, particle_radius,
        soft_contact_ke, soft_contact_kd, mu, friction_epsilon, shape_body, body_q, body_q_prev, body_qd,
        body_com, contact_shape, contact_body_pos, contact_body_vel, contact_normal, shape_margin, dt, use_log_barrier,
    )
    com = wp.transform_point(body_q[body], body_com[body])
    reaction = -force
    torque = wp.cross(contact_pos - com, reaction)
    wp.atomic_add(body_f, body, wp.spatial_vector(reaction[0], reaction[1], reaction[2],
                                                  torque[0], torque[1], torque[2]))


def contact_filter_data(model, contacts, particle_q, body_q) -> _SoftContactFilterData:
    """The filter VBD selects its contacts' forces by, bound to ``contacts`` and the positions they are evaluated at
    (references to existing arrays; nothing is allocated). Every record passes when ``contacts`` holds no mesh
    features."""
    return _make_soft_contact_filter_data(model, contacts, particle_q, body_q)
