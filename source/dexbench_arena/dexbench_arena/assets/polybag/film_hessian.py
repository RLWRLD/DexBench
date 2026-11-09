# SPDX-License-Identifier: Apache-2.0
"""The film springs' Hessian in SolverVBD's per-vertex step: Newton's own, or (psd) with its transverse term clamped."""

from __future__ import annotations

import warp as wp

from newton._src.solvers.vbd import solver_vbd as _solver_vbd

FILM_HESSIANS = ("newton", "psd")


@wp.kernel
def accumulate_spring_force_and_psd_hessian(
    dt: float,
    current_color: int,
    pos_anchor: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    particle_colors: wp.array(dtype=int),
    num_springs: int,
    spring_indices: wp.array(dtype=int),
    spring_rest_length: wp.array(dtype=float),
    spring_stiffness: wp.array(dtype=float),
    spring_damping: wp.array(dtype=float),
    particle_forces: wp.array(dtype=wp.vec3),
    particle_hessians: wp.array(dtype=wp.mat33),
):
    """Newton's ``accumulate_spring_force_and_hessian`` (same inputs and outputs), the Hessian's transverse
    coefficient ``1 - l0 / l`` clamped at 0."""
    s = wp.tid()
    if s >= num_springs:
        return
    v0 = spring_indices[s * 2]
    v1 = spring_indices[s * 2 + 1]
    c0 = particle_colors[v0]
    c1 = particle_colors[v1]
    if c0 != current_color and c1 != current_color:
        return
    diff = pos[v0] - pos[v1]
    length = wp.max(wp.length(diff), 1e-8)
    l0 = spring_rest_length[s]
    k = spring_stiffness[s]
    d = diff / length
    ddt = wp.outer(d, d)
    force = k * (l0 - length) / length * diff
    hessian = k * (ddt + wp.max(1.0 - l0 / length, 0.0) * (wp.identity(3, float) - ddt))
    length_anchor = wp.max(wp.length(pos_anchor[v0] - pos_anchor[v1]), 1e-8)
    damping = spring_damping[s] / dt
    force_damping = -d * (damping * (length - length_anchor))
    hessian = hessian + ddt * damping
    if c0 == current_color:
        wp.atomic_add(particle_forces, v0, force + force_damping)
        wp.atomic_add(particle_hessians, v0, hessian)
    if c1 == current_color:
        wp.atomic_add(particle_forces, v1, -force - force_damping)
        wp.atomic_add(particle_hessians, v1, hessian)


class PsdFilmHessian:
    """Mixin for SolverVBD (or a subclass): its spring launches use :func:`accumulate_spring_force_and_psd_hessian`."""

    def _solve_particle_iteration(self, state_in, state_out, contacts, dt):
        newton_kernel = _solver_vbd.accumulate_spring_force_and_hessian
        _solver_vbd.accumulate_spring_force_and_hessian = accumulate_spring_force_and_psd_hessian
        try:
            return super()._solve_particle_iteration(state_in, state_out, contacts, dt)
        finally:
            _solver_vbd.accumulate_spring_force_and_hessian = newton_kernel


def with_film_hessian(solver_class: type, film_hessian: str) -> type:
    """``solver_class`` for ``film_hessian`` newton; a subclass with :class:`PsdFilmHessian` for psd."""
    if film_hessian not in FILM_HESSIANS:
        raise ValueError(f"film_hessian {film_hessian!r}: one of {FILM_HESSIANS}")
    if film_hessian == "newton":
        return solver_class
    return type(f"PsdFilm{solver_class.__name__}", (PsdFilmHessian, solver_class), {})


def load(device) -> None:
    """Compile this module's kernel before the manager captures its CUDA graph."""
    wp.load_module(module=__name__, device=device)
