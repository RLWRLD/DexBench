# SPDX-License-Identifier: Apache-2.0
"""A bag that waits off stage, out of the simulation (Warp and Newton only)."""

from __future__ import annotations

import re

import newton
import warp as wp


@wp.kernel
def _set_particles_active(
    particle_flags: wp.array(dtype=wp.int32),
    particle_offsets: wp.array(dtype=wp.int32),
    env_ids: wp.array(dtype=wp.int32),
    active_bit: wp.int32,
    active: wp.int32,
):
    i, j = wp.tid()
    p = particle_offsets[env_ids[i]] + j
    if active != 0:
        particle_flags[p] = particle_flags[p] | active_bit
    else:
        particle_flags[p] = particle_flags[p] & ~active_bit


def set_particles_active(particle_flags: wp.array, particle_offsets: wp.array, env_ids: wp.array,
                         particles_per_object: int, active: bool) -> None:
    """Set or clear ``ParticleFlags.ACTIVE`` on the particles of one object in each of ``env_ids``."""
    if len(env_ids) == 0 or particles_per_object == 0:
        return
    wp.launch(_set_particles_active, dim=(len(env_ids), particles_per_object),
              inputs=[particle_flags, particle_offsets, env_ids, int(newton.ParticleFlags.ACTIVE), int(bool(active))],
              device=particle_flags.device)


def stop_particle_collision(builder, patterns: tuple[str, ...] | list[str]) -> list[str]:
    """Clear ``COLLIDE_PARTICLES`` on the builder's shapes whose label matches any of ``patterns`` (regexes)."""
    collide = int(newton.ShapeFlags.COLLIDE_PARTICLES)
    changed = []
    for pattern in patterns:
        hits = [s for s in range(builder.shape_count)
                if re.search(pattern, str(builder.shape_label[s])) and int(builder.shape_flags[s]) & collide]
        if not hits:
            raise ValueError(f"stop particle collision: no particle-colliding shape matches {pattern!r}")
        for s in hits:
            builder.shape_flags[s] = int(builder.shape_flags[s]) & ~collide
            changed.append(str(builder.shape_label[s]))
    return changed
