# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Volume-first seeding inside an axisymmetric vessel (metres, kilograms)."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class VolumeSeed:
    positions: np.ndarray
    volume_ml: float
    density_kg_m3: float
    spacing_m: float
    surface_z_m: float
    capacity_ml: float

    @property
    def count(self):
        return len(self.positions)

    @property
    def particle_volume_m3(self):
        return self.volume_ml * 1e-6 / self.count

    @property
    def particle_mass_kg(self):
        return self.particle_volume_m3 * self.density_kg_m3

    @property
    def particle_radius_m(self):
        return 0.5 * np.cbrt(self.particle_volume_m3)


def seed_profile(profile_path: str | Path, volume_ml=800.0,
                 spacing_m=0.005 / 1.2, density_kg_m3=1000.0,
                 axis_xy=(0.0, 0.0), wall_clearance_m=0.0, floor_clearance_m=0.0) -> VolumeSeed:
    """Fill the profile to the requested volume and normalize actual N particles."""
    values = (volume_ml, spacing_m, density_kg_m3)
    if not all(np.isfinite(v) and v > 0 for v in values):
        raise ValueError('Volume, spacing and density must be positive and finite')
    if np.shape(axis_xy) != (2,) or not np.isfinite(axis_xy).all():
        raise ValueError('axis_xy must contain two finite coordinates')
    if not all(np.isfinite(v) and v >= 0 for v in (wall_clearance_m, floor_clearance_m)):
        raise ValueError('Clearances must be finite and non-negative')
    with np.load(profile_path, allow_pickle=False) as profile:
        z = np.asarray(profile['z'], dtype=float)
        r = np.asarray(profile['r_inner'], dtype=float)
        floor, rim = float(profile['floor_z']), float(profile['rim_z'])
    if (z.ndim != 1 or z.shape != r.shape or len(z) < 2 or
            not np.isfinite(z).all() or not np.isfinite(r).all() or
            np.any(np.diff(z) <= 0) or np.any(r < 0) or
            not z[0] <= floor < rim <= z[-1]):
        raise ValueError('Invalid vessel profile')
    floor = floor + floor_clearance_m
    if not floor < rim:
        raise ValueError('Floor clearance leaves no cavity below the rim')
    zz = np.r_[floor, z[(z > floor) & (z < rim)], rim]
    rr = np.maximum(np.interp(zz, z, r) - wall_clearance_m, 0.0)
    segment_v = np.pi * np.diff(zz) * (rr[:-1]**2 + rr[:-1]*rr[1:] + rr[1:]**2) / 3
    cumulative = np.r_[0.0, np.cumsum(segment_v)]
    capacity = float(cumulative[-1] * 1e6)
    requested = volume_ml * 1e-6
    if requested > cumulative[-1] * (1 + 1e-12):
        raise ValueError(f'Requested {volume_ml:g} mL exceeds profile capacity {capacity:.3f} mL')
    seg = min(int(np.searchsorted(cumulative, requested, side='right') - 1), len(zz)-2)
    low, high = float(zz[seg]), float(zz[seg+1])
    for _ in range(60):
        mid = (low + high) / 2
        rm = float(np.interp(mid, zz, rr))
        v = cumulative[seg] + np.pi * (mid-zz[seg]) * (rr[seg]**2 + rr[seg]*rm + rm**2) / 3
        if v < requested:
            low = mid
        else:
            high = mid
    surface = (low + high) / 2
    radius_max = float(rr.max())
    width = int(np.ceil(2 * radius_max / spacing_m))
    level_count = max(int(np.ceil((surface-floor)/spacing_m)), 1)
    if width * width * level_count > 5_000_000:
        raise ValueError('Sampling grid exceeds 5 million cells; increase spacing')
    levels = np.arange(floor + spacing_m/2, surface, spacing_m)
    xy = (np.arange(width) - (width-1)/2) * spacing_m
    xx, yy = np.meshgrid(xy, xy, indexing='ij')
    radial = np.hypot(xx, yy)
    layers = []
    for height in levels:
        valid = radial <= float(np.interp(height, zz, rr)) - spacing_m/2
        if valid.any():
            layers.append(np.column_stack((xx[valid]+axis_xy[0], yy[valid]+axis_xy[1],
                                           np.full(valid.sum(), height))))
    if not layers:
        raise ValueError('No particle samples fit; reduce spacing or increase volume')
    return VolumeSeed(np.concatenate(layers), float(volume_ml), float(density_kg_m3),
                      float(spacing_m), surface, capacity)


def add_seed_particles(builder, seed: VolumeSeed, positions=None):
    """Insert equal-mass samples into a Newton builder before finalize()."""
    positions = seed.positions if positions is None else np.asarray(positions)
    if positions.shape != seed.positions.shape or not np.isfinite(positions).all():
        raise ValueError('World positions must match the seed shape and be finite')
    builder.add_particles(pos=positions.tolist(), vel=np.zeros_like(positions).tolist(),
                          mass=[seed.particle_mass_kg]*seed.count,
                          radius=[float(seed.particle_radius_m)]*seed.count)
