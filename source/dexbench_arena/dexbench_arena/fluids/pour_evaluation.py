# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Robot-independent, mass-conserving pour accounting in SI units."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Cavity:
    z: np.ndarray
    radius: np.ndarray

    def __post_init__(self):
        self.z, self.radius = np.asarray(self.z, float), np.asarray(self.radius, float)
        if (self.z.ndim != 1 or self.z.shape != self.radius.shape or len(self.z) < 2
                or not np.isfinite(self.z).all() or not np.isfinite(self.radius).all()
                or np.any(np.diff(self.z) <= 0) or np.any(self.radius < 0)):
            raise ValueError('Invalid cavity profile')

    @classmethod
    def load(cls, path):
        with np.load(path) as p:
            floor, rim = float(p['floor_z']), float(p['rim_z'])
            z = np.r_[floor, p['z'][(p['z'] > floor) & (p['z'] < rim)], rim]
            return cls(z, np.interp(z, p['z'], p['r_inner']))

    @property
    def capacity_ml(self):
        r = self.radius
        return float(np.sum(np.pi * np.diff(self.z) * (r[:-1]**2+r[:-1]*r[1:]+r[1:]**2)/3)*1e6)

    def contains(self, world, position, rotation):
        local = (np.asarray(world)-position) @ rotation
        return (np.isfinite(local).all(axis=1) & (local[:, 2] >= self.z[0])
                & (local[:, 2] < self.z[-1])
                & (np.linalg.norm(local[:, :2], axis=1) <= np.interp(local[:, 2], self.z, self.radius)))


def measure(positions, masses_kg, density, cup, cup_pose, jug, jug_pose, table_z=0.):
    p, mass = np.asarray(positions, float), np.asarray(masses_kg, float)
    if (p.shape != (len(mass), 3) or mass.ndim != 1 or not np.isfinite(mass).all()
            or np.any(mass <= 0) or not np.isfinite(density) or density <= 0):
        raise ValueError('Expected Nx3 positions, N positive masses, positive density')
    valid = np.isfinite(p).all(axis=1)
    source = valid & jug.contains(p, *jug_pose)
    target = valid & cup.contains(p, *cup_pose) & ~source
    outside = valid & ~source & ~target
    spill_z = max(table_z, float(cup_pose[0][2]+cup.z[0]))
    spilled = outside & (p[:, 2] < spill_z)
    masks = dict(target=target, source=source, inflight=outside & ~spilled,
                 spilled=spilled, invalid=~valid)
    result = {'total_ml': float(mass.sum()/density*1e6)}
    for name, mask in masks.items():
        result[name+'_count'] = int(mask.sum())
        result[name+'_ml'] = float(mass[mask].sum()/density*1e6)
    return result, masks
