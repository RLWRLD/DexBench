# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""A container's interior as a lathe profile, and the pull that keeps a rendered water surface inside it."""

from __future__ import annotations

import numpy as np


class LatheCavity:
    """Inner radius ``r(z)`` between ``floor_z`` and ``rim_z``, in the container's frame with z along its axis."""

    def __init__(self, z, r, floor_z: float, rim_z: float):
        z, r = np.asarray(z, float), np.asarray(r, float)
        keep = (z >= floor_z) & (z <= rim_z)
        zz, rr = z[keep], r[keep]
        poly = np.vstack([[0.0, zz[0]], np.column_stack([rr, zz])])
        self.a, self.b = poly[:-1], poly[1:]
        self.z, self.r, self.floor_z, self.rim_z = z, r, float(floor_z), float(rim_z)
        self.r_rim = float(np.interp(self.rim_z, z, r))

    @classmethod
    def from_npz(cls, path) -> "LatheCavity":
        with np.load(path) as profile:
            return cls(profile["z"], profile["r_inner"], float(profile["floor_z"]), float(profile["rim_z"]))

    def inside(self, rxy, lz):
        return (lz >= self.floor_z) & (lz <= self.rim_z) & (rxy <= np.interp(lz, self.z, self.r))

    def nearest(self, rxy, lz):
        """The nearest boundary point ``(r, z)`` to each query point, and its distance."""
        p = np.column_stack([rxy, lz])[:, None, :]
        ab = (self.b - self.a)[None]
        t = np.clip(((p - self.a[None]) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-18), 0.0, 1.0)
        q = self.a[None] + t[..., None] * ab
        d = np.linalg.norm(p - q, axis=-1)
        k = d.argmin(axis=1)
        idx = np.arange(len(k))
        return q[idx, k], d[idx, k]

    def hold(self, local, particles, eps: float = 0.0005, reach: float = 0.025):
        """Pull the surface points that water held in the cavity pushed out of it back inside."""
        out = np.array(local, dtype=np.float64, copy=True)
        moved = np.zeros(len(out), bool)
        rxy, lz = np.hypot(out[:, 0], out[:, 1]), out[:, 2]
        wall = (~self.inside(rxy, lz) & (lz < self.rim_z) & (lz > self.floor_z - reach)
                & (rxy < self.r.max() + reach))
        mouth = (lz >= self.rim_z) & (lz < self.rim_z + reach) & (rxy > self.r_rim) & (rxy < self.r_rim + reach)
        if not (wall | mouth).any() or len(particles) == 0:
            return out, moved
        particles = np.asarray(particles, dtype=np.float64)
        pr = np.hypot(particles[:, 0], particles[:, 1])
        near = ((pr < self.r.max() + 2 * reach) & (particles[:, 2] > self.floor_z - 2 * reach)
                & (particles[:, 2] < self.rim_z + 2 * reach))
        P = particles[near]
        if len(P) == 0:
            return out, moved
        held = self.inside(pr[near], P[:, 2]) | ((P[:, 2] >= self.rim_z) & (P[:, 2] < self.rim_z + reach)
                                                 & (pr[near] <= self.r_rim))
        idx = np.flatnonzero(wall | mouth)
        nearest = np.empty(len(idx), int)
        for s in range(0, len(idx), 1024):
            nearest[s:s + 1024] = ((out[idx[s:s + 1024], None, :] - P[None, :, :]) ** 2).sum(-1).argmin(axis=1)
        sel = idx[held[nearest]]
        if len(sel) == 0:
            return out, moved
        sel_wall, sel_mouth = sel[wall[sel]], sel[mouth[sel]]
        q_wall = self.nearest(rxy[sel_wall], lz[sel_wall])[0] if len(sel_wall) else np.zeros((0, 2))
        q = np.vstack([q_wall, np.column_stack([np.full(len(sel_mouth), self.r_rim), lz[sel_mouth]])])
        sel = np.concatenate([sel_wall, sel_mouth])
        r_sel = rxy[sel]
        step = q - np.column_stack([r_sel, lz[sel]])
        target = q + eps * step / np.maximum(np.linalg.norm(step, axis=1, keepdims=True), 1e-12)
        scale = np.where(r_sel > 1e-9, np.maximum(target[:, 0], 0.0) / np.maximum(r_sel, 1e-9), 0.0)
        out[sel, 0] *= scale
        out[sel, 1] *= scale
        out[sel, 2] = target[:, 1]
        moved[sel] = True
        return out, moved
