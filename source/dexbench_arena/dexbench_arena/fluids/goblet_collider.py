# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Closed solid shell around an open goblet cavity, for Newton MPM only."""
from pathlib import Path
import numpy as np
import warp as wp
from .pour_evaluation import Cavity


def build_goblet_shell(profile_path, thickness=.016, segments=64, rings=48):
    """Offset the bowl meridian outward, retaining the measured inner surface."""
    if not np.isfinite(thickness) or thickness <= 0 or segments < 12 or rings < 8:
        raise ValueError('Positive thickness, >=12 segments and >=8 rings required')
    cavity = Cavity.load(profile_path)
    arc = np.r_[0.,np.cumsum(np.hypot(np.diff(cavity.radius),np.diff(cavity.z))) ]
    samples = np.linspace(0.,arc[-1],rings)
    r = np.interp(samples,arc,cavity.radius)
    z = np.interp(samples,arc,cavity.z)
    if np.any(r <= 0):
        raise ValueError('Profile must have positive radii; use the first enclosed section above the floor')
    dr,dz = np.gradient(r,samples),np.gradient(z,samples)
    norm = np.hypot(dr,dz)
    outer_r = r+thickness*dz/norm
    outer_z = z-thickness*dr/norm
    theta = np.arange(segments)*2*np.pi/segments
    rz = list(zip(r,z))+list(zip(outer_r[::-1],outer_z[::-1]))
    vertices = [[rad*np.cos(t),rad*np.sin(t),height] for rad,height in rz for t in theta]
    faces=[]
    for ring in range(2*rings-1):
        for j in range(segments):
            a=ring*segments+j;b=ring*segments+(j+1)%segments
            c=a+segments;d=b+segments
            faces.extend(((a,b,c),(b,d,c)))
    inner_center=len(vertices);vertices.append([0,0,z[0]])
    outer_center=len(vertices);vertices.append([0,0,z[0]-thickness])
    base=(2*rings-1)*segments
    for j in range(segments):
        k=(j+1)%segments
        faces.extend(((inner_center,k,j),(outer_center,base+j,base+k)))
    vertices=np.asarray(vertices,dtype=np.float64)
    faces=np.asarray(faces,dtype=np.int32)
    tri=vertices[faces]
    volume=np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2])).sum()/6
    if volume < 0:
        faces=faces[:,::-1].copy()
    return vertices.astype(np.float32),faces


class GobletFluidCollider:
    def __init__(self, profile_path, position, rotation, voxel_size=.008):
        self.vertices,self.triangles=build_goblet_shell(profile_path,2*voxel_size)
        world=self.vertices@np.asarray(rotation).T+np.asarray(position)
        self.mesh=wp.Mesh(wp.array(world,dtype=wp.vec3),
                          wp.array(self.triangles.reshape(-1),dtype=wp.int32),
                          wp.zeros(len(world),dtype=wp.vec3))
        self.thickness=2*voxel_size
        self._prev_world=world.astype(np.float32)

    def update(self, position, rotation, dt=None):
        """Mirror the glass's measured pose so a lifted glass carries its water."""
        world=(self.vertices@np.asarray(rotation).T+np.asarray(position)).astype(np.float32)
        self.mesh.points.assign(wp.array(world,dtype=wp.vec3))
        if dt:
            self.mesh.velocities.assign(
                wp.array((world-self._prev_world)/float(dt),dtype=wp.vec3))
        self._prev_world=world
        self.mesh.refit()

    def save(self, path):
        np.savez_compressed(Path(path),vertices=self.vertices,triangles=self.triangles,
                            thickness_m=self.thickness)
