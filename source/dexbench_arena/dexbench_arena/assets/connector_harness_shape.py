# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""The anchored DB-M2 cable's shape at a reset's tape spot."""
from __future__ import annotations

import math

import torch
from isaaclab.utils.math import matrix_from_quat, quat_from_angle_axis
from isaaclab_arena.utils.yaw import rotate_points_by_yaw
from pxr import UsdPhysics

from dexbench_arena.assets.connector_harness import harness_arc

_DTYPE = torch.float64
_Z = torch.tensor([0.0, 0.0, 1.0], dtype=_DTYPE)


def _matrix(xyzw) -> torch.Tensor:
    return matrix_from_quat(torch.as_tensor(xyzw, dtype=_DTYPE))


def _turn(angle: float, axis: torch.Tensor) -> torch.Tensor:
    """The rotation by ``angle`` about the unit ``axis``."""
    return matrix_from_quat(quat_from_angle_axis(torch.tensor([angle], dtype=_DTYPE), axis[None]))[0]


def _onto(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """The smallest rotation taking unit vector ``a`` onto unit vector ``b``."""
    axis = torch.linalg.cross(a, b)
    sine = float(torch.linalg.vector_norm(axis))
    if sine < 1e-12:
        return torch.eye(3, dtype=_DTYPE)
    return _turn(math.atan2(sine, float(a @ b)), axis / sine)


def intrinsic_xyz(rotation: torch.Tensor) -> torch.Tensor:
    """``(a, b, c)`` with ``rotation = Rx(a) Ry(b) Rz(c)``: a three-axis joint's coordinates in Newton's D6 convention
    (``compute_3d_rotational_dofs``: each axis turned by the rotations before it). Written out because the pin has
    only the forward map (``matrix_from_euler(..., "XYZ")``); ``euler_xyz_from_quat`` is the extrinsic one."""
    b = torch.asin(rotation[0, 2].clamp(-1.0, 1.0))
    a = torch.atan2(-rotation[1, 2], rotation[2, 2])
    c = torch.atan2(-rotation[0, 1], rotation[0, 0])
    return torch.stack([a, b, c])


class HarnessStartShape:
    """The cable's joint coordinates at a tape spot, for a build that authored it at another (``extend_harness``)."""

    CLOSE_ITERATIONS = 50
    CLOSE_TOLERANCE_M = 1e-10

    def __init__(self, nodes, rotations, child_frames, anchor, plug_xyz, spot):
        self.nodes = torch.as_tensor(nodes, dtype=_DTYPE)
        self.rotations = torch.as_tensor(rotations, dtype=_DTYPE)
        self.child_frames = torch.as_tensor(child_frames, dtype=_DTYPE)
        self.anchor = torch.as_tensor(anchor, dtype=_DTYPE)
        self.plug_xyz = torch.as_tensor(plug_xyz, dtype=_DTYPE)
        self.spot = torch.as_tensor(spot, dtype=_DTYPE)
        self.lengths = torch.linalg.vector_norm(torch.diff(self.nodes, dim=0), dim=1)
        self.length_m = float(self.lengths.sum())
        before = torch.cat([torch.zeros(1, dtype=_DTYPE), torch.cumsum(self.lengths, 0)[:-1]])
        self._carry = 1.0 - before / self.length_m

    @classmethod
    def from_stage(cls, stage, chain_path: str, plug_pose, spot) -> HarnessStartShape:
        """The chain ``extend_harness`` authored under ``chain_path`` (``<package>/Task/Wires/<key>_0``) with the plug
        at ``plug_pose`` (``(xyz, xyzw)``, the pose it was authored at, ``spot`` from the tape release), read from its
        joints: ``ToConnector`` fixes the first link to the plug, ``J_<i>`` turns link ``i + 1`` about the end of link
        ``i`` (its frame on link ``i`` unturned, on link ``i + 1`` at the link's origin), ``AtTape`` holds the last
        link's end at the anchor."""
        chain = stage.GetPrimAtPath(chain_path)
        if not chain:
            raise ValueError(f"no anchored cable at {chain_path}")

        def frames(name):
            joint = UsdPhysics.Joint(chain.GetChild(name))
            if not joint:
                raise ValueError(f"{chain_path} has no joint {name}")
            return [(torch.tensor(list(position.Get()), dtype=_DTYPE),
                     _matrix(list(rotation.Get().GetImaginary()) + [rotation.Get().GetReal()]))
                    for position, rotation in ((joint.GetLocalPos0Attr(), joint.GetLocalRot0Attr()),
                                               (joint.GetLocalPos1Attr(), joint.GetLocalRot1Attr()))]

        xyz, xyzw = plug_pose
        (attach_at, attach_turn), (link_at, link_turn) = frames("ToConnector")
        plug_turn = _matrix(xyzw)
        rotations = [plug_turn @ attach_turn @ link_turn.T]
        nodes = [torch.as_tensor(xyz, dtype=_DTYPE) + plug_turn @ attach_at - rotations[0] @ link_at]
        child_frames = []
        names = sorted(prim.GetName() for prim in chain.GetChildren() if prim.GetName().startswith("J_"))
        for name in names:
            (end, turn), (origin, child) = frames(name)
            if float(torch.linalg.vector_norm(origin)) > 1e-9 or not torch.allclose(turn, torch.eye(3, dtype=_DTYPE)):
                raise ValueError(f"{chain_path}/{name} is not a joint extend_harness authors")
            nodes.append(nodes[-1] + rotations[-1] @ end)
            rotations.append(rotations[-1] @ child.T)
            child_frames.append(child)
        (anchor, _), (tail, _) = frames("AtTape")
        nodes.append(nodes[-1] + rotations[-1] @ tail)
        return cls(torch.stack(nodes), torch.stack(rotations), torch.stack(child_frames), anchor, xyz, spot)

    @property
    def joints(self) -> int:
        return len(self.child_frames)

    def nodes_at(self, spot) -> tuple[torch.Tensor, torch.Tensor]:
        """The chain's nodes and link rotations with the plug at tape spot ``spot`` (``[dx, dy, yaw]``)."""
        delta = torch.as_tensor(spot, dtype=_DTYPE) - self.spot
        yaw = float(delta[2])
        shift = torch.stack([delta[0], delta[1], torch.zeros((), dtype=_DTYPE)])
        held = rotate_points_by_yaw(self.nodes[:2] - self.plug_xyz, yaw) + self.plug_xyz + shift
        reach, slack = float(torch.linalg.vector_norm(self.anchor - held[1])), float(self.lengths[1:].sum())
        if reach >= slack:
            raise ValueError(f"the cable cannot reach its anchor from this spot: {reach * 1e3:.2f} mm to span with "
                             f"{slack * 1e3:.2f} mm past its first link")
        nodes = torch.as_tensor(harness_arc(held[0].numpy(), self.anchor.numpy(), len(self.lengths), self.length_m),
                                dtype=_DTYPE)
        nodes[:2], nodes[-1] = held, self.anchor
        nodes = self._close(nodes)
        rotations = torch.empty_like(self.rotations)
        for i, rotation in enumerate(self.rotations):
            carried = _turn(yaw * float(self._carry[i]), _Z) @ rotation
            direction = nodes[i + 1] - nodes[i]
            rotations[i] = _onto(carried @ _Z, direction / torch.linalg.vector_norm(direction)) @ carried
        return nodes, rotations

    def _close(self, nodes: torch.Tensor) -> torch.Tensor:
        """Move the free nodes (2 to n - 1) onto the link lengths, the first two and the last held: Gauss-Newton steps
        of least change (``-J^T (J J^T)^-1 C``), which close a chain this near its lengths in a few."""
        lengths = self.lengths[1:]
        free = len(nodes) - 3
        rows = torch.arange(len(lengths))
        for step_count in range(self.CLOSE_ITERATIONS + 1):
            segments = nodes[2:] - nodes[1:-1]
            distances = torch.linalg.vector_norm(segments, dim=1)
            error = distances - lengths
            if float(error.abs().max()) < self.CLOSE_TOLERANCE_M:
                return nodes
            if step_count == self.CLOSE_ITERATIONS:
                break
            units = segments / distances[:, None]
            jacobian = torch.zeros(len(lengths), free, 3, dtype=_DTYPE)
            jacobian[rows[:-1], rows[:-1]] = units[:-1]
            jacobian[rows[1:], rows[1:] - 1] = -units[1:]
            jacobian = jacobian.reshape(len(lengths), 3 * free)
            step = jacobian.T @ torch.linalg.solve(jacobian @ jacobian.T, error)
            nodes = nodes.clone()
            nodes[2:-1] -= step.reshape(free, 3)
        raise RuntimeError(f"the cable's lengths did not close in {self.CLOSE_ITERATIONS} steps "
                           f"({float(error.abs().max()) * 1e3:.4f} mm off)")

    def joint_coordinates(self, spot) -> torch.Tensor:
        """``(joints, 3)`` coordinates of ``J_00`` .. ``J_nn`` (Newton D6, X Y Z) with the plug at ``spot``."""
        _nodes, rotations = self.nodes_at(spot)
        relative = rotations[:-1].transpose(1, 2) @ rotations[1:] @ self.child_frames
        return torch.stack([intrinsic_xyz(rotation) for rotation in relative])
