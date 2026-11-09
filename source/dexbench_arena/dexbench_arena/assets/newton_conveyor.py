# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Friction-driven conveyor belt for the Newton / MuJoCo-Warp backend."""

from __future__ import annotations

import math
import re
from typing import Literal

import newton
import numpy as np
import warp as wp
from isaaclab.utils.configclass import configclass


_FREE_JOINT_TYPE = int(newton.JointType.FREE)


@wp.kernel
def _drive_belts(
    joint_q_start: wp.array(dtype=wp.int32),
    joint_qd_start: wp.array(dtype=wp.int32),
    body_index: wp.array(dtype=wp.int32),
    rest_pose: wp.array(dtype=wp.transform),
    drive_linear: wp.array(dtype=wp.vec3),
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
):
    """Pin each belt's pose and prescribe its surface velocity."""

    i = wp.tid()

    pose = rest_pose[i]
    p = wp.transform_get_translation(pose)
    r = wp.transform_get_rotation(pose)
    v = drive_linear[i]

    q_start = joint_q_start[i]
    joint_q[q_start + 0] = p[0]
    joint_q[q_start + 1] = p[1]
    joint_q[q_start + 2] = p[2]
    joint_q[q_start + 3] = r[0]
    joint_q[q_start + 4] = r[1]
    joint_q[q_start + 5] = r[2]
    joint_q[q_start + 6] = r[3]

    qd_start = joint_qd_start[i]
    joint_qd[qd_start + 0] = v[0]
    joint_qd[qd_start + 1] = v[1]
    joint_qd[qd_start + 2] = v[2]
    joint_qd[qd_start + 3] = 0.0
    joint_qd[qd_start + 4] = 0.0
    joint_qd[qd_start + 5] = 0.0

    body = body_index[i]
    body_q[body] = pose
    body_qd[body] = wp.spatial_vector(v, wp.vec3(0.0, 0.0, 0.0))


class BeltTreadmill:
    """A set of belts whose surfaces carry a prescribed velocity."""

    def __init__(self, model):
        self._model = model
        self._device = model.device

        self._shape_body = model.shape_body.numpy()
        self._joint_child = model.joint_child.numpy()
        self._joint_type = model.joint_type.numpy()
        self._joint_q_start = model.joint_q_start.numpy()
        self._joint_qd_start = model.joint_qd_start.numpy()

        self._shapes: list[int] = []
        self._bodies: list[int] = []
        self._joints: list[int] = []
        self._q_starts: list[int] = []
        self._qd_starts: list[int] = []
        self._rest_poses: list[tuple[float, ...]] = []
        self._velocities: list[tuple[float, float, float]] = []

        self._args: tuple | None = None

    @property
    def belt_count(self) -> int:
        """Number of registered belts, one per cloned environment."""

        return len(self._shapes)

    def add_belt(self, shape_index: int, state, *, velocity) -> None:
        """Register one belt collider and the world velocity of its surface."""

        body = int(self._shape_body[shape_index])
        if body < 0:
            raise ValueError(
                f"belt shape {shape_index} is a static collider with no rigid body, so it has no "
                "joint to prescribe motion on; the shared conveyor module authors the belt as a "
                "kinematic rigid body, so this means the USD parse dropped it"
            )

        joints = np.nonzero(self._joint_child == body)[0]
        if joints.size != 1:
            raise ValueError(
                f"belt body {body} is the child of {joints.size} joints; expected exactly one, "
                "the FREE joint Newton gives an unparented rigid body"
            )
        joint = int(joints[0])

        joint_type = int(self._joint_type[joint])
        if joint_type != _FREE_JOINT_TYPE:
            raise ValueError(
                f"belt joint {joint} has Newton joint type {joint_type}, but the drive writes the "
                f"FREE layout ({_FREE_JOINT_TYPE}); the belt's USD authoring must have changed"
            )

        pose = state.body_q.numpy()[body]

        self._shapes.append(int(shape_index))
        self._bodies.append(body)
        self._joints.append(joint)
        self._q_starts.append(int(self._joint_q_start[joint]))
        self._qd_starts.append(int(self._joint_qd_start[joint]))
        self._rest_poses.append(tuple(float(v) for v in pose[:7]))
        self._velocities.append(tuple(float(v) for v in velocity))

    def finalize(self) -> None:
        """Upload the per-belt arrays. Call once, before any CUDA-graph capture."""

        if not self._shapes:
            raise ValueError("no belts registered")

        self._args = (
            wp.array(self._q_starts, dtype=wp.int32, device=self._device),
            wp.array(self._qd_starts, dtype=wp.int32, device=self._device),
            wp.array(self._bodies, dtype=wp.int32, device=self._device),
            wp.array(
                [wp.transform(p[:3], p[3:7]) for p in self._rest_poses],
                dtype=wp.transform,
                device=self._device,
            ),
            wp.array(self._velocities, dtype=wp.vec3, device=self._device),
        )

    def apply(self, state) -> None:
        """Re-assert every belt's pose and surface velocity. Graph-safe."""

        if self._args is None:
            return
        wp.launch(
            _drive_belts,
            dim=self.belt_count,
            inputs=self._args,
            outputs=[state.joint_q, state.joint_qd, state.body_q, state.body_qd],
            device=self._device,
        )


@configclass
class BeltDriveCfg:
    """One conveyor belt a Newton scene moves: its collider, its surface velocity, its grip and its texture."""

    shape_pattern: str = ""
    """Regex on Newton shape labels (prim paths): the belt's collider, the kinematic rigid body the treadmill drives.
    Every shape colliding with shapes that matches is a belt, one per cloned environment."""

    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """The belt surface's velocity [m/s], in ``frame``."""

    frame: Literal["world", "belt"] = "world"
    """world: ``velocity`` is in the world frame; belt: in the belt collider's own frame (its body's world rotation
    times the shape's), so a yawed belt needs no other number."""

    friction: float | None = None
    """The belt collider's friction coefficient (``model.shape_material_mu``); None: as the asset authors it."""

    texture_pattern: str | None = None
    """Regex on stage prim paths: the ``UsdTransform2d`` shaders whose ``inputs:translation`` u slides at
    ``texture_rate``, the look of a moving belt; None: the texture stands still."""

    texture_rate: float = 0.0
    """How fast the texture slides along u [UV units per second of simulated time]."""


def belt_shapes(model, pattern: str) -> list[int]:
    """The shapes colliding with shapes whose label matches ``pattern``; ValueError when there are none."""

    flags = model.shape_flags.numpy()
    collide = int(newton.ShapeFlags.COLLIDE_SHAPES)
    shapes = [
        s for s, label in enumerate(model.shape_label) if re.search(pattern, str(label)) and int(flags[s]) & collide
    ]
    if not shapes:
        raise ValueError(f"belt: no shape colliding with shapes matches {pattern!r}")
    return shapes


def belt_world_velocity(model, state, shape: int, belt: BeltDriveCfg):
    """``belt.velocity`` in the world frame, for its collider ``shape``."""

    if belt.frame == "world":
        return wp.vec3(*(float(v) for v in belt.velocity))
    if belt.frame != "belt":
        raise ValueError(f"belt frame {belt.frame!r}: world or belt")
    shape_q = model.shape_transform.numpy()[shape]
    rotation = wp.quat(*(float(v) for v in shape_q[3:7]))
    body = int(model.shape_body.numpy()[shape])
    if body >= 0:
        body_q = state.body_q.numpy()[body]
        rotation = wp.quat(*(float(v) for v in body_q[3:7])) * rotation
    return wp.quat_rotate(rotation, wp.vec3(*(float(v) for v in belt.velocity)))


def build_belt_treadmill(model, state, belts) -> tuple[BeltTreadmill | None, bool]:
    """The treadmill for ``belts`` (BeltDriveCfg), and whether a belt's friction was written into
    ``model.shape_material_mu``, for the caller to notify the solver (``newton.ModelFlags.SHAPE_PROPERTIES``).
    (None, False) for no belts. Before graph capture (it allocates)."""

    if not belts:
        return None, False
    treadmill = BeltTreadmill(model)
    mu = None
    for belt in belts:
        shapes = belt_shapes(model, belt.shape_pattern)
        for s in shapes:
            treadmill.add_belt(s, state, velocity=belt_world_velocity(model, state, s, belt))
        if belt.friction is not None:
            mu = model.shape_material_mu.numpy() if mu is None else mu
            mu[shapes] = float(belt.friction)
        print(f"[dexbench] belt treadmill: {len(shapes)} belt(s) {belt.shape_pattern!r} at {tuple(belt.velocity)} m/s "
              f"({belt.frame} frame), friction {'as authored' if belt.friction is None else belt.friction}", flush=True)
    if mu is not None:
        model.shape_material_mu.assign(mu)
    treadmill.finalize()
    return treadmill, mu is not None


def resolve_belt_textures(belts) -> list:
    """``(inputs:translation attribute, rate, its v)`` for every stage shader a belt's ``texture_pattern`` matches;
    ValueError for a pattern that matches none."""

    import omni.usd

    stage = omni.usd.get_context().get_stage()
    found = []
    for belt in belts:
        if belt.texture_pattern is None:
            continue
        matched = 0
        for prim in stage.Traverse():
            if not re.search(belt.texture_pattern, str(prim.GetPath())):
                continue
            attr = prim.GetAttribute("inputs:translation")
            if not attr:
                continue
            current = attr.Get()
            found.append((attr, float(belt.texture_rate), float(current[1]) if current is not None else 0.0))
            matched += 1
        if not matched:
            raise ValueError(f"belt texture: no shader with inputs:translation matches {belt.texture_pattern!r}")
        print(f"[dexbench] belt texture {belt.texture_pattern!r} at {belt.texture_rate:+.3f} UV/s on {matched} "
              "shader(s)", flush=True)
    return found


def scroll_belt_textures(textures: list, sim_time_s: float) -> None:
    """Slide each resolved belt texture to where simulated time puts it."""

    from pxr import Gf

    for attr, rate, v in textures:
        attr.Set(Gf.Vec2f(math.fmod(rate * sim_time_s, 1.0), v))
