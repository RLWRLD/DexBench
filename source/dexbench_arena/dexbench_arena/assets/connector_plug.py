# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""A harness plug inside a spawned package, as an Arena articulation that resets to a set offset from its CAD pose."""

from __future__ import annotations

from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_reference import ObjectReference
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.utils.pose import Pose


class ConnectorPlugReference(ObjectReference):
    """A plug articulation in ``parent_asset``, reset ``start_offset_xyz`` (package frame) from its CAD pose."""

    def __init__(
        self,
        name: str,
        prim_path: str,
        parent_asset: Object,
        start_offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
        start_rotation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        wire_drive: tuple[float, float] | None = None,
        wire_actuator: bool = False,
    ):
        self._wire_drive = wire_drive
        self._wire_actuator = wire_actuator
        self._start_offset_xyz = tuple(float(v) for v in start_offset_xyz)
        self._start_rotation_xyzw = tuple(float(v) for v in start_rotation_xyzw)
        super().__init__(
            name=name,
            prim_path=prim_path,
            parent_asset=parent_asset,
            object_type=ObjectType.ARTICULATION,
        )

    def _generate_articulation_cfg(self):
        """Arena's articulation cfg; with ``wire_drive`` (stiffness N m/rad, damping N m s/rad) every joint of the
        wire gets an implicit drive to its authored shape. With ``wire_actuator`` alone the group keeps the package's
        own drives (gains None: the USD's), so position targets set through Isaac Lab reach them, a saved state's
        included; without a group Isaac Lab leaves those targets to Newton."""
        cfg = super()._generate_articulation_cfg()
        if self._wire_drive is not None or self._wire_actuator:
            from isaaclab.actuators import ImplicitActuatorCfg

            stiffness, damping = self._wire_drive if self._wire_drive is not None else (None, None)
            cfg.actuators = {"wire": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=stiffness, damping=damping)}
        return cfg

    @property
    def mated_pose_relative_to_parent(self) -> Pose:
        """The plug's pose in the package file (mated), in the package frame."""
        return self.initial_pose_relative_to_parent

    def get_initial_pose(self) -> Pose:
        """The CAD pose moved by the start offset and turned, in the env frame: T_E_P * (offset * rotation * T_P_O)."""
        T_P_O = self.initial_pose_relative_to_parent
        turned = Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=self._start_rotation_xyzw).multiply(
            Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=T_P_O.rotation_xyzw))
        moved = Pose(
            position_xyz=tuple(p + d for p, d in zip(T_P_O.position_xyz, self._start_offset_xyz)),
            rotation_xyzw=turned.rotation_xyzw,
        )
        return self.get_parent_pose().multiply(moved)
