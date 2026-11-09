# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Markers for the operator (``--teleop_target_marker``): over the task's target, and on its slot."""

from __future__ import annotations

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg, ManagerTermBase
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import quat_from_angle_axis
from pxr import Usd, UsdGeom

from dexbench_arena.tasks.predicates.geometry import per_env_rows
from dexbench_arena.tasks.predicates.placement import object_standing_in_slot

TARGET_MARKER_PRIM_PATH = "/Visuals/dexbench_teleop_target_marker"
TARGET_MARKER_CLEARANCE_M = 0.06
TARGET_MARKER_HEIGHT_M = 0.08
TARGET_MARKER_RADIUS_M = 0.03

TARGET_MARKER_CFG = VisualizationMarkersCfg(
    prim_path=TARGET_MARKER_PRIM_PATH,
    markers={
        "cone": sim_utils.ConeCfg(
            radius=TARGET_MARKER_RADIUS_M,
            height=TARGET_MARKER_HEIGHT_M,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.85, 0.0), emissive_color=(1.0, 0.6, 0.0), opacity=0.9
            ),
        ),
    },
)


def _top_above_root_m(asset) -> float:
    """How far the object's top (its world bounds at spawn, env 0) sits above its root [m]."""
    prim_path = sim_utils.find_matching_prim_paths(asset.cfg.prim_path)[0]
    prim = sim_utils.get_current_stage().GetPrimAtPath(prim_path)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    top_z = cache.ComputeWorldBound(prim).ComputeAlignedRange().GetMax()[2]
    root_z = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()[2]
    return float(top_z - root_z)


class TeleopTargetMarker(ManagerTermBase):
    """Startup event term: a cone over ``object_name`` in every env, redrawn on every render tick."""

    def __init__(self, cfg: EventTermCfg, env):
        super().__init__(cfg, env)
        self._object = env.scene[cfg.params["object_name"]]
        self._offset_z = _top_above_root_m(self._object) + TARGET_MARKER_CLEARANCE_M + TARGET_MARKER_HEIGHT_M / 2
        self._markers = VisualizationMarkers(TARGET_MARKER_CFG)
        down = quat_from_angle_axis(torch.tensor([math.pi]), torch.tensor([[1.0, 0.0, 0.0]]))
        self._orientations = down.repeat(env.num_envs, 1).to(env.device)
        self._debug_vis_handle = env.sim.vis_marker_registry.add_debug_vis_callback(self)

    def _debug_vis_callback(self, event=None) -> None:
        del event
        translations = self._object.data.root_pos_w.clone()
        translations[:, 2] += self._offset_z
        self._markers.visualize(translations=translations, orientations=self._orientations)

    def __call__(self, env, env_ids, object_name: str) -> None:
        del env, env_ids, object_name
        self._debug_vis_callback()

    def __del__(self):
        registry = getattr(getattr(self._env, "sim", None), "vis_marker_registry", None)
        if registry is not None:
            registry.clear_debug_vis_callback(self)


def teleop_target_marker_event(object_name: str) -> EventTermCfg:
    """The startup event term that puts :class:`TeleopTargetMarker` over ``object_name``."""
    return EventTermCfg(func=TeleopTargetMarker, mode="startup", params={"object_name": object_name})


SLOT_MARKER_PRIM_PATH = "/Visuals/dexbench_teleop_slot_marker"
SLOT_MARKER_THICKNESS_M = 0.004
SLOT_MARKER_LIFT_M = 0.001
SLOT_MARKER_PENDING, SLOT_MARKER_SEATED = 0, 1

SLOT_MARKER_CFG = VisualizationMarkersCfg(
    prim_path=SLOT_MARKER_PRIM_PATH,
    markers={
        "pending": sim_utils.CuboidCfg(
            size=(1.0, 1.0, 1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.3, 0.0), emissive_color=(0.6, 0.15, 0.0), opacity=0.5
            ),
        ),
        "seated": sim_utils.CuboidCfg(
            size=(1.0, 1.0, 1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 1.0, 0.2), emissive_color=(0.0, 0.7, 0.1), opacity=0.6
            ),
        ),
    },
)


class TeleopSlotMarker(ManagerTermBase):
    """Startup event term: the task's slot region on the shelf top in every env, coloured by its seated check."""

    def __init__(self, cfg: EventTermCfg, env):
        super().__init__(cfg, env)
        params = cfg.params
        origins = env.scene.env_origins
        center = per_env_rows(params["center_xy"], env.num_envs, 1, origins)
        half = per_env_rows(params["center_halfsize_xy"], env.num_envs, 1, origins) + per_env_rows(
            params["object_half_extents_xy"], env.num_envs, 1, origins
        )
        z = torch.full_like(center[:, :1], params["top_z_m"] + SLOT_MARKER_LIFT_M + SLOT_MARKER_THICKNESS_M / 2)
        self._translations = origins + torch.cat((center, z), dim=-1)
        self._scales = torch.cat((2.0 * half, torch.full_like(z, SLOT_MARKER_THICKNESS_M)), dim=-1)
        self._seated = params["seated"]
        self._markers = VisualizationMarkers(SLOT_MARKER_CFG)
        self._debug_vis_handle = env.sim.vis_marker_registry.add_debug_vis_callback(self)

    def _debug_vis_callback(self, event=None) -> None:
        del event
        seated = object_standing_in_slot(self._env, **self._seated)
        indices = torch.where(seated, SLOT_MARKER_SEATED, SLOT_MARKER_PENDING)
        self._markers.visualize(translations=self._translations, scales=self._scales, marker_indices=indices)

    def __call__(
        self, env, env_ids, center_xy, center_halfsize_xy, object_half_extents_xy, top_z_m: float, seated: dict
    ) -> None:
        del env, env_ids, center_xy, center_halfsize_xy, object_half_extents_xy, top_z_m, seated
        self._debug_vis_callback()

    def __del__(self):
        registry = getattr(getattr(self._env, "sim", None), "vis_marker_registry", None)
        if registry is not None:
            registry.clear_debug_vis_callback(self)


def teleop_slot_marker_event(region: dict) -> EventTermCfg:
    """The startup event term that draws :class:`TeleopSlotMarker` over a task's ``slot_region()``."""
    return EventTermCfg(func=TeleopSlotMarker, mode="startup", params=dict(region))
