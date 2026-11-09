# SPDX-License-Identifier: Apache-2.0
"""Camera extrinsics variation that offsets each pose view in its own local frame."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp
import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import Camera, TiledCamera
from isaaclab.sim.views.usd_frame_view import UsdFrameView
from isaaclab.utils.math import quat_apply

from isaaclab_arena.variations.camera_extrinsics_variation import CameraExtrinsicsVariation
from isaaclab_arena.variations.continuous_sampler import ContinuousSampler

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


POSITION_TOLERANCE_M = 1.0e-4
"""How far the two views may place the camera apart in its body's frame (Arena's writer: 1e-4 m on local poses)."""
ALIGNMENT_TOLERANCE = 1.0e-3
"""How far below 1 the two views' orientations' absolute dot product may be (Arena's writer: 1e-3)."""


def usd_camera_poses_in_body_frame(prim_path: str, stage=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Each matching camera prim's pose in the frame of its nearest rigid-body (or articulation-root) ancestor, as the
    USD stage composes it: ``(translations (N, 3), orientations x, y, z, w (N, 4))``, in ``UsdFrameView``'s prim order."""
    from pxr import UsdPhysics  # noqa: PLC0415

    translations, orientations = [], []
    for prim in sim_utils.find_matching_prims(prim_path, stage=stage):
        body = prim.GetParent()
        while body and body.IsValid() and not (
            body.HasAPI(UsdPhysics.RigidBodyAPI) or body.HasAPI(UsdPhysics.ArticulationRootAPI)
        ):
            body = body.GetParent()
        if not (body and body.IsValid()):
            raise RuntimeError(f"camera prim {prim.GetPath()} has no rigid-body ancestor for its Newton site")
        position, orientation = sim_utils.resolve_prim_pose(prim, body)
        translations.append([float(v) for v in position])
        orientations.append([float(v) for v in orientation])
    return torch.tensor(translations), torch.tensor(orientations)


class MountedCameraExtrinsicsVariation(CameraExtrinsicsVariation):
    """Arena's camera extrinsics variation, realized per pose view (:class:`ApplyMountedCameraExtrinsics`)."""

    def build_event_cfg(self) -> tuple[str, EventTermCfg]:
        event_name, event_cfg = super().build_event_cfg()
        event_cfg.func = ApplyMountedCameraExtrinsics
        return event_name, event_cfg


class ApplyMountedCameraExtrinsics(ManagerTermBase):
    """Event term: offset a camera by a sampler-drawn translation in its ROS optical frame (+X right, +Y down,
    +Z forward), in the physics FrameView's frame and in the USD prim's frame alike."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        sampler: ContinuousSampler = cfg.params["sampler"]
        camera = env.scene[asset_cfg.name]
        if not isinstance(camera, (Camera, TiledCamera)):
            raise TypeError(f"scene['{asset_cfg.name}'] is a {type(camera).__name__}, not a Camera or TiledCamera")
        if tuple(sampler.shape_per_sample) != (3,):
            raise ValueError(f"the extrinsics sampler draws {tuple(sampler.shape_per_sample)}, not an XYZ (3,)")
        self._camera = camera
        self._nominal: list[tuple[object, torch.Tensor, torch.Tensor]] | None = None
        """``(view, translation, orientation xyzw)`` of the physics FrameView and the USD prim, at their nominal pose."""

    def _snapshot(self) -> list[tuple[object, torch.Tensor, torch.Tensor]]:
        assert self._camera._view is not None, "the camera's FrameView is not initialized"
        nominal = []
        for view in (self._camera._view, UsdFrameView(self._camera.cfg.prim_path)):
            translations, orientations = view.get_local_poses()
            nominal.append((view, translations.torch.detach().clone(), orientations.torch.detach().clone()))
        self._assert_views_hold_one_camera(nominal[0][1], nominal[0][2])
        return nominal

    def _assert_views_hold_one_camera(self, physics_t: torch.Tensor, physics_q: torch.Tensor) -> None:
        """Refuse a camera whose physics FrameView (a site, local to its body) and USD prim (composed into the same
        body's frame) are not at one pose: then the renderer and ``camera.data`` would see different cameras."""
        usd_t, usd_q = usd_camera_poses_in_body_frame(self._camera.cfg.prim_path)
        usd_t, usd_q = usd_t.to(physics_t), usd_q.to(physics_q)
        name = self._camera.cfg.prim_path
        if usd_t.shape != physics_t.shape:
            raise RuntimeError(f"{name}: {tuple(usd_t.shape)[0]} USD camera prims, {tuple(physics_t.shape)[0]} sites")
        if not torch.allclose(physics_t, usd_t, atol=POSITION_TOLERANCE_M):
            raise RuntimeError(f"{name}: the physics FrameView and the USD camera prim place the camera apart in its "
                               f"body's frame (physics={physics_t.tolist()}, usd={usd_t.tolist()})")
        alignment = (physics_q * usd_q).sum(dim=-1).abs()
        if not bool(torch.all(alignment > 1.0 - ALIGNMENT_TOLERANCE)):
            raise RuntimeError(f"{name}: the physics FrameView and the USD camera prim orient the camera differently "
                               f"in its body's frame (alignment={alignment.tolist()})")

    def __call__(
        self,
        env: ManagerBasedEnv,  # noqa: ARG002
        env_ids: torch.Tensor,
        asset_cfg: SceneEntityCfg,  # noqa: ARG002
        sampler: ContinuousSampler,
    ) -> None:
        if self._nominal is None:
            self._nominal = self._snapshot()
        offset_ros = sampler.sample(num_samples=len(env_ids), env_ids=env_ids)
        for view, translations, orientations in self._nominal:
            ids = env_ids.to(translations.device)
            offset = offset_ros.to(device=translations.device, dtype=translations.dtype)
            ros_to_opengl = offset.new_tensor((1.0, 0.0, 0.0, 0.0)).expand(len(ids), 4)
            offset_in_view = quat_apply(orientations[ids], quat_apply(ros_to_opengl, offset))
            view.set_local_poses(translations=translations[ids] + offset_in_view, orientations=None,
                                 indices=wp.from_torch(ids.to(torch.int32)))
        self._camera.reset(env_ids)
