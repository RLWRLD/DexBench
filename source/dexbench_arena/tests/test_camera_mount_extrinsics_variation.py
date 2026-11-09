"""A palm camera's extrinsics draw moves the physics view and the USD prim to one camera pose."""


import math
import types

import pytest

pytest.importorskip("isaaclab.sensors")

import torch
import warp as wp
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.sensors import Camera
from isaaclab.utils.math import quat_apply, quat_mul

from dexbench_arena.embodiments.rby1_wuji2.rby1_wuji2 import DexBenchRby1Wuji2JointEmbodiment
from dexbench_arena.variations import camera_mount_extrinsics_variation as mounted

CAMERA = "wrist_left_cam"


def _quat_xyzw(axis, angle_rad):
    axis = torch.tensor(axis, dtype=torch.float32)
    return torch.cat([axis * math.sin(angle_rad / 2), torch.tensor([math.cos(angle_rad / 2)])])


class _Proxy:
    def __init__(self, tensor):
        self.torch = tensor


class _View:
    def __init__(self, translations, orientations):
        self.translations, self.orientations = translations.clone(), orientations.clone()

    def get_local_poses(self):
        return _Proxy(self.translations.clone()), _Proxy(self.orientations.clone())

    def set_local_poses(self, translations, orientations, indices):
        assert orientations is None
        self.translations[wp.to_torch(indices).long()] = translations


class _Camera(Camera):
    def __init__(self, view):
        self._view = view
        self.cfg = types.SimpleNamespace(prim_path=f"/World/envs/env_.*/Robot/link_left_arm_6/wrist_camera/{CAMERA}")
        self.resets = []

    def reset(self, env_ids=None):
        self.resets.append(env_ids)

    def __del__(self):
        pass


class _Sampler:
    shape_per_sample = (3,)

    def __init__(self):
        self.draws = []

    def sample(self, num_samples, env_ids):
        return self.draws.pop(0)[:num_samples]


def test_the_draw_lands_at_one_camera_pose_in_both_views(monkeypatch):
    attach_q, attach_t = _quat_xyzw([0, 0, 1], math.pi / 2), torch.tensor([0.10, 0.02, -0.03])
    mount_q = _quat_xyzw([1, 0, 0], math.radians(30)).expand(2, 4)
    mount_t = torch.tensor([[-0.005, -0.065, 0.059]] * 2)
    body_q = quat_mul(attach_q.expand(2, 4), mount_q)
    body_t = attach_t + quat_apply(attach_q.expand(2, 4), mount_t)
    physics, usd = _View(body_t, body_q), _View(mount_t, mount_q)
    monkeypatch.setattr(mounted, "UsdFrameView", lambda prim_path: usd)
    monkeypatch.setattr(mounted, "usd_camera_poses_in_body_frame", lambda prim_path: (body_t.clone(), -body_q))
    camera, sampler = _Camera(physics), _Sampler()
    env = types.SimpleNamespace(scene={CAMERA: camera}, num_envs=2, device="cpu")
    cfg = EventTermCfg(func=mounted.ApplyMountedCameraExtrinsics, mode="reset",
                       params={"asset_cfg": SceneEntityCfg(CAMERA), "sampler": sampler})
    term = mounted.ApplyMountedCameraExtrinsics(cfg, env)

    ros_to_opengl = _quat_xyzw([1, 0, 0], math.pi).expand(2, 4)
    for draw in (torch.tensor([[0.03, -0.01, 0.02], [-0.03, 0.03, -0.02]]), torch.tensor([[0.0, 0.0, 0.03]] * 2)):
        sampler.draws.append(draw)
        term(env, torch.tensor([0, 1]), SceneEntityCfg(CAMERA), sampler)
        assert torch.allclose(physics.translations, attach_t + quat_apply(attach_q.expand(2, 4), usd.translations),
                              atol=1e-6)
        expected = body_t + quat_apply(body_q, quat_apply(ros_to_opengl, draw))
        assert torch.allclose(physics.translations, expected, atol=1e-6)
    assert torch.equal(physics.orientations, body_q) and torch.equal(usd.orientations, mount_q)
    assert len(camera.resets) == 2


def _term_on(monkeypatch, physics, usd, usd_in_body):
    monkeypatch.setattr(mounted, "UsdFrameView", lambda prim_path: usd)
    monkeypatch.setattr(mounted, "usd_camera_poses_in_body_frame", lambda prim_path: usd_in_body)
    camera, sampler = _Camera(physics), _Sampler()
    env = types.SimpleNamespace(scene={CAMERA: camera}, num_envs=1, device="cpu")
    cfg = EventTermCfg(func=mounted.ApplyMountedCameraExtrinsics, mode="reset",
                       params={"asset_cfg": SceneEntityCfg(CAMERA), "sampler": sampler})
    return mounted.ApplyMountedCameraExtrinsics(cfg, env), env, sampler


@pytest.mark.parametrize("shift", ["translation", "orientation"])
def test_views_that_place_the_camera_apart_are_refused_before_any_write(monkeypatch, shift):
    site_t, site_q = torch.tensor([[-0.0652, 0.0039, -0.0958]]), _quat_xyzw([0, 1, 0], 0.3).expand(1, 4)
    prim_t, prim_q = site_t.clone(), site_q.clone()
    if shift == "translation":
        prim_t[0, 2] += 0.002
    else:
        prim_q = _quat_xyzw([0, 1, 0], 0.4).expand(1, 4)
    physics = _View(site_t, site_q)
    term, env, sampler = _term_on(monkeypatch, physics, _View(torch.zeros(1, 3), site_q), (prim_t, prim_q))
    sampler.draws.append(torch.tensor([[0.0, 0.0, 0.005]]))
    with pytest.raises(RuntimeError, match="body's frame"):
        term(env, torch.tensor([0]), SceneEntityCfg(CAMERA), sampler)
    assert torch.equal(physics.translations, site_t)


def test_the_usd_pose_is_composed_into_the_rigid_bodys_frame():
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    link = UsdGeom.Xform.Define(stage, "/World/envs/env_0/Robot/link_left_arm_6")
    link.AddTranslateOp().Set(Gf.Vec3d(0.4, -0.2, 1.1))
    UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
    attach = UsdGeom.Xform.Define(stage, "/World/envs/env_0/Robot/link_left_arm_6/wrist_camera")
    attach.AddTransformOp().Set(Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), 90)) *
                                Gf.Matrix4d().SetTranslate(Gf.Vec3d(0.10, 0.02, -0.03)))
    camera = UsdGeom.Camera.Define(stage, f"/World/envs/env_0/Robot/link_left_arm_6/wrist_camera/{CAMERA}")
    camera.AddTranslateOp().Set(Gf.Vec3d(-0.005, -0.065, 0.059))
    camera.AddOrientOp().Set(Gf.Quatf(math.cos(math.radians(15)), math.sin(math.radians(15)), 0, 0))

    translations, orientations = mounted.usd_camera_poses_in_body_frame(
        f"/World/envs/env_.*/Robot/link_left_arm_6/wrist_camera/{CAMERA}", stage=stage)
    attach_q = _quat_xyzw([0, 0, 1], math.pi / 2)
    expected_t = torch.tensor([0.10, 0.02, -0.03]) + quat_apply(attach_q[None], torch.tensor([[-0.005, -0.065, 0.059]]))
    expected_q = quat_mul(attach_q[None], _quat_xyzw([1, 0, 0], math.radians(30))[None])
    assert torch.allclose(translations, expected_t, atol=1e-6)
    assert torch.allclose((orientations * expected_q).sum(-1).abs(), torch.ones(1), atol=1e-6)


def test_the_embodiment_registers_every_camera_extrinsics_under_arenas_key():
    embodiment = DexBenchRby1Wuji2JointEmbodiment(enable_cameras=True, wrist_cameras=True)
    for camera in ("robot_pov_left_cam", "robot_pov_right_cam", "wrist_left_cam", "wrist_right_cam"):
        variation = embodiment.get_variation(f"camera_extrinsics_{camera}")
        assert isinstance(variation, mounted.MountedCameraExtrinsicsVariation)
        assert variation.camera_name == camera
