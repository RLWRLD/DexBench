# SPDX-License-Identifier: Apache-2.0
"""record_demos.py's episode videos: a clip per episode from the scene camera, named after the demo it became, and an
index of every clip. Encoded with FFmpeg against stand-ins for the env, its camera and its recorder manager."""

from __future__ import annotations

import importlib
import json
import shutil
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("isaaclab.managers", exc_type=ImportError)
if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
    pytest.skip("needs ffmpeg and ffprobe", allow_module_level=True)


@pytest.fixture(scope="module")
def video():
    """The record_demos.py wrapper's helper module, beside dexbench_arena.video.video_io.VideoWriter, which it
    writes with."""
    return importlib.import_module("dexbench_arena.video.episode_video")


def _env(camera_name):
    camera = NS(data=NS(output={"rgb": torch.zeros(1, 48, 64, 3, dtype=torch.uint8)}))
    return NS(step_dt=1.0 / 30.0, scene=NS(sensors={camera_name: camera}),
              recorder_manager=NS(exported_successful_episode_count=0))


def _steps(term, env, n):
    for i in range(n):
        next(iter(env.scene.sensors.values())).data.output["rgb"][:] = i
        term.record_post_step()


def test_every_episode_gets_a_clip_and_an_exported_one_its_demo_name(video, tmp_path):
    env = _env(video.CAMERA_NAME)
    cfg = video.EpisodeVideoRecorderCfg(output_dir=str(tmp_path), quality=30, codec="libx264")
    term = cfg.class_type(cfg, env)
    _steps(term, env, 5)
    term.record_pre_reset([0])
    env.recorder_manager.exported_successful_episode_count = 1
    term.record_pre_reset([0])
    term.record_post_reset([0])
    _steps(term, env, 3)
    term.record_pre_reset([0])
    term.record_post_reset([0])
    _steps(term, env, 4)
    term.close("dataset.hdf5")

    index = [json.loads(line) for line in (tmp_path / "videos.jsonl").read_text().splitlines()]
    assert [(r["video"], r["frames"], r["demo"]) for r in index] == [
        ("ep000_demo_0.mp4", 5, "demo_0"), ("ep001.mp4", 3, None), ("ep002.mp4", 4, None)]
    assert sorted(p.name for p in tmp_path.glob("*.mp4")) == ["ep000_demo_0.mp4", "ep001.mp4", "ep002.mp4"]

    env = _env(video.CAMERA_NAME)
    term = cfg.class_type(cfg, env)
    _steps(term, env, 2)
    term.close("dataset.hdf5")
    assert (tmp_path / "ep003.mp4").exists() and len((tmp_path / "videos.jsonl").read_text().splitlines()) == 4


def test_the_demo_that_ends_a_session_at_its_demo_count_keeps_its_demo_name(video, tmp_path):
    """record_demos.py stops at --num_demos right after the success export, with no reset after it."""
    env = _env(video.CAMERA_NAME)
    cfg = video.EpisodeVideoRecorderCfg(output_dir=str(tmp_path), quality=30, codec="libx264")
    term = cfg.class_type(cfg, env)
    _steps(term, env, 4)
    term.record_pre_reset([0])
    env.recorder_manager.exported_successful_episode_count = 1
    term.close("dataset.hdf5")
    index = [json.loads(line) for line in (tmp_path / "videos.jsonl").read_text().splitlines()]
    assert [(r["video"], r["demo"]) for r in index] == [("ep000_demo_0.mp4", "demo_0")]


def test_the_camera_sits_at_the_task_viewer_eye_looking_at_its_target(video, tmp_path):
    pytest.importorskip("isaaclab.sensors", exc_type=ImportError)
    from isaaclab.envs.common import ViewerCfg, _apply_deprecated_viewer_cfg
    from isaaclab.utils.math import quat_apply

    eye, lookat = (2.6, 0.15, 2.2), (2.6, 1.4, 0.8)
    cfg = NS(viewer=ViewerCfg(eye=eye, lookat=lookat), sim=NS(default_visualizer_cfg=None), scene=NS(), recorders=NS())
    _apply_deprecated_viewer_cfg(cfg)
    cfg.viewer = ViewerCfg()
    video.install_episode_video(cfg, str(tmp_path))
    camera = getattr(cfg.scene, video.CAMERA_NAME)
    assert camera.offset.convention == "opengl" and camera.offset.pos == eye
    forward = quat_apply(torch.tensor([camera.offset.rot]), torch.tensor([[0.0, 0.0, -1.0]]))[0]
    want = torch.tensor(lookat) - torch.tensor(eye)
    assert torch.allclose(forward, want / want.norm(), atol=1e-5)
    assert cfg.recorders.record_episode_video.output_dir == str(tmp_path)


def test_the_camera_takes_the_rtx_settings_the_scene_cameras_carry(video, tmp_path):
    """Isaac RTX's global settings are process-wide and Isaac Lab refuses two cameras that disagree; an E2 scene's
    cameras carry the delivery's tonemap and exposure (cli.set_rtx_global_settings)."""
    pytest.importorskip("isaaclab.sensors", exc_type=ImportError)
    from isaaclab.envs.common import ViewerCfg
    from isaaclab_physx.renderers import IsaacRtxRendererCfg
    from isaaclab_tasks.utils.presets import set_isaac_rtx_global_settings

    head = NS(renderer_cfg=IsaacRtxRendererCfg())
    set_isaac_rtx_global_settings(head.renderer_cfg, carb_settings={"/rtx/post/tonemap/op": 6})
    cfg = NS(viewer=ViewerCfg(eye=(2.6, 0.15, 2.2), lookat=(2.6, 1.4, 0.8)), scene=NS(head=head), recorders=NS())
    video.install_episode_video(cfg, str(tmp_path))
    rendered = getattr(cfg.scene, video.CAMERA_NAME).renderer_cfg
    assert rendered.global_settings == head.renderer_cfg.global_settings != IsaacRtxRendererCfg().global_settings
