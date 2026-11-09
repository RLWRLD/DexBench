# SPDX-License-Identifier: Apache-2.0
"""Episode videos for record_demos.py: one MP4 per teleop episode from a fixed scene camera."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from dexbench_arena.video.video_io import VideoWriter

from isaaclab.managers import RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

CAMERA_NAME = "dexbench_episode_video_camera"
CAMERA_PRIM_PATH = "{ENV_REGEX_NS}/DexBenchEpisodeVideoCamera"
"""Under the environment's namespace: Isaac Lab's RTX camera at this pin renders a camera prim outside it black.
With one environment its origin is the world's, so the viewer's world-frame pose is the camera's offset."""
RESOLUTION = (1280, 720)
FOCAL_LENGTH_MM = 12.0
"""About 82 degrees across on the default 20.955 mm aperture: a work cell from two metres."""


def camera_frame(env, sensor_name: str = CAMERA_NAME):
    """Env 0's RGB image from a scene camera, as the last render left it: an HxWx3 uint8 array."""
    rgb = env.scene.sensors[sensor_name].data.output["rgb"]
    rgb = rgb.torch if hasattr(rgb, "torch") else rgb
    return rgb[0, ..., :3].cpu().numpy()


class EpisodeVideoRecorder(RecorderTerm):
    """Writes the scene camera's frame after every step; a clip per episode, named after the demo it became."""

    def __init__(self, cfg: EpisodeVideoRecorderCfg, env):
        super().__init__(cfg, env)
        self._dir = Path(cfg.output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._fps = cfg.fps or round(1.0 / env.step_dt)
        self._writer: VideoWriter | None = None
        taken = [int(p.stem[2:5]) for p in self._dir.glob("ep[0-9][0-9][0-9]*.mp4") if p.stem[2:5].isdigit()]
        self._episode = max(taken) + 1 if taken else 0
        self._exported_at_start: int | None = None
        self._finished: tuple[int, Path, int] | None = None
        self._failed = False

    def _exported(self) -> int:
        return int(self._env.recorder_manager.exported_successful_episode_count)

    def record_post_step(self):
        if self._failed:
            return None, None
        if self._exported_at_start is None:
            self._exported_at_start = self._exported()
        try:
            frame = camera_frame(self._env, self.cfg.sensor_name)
            if self._writer is None:
                self._writer = VideoWriter(self._dir / f"ep{self._episode:03d}.mp4", frame.shape, self._fps,
                                           codec=self.cfg.codec, quality=self.cfg.quality)
            self._writer.write(frame)
        except Exception as error:
            self._fail(error)
        return None, None

    def record_pre_reset(self, env_ids):
        if self._writer is not None and (env_ids is None or 0 in [int(e) for e in env_ids]):
            self._finish()
        return None, None

    def record_post_reset(self, env_ids):
        if self._finished is not None:
            exported = self._exported()
            self._publish(exported > (self._exported_at_start or 0))
            self._exported_at_start = exported
        return None, None

    def close(self, file_path):
        if self._writer is not None:
            self._finish()
        if self._finished is not None:
            self._publish(self._exported() > (self._exported_at_start or 0))

    def _finish(self) -> None:
        writer, self._writer = self._writer, None
        try:
            writer.finish(writer.frames)
        except Exception as error:
            self._fail(error)
            return
        self._finished = (self._episode, writer.path, writer.frames)
        self._episode += 1

    def _publish(self, exported: bool) -> None:
        episode, path, frames = self._finished
        self._finished = None
        demo = f"demo_{self._exported_at_start or 0}" if exported else None
        if demo is not None:
            path = path.rename(path.with_name(f"{path.stem}_{demo}.mp4"))
        with open(self._dir / "videos.jsonl", "a") as index:
            index.write(json.dumps({"episode": episode, "video": path.name, "frames": frames, "fps": self._fps,
                                    "demo": demo}) + "\n")
        print(f"[dexbench] episode video -> {path} ({frames} frames, {demo or 'not in the dataset'})", flush=True)

    def _fail(self, error: Exception) -> None:
        self._failed = True
        if self._writer is not None:
            self._writer.abort()
            self._writer = None
        print(f"[dexbench] episode video: stopped recording videos after an error: {error}", flush=True)


@configclass
class EpisodeVideoRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = EpisodeVideoRecorder

    output_dir: str = "videos"
    """Where the clips and ``videos.jsonl`` go."""

    sensor_name: str = CAMERA_NAME
    """The scene camera the frames come from (its ``rgb`` output)."""

    fps: int | None = None
    """Frames per second of the clips; None: the control rate, so a clip plays in simulated time."""

    codec: str = "auto"
    """``auto`` (NVENC when the GPU has it, else libx264), ``h264_nvenc`` or ``libx264``."""

    quality: int = 23
    """The encoder's constant quality (x264 CRF or NVENC CQ, 0-51)."""


def install_episode_video(env_cfg, output_dir: str) -> None:
    """Put the episode video camera and its recorder term on a composed env cfg (one environment, world frame)."""
    environments = getattr(env_cfg.scene, "num_envs", 1)
    if environments != 1:
        raise ValueError(f"--episode_video_dir films one environment at the world-frame viewer pose; the scene has "
                         f"{environments}")
    add_video_camera(env_cfg)
    env_cfg.recorders.record_episode_video = EpisodeVideoRecorderCfg(output_dir=str(output_dir))
    print(f"[dexbench] episode video: clips -> {output_dir}", flush=True)


def viewer_pose(env_cfg) -> tuple[str, tuple[float, ...], tuple[float, ...]]:
    """``(origin type, eye, look-at)`` of the task's viewer on a composed env cfg: where the viewport starts."""
    visualizer = getattr(getattr(env_cfg, "sim", None), "default_visualizer_cfg", None)
    source = env_cfg.viewer if visualizer is None else visualizer
    return getattr(source, "origin_type", "world"), tuple(source.eye), tuple(source.lookat)


def add_video_camera(env_cfg) -> None:
    """Put the episode video camera (scene sensor ``CAMERA_NAME``) at the task's viewer pose on a composed env cfg."""
    import torch

    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg
    from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix
    from isaaclab_physx.renderers import IsaacRtxRendererCfg

    origin_type, eye_xyz, lookat_xyz = viewer_pose(env_cfg)
    if origin_type != "world":
        raise ValueError(f"--episode_video_dir takes the task's viewer pose in the world frame, not {origin_type!r}")
    eye = torch.tensor([eye_xyz], dtype=torch.float32)
    rotation = quat_from_matrix(create_rotation_matrix_from_view(eye, torch.tensor([lookat_xyz]), up_axis="Z"))
    width, height = RESOLUTION
    renderer_cfg = IsaacRtxRendererCfg()
    for entry in vars(env_cfg.scene).values():
        other = getattr(entry, "renderer_cfg", None)
        if getattr(other, "renderer_type", None) == "isaac_rtx" and hasattr(other, "global_settings"):
            renderer_cfg.global_settings = copy.deepcopy(other.global_settings)
            break
    setattr(env_cfg.scene, CAMERA_NAME, CameraCfg(
        prim_path=CAMERA_PRIM_PATH,
        update_period=0.0,
        width=width,
        height=height,
        data_types=["rgb"],
        renderer_cfg=renderer_cfg,
        spawn=sim_utils.PinholeCameraCfg(focal_length=FOCAL_LENGTH_MM, clipping_range=(0.05, 50.0)),
        offset=CameraCfg.OffsetCfg(pos=eye_xyz, rot=tuple(rotation[0].tolist()), convention="opengl"),
    ))
    print(f"[dexbench] episode video: a {width}x{height} camera at {eye_xyz} looking at {lookat_xyz}", flush=True)
