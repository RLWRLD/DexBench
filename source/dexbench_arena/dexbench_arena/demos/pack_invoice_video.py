"""Headless MP4 capture of the pack-invoice bench for driver scripts."""

from __future__ import annotations

from pathlib import Path

import av


DEFAULT_EYE = (3.55, 2.95, 1.75)
DEFAULT_TARGET = (4.75, 3.45, 1.05)
DEFAULT_RESOLUTION = (1280, 720)
DEFAULT_FOCAL = 18.0


class PackInvoiceVideoRecorder:
    """Capture the composed USD stage (including VBD meshes) directly to MP4."""

    def __init__(self, env, output_path: Path | None, *, eye=DEFAULT_EYE, target=DEFAULT_TARGET,
                 resolution=DEFAULT_RESOLUTION, focal: float = DEFAULT_FOCAL,
                 fps: int = 30, warmup_frames: int = 12, render: bool = True,
                 camera_path: str = "/World/dexbench_pack_invoice_video_camera"):
        import omni.kit.app
        import omni.replicator.core as rep
        import isaaclab.sim as sim_utils
        from pxr import Gf, UsdGeom

        self._av = av
        self._env = env
        self._render = render
        self._fps = max(int(fps), 1)
        self._container = None
        self.output_path: Path | None = None
        self.frames = 0
        width, height = int(resolution[0]), int(resolution[1])
        self._size = (width, height)

        stage = sim_utils.get_current_stage()
        camera = UsdGeom.Camera.Define(stage, camera_path)
        camera.CreateFocalLengthAttr(float(focal))
        view = Gf.Matrix4d()
        view.SetLookAt(
            Gf.Vec3d(*[float(v) for v in eye]),
            Gf.Vec3d(*[float(v) for v in target]),
            Gf.Vec3d(0.0, 0.0, 1.0),
        )
        UsdGeom.Xformable(camera.GetPrim()).AddTransformOp().Set(view.GetInverse())

        self._render_product = rep.create.render_product(camera_path, (width, height))
        self._annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        self._annotator.attach(self._render_product)

        self._app = omni.kit.app.get_app()
        self._pump_app = not any(viz.pumps_app_update() for viz in env.unwrapped.sim.visualizers)
        for _ in range(warmup_frames):
            self._render_frame()
        if output_path is not None:
            self.open(output_path)

    def _render_frame(self) -> None:
        self._env.unwrapped.sim.render()
        if self._pump_app:
            self._app.update()

    @property
    def is_open(self) -> bool:
        return self._container is not None

    def open(self, output_path: Path) -> None:
        """Start a new MP4 (finishing the current one first)."""

        self.close()
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.frames = 0
        self._container = self._av.open(str(self.output_path), mode="w")
        try:
            self._stream = self._container.add_stream("libx264", rate=self._fps)
        except Exception:
            self._stream = self._container.add_stream("mpeg4", rate=self._fps)
        self._stream.width, self._stream.height = self._size
        self._stream.pix_fmt = "yuv420p"
        self._stream.bit_rate = 8_000_000

    def capture(self) -> None:
        import numpy as np

        if self._container is None:
            return
        if self._render:
            self._env.unwrapped.sim.render()
        if self._pump_app:
            self._app.update()
        data = self._annotator.get_data()
        if isinstance(data, dict):
            data = data.get("data")
        rgb = np.asarray(data)
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            raise RuntimeError(f"recording camera returned an invalid RGB frame: shape={rgb.shape}")
        rgb = rgb[:, :, :3]
        if rgb.dtype != np.uint8:
            scale = 255.0 if np.issubdtype(rgb.dtype, np.floating) and float(rgb.max()) <= 1.0 else 1.0
            rgb = np.clip(rgb * scale, 0, 255).astype(np.uint8)
        frame = self._av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24")
        for packet in self._stream.encode(frame):
            self._container.mux(packet)
        self.frames += 1

    def close(self) -> None:
        if self._container is None:
            return
        try:
            for packet in self._stream.encode():
                self._container.mux(packet)
        finally:
            self._container.close()
            self._container = None
        print(f"[video] saved {self.frames} frames: {self.output_path}")
