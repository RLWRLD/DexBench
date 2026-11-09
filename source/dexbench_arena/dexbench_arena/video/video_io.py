# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""H.264 MP4 encoding of camera frames (``VideoWriter``) and the FFprobe metadata inspection that video writers
and readers share."""
from __future__ import annotations

import json
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from contextlib import suppress
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import numpy as np


def probe_video(path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,nb_frames,avg_frame_rate,codec_name,pix_fmt", "-of", "json", str(path)],
        capture_output=True, text=True, timeout=30, check=True,
    )
    streams = json.loads(result.stdout)["streams"]
    if len(streams) != 1:
        raise ValueError(f"Expected one video stream: {path}")
    return streams[0]


@lru_cache(maxsize=None)
def resolve_codec(requested: str, width: int, height: int) -> str:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"{tool} is required for external video storage")
    if requested not in {"auto", "libx264", "h264_nvenc"}:
        raise ValueError(f"Unsupported video codec: {requested}")
    if requested == "auto":
        try:
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                 f"color=size={width}x{height}:rate=30", "-frames:v", "1", "-c:v",
                 "h264_nvenc", "-f", "null", "-"],
                capture_output=True, timeout=30,
            )
        except subprocess.TimeoutExpired:
            return "libx264"
        return "h264_nvenc" if result.returncode == 0 else "libx264"
    return requested


class VideoWriter:
    """One encoder process per camera, with bounded queues and no dropped frames."""

    def __init__(self, path, shape, fps, *, codec="auto", quality=18, queue_size=8):
        self.path = Path(path)
        self.shape = tuple(shape)
        if (len(self.shape) != 3 or self.shape[2] != 3
                or min(self.shape[:2]) <= 0 or any(v % 2 for v in self.shape[:2])):
            raise ValueError(f"H.264 yuv420p requires even HxWx3 RGB frames, got {shape}")
        if not 0 <= quality <= 51 or queue_size < 1:
            raise ValueError("video quality must be 0..51 and queue size must be positive")
        self.rate = Fraction(str(fps)).limit_denominator(1_000_000)
        if self.rate <= 0:
            raise ValueError("Video FPS must be positive")
        height, width, _ = self.shape
        self.codec = resolve_codec(codec, width, height)
        self.quality = quality
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stderr = tempfile.TemporaryFile()
        self._queue = queue.Queue(maxsize=queue_size)
        self._error = None
        self._done = threading.Event()
        self._closed = False
        self.frames = 0
        self._written = 0
        options = (["-preset", "p4", "-rc", "vbr", "-cq", str(quality), "-b:v", "0"]
                   if self.codec == "h264_nvenc"
                   else ["-preset", "veryfast", "-crf", str(quality)])
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s:v", f"{width}x{height}",
            "-framerate", str(self.rate), "-i", "pipe:0", "-an", "-c:v", self.codec,
            *options, "-threads", "1", "-pix_fmt", "yuv420p", "-g",
            str(max(1, round(float(self.rate)))), "-bf", "0", "-movflags", "+faststart", str(self.path),
        ]
        try:
            self._process = subprocess.Popen(command, stdin=subprocess.PIPE,
                                             stdout=subprocess.DEVNULL, stderr=self._stderr)
        except BaseException:
            self._stderr.close()
            raise
        self._thread = threading.Thread(target=self._encode, daemon=True)
        self._thread.start()

    def _encode(self):
        try:
            while True:
                frame = self._queue.get()
                if frame is None:
                    break
                self._process.stdin.write(memoryview(frame).cast("B"))
                self._written += 1
            self._process.stdin.close()
            if self._process.wait(timeout=60) != 0:
                raise RuntimeError("FFmpeg exited unsuccessfully")
        except BaseException as error:
            self._error = error
            if self._process.poll() is None:
                self._process.kill()
            self._process.wait()
        finally:
            self._done.set()

    def _check(self):
        if self._error is not None:
            self._stderr.seek(0)
            detail = self._stderr.read().decode(errors="replace")[-4000:]
            raise RuntimeError(f"Video encoder failed for {self.path}: {detail}") from self._error

    def _put(self, frame):
        deadline = time.monotonic() + 120
        while True:
            self._check()
            if self._done.is_set():
                self._check()
                raise RuntimeError("Video encoder stopped before all frames were submitted")
            try:
                self._queue.put(frame, timeout=0.1)
                return
            except queue.Full:
                if time.monotonic() > deadline:
                    raise TimeoutError("Video encoder queue did not make progress")

    def write(self, frame):
        if self._closed:
            raise RuntimeError("Video writer is closed")
        frame = np.asarray(frame)
        if frame.shape != self.shape or frame.dtype != np.uint8:
            raise ValueError(f"Expected uint8 {self.shape}, got {frame.dtype} {frame.shape}")
        self._put(frame.copy(order="C"))
        self.frames += 1

    def finish(self, expected_frames):
        try:
            self._put(None)
            self._thread.join(timeout=120)
            if self._thread.is_alive():
                raise TimeoutError("Video encoder did not finish")
            self._check()
            if self.frames != expected_frames or self._written != expected_frames:
                raise ValueError("Video/action frame counts differ")
            info = probe_video(self.path)
            if (int(info["nb_frames"]) != expected_frames
                    or (int(info["height"]), int(info["width"]), 3) != self.shape
                    or Fraction(info["avg_frame_rate"]) != self.rate):
                raise ValueError(f"Encoded video disagrees with submitted frames: {info}")
            return info
        finally:
            self.abort()

    def abort(self):
        if self._closed:
            return
        if self._process.poll() is None:
            self._process.kill()
            self._process.wait()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=5)
        with suppress(BrokenPipeError, OSError):
            self._process.stdin.close()
        self._stderr.close()
        self._closed = True
        while not self._queue.empty():
            self._queue.get_nowait()
