# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""The camera-frame transform every DexBench GR00T client applies before a frame leaves it."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

IMAGE_PREPROCESSING = "rgb_area_v1"
"""Name of this transform in evaluation records; a different transform gets a new name."""


def resize_rgb_frames(frames: np.ndarray, height: int, width: int) -> np.ndarray:
    """Return ``(N, height, width, 3)`` uint8 RGB frames, area-downscaled from ``frames``."""

    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[-1] < 3 or frames.dtype != np.uint8:
        raise ValueError(f"Expected (N, H, W, C>=3) uint8 RGB frames, got {frames.shape} {frames.dtype}")
    frames = np.ascontiguousarray(frames[..., :3])
    check_resizable(frames.shape[1:3], height, width)
    if frames.shape[1:3] == (height, width):
        return frames
    return np.stack([cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA) for frame in frames])


def check_resizable(source_hw: tuple[int, int], height: int, width: int) -> None:
    """Raise the :func:`resize_rgb_frames` refusal for frames of ``source_hw``, if it has one."""

    source_height, source_width = (int(v) for v in source_hw)
    if (source_height, source_width) == (height, width):
        return
    if source_height < height or source_width < width:
        raise ValueError(
            f"Camera frames are {source_width}x{source_height}, smaller than the policy's "
            f"{width}x{height}; render at least at the policy resolution instead of upsampling."
        )
    if abs(source_height * width - source_width * height) > width:
        raise ValueError(
            f"Camera frames are {source_width}x{source_height}, a different aspect ratio from the "
            f"policy's {width}x{height}; render at the policy's aspect ratio. No checkpoint was "
            "trained on stretched, letterboxed or cropped frames."
        )


def image_preprocessing_record(source_hw: tuple[int, int] | None, target_hw: tuple[int, int]) -> dict[str, Any]:
    """Describe the transform :func:`resize_rgb_frames` applies, for an evaluation record."""

    return {
        "name": IMAGE_PREPROCESSING,
        "source_hw": None if source_hw is None else [int(v) for v in source_hw],
        "target_hw": [int(v) for v in target_hw],
        "resize": "none" if source_hw is not None and tuple(source_hw) == tuple(target_hw) else "cv2.INTER_AREA",
        "channels": "RGB",
    }
