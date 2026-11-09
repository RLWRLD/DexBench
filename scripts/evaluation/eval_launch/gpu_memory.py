"""The peak memory used on a simulator's GPU while its unit runs, as ``nvidia-smi`` reports it for the whole GPU: the
simulator's own, and every process sharing the GPU with it (its station's server, other simulators of a layout that
shares it). What tells whether a layout fits, which Ray's fractional GPUs do not hold any process to (``layout``).

Standard library only. Without ``nvidia-smi``, or without one GPU to name, nothing is measured (``None``).
"""

from __future__ import annotations

import shutil
import subprocess
import threading
from collections.abc import Callable
from typing import Any

POLL_S = 5.0
"""How often the GPU is read while a unit runs: memory grows over seconds (a Run's build, the first renders)."""


def query(gpu: str | None) -> tuple[float, float] | None:
    """``(used, total)`` MiB of ``gpu`` (an index or UUID, as ``CUDA_VISIBLE_DEVICES`` names it; None: the only GPU this
    process sees), or None when ``nvidia-smi`` cannot say."""
    if shutil.which("nvidia-smi") is None or (gpu is not None and "," in gpu):
        return None
    command = ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"]
    if gpu is not None:
        command += ["-i", gpu]
    try:
        lines = subprocess.run(command, capture_output=True, text=True, timeout=30, check=True).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        return None
    if len(lines) != 1:
        return None
    try:
        used, total = (float(value) for value in lines[0].split(","))
    except ValueError:
        return None
    return used, total


class Peak:
    """Reads ``gpu`` every :data:`POLL_S` from ``start`` to ``stop``; :meth:`result` is the peak."""

    def __init__(self, gpu: str | None, read: Callable[[str | None], tuple[float, float] | None] | None = None):
        self.gpu, self.read = gpu, read or query
        self.peak_used = self.total = None
        self.samples = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> None:
        reading = self.read(self.gpu)
        if reading is not None:
            used, self.total = reading
            self.peak_used = used if self.peak_used is None else max(self.peak_used, used)
            self.samples += 1

    def _run(self) -> None:
        self._sample()
        while not self._stop.wait(POLL_S):
            self._sample()

    def start(self) -> Peak:
        self._thread.start()
        return self

    def stop(self) -> dict[str, Any] | None:
        self._stop.set()
        self._thread.join(timeout=60)
        return self.result()

    def result(self) -> dict[str, Any] | None:
        """``{"gpu", "peak_used_mib", "total_mib", "samples"}``, or None when nothing was read."""
        if not self.samples:
            return None
        return {"gpu": self.gpu, "peak_used_mib": self.peak_used, "total_mib": self.total, "samples": self.samples}
