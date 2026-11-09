# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""JSONL trace helpers for teleoperation diagnostics."""

from __future__ import annotations

import json
import math
import os
import threading
import time
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()
_COUNTERS: dict[str, int] = {}
_EVENT_COUNT = 0
_PATH: Path | None = None


def is_enabled() -> bool:
    """Return whether teleop tracing is enabled by environment."""

    flag = os.environ.get("DEXBENCH_TELEOP_TRACE", "").lower()
    return flag in {"1", "true", "yes", "on"} or bool(os.environ.get("DEXBENCH_TELEOP_TRACE_PATH"))


def get_trace_path() -> Path | None:
    """Return the configured trace path, or the default path when tracing is enabled."""

    global _PATH
    if _PATH is not None:
        return _PATH

    configured = os.environ.get("DEXBENCH_TELEOP_TRACE_PATH")
    if configured:
        _PATH = Path(configured).expanduser()
    elif is_enabled():
        _PATH = Path("/tmp/dexbench_teleop_trace.jsonl")
    return _PATH


def start_session(metadata: dict[str, Any] | None = None, *, truncate: bool = True) -> Path | None:
    """Create the trace file and write a session-start record."""

    if not is_enabled():
        return None

    path = get_trace_path()
    if path is None:
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    if truncate:
        with path.open("w", encoding="utf-8"):
            pass

    record_event("trace_session_start", metadata or {}, force=True)
    return path


def record_event(kind: str, payload: dict[str, Any] | None = None, *, step: int | None = None, force: bool = False) -> None:
    """Append one JSONL event if tracing is enabled."""

    global _EVENT_COUNT

    if not force and not is_enabled():
        return

    path = get_trace_path()
    if path is None:
        return

    max_events = _read_int_env("DEXBENCH_TELEOP_TRACE_MAX_EVENTS", 0)
    if max_events > 0 and _EVENT_COUNT >= max_events and not force:
        return

    key = kind
    if payload is not None and "side" in payload:
        key = f"{kind}:{payload['side']}"

    if not force:
        every = max(1, _read_int_env("DEXBENCH_TELEOP_TRACE_EVERY", 1))
        count = _COUNTERS.get(key, 0) + 1
        _COUNTERS[key] = count
        if (count - 1) % every != 0:
            return

    event = {
        "time": time.time(),
        "kind": kind,
        "step": step,
        "payload": to_jsonable(payload or {}),
    }

    line = json.dumps(event, separators=(",", ":"), allow_nan=False)
    with _LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file:
            file.write(line)
            file.write("\n")
        _EVENT_COUNT += 1


def to_jsonable(value: Any) -> Any:
    """Convert tensors, arrays, scalars, and containers into strict JSON values."""

    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [to_jsonable(item) for item in value]

    detach = getattr(value, "detach", None)
    cpu = getattr(value, "cpu", None)
    if callable(detach) and callable(cpu):
        return to_jsonable(value.detach().cpu().tolist())

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return to_jsonable(tolist())

    item = getattr(value, "item", None)
    if callable(item):
        return to_jsonable(item())

    return str(value)


def _read_int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default

