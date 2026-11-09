# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Golden-equality probe for every resolved camera config in the repo."""

from __future__ import annotations

import importlib.util
import json
import pytest
from dataclasses import fields, is_dataclass
from pathlib import Path

GOLDEN_PATH = Path(__file__).with_name("data") / "camera_catalog_golden.json"

SCALES = ("1.0", "0.25")


def _serialize(value):
    """Recursively render a config value as JSON-safe, order-stable, exact data."""
    if is_dataclass(value) and not isinstance(value, type):
        out = {"__class__": type(value).__name__}
        for f in sorted(fields(value), key=lambda f: f.name):
            out[f.name] = _serialize(getattr(value, f.name))
        return out
    if isinstance(value, type):
        return f"<class {value.__module__}.{value.__qualname__}>"
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, dict):
        return {str(k): _serialize(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_serialize(v) for v in value]
    return repr(value)


def _rig_instances(scale: float):
    """Yield ``(label, camera_cfg_instance)`` for every camera rig in the repo."""
    from dexbench_arena.embodiments.rby1_wuji2.rby1_wuji2 import RBY1_WUJI2_HEAD_RIG, Rby1Wuji2CameraCfg

    for is_tiled in (True, False):
        suffix = "tiled" if is_tiled else "plain"

        rby1_wuji2 = Rby1Wuji2CameraCfg(**RBY1_WUJI2_HEAD_RIG.build(scale))
        rby1_wuji2.set_use_tiled_camera(is_tiled)
        yield f"rby1_wuji2/{suffix}", rby1_wuji2.get_cfg()

def resolve_camera_configs() -> dict:
    """Resolve every camera on every rig, at every pinned scale, into plain data."""
    from isaaclab.sensors import CameraCfg

    dump: dict[str, dict] = {}
    for scale in SCALES:
        for label, rig in _rig_instances(float(scale)):
            for f in sorted(fields(rig), key=lambda f: f.name):
                camera = getattr(rig, f.name)
                if not isinstance(camera, CameraCfg):
                    continue
                dump[f"scale={scale}/{label}/{f.name}"] = _serialize(camera)
    return dump


def _dumps(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pxr") is None,
    reason="needs the Isaac Sim runtime (pxr); run inside the isaaclab_arena container",
)


def test_camera_configs_match_golden():
    assert GOLDEN_PATH.exists(), f"missing golden file {GOLDEN_PATH}; regenerate with --write"
    golden = json.loads(GOLDEN_PATH.read_text())
    resolved = resolve_camera_configs()

    missing = sorted(set(golden) - set(resolved))
    added = sorted(set(resolved) - set(golden))
    assert not missing, f"cameras disappeared from the resolved set: {missing}"
    assert not added, f"new cameras appeared and are not in the golden: {added}"

    drifted = {k: (golden[k], resolved[k]) for k in sorted(golden) if golden[k] != resolved[k]}
    assert not drifted, (
        "resolved camera configs differ from the golden calibration:\n"
        + "\n".join(f"  {k}\n    golden:   {g}\n    resolved: {r}" for k, (g, r) in drifted.items())
    )


def test_observation_keys_are_pinned():
    """The dataset/policy-facing camera names must never be renamed by a refactor."""
    golden = json.loads(GOLDEN_PATH.read_text()) if GOLDEN_PATH.exists() else {}
    names = {key.rsplit("/", 1)[-1] for key in golden}
    assert names == {"robot_pov_left_cam", "robot_pov_right_cam"}, names


if __name__ == "__main__":
    import sys

    if "--write" not in sys.argv:
        raise SystemExit("pass --write to regenerate the golden calibration file")
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(_dumps(resolve_camera_configs()))
    print(f"wrote {GOLDEN_PATH}")


@pytest.mark.parametrize("embodiment_name", [
    "DexBenchRby1Wuji2Embodiment",
    "DexBenchRby1Wuji2JointEmbodiment",
])
@pytest.mark.parametrize("scale, expected", [(None, (640, 360)), (0.25, (320, 180))])
def test_default_resolution_and_explicit_overrides(monkeypatch, embodiment_name, scale, expected):
    from dexbench_arena.embodiments.rby1_wuji2 import rby1_wuji2

    monkeypatch.setenv("DEXBENCH_CAMERA_SCALE", "1.0")
    cls = getattr(rby1_wuji2, embodiment_name)
    low = cls(camera_scale=scale).camera_config
    high = cls(camera_scale=1.0).camera_config
    for name in low.camera_names():
        small, native = getattr(low, name), getattr(high, name)
        assert (small.width, small.height) == expected
        assert (native.width, native.height) == (1280, 720)
        assert small.renderer_cfg.global_settings.antialiasing_mode == "DLAA"
        assert native.renderer_cfg.global_settings.antialiasing_mode == "DLAA"
        a, b = _serialize(small), _serialize(native)
        for key in ('width', 'height'):
            del a[key], b[key]
        assert a == b
