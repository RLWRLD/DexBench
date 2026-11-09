# Copyright (c) 2026, RLWRLD Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Quaternions are (x, y, z, w) everywhere at the pinned Isaac Lab."""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
IDENTITY_XYZW = (0.0, 0.0, 0.0, 1.0)
YAW_90_XYZW = (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))


def test_every_robot_cfg_declares_the_xyzw_identity():
    """Arena only overwrites ``init_state.rot`` when a scene sets a pose; the literal must be the identity."""
    pytest.importorskip("isaaclab.sim", exc_type=ImportError)
    import importlib
    import importlib.util
    import pkgutil

    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase

    import dexbench_arena.embodiments as embodiments

    for package in pkgutil.iter_modules(embodiments.__path__):
        robot_module = f"{embodiments.__name__}.{package.name}.{package.name}"
        if package.ispkg and importlib.util.find_spec(robot_module) is not None:
            importlib.import_module(robot_module)
    registry = AssetRegistry()
    checked, missing = [], []
    for name in registry.get_all_keys():
        cls = registry.get_asset_by_name(name)
        if name.startswith("dexbench_") and isinstance(cls, type) and issubclass(cls, EmbodimentBase):
            try:
                embodiment = cls()
            except FileNotFoundError as exc:
                missing.append(f"{name}: {exc}")
                continue
            assert tuple(embodiment.scene_config.robot.init_state.rot) == IDENTITY_XYZW, name
            checked.append(name)
    if not checked:
        pytest.skip("assets payload missing for every embodiment: " + "; ".join(missing))


@pytest.mark.parametrize("rotation", [IDENTITY_XYZW, YAW_90_XYZW, (0.5, 0.5, 0.5, 0.5)])
def test_procedural_assets_pass_the_pose_rotation_through(rotation):
    """``Pose.rotation_xyzw`` reaches ``InitialStateCfg.rot`` unchanged; the R3 slot frame keeps its yaw."""
    pytest.importorskip("isaaclab.sim", exc_type=ImportError)
    from isaaclab_arena.utils.pose import Pose

    from dexbench_arena.assets import object_library

    pose = Pose(position_xyz=(1.0, 2.0, 3.0), rotation_xyzw=rotation)
    assets = (
        object_library.DexBenchTaskFrame(instance_name="probe_frame", initial_pose=pose),
        object_library.DexBenchProceduralTable(initial_pose=pose),
        object_library.DexBenchProceduralBlock(instance_name="probe_block", initial_pose=pose),
        object_library.DexBenchTaskFrame(instance_name="default_frame"),
    )
    for asset in assets[:-1]:
        _, cfg = asset.get_object_cfg()
        assert tuple(cfg.init_state.rot) == pytest.approx(rotation)
        assert tuple(cfg.init_state.pos) == pytest.approx((1.0, 2.0, 3.0))
    assert tuple(assets[-1].get_object_cfg()[1].init_state.rot) == IDENTITY_XYZW


_WXYZ_IDENTITY = re.compile(
    r"(?:\brot(?:ation_xyzw)?\s*=\s*|\brotate_quat_by_yaw\(\s*)"
    r"\(\s*1(?:\.0)?\s*,\s*0(?:\.0)?\s*,\s*0(?:\.0)?\s*,\s*0(?:\.0)?\s*\)"
)


def test_no_rotation_field_receives_the_wxyz_identity():
    """``rot=(1, 0, 0, 0)`` is a 180-degree X flip in (x, y, z, w); the identity is (0, 0, 0, 1)."""
    files = sorted(path.relative_to(REPO_ROOT).as_posix() for top in ("source/dexbench_arena/dexbench_arena", "scripts")
                   for path in (REPO_ROOT / top).rglob("*.py"))
    offenders = []
    for rel in files:
        text = (REPO_ROOT / rel).read_text()
        for match in _WXYZ_IDENTITY.finditer(text):
            offenders.append(f"{rel}:{text.count(chr(10), 0, match.start()) + 1}: {match.group(0)}")
    assert not offenders, "w-first identity quaternion handed to an xyzw field or rotate_quat_by_yaw:\n" + "\n".join(
        offenders
    )
