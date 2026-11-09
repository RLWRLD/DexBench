# Copyright (c) 2026, RLWRLD Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pxr") is None,
    reason="needs the Isaac Sim runtime (pxr); run inside the DexBench container",
)


def test_camera_body_mesh_paths_match_the_staged_v2_asset():
    from pxr import Usd, UsdGeom

    from dexbench_arena.embodiments.rby1_wuji2.constants import RBY1_WUJI2_USD_PATH
    from dexbench_arena.embodiments.rby1_wuji2.rby1_wuji2 import RBY1_WUJI2_CAMERA_BODY_MESH_RELATIVE_PATHS

    if not RBY1_WUJI2_USD_PATH.exists():
        pytest.skip(f"staged RBY1 Wuji v2 asset is unavailable: {RBY1_WUJI2_USD_PATH}")

    stage = Usd.Stage.Open(str(RBY1_WUJI2_USD_PATH))
    root_path = stage.GetDefaultPrim().GetPath()
    for relative_path in RBY1_WUJI2_CAMERA_BODY_MESH_RELATIVE_PATHS:
        prim = stage.GetPrimAtPath(root_path.AppendPath(relative_path))
        assert prim.IsValid(), f"missing camera housing prim: {root_path}/{relative_path}"
        assert prim.IsA(UsdGeom.Mesh), f"camera housing prim is not a mesh: {prim.GetPath()}"


def test_camera_body_visibility_uses_the_exact_descendant_path(monkeypatch):
    from dexbench_arena.embodiments.rby1_wuji2 import rby1_wuji2

    class Prim:
        def GetPath(self):
            return "/World/envs/env_0/Robot/ego_zed_m_camera/visuals/geom_0"

    class Scene:
        env_regex_ns = "/World/envs/env_.*"

    class Env:
        scene = Scene()

    queries = []
    hidden = []
    monkeypatch.setattr(rby1_wuji2.sim_utils, "find_matching_prims", lambda path: queries.append(path) or [Prim()])
    monkeypatch.setattr(
        rby1_wuji2.sim_utils,
        "set_prim_visibility",
        lambda prim, visible: hidden.append((prim, visible)),
    )

    rby1_wuji2.hide_rby1_wuji2_camera_body_mesh(Env())

    assert queries == ["/World/envs/env_.*/Robot/ego_zed_m_camera/visuals/geom_0"]
    assert len(hidden) == 1
    assert hidden[0][1] is False


def test_camera_body_visibility_rejects_a_zero_match(monkeypatch):
    from dexbench_arena.embodiments.rby1_wuji2 import rby1_wuji2

    class Scene:
        env_regex_ns = "/World/envs/env_.*"

    class Env:
        scene = Scene()

    monkeypatch.setattr(rby1_wuji2.sim_utils, "find_matching_prims", lambda path: [])

    with pytest.raises(RuntimeError, match="housing meshes were not found"):
        rby1_wuji2.hide_rby1_wuji2_camera_body_mesh(Env())
