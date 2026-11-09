# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""``--teleop_target_marker`` adds the marker's event term only when it is on (and the slot region's for a task
that has one), and a replay of a stored command line that carries it leaves it out. Kit-free."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("isaaclab_arena.assets.register", reason="needs IsaacLab-Arena importable")
pytest.importorskip("isaaclab.markers", reason="needs Isaac Lab importable")

from dexbench_arena.environments.manipulation.scene_environment_base import (  # noqa: E402
    DexBenchEnvironmentCfg,
    teleop_target_marker_callback,
)
from dexbench_arena.tasks.place_in_slot import DexBenchPlaceInSlotTask  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts/imitation_learning"))
from dexbench_arena.recording.rerecord_metadata import without_operator_switches  # noqa: E402

TASK = SimpleNamespace(object_name="r2_target_drink")


def _composed_events(enabled: bool, task=TASK) -> dict:
    callback = teleop_target_marker_callback(None, DexBenchEnvironmentCfg(teleop_target_marker=enabled), task)
    env_cfg = SimpleNamespace(events=SimpleNamespace())
    if callback is not None:
        env_cfg = callback(env_cfg)
    return vars(env_cfg.events)


def test_the_marker_is_composed_only_when_the_flag_is_on():
    assert _composed_events(False) == {}
    term = _composed_events(True)["teleop_target_marker"]
    assert term.mode == "startup"
    assert term.params == {"object_name": "r2_target_drink"}


def _slot_task() -> DexBenchPlaceInSlotTask:
    return DexBenchPlaceInSlotTask(
        object_name="supply_item", object_center_offset_xyz=(0.0, 0.0, 0.05), slot_center_xy=(0.9555, 0.7365),
        slot_center_halfsize_xy=(0.02, 0.0125), object_half_extents_xy=(0.025, 0.07), shelf_top_z_m=1.0,
        center_height_range_m=(0.02, 0.04), workspace_xy_bounds=((0.0, 0.0), (2.0, 2.0)), object_dropped_min_z_m=0.3,
        front_plane_xy=(0.9805, 0.7365), outward_xy=(1.0, 0.0))


def test_a_slot_task_also_gets_its_region_drawn():
    assert _composed_events(False, _slot_task()) == {}
    events = _composed_events(True, _slot_task())
    assert events["teleop_target_marker"].params == {"object_name": "supply_item"}
    region = events["teleop_slot_marker"]
    assert region.mode == "startup"
    assert region.params["center_xy"] == (0.9555, 0.7365)
    assert region.params["center_halfsize_xy"] == (0.02, 0.0125)
    assert region.params["object_half_extents_xy"] == (0.025, 0.07)
    assert region.params["top_z_m"] == 1.0
    assert region.params["seated"]["min_up_alignment"] == 0.90
    assert "neighbor_names" not in region.params["seated"] and "hold_time_s" not in region.params["seated"]


def test_the_flag_is_refused_for_a_task_without_a_target():
    with pytest.raises(ValueError, match="names no target object"):
        teleop_target_marker_callback(None, DexBenchEnvironmentCfg(teleop_target_marker=True), SimpleNamespace())


def test_a_replay_of_an_older_sidecar_leaves_the_marker_out(tmp_path):
    """Some sidecars (the published DB-R2 reteleop ones) carry the switch."""
    sidecar = tmp_path / "demo.job.json"
    sidecar.write_text(json.dumps({"env": "dexbench_retail_v1_environment", "args": [
        "--task", "retrieve_beverage_from_clutter", "--teleop_target_marker", "--r2_target_sku", "TC1",
        "--teleop-session", "cage_l2_package.pose.enabled=false"]}))
    argv = ["--input_file", "demo.hdf5", "dexbench_retail_v1_environment", *json.loads(sidecar.read_text())["args"]]
    kept, dropped = without_operator_switches(argv)
    assert kept == ["--input_file", "demo.hdf5", "dexbench_retail_v1_environment", "--task",
                    "retrieve_beverage_from_clutter", "--r2_target_sku", "TC1", "cage_l2_package.pose.enabled=false"]
    assert dropped == ["--teleop_target_marker", "--teleop-session"]
