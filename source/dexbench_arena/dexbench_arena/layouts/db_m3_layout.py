# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""M3 CAD sites and station layout, importable before Kit starts."""

import json
from dataclasses import replace

from dexbench_arena import ASSETS_DIR
from dexbench_arena.tasks.object_axis import ObjectAxis, TaskObject

PACKAGE = ASSETS_DIR / "object_library" / "center_hinge_station"
MANIFEST = json.loads((PACKAGE / "manifest.json").read_text())
BOLT_NAMES = tuple(bolt["id"] for bolt in MANIFEST["bolts"])
REFRIGERATOR_XYZ = (0.0, 0.0, 0.325)
SEATED_HEADS = tuple(
    (bolt["seated_head_xyz"][0], bolt["seated_head_xyz"][1], bolt["seated_head_xyz"][2] + REFRIGERATOR_XYZ[2])
    for bolt in MANIFEST["bolts"]
)
EPISODE_LENGTH_S = 60.0
TOOL_NAME = "power_driver"
TOOL_TIP_LOCAL = (0.0, -0.11580302566289902, 0.17578961337858345)
TOOL_AXIS_LOCAL = (0.0, -1.0, 0.0)
TOOL_USD = ASSETS_DIR / "object_library/power_drill/power_drill_1.0.0_SimReady/Power_Drill.usd"


def station_seated_heads(offset_y_m: float = 0.0, offset_x_m: float = 0.0, offset_z_m: float = 0.0):
    """Translate canonical CAD sites with the complete conveyor station."""
    return tuple((x + offset_x_m, y + offset_y_m, z + offset_z_m) for x, y, z in SEATED_HEADS)

M3_OBJECT_AXIS = ObjectAxis(
    field="m3_fixture", default="center_hinge_fixture",
    objects=(TaskObject("center_hinge_fixture", "center_hinge_fixture", "train",
                        "LG M-Next3 open-door centre-hinge station",
                        (1.094523549, 1.412809789, 1.859850364),
                        benchmark_asset_major_renames=((1, 2,
                            "assets/object_library/center_hinge_station/CHANGELOG.md"),)),),
    size_axes="source CAD bounds at 100-degree doors; runtime opening is recorded separately",
    scene_layout="m3_center_hinge_right_two_bolt_trigger_v4",
    fixture_profile={"m3_fastening_mode": "contact_torque"},
)

E3_M3_CONVEYOR_LENGTH_M = 2.4
E3_M3_CAD_MIN_Z_M = -0.030049996450543404
E3_M3_DEACTIVATED_SLOTS = (
    "Slot_Workstation", "Slot_Workbench", "Slot_BenchJig", "Slot_BenchFeedTray",
    "Slot_ToolZone_DriverStand", "Slot_ToolZone_NutTray",
    "Slot_StagingShelf", "Slot_PartsBin",
)
M3_E3_OBJECT_AXIS = replace(M3_OBJECT_AXIS, scene_layout="e3_m1_bay_m3_low_conveyor_v2")

ROBOT_BASE_ANCHORS_XYZ = {"A_m3": (-.1234, -1.07, 0.)}
ANCHOR_FACING_TARGETS_XY = {"A_m3": (.0766, -.35)}
