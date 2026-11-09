# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Frozen, Kit-free inputs to DB-M1's start-state grasp."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

GRASP_PLAN_PATH = Path(__file__).with_name("db_m1_grasp_plan_v1.json")


@dataclass(frozen=True)
class GraspPad:
    key: str
    body: str
    contact_bodies: tuple[str, ...]
    joint_names: tuple[str, ...]


@dataclass(frozen=True)
class GraspHand:
    side: str
    wrist_body: str
    arm_joint_names: tuple[str, ...]
    fingers: tuple[tuple[str, float], ...]
    wrist_in_drawer: tuple[float, ...]
    pads: tuple[GraspPad, ...]


@dataclass(frozen=True)
class GraspSite:
    name: str
    position_m: tuple[float, ...]
    normal: tuple[float, ...]
    press_m: float
    extent: tuple[float, ...]
    extent_x: tuple[float, ...]
    extent_z: tuple[float, ...]


@dataclass(frozen=True)
class GraspController:
    physics_dt_s: float
    close_s: float
    thumb_steps: int
    hold_steps: int
    wrap_open: float
    pad_step: float
    arm_step: float
    wrist_dls_lambda: float
    pad_dls_squared_lambda: float
    hook_push: tuple[float, ...]
    thumb_press_m: float
    force_band: bool
    f_hook_on_n: float
    f_thumb_on_n: float
    preopen: float
    approach_mm: float
    place_servo: bool
    force_cap: bool
    closed_loop: bool
    legacy_exit: bool
    start_depth_m: float
    base_noise: tuple[float, ...]
    grasp_tilt_deg: tuple[float, ...]
    torso_joint_names: tuple[str, ...]
    torso_stiffness: float
    torso_damping: float


@dataclass(frozen=True)
class InsertionPosture:
    name: str
    depth_m: float
    joint_positions: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class InsertionPlan:
    rear_lead_m: tuple[float, ...]
    sensor_limits: tuple[tuple[str, float], ...]
    posture_keyframes: tuple[InsertionPosture, ...]


@dataclass(frozen=True)
class GraspPlan:
    source_sha256: tuple[tuple[str, str], ...]
    keyframe: str
    joint_positions: tuple[tuple[str, float], ...]
    keyframe_joint_names: tuple[str, ...]
    hands: tuple[GraspHand, ...]
    hook_fingers: tuple[str, ...]
    contact_pt_mm: tuple[tuple[str, tuple[float, ...]], ...]
    clamp_point_m: tuple[float, ...]
    drawer_sites: tuple[GraspSite, ...]
    right_palm_centre: tuple[float, ...]
    base_position_m: tuple[float, ...]
    base_yaw_deg: float
    drawer_position_m: tuple[float, ...]
    controller: GraspController
    insertion: InsertionPlan


def load_grasp_plan(path: str | Path = GRASP_PLAN_PATH) -> GraspPlan:
    """Load immutable plan values, rejecting non-finite values and invalid poses."""
    data = json.loads(Path(path).read_text())

    def check_finite(value):
        if isinstance(value, dict):
            for item in value.values():
                check_finite(item)
        elif isinstance(value, list):
            for item in value:
                check_finite(item)
        elif isinstance(value, (float, int)) and not math.isfinite(value):
            raise ValueError("Grasp plan numbers must be finite")

    check_finite(data)
    if data["schema"] != "dexbench.m1_grasp_plan/1":
        raise ValueError("Unsupported DB-M1 grasp plan schema")

    def vector(values, length):
        result = tuple(float(v) for v in values)
        if len(result) != length or not all(math.isfinite(v) for v in result):
            raise ValueError(f"Expected {length} finite numbers")
        return result

    def pose(row):
        result = vector(row["position"] + row["quat_xyzw"], 7)
        if abs(math.sqrt(sum(v * v for v in result[3:])) - 1) > 1e-6:
            raise ValueError("Grasp pose quaternion must be unit xyzw")
        return result

    def joints(row):
        result = tuple((name, float(v)) for name, v in row.items())
        if not all(math.isfinite(v) for _, v in result):
            raise ValueError("Joint values must be finite")
        return result

    hands = tuple(
        GraspHand(
            h["side"],
            h["wrist_body"],
            tuple(h["arm_joint_names"]),
            joints(h["fingers"]),
            pose(h["wrist_in_drawer"]),
            tuple(
                GraspPad(
                    p["key"],
                    p["body"],
                    tuple(p["contact_bodies"]),
                    tuple(p["joint_names"]),
                )
                for p in h["pads"]
            ),
        )
        for h in data["hands"]
    )
    sites = tuple(
        GraspSite(
            name,
            vector(s["pos"], 3),
            vector(s["normal"], 3),
            float(s["press_m"]),
            tuple(s.get("extent", ())),
            tuple(s.get("extent_x", ())),
            tuple(s.get("extent_z", ())),
        )
        for name, s in data["drawer_sites"].items()
    )
    controller = dict(data["controller"])
    for name in ("hook_push", "base_noise", "grasp_tilt_deg", "torso_joint_names"):
        controller[name] = tuple(controller[name])
    palm = data["right_palm_centre"]
    insertion = data["insertion"]
    postures = tuple(
        InsertionPosture(row["name"], float(row["depth_m"]), joints(row["joint_positions"]))
        for row in insertion["posture_keyframes"]
    )
    if not postures or any(a.depth_m >= b.depth_m for a, b in zip(postures[:-1], postures[1:])):
        raise ValueError("Insertion posture depths must be strictly increasing")
    return GraspPlan(
        tuple(data["source_sha256"].items()),
        data["keyframe"],
        joints(data["joint_positions"]),
        tuple(data["keyframe_joint_names"]),
        hands,
        tuple(data["hook_fingers"]),
        tuple((k, vector(v, 3)) for k, v in data["contact_pt_mm"].items()),
        vector(data["clamp_point_m"], 3),
        sites,
        vector(palm["pos"] + palm["quat"], 7),
        vector(data["base_position_m"], 3),
        float(data["base_yaw_deg"]),
        vector(data["drawer_position_m"], 3),
        GraspController(**controller),
        InsertionPlan(vector(insertion["rear_lead_m"], 3), joints(insertion["sensor_limits"]), postures),
    )
