# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Explicit diagnostic gates; these do not change task success predicates."""

import math

import numpy as np
import torch
from isaaclab.utils.math import quat_error_magnitude


def evaluate_hold(summary: dict, tracked_objects) -> dict[str, bool]:
    """The hold gates of one trial. An object is supported when a contact under it (normal at least
    60 deg from horizontal, gap at most 1.5 mm) is with anything but the robot: the experiment
    names no support per object, and resting on the robot is not resting."""
    initial, final = summary["initial_contacts"], summary["final_contacts"]

    def depth(rows, *, robot=False):
        relevant = (
            [r for r in rows if r["robot"]]
            if robot
            else [r for r in rows if r["objects"] and not r["robot"]]
        )
        return max([max(0.0, -r["gap_m"]) for r in relevant] or [0.0])

    supported = {n: False for n in tracked_objects}
    for row in final:
        for name in row["objects"]:
            on_a = f"/{name}/" in row["shape0"]
            nz = -row["normal"][2] if on_a else row["normal"][2]
            if not row["robot"] and nz > 0.5 and row["gap_m"] <= 0.0015:
                supported[name] = True
    return {
        "finite": summary["finite"],
        "finite_contacts": all(math.isfinite(v) for r in [*initial, *final] for v in [r["gap_m"], *r["normal"]]),
        "initial_object_penetration_le_0_2mm": depth(initial) <= 0.0002,
        "final_object_penetration_le_2mm": depth(final) <= 0.002,
        "initial_robot_penetration_le_0_2mm": depth(initial, robot=True) <= 0.0002,
        "final_robot_penetration_le_0_2mm": depth(final, robot=True) <= 0.0002,
        "objects_supported": all(supported.values()),
        "object_displacement_le_5mm": summary["max_displacement_m"] <= 0.005,
        "object_rotation_le_15deg": summary["max_rotation_deg"] <= 15,
        "settled_drift_le_2mm": summary["settled_drift_m"] <= 0.002,
        "settled_window_motion_le_0_5mm": summary["settled_window_motion_m"] <= 0.0005,
        "settled_window_rotation_le_0_5deg": summary["settled_window_rotation_deg"] <= 0.5,
    }


def window_motion(samples_by_object: dict, window: int) -> tuple[float, float]:
    """Largest position (m) and rotation (deg) change of any object between any two samples at most ``window`` apart."""

    worst_m = worst_deg = 0.0
    for samples in samples_by_object.values():
        a = np.asarray(samples, dtype=float)
        if len(a) < 2:
            continue
        q = torch.as_tensor(a[:, 3:7], dtype=torch.float64)
        q = q / q.norm(dim=1, keepdim=True)
        for i in range(0, max(1, len(a) - window)):
            span, qs = a[i:i + window + 1], q[i:i + window + 1]
            worst_m = max(worst_m, float(np.linalg.norm(np.ptp(span[:, :3], axis=0))))
            n = qs.shape[0]
            pairs = quat_error_magnitude(qs.repeat_interleave(n, 0), qs.repeat(n, 1))
            worst_deg = max(worst_deg, float(np.rad2deg(pairs.max())))
    return worst_m, worst_deg

