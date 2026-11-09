# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""The pack-invoice scene's ranges: how far each part of the layout moves, and as what it is drawn."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

from dexbench_arena.assets.e2_logistics_cell import E2_PACK_INVOICE_BOX_SCALE_XYZ

PACK_INVOICE_INSTRUCTION = "Pick up the invoice sheet and put it in the carton."

Interval = tuple[float, float]


@dataclass(frozen=True)
class PackInvoiceSceneRanges:
    """Each axis's offset from the default layout, as (low, high) [m, deg]; a GA-S Run's fixed axis has low == high."""

    paper_x: Interval
    paper_y: Interval
    paper_yaw_deg: Interval
    box_x: Interval
    box_y: Interval
    box_yaw_deg: Interval
    bench_z: Interval
    base_x: Interval
    base_y: Interval
    base_yaw_deg: Interval

    def __post_init__(self) -> None:
        for name, (low, high) in asdict(self).items():
            if low > high:
                raise ValueError(f"{name}: an interval is (low, high) with low <= high; got {(low, high)}")

    def scaled(self, factor: float) -> PackInvoiceSceneRanges:
        return PackInvoiceSceneRanges(**{k: (low * factor, high * factor) for k, (low, high) in asdict(self).items()})


REACH_LIMITS = PackInvoiceSceneRanges(
    paper_x=(-0.04, 0.12), paper_y=(-0.09, 0.15), paper_yaw_deg=(-30.0, 40.0),
    box_x=(-0.12, 0.04), box_y=(-0.03, 0.04), box_yaw_deg=(-10.0, 25.0),
    bench_z=(-0.08, 0.03),
    base_x=(-0.02, 0.04), base_y=(-0.04, 0.03), base_yaw_deg=(-5.0, 4.0),
)
"""The widest offsets, side by side, at which the robot still reaches the pile and the box and the
hand stays in the 320x180 ego frames."""

TRAINING_RANGES = REACH_LIMITS.scaled(0.5)
"""The collection's and the Train cell's draws: half of ``REACH_LIMITS`` on each side."""

BEYOND_FACTORS = (1.5,)
"""Where a GA-S Run fixes its axis: these times the training range's extent on that side. 1.5 alone, not
also 2 (the limit itself): reach and view are checked one axis at a time, not with the other axes
drawn across training as a GA-S row draws them, so the point nearer training is the safer one. DB-R3
also fixes one point per direction."""

PAPER_PILE = "packing_bench.paper_pile"
BOX_POSE = "outbound_box.pose"
BENCH_HEIGHT = "packing_bench.height"
STATION_VARIATIONS = (PAPER_PILE, BOX_POSE, BENCH_HEIGHT)
"""The station's variation keys (the base's is the embodiment's)."""

AXES = {
    "paper_x": (PAPER_PILE, 0), "paper_y": (PAPER_PILE, 1), "paper_yaw_deg": (PAPER_PILE, 2),
    "box_x": (BOX_POSE, 0), "box_y": (BOX_POSE, 1), "box_yaw_deg": (BOX_POSE, 2),
    "bench_z": (BENCH_HEIGHT, 0),
    "base_x": ("base_jitter", 0), "base_y": ("base_jitter", 1), "base_yaw_deg": ("base_jitter", 2),
}
"""Axis -> (the variation, or the base's environment fields, that draws it; its component)."""


def scene_fields(ranges: PackInvoiceSceneRanges) -> tuple[dict, dict]:
    """``(environment, variations)`` blocks that draw every axis uniformly in ``ranges``."""

    bounds = {key: ([0.0] * (1 if key == BENCH_HEIGHT else 3), [0.0] * (1 if key == BENCH_HEIGHT else 3))
              for key in (*STATION_VARIATIONS, "base_jitter")}
    for axis, (low, high) in asdict(ranges).items():
        key, index = AXES[axis]
        to_unit = math.radians if axis.endswith("_deg") and key != "base_jitter" else float
        bounds[key][0][index], bounds[key][1][index] = round(to_unit(low), 7) + 0.0, round(to_unit(high), 7) + 0.0
    environment = {"base_jitter_low": bounds["base_jitter"][0], "base_jitter_high": bounds["base_jitter"][1]}
    variations = {}
    for key in STATION_VARIATIONS:
        asset, name = key.split(".")
        variations.setdefault(asset, {})[name] = {
            "enabled": True, "sampler_cfg": {"low": bounds[key][0], "high": bounds[key][1]},
        }
    return environment, variations


def beyond_settings(
    training: PackInvoiceSceneRanges = TRAINING_RANGES, factors: tuple[float, ...] = BEYOND_FACTORS,
) -> list[tuple[str, str, PackInvoiceSceneRanges]]:
    """The GA-S Runs: ``(key, note, ranges)``, one per axis, side and factor."""

    runs = []
    for axis, (low, high) in asdict(training).items():
        for side, extent in (("neg", low), ("pos", high)):
            if not extent:
                continue
            for factor in factors:
                point = round(extent * factor, 6) + 0.0
                ranges = PackInvoiceSceneRanges(**{**asdict(training), axis: (point, point)})
                unit = "deg" if axis.endswith("_deg") else "m"
                runs.append((f"{axis.removesuffix('_deg')}_{side}_{factor:g}x".replace(".", "p"),
                             f"{axis} fixed at {point:+g} {unit} ({factor:g}x the training extent)", ranges))
    return runs


SCENE_ARG_NAMES = ("outbound_box_scale_x", "outbound_box_scale_y", "outbound_box_scale_z")
"""The environment arguments that build the scene a recording needs: the box's size."""

SCENE_ARG_DEFAULTS = dict(zip(SCENE_ARG_NAMES, E2_PACK_INVOICE_BOX_SCALE_XYZ))
"""Their environment defaults."""

DELIVERED_BOX_SCALE = {name: 1.0 for name in SCENE_ARG_NAMES}
"""What a recording that names no box scale was made with: the delivered box, not the task's carton."""

PAPER_PHYSICS_ARGS = {
    "paper_coupling": "one_way_proxy",
    "paper_hand_colliders": "official_hull",
    "paper_num_sheets": 4,
    "paper_resolution": "4x6",
    "paper_substeps": 16,
    "paper_iterations": 10,
    "paper_contact_mu": 0.2,
    "paper_self_contact_radius": 0.002,
}


def recorded_scene_args(attrs) -> dict[str, float]:
    """The box size a paper demo HDF5 was recorded with: its ``scene_args`` attr, or the delivered box."""

    args = dict(DELIVERED_BOX_SCALE)
    if "scene_args" in attrs:
        args.update({name: float(value) for name, value in json.loads(attrs["scene_args"]).items() if name in args})
    return args


def apply_recorded_scene_args(args_cli, attrs, log=print) -> None:
    """Set ``recorded_scene_args`` on parsed environment arguments; refuse a contradicting flag."""

    for name, value in recorded_scene_args(attrs).items():
        given = getattr(args_cli, name)
        if given not in (SCENE_ARG_DEFAULTS[name], value):
            raise ValueError(f"--{name} {given} contradicts the recording's {value}")
        setattr(args_cli, name, value)
        log(f"recorded scene: {name} {value:+.4f}")


def recorded_variation_overrides(h5) -> list[str]:
    """Arena variation overrides that redraw a recording's station exactly: each station variation
    the file has under ``variations/`` enabled with its range collapsed to the recorded draw."""

    overrides = []
    for key in STATION_VARIATIONS:
        path = "variations/" + key.replace(".", "/")
        if path in h5:
            value = "[" + ",".join(repr(float(v)) for v in h5[path][0]) + "]"
            overrides += [f"{key}.enabled=true", f"{key}.sampler_cfg.low={value}", f"{key}.sampler_cfg.high={value}"]
    return overrides
