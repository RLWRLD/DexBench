# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""DB-R1 piles: the fruit identities, the pile tray, the one-layer planner, recipes and layout sets."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from pathlib import Path
from types import SimpleNamespace

from dexbench_arena import ASSETS_DIR
from dexbench_arena.tasks.object_axis import ObjectAxis, TaskObject, task_object_record

R1_TASK_DESCRIPTION = (
    "Pick one fruit from the produce display and place it in the basket. "
    "Keep all other fruit on the tray."
)

R1_FRUIT_KINDS = {
    "plum": ("small", "train", 52),
    "lime": ("small", "eval", 58),
    "mandarin": ("small", "eval", 62),
    "peach": ("small", "train", 72),
    "lemon": ("large", "eval", 75),
    "apple": ("large", "train", 80),
    "pear": ("large", "train", 100),
    "bell_pepper": ("large", "eval", 110),
}
VARIANT_INDICES = range(1, 6)
FRUITS = {
    f"{fruit}_pile": {"fruit": fruit, "size_group": size, "split": split, "nominal_mm": nominal,
                      "variants": tuple(f"{fruit}_v{i:02d}" for i in VARIANT_INDICES)}
    for fruit, (size, split, nominal) in R1_FRUIT_KINDS.items()
}
FILL_SHARE = .72
COUNT_RANGE = (15, 30)
FRONT_FILL_SHARE = .66
PILE_TRAY_BOUNDS = ((3.0423, 0.50, 0.9734), (3.5764, 0.90, 1.33))
"""World ``(min xyz, max xyz)`` a fruit's centre stays in while it is on the pile tray: lip plane to back wall,
divider to divider, the board's front top up (measured on produce_display_fc.usd 2.0.0)."""
TRAY_FLOOR_FRONT_Z = 1.00
TRAY_FLOOR_SLOPE = .171
DROP_GAP_M = .003
WALL_MARGIN_M = .012
ELONGATED_RATIO = 1.15
REST_TILT_DEG = 12.0
RELAX_ITERATIONS = 600
EXACT_ATTEMPTS = 20


def random_quat_xyzw(rng: random.Random) -> tuple[float, float, float, float]:
    """Uniformly distributed rotation (Shoemake)."""
    u1, u2, u3 = rng.random(), rng.random(), rng.random()
    a, b = math.sqrt(1 - u1), math.sqrt(u1)
    return (a * math.sin(2 * math.pi * u2), a * math.cos(2 * math.pi * u2),
            b * math.sin(2 * math.pi * u3), b * math.cos(2 * math.pi * u3))


def rotate_xyzw(q, v):
    x, y, z, w = q
    tx, ty, tz = 2 * (y * v[2] - z * v[1]), 2 * (z * v[0] - x * v[2]), 2 * (x * v[1] - y * v[0])
    return (v[0] + w * tx + y * tz - z * ty, v[1] + w * ty + z * tx - x * tz, v[2] + w * tz + x * ty - y * tx)


def root_from_center(center, rotation, variant):
    """Root (bbox bottom-centre origin) position of a fruit whose bbox centre is ``center``."""
    offset = rotate_xyzw(rotation, (0.0, 0.0, variant["size_m"][2] * variant["scale"] / 2))
    return tuple(c - o for c, o in zip(center, offset))


SEED_BLOCK = 10_000


def seed_base(fruit_id: str) -> int:
    """First seed of a kind's own block: kinds never share a layout seed (plum 10000, lime 20000, ...)."""
    return SEED_BLOCK * (list(FRUITS).index(fruit_id) + 1)


def variant_asset_name(package: str) -> str:
    return f"dexbench_retail_e1_fc_{package}"


def registered_variants(fruit_id: str, assets_dir: Path = ASSETS_DIR) -> list[dict]:
    """The five variant records of a kind, read from their package sidecars."""
    rows = []
    for package in FRUITS[fruit_id]["variants"]:
        sidecar = assets_dir / "object_library" / package / "usd" / f"{package}.meta.json"
        asset = json.loads(sidecar.read_text())["asset"]
        material = asset["physics_material"]
        rows.append({"variant_id": package, "asset": variant_asset_name(package), "scale": 1.0,
                     "size_m": asset["geometry"]["size_m"], "mass_kg": asset["mass_kg"],
                     "friction": [material["static_friction"], material["dynamic_friction"]],
                     "asset_version": asset["version"]})
    return rows


def _quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz)


def _axis_quat(axis, angle):
    s = math.sin(angle / 2)
    return (axis[0] * s, axis[1] * s, axis[2] * s, math.cos(angle / 2))


def _rest_rotation(rng, variant):
    """Random resting rotation: any for a round fruit; the authored lying pose, turned and tilted, otherwise."""
    size = variant["size_m"]
    if max(size) / min(size) <= ELONGATED_RATIO:
        return random_quat_xyzw(rng)
    tilt_axis = rng.uniform(0, 2 * math.pi)
    tilt = _axis_quat((math.cos(tilt_axis), math.sin(tilt_axis), 0.0), math.radians(rng.uniform(0, REST_TILT_DEG)))
    return _quat_mul(tilt, _axis_quat((0.0, 0.0, 1.0), rng.uniform(0, 2 * math.pi)))


def _footprint(variant, rotation):
    """A fruit seen from above as a capsule: (segment half-vector, radius); a round fruit is a disk."""
    size = [v * variant["scale"] for v in variant["size_m"]]
    if max(size) / min(size) <= ELONGATED_RATIO:
        return ((0.0, 0.0), max(size) / 2)
    long_axis = 0 if size[0] >= size[1] else 1
    axis = rotate_xyzw(rotation, (1.0, 0.0, 0.0) if long_axis == 0 else (0.0, 1.0, 0.0))
    half = (size[long_axis] - size[1 - long_axis]) / 2
    radius = max(size[1 - long_axis], size[2] * math.sin(math.radians(REST_TILT_DEG)) * 2) / 2
    return ((axis[0] * half, axis[1] * half), radius)


def _support(shape, axis):
    (hx, hy), radius = shape
    return abs(hx * axis[0] + hy * axis[1]) + radius


def _closest_on_segments(p0, d0, p1, d1):
    """Closest points of segments p0 + s d0 and p1 + t d1, s, t in [-1, 1]."""
    rx, ry = p0[0] - p1[0], p0[1] - p1[1]
    a = d0[0] * d0[0] + d0[1] * d0[1]
    e = d1[0] * d1[0] + d1[1] * d1[1]
    f = d1[0] * rx + d1[1] * ry
    if a < 1e-12 and e < 1e-12:
        return p0, p1
    if a < 1e-12:
        s, t = 0.0, max(-1.0, min(1.0, f / e))
    else:
        c = d0[0] * rx + d0[1] * ry
        if e < 1e-12:
            s, t = max(-1.0, min(1.0, -c / a)), 0.0
        else:
            b = d0[0] * d1[0] + d0[1] * d1[1]
            denom = a * e - b * b
            s = max(-1.0, min(1.0, (b * f - c * e) / denom)) if denom > 1e-12 else 0.0
            t = (b * s + f) / e
            if t < -1.0 or t > 1.0:
                t = max(-1.0, min(1.0, t))
                s = max(-1.0, min(1.0, (t * b - c) / a))
    return (p0[0] + s * d0[0], p0[1] + s * d0[1]), (p1[0] + t * d1[0], p1[1] + t * d1[1])


def _separation(a, pa, b, pb, gap):
    """Overlap depth and unit push direction (from a to b) of two capsules closer than ``gap``, or None."""
    qa, qb = _closest_on_segments(pa, a[0], pb, b[0])
    dx, dy = qb[0] - qa[0], qb[1] - qa[1]
    dist = math.hypot(dx, dy)
    overlap = a[1] + b[1] + gap - dist
    if overlap <= 0:
        return None
    if dist < 1e-9:
        dx, dy, dist = pb[0] - pa[0] + 1e-6, pb[1] - pa[1], math.hypot(pb[0] - pa[0] + 1e-6, pb[1] - pa[1])
    return overlap, (dx / dist, dy / dist)


def _half_height(variant, rotation):
    size = [v * variant["scale"] for v in variant["size_m"]]
    axes = [rotate_xyzw(rotation, e) for e in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
    return sum(abs(axis[2]) * s / 2 for axis, s in zip(axes, size))


def _relax(shapes, positions, box, gap, iterations):
    """Push overlapping footprints apart inside ``box`` until none overlap; True on success."""
    (lo_x, lo_y), (hi_x, hi_y) = box
    for _ in range(iterations):
        moved = False
        for i in range(len(shapes)):
            for j in range(i + 1, len(shapes)):
                hit = _separation(shapes[i], positions[i], shapes[j], positions[j], gap)
                if hit is None:
                    continue
                moved = True
                push, (nx, ny) = hit[0] / 2 + 1e-4, hit[1]
                positions[i] = (positions[i][0] - nx * push, positions[i][1] - ny * push)
                positions[j] = (positions[j][0] + nx * push, positions[j][1] + ny * push)
        for i, shape in enumerate(shapes):
            rx, ry = _support(shape, (1.0, 0.0)), _support(shape, (0.0, 1.0))
            positions[i] = (min(max(positions[i][0], lo_x + rx), hi_x - rx),
                            min(max(positions[i][1], lo_y + ry), hi_y - ry))
        if not moved:
            return True
    return False


def make_recipe(fruit_id: str, count: int | None = None, seed: int = 0,
                phase: str = "collection", variants: list[dict] | None = None, exact: bool = False) -> dict:
    """Plan one dense, random layer over the task compartment, five variants balanced (counts differ by <=1)."""
    if fruit_id not in FRUITS:
        raise ValueError(f'unknown fruit identity: {fruit_id}')
    fruit = FRUITS[fruit_id]
    if phase not in ("collection", "evaluation"):
        raise ValueError("phase must be collection or evaluation")
    if not isinstance(seed, int) or not 0 <= seed < 1_000_000:
        raise ValueError("seed must be an integer in [0, 1000000)")
    if variants is None:
        variants = registered_variants(fruit_id)
    if len(variants) != 5 or len({v["variant_id"] for v in variants}) != 5:
        raise ValueError("each fruit requires five distinct variants")
    for variant in variants:
        if not variant.get('asset') or not math.isfinite(variant['scale']) or variant['scale'] <= 0:
            raise ValueError('each variant requires a registered asset and positive scale')
        if len(variant.get('size_m', ())) != 3:
            raise ValueError('variants require measured local bbox size_m for separated drops')
    rng = random.Random(seed + (0 if phase == "collection" else 1_000_000))
    (x0, y0, _), (x1, y1, _) = PILE_TRAY_BOUNDS
    box = ((x0 + WALL_MARGIN_M, y0 + WALL_MARGIN_M), (x1 - WALL_MARGIN_M, y1 - WALL_MARGIN_M))
    area = (box[1][0] - box[0][0]) * (box[1][1] - box[0][1])
    if count is None:
        def covered(v):
            (hx, hy), radius = _footprint(v, (0.0, 0.0, 0.0, 1.0))
            return math.pi * radius ** 2 + 4 * radius * math.hypot(hx, hy)
        mean = sum(covered(v) for v in variants) / len(variants)
        count = min(max(round(FILL_SHARE * area / mean), COUNT_RANGE[0]), COUNT_RANGE[1])
    if not isinstance(count, int) or not 5 <= count <= 60:
        raise ValueError("count must be an integer in [5, 60]")
    width = box[1][1] - box[0][1]
    footprint = sum(math.pi * r ** 2 + 4 * r * math.hypot(hx, hy)
                    for (hx, hy), r in (_footprint(variants[i % 5], (0.0, 0.0, 0.0, 1.0)) for i in range(count)))
    depth = min(box[1][0] - box[0][0], footprint / FRONT_FILL_SHARE / width)
    box = (box[0], (box[0][0] + depth, box[1][1]))
    attempts = [count] * EXACT_ATTEMPTS if exact else range(count, 4, -1)
    for n in attempts:
        order = [variants[i % 5] for i in range(n)]
        rotations = [_rest_rotation(rng, v) for v in order]
        shapes = [_footprint(v, q) for v, q in zip(order, rotations)]
        positions = [(rng.uniform(box[0][0], box[1][0]), rng.uniform(box[0][1], box[1][1])) for _ in order]
        if _relax(shapes, positions, box, DROP_GAP_M, RELAX_ITERATIONS):
            break
    else:
        raise ValueError(f"no one-layer layout of {'exactly ' + str(count) if exact else 'at least 5'} {fruit_id} fruit")
    objects = []
    for i, (v, q, shape, (x, y)) in enumerate(zip(order, rotations, shapes, positions)):
        reach = max(_support(shape, (1.0, 0.0)), _support(shape, (0.0, 1.0)))
        z = (TRAY_FLOOR_FRONT_Z + TRAY_FLOOR_SLOPE * (x + reach - x0) + _half_height(v, q) + DROP_GAP_M)
        objects.append({**v, "fruit_id": fruit_id, "name": f"r1_fruit_{i:03d}",
                        "position_xyz": root_from_center((x, y, z), q, v), "rotation_xyzw": q})
    return {"schema_version": 1, "fruit_id": fruit_id, "split": fruit["split"],
            "size_group": fruit["size_group"], "phase": phase, "seed": seed, "count": len(objects),
            "layout": {"mode": "dense_layer", "fill_share": FILL_SHARE, "requested": count,
                       "elongated_ratio": ELONGATED_RATIO, "rest_tilt_deg": REST_TILT_DEG},
            "status": "authored", "objects": objects}


def load_recipe(path: str | Path, fruit_id: str | None = None, settled: bool = False) -> dict:
    recipe = json.loads(Path(path).read_text())
    if recipe.get("schema_version") != 1 or recipe.get("fruit_id") not in FRUITS:
        raise ValueError("unsupported R1 recipe or fruit identity")
    if fruit_id is not None and recipe["fruit_id"] != fruit_id:
        raise ValueError(f"recipe fruit {recipe['fruit_id']} differs from instructed {fruit_id}")
    if settled and recipe.get("status") != "settled":
        raise ValueError("task requires a settled recipe; run bake_r1_layouts.py first")
    objects = recipe["objects"]
    fruit = FRUITS[recipe['fruit_id']]
    if recipe.get('split') != fruit['split'] or recipe.get('size_group') != fruit['size_group']:
        raise ValueError('recipe split/size must match its fruit identity')
    if len(objects) != recipe["count"] or not 5 <= len(objects) <= 60:
        raise ValueError("recipe count must match all planned fruit, within [5, 60]")
    if len({o["name"] for o in objects}) != len(objects):
        raise ValueError("duplicate fruit instance names")
    if any(o["fruit_id"] != recipe["fruit_id"] for o in objects):
        raise ValueError("a pile must contain the instructed fruit kind only")
    if len({o["variant_id"] for o in objects}) != 5:
        raise ValueError("a pile must mix all five variants")
    foreign = sorted({(o["variant_id"], o["asset"]) for o in objects
                      if o["variant_id"] not in fruit["variants"] or o["asset"] != variant_asset_name(o["variant_id"])})
    if foreign:
        raise ValueError(f"a {recipe['fruit_id']} pile spawns only {list(fruit['variants'])}, each as its own asset; "
                         f"got {foreign[:3]}")
    for obj in objects:
        if len(obj['position_xyz']) != 3 or len(obj['rotation_xyzw']) != 4 or not obj.get('asset'):
            raise ValueError('each fruit needs XYZ, xyzw and a registered asset')
        values = (*obj["position_xyz"], *obj["rotation_xyzw"], obj["scale"])
        if not all(math.isfinite(v) for v in values) or obj["scale"] <= 0:
            raise ValueError("invalid fruit pose or scale")
        if abs(sum(v*v for v in obj["rotation_xyzw"]) - 1) > .001:
            raise ValueError("rotation must be a unit xyzw quaternion")
        if 'mass_kg' in obj and not (math.isfinite(obj['mass_kg']) and obj['mass_kg'] > 0):
            raise ValueError('mass_kg must be finite and positive')
    return recipe


R1_LAYOUT_VARIATION = "layout"
"""The per-reset layout draw of a layout set, on the set's first fruit: ``r1_fruit_000.layout``."""


def _bodies(recipe: dict) -> list[tuple]:
    return [(o["name"], o["asset"], o["scale"], o.get("mass_kg")) for o in recipe["objects"]]


def layout_set(directory: str | Path, fruit_id: str) -> dict[str, dict]:
    """The settled layouts of ``fruit_id`` in ``directory``, by label (the file stem), in seed order."""
    root = Path(directory)
    if not root.is_dir():
        raise ValueError(f"r1_layouts names {directory}, which is no directory")
    pattern = re.compile(rf"{re.escape(fruit_id)}_(collection|evaluation)_s\d+")
    paths = sorted(path for path in root.glob(f"{fruit_id}_*.json") if pattern.fullmatch(path.stem))
    if not paths:
        raise ValueError(f"{directory} holds no {fruit_id}_<phase>_s<seed>.json layout")
    layouts = dict(sorted(((path.stem, load_recipe(path, fruit_id, settled=True)) for path in paths),
                          key=lambda item: item[1]["seed"]))
    first = next(iter(layouts.values()))
    for label, recipe in layouts.items():
        if _bodies(recipe) != _bodies(first):
            raise ValueError(f"{root / label}.json spawns other fruit bodies than {next(iter(layouts))}.json; "
                             "a layout set is one exact-count recipe family (make_recipe(..., exact=True))")
        if recipe["phase"] != first["phase"]:
            raise ValueError(f"{directory} mixes {first['phase']} and {recipe['phase']} layouts")
    if len({recipe["seed"] for recipe in layouts.values()}) != len(layouts):
        raise ValueError(f"{directory} holds two layouts of one seed")
    return layouts


def layout_set_sha256(directory: str | Path, labels) -> str:
    """One digest of the set's files: each label and its file's SHA-256, a line each, in the set's order."""
    lines = "".join(f"{label} {hashlib.sha256((Path(directory) / f'{label}.json').read_bytes()).hexdigest()}\n"
                    for label in labels)
    return hashlib.sha256(lines.encode()).hexdigest()


def r1_recipe_file(cfg) -> str | None:
    """The recipe a pile build spawns: ``cfg.r1_clutter_file``, or the first layout of ``cfg.r1_layouts``
    (which then draws one of its layouts at every reset); None for neither, which no DB-R1 build takes."""
    if cfg.r1_layouts is None:
        return cfg.r1_clutter_file
    if cfg.r1_clutter_file is not None:
        raise ValueError("r1_layouts spawns the first layout of its set and draws one at every reset; "
                         "drop r1_clutter_file (one fixed layout) or r1_layouts")
    return str(Path(cfg.r1_layouts) / f"{next(iter(layout_set(cfg.r1_layouts, cfg.r1_target_sku)))}.json")


def recipe_fit(obj, cfg) -> dict:
    """The pile's fit; a layout set's names its size and digest where one recipe names its seed and file's."""
    if cfg.r1_layouts is None:
        recipe = load_recipe(cfg.r1_clutter_file, obj.key, settled=True)
        return {k: recipe[k] for k in ("size_group", "phase", "seed", "count")} | {
            "recipe_sha256": hashlib.sha256(Path(cfg.r1_clutter_file).read_bytes()).hexdigest(),
            "variants": sorted({o["variant_id"] for o in recipe["objects"]})}
    if cfg.r1_clutter_file is not None:
        r1_recipe_file(cfg)
    layouts = layout_set(cfg.r1_layouts, obj.key)
    first = next(iter(layouts.values()))
    return {k: first[k] for k in ("size_group", "phase", "count")} | {
        "layouts": len(layouts), "layouts_sha256": layout_set_sha256(cfg.r1_layouts, layouts),
        "variants": sorted({o["variant_id"] for o in first["objects"]})}


R1_OBJECT_AXIS = ObjectAxis(
    field="r1_target_sku", default="plum_pile", scene_layout="fc_260821",
    objects=tuple(TaskObject(key=k, asset=variant_asset_name(v["variants"][2]), split=v["split"], label=v["fruit"],
                             size_m=(v["nominal_mm"] / 1000,))
                  for k, v in FRUITS.items()),
    size_axes="nominal size (delivery preset)", fit=recipe_fit,
    source_fit=("layouts", "layouts_sha256"),
)


def layout_rows(directory: str | Path, fruit_id: str) -> list[dict]:
    """One row per layout of the set (:func:`layout_set`), in its order."""
    layouts = layout_set(directory, fruit_id)
    task_object, fit = R1_OBJECT_AXIS.resolve(SimpleNamespace(
        r1_target_sku=fruit_id, r1_clutter_file=None, r1_layouts=str(directory)))
    first = next(iter(layouts.values()))
    version = next((o.get("asset_version") for o in first["objects"] if o["asset"] == task_object.asset), None)
    record = task_object_record(R1_OBJECT_AXIS, task_object, fit, version)
    key = f"{first['objects'][0]['name']}.{R1_LAYOUT_VARIATION}"
    return [{"variations": {key: label}, "task_object": record} for label in layouts]
