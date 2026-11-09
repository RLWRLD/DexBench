# SPDX-License-Identifier: Apache-2.0
"""DB-L1 collection plans: where each item of a stream episode goes and how it is turned, episode by episode."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PLAN_FORMAT = "dexbench_l1_stream_plan"
PLAN_VERSION = 1


@dataclass
class PlanEpisode:
    """One plan entry as the stream takes it, per item in the stream's items order (bags, then boxes). Not frozen:
    Isaac Lab's configclass walks an event term's params and writes back into the dataclasses it finds there."""

    id: str
    release_t: tuple[float, ...]
    """Release time [s from the episode's start]; inf for an item the episode leaves out."""
    lateral_y: tuple[float, ...]
    """Centre across the belt [m, env frame]."""
    yaw_rad: tuple[float, ...]
    """Turn about z [rad]."""


def across_belt_half_width(half_extent_xy: Sequence[float], yaw_rad: float) -> float:
    """Half the width across the belt (y) of a footprint with half extents (x, y) at yaw 0, turned by ``yaw_rad``
    (its bounding rectangle)."""
    hx, hy = (float(v) for v in half_extent_xy)
    return abs(math.sin(yaw_rad)) * hx + abs(math.cos(yaw_rad)) * hy


def off_deck(half_extent_xy: Sequence[float], lateral_y: float, yaw_rad: float, deck_y: Sequence[float]) -> str | None:
    """Why a footprint centred at ``lateral_y`` and turned by ``yaw_rad`` is not on the deck (y in ``deck_y``), or
    None when it is."""
    half = across_belt_half_width(half_extent_xy, yaw_rad)
    lo, hi = (float(v) for v in deck_y)
    if lateral_y - half < lo or lateral_y + half > hi:
        return (f"y {lateral_y:.3f} at yaw {math.degrees(yaw_rad):.1f} deg spans {lateral_y - half:.3f}-"
                f"{lateral_y + half:.3f} m, off the deck ({lo:.2f}-{hi:.2f})")
    return None


def off_belt_deg(long_axis_deg: float) -> float:
    """The angle [deg, 0-90] between a long side pointing ``long_axis_deg`` (about z, from +X) and the belt's
    direction (+X): an outline repeats every half turn."""
    return abs(math.remainder(float(long_axis_deg), 180.0))


def _number(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{what}: {value!r} is not a finite number")
    return float(value)


def parse_stream_plan(plan: dict, *, bag_variants: Sequence[str], half_extents: Sequence[Sequence[float]],
                      boxes: int, belt_speed: float, deck_y: Sequence[float], spacing_m: float,
                      lead: str | None = None, bag_band_deg: Sequence[float] | None = None,
                      rest_long_axis_deg: Sequence[float] | None = None, episode_bags: int | None = None,
                      episode_boxes: int | None = None, source: str = "plan") -> tuple[PlanEpisode, ...]:
    """A plan's episodes for a session with ``bag_variants`` (one per bag, in order) and ``boxes`` boxes, whose items
    (bags, then boxes) have the footprint ``half_extents`` (x, y at yaw 0), and whose episodes put down
    ``episode_bags`` of the bags and ``episode_boxes`` of the boxes (None: all of them); ``ValueError`` names what is
    wrong."""
    if not isinstance(plan, dict) or plan.get("format") != PLAN_FORMAT:
        found = plan.get("format") if isinstance(plan, dict) else type(plan).__name__
        raise ValueError(f"{source}: not a {PLAN_FORMAT} file (format {found!r})")
    if plan.get("version") != PLAN_VERSION:
        raise ValueError(f"{source}: version {plan.get('version')!r}; this reads version {PLAN_VERSION}")
    bags = len(bag_variants)
    episode_bags = bags if episode_bags is None else int(episode_bags)
    episode_boxes = boxes if episode_boxes is None else int(episode_boxes)
    if not (1 <= episode_bags <= bags and 0 <= episode_boxes <= boxes):
        raise ValueError(f"{source}: episodes of {episode_bags} bags and {episode_boxes} boxes in a session of {bags} "
                         f"and {boxes}")
    if len(half_extents) != bags + boxes:
        raise ValueError(f"{source}: {len(half_extents)} footprints for {bags} bags and {boxes} boxes")
    if belt_speed <= 0.0:
        raise ValueError(f"{source}: a plan needs a moving belt, not {belt_speed} m/s")
    session = plan.get("session") or {}
    if "bag_variants" in session and list(session["bag_variants"]) != list(bag_variants):
        raise ValueError(f"{source}: planned for the bags {list(session['bag_variants'])}, the session has "
                         f"{list(bag_variants)} (in order)")
    if "boxes" in session and int(session["boxes"]) != boxes:
        raise ValueError(f"{source}: planned for {session['boxes']} boxes, the session has {boxes}")
    if "belt_speed" in session and abs(float(session["belt_speed"]) - belt_speed) > 1e-9:
        raise ValueError(f"{source}: planned for a belt at {session['belt_speed']} m/s, the session's runs at "
                         f"{belt_speed} (--belt_speed)")
    if lead not in (None, "bag", "box"):
        raise ValueError(f"{source}: lead {lead!r}: bag, box or None")
    if bag_band_deg is not None and (rest_long_axis_deg is None or len(rest_long_axis_deg) != bags):
        raise ValueError(f"{source}: a bag band needs each of the {bags} bags' long-axis direction at rest")
    episodes = plan.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f"{source}: no episodes")
    out, ids = [], set()
    for number, episode in enumerate(episodes):
        name = str(episode.get("id", number)) if isinstance(episode, dict) else str(number)
        where = f"{source}: episode {name}"
        if not isinstance(episode, dict) or not isinstance(episode.get("items"), list):
            raise ValueError(f"{where}: expected an object with an 'items' list")
        if name in ids:
            raise ValueError(f"{where}: the id repeats")
        ids.add(name)
        release, lateral, yaw = [None] * (bags + boxes), [None] * (bags + boxes), [None] * (bags + boxes)
        distance = 0.0
        for position, item in enumerate(episode["items"]):
            kinds = [kind for kind in ("bag", "box") if kind in item] if isinstance(item, dict) else []
            if len(kinds) != 1:
                raise ValueError(f"{where}, item {position}: name exactly one of 'bag' or 'box'")
            kind, index = kinds[0], item[kinds[0]]
            count = bags if kind == "bag" else boxes
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < count:
                raise ValueError(f"{where}, item {position}: {kind} {index!r}; the session has {count} "
                                 f"{'boxes' if kind == 'box' else 'bags'} (counted from 0)")
            k = index if kind == "bag" else bags + index
            if release[k] is not None:
                raise ValueError(f"{where}: {kind} {index} is listed twice")
            gap = _number(item.get("gap_m", 0.0 if position == 0 else spacing_m), f"{where}, {kind} {index} gap_m")
            if gap < 0.0:
                raise ValueError(f"{where}, {kind} {index}: gap_m {gap} is negative")
            distance += gap
            release[k] = distance / belt_speed
            lateral[k] = _number(item.get("lateral_y"), f"{where}, {kind} {index} lateral_y")
            yaw[k] = math.remainder(math.radians(_number(item.get("yaw_deg"), f"{where}, {kind} {index} yaw_deg")),
                                    2.0 * math.pi)
            why = off_deck(half_extents[k], lateral[k], yaw[k], deck_y)
            label = bag_variants[index] if kind == "bag" else f"box {index}"
            if why is not None:
                raise ValueError(f"{where}, {kind} {index} ({label}): {why}")
            if kind == "bag" and bag_band_deg is not None:
                low, high = (float(v) for v in bag_band_deg)
                angle = off_belt_deg(float(rest_long_axis_deg[index]) + math.degrees(yaw[k]))
                if not low - 0.5 <= angle <= high + 0.5:
                    raise ValueError(f"{where}, bag {index} ({label}): yaw {math.degrees(yaw[k]):.1f} deg puts its "
                                     f"long side {angle:.1f} deg off the belt, outside the run's {low:g}-{high:g}")
        listed = (sum(t is not None for t in release[:bags]), sum(t is not None for t in release[bags:]))
        if listed != (episode_bags, episode_boxes):
            raise ValueError(f"{where}: {listed[0]} bags and {listed[1]} boxes; an episode puts down {episode_bags} "
                             f"and {episode_boxes}")
        if lead is not None:
            kinds = [next(kind for kind in ("bag", "box") if kind in item) for item in episode["items"]]
            if kinds != alternating_kinds(lead, episode_bags, episode_boxes):
                raise ValueError(f"{where}: releases {kinds}, not alternating from a {lead}")
        out.append(PlanEpisode(name, tuple(math.inf if t is None else t for t in release),
                               tuple(0.0 if v is None else v for v in lateral),
                               tuple(0.0 if v is None else v for v in yaw)))
    return tuple(out)


def alternating_kinds(lead: str, bags: int, boxes: int) -> list[str]:
    """The release order of ``bags`` bags and ``boxes`` boxes alternating from ``lead`` while both are left."""
    left = {"bag": int(bags), "box": int(boxes)}
    order, kind = [], lead
    while left["bag"] or left["box"]:
        if not left[kind]:
            kind = "box" if kind == "bag" else "bag"
        order.append(kind)
        left[kind] -= 1
        kind = "box" if kind == "bag" else "bag"
    return order


LAYOUT_DRAW_PER_PLACE = ("gap", "lateral", "turn", "sign", "flip")
"""The coordinates an episode's layout draw (``l1_park_slab.stream_layout``) holds per place of its release order, each
in [0, 1]: the belt's run from the previous release (as a fraction of the spacing range; the first place's is
unused, it goes down as the episode starts), the lateral centre (of the lateral range), the long side's angle off the
belt (of its kind's band), which side of the belt's direction (under 0.5: to its left) and which end leads (under 0.5:
the rest pose's)."""


def layout_draw_size(scene_bags: int, scene_boxes: int, places: int) -> int:
    """How many coordinates an episode's layout draw holds: an ordering key per bag and per box of the scene, then
    ``LAYOUT_DRAW_PER_PLACE`` for each of the episode's ``places`` releases."""
    return int(scene_bags) + int(scene_boxes) + len(LAYOUT_DRAW_PER_PLACE) * int(places)


def split_layout_draw(draw: Sequence[float], scene_bags: int, scene_boxes: int,
                      places: int) -> tuple[list[int], list[int], list[tuple[float, ...]]]:
    """An episode's layout draw (``layout_draw_size`` values in [0, 1]) as (the bags in the order the episode takes
    them, the boxes likewise, each place's ``LAYOUT_DRAW_PER_PLACE`` values): the order is the ordering keys' sort,
    smallest first (ties to the earlier one). ValueError for a draw of another size or outside [0, 1]."""
    values = [float(v) for v in draw]
    size = layout_draw_size(scene_bags, scene_boxes, places)
    if len(values) != size:
        raise ValueError(f"a layout draw of {len(values)} values; {scene_bags} bags, {scene_boxes} boxes and {places} "
                         f"places take {size}")
    if not all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in values):
        raise ValueError(f"a layout draw's values lie in [0, 1]: {values}")
    bags, boxes = values[:scene_bags], values[scene_bags:scene_bags + scene_boxes]
    per = len(LAYOUT_DRAW_PER_PLACE)
    rest = values[scene_bags + scene_boxes:]
    return (sorted(range(scene_bags), key=lambda i: (bags[i], i)), sorted(range(scene_boxes), key=lambda i: (boxes[i], i)),
            [tuple(rest[place * per:(place + 1) * per]) for place in range(places)])


def laid_out_episode(draw: Sequence[float], params: dict) -> list[dict]:
    """The episode a layout draw lays out, read without the stream: per release, in order, the item (its index in
    ``params["items"]``, kind and variant), its planned release time [s], its lateral centre [m] (before the put-down
    moves it onto the clear belt) and its long side's angle off the belt [deg]. ``params`` are the stream's
    (``evaluation_settings.events.l1_stream.params`` in a condition row); the stream's ``_lay_out`` lays the same
    order, times, centres and turns out. Nothing in this repository calls it: the DB-L1 collection console, which
    is not in this repository, reads it."""
    items = list(params["items"])
    bags = [k for k, it in enumerate(items) if it["kind"] == "bag"]
    boxes = [k for k, it in enumerate(items) if it["kind"] == "box"]
    episode_bags = int(params.get("episode_bags") or len(bags))
    episode_boxes = len(boxes) if params.get("episode_boxes") is None else int(params["episode_boxes"])
    kinds = alternating_kinds(params["lead"], episode_bags, episode_boxes)
    bag_order, box_order, places = split_layout_draw(draw, len(bags), len(boxes), len(kinds))
    queues = {"bag": [bags[i] for i in bag_order], "box": [boxes[i] for i in box_order]}
    low, high = (float(v) for v in params["spacing_m"])
    lo, hi = (float(v) for v in params["lateral_y"])
    out, run = [], 0.0
    for place, kind in enumerate(kinds):
        gap, lateral, turn = places[place][:3]
        run += 0.0 if place == 0 else low + (high - low) * gap
        k = queues[kind].pop(0)
        band = params["bag_yaw_off_belt_deg" if kind == "bag" else "box_yaw_off_belt_deg"]
        out.append({"item": k, "kind": kind, "variant": items[k].get("variant"),
                    "release_t_planned_s": run / float(params["belt_speed"]), "lateral_y_m": lo + (hi - lo) * lateral,
                    "off_belt_deg": float(band[0]) + (float(band[1]) - float(band[0])) * turn})
    return out


def load_stream_plan(path: str | Path, **session) -> tuple[tuple[PlanEpisode, ...], str]:
    """``parse_stream_plan`` on the JSON file at ``path`` -> (its episodes, the file's SHA-256)."""
    raw = Path(path).read_bytes()
    try:
        plan = json.loads(raw)
    except ValueError as error:
        raise ValueError(f"{path}: not JSON: {error}") from error
    return parse_stream_plan(plan, source=str(path), **session), hashlib.sha256(raw).hexdigest()
