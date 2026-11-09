# SPDX-License-Identifier: Apache-2.0
"""What a recorded DB-L1 stream demo says about its episode, read without Isaac Lab."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

STATE_NAMES = ("waiting", "in_play", "placed", "wrong_tote", "dropped", "missed")
"""A bag's states, in the order of ``polybag_stream``'s ``WAITING`` ... ``MISSED``."""
EPISODE_RECORD_ATTR = "dexbench_l1_stream"
"""The per-demo HDF5 attribute holding ``PolybagStream.episode_record`` as JSON."""
EPISODE_RECORD_VERSION = 3
"""3: the stream's ``mode``; per item whether it is in the episode (``used``; an episode draws three of
the scene's four bags and two of its four boxes, an item left out has ``release_t_planned_s`` null), per box whether
it was disturbed; the episode's ``success`` and ``failed``; and for a continuous stream's demo, its ``segment`` (the
stream's state at its start and its events, steps counted from its start) and the ``grasp_style`` label (null until
a review sets it). 2 (261014): an item's turn is recorded with its long side's angle off the belt
(``off_belt_deg``, a bag's measured one at the pick zone ``off_belt_at_pick_zone_deg``), the session with the lead,
the spacing range and the orientation bands; 1 (260827) gave a turn's ``yaw_mod180_deg`` and one spacing with its
jitter. What this module reads (the items' kinds, states and steps, the plan) is in all of them; an item of a record
before 3 is in the episode."""
REPLAYABLE_VERSIONS = (2, 3)
FINAL_STATE_NAMES = ("placed", "wrong_tote", "dropped", "missed")


def bag_states_by_step(record: dict) -> list[list[str]]:
    """Each bag's state (``STATE_NAMES``) at the end of every control step of a recorded episode, from its
    ``episode_record``: row r is step r + 1, the step taking the demo's ``actions[r]``; columns are the bags in items
    order. A bag final in row r was judged on the demo's ``states[r]``; one released in row r is on the belt from
    ``states[r + 1]`` on. Exact, since a bag only goes waiting -> in play -> one final state."""
    bags = [item for item in record["items"] if item["kind"] == "bag"]
    rows = []
    for step in range(1, int(record["steps"]) + 1):
        rows.append(["waiting" if not 0 <= bag["release_step"] <= step else
                     bag["state"] if 0 <= bag["final_step"] <= step else "in_play" for bag in bags])
    return rows


def placed_in_order_steps(record: Mapping) -> list[int]:
    """The steps the episode's leading bags were placed in, in stream (release) order, up to its first failure: one
    entry per bag counted from the first released, stopping at the first that is not ``placed`` or was placed in or
    after the step of the episode's first failure (a bag of the episode dropped, missed or in the wrong tote, or a
    box disturbed: ``failed`` in ``polybag_stream``). Its length is the episode's ``bags_placed_in_order``; DB-L1's
    headline success is that length reaching ``headline_bags`` (2), decided in the step its last entry names."""
    items = [item for item in record["items"] if item.get("used", True)]
    failures = [int(item["final_step"]) for item in items
                if item["kind"] == "bag" and item["state"] in FINAL_STATE_NAMES[1:]]
    failures += [int(item["disturbed_step"]) for item in items if item["kind"] == "box" and item.get("disturbed")]
    first_failure = min(failures, default=math.inf)
    steps = []
    for k in record["order"]:
        item = record["items"][k]
        if item["kind"] != "bag" or not item.get("used", True):
            continue
        step = int(item["final_step"])
        if item["state"] != "placed" or not 0 <= step < first_failure:
            break
        steps.append(step)
    return steps


def recorded_layout(record: dict,
                    items: Sequence[tuple[str, str | None]]) -> tuple[list[float], list[float], list[float]]:
    """The layout a recorded episode ran, for ``PolybagStream.schedule_next``: per item, in the record's order (the
    stream's items order), its release time as planned at the episode's start [s; ``math.inf`` for an item left out
    of the episode], its lateral centre [m, env frame] and its turn [rad]: the planned releases, places and turns the
    recording drew, so a replay does not draw its own (an item is still held back while the replay's belt is blocked,
    as the recording's was). ``items`` are the replaying stream's (entity, variant) pairs; ValueError unless the
    record lists the same, in the same order (a recording of another bag set or stream), for a record of a version
    this cannot replay, or for a continuous stream's segment (resumed, not laid out)."""
    if record.get("version") not in REPLAYABLE_VERSIONS:
        raise ValueError(f"recorded_layout: an episode record of version {record.get('version')!r}; this stream "
                         f"replays versions {REPLAYABLE_VERSIONS}")
    if record.get("mode", "episode") != "episode":
        raise ValueError(f"recorded_layout: a {record['mode']} stream's segment is resumed from its record "
                         "(PolybagStream.schedule_resume), not laid out")
    recorded = [(item["entity"], item["variant"]) for item in record["items"]]
    if recorded != [(entity, variant or None) for entity, variant in items]:
        raise ValueError(f"recorded_layout: the record's items {recorded} are not this stream's {list(items)}")
    return ([math.inf if item["release_t_planned_s"] is None else float(item["release_t_planned_s"])
             for item in record["items"]],
            [float(item["lateral_y_m"]) for item in record["items"]],
            [math.radians(float(item["yaw_deg"])) for item in record["items"]])


def drawn_on(record: Mapping) -> int | None:
    """The stream seed a recorded demo's episode was drawn on, or None for one laid out otherwise (a collection plan,
    a scheduled layout, or the layout draw of a collection row, ``plan.source`` "variation"). KeyError for a drawn
    record without its session's seed."""
    if (record.get("plan") or {}).get("source") != "drawn":
        return None
    return int(record["session"]["seed"])


def reused_seed_refusal(seed: int, drawn: Mapping[int, Sequence[str]], directory) -> str | None:
    """Why a recording session on stream seed ``seed`` is refused, or None when it is free. ``drawn`` maps each seed
    (``drawn_on``) to the demos drawn on it (``file.hdf5:demo_N``) of every recording in ``directory``, the session's
    own file included. The draw is seeded by the seed and the process's episode count alone, so a later session on
    that seed (a crashed one run again included) would lay its episodes out as the earlier one did; the message names
    the next seed none of them drew on."""
    if seed not in drawn:
        return None
    free = next(s for s in range(seed + 1, seed + 1 + len(drawn) + 1) if s not in drawn)
    demos = drawn[seed]
    return (f"l1_stream_seed {seed}: {len(demos)} demos in {directory} were drawn on it "
            f"({', '.join(demos[:3])}{', ...' if len(demos) > 3 else ''}), and this session would lay its episodes "
            f"out again as theirs; record on --l1_stream_seed {free} (into a new --dataset_file)")
