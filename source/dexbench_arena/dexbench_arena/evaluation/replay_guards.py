# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Guards for a run that reuses what another run recorded: same object, nothing random left unreplayed."""

from __future__ import annotations

import dataclasses
import json
import math
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dexbench_arena.evaluation.artifacts import is_finite, numeric_draw, recorded
from dexbench_arena.tasks.object_axis import asset_major_problem

REPLAY_GUARD_MODE = "warn"
"""``"warn"`` prints what does not match and goes on; ``"refuse"``
raises :class:`ReplayGuardError` instead. Switch to ``"refuse"`` for the release."""

TASK_OBJECT_KEY = "task_object"
"""Where a source records its object: an episode row's field and a demo file's ``env_args`` key."""


class ReplayGuardError(RuntimeError):
    """A replay or Mimic source that does not match the build: under ``REPLAY_GUARD_MODE = "refuse"``, and
    always from the strict benchmark checks."""


def enforce(problems: Sequence[str], what: str) -> None:
    """Report ``problems`` with ``what`` the run is doing, as :data:`REPLAY_GUARD_MODE` says."""
    if not problems:
        return
    message = f"[DexBench replay guard] {what}:\n  - " + "\n  - ".join(problems)
    if REPLAY_GUARD_MODE == "refuse":
        raise ReplayGuardError(message)
    print(
        f"\n{message}\n  (a warning: dexbench_arena.evaluation.replay_guards.REPLAY_GUARD_MODE is 'warn'; "
        "'refuse' stops the run)\n",
        file=sys.stderr,
        flush=True,
    )


@dataclass(frozen=True)
class ObjectIdentity:
    """The object a build is run with, and its fixture profile, in the terms a source records them."""

    field: str
    """The axis' config field (``r2_target_sku``)."""
    key: str
    default: str
    """What a source that records no object was made with (``ObjectAxis.default`` never changes)."""
    fixture: Mapping[str, Any]
    """The build's values of the axis' fixture fields, and its ``scene_layout`` when the axis names one."""
    fixture_defaults: Mapping[str, Any]
    """What a source that records no fixture value was made with: the config's defaults, and the
    axis' ``unlabelled_scene_layout``."""
    fit_form: Callable[[str, dict[str, Any]], dict[str, Any]] | None = dataclasses.field(
        default=None, compare=False, repr=False)
    """``fit_form(key, values)``: fixture values a source names outside its fit (``environment_cfg``, a
    sidecar's command line) in the form the axis' fit records them, as :meth:`of` puts the build's."""
    asset_version: str | None = None
    benchmark_asset_major: int | None = None
    benchmark_asset_major_renames: tuple[tuple[int, int, str], ...] = ()

    @classmethod
    def of(cls, axis, cfg, *, asset_version: str | None = None) -> ObjectIdentity:
        """The identity of a build from ``cfg`` on ``axis`` (``tasks.object_axis.ObjectAxis``)."""
        defaults = {
            item.name: item.default
            for item in dataclasses.fields(type(cfg))
            if item.name in axis.fixture_profile and item.default is not dataclasses.MISSING
        }
        task_object, fit = axis.resolve(cfg)
        fixture = {name: fit[name] if name in fit else getattr(cfg, name, None) for name in axis.fixture_profile}
        fixture.update({name: fit.get(name) for name in axis.source_fit})
        if axis.scene_layout is not None:
            fixture["scene_layout"] = axis.scene_layout
            defaults["scene_layout"] = axis.unlabelled_scene_layout

        def fit_form(key: str, values: dict[str, Any]) -> dict[str, Any]:
            try:
                _, source_fit = axis.resolve(dataclasses.replace(cfg, **{axis.field: key, **values}))
            except (TypeError, ValueError):
                return values
            return {name: source_fit.get(name, value) for name, value in values.items()}

        return cls(
            field=axis.field,
            key=task_object.key,
            default=axis.unlabelled_default or axis.default,
            fixture=fixture,
            fixture_defaults=defaults,
            fit_form=fit_form,
            asset_version=asset_version,
            benchmark_asset_major=task_object.benchmark_asset_major,
            benchmark_asset_major_renames=task_object.benchmark_asset_major_renames,
        )

    def source_identity(self, record: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
        """``(key, fixture)`` a source record was made with."""
        key, fixture = recorded_object_key(record, self.field), {}
        task_object = recorded(record, TASK_OBJECT_KEY)
        if isinstance(task_object, Mapping) and task_object.get("axis") == self.field:
            fit = task_object.get("fit") or {}
            fixture.update({name: fit[name] for name in self.fixture if name in fit})
        environment_cfg = recorded(record, "environment_cfg") or {}
        arguments = _flag_values(record.get("args") or (), self.fixture)
        raw: dict[str, Any] = {}
        for source in (environment_cfg, arguments):
            raw.update({name: source[name] for name in self.fixture if name not in fixture and name not in raw
                        and name in source})
        if raw and self.fit_form is not None:
            raw = self.fit_form(key or self.default, raw)
        fixture.update(raw)
        for name in self.fixture:
            fixture.setdefault(name, self.fixture_defaults.get(name))
        return (key or self.default), fixture

    def differences(self, records: Iterable[tuple[str, Mapping[str, Any]]]) -> list[str]:
        """One line per distinct (object, fixture) a source made something with that this build is not."""
        built = self._describe(self.key, self.fixture)
        seen: Counter[str] = Counter()
        where: dict[str, str] = {}
        total = 0
        for label, record in records:
            total += 1
            key, fixture = self.source_identity(record)
            if key == self.key and all(_same(fixture[name], self.fixture[name]) for name in self.fixture):
                continue
            described = self._describe(key, fixture)
            seen[described] += 1
            where.setdefault(described, label)
        return [
            f"{count} of {total} source records (first: {where[described]}) were made with {described}; "
            f"this build is {built}"
            for described, count in seen.items()
        ]

    def _describe(self, key: str, fixture: Mapping[str, Any]) -> str:
        settings = "".join(f", {name}={value}" for name, value in fixture.items())
        return f"{self.field}={key}{settings}"


def recorded_object_key(record: Mapping[str, Any], field: str) -> str | None:
    """The object a source record names on axis ``field``: its ``task_object`` key, else its
    ``environment_cfg`` value, else its replay sidecar's ``--field``; None when it names none (it was made
    with the axis' default, which only the environment knows)."""
    task_object = recorded(record, TASK_OBJECT_KEY)
    if isinstance(task_object, Mapping) and task_object.get("axis") == field and task_object.get("key") is not None:
        return task_object["key"]
    environment_cfg = recorded(record, "environment_cfg") or {}
    if environment_cfg.get(field) is not None:
        return environment_cfg[field]
    value = _flag_values(record.get("args") or (), (field,)).get(field)
    return None if value is None else str(value)


def _same(a: Any, b: Any) -> bool:
    try:
        return math.isclose(float(a), float(b), abs_tol=1e-9)
    except (TypeError, ValueError):
        return a == b


def _flag_values(arguments: Sequence[str], names: Iterable[str]) -> dict[str, Any]:
    """``{name: value}`` of ``--name value`` / ``--name=value`` on a command line; numbers parsed."""
    found, tokens = {}, list(arguments)
    for name in names:
        for index, token in enumerate(tokens):
            for flag in (f"--{name}", f"--{name.replace('_', '-')}"):
                if token == flag and index + 1 < len(tokens):
                    found[name] = _number(tokens[index + 1])
                elif token.startswith(flag + "="):
                    found[name] = _number(token.split("=", 1)[1])
    return found


def _number(text: str) -> Any:
    try:
        return float(text)
    except ValueError:
        return text


def read_episode_records(path: str | Path) -> list[tuple[str, dict]]:
    """``(label, row)`` for every row of an episode JSONL (Arena's ``episode_results_*.jsonl``)."""
    rows = []
    for number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if line.strip():
            rows.append((f"{Path(path).name}:{number}", json.loads(line)))
    return rows


def read_dataset_sources(path: str | Path) -> list[tuple[str, dict]]:
    """What a demo HDF5 says it was made with: its ``env_args``, else its replay sidecar, else its episode rows."""
    path = Path(path)
    import h5py

    with h5py.File(path, "r") as file:
        env_args = json.loads(file["data"].attrs.get("env_args", "{}")) if "data" in file else {}
    if TASK_OBJECT_KEY in env_args:
        return [(f"{path.name} env_args", env_args)]
    sidecar = path.with_suffix(".job.json")
    if sidecar.is_file():
        return [(sidecar.name, json.loads(sidecar.read_text()))]
    rows = path.with_suffix(".episodes.jsonl")
    if rows.is_file():
        return read_episode_records(rows)
    return [(f"{path.name} (records no object)", env_args)]


def check_replay_source(identity: ObjectIdentity | None, path: str | Path, *, what: str) -> None:
    """Warn or refuse (:data:`REPLAY_GUARD_MODE`) when ``path`` was made with another object or fixture profile."""
    if identity is None:
        return
    path = Path(path)
    records = read_episode_records(path) if path.suffix.lower() == ".jsonl" else read_dataset_sources(path)
    enforce(identity.differences(records), f"{what} from {path}")



DETERMINISTIC_RESET_ATTRIBUTE = "dexbench_deterministic_reset"

DRAWS_THROUGH_VARIATION_ATTRIBUTE = "dexbench_draws_through_variation"

UPSTREAM_DETERMINISTIC_RESET_EVENTS = frozenset(
    {
        "isaaclab_arena.terms.events.set_object_pose",
        "isaaclab_arena.terms.events.set_object_pose_per_env",
        "isaaclab_arena.terms.events.reset_placement_asset_pose",
        "isaaclab_arena.terms.events.reset_placement_asset_pose_per_env",
        "isaaclab_arena.terms.events.reset_articulation_pose_and_joints",
        "isaaclab_arena.terms.events.reset_articulation_joints",
        "isaaclab_arena.terms.events.reset_all_articulation_joints",
        "isaaclab_arena.terms.events.ResetBackgroundPhysics",
    }
)
"""Not here, because they draw: ``randomize_object_pose`` (a ``PoseRange`` initial pose) and
``solve_and_place_objects`` while it solves live (rows without ``placement``: written by an Arena
without IsaacLab-Arena#1427, or a scene Arena does not record, see ``supports_recorded_placement``)."""


def deterministic_reset(func):
    """Declare a reset event function or term class as drawing no random numbers (replays exactly)."""
    setattr(func, DETERMINISTIC_RESET_ATTRIBUTE, True)
    return func


def draws_through_variation(event_name: str):
    """Declare a reset or interval event function or term class as drawing only through the variation whose event
    Arena composes as ``event_name`` (``build_event_cfg``'s name, the key of :func:`bound_variation_events`)."""

    def mark(func):
        setattr(func, DRAWS_THROUGH_VARIATION_ATTRIBUTE, str(event_name))
        return func

    return mark


def _qualified_name(func) -> str:
    return f"{getattr(func, '__module__', '')}.{getattr(func, '__qualname__', repr(func))}"


def _replays_recorded_placement(params: Mapping[str, Any]) -> bool:
    """Whether ``params`` are those of Arena's placement event bound to recorded rows."""
    handle = params.get("placement_sampler")
    return handle is not None and hasattr(handle, "sampler") and handle.pool is None


def _replays_with(sampler, bound) -> bool:
    """Whether ``sampler`` (an event's, a deep copy) draws through ``bound``'s replay (a bound variation's sampler)."""
    sampler = getattr(sampler, "replays_through", sampler)
    replay = getattr(bound, "_replay_sampler", None)
    return type(sampler) is type(bound) and replay is not None and getattr(sampler, "_replay_sampler", None) is replay


def undeclared_reset_events(events_cfg, modes: tuple[str, ...] = ("reset", "interval"),
                            bound_events: Mapping[str, Any] | None = None) -> list[str]:
    """Events of ``events_cfg`` in ``modes`` that are neither variations nor declared deterministic."""
    terms = {name: term for name, term in (vars(events_cfg) if events_cfg is not None else {}).items()
             if callable(getattr(term, "func", None))}
    replayed_variations = set()
    undeclared = []
    for name, term in terms.items():
        params = getattr(term, "params", None) or {}
        samplers = [value for value in params.values() if hasattr(value, "set_replay_sampler")]
        if not samplers:
            continue
        if bound_events is None or (
                name in bound_events and all(_replays_with(value, bound_events[name]) for value in samplers)):
            replayed_variations.add(name)
        elif getattr(term, "mode", None) in modes:
            undeclared.append(f"{name} ({_qualified_name(term.func)}, mode {term.mode}) holds a sampler that no "
                              "recorded variation binds, so it draws live")
    for name, term in terms.items():
        params = getattr(term, "params", None) or {}
        if getattr(term, "mode", None) not in modes or name in replayed_variations or any(
                hasattr(value, "set_replay_sampler") for value in params.values()):
            continue
        if _replays_recorded_placement(params):
            continue
        func = term.func
        if getattr(func, DETERMINISTIC_RESET_ATTRIBUTE, False) or _qualified_name(func) in UPSTREAM_DETERMINISTIC_RESET_EVENTS:
            continue
        through = getattr(func, DRAWS_THROUGH_VARIATION_ATTRIBUTE, None)
        if through is not None and through in replayed_variations:
            continue
        if through is not None:
            why = "no recorded variation binds" if through in terms else "is not composed (its variation is disabled)"
            undeclared.append(f"{name} ({_qualified_name(func)}, mode {term.mode}) draws through the variation event "
                              f"{through}, which {why}, so it draws live")
            continue
        undeclared.append(f"{name} ({_qualified_name(func)}, mode {term.mode})")
    return undeclared


def bound_variation_events(variations: Mapping[str, Sequence[Any]]) -> dict[str, Any]:
    """``{event name: sampler}`` of every enabled run-time variation: the term Arena composes for it
    (``build_event_cfg``, the name ``ArenaEnvBuilder`` gives it) and the sampler a benchmark replay
    checked is recorded and bound. Matched by name, not ``id()``: the env config holds deep copies."""
    return {variation.build_event_cfg()[0]: variation.sampler for hosted in variations.values() for variation in hosted
            if variation.enabled and callable(getattr(variation, "build_event_cfg", None))}


def check_replayed_reset_events(events_cfg, *, what: str, strict: bool = False,
                                bound_events: Mapping[str, Any] | None = None) -> None:
    """Warn or refuse (:data:`REPLAY_GUARD_MODE`) when a replayed env has reset events that may draw;
    ``strict`` (a benchmark replay) refuses whatever the mode, audits startup events too, and with
    ``bound_events`` exempts only the events of the variations the replay bound, drawing through it."""
    problems = [
        f"reset event {event} is not declared deterministic: if it draws, replayed episodes differ from the "
        "recorded ones (declare it with replay_guards.deterministic_reset, or make the draw an Arena variation)"
        for event in undeclared_reset_events(events_cfg, ("reset", "interval", "startup") if strict else ("reset", "interval"),
                                             bound_events)
    ]
    if strict and problems:
        raise ReplayGuardError(f"[DexBench benchmark replay] {what}:\n  - " + "\n  - ".join(problems))
    enforce(problems, what)




def check_recorded_draws(variations: Mapping[str, Sequence[Any]], path: str | Path) -> None:
    """Hand each variation that checks its recorded draws (``check_recorded_draws``, :class:`ObjectPoseVariation`)
    every row's draw for its key, before Arena binds and builds: a draw outside its domain refuses the build."""
    records = read_episode_records(path)
    for asset, hosted in variations.items():
        for variation in hosted:
            check = getattr(variation, "check_recorded_draws", None)
            if check is None:
                continue
            key = f"{asset}.{variation.name}"
            draws = [(label, record["variations"][key]) for label, record in records
                     if isinstance(record.get("variations"), Mapping) and key in record["variations"]]
            if draws and len(draws) == len(records):
                check(draws)


def enabled_variation_keys(variations: Mapping[str, Sequence[Any]]) -> set[str]:
    """``<asset>.<variation>`` of every enabled variation in ``{asset: [variation, ...]}`` (Arena's record keys)."""
    return {f"{asset}.{variation.name}" for asset, hosted in variations.items() for variation in hosted if variation.enabled}


def variation_ranges(variations: Mapping[str, Sequence[Any]]) -> dict[str, tuple[list[float], list[float]]]:
    """``{key: (low, high)}`` of every enabled variation whose sampler draws a numeric box (after the overrides)."""
    ranges = {}
    for asset, hosted in variations.items():
        for variation in hosted:
            sampler_cfg = getattr(getattr(variation, "cfg", None), "sampler_cfg", None)
            low, high = getattr(sampler_cfg, "low", None), getattr(sampler_cfg, "high", None)
            if variation.enabled and low is not None and high is not None:
                ranges[f"{asset}.{variation.name}"] = ([float(v) for v in low], [float(v) for v in high])
    return ranges


RANGE_TOLERANCE = 1e-6
"""How far past its range a recorded draw may sit (a float32 draw's JSON round trip)."""


def benchmark_replay_problems(
    identity: ObjectIdentity | None, records: Sequence[tuple[str, Mapping[str, Any]]], enabled_keys: Iterable[str],
    ranges: Mapping[str, tuple[Sequence[float], Sequence[float]]] | None = None,
) -> list[str]:
    """What keeps ``records`` from being a strict replay on this build: one line per kind, with its first row."""
    if identity is None:
        return ["the task has no object axis, so its rows cannot be checked against the build's object"]
    if not records:
        return ["the rows file holds no rows"]
    enabled = sorted(enabled_keys)
    found: dict[str, tuple[int, str]] = {}

    def note(problem: str, label: str) -> None:
        count, first = found.get(problem, (0, label))
        found[problem] = (count + 1, first)

    for label, record in records:
        task_object = recorded(record, TASK_OBJECT_KEY)
        if not isinstance(task_object, Mapping) or task_object.get("axis") != identity.field:
            note(f"records no task_object on axis {identity.field!r} (a benchmark row never falls back to a default)", label)
        else:
            if task_object.get("key") != identity.key:
                note(f"was made with {identity.field}={task_object.get('key')}, this build is {identity.key}", label)
            major_problem = asset_major_problem(identity.benchmark_asset_major,
                                                {"frozen source": task_object.get("asset_version"),
                                                 "runtime sidecar": identity.asset_version},
                                                rename_only=identity.benchmark_asset_major_renames)
            if major_problem:
                note(major_problem, label)
            fit = task_object.get("fit") or {}
            environment_cfg = recorded(record, "environment_cfg") or {}
            snapshot = {name: environment_cfg[name] for name in identity.fixture if name in environment_cfg and name not in fit}
            if snapshot and identity.fit_form is not None:
                snapshot = identity.fit_form(task_object.get("key") or identity.default, snapshot)
            fixture = {**environment_cfg, **snapshot, **fit}
            for name, value in identity.fixture.items():
                if name not in fixture:
                    note(f"records no {name} (in its task_object fit or environment_cfg)", label)
                elif not _same(fixture[name], value):
                    note(f"was made with {name}={fixture[name]}, this build is {value}", label)
        if "placement" in record:
            note("records a relation placement, which the benchmark does not admit (no task of it solves one)", label)
        draws = record.get("variations")
        if not isinstance(draws, Mapping):
            draws = {}
        missing = [key for key in enabled if key not in draws]
        if missing:
            note(f"does not record the enabled variation(s) {missing}, which would draw live", label)
        for key in enabled:
            if key not in draws:
                continue
            if draws[key] is None:
                note(f"records no value for {key} (null)", label)
            if key not in (ranges or {}):
                continue
            low, high = ranges[key]
            numbers = numeric_draw(draws[key])
            if numbers is None or len(numbers) != len(low):
                note(f"records {key} as {draws[key]!r}, not {len(low)} finite numbers", label)
            elif any(value < lo - RANGE_TOLERANCE or value > hi + RANGE_TOLERANCE for value, lo, hi in zip(numbers, low, high)):
                note(f"records {key} outside its declared range {low} .. {high}", label)
        if not is_finite(draws):
            note("records a non-finite draw", label)
    return [f"{count} of {len(records)} rows (first: {first}) {problem}" for problem, (count, first) in found.items()]


def check_benchmark_replay(identity: ObjectIdentity | None, path: str | Path, enabled_keys: Iterable[str], *, what: str,
                           ranges: Mapping[str, tuple[Sequence[float], Sequence[float]]] | None = None) -> None:
    """Refuse a benchmark replay of ``path`` that :func:`benchmark_replay_problems` finds fault with, whatever the mode."""
    problems = benchmark_replay_problems(identity, read_episode_records(path), enabled_keys, ranges)
    if problems:
        raise ReplayGuardError(f"[DexBench benchmark replay] {what} from {path}:\n  - " + "\n  - ".join(problems))


__all__ = [
    "DETERMINISTIC_RESET_ATTRIBUTE",
    "DRAWS_THROUGH_VARIATION_ATTRIBUTE",
    "benchmark_replay_problems",
    "check_benchmark_replay",
    "check_recorded_draws",
    "enabled_variation_keys",
    "variation_ranges",
    "UPSTREAM_DETERMINISTIC_RESET_EVENTS",
    "bound_variation_events",
    "check_replayed_reset_events",
    "deterministic_reset",
    "draws_through_variation",
    "undeclared_reset_events",
    "REPLAY_GUARD_MODE",
    "ObjectIdentity",
    "ReplayGuardError",
    "check_replay_source",
    "enforce",
    "read_dataset_sources",
    "read_episode_records",
    "recorded_object_key",
]
