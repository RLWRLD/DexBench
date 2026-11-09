# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Which object a task is run with: one table per task, the only list of its train and held-out objects."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

SPLITS = ("train", "eval", "stand-in", "unfit")
"""``train`` and ``eval`` are the benchmark split; ``stand-in`` builds but is in neither
(E2's parcel); ``unfit`` is listed so it is not dropped silently, and refused."""


@dataclass(frozen=True)
class TaskObject:
    """One object a task can be run with."""

    key: str
    """The value of the axis' config field, and the recording ID (``TC1``, ``train_0``, ...)."""
    asset: str
    """The registered Arena asset that spawns it."""
    split: str
    """One of :data:`SPLITS`."""
    label: str
    """What the task language calls it."""
    size_m: tuple[float, ...]
    """The measured size the task's fit rules read (the axis says which dimensions)."""
    unfit_reason: str | None = None
    benchmark_asset_major: int | None = None
    """An additional required MAJOR, including a refusal of missing stamps."""
    benchmark_asset_major_renames: tuple[tuple[int, int, str], ...] = ()
    """Exact old/new MAJOR pairs whose only breaking change was a name migration,
    with a repository evidence path. No transitive or future-major allowance."""

    def __post_init__(self) -> None:
        if self.split not in SPLITS:
            raise ValueError(f"{self.key}: split {self.split!r} is not one of {SPLITS}")
        if (self.split == "unfit") != (self.unfit_reason is not None):
            raise ValueError(f"{self.key}: an unfit object, and only one, says why")
        if self.benchmark_asset_major is not None and (type(self.benchmark_asset_major) is not int or self.benchmark_asset_major < 0):
            raise ValueError(f"{self.key}: benchmark_asset_major must be a nonnegative integer")
        _rename_major_pairs(self.benchmark_asset_major_renames)


@dataclass(frozen=True)
class ObjectAxis:
    """The objects one task is run with, selected by ``field`` of its environment config."""

    field: str
    """The typed environment-config field that names the object for a Run."""
    default: str
    """The object a config that does not set ``field`` builds. A recording without a
    ``task_object`` row was made with this object, unless ``unlabelled_default`` says otherwise."""
    objects: tuple[TaskObject, ...]
    size_axes: str
    """What ``TaskObject.size_m`` holds, e.g. ``"diameter, height"``."""
    fixture_profile: Mapping[str, Any] = field(default_factory=dict)
    """Config fields every Run of the split shares and the split's values (DB-R2's Shelf3
    raise). An experiment sets them in ``shared:``; the default config may differ (it builds
    the scene the published recordings were made in)."""
    fit: Callable[[TaskObject, Any], dict[str, Any]] | None = None
    """``fit(task_object, cfg)``: the geometry the factory derives from the object under the
    config's fixture settings, as JSON values; raises ValueError, with the measured shortfall,
    for an object the fixture profile cannot hold."""
    unlabelled_default: str | None = None
    """What a recording without a ``task_object`` row was made with, where ``default`` has moved since."""
    scene_layout: str | None = None
    """The layout the objects are placed in, where the scene has changed under the same object
    keys (DB-L2: PSL's E2 delivery). ``resolve`` puts it in the fit, so every ``task_object``
    row records it, and the replay guard compares it as one more fixture value."""
    unlabelled_scene_layout: str | None = None
    """What a source whose fit records no ``scene_layout`` was made on: the layout before the axis named one."""
    source_fit: tuple[str, ...] = ()
    """Fit values a source must have been made with, beyond the fixture profile: digests of the build's inputs
    (DB-R1's layout set, its size and SHA-256). The replay guard compares them as fixture values, so a replay of rows
    made from other inputs is refused before it starts; a source whose fit lacks one counts as made with None."""

    def __post_init__(self) -> None:
        keys = [obj.key for obj in self.objects]
        if len(set(keys)) != len(keys):
            raise ValueError(f"{self.field}: duplicate object keys {keys}")
        if self.default not in keys:
            raise ValueError(f"{self.field}: the default {self.default!r} is not one of {keys}")

    def keys(self, split: str | None = None) -> tuple[str, ...]:
        return tuple(obj.key for obj in self.objects if split is None or obj.split == split)

    def train(self) -> tuple[str, ...]:
        return self.keys("train")

    def eval(self) -> tuple[str, ...]:
        return self.keys("eval")

    def get(self, key: str) -> TaskObject:
        for obj in self.objects:
            if obj.key == key:
                return obj
        raise ValueError(f"{key!r} is not an object of {self.field}; choose from {list(self.keys())}")

    def selected(self, cfg) -> TaskObject:
        """The object ``cfg`` names (the default when it names none)."""
        return self.get(getattr(cfg, self.field, None) or self.default)

    def resolve(self, cfg) -> tuple[TaskObject, dict[str, Any]]:
        """The object ``cfg`` builds and its derived fit, with the axis' ``scene_layout``; ValueError
        for one it cannot build."""
        obj = self.selected(cfg)
        if obj.split == "unfit":
            raise ValueError(f"{self.field} {obj.key}: {obj.unfit_reason}")
        fit = dict(self.fit(obj, cfg)) if self.fit is not None else {}
        if self.scene_layout is not None:
            fit["scene_layout"] = self.scene_layout
        return obj, fit

    def profile_differences(self, cfg) -> dict[str, tuple[Any, Any]]:
        """``{field: (cfg's value, the profile's)}`` for every fixture setting ``cfg`` does not share."""
        return {
            name: (getattr(cfg, name, None), value)
            for name, value in self.fixture_profile.items()
            if getattr(cfg, name, None) != value
        }


def task_object_record(axis: ObjectAxis, task_object: TaskObject, fit: Mapping[str, Any], asset_version: str | None) -> dict:
    """What an episode row and a demo file carry as ``task_object``: what the build resolved, not its input."""
    return {
        "axis": axis.field,
        "key": task_object.key,
        "asset": task_object.asset,
        "split": task_object.split,
        "fit": dict(fit),
        "asset_version": asset_version,
    }


def _rename_major_pairs(renames) -> set[tuple[int, int]]:
    pairs = set()
    for old, new, evidence in renames:
        if (type(old) is not int or type(new) is not int or not 0 <= old < new
                or not isinstance(evidence, str) or not evidence.strip() or (old, new) in pairs):
            raise ValueError("A rename-only asset MAJOR exception needs a distinct increasing pair and evidence")
        pairs.add((old, new))
    return pairs


def asset_major_problem(required_major: int | None, versions: Mapping[str, Any], *, rename_only=()) -> str | None:
    """Compare known stamps with the last (current) version, permitting only named rename pairs."""
    pairs = _rename_major_pairs(rename_only)
    majors = {}
    for label, version in versions.items():
        if version is None and required_major is None:
            majors[label] = None
            continue
        if not isinstance(version, str) or re.fullmatch(r"\d+\.\d+\.\d+", version) is None:
            return f"{label} records no valid asset_version ({version!r}); required asset MAJOR is {required_major}"
        majors[label] = int(version.split(".", 1)[0])
        if required_major is not None and majors[label] != required_major:
            return f"{label} asset_version {version} differs from required asset MAJOR {required_major}"
    if not majors:
        return None
    current_label = next(reversed(majors))
    current = majors[current_label]
    source_label = next(iter(majors))
    for label, major in majors.items():
        if major is None:
            if label != source_label and majors[source_label] is not None:
                return f"{label} has no asset_version for known {source_label} asset MAJOR {majors[source_label]}"
            continue
        if current is None:
            return f"{current_label} has no asset_version for known {label} asset MAJOR {major}"
        if major != current and (major, current) not in pairs:
            return f"{label} asset MAJOR {major} differs from {current_label} asset MAJOR {current}"
    return None


__all__ = ["SPLITS", "ObjectAxis", "TaskObject", "asset_major_problem", "task_object_record"]
