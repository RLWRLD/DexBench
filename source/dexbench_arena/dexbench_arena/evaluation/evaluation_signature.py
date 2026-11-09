# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""What an episode was evaluated under, beyond its scene: so two models are never compared across settings."""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import inspect
import json
import math
import os
import re
import subprocess
from collections.abc import Collection, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

from dexbench_arena import REPO_ROOT
from dexbench_arena.evaluation.artifacts import numeric_draw, recorded

EVALUATION_SETTINGS_KEY = "evaluation_settings"
_MAX_DEPTH = 3
_CACHE_ATTRIBUTE = "_dexbench_evaluation_settings"
RECORDING_FIELDS = ("record_dataset", "record_dataset_images")
"""What a run keeps on disk, not how it is scored: outside the signature and outside a cell's declaration
(benchmark.expand), so a benchmark run may keep its per-step state or not and score the same."""
_NOT_SETTINGS = frozenset({"enable_cameras", "camera_scale", "record_settle", *RECORDING_FIELDS})


NEWTON_SHAPE_OVERRIDES_ATTRIBUTE = "dexbench_newton_shape_overrides"
"""Attribute of a composed env cfg that lists the per-shape Newton contact an environment installed through
builder hooks (DB-R2's basket and drinks), as plain values: those changes are made on the model builder, not in
``sim.physics``, so they reach the recorded physics only through it. Absent where nothing was installed."""

SETTINGS_VERSION = 3
"""The ``evaluation_settings`` payload's version; a row with another, or without one of
:data:`REQUIRED_SETTINGS_KEYS`, is not scored (its settings are not known in full). Version 2 cut a term's
parameters three containers deep, so two of L2's success regions with as many rectangles recorded alike."""
REQUIRED_SETTINGS_KEYS = ("version", "physics", "control", "episode_length_s", "terminations", "events", "code", "policy")
SUCCESS_CRITERIA_KEY = "success_criteria"
"""The task's completion criteria (:func:`task_success_criteria`). Not one of :data:`REQUIRED_SETTINGS_KEYS`
and no new :data:`SETTINGS_VERSION`: version-3 rows without it score, and the scorer compares
the criteria only between rows that both carry them."""
UNENCODABLE_PREDICATE = "unencodable"
"""Marks a recorded predicate whose parameters live in the callable's own state (:func:`_predicate`)."""


MOVED_DEFINITIONS: Mapping[str, str] = MappingProxyType({
    "dexbench_arena.db_l2_mimic_utils.package_clear_of_cage":
        "dexbench_arena.tasks.cage_clear.package_clear_of_cage",
    "dexbench_arena.db_l2_mimic_utils.reset_cage_clear":
        "dexbench_arena.tasks.cage_clear.reset_cage_clear",
    "dexbench_arena.scripted.db_l2.mimic_utils.package_clear_of_cage":
        "dexbench_arena.tasks.cage_clear.package_clear_of_cage",
    "dexbench_arena.scripted.db_l2.mimic_utils.reset_cage_clear":
        "dexbench_arena.tasks.cage_clear.reset_cage_clear",
})
"""Functions and classes that moved without a change, as {the path a row recorded: the current path}."""


def _current_names(value: Any) -> Any:
    """``value`` (recorded settings) with each :data:`MOVED_DEFINITIONS` path read as its current one."""
    if isinstance(value, str):
        if value in MOVED_DEFINITIONS:
            return MOVED_DEFINITIONS[value]
        if value.startswith("<callable:") and value.endswith(">") and value[10:-1] in MOVED_DEFINITIONS:
            return f"<callable:{MOVED_DEFINITIONS[value[10:-1]]}>"
        return value
    if isinstance(value, Mapping):
        return {key: _current_names(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_current_names(item) for item in value]
    return value


def _qualified(func) -> str:
    return f"{getattr(func, '__module__', '?')}.{getattr(func, '__qualname__', type(func).__name__)}"


def _plain(value: Any, depth: int = 0, _seen: frozenset[int] = frozenset(), numbers: bool = False) -> Any:
    """``value`` as JSON values. Mappings, lists and tuples are kept whole however deep they nest (L2's
    ``surface_xy_rects`` is rectangles of corners of coordinates), with a marker in place of a container that
    contains itself; ``depth`` bounds only how many config dataclasses deep the walk goes. With ``numbers``, a
    tensor or an array (anything with ``tolist()``) is its numbers; the completion criteria are recorded so
    (:func:`task_success_criteria`), while the terms and events keep the marker that version-3 rows record,
    so their signatures do not change. What has no plain form (an asset handle, a sampler, a function),
    or a dataclass past that bound, is recorded as a marker naming its type, never dropped; such a marker does
    not tell two values of the same type apart."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if id(value) in _seen:
        return f"<cycle:{type(value).__name__}>"
    seen = _seen | {id(value)}
    if isinstance(value, Mapping):
        return {str(key): _plain(item, depth, seen, numbers) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item, depth, seen, numbers) for item in value]
    tolist = getattr(value, "tolist", None) if numbers else None
    if callable(tolist) and not isinstance(value, type):
        return _plain(tolist(), depth, seen, numbers)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        if depth >= _MAX_DEPTH:
            return f"<truncated:{type(value).__name__}>"
        fields = {field.name: _plain(getattr(value, field.name, None), depth + 1, seen, numbers)
                  for field in dataclasses.fields(value)}
        return {"type": f"{type(value).__module__}.{type(value).__qualname__}", **fields}
    if callable(value):
        return f"<callable:{_qualified(value)}>"
    return f"<unencodable:{type(value).__name__}>"


def _term(term, numbers: bool = False) -> dict[str, Any]:
    """A manager term as its callable (module.qualname), mode, interval and plain parameters."""
    record = {"func": _qualified(getattr(term, "func", None)),
              "params": _plain(dict(getattr(term, "params", None) or {}), depth=1, numbers=numbers)}
    for attribute in ("mode", "interval_range_s", "is_global_time"):
        if hasattr(term, attribute):
            record[attribute] = _plain(getattr(term, attribute), depth=1, numbers=numbers)
    return record


def _manager_terms(group, modes: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Manager cfgs and dicts are both supported; removed/empty groups have no terms."""
    terms = {}
    entries = group if isinstance(group, dict) else vars(group) if group is not None else {}
    for name, term in sorted(entries.items()):
        if not callable(getattr(term, "func", None)):
            continue
        if modes is not None and getattr(term, "mode", None) not in modes:
            continue
        terms[name] = _term(term)
    return terms


def _predicate(predicate) -> dict[str, Any]:
    """One predicate of a completion criteria: a term (its callable and parameters, :func:`_term`), a
    consecutive-steps requirement around one (Arena's ``TrueForConsecutiveStepsCfg``), or a bare callable
    or ``functools.partial``."""
    if hasattr(predicate, "predicate") and hasattr(predicate, "required_steps"):
        return {"type": _qualified(type(predicate)), "predicate": _predicate(predicate.predicate),
                "required_steps": _plain(predicate.required_steps, numbers=True)}
    if isinstance(predicate, functools.partial):
        return {"func": _qualified(predicate.func), "args": _plain(list(predicate.args), depth=1, numbers=True),
                "params": _plain(dict(predicate.keywords or {}), depth=1, numbers=True)}
    if callable(getattr(predicate, "func", None)):
        return _term(predicate, numbers=True)
    if inspect.isfunction(predicate) or inspect.isbuiltin(predicate):
        return {"func": _qualified(predicate), "params": {}}
    if dataclasses.is_dataclass(predicate) and not isinstance(predicate, type):
        return _plain(predicate, numbers=True)
    return {"func": _qualified(type(predicate)), UNENCODABLE_PREDICATE: True}


def _has_unencodable_predicate(value: Any) -> bool:
    if isinstance(value, Mapping):
        return bool(value.get(UNENCODABLE_PREDICATE)) or any(_has_unencodable_predicate(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_unencodable_predicate(item) for item in value)
    return False


def _sequence(sequence) -> list[Any]:
    """A predicate sequence as declared: predicates, or ``(predicate, score)`` pairs."""
    return [{"predicate": _predicate(item[0]), "score": _plain(item[1])} if isinstance(item, tuple) else _predicate(item)
            for item in sequence or ()]


def task_success_criteria(env_cfg) -> dict[str, Any] | None:
    """The completion criteria Arena's ``ProgressTracker`` decides success and progress from
    (``env_cfg.task_progress``): every criteria, the success predicate's and the partial-credit chain's
    (``tasks.task_termination``), with its predicate sequences (every predicate's callable and parameters),
    how its sequences combine, and the subtask it belongs to (``parent_subtask_idx``, which the tracker pairs
    with ``desired_subtask_success_state``); and that desired state. None for an env without ``task_progress``."""
    progress = getattr(env_cfg, "task_progress", None)
    if progress is None:
        return None
    criteria = []
    for item in getattr(progress, "success_criteria", None) or ():
        sequences = getattr(item, "predicate_sequences", None)
        if sequences is None:
            sequences = {"default_sequence": getattr(item, "predicate_sequence", None)}
        logical = getattr(item, "logical", None)
        criteria.append({"name": getattr(item, "name", None), "logical": getattr(logical, "value", logical),
                         "K": getattr(item, "K", None), "score": getattr(item, "score", None),
                         "parent_subtask_idx": getattr(item, "parent_subtask_idx", None),
                         "sequences": {name: _sequence(sequence) for name, sequence in sequences.items()}})
    return {"criteria": criteria,
            "subtasks_are_sequential": getattr(progress, "subtasks_are_sequential", None),
            "desired_subtask_success_state": _plain(getattr(progress, "desired_subtask_success_state", None))}


SOURCE_COMMIT_FILE = ".source-commit"
"""The commit a staged tree without ``.git`` was archived from (the pool's git-archive staging writes it)."""
IMAGE_REVISION_VARIABLE = "DEXBENCH_IMAGE_REVISION"
"""The commit an image baked its workspace from, without ``.git`` (as ``recording.rerecord_metadata.source_revision`` reads it)."""


def _commit(text: str) -> str | None:
    words = text.split()
    return words[0] if words and re.fullmatch(r"[0-9a-f]{40}", words[0]) else None


@functools.lru_cache(maxsize=1)
def _process_code_revision(root: Path, process_id: int) -> dict[str, Any]:
    return code_revision(root=root)


def code_revision(root: str | Path | None = None, environ: Mapping[str, str] | None = None,
                  *, refresh: bool = False) -> dict[str, Any]:
    """The source commit and the assets tree the env runs from, and where the commit was read
    (``revision_source``): git; else the tree's ``.source-commit`` (a git-archive staging); else
    ``DEXBENCH_IMAGE_REVISION``; else "unknown". The assets tree comes from git only.
    Default calls share one source snapshot per process, returned as independent dictionaries.
    Submission preparation passes ``refresh=True`` to recheck the checkout and replace that snapshot;
    an explicit root or environment always reads afresh. Each env also caches its settings."""
    if root is None and environ is None:
        if refresh:
            _process_code_revision.cache_clear()
        return dict(_process_code_revision(REPO_ROOT, os.getpid()))
    root = (Path(root) if root is not None else REPO_ROOT).resolve()
    environ = os.environ if environ is None else environ

    has_git = (root / ".git").exists() or bool(environ.get("GIT_DIR"))

    git = ["git", "-c", f"safe.directory={root}", "-C", str(root)]

    def read(*arguments: str) -> str:
        if not has_git:
            return "unknown"
        try:
            result = subprocess.run([*git, *arguments], capture_output=True, text=True, timeout=5)
        except (OSError, subprocess.SubprocessError):
            return "unknown"
        return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else "unknown"

    revision, source = read("rev-parse", "HEAD"), "git"
    if revision == "unknown":
        staged = root / SOURCE_COMMIT_FILE
        try:
            revision, source = _commit(staged.read_text()) or "unknown", SOURCE_COMMIT_FILE
        except OSError:
            revision = "unknown"
        if revision == "unknown":
            revision, source = _commit(environ.get(IMAGE_REVISION_VARIABLE, "")) or "unknown", IMAGE_REVISION_VARIABLE
        if revision == "unknown":
            source = "unknown"
    dirty = None
    if has_git:
        try:
            status = subprocess.run([*git, "status", "--porcelain", "--untracked-files=normal"],
                                    capture_output=True, text=True, timeout=5)
            if status.returncode == 0:
                dirty = bool(status.stdout.strip())
        except (OSError, subprocess.SubprocessError):
            pass
    return {"revision": revision, "revision_source": source, "assets": read("rev-parse", "HEAD:assets"), "dirty": dirty}


def evaluation_settings(env_cfg, code: Mapping[str, str] | None = None) -> dict[str, Any]:
    """What an episode was evaluated under, beyond its scene (:data:`REQUIRED_SETTINGS_KEYS`):"""
    sim = getattr(env_cfg, "sim", None)
    physics = _plain(getattr(sim, "physics", None))
    overrides = getattr(env_cfg, NEWTON_SHAPE_OVERRIDES_ATTRIBUTE, None)
    if overrides and isinstance(physics, dict):
        physics = {**physics, "shape_overrides": _plain(overrides)}
    return {
        "version": SETTINGS_VERSION,
        "physics": physics,
        "control": {"sim_dt": getattr(sim, "dt", None), "decimation": getattr(env_cfg, "decimation", None)},
        "episode_length_s": getattr(env_cfg, "episode_length_s", None),
        "terminations": _manager_terms(getattr(env_cfg, "terminations", None)),
        "events": _manager_terms(getattr(env_cfg, "events", None), ("reset", "startup", "interval")),
        "code": dict(code) if code is not None else code_revision(),
        SUCCESS_CRITERIA_KEY: task_success_criteria(env_cfg),
    }


def settings_problems(settings: Any) -> list[str]:
    """Why ``settings`` (a row's ``evaluation_settings``) cannot be scored: another version, a required key missing,
    or a success criteria predicate recorded without its parameters."""
    if not isinstance(settings, Mapping):
        return ["records no evaluation_settings, so its settings are unknown"]
    if settings.get("version") == 2:
        return ["records evaluation_settings version 2, recorded before nested term parameters were kept whole "
                "(a <truncated:...> marker can stand for different values): its settings are not known in full"]
    if settings.get("version") != SETTINGS_VERSION:
        return [f"records evaluation_settings version {settings.get('version')!r}, not {SETTINGS_VERSION}: its settings are not known in full"]
    missing = [key for key in REQUIRED_SETTINGS_KEYS if key not in settings]
    if missing:
        return [f"records evaluation_settings without {missing}, so its settings are unknown"]
    if _has_unencodable_predicate(settings.get(SUCCESS_CRITERIA_KEY)):
        return ["records a success criteria predicate that keeps its parameters in its own state (a callable "
                "object), so its success definition is not known in full; make it a function, a partial or a term"]
    return []


DRAW_TOLERANCE = 1e-6
"""How far a recorded number may be from its declared or source value (the JSON round trip of a float32 tensor)."""


def same_value(recorded_value: Any, declared: Any) -> bool:
    """Equal, or both numbers (or lists of numbers) of one length within :data:`DRAW_TOLERANCE`."""
    ours, theirs = numeric_draw(recorded_value), numeric_draw(declared)
    if ours is not None and theirs is not None:
        return len(ours) == len(theirs) and all(abs(a - b) <= DRAW_TOLERANCE for a, b in zip(ours, theirs))
    return recorded_value == declared


def declaration_problems(environment_cfg: Mapping[str, Any], declared: Mapping[str, Any], policy: Mapping[str, Any],
                         model: str, horizon: int | None, policy_execution: Mapping[str, Any] | None = None) -> list[str]:
    """How a Run departs from what its cell declares for ``model``: each declared environment field (the row's
    own fields over the cell's), the model's ``policy_variant``, the executed horizon (None for a trial, a
    tool-calling planner with no chunk) and the policy rate. ``policy_execution`` is the manifest's for the task
    (``{"policy": ..., "environment": ...}``, ``protocol.PolicyExecution``), None for a task whose policy is asked
    every control step: a model with a horizon must have run that rate (a row whose policy stamped no rate reads as
    every step, so it matches None and nothing else); a trial keeps the task's own control step, so the
    rate's environment fields are not held to it. The runner checks the Runs it composed, the scorer the episodes
    that ran."""
    execution = policy_execution or {}
    if horizon is None:
        declared = {name: value for name, value in declared.items() if name not in (execution.get("environment") or {})}
    problems = [f"{name}={environment_cfg.get(name)!r}, its cell declares {value!r}"
                for name, value in declared.items() if not same_value(environment_cfg.get(name), value)]
    if policy.get("policy_variant") != model:
        problems.append(f"policy_variant={policy.get('policy_variant')!r}, its Run is model {model!r}")
    if horizon is not None and policy.get("action_chunk_length") != horizon:
        problems.append(f"action_chunk_length={policy.get('action_chunk_length')!r}, the protocol's horizon is {horizon}")
    if horizon is not None:
        expected = dict(execution.get("policy") or {"policy_hz": None, "policy_action_interpolation": "none"})
        ran = {"policy_hz": policy.get("policy_hz"),
               "policy_action_interpolation": policy.get("policy_action_interpolation") or "none"}
        if ran != expected:
            problems.append(f"ran the policy at {ran}, its cell asks {expected}")
    return problems


POLICY_SETTINGS_ATTRIBUTE = "dexbench_policy_settings"
"""Where a policy leaves what it runs with on the env it is bound to (:func:`stamp_policy_settings`)."""


def stamp_policy_settings(env, settings: Mapping[str, Any]) -> None:
    """Called by a policy when it binds to ``env``: its model (``policy_variant``) and executed horizon,
    which the episode rows then carry under ``evaluation_settings["policy"]``."""
    setattr(env, POLICY_SETTINGS_ATTRIBUTE, _plain(dict(settings)))


def record_evaluation_settings(env, env_id: int) -> dict:
    """Episode recorder term: ``{"evaluation_settings": ...}`` of the env's composed config (computed once)
    and of the policy that ran the episode (None for a policy that does not stamp one)."""
    del env_id
    settings = getattr(env, _CACHE_ATTRIBUTE, None)
    if settings is None:
        settings = evaluation_settings(env.cfg)
        setattr(env, _CACHE_ATTRIBUTE, settings)
    return {EVALUATION_SETTINGS_KEY: {**settings, "policy": getattr(env, POLICY_SETTINGS_ATTRIBUTE, None)}}


def _without_none(value: Any) -> Any:
    """``value`` with every mapping key whose value is None left out, however deep the mappings nest."""
    if isinstance(value, Mapping):
        return {key: _without_none(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_without_none(item) for item in value]
    return value


def current_environment_cfg(environment_cfg: Mapping[str, Any],
                            legacy_values: Mapping[str, Mapping[str, str]] | None = None,
                            other_task_fields: Collection[str] = ()) -> dict[str, Any]:
    """A row's ``environment_cfg`` snapshot as a row recorded today records the same settings."""
    legacy_values = legacy_values or {}
    renamed = {name: legacy_values.get(name, {}).get(value, value) if isinstance(value, str) else value
               for name, value in environment_cfg.items()
               if name not in other_task_fields}
    return _without_none(renamed)


def evaluation_signature(row: Mapping[str, Any], legacy_values: Mapping[str, Mapping[str, str]] | None = None,
                         with_success_criteria: bool = True, other_task_fields: Collection[str] = ()) -> str:
    """SHA-256 of what ``row`` was evaluated under: its config snapshot less the object
    (:func:`current_environment_cfg`, with ``legacy_values`` and ``other_task_fields``), and its evaluation settings."""
    environment_cfg = current_environment_cfg(recorded(row, "environment_cfg") or {}, legacy_values, other_task_fields)
    task_object = recorded(row, "task_object")
    if isinstance(task_object, Mapping) and task_object.get("axis"):
        environment_cfg.pop(task_object["axis"], None)
    for name in _NOT_SETTINGS:
        environment_cfg.pop(name, None)
    settings = _current_names(dict(recorded(row, EVALUATION_SETTINGS_KEY) or {}))
    settings.pop("policy", None)
    settings.pop("code", None)
    if not with_success_criteria or settings.get(SUCCESS_CRITERIA_KEY) is None:
        settings.pop(SUCCESS_CRITERIA_KEY, None)
    payload = {"environment_cfg": environment_cfg, EVALUATION_SETTINGS_KEY: settings}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


__all__ = [
    "MOVED_DEFINITIONS",
    "DRAW_TOLERANCE",
    "EVALUATION_SETTINGS_KEY",
    "REQUIRED_SETTINGS_KEYS",
    "SETTINGS_VERSION",
    "SUCCESS_CRITERIA_KEY",
    "settings_problems",
    "POLICY_SETTINGS_ATTRIBUTE",
    "RECORDING_FIELDS",
    "code_revision",
    "current_environment_cfg",
    "declaration_problems",
    "evaluation_settings",
    "evaluation_signature",
    "record_evaluation_settings",
    "same_value",
    "stamp_policy_settings",
    "task_success_criteria",
]
