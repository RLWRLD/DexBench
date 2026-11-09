# SPDX-License-Identifier: Apache-2.0
"""What each enabled variation drew for an environment's current episode, as numbers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def dotted_items(values: Mapping, prefix: str = "") -> list[tuple[str, object]]:
    """A nested ``variations`` mapping as ``(dotted.key, value)`` pairs in its own order, flat or nested alike."""
    pairs = []
    for key, value in values.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        pairs += dotted_items(value, path) if isinstance(value, Mapping) else [(path, value)]
    return pairs


def _numbers(row) -> list[float] | None:
    values = row.tolist() if hasattr(row, "tolist") else row
    if isinstance(values, (int, float)):
        return [float(values)]
    try:
        flat = []
        stack = [values]
        while stack:
            item = stack.pop(0)
            if isinstance(item, (list, tuple)):
                stack[:0] = list(item)
            else:
                flat.append(float(item))
        return flat
    except (TypeError, ValueError):
        return None


def episode_variation_draws(env, env_ids: Sequence[int]) -> dict[str, list[list[float]]]:
    """``<asset>/<variation>`` -> one row of numbers per environment in ``env_ids``."""

    recorder = getattr(env, "variation_recorder", None)
    draws: dict[str, list[list[float]]] = {}
    if recorder is None:
        return draws
    for key, record in recorder.records.items():
        encoded = getattr(getattr(record, "cfg", None), "dataset_values", None) or {}
        rows = []
        for env_id in env_ids:
            row = record.sample_for_episode(int(env_id), env.get_episode_index(int(env_id)))
            row = encoded.get(row, row) if isinstance(row, str) else row
            numbers = None if row is None or isinstance(row, str) else _numbers(row)
            if numbers is None:
                break
            rows.append(numbers)
        else:
            draws[key.replace(".", "/")] = rows
    return draws
