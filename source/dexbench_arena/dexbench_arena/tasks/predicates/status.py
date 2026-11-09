# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""What a success term saw at its last evaluation, per condition, for an operator watching a take."""

from __future__ import annotations

import torch

from dexbench_arena.tasks.predicates.object_state import get_env

_STATUS_ATTR = "_dexbench_condition_status"


def publish(env, task: str, values: dict) -> None:
    """Keep ``values`` (``{name: per-env tensor, bool or number}``) as ``task``'s newest status."""
    base = get_env(env)
    base.__dict__[_STATUS_ATTR] = {"task": task, "step": int(getattr(base, "common_step_counter", 0)), "values": values}


def latest(env, env_id: int = 0) -> dict | None:
    """The newest published status as plain values for ``env_id``, or None when no term published one."""
    status = get_env(env).__dict__.get(_STATUS_ATTR)
    if status is None:
        return None
    values = {}
    for name, value in status["values"].items():
        if isinstance(value, torch.Tensor):
            value = value[env_id] if value.ndim else value
            value = value.item()
        values[name] = round(value, 4) if isinstance(value, float) else value
    return {"task": status["task"], "step": status["step"], **values}
