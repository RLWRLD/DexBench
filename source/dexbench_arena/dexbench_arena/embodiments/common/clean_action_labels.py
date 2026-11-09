# SPDX-License-Identifier: Apache-2.0
"""Clean action labels for Mimic generation with action noise (DART-style)."""

from __future__ import annotations

import math

import torch
from isaaclab.managers import RecorderTerm, RecorderTermCfg
from isaaclab.utils.configclass import configclass

_STASH_ATTR = "_dexbench_clean_actions"


def mimic_action_noise(value: float) -> float:
    """Validate per-step Mimic action noise from the task cfg (default 0)."""
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"action_noise={value!r}: action noise must be a finite value >= 0")
    return value


def mimic_draws_noise(env_cfg) -> bool:
    """Whether any of the Mimic cfg's subtasks draws action noise."""
    return any(float(subtask.action_noise) > 0
               for subtasks in (getattr(env_cfg, "subtask_configs", None) or {}).values() for subtask in subtasks)


def stash_clean_action(env, env_id: int, clean: torch.Tensor, played: torch.Tensor) -> None:
    """Keep ``env_id``'s clean action next to the ``played`` one ``target_eef_pose_to_action`` returns."""
    stash = env.__dict__.setdefault(_STASH_ATTR, {})
    stash[int(env_id)] = (clean.detach().clone(), played.detach().clone())


def clean_action_batch(env, actions: torch.Tensor) -> torch.Tensor | None:
    """``actions`` with each row replaced by its env's clean action, or ``None`` when nothing was stashed."""
    stash = getattr(env, _STASH_ATTR, None)
    if not stash:
        return None
    clean = actions.clone()
    for env_id, (clean_row, played_row) in stash.items():
        if env_id < actions.shape[0] and torch.equal(played_row.to(actions.device), actions[env_id]):
            clean[env_id] = clean_row.to(actions.device)
    return clean


class CleanActionsRecorder(RecorderTerm):
    """``actions_clean``: the action before Mimic's noise, for the step ``actions`` records."""

    def record_pre_step(self):
        if not mimic_draws_noise(self._env.cfg):
            return None, None
        clean = clean_action_batch(self._env, self._env.action_manager.action)
        return (None, None) if clean is None else ("actions_clean", clean)


@configclass
class CleanActionsRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = CleanActionsRecorder
