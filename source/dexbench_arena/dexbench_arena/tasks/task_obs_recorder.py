# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Record a task's own observation group (``task_obs``) in every demo, beside the policy group."""

from __future__ import annotations

from isaaclab.managers.manager_term_cfg import RecorderTermCfg
from isaaclab.managers.recorder_manager import RecorderTerm
from isaaclab.utils.configclass import configclass

from dexbench_arena.embodiments.recorder_terms import add_declared_recorder_term

TASK_OBS_GROUP = "task_obs"


class TaskObservationsRecorder(RecorderTerm):
    """Record the task's observation group before each step (as the policy group's recorder does)."""

    def record_pre_step(self):
        group = getattr(self._env, "obs_buf", {}).get(TASK_OBS_GROUP)
        return (TASK_OBS_GROUP, group) if group is not None else (None, None)


@configclass
class TaskObservationsRecorderCfg(RecorderTermCfg):
    """Configuration for :class:`TaskObservationsRecorder`."""

    class_type: type[RecorderTerm] = TaskObservationsRecorder


def with_task_obs_recording(previous):
    """Chain an env cfg callback that adds :class:`TaskObservationsRecorder` to the demo and Mimic recorders."""

    def callback(env_cfg):
        if previous is not None:
            env_cfg = previous(env_cfg) or env_cfg
        add_declared_recorder_term(env_cfg, "record_task_obs", TaskObservationsRecorderCfg())
        return env_cfg

    return callback
