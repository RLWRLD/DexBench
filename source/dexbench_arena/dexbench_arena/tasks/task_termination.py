# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""A DexBench task's terms as Arena's ``TaskTerminationCfg``, scored as Isaac Lab termination terms."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

from isaaclab.managers import TerminationTermCfg

from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
from isaaclab_arena.progress_tracking.progress_tracking_utils import Predicate
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg

SUCCESS_CRITERIA_NAME = "success"


def task_termination_cfg(
    timeout_s: float | None,
    terms: object,
    partial_credit: Sequence[Predicate] | None = None,
    partial_credit_name: str = "partial_credit",
    partial_credit_description: str | None = None,
) -> TaskTerminationCfg:
    """``terms`` (a configclass with ``success`` and failure fields) as Arena's ``TaskTerminationCfg``."""
    by_name = {field.name: getattr(terms, field.name) for field in dataclasses.fields(terms)}
    success = by_name.pop(SUCCESS_CRITERIA_NAME)
    assert isinstance(success, TerminationTermCfg), "a DexBench task needs one success term"
    failures = {name: term for name, term in by_name.items() if term is not None}
    criteria = [
        CompletionCriteria(
            name=SUCCESS_CRITERIA_NAME,
            predicate_sequence=[success],
            score=0.0 if partial_credit else 1.0,
            parent_subtask_idx=0,
        )
    ]
    desired: list[bool | None] = [True]
    if partial_credit:
        criteria.append(
            CompletionCriteria(
                name=partial_credit_name,
                predicate_sequence=list(partial_credit),
                description=partial_credit_description,
                parent_subtask_idx=1,
            )
        )
        desired.append(None)
    return TaskTerminationCfg(
        timeout_s=timeout_s, success=criteria, failures=failures, desired_subtask_success_state=desired
    )


def composed_success_term(env_cfg) -> TerminationTermCfg:
    """The task's success predicate in an env cfg Arena composed from :func:`task_termination_cfg`."""
    criteria = {criteria.name: criteria for criteria in env_cfg.task_progress.success_criteria}
    return criteria[SUCCESS_CRITERIA_NAME].predicate_sequence[0]
