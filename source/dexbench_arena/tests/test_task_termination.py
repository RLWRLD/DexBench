# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""The TaskTerminationCfg adapter: success is the predicate at every step, partial credit only scores."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("isaaclab")
pytest.importorskip("isaaclab_arena")

from isaaclab.managers import TerminationManager, TerminationTermCfg  # noqa: E402
from isaaclab.utils.configclass import configclass  # noqa: E402
from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker  # noqa: E402

from dexbench_arena.tasks.task_termination import composed_success_term, task_termination_cfg  # noqa: E402


def _tracker(termination, env):
    return ProgressTracker(
        termination.success,
        num_envs=env.num_envs,
        device="cpu",
        env=env,
        subtasks_are_sequential=termination.subtasks_are_sequential,
        desired_subtask_success_state=termination.desired_subtask_success_state,
    )


def _flag(env, name):
    return getattr(env, name).clone()


@configclass
class _Terms:
    success: TerminationTermCfg = TerminationTermCfg(func=_flag, params={"name": "ok"})
    dropped: TerminationTermCfg = TerminationTermCfg(func=_flag, params={"name": "dropped"})


def test_success_is_the_predicate_now_and_partial_credit_never_gates_it():
    """Success follows the predicate step by step (no latch); the chain only scores."""
    env = SimpleNamespace(num_envs=2, scene=None, episode_length_buf=torch.zeros(2, dtype=torch.long))
    chain = [lambda env: env.lifted.clone(), lambda env: env.over.clone()]
    termination = task_termination_cfg(12.0, _Terms(), partial_credit=chain, partial_credit_name="carry")
    assert termination.timeout_s == 12.0
    assert set(termination.failures) == {"dropped"}
    tracker = _tracker(termination, env)
    T, F = True, False
    script = [
        ((F, F), (F, T), (F, F)),
        ((T, F), (F, T), (F, T)),
        ((T, F), (F, T), (F, F)),
        ((F, T), (F, F), (F, F)),
    ]
    for step, (ok, lifted, over) in enumerate(script, start=1):
        env.ok, env.lifted, env.over = (torch.tensor(v) for v in (ok, lifted, over))
        env.episode_length_buf[:] = step
        tracker.step(env, step_index=env.episode_length_buf.clone())
        assert tracker.is_complete().tolist() == list(ok)
    states = tracker.get_state()
    assert [s.overall_score for s in states] == [0.0, 1.0]
    assert [s.criteria_by_name["carry"].is_complete for s in states] == [False, True]


def test_without_partial_credit_the_success_criteria_carries_the_score():
    termination = task_termination_cfg(None, _Terms())
    assert termination.timeout_s is None
    assert [c.name for c in termination.success] == ["success"]
    assert termination.success[0].score == 1.0
    cfg = SimpleNamespace(task_progress=SimpleNamespace(success_criteria=termination.success))
    term = composed_success_term(cfg)
    assert term.func is _flag and term.params == {"name": "ok"}
