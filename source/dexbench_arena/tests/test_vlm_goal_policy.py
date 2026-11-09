# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""A paid VLM goal policy run: its key stays in the environment, and its cost cap holds across the runs sharing a ledger."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("isaaclab.utils.math", exc_type=ImportError)
pytest.importorskip("isaaclab_arena.policy.policy_base", exc_type=ImportError)

from dexbench_arena_vlm_agent import policy as vlm  # noqa: E402
from dexbench_arena_vlm_agent.cost_ledger import BudgetExhausted, CostLedger, ledger_cost_usd  # noqa: E402

KEY = "sk-proj-TESTONLYnotarealkey0123456789"


def _scored(tmp_path, **fields):
    return vlm.VlmGoalPolicyCfg(**{"policy_variant": "astra_l1", "task_recipe": "db_r2", "output_dir": str(tmp_path),
                                   "cost_budget_usd": 5.0, "cost_ledger_path": str(tmp_path / "study" / "costs.jsonl"),
                                   **fields})


def test_a_scored_policy_reads_its_key_from_the_environment_and_writes_it_nowhere(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    policy = vlm.VlmGoalPolicy(_scored(tmp_path))
    files = "".join(path.read_text() for path in tmp_path.rglob("*") if path.is_file())
    assert KEY not in files + json.dumps(policy.policy_settings()) + repr(policy.config)
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
        vlm.VlmGoalPolicy(_scored(tmp_path))


def test_the_cost_cap_refuses_a_request_whose_worst_case_would_pass_it_across_runs(tmp_path):
    ledger = CostLedger(tmp_path / "costs.jsonl", 5.0)
    other_run = CostLedger(tmp_path / "costs.jsonl", 5.0)
    first, second = ledger.reserve(2.0), other_run.reserve(2.5)
    with pytest.raises(BudgetExhausted):
        ledger.reserve(1.0)
    ledger.settle(first, 0.4)
    other_run.reserve(1.0)
    assert ledger_cost_usd(ledger.path) == pytest.approx(0.4 + 2.5 + 1.0)
    other_run.settle(second, 2.5)
