# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""The cost ledger that caps what paid VLM runs spend together: every paid request reserves its bound before it is sent."""

from __future__ import annotations

import fcntl
import json
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class BudgetExhausted(RuntimeError):
    """The next request's worst case would take the ledger past its cap."""


def ledger_cost_usd(path: str | Path) -> float:
    """What a ledger holds: settled costs, plus open reservations at their bound, plus lines without a ``kind``."""
    path = Path(path)
    if not path.is_file():
        return 0.0
    return _total(path.read_text())


def _total(text: str) -> float:
    reserved: dict[str, float] = {}
    total = 0.0
    for line in text.splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        kind = entry.get("kind")
        if kind == "reserve":
            reserved[entry["id"]] = float(entry["cost_usd"])
        elif kind == "settle":
            reserved.pop(entry["id"], None)
            total += float(entry["cost_usd"])
        else:
            total += float(entry.get("cost_usd", 0.0))
    return total + sum(reserved.values())


class CostLedger:
    """One ledger file; ``cap_usd`` None records without refusing (an unscored run)."""

    def __init__(self, path: str | Path, cap_usd: float | None):
        self.path = Path(path)
        self.cap_usd = cap_usd
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a"):
            pass

    @contextmanager
    def _locked(self) -> Iterator[Any]:
        with open(self.path, "a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                handle.seek(0)
                yield handle
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def total_usd(self) -> float:
        with self._locked() as handle:
            return _total(handle.read())

    def reserve(self, bound_usd: float, **record: Any) -> str:
        """Append a reservation of ``bound_usd`` and return its id; BudgetExhausted if it would pass the cap."""
        with self._locked() as handle:
            spent = _total(handle.read())
            if self.cap_usd is not None and spent + bound_usd > self.cap_usd:
                raise BudgetExhausted(
                    f"VLM cost budget: {spent:.2f} USD recorded in {self.path} plus this request's bound "
                    f"{bound_usd:.2f} USD would pass cost_budget_usd {self.cap_usd:.2f} USD; stopping the run")
            reservation = uuid.uuid4().hex
            handle.write(json.dumps({"kind": "reserve", "id": reservation, "time": time.time(),
                                     "cost_usd": bound_usd, **record}) + "\n")
            handle.flush()
        return reservation

    def settle(self, reservation: str, cost_usd: float, **record: Any) -> None:
        """Record what a reserved request cost; its reservation stops counting."""
        with self._locked() as handle:
            handle.seek(0, 2)
            handle.write(json.dumps({"kind": "settle", "id": reservation, "time": time.time(),
                                     "cost_usd": cost_usd, **record}) + "\n")
            handle.flush()
