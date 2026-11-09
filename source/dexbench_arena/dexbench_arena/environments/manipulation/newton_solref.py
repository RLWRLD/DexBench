# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""A MuJoCo solref as the Newton shape gains (ke, kd) that SolverMuJoCo turns back into it."""

from __future__ import annotations


def solref_to_newton_ke_kd(timeconst: float, dampratio: float) -> tuple[float, float]:
    """Newton shape (ke, kd) whose MuJoCo geom solref is (timeconst, dampratio)."""

    if timeconst <= 0.0 or dampratio <= 0.0:
        raise ValueError(f"solref needs a positive timeconst and dampratio, got ({timeconst}, {dampratio})")
    return 1.0 / (timeconst * dampratio) ** 2, 2.0 / timeconst
