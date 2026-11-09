# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""DexBench compatibility for Arena's config-derived policy CLI."""

from __future__ import annotations

import argparse

from isaaclab_arena.evaluation.policy_runner_cli import add_policy_cli_args as _add_policy_cli_args


def add_policy_cli_args(parser: argparse.ArgumentParser, policy_type: type) -> argparse.ArgumentParser:
    """Arena's flags from the policy's config, then the policy type's own changes to them."""
    parser = _add_policy_cli_args(parser, policy_type)
    adjust = getattr(policy_type, "adjust_policy_cli_args", None)
    if adjust is not None:
        adjust(parser)
    return parser
