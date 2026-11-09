# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Shared joint torque recording option for scene and example factories."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from dexbench_arena.environments.fingertip_contact import resolve_recording_switch

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


def add_joint_torque_cli_args(parser: argparse.ArgumentParser) -> None:
    """Expose the recording switch on every factory's CLI parser."""
    if "--record_joint_torque" not in parser._option_string_actions:
        parser.add_argument(
            "--record_joint_torque",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Record each robot joint's actuator effort (Isaac Lab's applied_effort, in the articulation's "
            "joint order) in demo, Mimic and --record_dataset datasets. On by default where the embodiment "
            "declares it (Wuji v2); --no-record_joint_torque turns it off, --record_joint_torque makes it required.",
        )


def configure_environment_joint_torque(environment: IsaacLabArenaEnvironment, cfg) -> None:
    """Enable the embodiment's joint torque recorder unless the run switched it off."""
    embodiment = environment.embodiment
    setting = getattr(cfg, "record_joint_torque", None)
    if embodiment is None and setting is not True:
        return
    enable = getattr(embodiment, "enable_joint_torque_recording", None)
    why_not = "needs an embodiment that declares joint torque recording (Wuji v2)" + (
        "; the environment has none." if embodiment is None
        else f", not {getattr(embodiment, 'name', type(embodiment).__name__)}."
    )
    if resolve_recording_switch(setting, enable is not None, why_not, "record_joint_torque"):
        enable()
