# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Shared fingertip contact recording option for scene and example factories."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


def add_fingertip_contact_cli_args(parser: argparse.ArgumentParser) -> None:
    """Expose the recording switch on every factory's CLI parser."""
    if "--record_fingertip_contact" not in parser._option_string_actions:
        parser.add_argument(
            "--record_fingertip_contact",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="Record each fingertip's contact force (net, normal, tangential; link frame) in demo, Mimic and "
            "--record_dataset datasets. On by default where supported (Wuji v2 on Newton, rigid objects; not with "
            "deformables); --no-record_fingertip_contact turns it off, --record_fingertip_contact makes it required.",
        )


def resolve_recording_switch(setting: bool | None, supported: bool, why_not: str, option: str) -> bool:
    """Opt-out switch: None records where supported, True requires it, False never records."""
    if setting is False:
        return False
    if supported:
        return True
    if setting:
        raise ValueError(f"--{option} {why_not}")
    print(f"[dexbench] --{option} is off: it {why_not}", flush=True)
    return False


def configure_environment_fingertip_contact(environment: IsaacLabArenaEnvironment, cfg) -> None:
    """Ask the embodiment to record its fingertip contact unless the run switched it off."""
    embodiment = environment.embodiment
    if embodiment is None:
        return
    enable = getattr(embodiment, "enable_fingertip_contact_recording", None)
    setting = getattr(cfg, "record_fingertip_contact", None)
    why_not = "needs an embodiment that declares fingertip contact bodies (Wuji v2)."
    if resolve_recording_switch(setting, enable is not None, why_not, "record_fingertip_contact"):
        enable(required=setting is True)
