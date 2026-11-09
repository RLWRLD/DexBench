# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Shared hand material configuration for scene and example factories."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from dexbench_arena.contact_materials import (
    HandContactMaterialCfg,
    resolve_hand_contact_material,
    validate_legacy_hand_friction_environment,
)

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


def add_hand_contact_material_cli_args(parser: argparse.ArgumentParser) -> None:
    """Expose the same run overrides on every factory's CLI parser."""
    validate_legacy_hand_friction_environment()
    options = {
        "--hand_static_friction": {
            "type": float,
            "help": "Override the selected task's static hand contact coefficient (zero is allowed).",
        },
        "--hand_dynamic_friction": {
            "type": float,
            "help": "Override the selected task's dynamic hand contact coefficient (zero is allowed).",
        },
        "--hand_friction_combine_mode": {
            "choices": ("average", "min", "multiply", "max"),
            "help": "Override the hand material's PhysX friction combine mode.",
        },
        "--disable_hand_contact_material": {
            "dest": "hand_contact_material_enabled",
            "action": "store_false",
            "help": "Put the robot model's own contact material on the hand (Wuji v2: its MJCF's 1.0 / 1.0, max)"
            " instead of the task's grasp material.",
        },
        "--enable_hand_contact_material": {
            "dest": "hand_contact_material_enabled",
            "action": "store_true",
            "help": "Enable the selected task's hand contact material.",
        },
    }
    for option, kwargs in options.items():
        if option not in parser._option_string_actions:
            parser.add_argument(option, default=None, **kwargs)


def configure_hand_contact_material(cfg, task, embodiment) -> HandContactMaterialCfg | None:
    """Resolve the task profile and validate overrides before physics starts."""
    validate_legacy_hand_friction_environment()
    overrides = {
        "static_friction": getattr(cfg, "hand_static_friction", None),
        "dynamic_friction": getattr(cfg, "hand_dynamic_friction", None),
        "friction_combine_mode": getattr(cfg, "hand_friction_combine_mode", None),
        "enabled": getattr(cfg, "hand_contact_material_enabled", None),
    }
    task_material = getattr(task, "hand_contact_material", None)
    configure = getattr(embodiment, "configure_hand_contact_material", None)
    if configure is None:
        if any(value is not None for value in overrides.values()):
            raise ValueError("Hand contact material configuration requires a supported embodiment (Wuji v2).")
        return None
    material, source = resolve_hand_contact_material(
        task_material, embodiment.default_hand_contact_material, **overrides
    )
    configure(material, source=source, overrides={name: value for name, value in overrides.items() if value is not None})
    return material


def configure_environment_hand_contact_material(environment: IsaacLabArenaEnvironment, cfg) -> None:
    """Configure the hand event and evaluation recorder from the environment's config."""
    material = configure_hand_contact_material(cfg, environment.task, environment.embodiment)
    if material is None:
        return

    from isaaclab_arena.recording.episode_recorder_manager import EpisodeRecorderTermCfg

    from dexbench_arena.embodiments.hand_contact import record_hand_contact_material

    environment.episode_recorder_terms["hand_contact_material"] = EpisodeRecorderTermCfg(
        func=record_hand_contact_material,
    )
