# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""The joint torque and fingertip contact channels the paper demo recorder stores beside its own rows."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from dexbench_arena.embodiments.fingertip_contact import (
    FINGERTIP_CONTACT_DATASET_KEY,
    FINGERTIP_CONTACT_SENSOR_NAME,
    FingertipContactReader,
    fingertip_contact_metadata,
)
from dexbench_arena.embodiments.joint_torque import (
    JOINT_TORQUE_DATASET_KEY,
    applied_joint_effort,
    joint_torque_metadata,
)

PAPER_DEMO_CHANNEL_TIMING = (
    "captured before the row's action: the value the previous env step left, the instant of the row's "
    "robot/joint_pos; the first row of an episode, read right after the reset, is not a reading of its own"
)


def _declared_term(env_cfg, name: str):
    """The embodiment's recorder term ``name`` on any of the cfgs ``add_declared_recorder_term`` writes, else None."""
    for attr in ("recorders", "demo_recorder_config", "mimic_recorder_config"):
        term = getattr(getattr(env_cfg, attr, None), name, None)
        if term is not None:
            return term
    return None


@dataclass
class PaperDemoChannel:
    """One group of per-row datasets, ``<key>/<field>``, read for env 0."""

    key: str
    metadata: dict
    read: Callable[[Any], dict[str, Any]]


def paper_demo_robot_channels(env, robot_name: str = "robot") -> list[PaperDemoChannel]:
    """The channels the env's embodiment records, each with its metadata; prints the ones it does not."""

    robot = env.scene[robot_name]
    channels = []
    if _declared_term(env.cfg, "record_post_step_joint_torque") is not None:
        metadata = {**joint_torque_metadata(robot.joint_names), "timing": PAPER_DEMO_CHANNEL_TIMING}
        channels.append(PaperDemoChannel(
            JOINT_TORQUE_DATASET_KEY, metadata, lambda e: {"applied_effort": applied_joint_effort(e.scene[robot_name])}
        ))
    else:
        print("[paper_demo] joint torque not recorded: the embodiment records none in this run")
    term = _declared_term(env.cfg, "record_post_step_fingertip_contact")
    if term is not None:
        reader = FingertipContactReader(term.body_names)
        metadata = {**fingertip_contact_metadata(reader.body_names), "timing": PAPER_DEMO_CHANNEL_TIMING}
        channels.append(PaperDemoChannel(
            FINGERTIP_CONTACT_DATASET_KEY, metadata,
            lambda e: reader.read(e.scene.sensors[FINGERTIP_CONTACT_SENSOR_NAME], e.scene[robot_name]),
        ))
    else:
        print("[paper_demo] fingertip contact not recorded: the embodiment records none in this run")
    return channels
