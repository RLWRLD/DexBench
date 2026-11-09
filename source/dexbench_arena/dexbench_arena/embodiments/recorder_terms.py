# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Add an embodiment's per-step recorder terms to the env's recorder configs."""

from __future__ import annotations

import dataclasses

from isaaclab.managers import RecorderTermCfg
from isaaclab_arena.utils.configclass import combine_configclass_instances, make_configclass


def _with_declared_term(recorder_cfg, name: str, term: RecorderTermCfg):
    """``recorder_cfg`` with ``name`` as a declared field holding ``term``, so that ``copy()`` keeps it."""
    fields = {f.name for f in dataclasses.fields(recorder_cfg)}
    if name not in fields:
        extra = {key: value for key, value in vars(recorder_cfg).items() if key not in fields}
        holder = make_configclass("RecorderTermHolderCfg", [(name, RecorderTermCfg | None, dataclasses.field(default=None))])
        cls = type(recorder_cfg)
        recorder_cfg = combine_configclass_instances(cls.__name__, recorder_cfg, holder(), bases=(cls,))
        for key, value in extra.items():
            setattr(recorder_cfg, key, value)
    setattr(recorder_cfg, name, term)
    return recorder_cfg


def add_declared_recorder_term(env_cfg, name: str, term: RecorderTermCfg) -> None:
    """Add a per-step term next to the embodiment's other per-step terms."""
    replaced = {}
    for attr in ("demo_recorder_config", "mimic_recorder_config"):
        recorder_cfg = getattr(env_cfg, attr, None)
        if recorder_cfg is not None:
            if id(recorder_cfg) not in replaced:
                replaced[id(recorder_cfg)] = _with_declared_term(recorder_cfg, name, term)
            setattr(env_cfg, attr, replaced[id(recorder_cfg)])
    recorders = getattr(env_cfg, "recorders", None)
    if recorders is not None and hasattr(recorders, "record_post_step_processed_actions"):
        env_cfg.recorders = _with_declared_term(recorders, name, term)
