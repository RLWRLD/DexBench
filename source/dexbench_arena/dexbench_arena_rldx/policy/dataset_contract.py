# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Joint groups shared with the native RLDX Wuji2 training configuration."""

import json
from pathlib import Path

from dexbench_arena.embodiments.rby1_wuji2.joint_groups import WUJI2_RECORDED_JOINT_GROUPS

DATASET_JOINT_GROUPS = tuple(WUJI2_RECORDED_JOINT_GROUPS)


def feature_names(info: dict, key: str) -> list[str]:
    """Read an ordered joint feature without guessing unnamed or ambiguous columns."""
    feature = info["features"][key]
    names = feature.get("names")
    if (not isinstance(names, list) or not names
            or not all(isinstance(name, str) and name for name in names)
            or len(set(names)) != len(names) or feature["shape"] != [len(names)]):
        raise ValueError(f"{key}: unique joint names matching the feature shape are required")
    return names


def load_dataset_joint_groups(metadata_dir: str | Path) -> dict[str, tuple[str, ...]]:
    """Validate an exported dataset against the versioned ``wuji2_rrc_v1`` layout."""
    root = Path(metadata_dir)
    info = json.loads((root / "info.json").read_text())
    modality = json.loads((root / "modality.json").read_text())
    groups = {}
    for name in DATASET_JOINT_GROUPS:
        resolved = []
        for kind, default_key in (("state", "observation.joint_position"), ("action", "action")):
            entry = modality[kind][name]
            key = entry.get("original_key", default_key)
            names = feature_names(info, key)
            start, end = entry["start"], entry["end"]
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(names):
                raise ValueError(f"{kind}.{name}: invalid joint slice [{start}, {end})")
            resolved.append(tuple(names[start:end]))
        if resolved[0] != resolved[1]:
            raise ValueError(f"{name}: state/action joint order differs")
        if resolved[0] != WUJI2_RECORDED_JOINT_GROUPS[name]:
            raise ValueError(f"{name}: joint order differs from wuji2_rrc_v1")
        groups[name] = resolved[0]
    flat = [joint for names in groups.values() for joint in names]
    if len(flat) != sum(map(len, WUJI2_RECORDED_JOINT_GROUPS.values())) or len(set(flat)) != len(flat):
        raise ValueError("Dataset groups must contain every controlled joint exactly once")
    return groups
