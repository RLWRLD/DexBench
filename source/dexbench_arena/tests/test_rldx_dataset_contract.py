# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Conversion metadata must match the versioned six-group inference layout."""

import json
import pytest

from dexbench_arena_rldx.policy.dataset_contract import DATASET_JOINT_GROUPS, feature_names, load_dataset_joint_groups
from dexbench_arena.embodiments.rby1_wuji2.joint_groups import WUJI2_RECORDED_JOINT_GROUPS


def test_feature_names_preserve_declared_order():
    info = {"features": {"action": {"names": ["b", "a"], "shape": [2]}}}
    assert feature_names(info, "action") == ["b", "a"]


@pytest.mark.parametrize("names", [["a", "a"], ["a"], None], ids=["duplicate", "wrong-width", "unnamed"])
def test_feature_names_reject_ambiguous_columns(names):
    info = {"features": {"action": {"names": names, "shape": [2]}}}
    with pytest.raises(ValueError, match="unique joint names matching the feature shape"):
        feature_names(info, "action")


@pytest.fixture
def metadata(tmp_path):
    names = [f"wheel_{i}" for i in range(4)] + [joint for group in WUJI2_RECORDED_JOINT_GROUPS.values() for joint in group]
    info = {"features": {key: {"names": names.copy(), "shape": [66]}
                         for key in ("observation.joint_position", "action")}}
    groups, start = {}, 4
    for name, width in zip(DATASET_JOINT_GROUPS, (6, 7, 20, 7, 20, 2)):
        groups[name] = {"start": start, "end": start + width}
        start += width
    modality = {key: groups.copy() for key in ("state", "action")}
    (tmp_path / "info.json").write_text(json.dumps(info))
    (tmp_path / "modality.json").write_text(json.dumps(modality))
    return tmp_path, info, modality


def test_preserves_declared_group_and_column_order(metadata):
    root, _, _ = metadata
    groups = load_dataset_joint_groups(root)
    assert tuple(groups) == DATASET_JOINT_GROUPS
    assert groups == WUJI2_RECORDED_JOINT_GROUPS


@pytest.mark.parametrize("fault", ["duplicate", "permutation", "slice", "overlap", "matching_permutation"])
def test_refuses_ambiguous_or_mismatched_columns(metadata, fault):
    root, info, modality = metadata
    if fault == "duplicate":
        info["features"]["action"]["names"][5] = "torso_0"
    elif fault == "permutation":
        info["features"]["action"]["names"][4:6] = ["torso_1", "torso_0"]
    elif fault == "slice":
        modality["action"]["neck_joints"] = {"start": 64, "end": 67}
    elif fault == "matching_permutation":
        for feature in info["features"].values():
            feature["names"][17:19] = list(reversed(feature["names"][17:19]))
    else:
        for kind in ("state", "action"):
            modality[kind]["neck_joints"] = {"start": 4, "end": 6}
    (root / "info.json").write_text(json.dumps(info))
    (root / "modality.json").write_text(json.dumps(modality))
    with pytest.raises(ValueError, match="joint|slice"):
        load_dataset_joint_groups(root)
