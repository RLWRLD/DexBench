# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Configuration precedence and isolation for task-selected hand materials."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from dexbench_arena.contact_materials import (
    HandContactMaterialCfg,
    WUJI_GRASP_CONTACT_MATERIAL,
    resolve_hand_contact_material,
)
from dexbench_arena.embodiments.hand_contact import HandContactMaterialMixin


@pytest.mark.parametrize("static,dynamic", [(-1, 0), (1, -1), (float("nan"), 0), (1, float("inf")), (0.2, 0.3)])
def test_invalid_coefficients_are_rejected(static, dynamic):
    with pytest.raises(ValueError):
        HandContactMaterialCfg(static, dynamic)


def test_profile_is_immutable_and_zero_is_distinct_from_disabled():
    zero = HandContactMaterialCfg(0.0, 0.0)
    assert zero.enabled
    with pytest.raises(FrozenInstanceError):
        zero.enabled = False
    disabled, source = resolve_hand_contact_material(zero, WUJI_GRASP_CONTACT_MATERIAL, enabled=False)
    assert not disabled.enabled
    assert zero.enabled
    assert source == "override"


def test_field_overrides_preserve_the_task_profile_and_embodiment_fallback():
    task = HandContactMaterialCfg(0.8, 0.6, "average")
    actual, source = resolve_hand_contact_material(task, WUJI_GRASP_CONTACT_MATERIAL, dynamic_friction=0.4)
    assert actual == HandContactMaterialCfg(0.8, 0.4, "average")
    assert source == "override"
    assert resolve_hand_contact_material(task, WUJI_GRASP_CONTACT_MATERIAL) == (task, "task")
    assert resolve_hand_contact_material(None, WUJI_GRASP_CONTACT_MATERIAL) == (
        WUJI_GRASP_CONTACT_MATERIAL,
        "embodiment",
    )
    assert task.dynamic_friction == 0.6
    assert WUJI_GRASP_CONTACT_MATERIAL == HandContactMaterialCfg(6.0, 5.0)


def test_task_profile_is_ignored_by_a_robot_without_hand_materials():
    from dexbench_arena.environments.hand_contact import configure_hand_contact_material

    args = SimpleNamespace()
    task = SimpleNamespace(hand_contact_material=WUJI_GRASP_CONTACT_MATERIAL)
    for embodiment in (None, SimpleNamespace()):
        assert configure_hand_contact_material(args, task, embodiment) is None
    with pytest.raises(ValueError, match="supported embodiment"):
        configure_hand_contact_material(SimpleNamespace(hand_static_friction=0.5), task, SimpleNamespace())


class Hand(HandContactMaterialMixin):
    model_hand_contact_material = HandContactMaterialCfg(1.0, 1.0)

    def __init__(self):
        self.event_config = SimpleNamespace(
            hand_contact_material_check=SimpleNamespace(params={"material_name": "hand_contact"})
        )
        self.scene_config = SimpleNamespace(robot=SimpleNamespace(spawn=SimpleNamespace()))


def test_configuring_one_hand_does_not_mutate_another_hand_or_the_preset():
    first, second = Hand(), Hand()
    first.configure_hand_contact_material(HandContactMaterialCfg(0.5, 0.4), source="task")
    assert first.event_config.hand_contact_material_check.params["static_friction"] == 0.5
    assert (first.scene_config.robot.spawn.hand_static_friction, first.scene_config.robot.spawn.hand_dynamic_friction) == (0.5, 0.4)
    assert second.event_config.hand_contact_material_check.params == {"material_name": "hand_contact"}
    assert vars(second.scene_config.robot.spawn) == {}
    assert second.hand_contact_material == WUJI_GRASP_CONTACT_MATERIAL


def test_recording_preserves_upstream_metadata_without_mutating_config_or_results():
    from dexbench_arena.embodiments.hand_contact import (
        install_hand_contact_recording_metadata,
        record_hand_contact_material,
    )

    upstream = {"existing": {"value": 7}, "sim_args": {"future_field": 42}}
    material = {"static_friction": 0.5, "overrides": {"dynamic_friction": 0.4}}
    env = SimpleNamespace(cfg=SimpleNamespace(dexbench_hand_contact_material=material))
    calls = []

    def upstream_get_ep_meta():
        calls.append(True)
        return upstream

    env.recorder_manager = SimpleNamespace(_env=env, get_ep_meta=upstream_get_ep_meta)
    install_hand_contact_recording_metadata(env)
    installed = env.recorder_manager.get_ep_meta
    install_hand_contact_recording_metadata(env)
    assert env.recorder_manager.get_ep_meta is installed
    metadata = json.loads(json.dumps(installed()))
    assert metadata["existing"] == {"value": 7}
    assert metadata["sim_args"] == {"future_field": 42}
    assert metadata["hand_contact_material"] == material
    assert not hasattr(env.cfg, "get_ep_meta")
    metadata["hand_contact_material"]["overrides"]["dynamic_friction"] = 99
    metadata["existing"]["value"] = 99
    upstream["sim_args"]["later_field"] = 43
    assert installed()["sim_args"]["later_field"] == 43
    assert len(calls) == 2
    assert upstream["existing"]["value"] == 7
    recorded = record_hand_contact_material(env, 0)
    recorded["hand_contact_material"]["overrides"]["dynamic_friction"] = 88
    assert material["overrides"]["dynamic_friction"] == 0.4


@pytest.mark.parametrize("task,source", [(WUJI_GRASP_CONTACT_MATERIAL, "task"), (None, "embodiment")])
def test_partial_override_error_identifies_inherited_values(task, source):
    with pytest.raises(ValueError) as error:
        resolve_hand_contact_material(task, WUJI_GRASP_CONTACT_MATERIAL, static_friction=0.5)
    assert "dynamic_friction must not exceed static_friction" in str(error.value)
    assert "static_friction=0.5 (override)" in str(error.value)
    assert f"dynamic_friction=5.0 ({source})" in str(error.value)


def test_explicit_override_provenance_is_copied_and_preserves_false():
    hand = Hand()
    overrides = {"enabled": False}
    hand.configure_hand_contact_material(HandContactMaterialCfg(6, 5, enabled=False), source="override", overrides=overrides)
    overrides["enabled"] = True
    assert hand.event_config.hand_contact_material_check.params["overrides"] == {"enabled": False}


@pytest.mark.parametrize("name", ["DEXBENCH_HAND_STATIC_FRICTION", "DEXBENCH_HAND_DYNAMIC_FRICTION"])
@pytest.mark.parametrize("value", ["", "0", "6"])
def test_legacy_environment_rejects_even_empty_values(monkeypatch, name, value):
    from dexbench_arena.contact_materials import validate_legacy_hand_friction_environment

    monkeypatch.delenv("DEXBENCH_HAND_STATIC_FRICTION", raising=False)
    monkeypatch.delenv("DEXBENCH_HAND_DYNAMIC_FRICTION", raising=False)
    validate_legacy_hand_friction_environment()
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        validate_legacy_hand_friction_environment()
