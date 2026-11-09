# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""The Newton physics runtime variant is selected on every rigid object that carries it."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from dexbench_arena.assets.physics_variants import (
    NEWTON_USD_PARSE_GUARD_ENV,
    apply_newton_usd_parse_guard,
    apply_physics_variants,
    physics_variants_enabled,
    physics_variants_of,
    select_newton_colliders,
)


@pytest.fixture
def packages(tmp_path):
    def package(name, physics, newton_colliders=None):
        usd = tmp_path / "object_library" / name / "usd"
        usd.mkdir(parents=True)
        (usd / f"{name}.usd").write_bytes(b"base")
        variants = {"physics": physics} if physics else {}
        if newton_colliders:
            variants["newton_colliders"] = newton_colliders
        (usd / f"{name}.meta.json").write_text(json.dumps({"asset": {"version": "1.3.0", "variants": variants}}))
        return str(usd / f"{name}.usd")

    return SimpleNamespace(
        bowl=package("bowl", {"Newton": "usd/runnables/physics/newton.usd", "PhysX": "usd/runnables/physics/physx.usd"}),
        can=package("can", {"Newton": "usd/runnables/physics/newton.usd"}),
        bolt=package("bolt", {"Newton": "usd/runnables/physics/newton.usd"}, {"primitives": "usd/runnables/physics/newton_primitives.usd"}),
        plain=package("plain", None),
    )


def asset(name, object_type, usd_path, variants=None):
    spawn = SimpleNamespace(usd_path=usd_path, variants=variants)
    return name, SimpleNamespace(object_type=SimpleNamespace(name=object_type), object_cfg=SimpleNamespace(spawn=spawn))


def scene_with(*assets):
    return SimpleNamespace(assets=dict(assets))


def test_the_sidecar_names_the_variants(packages):
    assert physics_variants_of(packages.bowl) == {"Newton": "usd/runnables/physics/newton.usd", "PhysX": "usd/runnables/physics/physx.usd"}
    assert physics_variants_of(packages.plain) == {}
    assert physics_variants_of("/nowhere/x.usd") == {}


def test_the_newton_variant_is_selected_on_rigid_and_articulated_objects_that_carry_it(packages):
    scene = scene_with(
        asset("bowl", "RIGID", packages.bowl),
        asset("cart", "ARTICULATION", packages.bowl, variants={"Look": "Red"}),
        asset("plain", "RIGID", packages.plain),
        asset("shelf", "BASE", packages.bowl),
    )
    assert apply_physics_variants(scene) == [("bowl", "Newton"), ("cart", "Newton")]
    assert scene.assets["bowl"].object_cfg.spawn.variants == {"Newton": "Enabled"}
    assert scene.assets["cart"].object_cfg.spawn.variants == {"Look": "Red", "Newton": "Enabled"}
    assert scene.assets["plain"].object_cfg.spawn.variants is None
    assert scene.assets["shelf"].object_cfg.spawn.variants is None


def test_a_multi_usd_spawn_takes_the_variant_only_when_every_member_carries_it(packages):
    scene = scene_with(asset("set", "RIGID", [packages.bowl, packages.can]), asset("mixed", "RIGID", [packages.bowl, packages.plain]))
    assert apply_physics_variants(scene) == [("set", "Newton")]


def test_selecting_twice_changes_nothing(packages):
    scene = scene_with(asset("bowl", "RIGID", packages.bowl))
    assert apply_physics_variants(scene) == [("bowl", "Newton")]
    assert apply_physics_variants(scene) == []


def test_a_physx_selection_on_the_spawn_is_replaced(packages):
    scene = scene_with(asset("bowl", "RIGID", packages.bowl, variants={"Look": "Red", "PhysX": "Enabled"}))
    assert apply_physics_variants(scene) == [("bowl", "Newton")]
    assert scene.assets["bowl"].object_cfg.spawn.variants == {"Look": "Red", "Newton": "Enabled"}


def test_a_collider_choice_is_selected_after_the_newton_variant_it_sits_under(packages):
    scene = scene_with(asset("bolt", "RIGID", packages.bolt))
    select_newton_colliders(scene.assets["bolt"].object_cfg.spawn, "primitives")
    assert apply_physics_variants(scene) == [("bolt", "Newton")]
    assert list(scene.assets["bolt"].object_cfg.spawn.variants.items()) == [("Newton", "Enabled"), ("NewtonColliders", "primitives")]


def test_a_collider_choice_the_package_does_not_offer_is_refused(packages):
    spawn = SimpleNamespace(usd_path=[packages.bolt, packages.can], variants=None)
    with pytest.raises(ValueError, match="can.usd"):
        select_newton_colliders(spawn, "primitives")
    assert spawn.variants is None


def test_the_run_flag_switches_the_selection_off():
    assert physics_variants_enabled(SimpleNamespace())
    assert physics_variants_enabled(SimpleNamespace(physics_variants=True))
    assert not physics_variants_enabled(SimpleNamespace(physics_variants=False))


def test_parse_guard_is_set_whatever_the_command_line_says(monkeypatch):
    monkeypatch.delenv(NEWTON_USD_PARSE_GUARD_ENV, raising=False)
    apply_newton_usd_parse_guard()
    assert os.environ[NEWTON_USD_PARSE_GUARD_ENV] == "1"
    monkeypatch.setenv(NEWTON_USD_PARSE_GUARD_ENV, "4")
    apply_newton_usd_parse_guard()
    assert os.environ[NEWTON_USD_PARSE_GUARD_ENV] == "4"
