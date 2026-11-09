# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Tearing a polybag scene's physics down forgets its bags, so the next Run's env in the same process builds
only its own (the experiment runner closes one Run's env and builds the next in one Kit process)."""

import pytest

for _module in ("warp", "newton", "torch", "pxr", "isaaclab", "isaaclab_newton", "isaaclab_contrib", "isaaclab_arena"):
    pytest.importorskip(_module)

from isaaclab_newton.physics import NewtonManager  # noqa: E402

from dexbench_arena.assets.polybag import polybag_object  # noqa: E402
from dexbench_arena.environments.manipulation import newton_build_scope  # noqa: E402
from dexbench_arena.assets.polybag.physics import NewtonPolybagCoupledManager, NewtonPolybagVBDManager  # noqa: E402
from dexbench_arena.assets.polybag.polybag_object import (  # noqa: E402
    PolybagObject,
    PolybagRegistryEntry,
    add_registered_polybags_to_builder,
    register_polybag,
    registered_polybags,
)

_MISSING = object()


@pytest.fixture
def polybag_globals():
    """Put the process-wide registry, render bindings, builder hooks and build cleanups back as they were."""
    registry = vars(NewtonManager).get("_polybag_registry", _MISSING)
    bindings = list(polybag_object._render_bindings)
    hooks = list(NewtonManager._per_world_builder_hooks)
    cleanups = list(newton_build_scope._CLEANUPS)
    yield
    newton_build_scope._CLEANUPS[:] = cleanups
    if registry is _MISSING:
        if "_polybag_registry" in vars(NewtonManager):
            del NewtonManager._polybag_registry
    else:
        NewtonManager._polybag_registry = registry
    polybag_object._render_bindings[:] = bindings
    NewtonManager._per_world_builder_hooks = hooks


@pytest.mark.parametrize("manager", [NewtonPolybagCoupledManager, NewtonPolybagVBDManager])
def test_the_next_scenes_polybags_are_built_alone_after_close(polybag_globals, manager):
    register_polybag(PolybagRegistryEntry(cfg="previous Run", template=None))
    polybag_object._render_bindings.append(object())

    manager.close()
    next_bag = PolybagRegistryEntry(cfg="next Run", template=None)
    register_polybag(next_bag)

    assert registered_polybags() == [next_bag]
    assert polybag_object._render_bindings == []
    assert manager._per_world_builder_hooks.count(add_registered_polybags_to_builder) == 1


def test_coupled_manager_close_drops_the_previous_particle_positions(monkeypatch):
    monkeypatch.setattr(NewtonPolybagCoupledManager, "_particle_q_prev", object())

    NewtonPolybagCoupledManager.close()

    assert NewtonPolybagCoupledManager._particle_q_prev is None


def test_a_bag_collected_after_the_next_build_replicated_an_equal_bag_leaves_that_bag(polybag_globals):
    template = object()
    previous = PolybagRegistryEntry(cfg="polybag_4 in slot 4", template=template)
    register_polybag(previous)
    previous.particle_offsets.append(0)
    previous_bag = object.__new__(PolybagObject)
    previous_bag._registry_entry = previous
    previous_bag._physics_ready_handle = None

    NewtonPolybagCoupledManager.close()
    next_bag = PolybagRegistryEntry(cfg="polybag_4 in slot 4", template=template)
    register_polybag(next_bag)
    next_bag.particle_offsets.append(0)

    del previous_bag

    assert len(registered_polybags()) == 1 and registered_polybags()[0] is next_bag
    assert NewtonManager._per_world_builder_hooks.count(add_registered_polybags_to_builder) == 1
