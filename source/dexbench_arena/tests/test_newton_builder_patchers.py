# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""The Newton ModelBuilder patchers change only the shapes or bodies they pick by label, and refuse a pattern that picks none."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

wp = pytest.importorskip("warp")

from dexbench_arena.embodiments.lookup import embodiment_class  # noqa: E402
from dexbench_arena.environments.manipulation.e3_manufacturing_db_m5_environment import (  # noqa: E402
    KIT_BOX_SHAPE_PATTERN,
    part_rolling_friction_pattern,
)
from dexbench_arena.environments.manipulation.newton_isotropic_inertia import make_inertia_isotropic  # noqa: E402
from dexbench_arena.environments.manipulation.newton_rolling_friction import (  # noqa: E402
    COLLIDE_SHAPES,
    CONDIM_ATTRIBUTE,
    set_rolling_friction,
)
from dexbench_arena.environments.manipulation.newton_shape_contact import (  # noqa: E402
    SOLMIX_ATTRIBUTE,
    ShapeContactRequest,
    set_shape_contact,
)

BOLT_AXIAL = 5.19e-5
BOLT_TRANSVERSE = 6.06e-4


def _mat(inertia) -> wp.mat33:
    return wp.mat33(*np.asarray(inertia, dtype=np.float64).flatten().tolist())


def _np(mat) -> np.ndarray:
    return np.array(mat, dtype=np.float64).reshape(3, 3)


class _Builder:
    def __init__(self, bodies):
        self.body_label = [label for label, _, _ in bodies]
        self.body_mass = [mass for _, mass, _ in bodies]
        self.body_inertia = [_mat(inertia) for _, _, inertia in bodies]
        self.body_inv_inertia = [
            _mat(np.linalg.inv(inertia)) if mass > 0 else _mat(np.zeros((3, 3))) for _, mass, inertia in bodies
        ]


BOLT = np.diag([BOLT_AXIAL, BOLT_TRANSVERSE, BOLT_TRANSVERSE])
TRIAXIAL = np.diag([1.0e-4, 3.0e-4, 6.0e-4])
TURNED = np.array([[3.5e-4, 2.5e-4, 0.0], [2.5e-4, 3.5e-4, 0.0], [0.0, 0.0, 3.0e-4]])


def test_sets_the_largest_principal_moment_on_every_axis_of_the_named_bodies_only():
    builder = _Builder(
        [
            ("/World/envs/env_0/bin_part_bolt_m24_00", 0.45, TURNED),
            ("/World/envs/env_0/kit_box_bolt_m24_02", 0.45, BOLT),
            ("/World/envs/env_0/belt_kit_box", 1.2, TRIAXIAL),
            ("/World/envs/env_0/static_bolt_m24", 0.0, np.zeros((3, 3))),
        ]
    )
    results = make_inertia_isotropic(builder, ("bolt_m24",))
    assert [r.index for r in results] == [0, 1]
    for index, largest in ((0, 6.0e-4), (1, BOLT_TRANSVERSE)):
        np.testing.assert_allclose(_np(builder.body_inertia[index]), np.eye(3) * largest, rtol=1e-6)
        np.testing.assert_allclose(_np(builder.body_inv_inertia[index]), np.eye(3) / largest, rtol=1e-6)
    np.testing.assert_allclose(_np(builder.body_inertia[2]), TRIAXIAL, rtol=1e-6)
    np.testing.assert_allclose(_np(builder.body_inertia[3]), np.zeros((3, 3)))
    with pytest.raises(RuntimeError, match="found no body"):
        make_inertia_isotropic(builder, ("elbow_50a",))


VISIBLE = 1 << 0
ENV = "/World/envs/env_0"
BIN = f"{ENV}/assembly_line/FixtureSlots/Slot_PartsBin/Asset/Geom/parts_bin_fc"


def _builder():
    shapes = [
        (f"{ENV}/bin_part_bolt_m24_00/Part_BoltM24/prim5_collision/cylinder_000", COLLIDE_SHAPES),
        (f"{ENV}/bin_part_bolt_m24_07/Part_BoltM24/prim5_collision/hexprism_001", COLLIDE_SHAPES),
        (f"{ENV}/kit_box_bolt_m24_00/Part_BoltM24/prim5_collision/cylinder_000", COLLIDE_SHAPES),
        (f"{ENV}/bin_part_bolt_m24_00/Part_BoltM24/PartFC_Bolt_Mesh_visonly", VISIBLE),
        (f"{BIN}/BinFC_Floor/BinFC_Floor_box", COLLIDE_SHAPES),
        (f"{ENV}/kit_box/Geom/KitBox_Floor", COLLIDE_SHAPES),
        (f"{ENV}/Robot/left_thumb_distal/collision", COLLIDE_SHAPES | VISIBLE),
        (f"{ENV}/bin_part_nema17_00/Part_Nema17/prim5_collision/box_001", COLLIDE_SHAPES),
    ]
    return SimpleNamespace(
        shape_label=[label for label, _ in shapes],
        shape_flags=[flags for _, flags in shapes],
        shape_material_mu_rolling=[1.0e-4] * len(shapes),
        custom_attributes={CONDIM_ATTRIBUTE: SimpleNamespace(values=None)},
    )


def test_only_the_parts_and_the_bin_get_rolling_friction_and_condim_6():
    builder = _builder()
    set_rolling_friction(builder, part_rolling_friction_pattern("bolt_m24"), 2.0e-4, "parts")
    assert builder.shape_material_mu_rolling == [2.0e-4, 2.0e-4, 2.0e-4, 1.0e-4, 2.0e-4, 1.0e-4, 1.0e-4, 1.0e-4]
    assert builder.custom_attributes[CONDIM_ATTRIBUTE].values == {0: 6, 1: 6, 2: 6, 4: 6}
    with pytest.raises(RuntimeError, match="found no colliding shape"):
        set_rolling_friction(builder, r"/bin_part_elbow_50a_\d+/", 2.0e-4, "parts")


ROBOT = f"{ENV}/Robot"
KIT_BOX = f"{ENV}/belt_kit_box"
WUJI2_HAND_BODY_PATTERN = embodiment_class("dexbench_rby1_wuji2").hand_body_label_pattern


def test_the_kit_box_and_the_hands_get_their_own_gains_and_solmix_and_nothing_else_does():
    shapes = [
        (f"{KIT_BOX}/Geom/kit_box_fc/KbFC_Floor/KbFC_Floor", KIT_BOX, True),
        (f"{KIT_BOX}/Geom/kit_box_fc/KbFC_Divider_X/KbFC_Divider_X", KIT_BOX, True),
        (f"{ENV}/kit_box_nema17_00/Newton/collision_0", f"{ENV}/kit_box_nema17_00", True),
        (f"{ROBOT}/left_wrist/collisions/geom_0", f"{ROBOT}/left_wrist", True),
        (f"{ROBOT}/left_wrist/visuals/geom_0", f"{ROBOT}/left_wrist", False),
        (f"{ROBOT}/right_index_finger_distal/collisions/geom_0", f"{ROBOT}/right_index_finger_distal", True),
        (f"{ROBOT}/link_right_arm_6/collisions/geom_0", f"{ROBOT}/link_right_arm_6", True),
        (f"{ENV}/assembly_line/FixtureSlots/Slot_PartsBin/Asset/PartsBin/BinFC_Floor", None, True),
    ]
    bodies = sorted({body for _, body, _ in shapes if body})
    builder = SimpleNamespace(
        shape_label=[label for label, _, _ in shapes],
        shape_body=[bodies.index(body) if body else -1 for _, body, _ in shapes],
        body_label=bodies,
        shape_flags=[COLLIDE_SHAPES if collides else 0 for _, _, collides in shapes],
        shape_material_ke=[625.0] * len(shapes),
        shape_material_kd=[100.0] * len(shapes),
        custom_attributes={SOLMIX_ATTRIBUTE: SimpleNamespace(values=None)},
    )
    set_shape_contact(builder, ShapeContactRequest("kit box", KIT_BOX_SHAPE_PATTERN, 10000.0, 200.0, 100.0))
    set_shape_contact(builder, ShapeContactRequest("hand", WUJI2_HAND_BODY_PATTERN, 2500.0, 50.0, 1.0, match="body"))

    assert builder.shape_material_ke == [10000.0, 10000.0, 625.0, 2500.0, 625.0, 2500.0, 625.0, 625.0]
    assert builder.shape_material_kd == [200.0, 200.0, 100.0, 50.0, 100.0, 50.0, 100.0, 100.0]
    assert builder.custom_attributes[SOLMIX_ATTRIBUTE].values == {0: 100.0, 1: 100.0, 3: 1.0, 5: 1.0}
    with pytest.raises(RuntimeError, match="found no colliding shape"):
        set_shape_contact(builder, ShapeContactRequest("hand", WUJI2_HAND_BODY_PATTERN, 2500.0, 50.0, 1.0))


def test_a_request_lasts_until_the_next_build_begins(monkeypatch):
    """Each patcher's requests belong to the build that installed them: the next environment's build starts without
    them, so a task composed after another in the same process (DB-R2 after DB-M5, DB-M5 after DB-M2) does not run
    with the first task's inertia, rolling friction or contact (newton_build_scope)."""
    nm = pytest.importorskip("isaaclab_newton.physics.newton_manager")
    from dexbench_arena.environments.manipulation import newton_build_scope as scope
    from dexbench_arena.environments.manipulation import newton_isotropic_inertia as inertia
    from dexbench_arena.environments.manipulation import newton_rolling_friction as rolling
    from dexbench_arena.environments.manipulation import newton_shape_contact as contact

    monkeypatch.setattr(nm.NewtonManager, "register_callback",
                        staticmethod(lambda *args, **kwargs: SimpleNamespace(deregister=lambda: None)))
    for module in (inertia, rolling, contact):
        monkeypatch.setattr(module, "_REQUESTS", {})
        monkeypatch.setattr(module, "_HANDLE", None)
    monkeypatch.setattr(scope, "_CLEANUPS", [])

    def install(name):
        inertia.install_isotropic_inertia(name, ("bolt_m24",))
        rolling.install_rolling_friction(name, "/bolt_m24/", 1e-4)
        contact.install_shape_contact(ShapeContactRequest(name, "/bolt_m24/", 1.0, 1.0, 1.0))

    scope.begin_newton_build()
    install("first_task")
    assert all(set(module._REQUESTS) == {"first_task"} for module in (inertia, rolling, contact))

    scope.begin_newton_build()
    install("second_task")
    assert all(set(module._REQUESTS) == {"second_task"} for module in (inertia, rolling, contact))

    scope.begin_newton_build()
    assert all(module._REQUESTS == {} for module in (inertia, rolling, contact))
