# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Fixed-root writes must change physical roots, not only reported poses."""

from types import SimpleNamespace

import pytest

from dexbench_arena.assets import newton_root_reset as compat


@pytest.mark.parametrize("by_mask", [False, True])
@pytest.mark.parametrize("from_link", [False, True])
@pytest.mark.parametrize("skip_forward", [False, True])
def test_root_write_order_and_com_semantics(monkeypatch, by_mask, from_link, skip_forward):
    events = []
    selection = [True, False] if by_mask else [1]
    field = "env_mask" if by_mask else "env_ids"
    asset = SimpleNamespace(
        root_view=SimpleNamespace(is_fixed_base=True),
        data=SimpleNamespace(_reset_pose=lambda **kw: events.append(("invalidate", kw))),
        _resolve_env_ids=lambda value: value,
        _ALL_ENV_MASK=[True, True],
    )
    model = object()
    manager = SimpleNamespace(
        get_model=lambda: model,
        _solver=SimpleNamespace(notify_model_changed=lambda flag: events.append(("notify", flag))),
    )
    monkeypatch.setattr(compat, "_runtime", lambda: (manager, SimpleNamespace(JOINT_PROPERTIES=123)))

    def original(self, *, root_pose, skip_forward, **kw):
        assert skip_forward
        assert kw == {field: selection}
        self.data.root_link_pose_w = root_pose if from_link else "SDK-converted link pose"
        events.append(("sdk", root_pose))
        return "original return"

    def copy(self, current_model, selected, *, by_mask):
        assert current_model is model
        assert selected is selection
        assert self.data.root_link_pose_w == ("input" if from_link else "SDK-converted link pose")
        events.append(("copy", by_mask))

    monkeypatch.setattr(compat, "_copy_fixed_root_to_model", copy)
    wrapped = compat._wrap_root_write(original, by_mask=by_mask, from_link=from_link)
    assert wrapped(asset, root_pose="input", skip_forward=skip_forward, **{field: selection}) == "original return"
    assert [e[0] for e in events] == (["sdk", "copy", "notify"] + ([] if skip_forward else ["invalidate"]))
    if not skip_forward:
        assert events[-1][1] == {"from_link": from_link, field: selection}


def test_nonfixed_roots_pass_through_and_installation_is_idempotent(monkeypatch):
    def original(self, **kwargs):
        return kwargs

    class Asset:
        root_view = SimpleNamespace(is_fixed_base=False)

    for frame in ("link", "com"):
        for selection in ("index", "mask"):
            setattr(Asset, f"write_root_{frame}_pose_to_sim_{selection}", original)
    monkeypatch.setattr(compat, "_runtime", lambda: pytest.fail("nonfixed roots must not access the model"))
    assert compat._install_on(Asset)
    first = Asset.write_root_link_pose_to_sim_index
    assert not compat._install_on(Asset)
    assert first is Asset.write_root_link_pose_to_sim_index
    assert first.__wrapped__ is original
    assert Asset().write_root_com_pose_to_sim_mask(root_pose="unchanged", skip_forward=True) == {
        "root_pose": "unchanged", "skip_forward": True
    }


@pytest.fixture
def newton_scene():
    """A real CPU Newton model; no renderer or scene assets are needed."""
    wp = pytest.importorskip("warp")
    newton = pytest.importorskip("newton")
    pytest.importorskip("torch")
    from newton.selection import ArticulationView

    wp.init()

    def build():
        world = newton.ModelBuilder()
        base = world.add_link(label="robot/base")
        arm = world.add_link(label="robot/arm")
        spacer = world.add_link(label="robot/spacer")
        hand = world.add_link(label="robot/hand")
        joints = [
            world.add_joint_fixed(-1, base),
            world.add_joint_revolute(base, arm, axis=wp.vec3(0, 0, 1)),
            world.add_joint_fixed(arm, spacer),
            world.add_joint_revolute(spacer, hand, axis=wp.vec3(1, 0, 0)),
        ]
        world.add_articulation(joints, label="robot")
        for name in ("source", "destination"):
            body = world.add_link(label=f"{name}/body")
            world.add_articulation([world.add_joint_free(body)], label=name)
        builder = newton.ModelBuilder()
        for i in range(3):
            builder.add_world(world, label_prefix=f"env{i}")
        model = builder.finalize(device="cpu")
        state = model.state()
        robot = ArticulationView(model, "*/robot", exclude_joint_types=[newton.JointType.FREE, newton.JointType.FIXED])
        objects = [ArticulationView(model, f"*/{name}") for name in ("source", "destination")]
        cached = wp.clone(robot.get_root_transforms(state)[:, 0])
        asset = SimpleNamespace(root_view=robot, data=SimpleNamespace(root_link_pose_w=SimpleNamespace(warp=cached)))
        return model, state, asset, objects

    return wp, newton, build


@pytest.mark.parametrize("by_mask", [False, True])
def test_selected_root_write_updates_real_bodies_and_preserves_other_variations(newton_scene, by_mask):
    import numpy as np
    import torch

    wp, newton, build = newton_scene
    model, state, asset, objects = build()
    nominal = state.body_q.numpy().copy()
    root = asset.data.root_link_pose_w.warp
    intended = wp.to_torch(root)
    intended[1, :3] = torch.tensor([-0.02, 0.01, 0.0])
    intended[1, 3:] = torch.tensor([0.0, 0.0, np.sin(0.1), np.cos(0.1)])
    for index, obj in enumerate(objects):
        value = wp.to_torch(obj.get_root_transforms(state))
        value[:, 0, :3] = torch.tensor([0.02, 0.01, 0.5]) * (index + 1)
    selection = wp.array([False, True, False], dtype=wp.bool, device="cpu") if by_mask else [1]
    compat._copy_fixed_root_to_model(asset, model, selection, by_mask=by_mask)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    actual = asset.root_view.get_link_transforms(state).numpy()[:, 0, 0]
    np.testing.assert_allclose(actual[1, :3], [-0.02, 0.01, 0.0], atol=1e-7)
    np.testing.assert_allclose(actual[1, 3:], [0.0, 0.0, np.sin(0.1), np.cos(0.1)], atol=1e-7)
    np.testing.assert_allclose(actual[[0, 2], :3], 0.0, atol=1e-7)
    for index, obj in enumerate(objects):
        np.testing.assert_allclose(
            obj.get_link_transforms(state).numpy()[:, 0, 0, :3],
            np.tile(np.array([0.02, 0.01, 0.5]) * (index + 1), (3, 1)), atol=1e-7,
        )
    assert not np.array_equal(state.body_q.numpy(), nominal)

    intended[1, :3] = 99.0
    intended[2, :3] = torch.tensor([0.02, -0.01, 0.0])
    selection = wp.array([False, False, True], dtype=wp.bool, device="cpu") if by_mask else [2]
    compat._copy_fixed_root_to_model(asset, model, selection, by_mask=by_mask)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    actual = asset.root_view.get_link_transforms(state).numpy()[:, 0, 0]
    np.testing.assert_allclose(actual[1, :3], [-0.02, 0.01, 0.0], atol=1e-7)
    np.testing.assert_allclose(actual[2, :3], [0.02, -0.01, 0.0], atol=1e-7)


def test_fixed_root_write_targets_the_given_model_after_a_rebuild(newton_scene):
    import numpy as np

    wp, newton, build = newton_scene
    old_model, _, asset, _ = build()
    new_model, new_state, replacement, _ = build()
    asset.data = replacement.data
    wp.to_torch(asset.data.root_link_pose_w.warp)[:, :3] = 0.025
    compat._copy_fixed_root_to_model(asset, new_model, None, by_mask=False)
    newton.eval_fk(new_model, new_state.joint_q, new_state.joint_qd, new_state)
    np.testing.assert_allclose(asset.root_view.get_link_transforms(new_state).numpy()[:, 0, 0, :3], 0.025, atol=1e-7)
    np.testing.assert_allclose(old_model.joint_X_p.numpy()[:, :3], 0.0, atol=1e-7)
