# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""DB-R1's placement rules (any fruit of the kind, others on the tray) and the per-fruit basket dwell (Kit-free)."""

import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from dexbench_arena.tasks.predicates.hold import sustained  # noqa: E402
from dexbench_arena.tasks.r1_clutter import PILE_TRAY_BOUNDS  # noqa: E402
from dexbench_arena.tasks.r1_rules import evaluate  # noqa: E402

RULES = {"source_bounds": PILE_TRAY_BOUNDS, "basket_bounds": ((2.4225, 0.1325, 0.603), (2.7775, 0.3675, 0.830)),
         "workspace_xy_bounds": ((2.0, -0.2), (3.7, 1.6)), "dropped_min_z_m": 0.45, "basket_margin_m": 0.004,
         "released_max_force_n": 1.0, "released_max_weight_fraction": 0.25, "still_gates": (0.1, 6.0)}
MASSES = torch.full((3,), .0483)


def test_any_matching_fruit_succeeds_but_drops_foreign_kinds_and_two_picks_fail():
    centers = torch.tensor([[[3.15,.7,1.08], [3.2,.65,1.08], [2.6,.25,.64]]])
    half = torch.full_like(centers, .018)
    velocity = torch.zeros_like(centers)
    forces = torch.zeros(1,3)
    eligible = torch.ones(3,dtype=torch.bool)
    def run():
        return evaluate(centers, half, velocity, velocity, forces, eligible, MASSES, **RULES)
    assert run()['candidates'].tolist() == [[False,False,True]]
    centers[0,0] = torch.tensor([3.5,.82,1.08])
    assert run()['candidates'][0,2]
    eligible[2] = False
    assert run()['wrong_kind'][0] and not run()['candidates'].any()
    eligible[2] = True
    forces[0,2] = 2.
    assert not run()['candidates'].any()
    forces.zero_()
    centers[0,0,2] = .3
    assert run()['failed'][0] and not run()['candidates'].any()
    centers[0,0] = torch.tensor([2.65,.25,.64])
    assert run()['multiple'][0] and run()['failed'][0]


def test_a_hand_still_bearing_a_light_fruit_has_not_let_go():
    centers = torch.tensor([[[3.15,.7,1.08], [3.2,.65,1.08], [2.6,.25,.64]]])
    half = torch.full_like(centers, .018)
    still = torch.zeros_like(centers)
    eligible = torch.ones(3, dtype=torch.bool)
    weight = .0483 * 9.81
    for force, released in ((weight, False), (.26 * weight, False), (.24 * weight, True)):
        masks = evaluate(centers, half, still, still, torch.tensor([[0., 0., force]]), eligible, MASSES, **RULES)
        assert masks['released'][0, 2].item() is released and masks['candidates'][0, 2].item() is released
    heavy = torch.full((3,), 1.)
    assert not evaluate(centers, half, still, still, torch.tensor([[0., 0., 1.01]]), eligible, heavy, **RULES)['released'][0, 2]


def test_a_fruit_let_go_over_the_basket_is_not_lost_while_it_falls_into_it():
    centers = torch.tensor([[[3.15,.7,1.08], [3.2,.65,1.08], [2.6,.25,.93]]])
    half = torch.full_like(centers, .018)
    velocity = torch.zeros_like(centers)
    velocity[0,2,2] = -1.4
    forces = torch.zeros(1,3)
    eligible = torch.ones(3,dtype=torch.bool)
    masks = evaluate(centers, half, velocity, torch.zeros_like(centers), forces, eligible, MASSES, **RULES)
    assert not masks['dropped'][0] and not masks['failed'][0]
    centers[0,2] = torch.tensor([2.2,.25,.04]); velocity.zero_()
    assert evaluate(centers, half, velocity, velocity, forces, eligible, MASSES, **RULES)['failed'][0]


def test_a_tumbled_fruit_at_rest_on_the_basket_floor_is_placed_not_lost():
    r1_pick = pytest.importorskip("dexbench_arena.tasks.r1_pick", reason="needs Isaac Lab and Warp")
    from isaaclab.utils.math import quat_from_matrix

    h = torch.tensor([.033, .035, .032])
    c, s = math.cos(math.pi / 4), math.sin(math.pi / 4)
    r = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1.]]) @ torch.tensor([[1., 0, 0], [0, c, -s], [0, s, c]])
    y = .3675 - .01 - .035
    assert y + (r.abs() @ h)[1] > .3675 + .004 > y + (r.square() @ h.square()).sqrt()[1]
    objects = [{"name": f"r1_fruit_{i:03d}", "fruit_id": "apple_pile", "center_local": (0., 0., 0.),
                "half_extents": tuple(h.tolist()), "mass_kg": .08} for i in range(3)]
    tumbled = tuple(quat_from_matrix(r).tolist())
    poses = {"r1_fruit_000": ((3.15, .7, 1.08), (0., 0., 0., 1.)), "r1_fruit_001": ((3.2, .65, 1.08), (0., 0., 0., 1.)),
             "r1_fruit_002": ((2.6, y, .64), tumbled)}
    rules = dict(RULES)
    placed = r1_pick.state(_pile_env(poses), objects, "apple_pile", rules)
    assert not placed["failed"][0] and placed["candidates"].tolist() == [[False, False, True]]
    poses["r1_fruit_002"] = ((2.6, y, .830 + .03), tumbled)
    assert r1_pick.state(_pile_env(poses), objects, "apple_pile", rules)["failed"][0]


def test_basket_dwell_cannot_transfer_between_fruit_and_resets_per_env():
    env = SimpleNamespace(common_step_counter=0, step_dt=.5, episode_length_buf=torch.tensor([0,0]))
    def step(ok, episode_steps):
        env.common_step_counter += 1
        env.episode_length_buf = torch.tensor(episode_steps)
        return sustained(env, torch.tensor(ok), 1., 'r1:test')
    assert not step([[True,False],[True,False]], [1,1]).any()
    assert step([[False,True],[True,False]], [2,2]).tolist() == [[False,False],[True,False]]
    assert step([[False,True],[True,False]], [3,0]).tolist() == [[False,True],[False,False]]


def _pile_env(poses, step=1):
    """A stand-in env for r1_pick.state: each fruit's root pose (env-local xyz, xyzw) at rest, no hand sensors."""
    import warp as wp

    def body(position, quat):
        return SimpleNamespace(data=SimpleNamespace(
            root_pos_w=wp.from_torch(torch.tensor([position])), root_quat_w=wp.from_torch(torch.tensor([quat])),
            root_lin_vel_w=wp.from_torch(torch.zeros(1, 3)), root_ang_vel_w=wp.from_torch(torch.zeros(1, 3))))

    scene = type("Scene", (dict,), {"env_origins": torch.zeros(1, 3), "sensors": {}})(
        {name: body(*pose) for name, pose in poses.items()})
    return SimpleNamespace(scene=scene, common_step_counter=step)


def test_the_pile_state_is_evaluated_once_per_control_step(monkeypatch):
    r1_pick = pytest.importorskip("dexbench_arena.tasks.r1_pick", reason="needs Isaac Lab and Warp")
    objects = [{"name": f"r1_fruit_{i:03d}", "fruit_id": "plum_pile", "center_local": (0., 0., 0.),
                "half_extents": (.018, .018, .018), "mass_kg": .0483} for i in range(3)]
    env = _pile_env({o["name"]: ((3.15 + .05 * i, .7, 1.08), (0., 0., 0., 1.)) for i, o in enumerate(objects)})
    calls = []
    evaluate_rules = r1_pick.r1_rules.evaluate
    monkeypatch.setattr(r1_pick.r1_rules, "evaluate", lambda *a, **k: calls.append(1) or evaluate_rules(*a, **k))
    rules = dict(RULES)
    first = r1_pick.state(env, objects, "plum_pile", rules)
    assert r1_pick.state(env, objects, "plum_pile", rules) is first and len(calls) == 1
    env.common_step_counter += 1
    r1_pick.state(env, objects, "plum_pile", rules)
    assert len(calls) == 2
