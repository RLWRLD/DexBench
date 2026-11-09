# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""DB-M5 success/failure terms against a fake env, without Kit."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dexbench_arena.layouts import db_m5_layout
from dexbench_arena.environments.manipulation import (
    e3_manufacturing_db_m5_environment as m5,
)
from dexbench_arena.layouts.db_m5_parts import get_m5_part
from dexbench_arena.tasks import kit_box_placement as placement

ApplyKitBoxEmptyCompartment = pytest.importorskip(
    "dexbench_arena.variations.kit_box_compartment_variation", exc_type=ImportError
).ApplyKitBoxEmptyCompartment

from isaaclab_arena.variations.choice_sampler import ChoiceSampler  # noqa: E402

BELT_SPEED = 0.15
EMPTY_COMPARTMENT = {
    "compartment_min_xy_local": db_m5_layout.KIT_BOX_EMPTY_COMPARTMENT_MIN_XY_LOCAL,
    "compartment_max_xy_local": db_m5_layout.KIT_BOX_EMPTY_COMPARTMENT_MAX_XY_LOCAL,
    "compartment_floor_z_local": db_m5_layout.KIT_BOX_COMPARTMENT_FLOOR_TOP_Z,
}
COMPARTMENT = {
    "compartments_min_xy_local": db_m5_layout.KIT_BOX_COMPARTMENTS_MIN_XY_LOCAL,
    "compartments_max_xy_local": db_m5_layout.KIT_BOX_COMPARTMENTS_MAX_XY_LOCAL,
    "default_empty_compartment": db_m5_layout.DEFAULT_EMPTY_COMPARTMENT,
    "compartment_floor_z_local": db_m5_layout.KIT_BOX_COMPARTMENT_FLOOR_TOP_Z,
    "part_center_local": get_m5_part("bolt_m24").center_local,
    "part_extents_local": get_m5_part("bolt_m24").extents_local,
    "part_end_points_local": get_m5_part("bolt_m24").end_points_local,
}
SEATED = {
    "kit_box_half_xy": db_m5_layout.KIT_BOX_OUTER_HALF_XY,
    "belt_y_range": (db_m5_layout.CONVEYOR_FRAME_S_BELT_FACE_Y_M, db_m5_layout.CONVEYOR_FRAME_N_BELT_FACE_Y_M),
    "belt_top_z": db_m5_layout.CONVEYOR_TOP_Z_M,
    "max_belt_gap_m": db_m5_layout.KIT_BOX_MAX_BELT_GAP_M,
}
EMPTY_COMPARTMENT_ROOT_LOCAL = (-0.087, -0.065, db_m5_layout.KIT_BOX_COMPARTMENT_FLOOR_TOP_Z)
NEAR_RIGHT_COMPARTMENT_ROOT_LOCAL = (0.087, -0.065, db_m5_layout.KIT_BOX_COMPARTMENT_FLOOR_TOP_Z)
BOLTS = ("bolt_a", "bolt_b")


def _yaw_xyzw(deg: float) -> list[float]:
    half = math.radians(deg) / 2
    return [0.0, 0.0, math.sin(half), math.cos(half)]


def make_env(box_xyz=(2.3, 3.2045, 0.75), box_yaw_deg=0.0):
    import warp as wp

    wp.init()
    origins = torch.tensor([[0.0, 0.0, 0.0], [8.0, 8.0, 2.0]])

    def body(pos, quat, mass_kg=0.45):
        return SimpleNamespace(
            data=SimpleNamespace(
                root_pos_w=wp.from_torch(torch.tensor([pos] * 2) + origins, dtype=wp.vec3),
                root_quat_w=wp.from_torch(torch.tensor([quat] * 2), dtype=wp.quat),
                body_mass=SimpleNamespace(torch=torch.full((2, 1, 1), mass_kg)),
            )
        )

    class Scene(dict):
        pass

    scene = Scene(
        kit_box=body(list(box_xyz), _yaw_xyzw(box_yaw_deg)),
        bolt_a=body([2.85, 2.76, 0.63], [0.0, 0.0, 0.0, 1.0]),
        bolt_b=body([2.90, 2.76, 0.63], [0.0, 0.0, 0.0, 1.0]),
    )
    scene.env_origins = origins
    scene.sensors = {
        f"hands_on_{name}": SimpleNamespace(
            data=SimpleNamespace(force_matrix_w=SimpleNamespace(torch=torch.zeros(2, 1, 1, 3)))
        )
        for name in BOLTS
    }
    compartment_draw = object.__new__(ApplyKitBoxEmptyCompartment)
    compartment_draw.empty_compartment = torch.full((2,), -1, dtype=torch.long)
    compartment_term = SimpleNamespace(func=compartment_draw, params={"sampler": ChoiceSampler()})
    events = SimpleNamespace(
        active_terms={"reset": ["kit_box_empty_compartment_variation"]},
        get_term_cfg=lambda name: compartment_term,
    )
    env = SimpleNamespace(
        scene=scene,
        event_manager=events,
        compartment_draw=compartment_draw,
        num_envs=2,
        device="cpu",
        step_dt=1 / 30,
        common_step_counter=0,
        episode_length_buf=torch.zeros(2, dtype=torch.long),
        riders=set(),
    )
    env.unwrapped = env
    return env


def _pos(env, name):
    import warp as wp

    return wp.to_torch(env.scene[name].data.root_pos_w)


def _quat(env, name):
    import warp as wp

    return wp.to_torch(env.scene[name].data.root_quat_w)


def _place(env, name, root_local, *, ride=True):
    box_quat = _quat(env, "kit_box")
    offset = placement.rotate_xyzw(box_quat, torch.tensor([root_local] * 2))
    _pos(env, name)[:] = _pos(env, "kit_box") + offset
    _quat(env, name)[:] = box_quat
    if ride:
        env.riders.add(name)


def _success(env):
    return placement.part_placed_in_kit_box_compartment(env, "kit_box", BOLTS, **COMPARTMENT, **SEATED)


def _advance(env, steps=1):
    """One control step: the belt carries the box (and its riders) +X, then the term is read."""
    result = None
    for _ in range(steps):
        step = torch.tensor([BELT_SPEED * env.step_dt, 0.0, 0.0])
        for name in ("kit_box", *env.riders):
            _pos(env, name)[:] += step
        env.common_step_counter += 1
        env.episode_length_buf += 1
        result = _success(env)
    return result


def test_a_bolt_counts_on_the_step_it_lands_in_the_compartment_its_reset_left_empty():
    env = make_env(box_yaw_deg=180.0)
    env.compartment_draw.empty_compartment = torch.tensor([1, -1])
    assert not _advance(env).any()
    _place(env, "bolt_a", NEAR_RIGHT_COMPARTMENT_ROOT_LOCAL)
    _place(env, "bolt_b", EMPTY_COMPARTMENT_ROOT_LOCAL)
    assert _success(env).tolist() == [True, True]
    env.riders.discard("bolt_b")
    _pos(env, "bolt_b")[:] = torch.tensor([2.90, 2.76, 0.63]) + env.scene.env_origins
    assert _advance(env).tolist() == [True, False]


def test_a_bolt_still_moving_relative_to_the_box_counts():
    env = make_env()
    _place(env, "bolt_a", EMPTY_COMPARTMENT_ROOT_LOCAL, ride=False)
    _quat(env, "bolt_a")[:] = torch.tensor(_yaw_xyzw(math.degrees(2.0 * env.step_dt)))
    assert _success(env).all()


def test_a_bolt_counts_only_once_the_hands_have_let_go_of_it_for_the_hold():
    pytest.importorskip("dexbench_arena.tasks.predicates.release", exc_type=ImportError)
    env = make_env()

    def success():
        return placement.part_placed_in_kit_box_compartment(
            env, "kit_box", BOLTS, **COMPARTMENT, **SEATED,
            hand_sensor_names=tuple(f"hands_on_{name}" for name in BOLTS), hold_time_s=0.5,
        )

    def advance():
        env.common_step_counter += 1
        env.episode_length_buf += 1
        return success()

    held = env.scene.sensors["hands_on_bolt_a"].data.force_matrix_w.torch
    _place(env, "bolt_a", EMPTY_COMPARTMENT_ROOT_LOCAL)
    held[..., 0] = 3.0
    env.scene.sensors["hands_on_bolt_b"].data.force_matrix_w.torch[..., 0] = 5.0
    assert not any(advance().any() for _ in range(30))
    held[..., 0] = 0.0
    assert not any(advance().any() for _ in range(10))
    held[..., 0] = 3.0
    advance()
    held[0, ..., 0] = 0.2
    assert [advance().tolist() for _ in range(15)][-2:] == [[False, False], [True, False]]


def test_a_light_part_still_gripped_under_1_n_does_not_count():
    pytest.importorskip("dexbench_arena.tasks.predicates.release", exc_type=ImportError)
    from dexbench_arena.tasks.place_part_in_moving_kit_box import PLACE_PART_RELEASE_MAX_WEIGHT_FRACTION

    env = make_env()
    env.scene["bolt_a"].data.body_mass.torch[:] = 0.038
    _place(env, "bolt_a", EMPTY_COMPARTMENT_ROOT_LOCAL)
    env.scene.sensors["hands_on_bolt_a"].data.force_matrix_w.torch[..., 0] = 0.707

    def held_for(fraction, key):
        for _ in range(30):
            env.common_step_counter += 1
            env.episode_length_buf += 1
            result = placement.part_placed_in_kit_box_compartment(
                env, "kit_box", BOLTS, **COMPARTMENT, **SEATED, hand_sensor_names=tuple(f"hands_on_{n}" for n in BOLTS),
                max_hand_weight_fraction=fraction, hold_time_s=0.5, hold_key=key,
            )
        return result.tolist()

    assert held_for(None, "cap_only") == [True, True]
    assert held_for(PLACE_PART_RELEASE_MAX_WEIGHT_FRACTION, "weight_share") == [False, False]
    env.scene.sensors["hands_on_bolt_a"].data.force_matrix_w.torch[..., 0] = 0.05
    assert held_for(PLACE_PART_RELEASE_MAX_WEIGHT_FRACTION, "weight_share") == [True, True]


def test_a_replay_judges_a_demo_against_the_compartment_it_was_recorded_with(tmp_path):
    import sys

    import h5py

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts/imitation_learning"))
    from dexbench_arena.recording.rerecord_metadata import demo_attributes
    from dexbench_arena.tasks.place_part_in_moving_kit_box import DexBenchPlacePartInMovingKitBoxTask

    recording = tmp_path / "takes.hdf5"
    with h5py.File(recording, "w") as file:
        demo = file.create_group("data/demo_0")
        demo.attrs["success"] = True
        demo.create_dataset("variations/kit_box/empty_compartment", data=[[3.0]])
    env = make_env()
    task = SimpleNamespace(kit_box_name="kit_box")
    DexBenchPlacePartInMovingKitBoxTask.prepare_replay_episode(task, env, demo_attributes(recording, "demo_0"))
    sampler = env.event_manager.get_term_cfg("kit_box_empty_compartment_variation").params["sampler"]
    assert [sampler.sample(1, choices=[0, 3]) for _ in range(8)] == [[3]] * 8
    with h5py.File(recording, "a") as file:
        del file["data/demo_0/variations"]
    with pytest.raises(ValueError, match="empty_compartment"):
        DexBenchPlacePartInMovingKitBoxTask.prepare_replay_episode(task, env, demo_attributes(recording, "demo_0"))


def _in_empty_compartment(key, root_local, rotation_xyzw) -> bool:
    """``parts_in_compartment`` for one part at a pose in the box frame (box at the origin, unturned)."""

    part = get_m5_part(key)
    inside, _ = placement.parts_in_compartment(
        torch.zeros(1, 3),
        torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        torch.tensor([[root_local]]),
        torch.tensor([[rotation_xyzw]]),
        **EMPTY_COMPARTMENT,
        part_center_local=part.center_local,
        part_extents_local=part.extents_local,
        part_end_points_local=part.end_points_local,
    )
    return bool(inside.item())


@pytest.mark.parametrize(
    ("key", "root_local", "rotation_xyzw"),
    [
        ("bolt_m24", (-0.087, -0.065, db_m5_layout.KIT_BOX_COMPARTMENT_FLOOR_TOP_Z + 0.02), (0.0, 0.0, 0.0, 1.0)),
        ("pom_roller", (-0.112, -0.027645, 0.079211), (0.653281, 0.270598, 0.653281, 0.270598)),
        ("bolt_m24", (-0.087, -0.065, -0.10), (0.0, 0.0, 0.0, 1.0)),
    ],
)
def test_a_part_off_the_floor_under_the_box_or_reaching_out_of_the_compartment_does_not_count(key, root_local, rotation_xyzw):
    assert not _in_empty_compartment(key, root_local, rotation_xyzw)


def _roll_xyzw(deg: float) -> list[float]:
    half = math.radians(deg) / 2
    return [math.sin(half), 0.0, 0.0, math.cos(half)]


@pytest.mark.parametrize(
    "unseat",
    [
        lambda pos, quat: pos.__setitem__((slice(None), 2), pos[:, 2] + 0.02),
        lambda pos, quat: pos.__setitem__((slice(None), 1), 3.1),
        lambda pos, quat: quat.__setitem__(slice(None), torch.tensor(_roll_xyzw(10.0))),
    ],
)
def test_a_placed_bolt_only_succeeds_while_the_box_is_seated(unseat):
    env = make_env()
    _place(env, "bolt_a", EMPTY_COMPARTMENT_ROOT_LOCAL)
    assert _advance(env).all()
    unseat(_pos(env, "kit_box"), _quat(env, "kit_box"))
    _place(env, "bolt_a", EMPTY_COMPARTMENT_ROOT_LOCAL)
    assert not _advance(env).any()


@pytest.mark.parametrize(("box_x", "expected"), [(4.3 - 0.185 - 0.01, False), (4.2, True)])
def test_the_belt_end_deadline_is_the_leading_edge_at_the_belt_end(box_x, expected):
    env = make_env(box_xyz=(box_x, 3.2045, 0.75))
    reached = placement.kit_box_reached_belt_end(
        env, "kit_box", db_m5_layout.KIT_BOX_LEADING_EDGE_LOCAL_X, db_m5_layout.CONVEYOR_BELT_END_X_M
    )
    assert reached.tolist() == [expected, expected]
