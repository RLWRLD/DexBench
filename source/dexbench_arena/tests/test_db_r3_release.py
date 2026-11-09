# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""DB-R3 succeeds only once the carton is seated and the hand has let go of it and backed out, held for the hold."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

wp = pytest.importorskip("warp")
pytest.importorskip("isaaclab_arena.tasks.predicates.object_settling", exc_type=ImportError)

from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab_arena.tasks.predicates.object_settling import ObjectInitialRestPoseRecorder  # noqa: E402

from dexbench_arena.tasks.predicates import status  # noqa: E402
from dexbench_arena.layouts.e1_retail_fc_layout import (  # noqa: E402
    FC_260821_R3_NEIGHBOR_ALONG_AXIS,
    FC_260821_R3_NEIGHBOR_DEPTH_AXIS,
    FC_260821_R3_NEIGHBOR_MAX_ALONG_M,
    FC_260821_R3_NEIGHBOR_MAX_DEPTH_M,
    FC_260821_R3_NEIGHBOR_MAX_DROP_M,
    FC_260821_R3_NEIGHBOR_MAX_LIFT_M,
    FC_260821_R3_NEIGHBOR_TOPPLE_MAX_UP_ALIGNMENT,
)
from dexbench_arena.tasks.place_in_slot import DexBenchPlaceInSlotTask  # noqa: E402
from dexbench_arena.tasks.predicates.e1_retail import db_r3_neighbor_toppled_or_fallen, db_r3_placed  # noqa: E402

HZ = 30
SLOT = (0.9555, 0.7365)
HALF_DEPTH = 0.025
UPRIGHT = (0.0, 0.0, 0.0, 1.0)
NEIGHBOUR_REST = (SLOT[0], 0.5765, 1.0)
KNOCK = {
    "neighbor_local_up_xyz": ((0.0, 0.0, 1.0),),
    "max_up_alignment": FC_260821_R3_NEIGHBOR_TOPPLE_MAX_UP_ALIGNMENT,
    "max_drop_m": FC_260821_R3_NEIGHBOR_MAX_DROP_M,
    "min_lift_m": FC_260821_R3_NEIGHBOR_MAX_LIFT_M,
    "max_along_m": FC_260821_R3_NEIGHBOR_MAX_ALONG_M,
    "max_depth_m": FC_260821_R3_NEIGHBOR_MAX_DEPTH_M,
    "along_axis": FC_260821_R3_NEIGHBOR_ALONG_AXIS,
    "depth_axis": FC_260821_R3_NEIGHBOR_DEPTH_AXIS,
}


def _quat_about_x(deg: float) -> tuple[float, ...]:
    half = math.radians(deg) / 2.0
    return (math.sin(half), 0.0, 0.0, math.cos(half))


class _Scene(dict):
    def __init__(self, entities: dict, sensors: dict):
        super().__init__(entities)
        self.sensors, self.env_origins, self.num_envs, self.device = sensors, torch.zeros(1, 3), 1, "cpu"
        self.deformable_objects = {}


def _rigid(position, quat=UPRIGHT):
    data = SimpleNamespace(root_pos_w=wp.array([position], dtype=wp.float32, device="cpu"),
                           root_quat_w=wp.array([quat], dtype=wp.float32, device="cpu"),
                           root_lin_vel_w=wp.zeros((1, 3), dtype=wp.float32, device="cpu"),
                           root_ang_vel_w=wp.zeros((1, 3), dtype=wp.float32, device="cpu"))
    return SimpleNamespace(data=data)


def _env(*, hand_force_n: float, wrist_out_m: float, carton_quat=UPRIGHT):
    carton = _rigid((SLOT[0], SLOT[1], 1.0), carton_quat)
    neighbour = _rigid(NEIGHBOUR_REST)
    wrist_x = SLOT[0] + HALF_DEPTH + wrist_out_m
    robot = SimpleNamespace(find_bodies=lambda name: ([0], [name]),
                            data=SimpleNamespace(body_pos_w=wp.array([[[wrist_x, SLOT[1], 1.1]]], dtype=wp.float32,
                                                                     device="cpu")))
    sensor = SimpleNamespace(data=SimpleNamespace(
        force_matrix_w=SimpleNamespace(torch=torch.tensor([[[[hand_force_n, 0.0, 0.0]]]]))))
    scene = _Scene({"target_carton": carton, "n0": neighbour, "robot": robot}, {"hand_on_object_contact": sensor})
    entities = {"target_carton": carton, "n0": neighbour}
    world = SimpleNamespace(
        get_root_linear_velocity_w=lambda name: torch.zeros(1, 3),
        get_root_angular_velocity_w=lambda name: torch.zeros(1, 3),
        get_position_w=lambda name: wp.to_torch(entities[name].data.root_pos_w),
    )
    env = SimpleNamespace(scene=scene, arena_world=world, object_initial_rest_pose_recorder=ObjectInitialRestPoseRecorder(1, "cpu"),
                          num_envs=1, device="cpu", step_dt=1.0 / HZ, common_step_counter=0,
                          episode_length_buf=torch.zeros(1, dtype=torch.long))
    env.unwrapped = env
    return env


def _first_success_step(env, steps: int = 2 * HZ, knock: dict | None = None, neighbour_at_step_2=None) -> int | None:
    """Step the stand-in env and return the control step the success term first reports success, or None."""
    params = dict(
        object_cfg=SceneEntityCfg("target_carton"), object_center_offset_xyz=(0.0, 0.0, 0.025), slot_center_xy=SLOT,
        slot_center_halfsize_xy=(0.010, 0.010), shelf_top_z_m=1.0, center_height_range_m=(0.002, 0.037),
        neighbor_names=["n0"], neighbor_reference_m=0.005, hand_sensor_name="hand_on_object_contact", wrist_body_name="right_wrist",
        front_plane_xy=(SLOT[0] + HALF_DEPTH, SLOT[1]), outward_xy=(1.0, 0.0), hold_time_s=1.0, neighbor_knock=knock,
    )
    for step in range(1, steps + 1):
        env.common_step_counter = step
        env.episode_length_buf[:] = step
        if step == 2 and neighbour_at_step_2 is not None:
            position, quat = neighbour_at_step_2
            data = env.scene["n0"].data
            data.root_pos_w = wp.array([position], dtype=wp.float32, device="cpu")
            data.root_quat_w = wp.array([quat], dtype=wp.float32, device="cpu")
        if bool(db_r3_placed(env, **params)[0]):
            return step
    return None


def test_a_seated_carton_still_in_the_hand_is_not_a_success():
    env = _env(hand_force_n=5.0, wrist_out_m=0.08)
    assert _first_success_step(env) is None
    seen = status.latest(env)
    assert seen["seated"] and not seen["released"] and seen["hand_force_n"] == pytest.approx(5.0)


def test_a_seated_released_carton_with_the_hand_backed_out_succeeds_after_the_hold():
    env = _env(hand_force_n=0.2, wrist_out_m=0.06)
    assert _first_success_step(env) == HZ
    seen = status.latest(env)
    assert seen["seated"] and seen["released"] and seen["success"]
    assert seen["hand_force_n"] == pytest.approx(0.2) and seen["back_out_cm"] == pytest.approx(6.0, abs=1e-3)
    assert seen["neighbours_max_mm"] == pytest.approx(0.0) and seen["held_s"] == pytest.approx(1.0)
    assert not seen["neighbours_gate"]


def test_a_hand_let_go_but_left_in_the_slot_is_not_a_success():
    assert _first_success_step(_env(hand_force_n=0.0, wrist_out_m=0.02)) is None


def test_a_released_carton_that_tipped_over_is_not_a_success():
    env = _env(hand_force_n=0.0, wrist_out_m=0.08, carton_quat=_quat_about_x(30.0))
    assert _first_success_step(env) is None
    seen = status.latest(env)
    assert seen["released"] and not seen["upright"] and seen["lean_deg"] == pytest.approx(30.0, abs=0.1)


def test_a_session_prints_what_the_term_saw_twice_a_second(capsys):
    """record_demos.py and teleop.py print the term's parts as one JSON line per half second of sim time."""
    import importlib
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "imitation_learning"))
    script = importlib.import_module("_dexbench_arena_script")
    env = _env(hand_force_n=0.2, wrist_out_m=0.06)
    _first_success_step(env, steps=3)
    env.step = lambda action: None
    script._print_task_status(env)
    for _ in range(HZ):
        env.step(None)
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("[DexBench status] ")]
    assert len(lines) == 2
    seen = json.loads(lines[-1].removeprefix("[DexBench status] "))
    assert seen["task"] == "db_r3" and seen["seated"] and seen["released"] and not seen["success"]


@pytest.mark.parametrize("name, moved, succeeds", [
    ("nudged 40 mm along the shelf", (NEIGHBOUR_REST[0], NEIGHBOUR_REST[1] - 0.040, NEIGHBOUR_REST[2]), True),
    ("nudged 40 mm in depth", (NEIGHBOUR_REST[0] - 0.040, *NEIGHBOUR_REST[1:]), True),
    ("pushed 60 mm along the shelf", (NEIGHBOUR_REST[0], NEIGHBOUR_REST[1] - 0.060, NEIGHBOUR_REST[2]), False),
    ("pushed 60 mm in depth", (NEIGHBOUR_REST[0] - 0.060, *NEIGHBOUR_REST[1:]), False),
    ("risen 25 mm", (*NEIGHBOUR_REST[:2], NEIGHBOUR_REST[2] + 0.025), False),
])
def test_a_nudged_neighbour_passes_and_a_knocked_one_fails_the_take(name, moved, succeeds):
    env = _env(hand_force_n=0.2, wrist_out_m=0.06)
    step = _first_success_step(env, knock=KNOCK, neighbour_at_step_2=(moved, UPRIGHT))
    assert (step == HZ) is succeeds, name


def test_a_toppled_or_fallen_neighbour_ends_the_episode():
    terms = {key: KNOCK[key] for key in ("neighbor_local_up_xyz", "max_up_alignment", "max_drop_m")}
    for moved, quat in (((NEIGHBOUR_REST), _quat_about_x(60.0)), ((*NEIGHBOUR_REST[:2], NEIGHBOUR_REST[2] - 0.040), UPRIGHT)):
        env = _env(hand_force_n=0.2, wrist_out_m=0.06)
        assert _first_success_step(env, knock=KNOCK, neighbour_at_step_2=(moved, quat)) is None
        assert bool(db_r3_neighbor_toppled_or_fallen(env, ["n0"], **terms)[0])
    env = _env(hand_force_n=0.2, wrist_out_m=0.06)
    _first_success_step(env, steps=2, knock=KNOCK)
    assert not bool(db_r3_neighbor_toppled_or_fallen(env, ["n0"], **terms)[0])


def test_the_task_ends_the_episode_on_a_toppled_neighbour_only_when_it_has_the_rule():
    def failures(knock):
        task = DexBenchPlaceInSlotTask(
            object_name="target_carton", object_center_offset_xyz=(0.0, 0.0, 0.025), slot_center_xy=SLOT,
            slot_center_halfsize_xy=(0.010, 0.010), object_half_extents_xy=(0.025, 0.07), shelf_top_z_m=1.0,
            center_height_range_m=(0.002, 0.037), workspace_xy_bounds=((0.0, 0.0), (2.0, 2.0)),
            object_dropped_min_z_m=0.3, front_plane_xy=(SLOT[0] + HALF_DEPTH, SLOT[1]), outward_xy=(1.0, 0.0),
            neighbor_names=["n0"], neighbor_reference_m=0.005, neighbor_knock=knock)
        return task.get_termination_cfg().failures

    term = failures(KNOCK)["neighbor_toppled"]
    assert term.func is db_r3_neighbor_toppled_or_fallen and term.params["max_drop_m"] == FC_260821_R3_NEIGHBOR_MAX_DROP_M
    assert "neighbor_toppled" not in failures(None)
