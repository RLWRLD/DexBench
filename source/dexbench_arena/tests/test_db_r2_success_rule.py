# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""DB-R2's strict success: the can contained and at rest for 90% of a sliding second, and no neighbour knocked."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

wp = pytest.importorskip("warp")
pytest.importorskip("isaaclab_arena.tasks.predicates.object_settling", exc_type=ImportError)

from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab_arena.tasks.predicates.object_settling import ObjectInitialRestPoseRecorder  # noqa: E402

from dexbench_arena.layouts.db_r2_layout import (  # noqa: E402
    FC_260821_R2_BASKET_CENTER_HEIGHT_RANGE_M,
    FC_260821_R2_BASKET_FLOOR_TOP_Z_M,
    FC_260821_R2_BASKET_INTERIOR_HALF_EXTENTS_XY_M,
    FC_260821_R2_BASKET_ORIGIN_XYZ_M,
    FC_260821_R2_FRIDGE_FRONT_PLANE_Y_M,
    FC_260821_R2_NEIGHBOR_KNOCK_LATERAL_M,
    FC_260821_R2_NEIGHBOR_KNOCK_LIFT_M,
    FC_260821_R2_NEIGHBOR_TOPPLE_MAX_UP_ALIGNMENT,
    FC_260821_R2_STRICT_HOLD_S_DEFAULT,
    FC_260821_R2_STRICT_HOLD_SHARE,
    FC_260821_R2_VELOCITY_THRESHOLD_M_S,
)
from dexbench_arena.tasks.predicates.e1_retail import db_r2_target_placed  # noqa: E402

HZ = 60
UPRIGHT = (0.0, 0.0, 0.0, 1.0)
CAN_IN_BASKET = (FC_260821_R2_BASKET_ORIGIN_XYZ_M[0], FC_260821_R2_BASKET_ORIGIN_XYZ_M[1], FC_260821_R2_BASKET_FLOOR_TOP_Z_M)
CAN_CENTER_OFFSET = (0.0, 0.0, 0.062)
NEIGHBOR_REST = (2.505, 2.52, 0.937)
STRICT = dict(
    object_cfg=SceneEntityCfg("r2_target_drink"),
    object_center_offset_xyz=CAN_CENTER_OFFSET,
    basket_center_xy=FC_260821_R2_BASKET_ORIGIN_XYZ_M[:2],
    basket_center_halfsize_xy=FC_260821_R2_BASKET_INTERIOR_HALF_EXTENTS_XY_M,
    basket_floor_z_m=FC_260821_R2_BASKET_FLOOR_TOP_Z_M,
    center_height_range_m=FC_260821_R2_BASKET_CENTER_HEIGHT_RANGE_M,
    neighbor_names=["r2_neighbor_0"],
    max_neighbor_displacement_m=None,
    neighbor_knock={
        "max_up_alignment": FC_260821_R2_NEIGHBOR_TOPPLE_MAX_UP_ALIGNMENT,
        "min_lift_m": FC_260821_R2_NEIGHBOR_KNOCK_LIFT_M,
        "plane_y_m": FC_260821_R2_FRIDGE_FRONT_PLANE_Y_M,
        "max_lateral_m": FC_260821_R2_NEIGHBOR_KNOCK_LATERAL_M,
    },
    velocity_threshold_m_s=FC_260821_R2_VELOCITY_THRESHOLD_M_S,
    hold_time_s=FC_260821_R2_STRICT_HOLD_S_DEFAULT,
    hold_share=FC_260821_R2_STRICT_HOLD_SHARE,
)


def _quat_about_x(deg: float) -> tuple[float, ...]:
    half = math.radians(deg) / 2.0
    return (math.sin(half), 0.0, 0.0, math.cos(half))


def _rigid(position, quat=UPRIGHT):
    data = SimpleNamespace(root_pos_w=wp.array([position], dtype=wp.float32, device="cpu"),
                           root_quat_w=wp.array([quat], dtype=wp.float32, device="cpu"),
                           root_lin_vel_w=wp.zeros((1, 3), dtype=wp.float32, device="cpu"),
                           root_ang_vel_w=wp.zeros((1, 3), dtype=wp.float32, device="cpu"))
    return SimpleNamespace(data=data)


def _set(entity, position=None, quat=None, ang_vel=0.0):
    if position is not None:
        entity.data.root_pos_w = wp.array([position], dtype=wp.float32, device="cpu")
    if quat is not None:
        entity.data.root_quat_w = wp.array([quat], dtype=wp.float32, device="cpu")
    entity.data.root_ang_vel_w = wp.array([(0.0, 0.0, ang_vel)], dtype=wp.float32, device="cpu")


class _Scene(dict):
    def __init__(self, entities: dict):
        super().__init__(entities)
        self.env_origins, self.num_envs, self.device = torch.zeros(1, 3), 1, "cpu"
        self.sensors, self.deformable_objects = {}, {}


def _env():
    entities = {"r2_target_drink": _rigid(CAN_IN_BASKET), "r2_neighbor_0": _rigid(NEIGHBOR_REST)}
    world = SimpleNamespace(
        get_root_linear_velocity_w=lambda name: wp.to_torch(entities[name].data.root_lin_vel_w),
        get_root_angular_velocity_w=lambda name: wp.to_torch(entities[name].data.root_ang_vel_w),
        get_position_w=lambda name: wp.to_torch(entities[name].data.root_pos_w),
    )
    env = SimpleNamespace(scene=_Scene(entities), arena_world=world,
                          object_initial_rest_pose_recorder=ObjectInitialRestPoseRecorder(1, "cpu"),
                          num_envs=1, device="cpu", step_dt=1.0 / HZ, common_step_counter=0,
                          episode_length_buf=torch.zeros(1, dtype=torch.long))
    env.unwrapped = env
    return env, entities


def _first_success_step(env, entities, *, can_spin_steps=(), neighbor=None, steps=3 * HZ):
    """Step the stand-in: the can rests in the basket except on ``can_spin_steps``; from step 2 on the
    neighbour is where ``neighbor`` puts it (its rest pose is recorded at step 1)."""
    for step in range(1, steps + 1):
        env.common_step_counter = step
        env.episode_length_buf[:] = step
        _set(entities["r2_target_drink"], ang_vel=0.5 if step in can_spin_steps else 0.0)
        if step == 2 and neighbor is not None:
            _set(entities["r2_neighbor_0"], **neighbor)
        if bool(db_r2_target_placed(env, **STRICT)[0]):
            return step
    return None


def test_a_can_at_rest_for_90_percent_of_the_second_succeeds_and_a_moving_one_does_not():
    env, entities = _env()
    assert _first_success_step(env, entities, can_spin_steps={10, 20, 30, 40, 50, 59}) == HZ
    env, entities = _env()
    assert _first_success_step(env, entities, can_spin_steps=set(range(5, 3 * HZ + 1, 5))) is None


@pytest.mark.parametrize("name, neighbor, succeeds", [
    ("nudged 25 mm back along its lane", {"position": (NEIGHBOR_REST[0], NEIGHBOR_REST[1] - 0.025, NEIGHBOR_REST[2])}, True),
    ("pushed 20 mm sideways, still in its lane", {"position": (NEIGHBOR_REST[0] + 0.020, *NEIGHBOR_REST[1:])}, True),
    ("toppled 60 deg", {"quat": _quat_about_x(60.0)}, False),
    ("risen 25 mm", {"position": (*NEIGHBOR_REST[:2], NEIGHBOR_REST[2] + 0.025)}, False),
    ("pushed half a lane sideways", {"position": (NEIGHBOR_REST[0] + FC_260821_R2_NEIGHBOR_KNOCK_LATERAL_M + 0.002,
                                                 *NEIGHBOR_REST[1:])}, False),
    ("out past the opening", {"position": (NEIGHBOR_REST[0], FC_260821_R2_FRIDGE_FRONT_PLANE_Y_M - 0.01,
                                           NEIGHBOR_REST[2])}, False),
])
def test_only_a_knocked_neighbour_fails_the_take(name, neighbor, succeeds):
    env, entities = _env()
    assert (_first_success_step(env, entities, neighbor=neighbor) == HZ) is succeeds, name
