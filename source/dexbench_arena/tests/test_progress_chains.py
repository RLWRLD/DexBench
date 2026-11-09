# SPDX-License-Identifier: Apache-2.0
"""The EA tasks' progress chains advance through their stages and never decide success."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

pytest.importorskip("isaaclab.sim", exc_type=ImportError)
pytest.importorskip("isaaclab_arena")

import torch  # noqa: E402
import warp as wp  # noqa: E402
from isaaclab.utils.warp.proxy_array import ProxyArray  # noqa: E402
from isaaclab_arena.environments.arena_world import ArenaWorld  # noqa: E402
from isaaclab_arena.progress_tracking.progress_tracker import ProgressTracker  # noqa: E402
from isaaclab_arena.tasks.predicates.object_settling import ObjectInitialRestPoseRecorder  # noqa: E402

from dexbench_arena.assets.e2_logistics_cell import E2_PARCEL_CAGE_SLOT_YAW_DEG  # noqa: E402
from dexbench_arena.tasks.conveyor_placement_config import (  # noqa: E402
    CONVEYOR_SET_DOWN_TOP_Z_M,
    CONVEYOR_SET_DOWN_XY_RECT_M,
    set_down_x_m,
)
from dexbench_arena.tasks.cage_clear import cage_front_plane  # noqa: E402
from dexbench_arena.environments.manipulation import e4_hotel_fb_environment as e4  # noqa: E402
from dexbench_arena.environments.manipulation import e2_logistics_environment as e2  # noqa: E402
from dexbench_arena.environments.manipulation import e3_assembly_line_environment as e3  # noqa: E402
from dexbench_arena.layouts import db_m1_layout as m1  # noqa: E402
from dexbench_arena.layouts.db_h1_layout import (  # noqa: E402
    ROUND_TABLE_TOP_Z_M,
    SOURCE_STACK_XY_M,
)
from dexbench_arena.tasks.task_termination import task_termination_cfg  # noqa: E402

UP = (0.0, 0.0, 0.0, 1.0)
STEP_DT = 1 / 30


def _proxy(values) -> ProxyArray:
    return ProxyArray(wp.from_torch(torch.as_tensor([values], dtype=torch.float32)))


class _Entity:
    def __init__(self):
        self.data = SimpleNamespace()

    def place(self, xyz, quat=UP, body_pos=None):
        pose = [*xyz, *quat]
        d = self.data
        d.root_pose_w, d.root_pos_w, d.root_quat_w = _proxy(pose), _proxy(list(xyz)), _proxy(list(quat))
        d.root_vel_w, d.root_lin_vel_w, d.root_ang_vel_w = _proxy([0.0] * 6), _proxy([0.0] * 3), _proxy([0.0] * 3)
        if body_pos is not None:
            d.body_pos_w = _proxy(body_pos)


class _Scene(dict):
    def __init__(self, rigid, articulated=()):
        super().__init__({name: _Entity() for name in (*rigid, *articulated)})
        self.rigid_objects = {name: self[name] for name in rigid}
        self.articulations = {name: self[name] for name in articulated}
        self.deformable_objects, self.extras, self.sensors = {}, {}, {}
        self.num_envs, self.device, self.env_origins = 1, "cpu", torch.zeros(1, 3)


class _Env:
    """What the predicates read from an Isaac Lab Arena env, for one environment."""

    def __init__(self, scene):
        self.scene, self.num_envs, self.device, self.step_dt, self.unwrapped = scene, 1, "cpu", STEP_DT, self
        self.arena_world = ArenaWorld(scene)
        self.object_initial_rest_pose_recorder = ObjectInitialRestPoseRecorder(1, "cpu")
        self.episode_length_buf, self.common_step_counter = torch.zeros(1, dtype=torch.long), 0


def _tracker(cfg, env):
    return ProgressTracker(cfg.success, 1, "cpu", env=env, subtasks_are_sequential=cfg.subtasks_are_sequential,
                           desired_subtask_success_state=cfg.desired_subtask_success_state)


def _run(task, rigid, articulated, phases):
    """Hold each phase's poses for its steps; return (progress score, success) after every phase."""
    envs = [_Env(_Scene(rigid, articulated)) for _ in range(2)]
    with_chain = _tracker(task.get_termination_cfg(), envs[0])
    without = _tracker(task_termination_cfg(task.episode_length_s, task.termination_cfg), envs[1])
    step, results = 0, []
    for poses, steps in phases:
        for _ in range(steps):
            for env, tracker in ((envs[0], with_chain), (envs[1], without)):
                for name, place in poses.items():
                    env.scene[name].place(*place)
                env.episode_length_buf, env.common_step_counter = torch.tensor([step + 1]), step + 1
                tracker.step(env, step_index=torch.tensor([step]))
            assert bool(with_chain.is_complete()[0]) == bool(without.is_complete()[0]), f"success flips at step {step}"
            step += 1
        state = with_chain.get_state()[0]
        results.append((round(state.overall_score, 3), state.all_complete))
    return results


def _factory(module, env_name, cfg_name, task_name, **fields):
    env_cls = getattr(module, env_name)
    return env_cls.available_tasks[task_name](object.__new__(env_cls), getattr(module, cfg_name)(**fields))


STAGES = [(0.0, False), (0.25, False), (0.5, False), (0.75, False), (1.0, True)]


def _l2_task():
    return _factory(e2, "DexBenchLogisticsCenterE2Environment", "DexBenchLogisticsCenterE2EnvironmentCfg",
                    "lift_to_conveyor", task="lift_to_conveyor")


def test_l2_parcel_out_of_the_cage_onto_the_conveyor():
    task = _l2_task()
    params = task.termination_cfg.success.params
    yaw = math.radians(E2_PARCEL_CAGE_SLOT_YAW_DEG)
    turned = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
    sx = params["box_sizes_xyz"][0][1]
    (_, y0), (_, y1) = CONVEYOR_SET_DOWN_XY_RECT_M
    belt = (set_down_x_m(sx), (y0 + y1) / 2)
    top = CONVEYOR_SET_DOWN_TOP_Z_M
    (front_x, front_y), (out_nx, out_ny) = cage_front_plane()

    def off_front(distance):
        return (front_x + out_nx * distance, front_y + out_ny * distance)

    def parcel(xyz):
        return {"cage_l2_package": (xyz, turned)}

    rest = (*off_front(-0.22), 0.87)
    phases = [
        (parcel(rest), 3),
        (parcel((*rest[:2], 0.90)), 3),
        (parcel((*off_front(0.02 + sx / 2 + 0.01), 0.95)), 3),
        (parcel((*belt, top - 0.30)), 3),
        (parcel((*belt, top + 0.10)), 3),
        (parcel((*belt, top)), 30),
    ]
    assert _run(task, ["cage_l2_package"], [], phases) == [*STAGES[:3], STAGES[2], *STAGES[3:]]

    assert _run(_l2_task(), ["cage_l2_package"], [], [(parcel(rest), 3), (parcel((*belt, top)), 30)]) == [
        (0.0, False), (0.0, True)
    ]


def test_m1_drawer_aligned_engaged_half_in_seated():
    task = _factory(e3, "DexBenchManufacturingE3M1Environment", "DexBenchManufacturingE3M1EnvironmentCfg",
                    "insert_drawer", task="insert_drawer")
    washer = (m1.WASHER_ROOT_XYZ, UP)
    off_axis = 2.5 * m1.PROGRESS_ALIGN_MAX_LATERAL_ERROR_M

    def drawer(depth, lateral=0.0):
        seat = m1.SEAT_XYZ
        return {"washer": washer, "drawer": ((seat[0] + lateral, seat[1] + depth, seat[2]),)}

    aligned = m1.PROGRESS_RAIL_ENTRY_DEPTH_M - m1.PROGRESS_ALIGN_LEAD_M / 2
    engaged = m1.PROGRESS_RAILS_ENGAGED_DEPTH_M + 0.01
    half = m1.PROGRESS_HALF_DEPTH_M + 0.01
    phases = [
        (drawer(m1.PROGRESS_RAIL_ENTRY_DEPTH_M + m1.START_DEPTH_M), 3),
        (drawer(aligned, lateral=off_axis), 3),
        (drawer(aligned), 3),
        (drawer(engaged), 3),
        (drawer(half, lateral=off_axis), 3),
        (drawer(half), 3),
        (drawer(m1.SEAT_PROXY_HOME_DY_M), 30),
    ]
    assert _run(task, ["drawer", "washer"], [], phases) == [
        STAGES[0], STAGES[0], STAGES[1], STAGES[2], STAGES[2], STAGES[3], STAGES[4]
    ]


@pytest.mark.parametrize("drawn", [False, True], ids=["nominal", "drawn"])
def test_h1_table_stack_onto_the_cart_stack(drawn):
    """With ``success_follows_dest_draw`` and a drawn cart stack, "over" follows the draw like success does."""
    cfg = e4.DexBenchHotelFBE4EnvironmentCfg(task="stack_plates")
    task = _factory(e4, "DexBenchHotelFBE4Environment", "DexBenchHotelFBE4EnvironmentCfg", "stack_plates",
                    task="stack_plates", success_follows_dest_draw=drawn)
    params = task.termination_cfg.success.params
    source, dest = e4.DexBenchHotelFBE4Environment._plate_names(cfg)
    pitch, top = params["nest_pitch_m"], params["surface_top_z_m"]
    cx, cy = params["stack_center_xy"]
    if drawn:
        dx = 2 * params["region_radius_m"]
        task.stack_center_offset(torch.tensor([[dx, 0.0, 0.0]]), [0])
        cx += dx
    far = [[0.0, 0.0, 0.0]] * 4

    def scene(x, y, z):
        poses = {name: ((cx, cy, top + k * pitch),) for k, name in enumerate(dest)}
        poses.update({name: ((x, y, z + k * pitch),) for k, name in enumerate(source)})
        poses["robot"] = ((0.0, 0.0, 0.0), UP, far)
        return poses

    seated = top + len(dest) * pitch
    phases = [
        (scene(*SOURCE_STACK_XY_M, ROUND_TABLE_TOP_Z_M), 3),
        (scene(*SOURCE_STACK_XY_M, ROUND_TABLE_TOP_Z_M + 0.03), 3),
        (scene(*SOURCE_STACK_XY_M, top + 0.05), 3),
        (scene(cx, cy, top + 0.30), 3),
        (scene(cx, cy, seated), 30),
    ]
    assert _run(task, [*source, *dest], ["robot"], phases) == STAGES
