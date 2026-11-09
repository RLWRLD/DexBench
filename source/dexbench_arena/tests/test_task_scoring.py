# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Each DexBench task scores the same through Arena's progress tracker as through its own success term (Kit tier)."""

from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("isaaclab")
pytest.importorskip("isaaclab_arena")

import gymnasium as gym  # noqa: E402
import warp as wp  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_from_matrix, quat_mul  # noqa: E402

from dexbench_arena.assets.e2_logistics_cell import E2_PARCEL_CAGE_SLOT_YAW_DEG  # noqa: E402
from dexbench_arena.tasks.conveyor_placement_config import (  # noqa: E402
    CONVEYOR_SET_DOWN_TOP_Z_M,
    CONVEYOR_SET_DOWN_XY_RECT_M,
    set_down_x_m,
)
from dexbench_arena.layouts.db_m4_layout import THREAD_PITCH_M  # noqa: E402
from dexbench_arena.layouts.db_m5_parts import get_m5_part  # noqa: E402
from dexbench_arena.tasks.extract_utensil import BIN_FLOOR, BIN_XY, TABLE_TOP  # noqa: E402
from dexbench_arena.tasks.kit_box_placement import kit_box_empty_compartment  # noqa: E402
from dexbench_arena.tasks.task_termination import composed_success_term  # noqa: E402

pytestmark = pytest.mark.usefixtures("simulation_app")

REFERENCE = "reference_success"
OUTPUTS = Path(__file__).resolve().parents[3] / "outputs"
FETCH = "scripts/download_datasets.sh --eval-conditions"
E1 = ["dexbench_retail_v1_environment", "--scene_layout", "fc_260821"]
WUJI2 = ["--embodiment", "dexbench_rby1_wuji2_joint", "--arm_mode", "dual_arm"]
M5_PART = "bolt_m24"
HOLD = {"h3": "targets"}
TASKS = {
    "r1": E1 + ["--task", "pick_fruit_from_display", "--anchor", "Fruit", "--start_pose", "teleop_home",
                "--r1_target_sku", "plum_pile", "--r1_layouts", str(OUTPUTS / "db_conditions/db_r1_layouts/eval")],
    "r2": E1 + ["--task", "retrieve_beverage_from_clutter", "--anchor", "Fridge"],
    "r3": E1 + ["--task", "place_box_in_slot", "--anchor", "Gondola"],
    "l2": ["dexbench_logistics_center_e2_environment", "--task", "lift_to_conveyor", "--anchor", "A1_CAGE"],
    "m1": ["dexbench_manufacturing_e3_m1_environment", "--task", "pick_insert_drawer", "--anchor", "A_m1",
           "--start_pose", "teleop_ready", "--m1_tabletop_base_xy", "0.15", "-0.9757",
           "--m1_tabletop_base_yaw_deg", "180", "--m1_tabletop_crouch_deg", "55",
           "--m1_tabletop_drawer_mass_kg", "0.55", "--m1_physics_profile", "damped_contacts"],
    "m2": ["dexbench_manufacturing_e3_m2_environment", "--task", "insert_connector", "--anchor", "A_m2",
           "--m2_connector", "S2", "--m2_plug_condim", "3", "--m2_fingertip_condim", "4",
           "--m2_wire_collision", "panel", "--m2_wire_rest_spot", "-0.008", "0.008", "0.0872665"],
    "m4": ["dexbench_manufacturing_e3_environment", "--task", "hand_start_leg", "--anchor", "A_m4"],
    "m5": ["dexbench_manufacturing_m5_moving_conveyor_kitting_environment", "--task", "place_part_in_kit_box",
           "--anchor", "A_m5", "--arm_mode", "left", "--part", M5_PART,
           "--parts_bin_clutter_dir", str(OUTPUTS / "db_conditions/db_m5_piles"),
           "--kit_box_empty_compartments", "0,3", "--parts_bin_seed_quantiles", "0.0,0.6"],
    "h1": ["dexbench_hotel_fb_e4_environment", "--task", "stack_plates"],
    "h3": ["dexbench_hotel_fb_e4_h3_environment", "--task", "extract_utensil", "--anchor", "A_h3",
           "--h3_object", "train_01_fork", "--h3_inputs_dir", str(OUTPUTS / "db_h3_inputs")],
}
DROPPED = {
    "r1": "fruit_lost", "r2": "target_dropped", "r3": "object_dropped", "l2": "object_dropped",
    "m1": "drawer_dropped", "m2": "plug_dropped", "m4": "leg_dropped", "m5": "kit_box_reached_belt_end",
    "h1": "plate_dropped", "h3": "dropped",
}


def _torch(value):
    return value if isinstance(value, torch.Tensor) else wp.to_torch(value)


def _place(env, name, local_xyz, quat_xyzw=None):
    """Put env 0's ``name`` at an env-local position (and orientation), at rest."""
    entity = env.scene[name]
    pose = _torch(entity.data.root_pose_w)[:1].clone()
    pose[0, :3] = torch.as_tensor(local_xyz, dtype=pose.dtype, device=pose.device) + env.scene.env_origins[0]
    if quat_xyzw is not None:
        pose[0, 3:7] = torch.as_tensor(quat_xyzw, dtype=pose.dtype, device=pose.device)
    entity.write_root_pose_to_sim_index(root_pose=pose)
    entity.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), dtype=pose.dtype, device=pose.device))


def _local(env, name):
    return _torch(env.scene[name].data.root_pos_w)[0] - env.scene.env_origins[0]


def _center_root(env, params, center_xyz, quat_xyzw, offset_xyz=None):
    """The root position that puts the predicate's object centre (``offset_xyz`` in the object's frame, by default
    the task's ``object_center_offset_xyz``) at ``center_xyz``."""
    quat = torch.as_tensor([quat_xyzw], dtype=torch.float32, device=env.device)
    offset_xyz = params["object_center_offset_xyz"] if offset_xyz is None else offset_xyz
    offset = torch.as_tensor([offset_xyz], dtype=torch.float32, device=env.device)
    return (torch.as_tensor(center_xyz, dtype=torch.float32, device=env.device) - quat_apply(quat, offset)[0]).tolist()


def _in_frame(env, name, local_xyz, quat_xyzw=(0.0, 0.0, 0.0, 1.0)):
    """The env-frame pose of a pose given in ``name``'s frame, as ``(xyz, xyzw)`` lists."""
    frame = env.arena_world.get_pose_e(name)[:1]
    point = torch.as_tensor([local_xyz], dtype=frame.dtype, device=frame.device)
    quat = torch.as_tensor([quat_xyzw], dtype=frame.dtype, device=frame.device)
    return (frame[0, :3] + quat_apply(frame[:, 3:], point)[0]).tolist(), quat_mul(frame[:, 3:], quat)[0].tolist()


def _yaw_xyzw(yaw):
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


def _set_success_state(task, env, params):
    """Put the task object(s) where the task's success predicate holds."""
    if task == "r1":
        fruit = next(o for o in params["objects"] if o["fruit_id"] == params["fruit_id"])
        (x0, y0, floor_z), (x1, y1, _) = params["rules"]["basket_bounds"]
        quat = (0.0, 0.0, 0.0, 1.0)
        center = ((x0 + x1) / 2, (y0 + y1) / 2, floor_z + fruit["half_extents"][2] + 0.003)
        _place(env, fruit["name"], _center_root(env, params, center, quat, fruit["center_local"]), quat)
    elif task == "r2":
        quat = (0.0, 0.0, 0.0, 1.0)
        x, y = (float(v) for v in params["basket_center_xy"])
        z = float(params["basket_floor_z_m"]) + float(params["object_center_offset_xyz"][2]) + 0.003
        _place(env, params["object_cfg"].name, _center_root(env, params, (x, y, z), quat), quat)
    elif task == "r3":
        quat = _yaw_xyzw(math.pi / 2)
        lo, hi = (float(v) for v in params["center_height_range_m"])
        x, y = (float(v) for v in params["slot_center_xy"])
        center = (x, y, float(params["shelf_top_z_m"]) + 0.5 * (lo + hi))
        _place(env, params["object_cfg"].name, _center_root(env, params, center, quat), quat)
    elif task == "l2":
        quat = _yaw_xyzw(math.radians(E2_PARCEL_CAGE_SLOT_YAW_DEG))
        (_, min_y), (_, max_y) = CONVEYOR_SET_DOWN_XY_RECT_M
        x = set_down_x_m(params["box_sizes_xyz"][0][1])
        _place(env, params["object_cfg"].name, (x, (min_y + max_y) / 2, CONVEYOR_SET_DOWN_TOP_Z_M), quat)
    elif task == "m1":
        lo, hi = params["depth_window_m"]
        x, y, z = params["seat_in_receiver_xyz"]
        _place(env, params["object_name"], *_in_frame(env, params["receiver_name"], (x, y + (lo + hi) / 2, z)))
    elif task == "m2":
        _place(env, params["plug_name"], params["seat_xyz"], params["seated_quat_xyzw"])
    elif task == "m4":
        depth = params["min_depth_m"] + 2 * THREAD_PITCH_M
        tip = [p + depth * a for p, a in zip(params["hole_in_receiver_xyz"], params["hole_axis_in_receiver"])]
        _place(env, params["object_name"], *_in_frame(env, params["receiver_name"], tip))
    elif task == "m5":
        index = int(kit_box_empty_compartment(env, params["default_empty_compartment"])[0])
        (x0, y0), (x1, y1) = params["compartments_min_xy_local"][index], params["compartments_max_xy_local"][index]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        (ox, oy, oz), quat = get_m5_part(M5_PART).kit_box_pose_local(cx)
        local = (cx + ox, cy + oy, params["compartment_floor_z_local"] + oz + 0.002)
        _place(env, params["part_names"][0], *_in_frame(env, params["kit_box_name"], local, quat))
    elif task == "h3":
        target = params["target_object"]
        lo, hi = (torch.as_tensor(v, dtype=torch.float32, device=env.device)
                  for v in params["object_bounds"][params["object_names"].index(target)])
        size = hi - lo
        longest, thinnest = int(size.argmax()), int(size.argmin())
        turn = torch.zeros(3, 3, device=env.device)
        turn[1, longest] = turn[2, thinnest] = turn[0, 3 - longest - thinnest] = 1.0
        turn[0, 3 - longest - thinnest] = float(torch.linalg.det(turn))
        quat = quat_from_matrix(turn).tolist()
        (bx0, by0), (bx1, by1) = BIN_XY
        center = ((bx0 + bx1) / 2, (by0 + by1) / 2, BIN_FLOOR + float(size[thinnest]) / 2 + 0.003)
        _place(env, target, _center_root(env, params, center, quat, ((lo + hi) / 2).tolist()), quat)
    else:
        center = torch.as_tensor(params["stack_center_xy"], dtype=torch.float32, device=env.device)
        names = list(params["plate_names"])
        in_region = [n for n in names if float(torch.norm(_local(env, n)[:2] - center)) <= params["region_radius_m"]]
        top = max(in_region, key=lambda n: float(_local(env, n)[2]))
        top_xy, quat = _local(env, top)[:2].tolist(), _torch(env.scene[top].data.root_quat_w)[0].tolist()
        for k, name in enumerate(n for n in names if n not in in_region):
            z = float(params["surface_top_z_m"]) + params["nest_pitch_m"] * (len(in_region) + k)
            _place(env, name, (*top_xy, z), quat)


def _drop(task, env, failure_params):
    """Put the task object below the height its dropped-object failure watches (DB-M5: the kit box at the belt
    end, its one failure)."""
    if task == "m5":
        name = failure_params["kit_box_name"]
        xyz = _local(env, name).tolist()
        xyz[0] = failure_params["belt_end_x"] - failure_params["leading_edge_local_x"] + 0.01
        _place(env, name, xyz)
        return
    if task == "h1":
        name, floor_z = failure_params["object_names"][0], float(failure_params["plane_y_m"])
    elif task == "h3":
        name, floor_z = failure_params["object_names"][0], TABLE_TOP - 0.15
    elif task == "r1":
        name, floor_z = failure_params["objects"][0]["name"], float(failure_params["rules"]["dropped_min_z_m"])
    elif task == "m2":
        name, floor_z = failure_params["plug_name"], float(failure_params["min_z_m"])
    else:
        name, floor_z = failure_params["asset_cfg"].name, float(failure_params["minimum_height"])
    xy = _local(env, name)[:2].tolist()
    _place(env, name, (*xy, floor_z - 0.15))


def _step(env, policy):
    """One control step of the do-nothing policy; the success and reference terms must agree on it."""
    actions = policy.get_action(env, None)
    with torch.inference_mode():
        _, _, terminated, truncated, _ = env.step(actions)
    manager = env.termination_manager
    success, reference = bool(manager.get_term("success")[0]), bool(manager.get_term(REFERENCE)[0])
    assert success == reference, f"tracker success {success} but the task's own term {reference}"
    fired = {name for name in manager.active_terms if bool(manager.get_term(name)[0])} - {REFERENCE}
    return fired, bool(terminated[0]), bool(truncated[0])


@pytest.mark.parametrize("task", TASKS, ids=TASKS.keys())
def test_success_failure_and_timeout_match_the_tasks_own_terms(task):
    from dexbench_arena.cli import get_arena_builder_from_cli, get_dexbench_arena_cli_parser
    from dexbench_arena.policy.hold_pose_policy import HoldPosePolicy, HoldPosePolicyCfg

    if missing := [arg for arg in TASKS[task] if arg.startswith(str(OUTPUTS)) and not Path(arg).exists()]:
        pytest.skip(f"{task} builds from {missing}; fetch them with {FETCH}")
    argv = TASKS[task][:1] + WUJI2 + TASKS[task][1:]
    builder = get_arena_builder_from_cli(get_dexbench_arena_cli_parser().parse_args(argv))
    name, cfg, kwargs = builder.build_registered()
    reference = copy.deepcopy(composed_success_term(cfg))
    setattr(cfg.terminations, REFERENCE, reference)
    cfg.recorders = cfg.metrics = None
    env = gym.make(name, cfg=cfg, **kwargs).unwrapped
    try:
        policy = HoldPosePolicy(HoldPosePolicyCfg(hold=HOLD.get(task, "default")))
        params = reference.params
        hold_steps = math.ceil(float(params.get("hold_time_s", 0.0)) / env.step_dt - 1e-9)

        env.reset()
        _set_success_state(task, env, params)
        for step in range(1, hold_steps + 61):
            fired, terminated, truncated = _step(env, policy)
            if "success" in fired:
                break
            assert not fired, f"step {step}: {sorted(fired)} fired before success"
        assert fired == {"success"} and terminated and not truncated, (step, fired)
        assert step >= hold_steps, f"success after {step} steps, inside the {hold_steps}-step hold"

        env.reset()
        failure = DROPPED[task]
        _drop(task, env, getattr(cfg.terminations, failure).params)
        fired, terminated, truncated = _step(env, policy)
        assert failure in fired and "success" not in fired and terminated and not truncated, fired

        env.reset()
        env.episode_length_buf[:] = env.max_episode_length - 1
        fired, terminated, truncated = _step(env, policy)
        assert fired == {"time_out"} and truncated and not terminated, fired
    finally:
        env.close()
