# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for E2 full-box success and its managed/extracted-term lifecycle."""

import math
from types import SimpleNamespace

import pytest
import torch

from dexbench_arena.tasks.conveyor_placement_config import conveyor_placement_params
from dexbench_arena.tasks import conveyor_placement as placement
from dexbench_arena.tasks.predicates.hold import sustained_count


SIZE = ((0.48, 0.38, 0.34),)
E2_DECKS = {key: conveyor_placement_params()[key] for key in ("surface_xy_rects", "surface_top_z_range_m")}
E2_PLACE_POS = (3.34, 0.80, 0.821)
PLATE_TOP, FB1_TOP = 0.821, 0.800


def quat(axis, degrees):
    v = [0.0] * 4
    v[axis] = math.sin(math.radians(degrees) / 2)
    v[3] = math.cos(math.radians(degrees) / 2)
    return v


def resting(centre_xy, rotation, top_z, size=SIZE[0]):
    """The root (bottom-face centre) of a box turned by ``rotation`` whose lowest corner sits at ``top_z``."""
    q = torch.tensor([rotation], dtype=torch.float64)
    root = torch.tensor([[*centre_xy, 0.0]], dtype=torch.float64) - placement.rotate_xyzw(
        q, torch.tensor([[0.0, 0.0, size[2] / 2]], dtype=torch.float64)
    )
    lowest = placement.box_corners(root, q, (size,))[0, :, 2].min()
    root[0, 2] += top_z - lowest
    return tuple(root[0].tolist())


def checks(pos, rotation=(0.0, 0.0, 0.0, 1.0), linear=(0.0, 0.0, 0.0), angular=(0.0, 0.0, 0.0)):
    return placement.box_placement_checks(
        *[torch.tensor([x], dtype=torch.float64) for x in (pos, rotation, linear, angular)],
        box_sizes_xyz=SIZE,
        **conveyor_placement_params(),
    )


def failed(result):
    return [name for name, value in result.items() if not bool(value.all())]


POSES = {
    "upright on the plate": ((3.34, 0.80), quat(2, 90), PLATE_TOP, []),
    "upright on FB1 past the plate": ((4.0, 0.80), quat(2, 90), FB1_TOP, []),
    "stood on an end": ((3.40, 0.80), quat(1, 90), PLATE_TOP, []),
    "upside down": ((3.40, 0.80), quat(0, 180), PLATE_TOP, []),
    "wedged across the plate's end into its notch": ((3.66, 0.80), quat(1, 4), FB1_TOP, []),
    "on a long side": ((3.40, 0.80), quat(0, 90), PLATE_TOP, ["not_on_long_side"]),
    "leaning 30 degrees": ((3.40, 0.80), quat(1, 30), PLATE_TOP, ["flat"]),
    "held 5 cm above": ((3.34, 0.80), quat(2, 90), PLATE_TOP + 0.05, ["seated"]),
    "carried 30 cm above": ((3.34, 0.80), quat(2, 90), PLATE_TOP + 0.30, ["seated"]),
    "beside FB1's belt, on its frame rail": ((4.0, 1.22), quat(2, 90), 0.794, ["over"]),
}


@pytest.mark.parametrize("centre_xy, rotation, top_z, failing", POSES.values(), ids=POSES.keys())
def test_a_box_resting_flat_on_a_carrying_surface_counts_unless_on_a_long_side(centre_xy, rotation, top_z, failing):
    assert failed(checks(resting(centre_xy, rotation, top_z), rotation)) == failing


def test_rotation_alone_and_translation_alone_prevent_success():
    assert failed(checks(E2_PLACE_POS, linear=(0.04, 0, 0), angular=(0, 0, 0.22))) == []
    assert failed(checks(E2_PLACE_POS, angular=(0, 0, 0.25))) == ["angular_slow"]
    assert failed(checks(E2_PLACE_POS, linear=(0.05, 0, 0))) == ["linear_slow"]


def test_mixed_members_use_own_size_in_round_robin_order():
    pos = torch.tensor([[3.40, 0.80, PLATE_TOP]] * 4)
    sizes = ((0.2, 0.2, 0.16), (0.48, 0.38, 0.34), (0.6, 0.6, 0.5))
    per_env = placement.box_sizes_per_env(sizes, pos)
    assert torch.allclose(per_env, torch.tensor([sizes[i] for i in [0, 1, 2, 0]]))
    rolled = torch.tensor([quat(0, 90)] * 4)
    result = placement.box_placement_checks(
        pos, rolled, torch.zeros(4, 3), torch.zeros(4, 3), sizes, **conveyor_placement_params()
    )
    assert result["not_on_long_side"].tolist() == [True, False, True, True]


def make_env():
    import warp as wp

    wp.init()
    origins = torch.tensor([[0.0, 0.0, 0.0], [8.0, 8.0, 2.0]])
    data = SimpleNamespace(
        root_pos_w=wp.from_torch(torch.tensor([E2_PLACE_POS] * 2) + origins, dtype=wp.vec3),
        root_quat_w=wp.from_torch(torch.tensor([quat(2, 90)] * 2), dtype=wp.quat),
        root_lin_vel_w=wp.from_torch(torch.zeros(2, 3), dtype=wp.vec3),
        root_ang_vel_w=wp.from_torch(torch.zeros(2, 3), dtype=wp.vec3),
    )

    class Scene(dict):
        pass

    scene = Scene(box=SimpleNamespace(data=data))
    scene.env_origins = origins
    env = SimpleNamespace(
        scene=scene, step_dt=1 / 30, common_step_counter=0, episode_length_buf=torch.zeros(2, dtype=torch.long)
    )
    env.unwrapped = env
    return env


def success(env):
    return placement.object_settled_on_conveyor(env, SimpleNamespace(name="box"), SIZE, **E2_DECKS)


def test_runtime_wrapper_reset_event_and_extracted_recorder_calls():
    env = make_env()
    assert success(env).tolist() == [False, False]
    for step in range(1, 16):
        env.common_step_counter = step
        env.episode_length_buf[:] = step
        expected = [step >= 15] * 2
        assert success(env).tolist() == expected
        assert success(env).tolist() == expected
    placement.reset_conveyor_placement(env, torch.tensor([0]), "box")
    env.episode_length_buf[0] = 0
    assert success(env).tolist() == [False, True]
    env.common_step_counter += 1
    env.episode_length_buf += 1
    assert success(env).tolist() == [False, True]
    placement.reset_conveyor_placement(env, None, "box")
    assert success(env).tolist() == [False, False]


def test_reset_outside_inference_mode_after_an_inference_mode_step():
    """The hold is built inside the env's inference-mode step; env.reset() runs its reset event outside it."""
    env = make_env()
    with torch.inference_mode():
        env.common_step_counter = 1
        env.episode_length_buf[:] = 1
        success(env)
    for _ in range(2):
        placement.reset_conveyor_placement(env, torch.tensor([0]), "box")
        placement.reset_conveyor_placement(env, None, "box")
    assert sustained_count(env, placement.placement_hold_key("box")).tolist() == [0, 0]
    env.common_step_counter = 2
    env.episode_length_buf[:] = 1
    success(env)
    assert sustained_count(env, placement.placement_hold_key("box")).tolist() == [1, 1]


def test_logging_names_what_fails_when_it_changes_then_the_hold_and_the_success(capsys):
    """log_checks prints at the decks only: a failing check with its value, once per change; then holding, placed."""
    import warp as wp

    env = make_env()
    pos = wp.to_torch(env.scene["box"].data.root_pos_w)
    capsys.readouterr()

    def step(n, local_xyz):
        pos[:] = torch.tensor([local_xyz] * 2) + env.scene.env_origins
        env.common_step_counter = n
        env.episode_length_buf[:] = n
        placement.object_settled_on_conveyor(env, SimpleNamespace(name="box"), SIZE, **E2_DECKS, log_checks=True)
        placement.object_settled_on_conveyor(env, SimpleNamespace(name="box"), SIZE, **E2_DECKS, log_checks=True)
        return capsys.readouterr().out.splitlines()

    hovering = (3.34, 0.80, 0.841)
    lines = step(1, hovering)
    assert [line.split(" t=")[0] for line in lines] == ["[lift_to_conveyor] env 0", "[lift_to_conveyor] env 1"]
    assert all(line.endswith("not placed -- seated: lowest corner z 0.841 not in [0.785, 0.836]") for line in lines)
    assert step(2, hovering) == []
    placement.reset_conveyor_placement(env, torch.tensor([0]), "box")
    assert [line.split(" t=")[0] for line in step(3, hovering)] == ["[lift_to_conveyor] env 0"]
    assert step(4, (3.34, 0.80, 1.12)) == []
    assert all(line.endswith("every check passes; holding for 0.50 s") for line in step(5, E2_PLACE_POS))
    assert [step(n, E2_PLACE_POS) for n in range(6, 19)] == [[]] * 13
    lines = step(19, E2_PLACE_POS)
    assert len(lines) == 2 and all(line.endswith("placed: every check held 0.50 s") for line in lines)
    velocity = wp.to_torch(env.scene["box"].data.root_lin_vel_w)
    velocity[:, 0] = 0.1
    lines = step(20, E2_PLACE_POS)
    assert len(lines) == 2 and all(line.endswith("not placed -- linear_slow: 0.100 m/s >= 0.050") for line in lines)
    velocity[:, 0] = 0.0
    assert step(21, E2_PLACE_POS) == []


def test_workspace_uses_each_members_rotated_center():
    import warp as wp

    env = make_env()
    wp.to_torch(env.scene["box"].data.root_pos_w)[:] = torch.tensor([[2.0, 0.1, 0.75]] * 2) + env.scene.env_origins
    wp.to_torch(env.scene["box"].data.root_quat_w)[:] = torch.tensor([quat(0, -90)] * 2)
    result = placement.conveyor_box_outside_workspace(
        env, SimpleNamespace(name="box"), ((0.2, 0.2, 0.1), (0.2, 0.2, 0.4)), ((-0.2, 0.2), (5.2, 2.8))
    )
    assert result.tolist() == [True, False]


def test_actual_task_config_with_isaac_managers_and_extracted_success():
    pytest.importorskip("isaaclab")
    pytest.importorskip("isaaclab_arena")
    from isaaclab.managers import EventManager, TerminationManager
    from dexbench_arena.environments.manipulation.e2_logistics_environment import (
        DexBenchLogisticsCenterE2EnvironmentCfg,
        _make_lift_to_conveyor_task,
    )

    task = _make_lift_to_conveyor_task(None, DexBenchLogisticsCenterE2EnvironmentCfg(object_box="train_1"))
    cfg = task.termination_cfg.success
    assert cfg.params["box_sizes_xyz"] == SIZE
    assert cfg.params["angular_velocity_threshold_rad_s"] == 0.25
    assert cfg.params["velocity_threshold_m_s"] == 0.05
    env = make_env()
    env.scene["cage_l2_package"] = env.scene.pop("box")
    env.num_envs, env.device = 2, "cpu"
    env.sim = SimpleNamespace(is_playing=lambda: True)
    terminations = TerminationManager({"success": cfg}, env)
    events = EventManager(task.get_events_cfg(), env)
    for step in range(1, 16):
        env.common_step_counter = step
        env.episode_length_buf[:] = step
        assert terminations.compute().tolist() == [step >= 15] * 2
        assert cfg.func(env, **cfg.params).tolist() == [step >= 15] * 2
    events.apply("reset", env_ids=torch.tensor([0]), global_env_step_count=15)
    env.episode_length_buf[0] = 0
    assert terminations.compute().tolist() == [False, True]

    events.apply("reset", env_ids=torch.tensor([0, 1]), global_env_step_count=15)
    for step in range(16, 31):
        env.common_step_counter = step
        env.episode_length_buf[:] = step - 15
        assert cfg.func(env, **cfg.params).tolist() == [step >= 30] * 2


@pytest.mark.parametrize("group", ["train", "eval"])
def test_factory_preserves_object_set_member_dimensions(group):
    pytest.importorskip("isaaclab")
    pytest.importorskip("isaaclab_arena")
    from dexbench_arena.assets.e2_logistics_cell import E2_LIFT_OBJECT_SETS, E2_LIFT_OBJECTS_BY_NAME
    from dexbench_arena.environments.manipulation.e2_logistics_environment import (
        DexBenchLogisticsCenterE2EnvironmentCfg,
        _make_lift_to_conveyor_task,
    )

    task = _make_lift_to_conveyor_task(None, DexBenchLogisticsCenterE2EnvironmentCfg(object_set=group))
    expected = tuple(E2_LIFT_OBJECTS_BY_NAME[n].size_xyz_m for n in E2_LIFT_OBJECT_SETS[group])
    assert task.termination_cfg.success.params["box_sizes_xyz"] == expected
    assert task.termination_cfg.object_outside_workspace.params["box_sizes_xyz"] == expected


def test_geometry_requires_explicit_thresholds():
    import inspect

    parameters = inspect.signature(placement.box_placement_checks).parameters
    for name in conveyor_placement_params():
        assert parameters[name].default is inspect.Parameter.empty


def test_task_and_offline_checks_share_all_placement_thresholds():
    pytest.importorskip("isaaclab")
    pytest.importorskip("isaaclab_arena")
    from dexbench_arena.tasks.conveyor_placement_config import CONVEYOR_PLACE_HOLD_TIME_S
    from dexbench_arena.environments.manipulation.e2_logistics_environment import (
        DexBenchLogisticsCenterE2EnvironmentCfg,
        _make_lift_to_conveyor_task,
    )

    task = _make_lift_to_conveyor_task(None, DexBenchLogisticsCenterE2EnvironmentCfg(object_set="eval"))
    params = task.termination_cfg.success.params
    assert {key: params[key] for key in conveyor_placement_params()} == conveyor_placement_params()
    assert params["hold_time_s"] == CONVEYOR_PLACE_HOLD_TIME_S
