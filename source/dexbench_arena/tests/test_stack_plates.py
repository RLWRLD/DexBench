# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""The H1 stack predicate judges each plate where it lies, the drop term ends only the environment whose
plate fell, and replay keeps H1 metric changes visible to the operator."""

import warnings
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("isaaclab.managers", exc_type=ImportError)

import warp as wp  # noqa: E402
from isaaclab.utils.warp import ProxyArray  # noqa: E402

from dexbench_arena.tasks.predicates.geometry import any_object_root_past_plane  # noqa: E402
from dexbench_arena.tasks.predicates.plates import plates_seated_in_stack  # noqa: E402
from dexbench_arena.tasks.stack_plates import DexBenchStackPlatesTask  # noqa: E402

PLATE_NAMES = ["table_plate_00", "table_plate_01", "table_plate_02", "clearing_plate_00", "clearing_plate_01"]


def warp_data(**fields):
    """Asset data as Isaac Lab 3 hands it out: each field a ``ProxyArray`` over warp memory shared with its
    tensor, so an edit through ``.torch`` is what the predicate reads next."""
    return SimpleNamespace(**{name: ProxyArray(wp.from_torch(value)) for name, value in fields.items()})


@pytest.mark.parametrize(
    "body_x,body_z,expected",
    [
        (1.65, 0.85, False),
        (1.38, 0.85, True),
        (1.54, 0.91, True),
        (1.80, 0.85, True),
    ],
)
def test_clearance_follows_displaced_bowls(body_x, body_z, expected):
    class Scene(dict):
        env_origins = torch.zeros(1, 3)

    scene = Scene()
    for i in range(5):
        scene[f"bowl_{i}"] = SimpleNamespace(
            data=warp_data(
                root_pos_w=torch.tensor([[1.54, 1.9, 0.8 + i * 0.01078]]),
                root_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
                root_lin_vel_w=torch.zeros(1, 3),
            )
        )
    scene["robot"] = SimpleNamespace(data=warp_data(body_pos_w=torch.tensor([[[body_x, 1.9, body_z]]])))
    env = SimpleNamespace(unwrapped=SimpleNamespace(scene=scene))
    success = plates_seated_in_stack(
        env,
        [f"bowl_{i}" for i in range(5)],
        (1.48, 1.9),
        0.8,
        0.0811,
        5,
        0.03,
        0.8660254,
        0.01078,
        robot_asset_name="robot",
        clear_radius_m=0.1111,
        clear_height_m=0.03,
    )
    assert bool(success[0]) is expected


@pytest.mark.parametrize("failure", [None, "held", "high", "offset", "moving", "missing"])
def test_pose_settling_only_replaces_velocity_measurement(failure):
    class Scene(dict):
        env_origins = torch.zeros(1, 3)

    scene = Scene()
    for i in range(5):
        scene[f"bowl_{i}"] = SimpleNamespace(
            data=warp_data(
                root_pos_w=torch.tensor([[1.48, 1.9, 0.8 + i * 0.0294]]),
                root_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
                root_lin_vel_w=torch.tensor([[0.3, 0.0, 0.0]]),
            )
        )
    scene["robot"] = SimpleNamespace(data=warp_data(body_pos_w=torch.tensor([[[1.0, 1.9, 0.85]]])))
    owner = SimpleNamespace(scene=scene, step_dt=1 / 30, episode_length_buf=torch.tensor([0]))
    env = SimpleNamespace(unwrapped=owner)
    if failure == "held":
        scene["robot"].data.body_pos_w.torch[0, 0, 0] = 1.48
    if failure == "high":
        scene["bowl_4"].data.root_pos_w.torch[0, 2] += 0.02
    if failure == "offset":
        scene["bowl_4"].data.root_pos_w.torch[0, 0] += 0.04
    if failure == "missing":
        scene["bowl_4"].data.root_pos_w.torch[0, 0] += 1.0
    for i in range(12):
        owner.episode_length_buf[0] = i
        if failure == "moving":
            scene["bowl_4"].data.root_pos_w.torch[0, 0] = 1.48 + 0.004 * (i % 2)
        success = plates_seated_in_stack(
            env,
            [f"bowl_{j}" for j in range(5)],
            (1.48, 1.9),
            0.8,
            0.14,
            5,
            0.03,
            0.8660254,
            0.0294,
            robot_asset_name="robot",
            clear_radius_m=0.14,
            clear_height_m=0.03,
            settle_window_s=0.25,
        )
        if i < 8:
            assert not bool(success[0])
    assert bool(success[0]) == (failure is None)


def test_vectorized_destination_regions_use_separate_reset_targets():
    class Scene(dict):
        env_origins = torch.tensor([[0.0, 0.0, 0.0], [8.0, 0.0, 0.0]])

    scene = Scene()
    centers = torch.tensor([[1.48, 1.9], [1.53, 1.95]])
    for i in range(5):
        position = torch.cat([centers, torch.full((2, 1), 0.8 + i * 0.0294)], dim=1) + scene.env_origins
        scene[f"bowl_{i}"] = SimpleNamespace(
            data=warp_data(
                root_pos_w=position,
                root_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 2),
                root_lin_vel_w=torch.zeros(2, 3),
            )
        )
    env = SimpleNamespace(unwrapped=SimpleNamespace(scene=scene))
    args = (env, [f"bowl_{i}" for i in range(5)], centers, 0.8, 0.04, 5, 0.03, 0.8660254, 0.0294)
    assert plates_seated_in_stack(*args).tolist() == [True, True]
    for i in range(5):
        scene[f"bowl_{i}"].data.root_pos_w.torch[1, 0] += 0.06
    assert plates_seated_in_stack(*args).tolist() == [True, False]


@pytest.mark.parametrize("aligned_feature,legacy_expected,rim_expected", [
    ("rims", False, True),
    ("roots", True, False),
    ("upright", True, True),
])
def test_rim_alignment_uses_rotated_rim_centers(aligned_feature, legacy_expected, rim_expected):
    math = pytest.importorskip("isaaclab.utils.math", exc_type=ImportError)

    class Scene(dict):
        env_origins = torch.zeros(1, 3)

    scene = Scene()
    for i in range(5):
        angle = torch.tensor([0.35 if i == 2 and aligned_feature != "upright" else 0.0])
        quat = math.quat_from_euler_xyz(torch.zeros(1), angle, torch.zeros(1))
        up = math.matrix_from_quat(quat)[0, :, 2]
        position = torch.tensor([[1.48, 1.9, 0.8 + i * 0.0294]])
        if aligned_feature == "rims":
            position[0, :2] -= 0.1 * up[:2]
        scene[f"bowl_{i}"] = SimpleNamespace(data=warp_data(
            root_pos_w=position, root_quat_w=quat, root_lin_vel_w=torch.zeros(1, 3),
        ))
    env = SimpleNamespace(unwrapped=SimpleNamespace(scene=scene))
    args = (env, [f"bowl_{i}" for i in range(5)], (1.48, 1.9), 0.8, 0.11,
            5, 0.03, 0.8660254, 0.0294)
    assert bool(plates_seated_in_stack(*args)[0]) is legacy_expected
    assert bool(plates_seated_in_stack(*args, rim_offset_xyz=(0, 0, 0.1))[0]) is rim_expected


def test_the_destination_region_follows_the_destination_stacks_draw():
    """The H1 eval cells' success: membership is judged around the destination stack where its
    dest_stack_pose draw put it, heard through Arena's sample listener."""
    import copy

    from dexbench_arena.variations.object_pose_variation import PlanarOffsetListener

    class Scene(dict):
        env_origins = torch.zeros(2, 3)

    scene = Scene()
    for i in range(5):
        position = torch.tensor([[1.48 + 0.06, 1.9, 0.8 + i * 0.0294]] * 2)
        scene[f"bowl_{i}"] = SimpleNamespace(
            data=warp_data(
                root_pos_w=position,
                root_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 2),
                root_lin_vel_w=torch.zeros(2, 3),
            )
        )
    env = SimpleNamespace(unwrapped=SimpleNamespace(scene=scene))
    args = (env, [f"bowl_{i}" for i in range(5)], (1.48, 1.9), 0.8, 0.04, 5, 0.03, 0.8660254, 0.0294)
    listener = PlanarOffsetListener()
    assert plates_seated_in_stack(*args, stack_center_offset=listener).tolist() == [False, False]
    held = copy.deepcopy({"stack_center_offset": listener})["stack_center_offset"]
    listener(torch.tensor([[0.06, 0.0, 0.0]]), torch.tensor([0]))
    assert plates_seated_in_stack(*args, stack_center_offset=held).tolist() == [True, False]
    assert plates_seated_in_stack(*args).tolist() == [False, False]

    math = pytest.importorskip("isaaclab.utils.math", exc_type=ImportError)
    quat = math.quat_from_euler_xyz(torch.zeros(2), torch.full((2,), 0.35), torch.zeros(2))
    scene["bowl_2"].data.root_quat_w.torch[:] = quat
    scene["bowl_2"].data.root_pos_w.torch[:, :2] -= 0.1 * math.matrix_from_quat(quat)[:, :2, 2]
    assert plates_seated_in_stack(*args, stack_center_offset=held).tolist() == [False, False]
    rim_aligned = plates_seated_in_stack(*args, stack_center_offset=held, rim_offset_xyz=(0, 0, 0.1))
    assert rim_aligned.tolist() == [True, False]


@pytest.fixture
def drop_term():
    task = DexBenchStackPlatesTask(
        plate_names=PLATE_NAMES,
        stack_center_xy=(1.48, 1.9),
        surface_top_z_m=0.8,
        region_radius_m=0.14,
        expected_count=5,
        max_rim_offset_mm=30,
        max_lean_deg=30,
        nest_pitch_m=0.0294,
        plate_dropped_min_z_m=0.65,
        max_seat_height_error_mm=10,
    )
    term = task.termination_cfg.plate_dropped
    assert term.func is any_object_root_past_plane
    assert term.params["object_names"] == PLATE_NAMES
    return term


@pytest.mark.parametrize("dropped_plate", PLATE_NAMES)
def test_any_source_or_destination_drop_terminates_only_its_environment(drop_term, dropped_plate):
    class Scene(dict):
        env_origins = torch.tensor([[0.0, 0.0, 0.0], [8.0, 0.0, 2.0], [0.0, 8.0, -1.0]])

    scene = Scene()
    for name in PLATE_NAMES:
        position = torch.tensor([[1.48, 1.9, 0.8]] * 3) + scene.env_origins
        scene[name] = SimpleNamespace(data=warp_data(root_pos_w=position))
    env = SimpleNamespace(unwrapped=SimpleNamespace(scene=scene, num_envs=3, device="cpu"))

    def evaluate():
        result = drop_term.func(env, **drop_term.params)
        assert result.dtype == torch.bool
        assert result.shape == (3,)
        return result.tolist()

    assert evaluate() == [False, False, False]
    position = scene[dropped_plate].data.root_pos_w.torch
    position[1, 2] = scene.env_origins[1, 2] + 0.64
    assert evaluate() == [False, True, False]
    position[1, 2] = scene.env_origins[1, 2] + 0.8
    position[0, 2] = 0.65
    assert evaluate() == [False, False, False]


@pytest.mark.parametrize("source,current", [("rim", "root"), ("root", "rim"), ("rim", "rim")])
@pytest.mark.parametrize("location", ["recording", "replay", "predicate"])
def test_replay_warns_only_when_the_recorded_alignment_changes(source, current, location):
    pytest.importorskip("isaaclab.managers", exc_type=ImportError)
    from dexbench_arena.tasks.stack_plates import DexBenchStackPlatesTask

    cfg = {"stack_alignment_point": source}
    offset = (0, 0, .1 if source == "rim" else 0)
    metadata = {
        "recording": {"dexbench_recording": {"environment_cfg": cfg}},
        "replay": {"dexbench_rerecord": {"runtime": {"environment_cfg": cfg}}},
        "predicate": {"dexbench_success_evaluation": {"parameters": {"rim_offset_xyz": offset}}},
    }[location]
    task = DexBenchStackPlatesTask.__new__(DexBenchStackPlatesTask)
    task.termination_cfg = SimpleNamespace(success=SimpleNamespace(
        params={"rim_offset_xyz": (0, 0, .1 if current == "rim" else 0)}))
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        task.validate_replay_dataset("demo.hdf5", metadata, "robot")
    assert len(recorded) == int(source != current)
    if recorded:
        assert f"source={source}, replay={current}" in str(recorded[0].message)
        assert f"--stack_alignment_point {source}" in str(recorded[0].message)
    assert task.termination_cfg.success.params["rim_offset_xyz"][2] == (.1 if current == "rim" else 0)
