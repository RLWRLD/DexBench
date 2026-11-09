# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""DB-M2's seat window, drop and neighbor terms, on poses placed along each connector's insertion axis."""

import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("isaaclab_arena", reason="needs isaaclab_arena on the path")

from isaaclab.utils.math import quat_from_angle_axis  # noqa: E402

from dexbench_arena.layouts.db_m2_layout import (  # noqa: E402
    CONNECTOR_SITES,
    NEIGHBOR_MAX_DISPLACEMENT_M,
    PLUG_DROP_M,
    SEAT_MAX_DEPTH_M,
    SEAT_MAX_LATERAL_M,
    SEAT_MAX_TILT_DEG,
    site_in_env,
)
from dexbench_arena.tasks.predicates.connector_insertion import (  # noqa: E402
    hand_plug_force,
    plug_along_axis,
    plug_below,
    plug_displaced,
    plug_in_seat_window,
)


def _env(position, tilt_deg=0.0, tilt_axis=(1.0, 0.0, 0.0)):
    quat = quat_from_angle_axis(
        torch.tensor([math.radians(tilt_deg)], dtype=torch.float64), torch.tensor([tilt_axis], dtype=torch.float64)
    )
    pose = torch.cat((torch.tensor([position], dtype=torch.float64), quat), -1)
    return SimpleNamespace(num_envs=1, arena_world=SimpleNamespace(get_pose_e={"plug": pose}.__getitem__))


def _along(site, depth_m, lateral_m=0.0):
    """A point ``depth_m`` out along the pull-out axis from the seat and ``lateral_m`` off it (env frame)."""
    seat = site_in_env(site.seated_xyz)
    axis = site.pull_out_axis
    side = (axis[1], -axis[0], 0.0)
    norm = math.hypot(*side)
    return tuple(s + depth_m * a + lateral_m * p / norm for s, a, p in zip(seat, axis, side))


def _seated(site, position, tilt_deg=0.0):
    env = _env(position, tilt_deg)
    return bool(
        plug_in_seat_window(
            env, "plug", site_in_env(site.seated_xyz), site.pull_out_axis, site.seated_quat_xyzw,
            SEAT_MAX_DEPTH_M, SEAT_MAX_LATERAL_M, SEAT_MAX_TILT_DEG,
        )[0]
    )


@pytest.mark.parametrize("key", ["S2"])
def test_seat_window(key):
    site = CONNECTOR_SITES[key]
    assert _seated(site, _along(site, 0.0))
    assert _seated(site, _along(site, 0.00034))
    assert _seated(site, _along(site, -0.0003, 0.0001))
    assert _seated(site, _along(site, 0.00095, 0.00045))
    assert not _seated(site, _along(site, 0.0011))
    assert not _seated(site, _along(site, 0.0, 0.0006))
    assert not _seated(site, _along(site, -0.0011))
    assert not _seated(site, _along(site, site.stroke_m))
    assert _seated(site, _along(site, 0.0), tilt_deg=SEAT_MAX_TILT_DEG * 0.9)
    assert not _seated(site, _along(site, 0.0), tilt_deg=SEAT_MAX_TILT_DEG * 1.1)


@pytest.mark.parametrize("key", ["S2"])
def test_half_way_stage(key):
    site = CONNECTOR_SITES[key]
    seat = site_in_env(site.seated_xyz)

    def stage(position):
        return bool(plug_along_axis(_env(position), "plug", seat, site.pull_out_axis, site.seated_quat_xyzw,
                                    site.stroke_m / 2, 0.0015, 5.0)[0])

    assert not stage(_along(site, site.stroke_m))
    assert stage(_along(site, site.stroke_m / 2 - 0.0005))
    assert not stage(_along(site, site.stroke_m / 2 - 0.0005, 0.002))


def test_drop_and_neighbor():
    site = CONNECTOR_SITES["S2"]
    seat = site_in_env(site.seated_xyz)
    assert not bool(plug_below(_env(seat), "plug", seat[2] - PLUG_DROP_M)[0])
    assert bool(plug_below(_env((seat[0], seat[1], seat[2] - PLUG_DROP_M - 0.01)), "plug", seat[2] - PLUG_DROP_M)[0])
    assert not bool(plug_displaced(_env(_along(site, 0.002)), "plug", seat, NEIGHBOR_MAX_DISPLACEMENT_M)[0])
    assert bool(plug_displaced(_env(_along(site, 0.004)), "plug", seat, NEIGHBOR_MAX_DISPLACEMENT_M)[0])


def _sensor_env(left_n, right_n=None):
    """An env whose hand sensors report one contact of ``left_n`` / ``right_n`` newtons on the plug (None: no sensor)."""
    sensors = {}
    for name, force in (("left_cs", left_n), ("right_cs", right_n)):
        if force is not None:
            matrix = torch.zeros(1, 3, 1, 3)
            matrix[0, 1, 0] = torch.tensor([0.6, 0.0, 0.8]) * force
            sensors[name] = SimpleNamespace(data=SimpleNamespace(force_matrix_w=matrix))
    return SimpleNamespace(num_envs=1, device="cpu", scene=SimpleNamespace(sensors=sensors))


def test_hand_force_reads_each_sensor_and_a_missing_one_as_zero():
    forces = hand_plug_force(_sensor_env(2.0, None), ("left_cs", "right_cs"))
    assert forces.shape == (1, 2)
    assert float(forces[0, 0]) == pytest.approx(2.0) and float(forces[0, 1]) == 0.0


def test_the_task_scores_no_force_and_observes_both_hands():
    from dexbench_arena.tasks.insert_connector import HAND_SENSORS, InsertConnectorObservationsCfg, InsertConnectorTerminationsCfg

    assert set(InsertConnectorTerminationsCfg.__dataclass_fields__) == {"success", "plug_dropped", "neighbor_unseated"}
    term = InsertConnectorObservationsCfg().task_obs.hand_plug_force
    assert term.func is hand_plug_force and tuple(term.params["sensor_names"]) == HAND_SENSORS == ("left_cs", "right_cs")
