# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Unit coverage for the shared recorded planar-pose variation factory."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from dexbench_arena.variations.object_pose_variation import make_planar_pose_variation


def test_planar_pose_variation_builds_symmetric_recorded_bounds():
    variation = make_planar_pose_variation(
        "robot",
        name="base_pose",
        xy_half_extent_m=0.05,
        yaw_half_extent_rad=0.25,
    )

    assert variation.asset_name == "robot"
    assert variation.name == "base_pose"
    assert variation.enabled
    assert variation.cfg.sampler_cfg.low == pytest.approx([-0.05, -0.05, -0.25])
    assert variation.cfg.sampler_cfg.high == pytest.approx([0.05, 0.05, 0.25])


def test_planar_pose_variation_centres_the_range_on_a_fixed_offset():
    """A GA-S fixed point: the jug held 3 cm toward -x and turned 15 deg, jittered only in yaw +-10 deg."""
    variation = make_planar_pose_variation(
        "idealisk_jug",
        name="pour_pose",
        xy_offset_m=(-0.03, 0.0),
        yaw_half_extent_rad=0.1745,
        yaw_offset_rad=0.2618,
    )

    assert variation.enabled
    assert variation.cfg.sampler_cfg.low == pytest.approx([-0.03, 0.0, 0.2618 - 0.1745])
    assert variation.cfg.sampler_cfg.high == pytest.approx([-0.03, 0.0, 0.2618 + 0.1745])


def test_a_yaw_offset_alone_enables_the_variation():
    variation = make_planar_pose_variation("idealisk_jug", yaw_offset_rad=-0.349)

    assert variation.enabled
    assert variation.cfg.sampler_cfg.low == pytest.approx([0.0, 0.0, -0.349])
    assert variation.cfg.sampler_cfg.high == pytest.approx([0.0, 0.0, -0.349])


@pytest.mark.parametrize("yaw_offset", [0.0, 0.4])
def test_asymmetric_bounds_keep_xy_and_yaw_offsets(yaw_offset):
    variation = make_planar_pose_variation(
        "robot", low=(-0.03, -0.01, -0.2), high=(0.01, 0.04, 0.1),
        xy_offset_m=(0.02, -0.03), yaw_offset_rad=yaw_offset,
    )

    assert variation.enabled
    assert variation.cfg.sampler_cfg.low == pytest.approx([-0.01, -0.04, -0.2 + yaw_offset])
    assert variation.cfg.sampler_cfg.high == pytest.approx([0.03, 0.01, 0.1 + yaw_offset])


def test_zero_planar_pose_variation_stays_registered_but_disabled():
    variation = make_planar_pose_variation("target")

    assert not variation.enabled
    assert variation.cfg.sampler_cfg.low == pytest.approx([0.0, 0.0, 0.0])
    assert variation.cfg.sampler_cfg.high == pytest.approx([0.0, 0.0, 0.0])


def test_base_condition_replay_applies_yaw_even_when_live_sampler_has_zero_yaw():
    """M2 enables base XY through CLI, then supplies all three values through replay."""
    pytest.importorskip("warp")
    from isaaclab.utils.math import quat_from_euler_xyz, quat_mul
    from isaaclab_arena.variations.variation_recorder import VariationRecorder

    variation = make_planar_pose_variation("robot", name="base_pose", xy_half_extent_m=.015)
    requested = torch.tensor([[.015, -.015, .026179939], [-.010, .010, -.017453293]])
    variation.set_replay_sampler(lambda count, env_ids: requested)
    scene = _FakeScene()
    default = torch.tensor([[.2, -.4, .1, 0., 0., .70710678, .70710678]] * 2)
    scene["robot"] = _FakeMember(default)
    env = SimpleNamespace(scene=scene, get_episode_index=lambda env_id: 0)
    recorder = VariationRecorder()
    recorder.attach({"dexbench_rby1_wuji2_joint": [variation]})
    recorder.bind_env(env)
    _, event = variation.build_event_cfg()
    event.func(event, env)(env, torch.tensor([0, 1]), **event.params)
    expected_position = default[:, :3] + scene.env_origins
    expected_position[:, :2] += requested[:, :2]
    expected_yaw = quat_from_euler_xyz(torch.zeros(2), torch.zeros(2), requested[:, 2])
    torch.testing.assert_close(scene["robot"].pose[:, :3], expected_position)
    torch.testing.assert_close(scene["robot"].pose[:, 3:], quat_mul(expected_yaw, default[:, 3:]))
    record = recorder["dexbench_rby1_wuji2_joint.base_pose"]
    torch.testing.assert_close(torch.stack([record.sample_for_episode(i, 0) for i in range(2)]), requested)


@pytest.mark.parametrize(
    ("xy_half_extent_m", "yaw_half_extent_rad"),
    [(-0.01, 0.0), (0.0, -0.01)],
)
def test_planar_pose_variation_rejects_negative_half_extents(
    xy_half_extent_m: float,
    yaw_half_extent_rad: float,
):
    with pytest.raises(ValueError, match="must be non-negative"):
        make_planar_pose_variation(
            "target",
            xy_half_extent_m=xy_half_extent_m,
            yaw_half_extent_rad=yaw_half_extent_rad,
        )


def test_stack_radius_check_refuses_a_mixed_corner_outside_the_region():
    """X and Y are drawn independently, so (low x, high y) is a reachable draw:
    low=(-0.10, 0), high=(0, 0.10) puts a plate 0.141 m out, past a 0.11 m region."""
    from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg

    from dexbench_arena.variations.object_pose_variation import (
        ObjectPoseVariation,
        ObjectPoseVariationCfg,
        PlanarPoseDomain,
    )

    def variation(low, high):
        return ObjectPoseVariation(
            "plate_00",
            cfg=ObjectPoseVariationCfg(sampler_cfg=UniformSamplerCfg(low=low, high=high)),
            domain=PlanarPoseDomain(max_xy_radius_m=0.11),
            member_names=["plate_00", "plate_01"],
        )

    with pytest.raises(ValueError, match="141 mm"):
        variation([-0.10, 0.0, 0.0], [0.0, 0.10, 0.0]).validate_range()
    variation([-0.07, 0.0, 0.0], [0.0, 0.07, 0.0]).validate_range()


def test_the_domain_is_checked_against_the_range_after_hydra_overrides():
    """The factory's range is only the default distribution: an override may
    widen or skew it, and only a range that can leave the physical domain is
    refused -- when the event is built, after the override has been applied."""
    from isaaclab_arena.variations.variations_hydra import apply_overrides

    from dexbench_arena.variations.object_pose_variation import PlanarPoseDomain

    def overridden(low, high):
        variation = make_planar_pose_variation(
            "target", xy_half_extent_m=0.01, domain=PlanarPoseDomain(max_abs_xy_m=(0.04, 0.02))
        )
        apply_overrides(
            {"target": [variation]},
            [f"target.pose.sampler_cfg.low={low}", f"target.pose.sampler_cfg.high={high}"],
        )
        return variation

    name, _ = overridden([-0.04, -0.005, -1.0], [0.0, 0.02, 1.0]).build_event_cfg()
    assert name == "target_pose_variation"
    with pytest.raises(ValueError, match=r"\|dx\| 50 mm"):
        overridden([-0.05, -0.01, 0.0], [0.01, 0.01, 0.0]).build_event_cfg()


class _FakeMember:
    """A rigid object as the reset term sees it: a default pose and the two indexed writers."""

    device = "cpu"

    def __init__(self, default_root_pose: torch.Tensor):
        import warp as wp

        wp.init()
        self.data = SimpleNamespace(
            default_root_pose=wp.from_torch(default_root_pose),
            default_root_vel=wp.from_torch(torch.ones(len(default_root_pose), 6)),
        )
        self.pose = self.velocity = None

    def write_root_pose_to_sim_index(self, root_pose, env_ids):
        self.pose = root_pose.clone()

    def write_root_velocity_to_sim_index(self, root_velocity, env_ids):
        self.velocity = root_velocity.clone()


class _FakeScene(dict):
    num_envs = 2
    env_origins = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])


def test_one_draw_per_env_moves_every_member_and_is_recorded_once():
    """A nested stack stays a stack only while its members share a pose: the
    reset event draws ONE [dx, dy, dyaw] per environment, records it under the
    registering asset, and applies it to every member."""
    pytest.importorskip("warp")
    from isaaclab.utils.math import quat_from_euler_xyz
    from isaaclab_arena.variations.variation_recorder import VariationRecorder

    names = ["plate_00", "plate_01", "plate_02"]
    variation = make_planar_pose_variation(
        "plate_00", xy_half_extent_m=0.015, yaw_half_extent_rad=0.26, name="dest_stack_pose", member_names=names
    )
    scene = _FakeScene()
    for height, name in enumerate(names):
        pose = torch.tensor([[0.5, 0.2, 0.8 + 0.01 * height, 0.0, 0.0, 0.0, 1.0]] * scene.num_envs)
        scene[name] = _FakeMember(pose)
    env = SimpleNamespace(scene=scene, get_episode_index=lambda env_id: 0)
    recorder = VariationRecorder()
    recorder.attach({"plate_00": [variation]})
    recorder.bind_env(env)

    event_name, event_cfg = variation.build_event_cfg()
    assert event_name == "plate_00_dest_stack_pose_variation"
    term = event_cfg.func(event_cfg, env)
    term(env, torch.tensor([0, 1]), **event_cfg.params)

    record = recorder["plate_00.dest_stack_pose"]
    draws = torch.stack([record.sample_for_episode(env_id, 0) for env_id in (0, 1)])
    assert draws.shape == (2, 3)
    assert not torch.equal(draws[0], draws[1]), "one draw per environment"
    expected_quat = quat_from_euler_xyz(torch.zeros(2), torch.zeros(2), draws[:, 2])
    for height, name in enumerate(names):
        member = scene[name]
        torch.testing.assert_close(
            member.pose[:, 0:2], scene.env_origins[:, 0:2] + torch.tensor([0.5, 0.2]) + draws[:, 0:2]
        )
        torch.testing.assert_close(member.pose[:, 2], torch.full((2,), 0.8 + 0.01 * height))
        torch.testing.assert_close(member.pose[:, 3:7], expected_quat)
        assert torch.count_nonzero(member.velocity) == 0


@pytest.mark.parametrize(
    "target", ["dexbench_retail_e1_fc_r3t1", "dexbench_retail_e1_fc_r3t3", "dexbench_retail_e1_fc_r3e1"]
)
def test_yaw_turns_the_r3_cartons_in_the_table_plane(target):
    """dyaw is a turn about world +Z for every asset. R3T1 stands on its local +Z
    (its default is a pure yaw), so a local-Z turn gives the same pose; R3T3 stands on its
    local +Y, so a local-Z turn would tip it over -- it must stay standing."""
    pytest.importorskip("warp")
    from isaaclab.utils.math import matrix_from_quat, quat_from_euler_xyz, quat_mul

    from dexbench_arena.layouts import e1_retail_fc_layout as layout

    default_xyzw = torch.tensor(layout.FC_260821_R3_TARGET_SKU_ROTATION_XYZW[target])
    local_up = torch.tensor(layout.FC_260821_R3_TARGET_SKU_LOCAL_UP_XYZ[target])

    dyaw = 0.3
    variation = make_planar_pose_variation("target", yaw_half_extent_rad=dyaw)
    variation.cfg.sampler_cfg.low = [0.0, 0.0, dyaw]
    variation.apply_cfg(variation.cfg)
    scene = _FakeScene()
    scene["target"] = _FakeMember(torch.cat([torch.tensor([0.5, 0.2, 0.9]), default_xyzw]).repeat(scene.num_envs, 1))
    env = SimpleNamespace(scene=scene, get_episode_index=lambda env_id: 0)
    _, event_cfg = variation.build_event_cfg()
    event_cfg.func(event_cfg, env)(env, torch.tensor([0, 1]), **event_cfg.params)
    quat = scene["target"].pose[:, 3:7]

    yaw = quat_from_euler_xyz(torch.zeros(2), torch.zeros(2), torch.full((2,), dyaw))
    torch.testing.assert_close(matrix_from_quat(quat), matrix_from_quat(quat_mul(yaw, default_xyzw.repeat(2, 1))))
    torch.testing.assert_close(matrix_from_quat(quat) @ local_up, torch.tensor([[0.0, 0.0, 1.0]] * 2))
    if target.endswith("r3t1"):
        torch.testing.assert_close(quat, quat_mul(default_xyzw.repeat(2, 1), yaw))


def test_on_applied_follows_the_written_members_with_the_draw():
    """``on_applied`` runs in the same reset event once every member is written, with the environments and the one
    draw they moved by (DB-M2's cable follows its plug so); without it the event's params carry no ``on_applied``."""
    pytest.importorskip("warp")

    calls = []

    def follow(env, env_ids, sample):
        calls.append((env_ids.clone(), sample.clone(), scene["target"].pose.clone()))

    variation = make_planar_pose_variation("target", xy_half_extent_m=0.01, yaw_half_extent_rad=0.2)
    assert "on_applied" not in variation.build_event_cfg()[1].params
    variation.on_applied = follow
    scene = _FakeScene()
    scene["target"] = _FakeMember(torch.tensor([[0.5, 0.2, 0.9, 0.0, 0.0, 0.0, 1.0]] * scene.num_envs))
    env = SimpleNamespace(scene=scene, get_episode_index=lambda env_id: 0)
    _, event_cfg = variation.build_event_cfg()
    event_cfg.func(event_cfg, env)(env, torch.tensor([0, 1]), **event_cfg.params)

    (env_ids, sample, pose_then), = calls
    assert env_ids.tolist() == [0, 1] and sample.shape == (2, 3)
    torch.testing.assert_close(pose_then, scene["target"].pose)
    torch.testing.assert_close(pose_then[:, :2], scene.env_origins[:, :2] + torch.tensor([0.5, 0.2]) + sample[:, :2])
