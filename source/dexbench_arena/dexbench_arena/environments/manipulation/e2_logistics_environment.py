# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""DexBench logistics-centre E2 Arena scene (PhysicsSim Lab delivery)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace

import argparse
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentFactory

from dexbench_arena import cli
from dexbench_arena.contact_materials import WUJI_GRASP_CONTACT_MATERIAL
from dexbench_arena.embodiments.lookup import start_poses
from dexbench_arena.embodiments.rby1m.constants import rby1m_torso_z_fold
from dexbench_arena.mimic_plugin import mimic_task_class, require_datagen
from dexbench_arena.tasks.conveyor_placement_config import (
    CONVEYOR_PLACE_TOP_Z_RANGE_M,
    CONVEYOR_PLACE_XY_RECTS_M,
    CONVEYOR_PLACE_BOTTOM_HEIGHT_TOLERANCE_M,
    CONVEYOR_PLACE_MAX_FACE_TILT_DEG,
    CONVEYOR_PLACE_LINEAR_SPEED_M_S,
    CONVEYOR_PLACE_ANGULAR_SPEED_RAD_S,
    CONVEYOR_PLACE_HOLD_TIME_S,
)
from dexbench_arena.assets.e2_logistics_cell import (
    E2_DEFAULT_LIFT_OBJECT,
    E2_LIFT_OBJECT_AXIS,
    E2_LIFT_OBJECT_SETS,
    E2_LIFT_OBJECTS_BY_NAME,
    E2_BENCH_STATION_REFERENCES,
    E2_INVOICE_OBJECT_AXIS,
    E2_PACK_INVOICE_BOX_SCALE_XYZ,
    E2LiftObject,
    e2_cage_box_asset,
)
from dexbench_arena.layouts.e2_logistics_layout import (
    E2_BELT_ITEM_PLACEMENTS,
    E2_LIGHTS,
    E2_OBJECT_SUPPORT_MARGIN_M,
    E2_RENDER_SETTINGS,
    E2CageLayout,
    E2CageLayoutParams,
    e2_cage_box_entity_name,
    e2_cage_box_park_position,
    e2_cage_layout,
    e2_cage_stack_scales,
    e2_footprint_half_extents_xy,
    e2_item_placements,
)
from dexbench_arena.deformables.pack_invoice_scene_randomization import PAPER_PHYSICS_ARGS
from dexbench_arena.environments.manipulation.scene_environment_base import (
    TASKS,
    DexBenchEnvironmentCfg,
    DexBenchSceneEnvironmentBase,
    _add_argument_if_missing,
)

if TYPE_CHECKING:
    from isaaclab_arena.scene.scene import Scene


ROBOT_BASE_ANCHORS_XYZ = {
    "A1_CAGE": (1.46, 1.6922, 0.0),
    "A3_BENCH": (4.72, 2.93, 0.0),
}

ANCHOR_FACING_TARGETS_XY = {
    "A1_CAGE": (0.46, 1.6922),
    "A3_BENCH": (4.72, 3.93),
}




def lift_object_front_clearance_m(lift_object: E2LiftObject) -> float:
    """How far behind the tier's front edge ``lift_object`` is set."""

    half_long, half_depth = (extent / 2.0 for extent in lift_object.footprint_xy_m)
    dx_cap, _dy_cap, dyaw_cap = CAGE_L2_PACKAGE_POSE_VARIATION_MAX_ABS
    turned_depth = e2_footprint_half_extents_xy(dyaw_cap, half_long, half_depth)[1]
    return E2_OBJECT_SUPPORT_MARGIN_M + dx_cap + (turned_depth - half_depth)


def lift_object_pose(
    layout: E2CageLayout, lift_object: E2LiftObject
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Where ``lift_object`` is spawned on ``layout``: origin (bottom-face centre) and rotation."""

    return layout.object_pose(lift_object.footprint_xy_m, lift_object_front_clearance_m(lift_object))


CAGE_L2_PACKAGE_POSE_VARIATION_MAX_ABS = (0.03, 0.10, 0.15)


def _cage_l2_package_room(layout: E2CageLayout, lift_object: E2LiftObject, dyaw: float) -> tuple[float, float, float, float]:
    """(-x, +x, -y, +y): how far ``lift_object`` turned by ``dyaw`` can move on ``layout`` before its edge meets the margin."""

    from isaaclab_arena.utils.yaw import yaw_from_quat_xyzw

    (min_x, min_y), (max_x, max_y) = layout.support_xy_bounds
    (centre_x, centre_y, _), rotation = lift_object_pose(layout, lift_object)
    half_long, half_depth = (extent / 2.0 for extent in lift_object.footprint_xy_m)
    margin = E2_OBJECT_SUPPORT_MARGIN_M
    extent_x, extent_y = e2_footprint_half_extents_xy(yaw_from_quat_xyzw(rotation) + dyaw, half_long, half_depth)
    return (
        centre_x - extent_x - (min_x + margin),
        (max_x - margin) - (centre_x + extent_x),
        centre_y - extent_y - (min_y + margin),
        (max_y - margin) - (centre_y + extent_y),
    )


def cage_l2_package_pose_variation_range(
    layout: E2CageLayout, lift_object: E2LiftObject
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """``(low, high)`` of ``lift_object``'s ``[dx, dy, dyaw]`` draw on ``layout``."""

    dx_cap, dy_cap, dyaw_cap = CAGE_L2_PACKAGE_POSE_VARIATION_MAX_ABS

    def room(dyaw: float) -> tuple[float, float, float, float]:
        return _cage_l2_package_room(layout, lift_object, dyaw)

    dyaw = dyaw_cap
    if min(room(dyaw)) < 0.0:
        low_yaw, high_yaw = 0.0, dyaw_cap
        for _ in range(40):
            mid = (low_yaw + high_yaw) / 2.0
            if min(room(mid)) >= 0.0:
                low_yaw = mid
            else:
                high_yaw = mid
        dyaw = low_yaw
    down_x, up_x, down_y, up_y = (max(0.0, value) for value in room(dyaw))
    low = (0.0 - round(min(dx_cap, down_x), 4), 0.0 - round(min(dy_cap, down_y), 4), 0.0 - round(dyaw, 4))
    high = (round(min(dx_cap, up_x), 4), round(min(dy_cap, up_y), 4), round(dyaw, 4))
    return low, high


@dataclass(frozen=True)
class CageL2PackagePoseDomain:
    """The object's physical placement domain: wholly on the tier top, margin kept, for every set and object spawned."""

    layouts: tuple[E2CageLayout, ...]
    lift_objects: tuple[E2LiftObject, ...]
    reason: str = "the object would overhang the tier top past E2_OBJECT_SUPPORT_MARGIN_M"

    _TOLERANCE_M = 1e-4
    _YAW_SAMPLES = 33

    def violation(self, low: Sequence[float], high: Sequence[float]) -> str | None:
        low_yaw, high_yaw = float(low[2]), float(high[2])
        yaws = [low_yaw + (high_yaw - low_yaw) * i / (self._YAW_SAMPLES - 1) for i in range(self._YAW_SAMPLES)]
        for layout in self.layouts:
            for lift_object in self.lift_objects:
                down_x, up_x, down_y, up_y = (
                    min(values)
                    for values in zip(*(_cage_l2_package_room(layout, lift_object, yaw) for yaw in yaws))
                )
                need = (-float(low[0]), float(high[0]), -float(low[1]), float(high[1]))
                if any(n > r + self._TOLERANCE_M for n, r in zip(need, (down_x, up_x, down_y, up_y))):
                    return (
                        f"'{lift_object.name}' on the box_scale {layout.params.box_scale} set has room "
                        f"x [{-down_x:+.4f}, {up_x:+.4f}] y [{-down_y:+.4f}, {up_y:+.4f}] m at that yaw"
                    )
        return None

def cage_l2_package_pose_variation_range_for_sets(
    layouts: Sequence[E2CageLayout], lift_objects: Sequence[E2LiftObject]
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """The draw range that fits every ``layouts`` x ``lift_objects`` pairing: the elementwise tightest."""

    assert layouts and lift_objects, "at least one layout and one object"
    ranges = [cage_l2_package_pose_variation_range(layout, obj) for layout in layouts for obj in lift_objects]
    low = tuple(max(values) for values in zip(*(r[0] for r in ranges)))
    high = tuple(min(values) for values in zip(*(r[1] for r in ranges)))
    return low, high


def cage_l2_package_pose_variation(
    asset_name: str, layouts: Sequence[E2CageLayout], lift_objects: Sequence[E2LiftObject]
):
    """The object's ``pose`` variation: disabled, ranged to fit every layout and object spawned, domain the support."""

    from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg

    from dexbench_arena.variations.object_pose_variation import ObjectPoseVariation, ObjectPoseVariationCfg

    low, high = cage_l2_package_pose_variation_range_for_sets(layouts, lift_objects)
    return ObjectPoseVariation(
        asset_name,
        cfg=ObjectPoseVariationCfg(sampler_cfg=UniformSamplerCfg(low=list(low), high=list(high))),
        domain=CageL2PackagePoseDomain(tuple(layouts), tuple(lift_objects)),
    )


LIFT_OBJECT_PAYLOAD_RANGE_KG = (2.0, 5.0)




LIFT_TO_CONVEYOR_WORKSPACE_XY_BOUNDS_M = ((-0.14, 0.20), (4.40, 2.80))
LIFT_TO_CONVEYOR_DROPPED_MIN_Z_M = 0.45
LIFT_TO_CONVEYOR_MAX_DESCENT_SPEED_M_S = 1.0


TORSO_CROUCH_EXTRA_RAD = 0.239110


def _lowered_torso_start_pose(
    home: Mapping[str, float] | None, extra_rad: float = TORSO_CROUCH_EXTRA_RAD
) -> dict[str, float] | None:
    """``home`` (an embodiment's replay home) with the torso crouched 0.10 m lower."""

    fold = rby1m_torso_z_fold(extra_rad)
    if not home or not fold.keys() <= home.keys():
        return None
    pose = dict(home)
    for name, offset in fold.items():
        pose[name] = pose[name] + offset
    return pose


PACK_INVOICE_TORSO_RISE_RAD = -0.29

PACK_INVOICE_RIGHT_ARM_OFFSETS_RAD = {"right_arm_0": 0.7, "right_arm_3": -0.15, "right_arm_5": -0.8}


def _pack_invoice_start_pose(home: Mapping[str, float] | None) -> Mapping[str, float] | None:
    """``home`` raised by ``PACK_INVOICE_TORSO_RISE_RAD`` with the right arm offset as above."""

    pose = _lowered_torso_start_pose(home, PACK_INVOICE_TORSO_RISE_RAD)
    if pose is None or not PACK_INVOICE_RIGHT_ARM_OFFSETS_RAD.keys() <= pose.keys():
        return pose or home
    for joint, offset in PACK_INVOICE_RIGHT_ARM_OFFSETS_RAD.items():
        pose[joint] = pose[joint] + offset
    return pose


def _make_lift_to_conveyor_task(env, cfg: DexBenchLogisticsCenterE2EnvironmentCfg):
    """Factory for the cage-to-conveyor lift task."""

    from dexbench_arena.tasks.lift_to_conveyor import DexBenchLiftToConveyorTask

    replay_home = start_poses(cfg.embodiment).get("replay_home")
    start_pose = _lowered_torso_start_pose(replay_home)
    lift_objects = DexBenchLogisticsCenterE2Environment.lift_objects_from_cli(cfg)
    mimic = {}
    if cfg.mimic_segments is not None:
        require_datagen("--mimic_segments")
        mimic["mimic_segments"] = cfg.mimic_segments

    return mimic_task_class(DexBenchLiftToConveyorTask)(
        hand_contact_material=WUJI_GRASP_CONTACT_MATERIAL,
        object_name="cage_l2_package",
        box_sizes_xyz=tuple(obj.size_xyz_m for obj in lift_objects),
        conveyor_xy_rects=CONVEYOR_PLACE_XY_RECTS_M,
        conveyor_top_z_range_m=CONVEYOR_PLACE_TOP_Z_RANGE_M,
        bottom_height_tolerance_m=CONVEYOR_PLACE_BOTTOM_HEIGHT_TOLERANCE_M,
        max_face_tilt_deg=CONVEYOR_PLACE_MAX_FACE_TILT_DEG,
        velocity_threshold_m_s=CONVEYOR_PLACE_LINEAR_SPEED_M_S,
        angular_velocity_threshold_rad_s=CONVEYOR_PLACE_ANGULAR_SPEED_RAD_S,
        hold_time_s=CONVEYOR_PLACE_HOLD_TIME_S,
        workspace_xy_bounds=LIFT_TO_CONVEYOR_WORKSPACE_XY_BOUNDS_M,
        object_dropped_min_z_m=LIFT_TO_CONVEYOR_DROPPED_MIN_Z_M,
        max_descent_speed_m_s=LIFT_TO_CONVEYOR_MAX_DESCENT_SPEED_M_S,
        robot_joint_pos_overrides=start_pose,
        log_placement_checks=cfg.log_placement_checks,
        **mimic,
    )


def _make_pack_invoice_task(env, cfg: DexBenchLogisticsCenterE2EnvironmentCfg):
    """Factory for the pack-invoice (DB-L3) task at the packing bench."""

    from dexbench_arena.tasks.pack_invoice import DexBenchPackInvoiceTask

    poses = start_poses(cfg.embodiment)
    home = poses.get("teleop_ready") or poses.get("replay_home")
    return DexBenchPackInvoiceTask(
        hand_contact_material=WUJI_GRASP_CONTACT_MATERIAL,
        robot_joint_pos_overrides=_pack_invoice_start_pose(home),
        paper_args={name: getattr(cfg, name) for name in PAPER_PHYSICS_ARGS},
        scene_args={f"outbound_box_scale_{a}": float(getattr(cfg, f"outbound_box_scale_{a}")) for a in "xyz"},
        paper_pile=env._paper_pile_listener,
    )


def apply_delivery_render_settings(env_cfg):
    """Apply the render settings the delivery hands its consumer (cli.apply_delivery_render_settings)."""

    return cli.apply_delivery_render_settings(env_cfg, E2_RENDER_SETTINGS, "the E2 delivery")


def with_delivery_render_settings(environment, enabled: bool):
    """``environment`` rendering at the delivery's settings when ``enabled`` (``--delivery_render_settings``)."""

    if enabled:
        previous_callback = environment.env_cfg_callback

        def configure_rendering(env_cfg):
            if previous_callback is not None:
                env_cfg = previous_callback(env_cfg)
            return apply_delivery_render_settings(env_cfg)

        environment.env_cfg_callback = configure_rendering
    return environment


def build_logistics_center_e2_scene(
    asset_registry,
    hide_room_shell: bool = False,
    bench_station: bool = False,
    paper_pile_listener=None,
    outbound_box_scale_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0),
    cage_layout: E2CageLayoutParams | None = None,
    cage_box_alt_scales: tuple[float, ...] = (),
    lift_objects: Sequence[E2LiftObject] = (),
    object_mass_kg: float | None = None,
    spawn_l2_task_items: bool = True,
    spawn_belt_items: bool = True,
    moving_belts: tuple[str, ...] = (),
) -> Scene:
    """Build the delivered E2 scene without an embodiment."""

    import isaaclab.sim as sim_utils
    from dexbench_arena.assets import background_library as _background_library  # noqa: F401
    from dexbench_arena.assets.object_library import e2_lift_box_set_member_usd
    from dexbench_arena.assets.object_set import SeatedObjectSet
    from dexbench_arena.variations.cage_stack_variation import CageStackVariation
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_arena.variations.object_mass_variation import ObjectMassVariationCfg
    from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg

    layout = e2_cage_layout(cage_layout)
    scales = e2_cage_stack_scales(layout.params, cage_box_alt_scales)
    set_layouts = [layout] + [e2_cage_layout(replace(layout.params, box_scale=scale)) for scale in scales[1:]]
    lift_objects = tuple(lift_objects) or (E2_LIFT_OBJECTS_BY_NAME[E2_DEFAULT_LIFT_OBJECT],)
    seats = [lift_object_pose(layout, obj) for obj in lift_objects]

    background = asset_registry.get_asset_by_name("dexbench_logistics_center_e2_cell")(
        hide_room_shell=hide_room_shell, moving_belts=moving_belts, bench_station=bench_station,
        **({"outbound_box_scale_xyz": outbound_box_scale_xyz} if bench_station else {}),
    )
    lights = []
    for light in E2_LIGHTS:
        light_class = asset_registry.get_asset_by_name(light.asset)
        spawner_cfg = None
        if light.size_m is not None:
            spawner_cfg = replace(light_class.default_spawner_cfg, width=light.size_m[0], height=light.size_m[1])
        lights.append(
            light_class(
                instance_name=light.name,
                initial_pose=Pose(position_xyz=light.position_xyz, rotation_xyzw=light.rotation_xyzw),
                spawner_cfg=spawner_cfg,
                position_range_m=light.position_range_m,
            )
        )
    cage_box_names = {placement.name for placement in layout.placements}
    left_out = cage_box_names if spawn_belt_items else cage_box_names | {item.name for item in E2_BELT_ITEM_PLACEMENTS}
    props = [
        asset_registry.get_asset_by_name(placement.asset)(
            instance_name=placement.name,
            initial_pose=Pose(position_xyz=placement.position_xyz, rotation_xyzw=placement.rotation_xyzw),
            scale=(placement.scale,) * 3,
        )
        for placement in e2_item_placements(layout)
        if placement.name not in left_out
    ]
    if not spawn_l2_task_items:
        return Scene(assets=[background, *lights, *props])
    cage_boxes = []
    for index, set_layout in enumerate(set_layouts):
        for placement in set_layout.placements:
            position = placement.position_xyz if index == 0 else e2_cage_box_park_position(placement.position_xyz, index)
            cage_boxes.append(
                asset_registry.get_asset_by_name(e2_cage_box_asset(placement.asset))(
                    instance_name=e2_cage_box_entity_name(placement.name, index),
                    initial_pose=Pose(position_xyz=position, rotation_xyzw=placement.rotation_xyzw),
                    scale=(placement.scale,) * 3,
                )
            )
    if len(lift_objects) == 1:
        position, rotation = seats[0]
        cage_package = asset_registry.get_asset_by_name(lift_objects[0].asset)(
            instance_name="cage_l2_package", initial_pose=Pose(position_xyz=position, rotation_xyzw=rotation)
        )
    else:
        cage_package = SeatedObjectSet(
            name="cage_l2_package",
            member_usd_paths=[e2_lift_box_set_member_usd(obj) for obj in lift_objects],
            seats=[Pose(position_xyz=position, rotation_xyzw=rotation) for position, rotation in seats],
            spawn_cfg_addon=asset_registry.get_asset_by_name(lift_objects[0].asset).spawn_cfg_addon,
            tags=["object", "logistics_center_e2"],
        )
    if object_mass_kg is not None:
        cage_package.object_cfg.spawn.mass_props = sim_utils.MassPropertiesCfg(mass=object_mass_kg)
    cage_package.get_variation("mass").apply_cfg(
        ObjectMassVariationCfg(
            sampler_cfg=UniformSamplerCfg(low=[LIFT_OBJECT_PAYLOAD_RANGE_KG[0]], high=[LIFT_OBJECT_PAYLOAD_RANGE_KG[1]])
        )
    )
    cage_package.add_variation(
        CageStackVariation(
            cage_package.name,
            nominal=layout.params,
            scales=scales,
            object_seats=tuple((*obj.footprint_xy_m, lift_object_front_clearance_m(obj)) for obj in lift_objects),
        )
    )
    cage_package.add_variation(cage_l2_package_pose_variation(cage_package.name, set_layouts, lift_objects))
    station = _bench_station_references(background, paper_pile_listener) if bench_station else []
    return Scene(assets=[background, *lights, *props, *cage_boxes, cage_package, *station])


def _bench_station_references(background, paper_pile_listener) -> list:
    """The bench station's packing bench and outbound box as object references, with their variations."""

    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.assets.object_type import ObjectType

    from dexbench_arena.variations.object_pose_variation import make_planar_pose_variation
    from dexbench_arena.variations.pack_invoice_station_variation import BenchHeightVariation, PaperPileVariation

    bench_name, box_name = E2_BENCH_STATION_REFERENCES
    bench = ObjectReference(
        name=bench_name, prim_path=f"{background.get_prim_path()}/{E2_BENCH_STATION_REFERENCES[bench_name]}",
        parent_asset=background, object_type=ObjectType.RIGID,
    )
    box = ObjectReference(
        name=box_name, prim_path=f"{background.get_prim_path()}/{E2_BENCH_STATION_REFERENCES[box_name]}",
        parent_asset=background, object_type=ObjectType.ARTICULATION,
    )
    bench.add_variation(BenchHeightVariation(bench_name, carried=(box_name,)))
    if paper_pile_listener is not None:
        bench.add_variation(PaperPileVariation(bench_name, paper_pile_listener))
    box.add_variation(make_planar_pose_variation(box_name, name="pose"))
    return [bench, box]


def add_e2_cell_cli_args(parser: argparse.ArgumentParser) -> None:
    """The flags every environment in the E2 cell takes: who drives the run and how the cell is rendered."""

    if "--teleop_session" not in parser._option_string_actions:
        parser.add_argument(
            "--teleop_session",
            "--teleop-session",
            action="store_true",
            help=(
                "This run is driven by an operator from inside the cell. Scripts that "
                "teleoperate set it themselves; it is exposed here so the same scene can "
                "be reproduced from the command line. What it changes is this "
                "environment's business -- see build_scene."
            ),
        )
    if "--delivery_render_settings" not in parser._option_string_actions:
        parser.add_argument(
            "--delivery_render_settings",
            "--delivery-render-settings",
            action=argparse.BooleanOptionalAction,
            default=True,
            help=(
                "Apply the tonemap and path-tracing settings the delivery embeds in its "
                "viz layer. On by default -- the delivery states these are the consumer's "
                "to apply and its lights are calibrated for them. Pass "
                "--no-delivery_render_settings to render at Arena's defaults instead."
            ),
        )


def _csv_floats(text: str) -> tuple[float, ...]:
    """argparse type for ``--cage_box_alt_scales``: ``"0.85,1.15"`` -> ``(0.85, 1.15)``."""

    return tuple(float(token) for token in text.split(",") if token.strip())


_LIFT_ONLY = {TASKS: ("lift_to_conveyor",)}
_PACK_ONLY = {TASKS: ("pack_invoice",)}


@dataclass
class DexBenchLogisticsCenterE2EnvironmentCfg(DexBenchEnvironmentCfg):
    """Typed config for DexBenchLogisticsCenterE2Environment: its flags, names and defaults."""

    task: str = "none"
    """Task variant the scene builds: ``lift_to_conveyor``, ``pack_invoice``, or ``none`` for the scene shell."""

    anchor: str = "A1_CAGE"
    """Robot base anchor name (A1_CAGE, facing the parcel cage; A3_BENCH, facing the packing bench)."""

    teleop_session: bool = False
    """The run is driven by an operator from inside the cell: the room shell is hidden (still solid)."""

    log_placement_checks: bool = field(default=False, metadata=_LIFT_ONLY)
    """Print, while the box sits at the conveyor's decks, which success check fails and by how much."""

    mimic_segments: int | None = field(default=None, metadata=_LIFT_ONLY)
    """lift_to_conveyor's Mimic subtasks per arm: 4 to generate from segment_db_l2_annotations.py output;
    None is the task's two."""

    delivery_render_settings: bool = True
    """False renders at Arena's defaults instead of the delivery's tonemap and exposure."""

    cage_box_scale: float = field(default=1.0, metadata=_LIFT_ONLY)
    """Uniform scale of the cage boxes; the object spawns on the scaled stack's top."""

    cage_stack_offset_depth: float = field(default=0.0, metadata=_LIFT_ONLY)
    """Slide the box stack toward the cage's back wall [m] (world -X as the cage stands)."""

    cage_stack_offset_lateral: float = field(default=0.0, metadata=_LIFT_ONLY)
    """Slide the box stack across the cage [m] (world +Y as the cage stands)."""

    cage_box_alt_scales: tuple[float, ...] = field(default=(), metadata=_LIFT_ONLY)
    """Box scales spawned as extra, parked sets for the cage_stack variation to swap in."""

    object_box: str = field(default=E2_DEFAULT_LIFT_OBJECT, metadata=_LIFT_ONLY)
    """Which E2_LIFT_OBJECTS entry the task lifts (train_0, train_1, eval_0..3, parcel)."""

    object_set: str | None = field(default=None, metadata=_LIFT_ONLY)
    """Spawn a whole group (train, eval), member ``e mod n`` in environment ``e``, instead of object_box."""

    object_mass_kg: float | None = field(default=None, metadata=_LIFT_ONLY)
    """Payload: the object's mass [kg]. None keeps the boxes' 3 kg (the parcel: its authored mass)."""

    invoice_sheet: str = field(default=E2_INVOICE_OBJECT_AXIS.default, metadata=_PACK_ONLY)
    """The sheet the task is run with (``E2_INVOICE_OBJECT_AXIS``; one)."""

    outbound_box_scale_x: float = field(default=E2_PACK_INVOICE_BOX_SCALE_XYZ[0], metadata=_PACK_ONLY)
    """Size of the outbound box along its x over the delivery's (``clear_e2_bench_station``)."""

    outbound_box_scale_y: float = field(default=E2_PACK_INVOICE_BOX_SCALE_XYZ[1], metadata=_PACK_ONLY)
    """Size of the outbound box along its y over the delivery's."""

    outbound_box_scale_z: float = field(default=E2_PACK_INVOICE_BOX_SCALE_XYZ[2], metadata=_PACK_ONLY)
    """Height of the outbound box over the delivery's."""

    paper_coupling: str = field(default=PAPER_PHYSICS_ARGS["paper_coupling"], metadata=_PACK_ONLY)
    """one_way_proxy, the proxy hand inside the solver (the recordings'): the only coupling. The paper's wrench is
    never fed back into the robot."""

    paper_hand_colliders: str = field(default=PAPER_PHYSICS_ARGS["paper_hand_colliders"], metadata=_PACK_ONLY)
    """official_hull, the robot USD's hand hulls: the only hand the paper sees."""

    paper_num_sheets: int = field(default=PAPER_PHYSICS_ARGS["paper_num_sheets"], metadata=_PACK_ONLY)
    """Sheets per pile."""

    paper_resolution: str = field(default=PAPER_PHYSICS_ARGS["paper_resolution"], metadata=_PACK_ONLY)
    """Cloth grid cells per sheet, NXxNY."""

    paper_substeps: int = field(default=PAPER_PHYSICS_ARGS["paper_substeps"], metadata=_PACK_ONLY)
    """Paper substeps per 1/30 s of simulated time, whatever the control rate."""

    paper_iterations: int = field(default=PAPER_PHYSICS_ARGS["paper_iterations"], metadata=_PACK_ONLY)
    """VBD iterations per paper substep."""

    paper_contact_mu: float = field(default=PAPER_PHYSICS_ARGS["paper_contact_mu"], metadata=_PACK_ONLY)
    """The paper's friction: sheet on sheet, and the geometric mean with the hand's, bench's and box's."""

    paper_self_contact_radius: float = field(default=PAPER_PHYSICS_ARGS["paper_self_contact_radius"],
                                             metadata=_PACK_ONLY)
    """Sheet-on-sheet contact thickness [m]; the margin is 1.5 and the pile's sheet spacing 2 times it."""


@register_environment
class DexBenchLogisticsCenterE2Environment(
    DexBenchSceneEnvironmentBase, ArenaEnvironmentFactory[DexBenchLogisticsCenterE2EnvironmentCfg]
):
    """PhysicsSim Lab E2 logistics centre, the robot facing the parcel cage beside the conveyor's infeed end."""

    newton_solver_options = {
        "solver": "cg",
        "num_substeps": 2,
        "tolerance": 1e-4,
        "nconmax": 1024,
        "njmax": 1024,
        "use_mujoco_contacts": True,
        "ccd_iterations": 50,
        "enable_multiccd": True,
    }

    def apply_newton_sim_settings(self, env_cfg, cfg: DexBenchLogisticsCenterE2EnvironmentCfg) -> None:
        """Keep only the cage's active box set responsive to contacts, also during replay."""
        super().apply_newton_sim_settings(env_cfg, cfg)
        nominal = self.cage_layout_params_from_cli(cfg)
        scales = e2_cage_stack_scales(nominal, self.cage_box_alt_scales_from_cli(cfg))
        if len(scales) == 1:
            return
        from dexbench_arena.assets.newton_parked_contacts import (
            NewtonParkedContactsMJWarpManager,
            ParkedContactsMJWarpSolverCfg,
        )

        offsets: list[tuple[str, float, float, float]] = []
        for index, scale in enumerate(scales):
            for placement in e2_cage_layout(replace(nominal, box_scale=scale)).placements:
                name = e2_cage_box_entity_name(placement.name, index)
                box_cfg = getattr(env_cfg.scene, name)
                parked = e2_cage_box_park_position(placement.position_xyz, index)
                spawn = placement.position_xyz if index == 0 else parked
                expression = box_cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_.*") + "(/.*)?"
                offsets.append((expression, *(a - b for a, b in zip(parked, spawn))))
        previous = env_cfg.sim.physics.solver_cfg
        values = {field.name: getattr(previous, field.name) for field in fields(previous)}
        values.update(class_type=NewtonParkedContactsMJWarpManager, parked_body_offsets=tuple(offsets))
        solver = ParkedContactsMJWarpSolverCfg(**values)
        env_cfg.sim.physics = replace(env_cfg.sim.physics, class_type=None, solver_cfg=solver)

    name = "dexbench_logistics_center_e2_environment"
    _legacy_argparse_cfg_type = DexBenchLogisticsCenterE2EnvironmentCfg
    robot_base_anchors_xyz = ROBOT_BASE_ANCHORS_XYZ
    anchor_facing_targets_xy = ANCHOR_FACING_TARGETS_XY
    default_anchor = "A1_CAGE"
    anchor_help = (
        "Robot base anchor: A1_CAGE, facing the parcel cage, the conveyor's infeed end on its left; "
        "A3_BENCH, facing the packing bench (pack_invoice)."
    )
    available_tasks = {
        "lift_to_conveyor": _make_lift_to_conveyor_task,
        "pack_invoice": _make_pack_invoice_task,
    }
    shared_settings = ("task", "teleop_session", "delivery_render_settings")
    object_axes = {"lift_to_conveyor": E2_LIFT_OBJECT_AXIS, "pack_invoice": E2_INVOICE_OBJECT_AXIS}
    bench_station_tasks = frozenset({"pack_invoice"})

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        super().add_cli_args(parser)
        add_e2_cell_cli_args(parser)
        _add_argument_if_missing(
            parser,
            "--log_placement_checks",
            "--log-placement-checks",
            dest="log_placement_checks",
            action="store_true",
            help=(
                "lift_to_conveyor: while the box sits at the conveyor's decks, print which success "
                "check keeps it from counting and its value, when that changes and once a second "
                "while it lasts; then the hold and the success. An operator's diagnostic: recorders "
                "leave it out of the replay arguments."
            ),
        )
        _add_argument_if_missing(
            parser,
            "--mimic_segments",
            type=int,
            choices=(2, 4),
            default=None,
            help=(
                "lift_to_conveyor Mimic subtasks per arm: 2 (unset; annotation always uses two) or 4 to "
                "generate from segment_db_l2_annotations.py output."
            ),
        )
        for axis, index, what in (("x", 0, "size along its x"), ("y", 1, "size along its y"), ("z", 2, "height")):
            _add_argument_if_missing(
                parser, f"--outbound_box_scale_{axis}", f"--outbound-box-scale-{axis}", dest=f"outbound_box_scale_{axis}",
                type=float, default=E2_PACK_INVOICE_BOX_SCALE_XYZ[index],
                help=(
                    f"pack_invoice: the outbound box's {what} over the delivered box's "
                    f"(default {E2_PACK_INVOICE_BOX_SCALE_XYZ[index]:.4f}, the task's carton; 1 is the delivered box)."
                ),
            )
        _add_argument_if_missing(
            parser,
            "--cage_box_scale",
            "--cage-box-scale",
            dest="cage_box_scale",
            type=float,
            default=1.0,
            help=(
                "Uniform scale of the eight cage boxes; the stack scales about its front "
                "plane and the cage floor, so the object spawns on the scaled tier's top "
                "(0.250 + 0.621 * scale m). 1.0 is the delivery; about 1.17 fills the "
                "cage's width, and the smallest tier an object fits on depends on the object."
            ),
        )
        _add_argument_if_missing(
            parser,
            "--cage_stack_offset_depth",
            "--cage-stack-offset-depth",
            dest="cage_stack_offset_depth",
            type=float,
            default=0.0,
            help=(
                "Slide the box stack toward the cage's back wall, in metres (world -X as "
                "the cage stands). 0 is the delivery, 50 mm behind the opening; the wall "
                "allows about 0.19 at scale 1."
            ),
        )
        _add_argument_if_missing(
            parser,
            "--cage_stack_offset_lateral",
            "--cage-stack-offset-lateral",
            dest="cage_stack_offset_lateral",
            type=float,
            default=0.0,
            help=(
                "Slide the box stack across the cage, in metres (world +Y as the cage "
                "stands). 0 centres it; the side walls allow about +/-0.085 at scale 1."
            ),
        )
        _add_argument_if_missing(
            parser,
            "--cage_box_alt_scales",
            "--cage-box-alt-scales",
            dest="cage_box_alt_scales",
            type=_csv_floats,
            default=(),
            metavar="S1,S2,...",
            help=(
                "Comma-separated box scales, e.g. 0.85,1.15: one more complete set of cage "
                "boxes is spawned per entry, parked 30 m outside the room, for the recorded "
                "cage_stack variation to swap into the cage at reset "
                "(cage_l2_package.cage_stack.enabled=true). Without that variation the sets "
                "just wait there. Each must fit the cage at the nominal offsets."
            ),
        )
        _add_argument_if_missing(
            parser, "--invoice_sheet", "--invoice-sheet", dest="invoice_sheet", type=str,
            default=E2_INVOICE_OBJECT_AXIS.default, choices=E2_INVOICE_OBJECT_AXIS.keys(),
            help="pack_invoice: the sheet the task is run with (one for now: the 202 x 276 mm invoice).",
        )
        paper_choices = {"paper_coupling": ("one_way_proxy",),
                         "paper_hand_colliders": ("official_hull",)}
        for name, default in PAPER_PHYSICS_ARGS.items():
            kwargs = ({"action": argparse.BooleanOptionalAction} if isinstance(default, bool)
                      else {"type": type(default), "choices": paper_choices.get(name)})
            _add_argument_if_missing(
                parser, f"--{name}", f"--{name.replace('_', '-')}", dest=name, default=default, **kwargs,
                help=f"pack_invoice: {name.removeprefix('paper_').replace('_', ' ')} of the paper sidecar "
                     f"(default {default}, the teleop recordings').",
            )
        _add_argument_if_missing(
            parser,
            "--object_box",
            "--object-box",
            dest="object_box",
            type=str,
            default=E2_DEFAULT_LIFT_OBJECT,
            choices=tuple(E2_LIFT_OBJECTS_BY_NAME),
            help=(
                "The box the task lifts: train_0 is Korea Post No.4 (410x310x280) and train_1 No.5 "
                "(480x380x340); eval_0..3 are 400x300x250, 450x350x300, 500x400x250 and 600x400x300; "
                "parcel is the post-office package the cell was first built with. Every box is its own "
                "object_library/box_<size> package; boxes weigh 3 kg unless --object_mass_kg says otherwise."
            ),
        )
        _add_argument_if_missing(
            parser,
            "--object_set",
            "--object-set",
            dest="object_set",
            type=str,
            default=None,
            choices=tuple(E2_LIFT_OBJECT_SETS),
            help=(
                "Spawn a whole group instead of --object_box: one rigid entity whose member in "
                "environment e is the group's e-th box (mod its size; the scene is cloned round-robin "
                "for it), so --num_envs 4 with eval puts one eval box in each environment. Per "
                "environment, not per reset."
            ),
        )
        _add_argument_if_missing(
            parser,
            "--object_mass_kg",
            "--object-mass-kg",
            dest="object_mass_kg",
            type=float,
            default=None,
            help=(
                "Payload knob: the object's mass in kg. Default keeps the boxes' 3 kg "
                "(the parcel its authored 0.3 kg). The recorded mass variation "
                "(cage_l2_package.mass.enabled=true) draws 2-5 kg instead."
            ),
        )

    def object_axis_for(self, cfg):
        """No single object with ``--object_set``: its members are seated one per env."""
        return None if getattr(cfg, "object_set", None) else super().object_axis_for(cfg)

    def task_object_fit(self, cfg, task_object) -> dict:
        """The box's seat on the upper tier: where ``lift_object_pose`` spawns it on the run's stack.
        pack_invoice's sheet has nothing to derive."""
        if task_object.key not in E2_LIFT_OBJECTS_BY_NAME:
            return {}
        position, _rotation = lift_object_pose(
            e2_cage_layout(self.cage_layout_params_from_cli(cfg)), E2_LIFT_OBJECTS_BY_NAME[task_object.key]
        )
        return {"seat_xyz_m": [round(float(value), 6) for value in position]}

    @staticmethod
    def lift_objects_from_cli(cfg) -> tuple[E2LiftObject, ...]:
        """The object(s) a config asks for: a set's members in order, or the one box (train_0 by default)."""

        group = getattr(cfg, "object_set", None)
        if group:
            return tuple(E2_LIFT_OBJECTS_BY_NAME[name] for name in E2_LIFT_OBJECT_SETS[group])
        return (E2_LIFT_OBJECTS_BY_NAME[getattr(cfg, "object_box", None) or E2_DEFAULT_LIFT_OBJECT],)

    @staticmethod
    def cage_layout_params_from_cli(cfg) -> E2CageLayoutParams:
        """The box-stack layout a config asks for (the delivery by default)."""

        def value(name: str, default: float) -> float:
            given = getattr(cfg, name, None)
            return default if given is None else float(given)

        return E2CageLayoutParams(
            box_scale=value("cage_box_scale", 1.0),
            stack_offset_depth_m=value("cage_stack_offset_depth", 0.0),
            stack_offset_lateral_m=value("cage_stack_offset_lateral", 0.0),
        )

    @staticmethod
    def cage_box_alt_scales_from_cli(cfg) -> tuple[float, ...]:
        """The extra box-set scales a config asks for (none by default)."""

        given = getattr(cfg, "cage_box_alt_scales", None)
        if given is None:
            return ()
        return _csv_floats(given) if isinstance(given, str) else tuple(float(scale) for scale in given)

    def task_object_usd_path(self, task_object) -> str | None:
        """pack_invoice's sheet is the paper sidecar's cloth (``E2_INVOICE_OBJECT_AXIS``), no registered
        asset, so its record carries no package version."""
        if not self.asset_registry.is_registered(task_object.asset):
            return None
        return super().task_object_usd_path(task_object)

    def build(self, cfg: DexBenchLogisticsCenterE2EnvironmentCfg):
        from dexbench_arena.variations.pack_invoice_station_variation import PaperPileListener

        self._paper_pile_listener = PaperPileListener() if cfg.task in self.bench_station_tasks else None
        return with_delivery_render_settings(super().build(cfg), cfg.delivery_render_settings)

    def build_scene(self, cfg: DexBenchLogisticsCenterE2EnvironmentCfg) -> Scene:
        return build_logistics_center_e2_scene(
            self.asset_registry,
            hide_room_shell=bool(cfg.teleop_session),
            bench_station=cfg.task in self.bench_station_tasks,
            paper_pile_listener=getattr(self, "_paper_pile_listener", None),
            outbound_box_scale_xyz=tuple(float(getattr(cfg, f"outbound_box_scale_{a}")) for a in "xyz"),
            cage_layout=self.cage_layout_params_from_cli(cfg),
            cage_box_alt_scales=self.cage_box_alt_scales_from_cli(cfg),
            lift_objects=self.lift_objects_from_cli(cfg),
            object_mass_kg=None if cfg.object_mass_kg is None else float(cfg.object_mass_kg),
        )
