"""DB-H1 -- plate transfer in the E4 hotel banquet hall, staged for teleop."""

from __future__ import annotations

from dataclasses import dataclass
import isaaclab.sim as sim_utils
import argparse
import math
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentFactory
from isaaclab_arena.utils.pose import Pose

from dexbench_arena.assets.e4_hotel_fb_dish_specs import (
    DEFAULT_DISH_VARIANT,
    DISH_OBJECT_AXIS,
    dish_spec,
    dish_variant_names,
)
from dexbench_arena.assets.e4_hotel_fb_lighting import E4_LIGHTING_ASSET
from dexbench_arena.assets.e4_hotel_fb_pour import POUR_JUGS
from dexbench_arena.cli import set_rtx_global_settings
from dexbench_arena.layouts.db_h1_layout import (
    ANCHOR_FACING_TARGETS_XY,
    CLEARING_ROBOT_EDGE_Y_M,
    CLEARING_TOP_Z_M,
    CLEARING_XY_M,
    DEFAULT_DEST_PLATE_COUNT,
    DEFAULT_SOURCE_PLATE_COUNT,
    DEST_STACK_XY_M,
    E4_RENDER_SETTINGS,
    H1_ARM_START_POSE_RAD,
    H1_HAND_DYNAMIC_FRICTION,
    H1_HAND_STATIC_FRICTION,
    H1_TORSO_CROUCH_RAD,
    HANDS_CLEAR_HEIGHT_M,
    HANDS_CLEAR_MARGIN_M,
    LEAN_BUDGET_DEG,
    PLATE_MAX_OVERHANG_FRACTION,
    POUR_DECOR_GOBLETS_XY_M,
    POUR_GOBLET_XYZ_M,
    POUR_JUG_XYZ_M,
    POUR_RENDER_SETTINGS,
    RIM_OFFSET_BUDGET_MM,
    ROBOT_BASE_ANCHORS_XYZ,
    ROUND_TABLE_TOP_Z_M,
    SEAT_HEIGHT_BUDGET_MM,
    SOURCE_STACK_XY_M,
    STACK_RIM_INSET_M,
)
from dexbench_arena.environments.manipulation.scene_environment_base import (
    DexBenchEnvironmentCfg,
    DexBenchSceneEnvironmentBase,
)
from dexbench_arena.variations.object_pose_variation import make_planar_pose_variation

if TYPE_CHECKING:
    from isaaclab_arena.scene.scene import Scene

def _stack_poses(center_xy, surface_z_m, count, nest_pitch_m):
    """Absolute poses for ``count`` dishes nested on a surface, bottom first."""
    from isaaclab_arena.utils.pose import Pose

    pitch = nest_pitch_m
    return [
        Pose(
            position_xyz=(center_xy[0], center_xy[1], surface_z_m + index * pitch),
            rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
        )
        for index in range(count)
    ]


def _pour_station_assets(
    asset_registry, jug: str = "idealisk",
    decor_goblets: bool = False, jug_jitter_xy_m: float = 0.0, jug_jitter_yaw_deg: float = 0.0,
    goblet_jitter_xy_m: float = 0.0, station_offset_xy_m: tuple[float, float] = (0.0, 0.0),
    jug_offset_xy_m: tuple[float, float] = (0.0, 0.0), jug_offset_yaw_deg: float = 0.0,
    goblet_offset_xy_m: tuple[float, float] = (0.0, 0.0),
) -> list:
    """The pour jug, the target goblet and, when asked for, the two decor goblets, at the delivered arrangement's
    spots (``db_h1_layout.POUR_JUG_XYZ_M``, ``POUR_GOBLET_XYZ_M``, ``POUR_DECOR_GOBLETS_XY_M``), the jug's handle
    toward +X."""
    identity = (0.0, 0.0, 0.0, 1.0)
    jug_asset, jug_instance = POUR_JUGS[jug]
    jug_x, jug_y = POUR_JUG_XYZ_M[0] + station_offset_xy_m[0], POUR_JUG_XYZ_M[1] + station_offset_xy_m[1]
    goblet_xy = (POUR_GOBLET_XYZ_M[0] + station_offset_xy_m[0], POUR_GOBLET_XYZ_M[1] + station_offset_xy_m[1])
    goblet = asset_registry.get_asset_by_name("dexbench_goblet_30cl")
    jug_object = asset_registry.get_asset_by_name(jug_asset)(
        instance_name=jug_instance,
        initial_pose=Pose(position_xyz=(jug_x, jug_y, POUR_JUG_XYZ_M[2]), rotation_xyzw=identity),
    )
    jug_object.add_variation(make_planar_pose_variation(
        jug_instance, xy_half_extent_m=jug_jitter_xy_m, yaw_half_extent_rad=math.radians(jug_jitter_yaw_deg),
        xy_offset_m=jug_offset_xy_m, yaw_offset_rad=math.radians(jug_offset_yaw_deg), name="pour_pose"))
    target = goblet(
        instance_name="pour_goblet",
        initial_pose=Pose(position_xyz=(*goblet_xy, POUR_GOBLET_XYZ_M[2]), rotation_xyzw=identity),
    )
    target.add_variation(make_planar_pose_variation(
        "pour_goblet", xy_half_extent_m=goblet_jitter_xy_m, xy_offset_m=goblet_offset_xy_m, name="pour_pose"))
    assets = [jug_object, target]
    for name, xy in (POUR_DECOR_GOBLETS_XY_M.items() if decor_goblets else ()):
        xy = (xy[0] + station_offset_xy_m[0], xy[1] + station_offset_xy_m[1])
        assets.append(
            goblet(instance_name=name, initial_pose=Pose(position_xyz=(*xy, POUR_GOBLET_XYZ_M[2]), rotation_xyzw=identity))
        )
    return assets


def build_e4_hotel_fb_scene(
    asset_registry,
    *,
    dish_variant: str = DEFAULT_DISH_VARIANT,
    plate_diameter_m: float | None = None,
    source_plate_count: int = DEFAULT_SOURCE_PLATE_COUNT,
    dest_plate_count: int = DEFAULT_DEST_PLATE_COUNT,
    source_stack_offset_xy_m: tuple[float, float] = (0.0, 0.0),
    dest_stack_offset_xy_m: tuple[float, float] = (0.0, 0.0),
    stack_jitter_xy_m: float = 0.0,
    stack_jitter_yaw_deg: float = 0.0,
    spawn_ground_plane: bool = True,
    idealisk_pour: bool = False,
    pour_jug: str = "idealisk",
    pour_decor_goblets: bool = False,
    pour_jug_jitter_xy_m: float = 0.0,
    pour_jug_jitter_yaw_deg: float = 0.0,
    pour_goblet_jitter_xy_m: float = 0.0,
    pour_station_offset_xy_m: tuple[float, float] = (0.0, 0.0),
    pour_jug_offset_xy_m: tuple[float, float] = (0.0, 0.0),
    pour_jug_offset_yaw_deg: float = 0.0,
    pour_goblet_offset_xy_m: tuple[float, float] = (0.0, 0.0),
) -> Scene:
    """E4 banquet hall (serving cart at A_h1's left) + two plate stacks, no delivered dishes."""

    from dexbench_arena.assets import background_library as _background_library  # noqa: F401
    from dexbench_arena.assets import e4_hotel_fb_dishes as _dishes  # noqa: F401  (registers the dish assets)
    from dexbench_arena.assets import object_library as _object_library  # noqa: F401
    from dexbench_arena.assets.e4_hotel_fb_scene import require_h1_destination
    from isaaclab_arena.scene.scene import Scene

    from dexbench_arena.layouts.db_h1_layout import (
        plate_radius_m,
        stack_region_radius_m,
    )

    spec = dish_spec(dish_variant)
    diameter = float(plate_diameter_m or spec.diameter_m)
    scale = spec.scale_for(diameter)
    scale_xyz = (scale, scale, scale)
    nest_pitch_m = spec.nest_pitch_for(diameter)

    background = asset_registry.get_asset_by_name("dexbench_hotel_fb_e4_260831")()
    require_h1_destination(background.usd_path, (CLEARING_XY_M[0], CLEARING_ROBOT_EDGE_Y_M, 0.0))
    lighting = asset_registry.get_asset_by_name(E4_LIGHTING_ASSET)()

    plate_asset = asset_registry.get_asset_by_name(spec.asset_name)

    def stack(prefix, center_xy, surface_z_m, count, offset):
        center = (center_xy[0] + offset[0], center_xy[1] + offset[1])
        return [
            plate_asset(
                instance_name=f"{prefix}_{index:02d}",
                initial_pose=pose,
                scale=scale_xyz,
            )
            for index, pose in enumerate(_stack_poses(center, surface_z_m, count, nest_pitch_m))
        ]

    source = stack(
        "table_plate",
        SOURCE_STACK_XY_M,
        ROUND_TABLE_TOP_Z_M,
        source_plate_count,
        source_stack_offset_xy_m,
    )
    dest = stack(
        "clearing_plate",
        DEST_STACK_XY_M,
        CLEARING_TOP_Z_M,
        dest_plate_count,
        dest_stack_offset_xy_m,
    )

    from dexbench_arena.variations.object_pose_variation import PlanarPoseDomain, make_planar_pose_variation

    max_jitter_xy_m = round(STACK_RIM_INSET_M - plate_radius_m(diameter) * (1.0 - PLATE_MAX_OVERHANG_FRACTION), 3)
    fixture_edge = (
        f"more would carry the near rim past the fixture edge by over {PLATE_MAX_OVERHANG_FRACTION:.0%} of a "
        f"{diameter * 1000:.0f} mm dish's radius; --stack_jitter_xy or the sampler range is too wide, or move "
        "the stacks inboard with --source/dest_stack_offset"
    )
    yaw_half = math.radians(stack_jitter_yaw_deg)
    source[0].add_variation(
        make_planar_pose_variation(
            "table_plate_00",
            member_names=[f"table_plate_{index:02d}" for index in range(source_plate_count)],
            xy_half_extent_m=stack_jitter_xy_m,
            yaw_half_extent_rad=yaw_half,
            name="source_stack_pose",
            domain=PlanarPoseDomain(max_abs_xy_m=(max_jitter_xy_m, max_jitter_xy_m), reason=fixture_edge),
        )
    )
    dest[0].add_variation(
        make_planar_pose_variation(
            "clearing_plate_00",
            member_names=[f"clearing_plate_{index:02d}" for index in range(dest_plate_count)],
            xy_half_extent_m=stack_jitter_xy_m,
            yaw_half_extent_rad=yaw_half,
            name="dest_stack_pose",
            domain=PlanarPoseDomain(
                max_abs_xy_m=(max_jitter_xy_m, max_jitter_xy_m),
                max_xy_radius_m=stack_region_radius_m(diameter),
                reason=f"{fixture_edge}; and the stack stays within one success-region radius of its nominal spot",
            ),
        )
    )

    assets = [lighting, background, *source, *dest]
    if idealisk_pour:
        assets.extend(
            _pour_station_assets(
                asset_registry, pour_jug, pour_decor_goblets,
                pour_jug_jitter_xy_m, pour_jug_jitter_yaw_deg, pour_goblet_jitter_xy_m, pour_station_offset_xy_m,
                jug_offset_xy_m=pour_jug_offset_xy_m, jug_offset_yaw_deg=pour_jug_offset_yaw_deg,
                goblet_offset_xy_m=pour_goblet_offset_xy_m,
            )
        )
    if spawn_ground_plane:
        assets.insert(0, asset_registry.get_asset_by_name("ground_plane")(spawner_cfg=sim_utils.GroundPlaneCfg(visible=False)))
    return Scene(assets=assets)


def apply_delivery_render_settings(env_cfg, extra_settings: dict | None = None):
    """Apply the exposure and bounce counts the r01 delivery hands its consumer."""

    settings = {**E4_RENDER_SETTINGS, **(extra_settings or {})}
    carb_settings = {"/" + key.replace(":", "/"): value for key, value in settings.items()}
    cameras = set_rtx_global_settings(env_cfg, carb_settings=carb_settings)
    print(
        f"[dexbench] {len(E4_RENDER_SETTINGS)} render settings from the E4 r01 delivery"
        f"{f' and {len(extra_settings)} for the pour glass' if extra_settings else ''} "
        f"({cameras} camera renderer cfg{'' if cameras == 1 else 's'})"
    )
    return env_cfg


def apply_pour_render_settings(env_cfg):
    """Apply ``POUR_RENDER_SETTINGS`` alone, for a pour scene built with ``--no-delivery_render_settings``."""

    carb_settings = {"/" + key.replace(":", "/"): value for key, value in POUR_RENDER_SETTINGS.items()}
    cameras = set_rtx_global_settings(env_cfg, carb_settings=carb_settings)
    print(f"[dexbench] {len(carb_settings)} render settings for the pour glass ({cameras} camera renderer cfgs)")
    return env_cfg


@dataclass
class DexBenchHotelFBE4EnvironmentCfg(DexBenchEnvironmentCfg):
    """Typed config for DB-H1 in the E4 banquet hall: its flags, names and defaults."""

    task: str = "none"
    """Task variant the scene builds: ``stack_plates``, or ``none`` for the scene shell."""
    anchor: str = "A_h1"
    """Robot base anchor name (the A_h1 banquet station)."""
    plate_variant: str = DEFAULT_DISH_VARIANT
    """Which DB-H1 dish both stacks are made of (``e4_hotel_fb_dish_specs.DISH_GEOMETRY`` key)."""
    plate_diameter_m: float | None = None
    """Rim diameter the chosen dish is rendered at [m]; None keeps the dish's own."""
    source_plate_count: int = DEFAULT_SOURCE_PLATE_COUNT
    """Plates on the banquet table to carry."""
    dest_plate_count: int = DEFAULT_DEST_PLATE_COUNT
    """Plates already on the clearing station."""
    source_stack_offset_x: float = 0.0
    """Table stack X shift [m]; positive is away from the robot."""
    source_stack_offset_y: float = 0.0
    """Table stack Y shift [m]."""
    dest_stack_offset_x: float = 0.0
    """Clearing stack X shift [m]."""
    dest_stack_offset_y: float = 0.0
    """Clearing stack Y shift [m]."""
    stack_jitter_xy: float = 0.0
    """Half-extent [m] of a per-reset uniform XY offset applied to each stack as a unit; 0 = off."""
    stack_jitter_yaw_deg: float = 0.0
    """Half-extent [deg] of a per-reset uniform yaw of each stack about its axis; 0 = off."""
    delivery_render_settings: bool = True
    """False renders at Arena's defaults instead of the delivery's tonemap and exposure."""
    stack_alignment_point: str = "root"
    """Lateral alignment feature: root or dish rim center."""
    success_follows_dest_draw: bool = False
    """Judge stack_plates' success around the destination stack where its dest_stack_pose draw put it,
    not its nominal spot. The H1 eval cells set it; collection and re-recording keep the nominal centre."""


@register_environment
class DexBenchHotelFBE4Environment(
    DexBenchSceneEnvironmentBase,
    ArenaEnvironmentFactory[DexBenchHotelFBE4EnvironmentCfg],
):
    """DB-H1: the E4 banquet hall staged for the plate-stack transfer."""

    newton_solver_options = {"use_mujoco_contacts": True, "enable_multiccd": True, "ccd_iterations": 35}

    def apply_newton_sim_settings(self, env_cfg, cfg: DexBenchHotelFBE4EnvironmentCfg) -> None:
        physics = getattr(env_cfg.sim, "physics", None)
        if cfg.task == "stack_plates" and getattr(getattr(physics, "solver_cfg", None), "solver_type", None) == "mujoco_warp":
            super().apply_newton_sim_settings(env_cfg, cfg)
            from dexbench_arena.assets.e4_hotel_fb_dishes import configure_h1_newton_physics

            configure_h1_newton_physics(env_cfg)

    name = "dexbench_hotel_fb_e4_environment"
    _legacy_argparse_cfg_type = DexBenchHotelFBE4EnvironmentCfg
    robot_base_anchors_xyz = ROBOT_BASE_ANCHORS_XYZ
    anchor_facing_targets_xy = ANCHOR_FACING_TARGETS_XY
    default_anchor = "A_h1"
    teleop_hidden_prim_patterns = (
        r"^Ceiling$", r"^Wall_[NESW]$", r"^BG_[ensw]wall$", r"^BG_ceiling$",
        r"^BG_chandelier$", r"^TbFC_Tablecloth_visonly",
    )
    replay_hidden_prim_patterns = (r"^TbFC_Tablecloth_visonly", r"^BG_chandelier$")

    def _make_stack_plates_task(self, cfg: DexBenchHotelFBE4EnvironmentCfg):
        """Seat the table stack onto the clearing-station stack."""
        if cfg.stack_alignment_point not in ("root", "rim"):
            raise ValueError("stack_alignment_point must be root or rim")
        from dexbench_arena.layouts.db_h1_layout import (
            PLATE_SPAWN_GAP_M,
            stack_region_radius_m,
            plate_radius_m,
        )
        from dexbench_arena.contact_materials import HandContactMaterialCfg
        from dexbench_arena.tasks.stack_plates import DexBenchStackPlatesTask
        from dexbench_arena.variations.object_pose_variation import PlanarOffsetListener

        spec = dish_spec(cfg.plate_variant)
        diameter = float(cfg.plate_diameter_m or spec.diameter_m)
        offset = (float(cfg.dest_stack_offset_x), float(cfg.dest_stack_offset_y))
        center = (DEST_STACK_XY_M[0] + offset[0], DEST_STACK_XY_M[1] + offset[1])

        source_names, dest_names = self._plate_names(cfg)
        names = source_names + dest_names

        hand_material = HandContactMaterialCfg(
            static_friction=H1_HAND_STATIC_FRICTION,
            dynamic_friction=H1_HAND_DYNAMIC_FRICTION,
            friction_combine_mode="max",
        )

        return DexBenchStackPlatesTask(
            robot_joint_pos_overrides=self._h1_start_pose(cfg),
            hand_contact_material=hand_material,
            plate_names=names,
            source_plate_names=source_names,
            stack_center_xy=center,
            surface_top_z_m=CLEARING_TOP_Z_M,
            region_radius_m=stack_region_radius_m(diameter),
            expected_count=len(names),
            max_rim_offset_mm=RIM_OFFSET_BUDGET_MM,
            rim_offset_xyz=(0.0, 0.0, spec.height_m * spec.scale_for(diameter) if cfg.stack_alignment_point == "rim" else 0.0),
            max_lean_deg=LEAN_BUDGET_DEG,
            max_seat_height_error_mm=SEAT_HEIGHT_BUDGET_MM,
            hands_clear_radius_m=plate_radius_m(diameter) + HANDS_CLEAR_MARGIN_M,
            hands_clear_height_m=HANDS_CLEAR_HEIGHT_M,
            nest_pitch_m=spec.nest_pitch_for(diameter) + PLATE_SPAWN_GAP_M,
            plate_dropped_min_z_m=CLEARING_TOP_Z_M - 0.15,
            stack_center_offset=PlanarOffsetListener() if cfg.success_follows_dest_draw else None,
            task_description=(
                f"Lift the stack of {spec.kind}s from the banquet table and seat it on the serving cart stack."
            ),
        )

    @staticmethod
    def _plate_names(cfg: DexBenchHotelFBE4EnvironmentCfg) -> tuple[list[str], list[str]]:
        """``(source, destination)`` plate instance names, bottom plate first."""
        source_count = int(cfg.source_plate_count)
        dest_count = int(cfg.dest_plate_count)
        return (
            [f"table_plate_{i:02d}" for i in range(source_count)],
            [f"clearing_plate_{i:02d}" for i in range(dest_count)],
        )


    def _h1_start_pose(self, cfg: DexBenchHotelFBE4EnvironmentCfg):
        """Return the crouch start pose for the selected embodiment, or None."""
        from dexbench_arena.layouts.db_h1_layout import (
            torso_crouch_joint_pos,
        )

        base = super()._get_embodiment_joint_pos_overrides(cfg)
        if not base:
            return None
        pose = torso_crouch_joint_pos(base, H1_TORSO_CROUCH_RAD)
        missing = [name for name in H1_ARM_START_POSE_RAD if name not in pose]
        if missing:
            raise KeyError(f"H1 arm start pose names joints the embodiment lacks: {missing}")
        pose.update(H1_ARM_START_POSE_RAD)
        return pose

    available_tasks = {
        "stack_plates": _make_stack_plates_task,
    }
    object_axes = {"stack_plates": DISH_OBJECT_AXIS}

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        super().add_cli_args(parser)

        def _add(*options, **kwargs):
            if any(option in parser._option_string_actions for option in options):
                return
            parser.add_argument(*options, **kwargs)

        _add(
            "--plate_diameter_m",
            "--plate-diameter-m",
            dest="plate_diameter_m",
            type=float,
            default=None,
            help="Render the chosen dish at this rim diameter [m] (GA-O size knob); default is "
            "the dish's own. The nest pitch and the success slots scale with it.",
        )
        _add(
            "--source_plate_count",
            "--source-plate-count",
            dest="source_plate_count",
            type=int,
            default=DEFAULT_SOURCE_PLATE_COUNT,
            help="Plates on the banquet table to carry.",
        )
        _add(
            "--dest_plate_count",
            "--dest-plate-count",
            dest="dest_plate_count",
            type=int,
            default=DEFAULT_DEST_PLATE_COUNT,
            help="Plates already on the clearing station.",
        )
        for axis in ("x", "y"):
            _add(
                f"--source_stack_offset_{axis}",
                dest=f"source_stack_offset_{axis}",
                type=float,
                default=0.0,
                help=f"Shift the table stack in {axis.upper()} [m]. Positive X is away "
                "from the robot; the rim already sits flush with the table edge.",
            )
            _add(
                f"--dest_stack_offset_{axis}",
                dest=f"dest_stack_offset_{axis}",
                type=float,
                default=0.0,
                help=f"Shift the clearing stack in {axis.upper()} [m].",
            )
        _add(
            "--stack_jitter_xy",
            "--stack-jitter-xy",
            dest="stack_jitter_xy",
            type=float,
            default=0.0,
            help="Half-extent [m] of a per-reset uniform XY offset applied to each stack as a "
            "unit (independent per stack), for collection diversity. 0 = off.",
        )
        _add(
            "--stack_jitter_yaw_deg",
            "--stack-jitter-yaw-deg",
            dest="stack_jitter_yaw_deg",
            type=float,
            default=0.0,
            help="Half-extent [deg] of a per-reset uniform yaw of each stack about its axis. 0 = off.",
        )
        _add(
            "--delivery_render_settings",
            "--delivery-render-settings",
            dest="delivery_render_settings",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Apply the tonemap and exposure the r01 delivery embeds in its viz layer. On by default: "
            "the delivered lights are tuned for it and render the room black without it. Pass "
            "--no-delivery_render_settings to render at Arena's defaults instead.",
        )
        _add("--stack_alignment_point", "--stack-alignment-point", choices=("root", "rim"), default="root")
        _add(
            "--success_follows_dest_draw",
            "--success-follows-dest-draw",
            dest="success_follows_dest_draw",
            action="store_true",
            help="Judge stack_plates' success around the destination stack where its draw put it "
            "(the DB-H1 eval cells), not its nominal spot.",
        )
        _add(
            "--plate_variant",
            "--plate-variant",
            dest="plate_variant",
            choices=dish_variant_names(),
            default=DEFAULT_DISH_VARIANT,
            help="Which DB-H1 dish both stacks are made of. A variant carries its own USD, native "
            "diameter and measured nest pitch, so the stacks are seeded and scored against the "
            "geometry actually spawned.",
        )

    def pour_station_dressing(self, cfg: DexBenchHotelFBE4EnvironmentCfg) -> dict | None:
        """The pour station's render settings: none in DB-H1's hall. The pour environment
        (``e4_hotel_fb_pour_environment``) returns its station's."""
        return None

    def build(self, cfg: DexBenchHotelFBE4EnvironmentCfg):
        environment = super().build(cfg)
        offset = getattr(environment.task, "stack_center_offset", None)
        if offset is not None:
            environment.scene.assets["clearing_plate_00"].get_variation("dest_stack_pose").add_sample_listener(offset)
        pour_settings = self.pour_station_dressing(cfg)
        if cfg.delivery_render_settings is not False or pour_settings:
            previous_callback = environment.env_cfg_callback

            def configure_rendering(env_cfg):
                if previous_callback is not None:
                    env_cfg = previous_callback(env_cfg)
                if cfg.delivery_render_settings is not False:
                    return apply_delivery_render_settings(env_cfg, pour_settings)
                return apply_pour_render_settings(env_cfg)

            environment.env_cfg_callback = configure_rendering
        return environment

    def build_scene(self, cfg: DexBenchHotelFBE4EnvironmentCfg) -> Scene:
        return build_e4_hotel_fb_scene(
            self.asset_registry,
            dish_variant=cfg.plate_variant,
            plate_diameter_m=cfg.plate_diameter_m,
            source_plate_count=int(cfg.source_plate_count),
            dest_plate_count=int(cfg.dest_plate_count),
            source_stack_offset_xy_m=(float(cfg.source_stack_offset_x), float(cfg.source_stack_offset_y)),
            dest_stack_offset_xy_m=(float(cfg.dest_stack_offset_x), float(cfg.dest_stack_offset_y)),
            stack_jitter_xy_m=float(cfg.stack_jitter_xy),
            stack_jitter_yaw_deg=float(cfg.stack_jitter_yaw_deg),
        )
