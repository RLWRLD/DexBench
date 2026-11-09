# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""DexBench retail_v1 Arena scene (from PhysicsSim Lab's E1 delivery)."""

from __future__ import annotations

from dataclasses import dataclass, field

import argparse
import math
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentFactory

from dexbench_arena.cli import apply_delivery_render_settings
from dexbench_arena.contact_materials import WUJI_GRASP_CONTACT_MATERIAL
from dexbench_arena.embodiments.lookup import embodiment_class, start_poses
from dexbench_arena.mimic_plugin import mimic_task_class
from dexbench_arena.environments.manipulation.scene_environment_base import (
    TASKS,
    DexBenchEnvironmentCfg,
    DexBenchSceneEnvironmentBase,
)
from dexbench_arena.layouts.e1_retail_fc_layout import (
    FC_260821_LAYOUT_SPEC,
    FC_260821_R3_NEIGHBOR_ALONG_AXIS,
    FC_260821_R3_NEIGHBOR_DEPTH_AXIS,
    FC_260821_R3_NEIGHBOR_INSTANCE_NAMES,
    FC_260821_R3_NEIGHBOR_LOCAL_UP_XYZ,
    FC_260821_R3_NEIGHBOR_MAX_ALONG_M,
    FC_260821_R3_NEIGHBOR_MAX_DEPTH_M,
    FC_260821_R3_NEIGHBOR_MAX_DROP_M,
    FC_260821_R3_NEIGHBOR_MAX_LIFT_M,
    FC_260821_R3_NEIGHBOR_TOPPLE_MAX_UP_ALIGNMENT,
    FC_260821_R3_OBJECT_AXIS,
    FC_260821_R3_RELEASE_DEPTH_WINDOW_M,
    FC_260821_R3_SHELF_TOP_Z_M,
    FC_260821_R3_SLOT_FRAME_INSTANCE_NAME,
    FC_260821_R3_TARGET_ASSET_NAME,
    FC_260821_R3_TARGET_ASSET_NAMES,
    FC_260821_R3_TARGET_INSTANCE_NAME,
    FC_260821_STOCK_LAYOUT,
    FC_260821_STOCK_SHELF2_GROUP,
    build_fc_260821_fixture_assets,
    build_fc_260821_r3_task_assets,
    build_fc_260821_produce_bake_assets,
    build_fc_260821_stock_assets,
    compute_fc_260821_r3_layout,
    fc_260821_r3_facing_width_m,
    fc_260821_resolved_sku,
)
from dexbench_arena.layouts.e1_retail_fc_layout import (
    FC_260821_PRODUCE_CLUTTER_DEFAULT,
    FC_260821_PRODUCE_CLUTTER_SET_IDS,
    FC_260821_PRODUCE_CLUTTER_TARGET_SKUS,
)
from dexbench_arena.environments.manipulation.db_r1_clutter_layout import build_r1_clutter_assets
from dexbench_arena.layouts.db_r1_layout import (
    FC_260821_R1_FIXTURE_INSTANCE_NAMES,
    FC_260821_R1_PILE_TRAY_BIN,
)
from dexbench_arena.layouts.db_r2_layout import (
    build_fc_260821_r2_task_assets,
    fc_260821_r2_stocking,
    FC_260821_R2_DEFAULT_STOCK_FILLERS,
    FC_260821_R2_DEFAULT_STOCK_RULE,
    FC_260821_R2_DEFAULT_TARGET_LANE,
    FC_260821_R2_DEFAULT_TARGET_SKU,
    FC_260821_R2_SHELF2_LANES,
    FC_260821_R2_STOCK_RULES,
    FC_260821_R2_FRIDGE_INSTANCE_NAME,
    FC_260821_R2_OBJECT_AXIS,
    FC_260821_R2_SKU_ASSET_NAMES,
    check_fridge_shelf3_raise,
    parse_r2_lane_draw,
)
from dexbench_arena.layouts.retail_v1_look import (
    FixtureLook,
    RETAIL_V1_FIXTURE_LOOKS,
    RETAIL_V1_LIGHTING_ASSET,
    RETAIL_V1_RENDER_SETTINGS,
    RETAIL_V1_ROOM_LOOK,
    build_fc_260821_retail_v1_dressing_assets,
)
from dexbench_arena.environments.manipulation.db_r1_task_factory import (
    PICK_FRUIT_TASK_NAME,
    R1_PILE_NEWTON_SOLVER_OPTIONS,
    apply_r1_newton_contact,
    make_pick_fruit_from_display_task,
)
from dexbench_arena.tasks.r1_clutter import FRUITS, R1_OBJECT_AXIS, load_recipe, r1_recipe_file
from dexbench_arena.environments.manipulation.db_r2_task_factory import (
    RETRIEVE_BEVERAGE_TASK_NAME,
    apply_r2_newton_contact,
    make_retrieve_beverage_from_clutter_task,
)

if TYPE_CHECKING:
    from isaaclab_arena.scene.scene import Scene


DEFAULT_SCENE_LAYOUT = "fc_260821"
SCENE_LAYOUT_BACKGROUND_ASSETS = {
    "fc_260821": "dexbench_retail_store_e1_fc_260821",
}


def scene_layout_background_asset(scene_layout: str | None) -> str:
    """The registered background asset a ``--scene_layout`` value names."""
    layout = scene_layout or DEFAULT_SCENE_LAYOUT
    try:
        return SCENE_LAYOUT_BACKGROUND_ASSETS[layout]
    except KeyError:
        raise ValueError(
            f"Unknown retail_v1 scene layout {layout!r}; choose one of {sorted(SCENE_LAYOUT_BACKGROUND_ASSETS)}"
        ) from None
FC_SHELF2_SKU_ASSETS = {
    "TC2": "dexbench_retail_e1_fc_tc2",
    "EC1": "dexbench_retail_e1_fc_ec1",
    "TC1": "dexbench_retail_e1_fc_tc1",
    "EC2": "dexbench_retail_e1_fc_ec2",
}
FC_SHELF2_EXPECTED_ITEM_COUNT = 32
PLACE_IN_SLOT_RESTING_CLEARANCE_M = 0.002
PLACE_IN_SLOT_CENTER_HEIGHT_LOWER_MARGIN_M = 0.025
PLACE_IN_SLOT_CENTER_HEIGHT_UPPER_MARGIN_M = 0.010
PLACE_IN_SLOT_VELOCITY_THRESHOLD_M_S = 0.10
PLACE_IN_SLOT_MIN_UP_ALIGNMENT = 0.90
STRICT_NEIGHBOR_DISPLACEMENT_M = 0.005
DEFAULT_SLOT_XY_MARGIN_M = 0.0
WORKSPACE_XY_BOUNDS = ((0.75, 0.20), (2.60, 2.20))
OBJECT_DROPPED_MIN_Z_M = 0.55


def _add_argument_if_missing(parser: argparse.ArgumentParser, *option_strings: str, **kwargs) -> None:
    if any(option in parser._option_string_actions for option in option_strings):
        return
    parser.add_argument(*option_strings, **kwargs)


def _build_fc_shelf2_interaction_assets(asset_registry) -> list:
    """Build the 32 Shelf2 drinks as real rigid entities at the transcribed stock poses."""
    from isaaclab_arena.utils.pose import Pose

    assets = []
    for placement in sorted(
        (p for p in FC_260821_STOCK_LAYOUT if p.group == FC_260821_STOCK_SHELF2_GROUP),
        key=lambda p: p.source_prim_name,
    ):
        sku = fc_260821_resolved_sku(placement.sku)
        try:
            asset_name = FC_SHELF2_SKU_ASSETS[sku]
        except KeyError as error:
            raise RuntimeError(f"No dynamic FC Shelf2 asset registered for SKU {sku!r}") from error
        assets.append(
            asset_registry.get_asset_by_name(asset_name)(
                instance_name=f"fridge_shelf2_{placement.source_prim_name.lower()}",
                initial_pose=Pose(position_xyz=placement.position_xyz, rotation_xyzw=placement.rotation_xyzw),
            )
        )

    if len(assets) != FC_SHELF2_EXPECTED_ITEM_COUNT:
        raise RuntimeError(
            f"Expected {FC_SHELF2_EXPECTED_ITEM_COUNT} FC Shelf2 drinks, found {len(assets)}"
        )
    return assets


def build_e1_retail_scene(
    asset_registry,
    *,
    spawn_slot_items: bool = False,
    spawn_interaction_items: bool = False,
    spawn_full_items: bool = False,
    spawn_r2_task_items: bool = False,
    r2_lane_draw: tuple[int, ...] = (),
    r2_target_sku: str = FC_260821_R2_DEFAULT_TARGET_SKU,
    r2_target_lane: int = FC_260821_R2_DEFAULT_TARGET_LANE,
    r2_stock_rule: str = FC_260821_R2_DEFAULT_STOCK_RULE,
    r2_stock_fillers: str = FC_260821_R2_DEFAULT_STOCK_FILLERS,
    fridge_shelf3_raise_m: float = 0.0,
    spawn_r1_task_items: bool = False,
    r1_target_sku: str = FC_260821_PRODUCE_CLUTTER_DEFAULT[1],
    r1_clutter_file: str | None = None,
    r1_layouts: str | None = None,
    minimal_scene: bool = False,
    produce_clutter_set: int = FC_260821_PRODUCE_CLUTTER_DEFAULT[0],
    produce_clutter_bake: str | None = None,
    r3_target_asset: str = FC_260821_R3_TARGET_ASSET_NAME,
    r3_supply_offset_xy: tuple[float, float] = (0.0, 0.0),
    facing_width_m: float | None = None,
    box_jitter_xy_m: float = 0.0,
    box_jitter_yaw_deg: float = 0.0,
    box_offset_xy_m: tuple[float, float] = (0.0, 0.0),
    scene_layout: str = DEFAULT_SCENE_LAYOUT,
) -> Scene:
    """Build the retail_v1 scene without an embodiment."""

    if spawn_r2_task_items and spawn_slot_items:
        raise ValueError("DB-R2 and DB-R3 task item sets are mutually exclusive; drop --with_r2_items or --with_r3_items")
    if spawn_r1_task_items and (spawn_r2_task_items or spawn_slot_items):
        raise ValueError("DB-R1 is one task set per scene; drop --with_r1_items or the DB-R2 / DB-R3 flags")

    from dexbench_arena.assets import background_library as _background_library  # noqa: F401
    from dexbench_arena.assets import object_library as _object_library  # noqa: F401
    from isaaclab_arena.scene.scene import Scene

    if len(r3_supply_offset_xy) != 2 or not all(math.isfinite(v) for v in r3_supply_offset_xy):
        raise ValueError("r3_supply_offset_xy must contain two finite offsets")
    if any(r3_supply_offset_xy) and scene_layout != "fc_260821":
        raise ValueError("R3 supply offsets require the FC tote-stack layout")
    background = asset_registry.get_asset_by_name(scene_layout_background_asset(scene_layout))(
        look=FixtureLook(None) if minimal_scene else RETAIL_V1_ROOM_LOOK
    )

    light = asset_registry.get_asset_by_name(RETAIL_V1_LIGHTING_ASSET)()

    assets = [light, background]
    fridge_shelf3_raise_m = check_fridge_shelf3_raise(fridge_shelf3_raise_m)
    fixture_kwargs = {FC_260821_R2_FRIDGE_INSTANCE_NAME: {"shelf3_raise_m": fridge_shelf3_raise_m}} if fridge_shelf3_raise_m else {}
    assets.extend(
        build_fc_260821_fixture_assets(
            asset_registry,
            supply_offset_xy=r3_supply_offset_xy,
            fixture_kwargs=fixture_kwargs,
            fixture_looks=RETAIL_V1_FIXTURE_LOOKS,
            instances=FC_260821_R1_FIXTURE_INSTANCE_NAMES if minimal_scene else None,
        )
    )

    if spawn_r1_task_items:
        if r1_target_sku not in FRUITS:
            raise ValueError(f"the DB-R1 pile's --r1_target_sku must be one of {list(FRUITS)}, got {r1_target_sku!r} "
                             "(apple, plum and strawberry only dress the display)")
        if r1_clutter_file is None:
            raise ValueError("the DB-R1 pile needs a settled pile: --r1_layouts <layout set> or --r1_clutter_file <recipe>")
        load_recipe(r1_clutter_file, r1_target_sku)
    elif r1_target_sku not in FC_260821_PRODUCE_CLUTTER_TARGET_SKUS:
        raise ValueError(f"--r1_target_sku must be one of {FC_260821_PRODUCE_CLUTTER_TARGET_SKUS} without the DB-R1 pile, "
                         f"got {r1_target_sku!r}")
    if minimal_scene and (not spawn_r1_task_items or spawn_full_items):
        raise ValueError("--r1_minimal_scene draws only DB-R1's fixtures; it needs the R1 task items and no --with_full_items")
    if produce_clutter_set not in FC_260821_PRODUCE_CLUTTER_SET_IDS:
        raise ValueError(f"--produce_clutter_set must be one of {FC_260821_PRODUCE_CLUTTER_SET_IDS}, got {produce_clutter_set!r}")
    if produce_clutter_bake is not None:
        if spawn_r1_task_items or spawn_r2_task_items or spawn_slot_items or spawn_interaction_items or spawn_full_items:
            raise ValueError("--produce_clutter_bake_json spawns one clutter set as rigid bodies; combine it with no task or stock flags")
    if spawn_r2_task_items and spawn_interaction_items:
        raise ValueError("DB-R2 task items and the 32-drink interaction set would duplicate Shelf2 prims")
    if r2_lane_draw and not spawn_r2_task_items:
        raise ValueError("--r2_lane_draw moves the DB-R2 task set; it needs --task retrieve_beverage_from_clutter")
    if r2_target_sku != FC_260821_R2_DEFAULT_TARGET_SKU and not spawn_r2_task_items:
        raise ValueError("--r2_target_sku picks the DB-R2 target; it needs --task retrieve_beverage_from_clutter")
    r2_stocking = fc_260821_r2_stocking(
        r2_target_sku,
        target_lane=r2_target_lane,
        lane_draw=r2_lane_draw,
        stock_rule=r2_stock_rule,
        stock_fillers=r2_stock_fillers,
        shelf3_raise_m=fridge_shelf3_raise_m,
    )

    if spawn_full_items:
        assets.extend(
            build_fc_260821_stock_assets(
                asset_registry,
                omit_shelf2=spawn_interaction_items,
                omit_source_prim_names=r2_stocking.stock_omissions if spawn_r2_task_items else (),
                extra_placements=r2_stocking.backdrop_stock if spawn_r2_task_items else (),
                omit_r3_task_shelf=spawn_slot_items,
                produce_clutter=(produce_clutter_set,
                                 FC_260821_PRODUCE_CLUTTER_DEFAULT[1] if spawn_r1_task_items else r1_target_sku),
                omit_produce_bin=FC_260821_R1_PILE_TRAY_BIN if spawn_r1_task_items else None,
            )
        )

        assets.extend(
            build_fc_260821_retail_v1_dressing_assets(
                asset_registry, fridge_shelf3_raise_m, r2_stocking.hidden_skus if spawn_r2_task_items else ()
            )
        )

    if spawn_interaction_items:
        assets.extend(_build_fc_shelf2_interaction_assets(asset_registry))
    if spawn_r2_task_items:
        assets.extend(build_fc_260821_r2_task_assets(asset_registry, r2_stocking))
    if spawn_r1_task_items:
        assets.extend(build_r1_clutter_assets(asset_registry, r1_clutter_file, r1_target_sku, r1_layouts))
    if produce_clutter_bake is not None:
        assets.extend(build_fc_260821_produce_bake_assets(asset_registry, produce_clutter_bake, produce_clutter_set, r1_target_sku))
    if spawn_slot_items:
        assets.extend(
            build_fc_260821_r3_task_assets(
                asset_registry,
                facing_width_m=facing_width_m,
                target_asset_name=r3_target_asset,
                box_jitter_xy_m=box_jitter_xy_m,
                box_jitter_yaw_deg=box_jitter_yaw_deg,
                box_offset_xy_m=box_offset_xy_m,
                supply_offset_xy=r3_supply_offset_xy,
            )
        )

    return Scene(assets=assets)


def _xy_pair(text: str) -> tuple[float, float]:
    """argparse type for ``--box_offset_xy``: ``"0.04,0"`` -> ``(0.04, 0.0)``."""
    values = tuple(float(token) for token in text.split(","))
    if len(values) != 2:
        raise argparse.ArgumentTypeError(f"expected X,Y, got {text!r}")
    return values


PLACE_BOX_TASK_NAME = "place_box_in_slot"
_R2_ONLY = {TASKS: (RETRIEVE_BEVERAGE_TASK_NAME,)}
_R3_ONLY = {TASKS: (PLACE_BOX_TASK_NAME,)}
_R1_ONLY = {TASKS: (PICK_FRUIT_TASK_NAME,)}


@dataclass
class DexBenchRetailStoreE1EnvironmentCfg(DexBenchEnvironmentCfg):
    """Typed config for DexBenchRetailStoreE1Environment: its flags, names and defaults."""

    task: str = "none"
    """Task variant the scene builds: ``place_box_in_slot``, ``retrieve_beverage_from_clutter``,
    ``pick_fruit_from_display``, or ``none`` for the scene shell."""

    anchor: str = "Gondola"
    """Robot base anchor name (Fridge, Fruit, Gondola)."""

    scene_layout: str = DEFAULT_SCENE_LAYOUT
    """retail_v1 fixture payload, the name of a registered scene layout (``fc_260821``)."""


    r3_target_asset: str = field(default=FC_260821_R3_TARGET_ASSET_NAME, metadata=_R3_ONLY)
    """Stable R3 target SKU ID; the scene entity remains ``target_carton``."""

    box_jitter_xy: float = field(default=0.0, metadata=_R3_ONLY)
    """Per-reset XY jitter half-extent for the target carton [m], recorded as
    ``target_carton.pose``."""

    box_jitter_yaw_deg: float = field(default=0.0, metadata=_R3_ONLY)
    """Per-reset yaw jitter half-extent [deg]; same recording contract as XY."""

    box_offset_xy: tuple[float, float] | None = field(default=None, metadata=_R3_ONLY)
    """Fixed (x, y) offset of the target carton inside the top tote [m], the centre of the ``box_jitter_xy``
    draw; the tote does not move (that is ``r3_supply_offset_x/y``). Recorded in ``target_carton.pose`` with
    the jitter, as one draw: with no jitter every reset draws the offset. None is no offset, (0, 0): the
    value an episode row without this field reads as (the scorer compares declared fields)."""

    slot_xy_margin: float = field(default=DEFAULT_SLOT_XY_MARGIN_M, metadata=_R3_ONLY)
    """Extra slot clearance for the success check [m]; negative tightens."""

    with_r3_items: bool = False
    """Spawn the DB-R3 carton set and slot frame without ``task`` (scene-shell inspection and renders)."""

    facing_width_mm: float | None = field(default=None, metadata=_R3_ONLY)
    """Empty-facing width [mm]; the shelf repacks and tolerances follow. None derives it from the
    target carton (its width + 40 mm, ``fc_260821_r3_facing_width_m``); an explicit value is for a
    facing sensitivity run."""

    success_tier: str = "collection"
    """'collection' (teleop gate) or 'strict' (benchmark eval metrics; DB-R2 only: DB-R1 and DB-R3 have one tier)."""

    strict_hold_s: float | None = None
    """Hold window for the strict tier [s]; None takes the task's (DB-R2 1)."""

    with_r1_items: bool = False
    """Spawn the DB-R1 pile without ``task`` (scene-shell inspection, renders and pile baking)."""

    r1_target_sku: str = FC_260821_PRODUCE_CLUTTER_DEFAULT[1]
    """With the DB-R1 pile, its kind (``<fruit>_pile``, ``tasks.r1_clutter.FRUITS``); without it, the display
    dressing's fruit in the first tier's middle compartment (apple, plum, strawberry)."""

    r1_clutter_file: str | None = field(default=None, metadata=_R1_ONLY)
    """A settled DB-R1 pile recipe (one layout ``bake_r1_layouts.py`` wrote): one fixed start layout. Takes no
    ``r1_layouts``."""

    r1_layouts: str | None = field(default=None, metadata=_R1_ONLY)
    """Directory of a DB-R1 layout set (``bake_r1_layouts.py``'s ``<r1_target_sku>_<phase>_s<seed>.json``, one
    exact-count family): the scene spawns the first and every reset writes a drawn one over every fruit,
    recorded as ``r1_fruit_000.layout`` (``variations/db_r1_layout_variation.py``); replay its rows
    (``write_r1_layout_rows.py``) to collect each layout in turn. Takes no ``r1_clutter_file``."""

    r1_minimal_scene: bool = field(default=False, metadata=_R1_ONLY)
    """DB-R1 teleop: spawn only the produce display, its cart and basket, and draw none of the room (its floor still
    collides). Nothing the task touches changes, so a replay in the full scene recomputes the same success. Named for
    R1 like its other fields: the R2 and R3 physics and contract tools have a ``--minimal_scene`` of their own."""

    produce_clutter_bake_json: str | None = None
    """Bake mode for ``bake_produce_clutter.py``: this authored JSON's fruit as rigid bodies, nothing else that moves."""

    produce_clutter_set: int = FC_260821_PRODUCE_CLUTTER_DEFAULT[0]
    """Which physics-settled produce clutter set dresses the display."""

    with_r2_items: bool = False
    """Spawn the DB-R2 target drink, its neighbors and task frames without ``task``."""

    with_interaction_items: bool = False
    """Spawn the FC Shelf2 drinks as dynamic entities without the replay backdrop."""

    with_full_items: bool = False
    """Layer the full visual-only stock backdrop (use wherever cameras render)."""

    r3_supply_offset_x: float = 0.0
    """FC R3 tote stack and contents: world X placement offset [m]."""

    r3_supply_offset_y: float = 0.0
    """FC R3 tote stack and contents: world Y placement offset [m]."""

    r2_lane_draw: str | None = field(default=None, metadata=_R2_ONLY)
    """DB-R2 Shelf2 lanes to redraw the task set over at every reset, e.g. ``"3,4,5"`` (it must
    include ``r2_target_lane``); recorded per episode as ``r2_target_drink.lane``. None keeps the
    set in ``r2_target_lane``."""

    r2_target_lane: int = field(default=FC_260821_R2_DEFAULT_TARGET_LANE, metadata=_R2_ONLY)
    """DB-R2 nominal Shelf2 MiniFit lane (1-6) of the target: where it spawns, and where it stays without
    a draw. 6 is the reviewed baseline the published demos were recorded in; the release's cells
    and training demos use 3, 4 and 5, a spatial setting per Run (``benchmark.spec``), not part of
    ``FC_260821_R2_OBJECT_AXIS.fixture_profile``."""

    r2_stock_rule: str = field(default=FC_260821_R2_DEFAULT_STOCK_RULE, metadata=_R2_ONLY)
    """What else DB-R2's row may hold: ``authored`` (the delivery's stock, which repeats four of the
    SKUs in other lanes; the published recordings' scene) or ``unique_target`` (the target SKU only
    in the target lane, conflicting lanes restocked with ``r2_stock_fillers``; the release split)."""

    r2_stock_fillers: str = field(default=FC_260821_R2_DEFAULT_STOCK_FILLERS, metadata=_R2_ONLY)
    """Comma-separated SKUs a lane that would repeat the target is restocked with under
    ``unique_target``: the first that is not the target."""

    r2_target_sku: str = field(default=FC_260821_R2_DEFAULT_TARGET_SKU, metadata=_R2_ONLY)
    """DB-R2 target drink (GA-O), one per run: Train TC1/TC2/TP1, held-out EC1/EC2/EP2/EP3. The PETs
    fit only under a raised Shelf3 (``fridge_shelf3_raise_m``) and are refused with the shortfall
    otherwise. Each episode row records it in ``environment_cfg``."""

    fridge_shelf3_raise_m: float = 0.0
    """Raise of the R2 fridge's Shelf3 and its MiniFit modules [m], a fixture setting one split shares
    (set it in an experiment's ``shared:`` block, never per Run); 0 is the delivered fridge."""


def _millimetres_or_none(value: str) -> float | None:
    """``--facing_width_mm``: a width in mm, or ``none`` for the typed config's None (derive from the carton)."""
    return None if value.lower() == "none" else float(value)


RETAIL_V1_ENVIRONMENT = "dexbench_retail_v1_environment"
LEGACY_RETAIL_ENVIRONMENT = "dexbench_retail_store_e1_environment"


@register_environment
class DexBenchRetailStoreE1Environment(
    DexBenchSceneEnvironmentBase, ArenaEnvironmentFactory[DexBenchRetailStoreE1EnvironmentCfg]
):
    """The retail_v1 store (PhysicsSim Lab's E1 delivery), with the delivery's three robot anchors."""

    name = RETAIL_V1_ENVIRONMENT
    _legacy_argparse_cfg_type = DexBenchRetailStoreE1EnvironmentCfg
    robot_base_anchors_xyz = FC_260821_LAYOUT_SPEC.robot_base_anchors_xyz
    anchor_facing_targets_xy = FC_260821_LAYOUT_SPEC.anchor_facing_targets_xy
    default_anchor = "Gondola"
    anchor_help = "Robot base anchor: Fridge, Fruit, or Gondola station."
    newton_solver_options = {
        "num_substeps": 8,
        "use_mujoco_contacts": True,
        "ccd_iterations": 50,
        "enable_multiccd": True,
    }
    _RETAIL_V1_NEWTON_PIPELINE_OPTIONS = {
        **newton_solver_options,
        "use_mujoco_contacts": False,
        "collision_decimation": 1,
        "ls_iterations": 50,
    }
    task_newton_solver_options = {
        RETRIEVE_BEVERAGE_TASK_NAME: _RETAIL_V1_NEWTON_PIPELINE_OPTIONS,
        PLACE_BOX_TASK_NAME: _RETAIL_V1_NEWTON_PIPELINE_OPTIONS,
        PICK_FRUIT_TASK_NAME: R1_PILE_NEWTON_SOLVER_OPTIONS,
    }
    """The tasks whose solver options are not :attr:`newton_solver_options`. DB-R1's piles take DB-M5's
    Newton setup (``db_r1_task_factory``), with or without the task."""

    @staticmethod
    def _spawns_r1_pile(cfg: DexBenchRetailStoreE1EnvironmentCfg) -> bool:
        return cfg.task == PICK_FRUIT_TASK_NAME or cfg.with_r1_items

    def newton_solver_options_for(self, cfg: DexBenchRetailStoreE1EnvironmentCfg) -> dict:
        if self._spawns_r1_pile(cfg):
            return R1_PILE_NEWTON_SOLVER_OPTIONS
        return type(self).task_newton_solver_options.get(cfg.task, type(self).newton_solver_options)

    def apply_newton_sim_settings(self, env_cfg, cfg: DexBenchRetailStoreE1EnvironmentCfg) -> None:
        """The solver options, DB-R2's own contact on its basket and drinks, and DB-R1's pile contact (with the
        hands' contact on the bodies the embodiment's ``hand_body_label_pattern`` names); every other build removes
        both."""
        super().apply_newton_sim_settings(env_cfg, cfg)
        apply_r2_newton_contact(env_cfg, cfg.task)
        embodiment = embodiment_class(getattr(cfg, "embodiment", None))
        apply_r1_newton_contact(env_cfg, self._spawns_r1_pile(cfg), getattr(embodiment, "hand_body_label_pattern", None))

    def _layout_spec_for_args(self, cfg: DexBenchRetailStoreE1EnvironmentCfg):
        """The layout spec of the background --scene_layout selects."""
        from dexbench_arena.assets import background_library as _background_library  # noqa: F401

        background = self.asset_registry.get_asset_by_name(scene_layout_background_asset(cfg.scene_layout))
        return background.layout_spec

    def _robot_base_anchors_for_args(self, cfg: DexBenchRetailStoreE1EnvironmentCfg):
        return self._layout_spec_for_args(cfg).robot_base_anchors_xyz

    def _anchor_facing_targets_for_args(self, cfg: DexBenchRetailStoreE1EnvironmentCfg):
        return self._layout_spec_for_args(cfg).anchor_facing_targets_xy

    def _embodiment_anchor_overrides_for_args(self, cfg: DexBenchRetailStoreE1EnvironmentCfg, base: str | None):
        overrides = dict(super()._embodiment_anchor_overrides_for_args(cfg, base))
        overrides.update(self._layout_spec_for_args(cfg).anchor_overrides_by_base.get(base, {}))
        return overrides

    @staticmethod
    def _rby1_manipulation_gaze(cfg: DexBenchRetailStoreE1EnvironmentCfg):
        """The embodiment's ``teleop_home`` pose, with the neck aimed at the workspace."""
        return start_poses(cfg.embodiment).get("teleop_home")

    def _make_place_box_in_slot_task(self, cfg: DexBenchRetailStoreE1EnvironmentCfg):
        from dexbench_arena.assets import object_library as _object_library  # noqa: F401
        from dexbench_arena.tasks.place_in_slot import PLACE_IN_SLOT_EPISODE_LENGTH_S, DexBenchPlaceInSlotTask

        margin = cfg.slot_xy_margin

        def _sku_geometry(asset_name: str) -> tuple[tuple[float, float, float], float]:
            """(local bbox center, resting center height) for one SKU."""
            bbox = self.asset_registry.get_asset_by_name(asset_name)(
                instance_name="_geom_probe"
            ).get_bounding_box()
            center_local = tuple(float(v) for v in bbox.center[0].tolist())
            bottom_local_z = float(bbox.min_point[0, 2].item())
            resting = center_local[2] - bottom_local_z + PLACE_IN_SLOT_RESTING_CLEARANCE_M
            return center_local, resting

        object_name = FC_260821_R3_TARGET_INSTANCE_NAME
        slot_ref_name = FC_260821_R3_SLOT_FRAME_INSTANCE_NAME
        shelf_top_z = FC_260821_R3_SHELF_TOP_Z_M
        target_asset_name = cfg.r3_target_asset
        facing_width_m = fc_260821_r3_facing_width_m(target_asset_name, cfg.facing_width_mm)
        fc_layout = compute_fc_260821_r3_layout(
            facing_width_m, target_asset_name=target_asset_name
        )
        slot_center_xy = fc_layout["slot_center_xy"]
        slot_half_x, slot_half_y = fc_layout["slot_halfsize_xy"]
        supply_half_xy = fc_layout["target_half_extents_xy"]
        slot_center_halfsize_xy = (
            max(0.001, slot_half_x - supply_half_xy[0] + margin),
            max(0.001, slot_half_y - supply_half_xy[1] + margin),
        )
        object_center_offset_xyz, _ = _sku_geometry(target_asset_name)
        resting_center_height_m = (
            fc_layout["target_half_height_m"] + PLACE_IN_SLOT_RESTING_CLEARANCE_M
        )
        object_local_up_xyz = fc_layout["target_local_up_xyz"]
        center_height_range_m = (
            max(0.0, resting_center_height_m - PLACE_IN_SLOT_CENTER_HEIGHT_LOWER_MARGIN_M),
            resting_center_height_m + PLACE_IN_SLOT_CENTER_HEIGHT_UPPER_MARGIN_M,
        )
        neighbor_names = list(FC_260821_R3_NEIGHBOR_INSTANCE_NAMES)
        task_description = (
            f"Take the {FC_260821_R3_OBJECT_AXIS.get(target_asset_name).label} from the raised tote "
            "and place it in the empty slot on the shelf."
        )

        if cfg.success_tier != "collection":
            raise ValueError(f"place_box_in_slot has one success tier, 'collection'; got --success_tier {cfg.success_tier}")
        if cfg.strict_hold_s is not None:
            raise ValueError("place_box_in_slot has no strict tier, so --strict_hold_s does not apply")

        return mimic_task_class(DexBenchPlaceInSlotTask)(
            hand_contact_material=WUJI_GRASP_CONTACT_MATERIAL,
            object_name=object_name,
            object_center_offset_xyz=object_center_offset_xyz,
            slot_center_xy=slot_center_xy,
            slot_center_halfsize_xy=slot_center_halfsize_xy,
            object_half_extents_xy=supply_half_xy,
            shelf_top_z_m=shelf_top_z,
            center_height_range_m=center_height_range_m,
            object_local_up_xyz=object_local_up_xyz,
            workspace_xy_bounds=WORKSPACE_XY_BOUNDS,
            object_dropped_min_z_m=OBJECT_DROPPED_MIN_Z_M,
            front_plane_xy=fc_layout["front_plane_xy"],
            outward_xy=fc_layout["outward_xy"],
            neighbor_names=neighbor_names,
            neighbor_reference_m=STRICT_NEIGHBOR_DISPLACEMENT_M,
            neighbor_knock={
                "neighbor_local_up_xyz": FC_260821_R3_NEIGHBOR_LOCAL_UP_XYZ,
                "max_up_alignment": FC_260821_R3_NEIGHBOR_TOPPLE_MAX_UP_ALIGNMENT,
                "max_drop_m": FC_260821_R3_NEIGHBOR_MAX_DROP_M,
                "min_lift_m": FC_260821_R3_NEIGHBOR_MAX_LIFT_M,
                "max_along_m": FC_260821_R3_NEIGHBOR_MAX_ALONG_M,
                "max_depth_m": FC_260821_R3_NEIGHBOR_MAX_DEPTH_M,
                "along_axis": FC_260821_R3_NEIGHBOR_ALONG_AXIS,
                "depth_axis": FC_260821_R3_NEIGHBOR_DEPTH_AXIS,
            },
            min_up_alignment=PLACE_IN_SLOT_MIN_UP_ALIGNMENT,
            velocity_threshold_m_s=PLACE_IN_SLOT_VELOCITY_THRESHOLD_M_S,
            slot_ref_name=slot_ref_name,
            task_description=task_description,
            robot_joint_pos_overrides=self._rby1_manipulation_gaze(cfg),
        )

    available_tasks = {
        PLACE_BOX_TASK_NAME: _make_place_box_in_slot_task,
        RETRIEVE_BEVERAGE_TASK_NAME: make_retrieve_beverage_from_clutter_task,
        PICK_FRUIT_TASK_NAME: make_pick_fruit_from_display_task,
    }
    object_axes = {
        PICK_FRUIT_TASK_NAME: R1_OBJECT_AXIS,
        PLACE_BOX_TASK_NAME: FC_260821_R3_OBJECT_AXIS,
        RETRIEVE_BEVERAGE_TASK_NAME: FC_260821_R2_OBJECT_AXIS,
    }
    shared_settings = (
        "task", "scene_layout", "success_tier", "strict_hold_s", "fridge_shelf3_raise_m",
        "with_r1_items", "with_r2_items", "with_r3_items", "with_interaction_items", "with_full_items",
        "r1_target_sku", "produce_clutter_set", "produce_clutter_bake_json", "r3_supply_offset_x", "r3_supply_offset_y",
    )

    def build_scene(self, cfg: DexBenchRetailStoreE1EnvironmentCfg) -> Scene:
        return build_e1_retail_scene(
            self.asset_registry,
            spawn_slot_items=cfg.task == PLACE_BOX_TASK_NAME or cfg.with_r3_items,
            spawn_r2_task_items=cfg.task == RETRIEVE_BEVERAGE_TASK_NAME or cfg.with_r2_items,
            r2_lane_draw=parse_r2_lane_draw(cfg.r2_lane_draw),
            r2_target_sku=cfg.r2_target_sku,
            r2_target_lane=int(cfg.r2_target_lane),
            r2_stock_rule=cfg.r2_stock_rule,
            r2_stock_fillers=cfg.r2_stock_fillers,
            fridge_shelf3_raise_m=float(cfg.fridge_shelf3_raise_m),
            spawn_r1_task_items=cfg.task == PICK_FRUIT_TASK_NAME or cfg.with_r1_items,
            r1_target_sku=cfg.r1_target_sku,
            r1_clutter_file=r1_recipe_file(cfg),
            r1_layouts=cfg.r1_layouts,
            minimal_scene=cfg.r1_minimal_scene,
            produce_clutter_set=int(cfg.produce_clutter_set),
            produce_clutter_bake=cfg.produce_clutter_bake_json,
            spawn_interaction_items=cfg.with_interaction_items,
            spawn_full_items=cfg.with_full_items,
            r3_target_asset=cfg.r3_target_asset,
            r3_supply_offset_xy=(float(cfg.r3_supply_offset_x), float(cfg.r3_supply_offset_y)),
            facing_width_m=fc_260821_r3_facing_width_m(cfg.r3_target_asset, cfg.facing_width_mm),
            box_jitter_xy_m=cfg.box_jitter_xy,
            box_jitter_yaw_deg=cfg.box_jitter_yaw_deg,
            box_offset_xy_m=tuple(float(value) for value in (cfg.box_offset_xy or (0.0, 0.0))),
            scene_layout=cfg.scene_layout,
        )

    def build(self, cfg: DexBenchRetailStoreE1EnvironmentCfg):
        environment = super().build(cfg)
        previous_callback = environment.env_cfg_callback

        def configure_rendering(env_cfg):
            if previous_callback is not None:
                env_cfg = previous_callback(env_cfg)
            return apply_delivery_render_settings(env_cfg, RETAIL_V1_RENDER_SETTINGS, "the retail_v1 look")

        environment.env_cfg_callback = configure_rendering
        return environment

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        for axis in ("x", "y"):
            _add_argument_if_missing(
                parser, f"--r3_supply_offset_{axis}", type=float, default=0.0,
                help=f"FC R3 tote stack and contents: world {axis.upper()} placement offset [m].",
            )
        super().add_cli_args(parser)
        _add_argument_if_missing(
            parser,
            "--scene_layout",
            "--scene-layout",
            dest="scene_layout",
            choices=tuple(SCENE_LAYOUT_BACKGROUND_ASSETS),
            default=DEFAULT_SCENE_LAYOUT,
            help="retail_v1 fixture payload, a registered scene layout. fc_260821 is the reviewed "
            "six-tote-stack, two-fridge layout with the R3-T2/tote place_box_in_slot baseline.",
        )
        _add_argument_if_missing(
            parser,
            "--r3_target_asset",
            "--r3-target-asset",
            dest="r3_target_asset",
            choices=FC_260821_R3_TARGET_ASSET_NAMES,
            default=FC_260821_R3_TARGET_ASSET_NAME,
            help="SKU used as the FC DB-R3 target: a Train carton (R3-T1..T4) or a held-out Eval "
            "carton (R3-E1, E2, E4, E5, E6). The scene entity remains target_carton.",
        )
        _add_argument_if_missing(
            parser,
            "--box_jitter_xy",
            "--box-jitter-xy",
            dest="box_jitter_xy",
            type=float,
            default=0.0,
            help="Half-extent, in meters, of the per-reset uniform jitter on the supply box's XY "
            "position (0 = pinned). FC records each target_carton.pose draw; leave it at 0 "
            "for teleop and raise one axis at a time for generation/evaluation.",
        )
        _add_argument_if_missing(
            parser,
            "--box_offset_xy",
            "--box-offset-xy",
            dest="box_offset_xy",
            type=_xy_pair,
            default=None,
            metavar="X,Y",
            help="Fixed offset, in meters, of the supply box inside the top tote, e.g. 0.04,0: the centre "
            "of the --box_jitter_xy draw (the tote stays). Recorded with the jitter as target_carton.pose.",
        )
        _add_argument_if_missing(
            parser,
            "--box_jitter_yaw_deg",
            "--box-jitter-yaw-deg",
            dest="box_jitter_yaw_deg",
            type=float,
            default=0.0,
            help="Half-extent, in degrees, of the per-reset uniform jitter on the supply box's "
            "yaw around its standing orientation (0 = fixed). FC records the same pose factor.",
        )
        _add_argument_if_missing(
            parser,
            "--slot_xy_margin",
            "--slot-xy-margin",
            dest="slot_xy_margin",
            type=float,
            default=DEFAULT_SLOT_XY_MARGIN_M,
            help="Extra allowance, in meters, on the slot success rectangle. 0 requires geometric "
            "containment in the empty facing; positive loosens the tier, negative tightens it.",
        )
        _add_argument_if_missing(
            parser,
            "--with_r3_items",
            "--with-r3-items",
            dest="with_r3_items",
            action="store_true",
            help="Spawn the DB-R3 carton neighbours, the target carton in the tote and the slot frame even without --task "
            "(for scene-shell inspection and renders).",
        )
        _add_argument_if_missing(
            parser,
            "--facing_width_mm",
            "--facing-width-mm",
            dest="facing_width_mm",
            type=_millimetres_or_none,
            default=None,
            help="Empty-facing width on shelf 3, in millimeters, for a facing sensitivity run; the shelf "
            "repacks and the success rectangle follows it. Unset or 'none' derives it from the target "
            "carton (its width + 40 mm), as an experiment's facing_width_mm: null does.",
        )
        _add_argument_if_missing(
            parser,
            "--success_tier",
            "--success-tier",
            dest="success_tier",
            type=str,
            default="collection",
            choices=("collection", "strict"),
            help="Success gate tier. 'collection' is the teleop/record gate; 'strict' is DB-R2's "
            "benchmark eval metrics, sustained for --strict_hold_s. DB-R1 (pick_fruit_from_display) and DB-R3 "
            "(place_box_in_slot) have one tier, 'collection'.",
        )
        _add_argument_if_missing(
            parser,
            "--strict_hold_s",
            "--strict-hold-s",
            dest="strict_hold_s",
            type=float,
            default=None,
            help="Hold window for --success_tier strict, in seconds (default: DB-R2 1).",
        )
        _add_argument_if_missing(
            parser,
            "--with_r1_items",
            "--with-r1-items",
            dest="with_r1_items",
            action="store_true",
            help="For fc_260821, spawn the DB-R1 pile (--r1_target_sku <fruit>_pile from --r1_layouts or "
            "--r1_clutter_file) even without --task (scene-shell inspection, renders and pile baking).",
        )
        _add_argument_if_missing(
            parser,
            "--r1_target_sku",
            "--r1-target-sku",
            dest="r1_target_sku",
            type=str,
            choices=list(FC_260821_PRODUCE_CLUTTER_TARGET_SKUS) + list(FRUITS),
            default=FC_260821_PRODUCE_CLUTTER_DEFAULT[1],
            help="With the DB-R1 pile (--task pick_fruit_from_display or --with_r1_items), its kind (<fruit>_pile); "
            "otherwise the display dressing's fruit in the first tier's middle compartment.",
        )
        _add_argument_if_missing(parser, "--r1_clutter_file", type=str, default=None,
                                 help="A settled DB-R1 pile recipe (one layout bake_r1_layouts.py wrote): one fixed start layout.")
        _add_argument_if_missing(parser, "--r1_layouts", type=str, default=None,
                                 help="Directory of a DB-R1 layout set (bake_r1_layouts.py's out-dir): spawn its first "
                                 "layout and draw one at every reset (r1_fruit_000.layout). Takes no --r1_clutter_file.")
        _add_argument_if_missing(
            parser,
            "--produce_clutter_bake_json",
            "--produce-clutter-bake-json",
            dest="produce_clutter_bake_json",
            type=str,
            default=None,
            help="Bake mode for the maintainers' produce clutter bake: spawn every fruit of the (--produce_clutter_set, "
            "--r1_target_sku) set in this authored JSON as a rigid body and nothing else that moves, so Arena's physics "
            "and settle check produce the display's poses.",
        )
        _add_argument_if_missing(
            parser,
            "--produce_clutter_set",
            "--produce-clutter-set",
            dest="produce_clutter_set",
            type=int,
            choices=list(FC_260821_PRODUCE_CLUTTER_SET_IDS),
            default=FC_260821_PRODUCE_CLUTTER_DEFAULT[0],
            help="Which physics-settled produce clutter set dresses the display.",
        )
        _add_argument_if_missing(
            parser,
            "--with_r2_items",
            "--with-r2-items",
            dest="with_r2_items",
            action="store_true",
            help="For fc_260821, spawn the DB-R2 target drink, its watched neighbors and the two "
            "task frames even without --task (scene-shell inspection and renders).",
        )
        _add_argument_if_missing(
            parser,
            "--r2_lane_draw",
            "--r2-lane-draw",
            dest="r2_lane_draw",
            type=str,
            default=None,
            help="DB-R2: Shelf2 MiniFit lanes (1-6, e.g. 3,4,5; must include --r2_target_lane) to redraw the "
            "target, its watched neighbors and the opening frame over at every reset. Each episode records "
            "its lane as r2_target_drink.lane. The target pair is the target SKU, the lateral drinks the "
            "filler (--r2_stock_rule unique_target) or the target SKU (authored), and the visual backdrop "
            "drops the front three slots of every lane the draw can occupy. Unset keeps the set in --r2_target_lane.",
        )
        _add_argument_if_missing(
            parser,
            "--r2_target_lane",
            "--r2-target-lane",
            dest="r2_target_lane",
            type=int,
            choices=FC_260821_R2_SHELF2_LANES,
            default=FC_260821_R2_DEFAULT_TARGET_LANE,
            help="DB-R2 nominal Shelf2 MiniFit lane of the target (6 = the reviewed baseline; the release "
            "split uses 4). Its watched neighbors are the rear drink and the lateral lanes' front and second "
            "drinks, on both sides for a middle lane.",
        )
        _add_argument_if_missing(
            parser,
            "--r2_stock_rule",
            "--r2-stock-rule",
            dest="r2_stock_rule",
            choices=FC_260821_R2_STOCK_RULES,
            default=FC_260821_R2_DEFAULT_STOCK_RULE,
            help="DB-R2 row: 'authored' keeps the delivery's stock (other lanes repeat TC1, TC2, EC1, EC2); "
            "'unique_target' restocks every other lane that would show the target SKU with --r2_stock_fillers, "
            "and refuses to build a row that still repeats it.",
        )
        _add_argument_if_missing(
            parser,
            "--r2_stock_fillers",
            "--r2-stock-fillers",
            dest="r2_stock_fillers",
            type=str,
            default=FC_260821_R2_DEFAULT_STOCK_FILLERS,
            help="DB-R2 unique_target filler SKUs, comma-separated; a conflicting lane takes the first that is not the target.",
        )
        _add_argument_if_missing(
            parser,
            "--r2_target_sku",
            "--r2-target-sku",
            dest="r2_target_sku",
            choices=tuple(FC_260821_R2_SKU_ASSET_NAMES),
            default=FC_260821_R2_DEFAULT_TARGET_SKU,
            help="DB-R2 target drink, one per run (GA-O): Train TC1, TC2, TP1; held-out EC1, EC2, EP2, EP3. Its "
            "lane is re-packed at its pitch. The PETs (TP1, EP2, EP3) "
            "fit only with --fridge_shelf3_raise_m and are refused with the measured shortfall without it.",
        )
        _add_argument_if_missing(
            parser,
            "--fridge_shelf3_raise_m",
            "--fridge-shelf3-raise-m",
            dest="fridge_shelf3_raise_m",
            type=float,
            default=0.0,
            help="DB-R2 fixture profile: raise the task fridge's Shelf3 and its MiniFit modules by this many metres, "
            "the same for every run of a split. 0 (default) is the delivered fridge.",
        )
        _add_argument_if_missing(
            parser,
            "--with_interaction_items",
            "--with-interaction-items",
            dest="with_interaction_items",
            action="store_true",
            help="For fc_260821, spawn the 32 Shelf2 drinks as dynamic interaction entities. "
            "This is independent from the visual-only --with_full_items backdrop.",
        )
        _add_argument_if_missing(
            parser,
            "--with_full_items",
            "--with-full-items",
            dest="with_full_items",
            action="store_true",
            help="Layer the delivery's non-interaction display items as replay-only visual "
            "scenery (rigid bodies disabled, not in scene state). On fc_260821 Shelf2 is "
            "visual-only unless --with_interaction_items explicitly requests dynamic drinks.",
        )
        _add_argument_if_missing(
            parser,
            "--r1_minimal_scene",
            "--r1-minimal-scene",
            dest="r1_minimal_scene",
            action="store_true",
            help="DB-R1 teleop: spawn only the produce display, its cart and basket, and draw none of the room "
            "(its floor still collides).",
        )


@dataclass
class DexBenchRetailStoreE1LegacyIdEnvironmentCfg(DexBenchRetailStoreE1EnvironmentCfg):
    """The retail_v1 config under its alias id: the same fields and defaults."""


@register_environment
class DexBenchRetailStoreE1LegacyIdEnvironment(
    DexBenchRetailStoreE1Environment, ArenaEnvironmentFactory[DexBenchRetailStoreE1LegacyIdEnvironmentCfg]
):
    """``dexbench_retail_store_e1_environment``, an alias id of the retail_v1 store."""

    name = LEGACY_RETAIL_ENVIRONMENT
    _legacy_argparse_cfg_type = DexBenchRetailStoreE1LegacyIdEnvironmentCfg
