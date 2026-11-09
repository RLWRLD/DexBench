# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""DB-R1 pick fruit from a pile: CLI -> pile -> task, on the FC produce display."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dexbench_arena.environments.manipulation.db_r1_clutter_layout import fruit_geometry
from dexbench_arena.environments.manipulation.newton_isotropic_inertia import (
    install_isotropic_inertia,
)
from dexbench_arena.environments.manipulation.newton_rolling_friction import (
    ROLLING_CONDIM,
    install_rolling_friction,
)
from dexbench_arena.environments.manipulation.newton_shape_contact import (
    ShapeContactRequest,
    install_shape_contact,
)
from dexbench_arena.environments.manipulation.newton_solref import solref_to_newton_ke_kd
from dexbench_arena.evaluation.evaluation_signature import NEWTON_SHAPE_OVERRIDES_ATTRIBUTE
from dexbench_arena.layouts.db_r1_layout import (
    FC_260821_R1_ANCHOR,
    FC_260821_R1_BASKET_BOUNDS,
    FC_260821_R1_BASKET_MARGIN_M,
    FC_260821_R1_EPISODE_LENGTH_S,
    FC_260821_R1_HOLD_S,
    FC_260821_R1_OBJECT_DROPPED_MIN_Z_M,
    FC_260821_R1_RELEASE_WEIGHT_FRACTION,
    FC_260821_R1_WORKSPACE_XY_BOUNDS,
)
from dexbench_arena.tasks.r1_clutter import FRUITS, PILE_TRAY_BOUNDS, load_recipe, r1_recipe_file

if TYPE_CHECKING:
    from dexbench_arena.environments.manipulation.e1_retail_environment import DexBenchRetailStoreE1EnvironmentCfg

PICK_FRUIT_TASK_NAME = "pick_fruit_from_display"

R1_PILE_NEWTON_SOLVER_OPTIONS = {
    "num_substeps": 8,
    "solver": "newton",
    "use_mujoco_contacts": False,
    "collision_decimation": 1,
    "tolerance": 1e-6,
    "iterations": 100,
    "ls_iterations": 50,
    "nconmax": 1024,
    "njmax": 4096,
    "bvh_constructor_geometry": "sah",
}
R1_PILE_SOLREF = (0.02, 2.0)
R1_PILE_CONTACT_GAP_M = 0.002
R1_FRUIT_ROLLING_FRICTION_M = 2.0e-4
R1_FRUIT_SHAPE_PATTERN = r"/r1_fruit_\d+/"
R1_FRUIT_BODY_TOKENS = ("/r1_fruit_",)
R1_FRUIT_ROLLING_FRICTION_NAME = "r1_fruit_rolling_friction"
R1_FRUIT_ISOTROPIC_INERTIA_NAME = "r1_fruit_isotropic_inertia"
R1_HAND_CONTACT_NAME = "r1_hand_contact"
R1_HAND_CONTACT_SOLREF = (0.02, 1.0)
R1_HAND_CONTACT_SOLMIX = 1.0


def apply_r1_newton_contact(env_cfg, pile: bool, hand_body_pattern: str | None = None) -> None:
    """Install DB-R1's pile physics for a pile build (``pile``); any other build installs none."""

    if not pile:
        return
    shape = env_cfg.sim.physics.default_shape_cfg
    shape.gap = R1_PILE_CONTACT_GAP_M
    shape.ke, shape.kd = solref_to_newton_ke_kd(*R1_PILE_SOLREF)
    own = {R1_FRUIT_ROLLING_FRICTION_NAME, R1_FRUIT_ISOTROPIC_INERTIA_NAME, R1_HAND_CONTACT_NAME}
    recorded = {kind: {name: v for name, v in entries.items() if name not in own}
                for kind, entries in (getattr(env_cfg, NEWTON_SHAPE_OVERRIDES_ATTRIBUTE, None) or {}).items()}
    install_rolling_friction(R1_FRUIT_ROLLING_FRICTION_NAME, R1_FRUIT_SHAPE_PATTERN, R1_FRUIT_ROLLING_FRICTION_M)
    recorded["rolling_friction"] = {**recorded.get("rolling_friction", {}), R1_FRUIT_ROLLING_FRICTION_NAME: {
        "pattern": R1_FRUIT_SHAPE_PATTERN, "mu_rolling": R1_FRUIT_ROLLING_FRICTION_M, "condim": ROLLING_CONDIM}}
    install_isotropic_inertia(R1_FRUIT_ISOTROPIC_INERTIA_NAME, R1_FRUIT_BODY_TOKENS, "max")
    recorded["isotropic_inertia"] = {R1_FRUIT_ISOTROPIC_INERTIA_NAME: {"tokens": list(R1_FRUIT_BODY_TOKENS),
                                                                       "mode": "max"}}
    if hand_body_pattern is not None:
        ke, kd = solref_to_newton_ke_kd(*R1_HAND_CONTACT_SOLREF)
        install_shape_contact(ShapeContactRequest(R1_HAND_CONTACT_NAME, hand_body_pattern, ke, kd,
                                                  R1_HAND_CONTACT_SOLMIX, match="body"))
        recorded["shape_contact"] = {**recorded.get("shape_contact", {}), R1_HAND_CONTACT_NAME: {
            "pattern": hand_body_pattern, "match": "body", "solref": list(R1_HAND_CONTACT_SOLREF),
            "ke": ke, "kd": kd, "solmix": R1_HAND_CONTACT_SOLMIX}}
    setattr(env_cfg, NEWTON_SHAPE_OVERRIDES_ATTRIBUTE, {kind: entries for kind, entries in recorded.items() if entries})


def make_pick_fruit_from_display_task(environment, cfg: DexBenchRetailStoreE1EnvironmentCfg):
    """Build DB-R1 for the CLI's pile kind and its settled pile (``--r1_layouts`` or ``--r1_clutter_file``)."""
    from dexbench_arena.assets.object_library import RETAIL_E1_PRODUCE_REST_GATES
    from dexbench_arena.tasks.predicates.release import HAND_RELEASED_MAX_FORCE_N
    from dexbench_arena.tasks.r1_pick import R1PickTask

    if cfg.scene_layout != "fc_260821":
        raise ValueError(
            f"--task {PICK_FRUIT_TASK_NAME} requires --scene_layout fc_260821 "
            "(the layout with the produce display and the fruit basket)."
        )
    anchor = cfg.anchor
    if anchor and anchor != FC_260821_R1_ANCHOR:
        print(f"[dexbench] WARNING: --task {PICK_FRUIT_TASK_NAME} is reviewed at --anchor {FC_260821_R1_ANCHOR}; got {anchor!r}.")
    if cfg.r1_target_sku not in FRUITS:
        raise ValueError(
            f"--task {PICK_FRUIT_TASK_NAME} picks from a pile: --r1_target_sku must be one of {list(FRUITS)}, "
            f"got {cfg.r1_target_sku!r} (apple, plum and strawberry only dress the display)."
        )
    if cfg.success_tier != "collection":
        raise ValueError(f"--task {PICK_FRUIT_TASK_NAME} has one success tier (collection); got --success_tier {cfg.success_tier}")
    if cfg.strict_hold_s is not None:
        raise ValueError(f"--task {PICK_FRUIT_TASK_NAME} has no strict tier; its hold is {FC_260821_R1_HOLD_S} s (drop --strict_hold_s)")
    recipe_file = r1_recipe_file(cfg)
    if recipe_file is None:
        raise ValueError(f"--task {PICK_FRUIT_TASK_NAME} needs a settled pile: --r1_layouts <layout set> or --r1_clutter_file <recipe>")

    recipe = load_recipe(recipe_file, cfg.r1_target_sku, settled=True)
    return R1PickTask(
        recipe,
        fruit_geometry(environment.asset_registry, recipe),
        source_bounds=PILE_TRAY_BOUNDS,
        basket_bounds=FC_260821_R1_BASKET_BOUNDS,
        workspace_xy_bounds=FC_260821_R1_WORKSPACE_XY_BOUNDS,
        dropped_min_z_m=FC_260821_R1_OBJECT_DROPPED_MIN_Z_M,
        basket_margin_m=FC_260821_R1_BASKET_MARGIN_M,
        released_max_force_n=HAND_RELEASED_MAX_FORCE_N,
        released_max_weight_fraction=FC_260821_R1_RELEASE_WEIGHT_FRACTION,
        still_gates=RETAIL_E1_PRODUCE_REST_GATES,
        hold_time_s=FC_260821_R1_HOLD_S,
        episode_length_s=FC_260821_R1_EPISODE_LENGTH_S,
        robot_joint_pos_overrides=environment._rby1_manipulation_gaze(cfg),
    )
