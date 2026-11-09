# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility re-exports; the predicates live in ``dexbench_arena.tasks.predicates``."""

from dexbench_arena.tasks.predicates.geometry import (  # noqa: F401
    _per_env_rows,
    _root_quat_w,
    any_object_toppled,
    object_center_outside_xy_bounds,
    object_center_within_xy_bounds,
    object_toppled,
    per_env_rows,
    root_quat_w,
)
from dexbench_arena.tasks.predicates.placement import (  # noqa: F401
    object_resting_on_disk,
    object_standing_in_slot,
)
from dexbench_arena.tasks.predicates.rest import (  # noqa: F401
    REST_ANG_VEL_THRESHOLD_RAD_S,
    REST_GATES_DEFAULT,
    RestGates,
    REST_LIN_VEL_THRESHOLD_M_S,
    SETTLED_ANG_VEL_THRESHOLD_RAD_S,
    object_descending_faster_than,
    object_fell_below_initial,
    object_fell_below_rest,
    object_lifted,
    object_settled,
    objects_within_rest_displacement,
    objects_within_spawn_displacement,
    record_rest_poses,
    rest_pos_w,
)

from dexbench_arena.tasks.predicates.plates import plates_seated_in_stack  # noqa: F401
