# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Load a portable, digest-bound H3 runtime input bundle before starting Kit."""

from pathlib import Path

import yaml

from dexbench_arena.tasks.h3_objects import read_h3_runtime_bundle
from dexbench_arena.tasks.h3_start_state import validate_grasp_start


def load_h3_runtime_bundle(path, digest):
    bundle = read_h3_runtime_bundle(path, digest)
    names = [a['object_name'] for a in bundle['assets']]
    starts = {}
    for row in bundle['starts']:
        data = (Path(path).resolve().parent / row['state']).read_bytes()
        starts[row['id']] = validate_grasp_start(
            dict(state=yaml.safe_load(data), joint_targets=row['joint_targets']), names)
    if 'adaptive_support_profile' in bundle:
        bundle['adaptive_support_profile'] = {k: v for k, v in bundle['adaptive_support_profile'].items()
            if k not in ('opening_resistance_rad', 'opening_follow_rate_rad_s', 'target_opening_scale')}
    bundle['loaded_starts'] = starts
    return bundle
