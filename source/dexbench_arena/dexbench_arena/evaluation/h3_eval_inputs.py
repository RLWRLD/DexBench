# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Resolve H3's frozen selection from portable, digest-bound runtime inputs."""

import json
import math
from pathlib import Path

from dexbench_arena import REPO_ROOT
from dexbench_arena.benchmark.protocol import ROWS_PER_CELL
from dexbench_arena.evaluation.artifacts import file_sha256
from dexbench_arena.tasks.h3_objects import (
    H3_OBJECT_AXIS, H3_OBJECTS, H3_GA_PROTOCOL, read_h3_runtime_bundle, resolve_h3_bundle,
)
from dexbench_arena.tasks.object_axis import task_object_record

SELECTION = REPO_ROOT / 'scripts/evaluation/experiment_configs/db_h3_selected_starts.json'


def cell_rows(inputs, selection, cell, *, verify_assets=True):
    """Resolve every selected row before writing any files; reject changed inputs."""
    if cell not in ('train', 'ga_os'):
        raise ValueError(f'Unsupported H3 cell: {cell}')
    selected = selection.get('cells', {}).get(cell)
    if not selected:
        raise ValueError(f'H3 {cell}: no qualified frozen selection')
    if cell == 'ga_os' and selection.get('ga_os_protocol') != H3_GA_PROTOCOL:
        raise ValueError('H3 GA-OS requires the declared coupled object-and-grasp protocol')
    package = json.loads((Path(inputs) / 'manifest.json').read_text())
    expected_report = selection['native_qualification_reports_sha256']
    if (package.get('qualification_reports_sha256') != expected_report
            or file_sha256(Path(inputs) / 'qualification_reports.json') != expected_report):
        raise ValueError('H3 native qualification evidence differs from the selected report')
    reports = json.loads((Path(inputs) / 'qualification_reports.json').read_text())
    keys = H3_OBJECT_AXIS.keys('train' if cell == 'train' else 'eval')
    count, remainder = divmod(ROWS_PER_CELL, len(keys))
    if remainder or set(selected['bundles']) != set(keys):
        raise ValueError('H3 selection must cover every declared trio/target equally')
    proof_path = Path(inputs) / 'preparation.json'
    if file_sha256(proof_path) != selection['preparation_sha256']:
        raise ValueError('H3 preparation evidence changed')
    proof = json.loads(proof_path.read_text())
    reserved_path = Path(inputs) / f'{cell}_reserved.jsonl'
    if (proof['training_selection_sha256'] != selection['training_plan_sha256']
            or proof['geometry_check']['rule'] != 'utensil_relative_6mm_10deg'
            or not proof['cells'][cell]['row_pool_complete']
            or file_sha256(reserved_path) != proof['cells'][cell]['reserved_sha256']):
        raise ValueError(f'H3 {cell} reservation does not match geometry-checked preparation')
    reserved = {(r['trio'] + '_' + r['target'], r['candidate_id'], r['state_sha256'])
                for line in reserved_path.read_text().splitlines() if (r := json.loads(line))}
    rows_by_key = {}
    training_hashes = set(selection['training_state_sha256'])
    for key in keys:
        expected = selected['bundles'][key]
        path, digest = resolve_h3_bundle(key, inputs)
        if digest != expected['runtime_bundle_sha256']:
            raise ValueError(f'{key}: bundle differs from frozen selection')
        bundle = read_h3_runtime_bundle(path, digest, verify_assets=verify_assets)
        provenance = package['bundles'][key]
        if any(provenance.get(field) != expected[field] for field in
               ('source_runtime_bundle_sha256', 'qualification_report')):
            raise ValueError(f'{key}: native qualification provenance differs from frozen selection')
        report = reports[expected['qualification_report']]
        if (report['runtime_bundle_sha256'] != expected['source_runtime_bundle_sha256']
                or not report['passed'] or report.get('retention_rule') != 'handle_anchor'
                or report.get('drive_profile') != 'collection_v1'
                or report.get('inertia_policy') != 'collection_isotropic_max'):
            raise ValueError(f'{key}: native qualification does not match the input bundle')
        outcomes = {r['id']: r for r in report['start_results']}
        for start in bundle['starts']:
            outcome = outcomes.get(start['id'])
            if outcome is None:
                raise ValueError(f'{key}: selected start lacks native qualification: {start["id"]}')
            if (not outcome['passed'] or not math.isfinite(outcome['hold_seconds'])
                    or outcome['hold_seconds'] < 10):
                raise ValueError(f'{key}: native hold qualification failed')
        if bundle['trio'] + '_' + bundle['target'] != key or bundle['cell'] != cell:
            raise ValueError(f'{key}: bundle has another trio, target or cell')
        assets = {a['type']: {'id': a['id'], 'sha256': a['sha256']} for a in bundle['assets']}
        if assets != H3_OBJECTS[key]['assets']:
            raise ValueError(f'{key}: bundle has other assets')
        starts = {s['id']: s for s in bundle['starts']}
        chosen = expected['starts']
        if (len(chosen) != count or len({r['state_sha256'] for r in chosen}) != count
                or len({r['id'] for r in chosen}) != count):
            raise ValueError(f'{key}: requires {count} distinct saved starts')
        obj = H3_OBJECT_AXIS.get(key)
        record = task_object_record(H3_OBJECT_AXIS, obj,
            dict(runtime_bundle_sha256=digest, utensil_assets=assets, target=bundle['target']), None)
        rows = []
        for row in chosen:
            if (key, row['id'], row['state_sha256']) not in reserved:
                raise ValueError(f'{key}: selected start was not geometry-qualified')
            state = starts.get(row['id'])
            if state is None:
                raise ValueError(f'{key}: selected start is missing: {row["id"]}')
            if state['state_sha256'] != row['state_sha256']:
                raise ValueError(f'{key}: selected state changed')
            if row['state_sha256'] in training_hashes:
                raise ValueError(f'{key}: training state leaked into evaluation')
            rows.append(dict(task_object=record,
                variations={bundle['assets'][-1]['object_name'] + '.start_state': row['id']},
                state_sha256=row['state_sha256'],
                reset_contract_sha256=state['reset_contract_sha256']))
        rows_by_key[key] = rows
    return rows_by_key

