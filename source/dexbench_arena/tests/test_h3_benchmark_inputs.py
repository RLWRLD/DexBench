# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""H3's categorical rows retain their input identity at strict replay admission."""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json

import pytest
import yaml

from dexbench_arena.evaluation.replay_guards import ObjectIdentity
from dexbench_arena.tasks import h3_objects
from dexbench_arena.tasks.h3_objects import H3_OBJECT_AXIS, H3_OBJECTS, H3_GA_PROTOCOL
from dexbench_arena.tasks.object_axis import task_object_record


@dataclass
class Config:
    h3_object: str = 'train_01_spoon'
    runtime_bundle: str = ''
    runtime_bundle_sha256: str = ''
    h3_inputs_dir: str = ''


def test_replay_identity_matches_recorded_fit_and_rejects_other_inputs(tmp_path):
    key = 'train_01_spoon'
    (tmp_path / 'runtime').mkdir()
    bundle_path = tmp_path / 'runtime/train_01_spoon.json'
    bundle_path.write_text(json.dumps({'format': 'dexbench_h3_runtime_v2'}))
    digest = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(format='dexbench_h3_inputs_v2',
        bundles={key: dict(path='runtime/train_01_spoon.json', sha256=digest)})))
    cfg = Config(h3_inputs_dir=str(tmp_path))
    obj, fit = H3_OBJECT_AXIS.resolve(cfg)
    row = {'task_object': task_object_record(H3_OBJECT_AXIS, obj, fit, None)}
    identity = ObjectIdentity.of(H3_OBJECT_AXIS, cfg)
    assert identity.differences([('saved row', row)]) == []
    assert fit == dict(runtime_bundle_sha256=digest,
        utensil_assets=H3_OBJECTS[key]['assets'], target='spoon')
    for field, replacement in [('runtime_bundle_sha256', 'b' * 64), ('target', 'knife'),
                               ('utensil_assets', {})]:
        changed = deepcopy(row)
        changed['task_object']['fit'][field] = replacement
        assert identity.differences([('changed row', changed)])
    explicit = Config(runtime_bundle=str(bundle_path), runtime_bundle_sha256=digest)
    assert ObjectIdentity.of(H3_OBJECT_AXIS, explicit).differences([('saved row', row)]) == []


def test_ga_os_requires_an_explicit_protocol_and_qualified_selection(tmp_path):
    from dexbench_arena.evaluation import h3_eval_inputs as writer
    for cells in ({}, {'ga_os': {'bundles': {}}}):
        with pytest.raises(ValueError, match='qualified frozen selection|declared coupled object-and-grasp protocol'):
            writer.cell_rows(tmp_path, {'cells': cells}, 'ga_os')


@pytest.mark.parametrize("cell", ("train", "ga_os"))
def test_row_writer_resolves_full_cell_and_refuses_changed_states_or_training_leaks(tmp_path, monkeypatch, cell):
    from dexbench_arena.evaluation import h3_eval_inputs as writer
    digest = lambda data: hashlib.sha256(data).hexdigest()
    reports = {}
    report = b'{}'
    (tmp_path / 'qualification_reports.json').write_bytes(report)
    manifest = dict(format='dexbench_h3_inputs_v2', bundles={}, qualification_reports_sha256=digest(report))
    selection = dict(training_state_sha256=[], native_qualification_reports_sha256=digest(report),
                     ga_os_protocol=H3_GA_PROTOCOL, cells={cell: {'bundles': {}}})
    package = tmp_path / 'assets/object_library/utensil/usd'
    package.mkdir(parents=True)
    (package / 'utensil.usd').write_bytes(b'geometry')
    monkeypatch.setattr(h3_objects, 'ASSETS_DIR', tmp_path / 'assets')
    palm = {'t': [1., 0., 0.], 'd': [0., 1., 0.]}
    for key in H3_OBJECT_AXIS.keys('train' if cell == 'train' else 'eval'):
        declaration = deepcopy(H3_OBJECTS[key])
        assets = []
        for kind, asset in declaration['assets'].items():
            asset['sha256'] = digest(b'geometry')
            assets.append(dict(asset, type=kind, object_name=kind, usd_path='object_library/utensil/usd/utensil.usd', files={}))
        monkeypatch.setitem(H3_OBJECTS, key, declaration)
        starts = []
        for index in range(10):
            body = dict(root_pose=[float(index), 0., 1., 0., 0., 0., 1.], root_velocity=[0.] * 6)
            state = dict(format='dexbench_h3_state_v1', frame='env', quaternion_order='xyzw',
                scene_asset='dexbench_hotel_fb_e4_260831',
                robot=dict(body, joint_positions={'finger': .3}, joint_velocities={'finger': 0.}),
                objects={a['object_name']: deepcopy(body) for a in assets})
            state['objects']['fork']['root_pose'][1] = index * .02
            data = yaml.safe_dump(state).encode()
            name = key + str(index) + '.yaml'
            (tmp_path / name).write_bytes(data)
            starts.append(dict(id=str(index), state=name, state_sha256=digest(data),
                joint_targets={'finger': .7}, reset_contract_sha256='r' * 64))
        bundle = dict(format='dexbench_h3_runtime_v2', trio=declaration['trio'], target=declaration['target'],
            cell=cell, assets=assets, starts=starts, palm_local=palm)
        data = json.dumps(bundle).encode()
        (tmp_path / (key + '.json')).write_bytes(data)
        manifest['bundles'][key] = dict(path=key+'.json', sha256=digest(data),
            source_runtime_bundle_sha256=digest(data), qualification_report=key)
        reports[key] = dict(runtime_bundle_sha256=digest(data), passed=True,
            retention_rule='handle_anchor', drive_profile='collection_v1', inertia_policy='collection_isotropic_max',
            start_results=[dict(id=r['id'], passed=True, hold_seconds=10.) for r in starts])
        selection['cells'][cell]['bundles'][key] = dict(runtime_bundle_sha256=digest(data),
            source_runtime_bundle_sha256=digest(data), qualification_report=key,
            starts=[{k: r[k] for k in ('id', 'state_sha256')} for r in starts])
    reserved = [dict(trio=H3_OBJECTS[key]['trio'], target=H3_OBJECTS[key]['target'],
                     candidate_id=r['id'], state_sha256=r['state_sha256'])
                for key, value in selection['cells'][cell]['bundles'].items() for r in value['starts']]
    data = ('\n'.join(json.dumps(r) for r in reserved) + '\n').encode()
    (tmp_path / f'{cell}_reserved.jsonl').write_bytes(data)
    proof = dict(training_selection_sha256='p' * 64,
        geometry_check={'rule': 'utensil_relative_6mm_10deg'},
        cells={cell: dict(row_pool_complete=True, reserved_sha256=digest(data))})
    data = json.dumps(proof).encode()
    (tmp_path / 'preparation.json').write_bytes(data)
    selection.update(preparation_sha256=digest(data), training_plan_sha256='p' * 64)
    report = json.dumps(reports).encode()
    (tmp_path / 'qualification_reports.json').write_bytes(report)
    manifest['qualification_reports_sha256'] = selection['native_qualification_reports_sha256'] = digest(report)
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    rows = writer.cell_rows(tmp_path, selection, cell)
    assert sum(map(len, rows.values())) == 120
    assert {r['task_object']['fit']['target'] for rs in rows.values() for r in rs} == {'fork', 'spoon', 'knife'}
    for key, records in rows.items():
        identity = ObjectIdentity.of(H3_OBJECT_AXIS, Config(h3_object=key, h3_inputs_dir=str(tmp_path)))
        assert identity.differences([(key, r) for r in records]) == []
    changed = deepcopy(manifest)
    first, second = list(changed['bundles'])[:2]
    for field in ('source_runtime_bundle_sha256', 'qualification_report'):
        changed['bundles'][first][field] = changed['bundles'][second][field]
    (tmp_path / 'manifest.json').write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='qualification provenance differs'):
        writer.cell_rows(tmp_path, selection, cell)
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    bad = deepcopy(selection)
    bad['training_state_sha256'] = [next(iter(rows.values()))[0]['state_sha256']]
    with pytest.raises(ValueError, match='training state leaked'):
        writer.cell_rows(tmp_path, bad, cell)
    (tmp_path / f'{cell}_reserved.jsonl').write_bytes(b'changed reserved rows')
    with pytest.raises(ValueError, match='geometry-checked preparation'):
        writer.cell_rows(tmp_path, selection, cell)
    (tmp_path / f'{cell}_reserved.jsonl').write_text('\n'.join(json.dumps(r) for r in reserved) + '\n')
    failed_reports = deepcopy(reports)
    failed_reports[first]['passed'] = False
    data = json.dumps(failed_reports).encode()
    bad_manifest, bad_selection = deepcopy(manifest), deepcopy(selection)
    bad_manifest['qualification_reports_sha256'] = bad_selection['native_qualification_reports_sha256'] = digest(data)
    (tmp_path / 'qualification_reports.json').write_bytes(data)
    (tmp_path / 'manifest.json').write_text(json.dumps(bad_manifest))
    with pytest.raises(ValueError, match='native qualification does not match'):
        writer.cell_rows(tmp_path, bad_selection, cell)
    for seconds in (float('nan'), float('inf')):
        failed_reports = deepcopy(reports)
        failed_reports[first]['start_results'][0]['hold_seconds'] = seconds
        data = json.dumps(failed_reports).encode()
        bad_manifest['qualification_reports_sha256'] = bad_selection['native_qualification_reports_sha256'] = digest(data)
        (tmp_path / 'qualification_reports.json').write_bytes(data)
        (tmp_path / 'manifest.json').write_text(json.dumps(bad_manifest))
        with pytest.raises(ValueError, match='native hold qualification failed'):
            writer.cell_rows(tmp_path, bad_selection, cell)
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    (tmp_path / 'qualification_reports.json').write_bytes(b'changed evidence')
    with pytest.raises(ValueError, match='qualification evidence differs'):
        writer.cell_rows(tmp_path, selection, cell)
    (tmp_path / 'qualification_reports.json').write_bytes(report)
    (package / 'utensil.usd').write_bytes(b'changed geometry')
    with pytest.raises(ValueError, match='asset SHA-256 mismatch'):
        writer.cell_rows(tmp_path, selection, cell)
    assert writer.cell_rows(tmp_path, selection, cell, verify_assets=False) == rows
    (package / 'utensil.usd').write_bytes(b'geometry')
    (tmp_path / (('train_01' if cell == 'train' else 'ga_os_01') + '_fork0.yaml')).write_text('changed state')
    with pytest.raises(ValueError, match='state SHA-256 mismatch'):
        writer.cell_rows(tmp_path, selection, cell)
