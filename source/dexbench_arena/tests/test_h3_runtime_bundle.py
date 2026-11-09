# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""A relocated runtime bundle keeps measured q, commands and asset-pool file identity distinct."""
import hashlib
import json

import pytest
import yaml

from dexbench_arena.tasks import h3_objects
from dexbench_arena.tasks.h3_runtime_bundle import load_h3_runtime_bundle


def test_bundle_checks_pool_package_bytes_before_restoring(tmp_path, monkeypatch):
    objects = ['fork_body', 'knife_body', 'spoon_body']
    body = dict(root_pose=[0., 0., 1., 0., 0., 0., 1.], root_velocity=[0.] * 6)
    state = dict(format='dexbench_h3_state_v1', frame='env', quaternion_order='xyzw',
                 scene_asset='dexbench_hotel_fb_e4_260831',
                 robot=dict(body, joint_positions={'finger': .3}, joint_velocities={'finger': 0.}),
                 objects={name: body for name in objects})
    data = yaml.safe_dump(state).encode()
    (tmp_path / 'state.yaml').write_bytes(data)
    assets = tmp_path / 'assets'
    usd = assets / 'object_library/utensil/usd'
    (usd / 'deps').mkdir(parents=True)
    (usd / 'utensil.usd').write_bytes(b'wrapper layer')
    (usd / 'deps/mesh.usd').write_bytes(b'referenced geometry')
    monkeypatch.setattr(h3_objects, 'ASSETS_DIR', assets)
    digest = lambda b: hashlib.sha256(b).hexdigest()
    bundle = dict(format='dexbench_h3_runtime_v2', target='knife', assets=[
        dict(id=name, type=kind, object_name=name, usd_path='object_library/utensil/usd/utensil.usd',
             sha256=digest(b'wrapper layer'), files={'deps/mesh.usd': digest(b'referenced geometry')})
        for name, kind in zip(objects, ('fork', 'knife', 'spoon'))], starts=[
        dict(id='a', state='state.yaml', state_sha256=digest(data), joint_targets={'finger': .7})])
    bundle['adaptive_support_profile'] = dict(hold_preload_rad=.001, opening_resistance_rad=.04,
                                             opening_follow_rate_rad_s=.04, target_opening_scale=.25)
    payload = json.dumps(bundle).encode()
    path = tmp_path / 'bundle.json'
    path.write_bytes(payload)
    loaded = load_h3_runtime_bundle(path, digest(payload))
    assert loaded['adaptive_support_profile'] == {'hold_preload_rad': .001}
    assert path.read_bytes() == payload
    assert loaded['loaded_starts']['a']['state']['robot']['joint_positions']['finger'] == .3
    assert loaded['loaded_starts']['a']['joint_targets']['finger'] == .7
    assert loaded['assets'][0]['usd_path'] == str(usd / 'utensil.usd')
    (usd / 'deps/mesh.usd').write_bytes(b'changed referenced geometry')
    with pytest.raises(ValueError, match='asset dependency SHA-256 mismatch'):
        load_h3_runtime_bundle(path, digest(payload))
    assert h3_objects.read_h3_runtime_bundle(path, digest(payload), verify_assets=False)['target'] == 'knife'
    (usd / 'deps/mesh.usd').write_bytes(b'referenced geometry')
    (usd / 'utensil.usd').write_bytes(b'changed wrapper layer')
    with pytest.raises(ValueError, match='asset SHA-256 mismatch'):
        load_h3_runtime_bundle(path, digest(payload))
    bundle['assets'][0]['usd_path'] = '../state.yaml'
    payload = json.dumps(bundle).encode()
    path.write_bytes(payload)
    with pytest.raises(ValueError, match='leaves assets/'):
        load_h3_runtime_bundle(path, digest(payload))
    payload = json.dumps(dict(bundle, format='dexbench_h3_runtime_v1')).encode()
    path.write_bytes(payload)
    with pytest.raises(ValueError, match='reinstall the H3 inputs'):
        load_h3_runtime_bundle(path, digest(payload))
