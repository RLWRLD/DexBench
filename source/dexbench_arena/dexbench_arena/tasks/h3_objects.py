# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""H3's utensil trios and requested targets, readable without the simulator."""

import json
from pathlib import Path
import subprocess

from dexbench_arena import ASSETS_DIR, REPO_ROOT
from dexbench_arena.assets.snapshot import restore_command
from dexbench_arena.tasks.object_axis import ObjectAxis, TaskObject
from dexbench_arena.evaluation.artifacts import file_sha256, sha256_bytes


_DECLARATION = json.loads(Path(__file__).with_name('h3_objects.json').read_text())
H3_OBJECTS = _DECLARATION['objects']
H3_PACKAGES = _DECLARATION['packages']
"""Asset id -> its package's main USD (relative to assets/) and the digest of the wrapper layer
DB-H3 loaded before the package moved into the pool (``source_sha256``)."""
H3_INPUTS_DIR = 'outputs/db_h3_inputs'
H3_INPUTS_FORMAT = 'dexbench_h3_inputs_v2'
H3_RUNTIME_FORMAT = 'dexbench_h3_runtime_v2'
H3_GA_PROTOCOL = 'held_out_objects_unseen_geometry_dependent_grasps_v1'

H3_TARGETS = ('spoon', 'fork', 'knife')
H3_TRIOS = {cell: tuple(sorted({row['trio'] for row in H3_OBJECTS.values() if row['split'] == split}))
            for cell, split in (('train', 'train'), ('ga_os', 'eval'))}


def h3_package_record(asset_id, assets_root=None):
    """What a runtime bundle binds for one utensil: its main USD (relative to assets/), that file's
    SHA-256 and the SHA-256 of every other Git-tracked file in the package's usd/ directory but the sidecar."""
    root = Path(assets_root or ASSETS_DIR)
    usd = root / H3_PACKAGES[asset_id]['usd_path']
    listed = subprocess.run(['git', '-C', str(usd.parent), 'ls-files', '-z', '--full-name', '.'],
                            check=True, capture_output=True).stdout.decode().split('\0')
    top = Path(subprocess.run(['git', '-C', str(usd.parent), 'rev-parse', '--show-toplevel'],
                              check=True, capture_output=True, text=True).stdout.strip())
    tracked = {(top / name).resolve() for name in listed if name}
    on_disk = {p.resolve() for p in usd.parent.rglob('*') if p.is_file()}
    if on_disk - tracked:
        untracked = sorted(p.relative_to(usd.parent.resolve()).as_posix() for p in on_disk - tracked)
        raise ValueError(f'{asset_id}: untracked files in its package: {untracked}')
    files = {p.relative_to(usd.parent.resolve()).as_posix(): file_sha256(p) for p in sorted(tracked)
             if p != usd.resolve() and not p.name.endswith('.meta.json')}
    return dict(usd_path=H3_PACKAGES[asset_id]['usd_path'], sha256=file_sha256(usd), files=files)


def relocate_h3_assets(assets, assets_root=None):
    """Bundle asset entries that name pool packages, from entries that named the wrapper layer the
    collection loaded. An entry is relocated only when its digest is the source_sha256 its pool
    package was made from; every other field is kept."""
    relocated = []
    for asset in assets:
        package = H3_PACKAGES.get(asset['id'])
        if package is None or asset['sha256'] != package['source_sha256']:
            raise ValueError(f"{asset['id']}: not the wrapper its pool package was made from")
        relocated.append(dict(asset, **h3_package_record(asset['id'], assets_root)))
    return relocated


def read_h3_runtime_bundle(path, digest, *, verify_assets=True):
    """Check a runtime bundle, its saved-state bytes and (``verify_assets``) every utensil file it binds."""
    path = Path(path).resolve()
    payload = path.read_bytes()
    if sha256_bytes(payload) != digest:
        raise ValueError('H3 runtime bundle SHA-256 mismatch')
    bundle = json.loads(payload)
    if bundle.get('format') != H3_RUNTIME_FORMAT:
        raise ValueError(f'Expected {H3_RUNTIME_FORMAT}; reinstall the H3 inputs with '
                         'scripts/download_datasets.sh --eval-conditions')
    assets = bundle['assets']
    if len(assets) != 3 or {a['type'] for a in assets} != {'fork', 'knife', 'spoon'}:
        raise ValueError('H3 needs exactly one fork, knife and spoon')
    names = [a['object_name'] for a in assets]
    if len(set(names)) != 3 or bundle['target'] not in {'fork', 'knife', 'spoon'}:
        raise ValueError('H3 needs unique object names and an explicit target')
    root = Path(ASSETS_DIR).resolve()
    for asset in assets:
        usd = (root / asset['usd_path']).resolve()
        if not usd.is_relative_to(root):
            raise ValueError(f"H3 asset path leaves assets/: {asset['id']}")
        if verify_assets:
            package = Path(asset['usd_path']).parent.parent
            hint = f" (restore its payload: {restore_command(REPO_ROOT, f'{package}/**')})"
            if not usd.is_file() or file_sha256(usd) != asset['sha256']:
                raise ValueError(f"H3 asset SHA-256 mismatch: {asset['id']}{hint}")
            for relative, expected in asset['files'].items():
                dependency = (usd.parent / relative).resolve()
                if (not dependency.is_relative_to(usd.parent) or not dependency.is_file()
                        or file_sha256(dependency) != expected):
                    raise ValueError(f"H3 asset dependency SHA-256 mismatch: {asset['id']}: {relative}{hint}")
        asset['usd_path'] = str(usd)
    labels = set()
    for row in bundle['starts']:
        if sha256_bytes((path.parent / row['state']).resolve().read_bytes()) != row['state_sha256']:
            raise ValueError(f"H3 state SHA-256 mismatch: {row['id']}")
        if row['id'] in labels:
            raise ValueError('Duplicate H3 start label')
        labels.add(row['id'])
    if not labels:
        raise ValueError('H3 runtime requires saved starts')
    return bundle


def h3_object_fit(task_object, cfg):
    """The same input identity for replay admission and episode recording."""
    path, digest = cfg.runtime_bundle, cfg.runtime_bundle_sha256
    if not cfg.runtime_bundle:
        path, digest = resolve_h3_bundle(task_object.key, cfg.h3_inputs_dir)
    data = Path(path).read_bytes()
    if sha256_bytes(data) != digest:
        raise ValueError('H3 runtime bundle SHA-256 mismatch')
    return dict(runtime_bundle_sha256=digest,
                utensil_assets=H3_OBJECTS[task_object.key]['assets'],
                target=H3_OBJECTS[task_object.key]['target'])


H3_OBJECT_AXIS = ObjectAxis(
    field='h3_object', default='train_01_spoon',
    objects=tuple(TaskObject(
        key=key, asset=row['asset'], split=row['split'],
        label='table knife' if row['target'] == 'knife' else row['target'],
        size_m=tuple(row['size_m']),
    ) for key, row in H3_OBJECTS.items()),
    size_axes='target utensil body-frame x, y, z extent',
    source_fit=('runtime_bundle_sha256', 'utensil_assets', 'target'),
    fit=h3_object_fit,
)


def resolve_h3_bundle(object_key, root):
    """Resolve a declared bundle from its package manifest, refusing unknown objects."""
    H3_OBJECT_AXIS.get(object_key)
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('format') != H3_INPUTS_FORMAT:
        raise ValueError(f'Expected {H3_INPUTS_FORMAT} input manifest in {root}; reinstall the H3 inputs '
                         'with scripts/download_datasets.sh --eval-conditions')
    entry = manifest['bundles'][object_key]
    return str(root / entry['path']), entry['sha256']
