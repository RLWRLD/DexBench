# staging_shelf_fc

Newest first. A version with no entry here is not a version.

## 2026-10-23 — 3.0.1

- The root layer's SimReady stamp (`customLayerData`) matches the sidecar again: `SimReady_Metadata.version` 3.0.1, `source_delivery` and `source_file` `deliveries/psl/e3_manufacturing/261008`, `source_generated` 261008, the validation record, and `usd_date_generated` 2026-10-23. It still carried what 1.0.1 wrote (version 1.0.0, `deliveries/psl/e3_manufacturing/260827`, 2026-09-14): `stamp_simready_metadata.py --write` only adds a stamp that is missing. Set from `stamp_simready_metadata.wanted_metadata`; the `Variants` block is kept. PATCH: layer metadata only; `compare_usd_geometry.py` against the delivery file: identical.
- Not 3.0.0 again: 3.0.0's bytes are already preserved on the HF mirror, so the new bytes take a new version.
- SimReady validated with simready-validate 2026.7.1 (SimReady Foundation 2026.07.1, tier-core d5c4c95) on 2026-10-23: Robotics-Prop 3.1.0 **pass** (scope static), with the same feature outcomes.

## 2026-10-23 — 3.0.0

- The shelf height from PSL's E3 r01 delivery 261008 (`deliveries/psl/e3_manufacturing/261008`, `usd/e3/fixtures/staging_shelf_fc.usdc`): the top board is 0.1 m higher, its top at 0.700 m, and the four legs reach it at 0.675 m. r01 authors this as xform ops on the part Xforms (`ShelfFC_Top` translated z +0.1, each `ShelfFC_Leg_*` scaled z 0.675/0.575); those specs are copied from the delivery layer onto 1.1.0, so the mesh points, the colliders, the materials and the Newton payload are as they were. MAJOR: the geometry moved. `compare_usd_geometry.py` against the delivery file: identical.
- No 2.0.0: internal PR 325 re-sources this package as 2.0.0 on its branch, which is open on origin with other bytes, so 2.0.0 already names that payload and reusing it would give two payloads one version.
- SimReady validated with simready-validate 2026.7.1 (SimReady Foundation 2026.07.1, tier-core d5c4c95) on 2026-10-23: Robotics-Prop 3.1.0 **pass** (scope static), with the same feature outcomes as 1.1.0.

## 2026-09-20 — 1.1.0

- Physics runtime variants `Newton` on the default prim (SimReady RV.001-RV.011), `Disabled` by default; `Enabled` composes `usd/runnables/physics/newton.usd`. The Newton payload annotates the colliders (`NewtonCollisionAPI`, `NewtonMeshCollisionAPI`) and the physics materials (`NewtonMaterialAPI`); contact gap and margin stay the engine's.

## 2026-09-14 — 1.0.1

- SimReady Core metadata (`SimReady_Metadata`, `asset_name`, `asset_type`, `source_file`, `usd_date_generated`) stamped into the root layer's customLayerData from the sidecar (NP.006/SR.001).

## 2026-09-15 — 1.0.0

Packaged as `object_library/staging_shelf_fc/usd/staging_shelf_fc.usd` from `object_library/fixtures/staging_shelf_fc.usdc` (source: deliveries/psl/e3_manufacturing/260827). No content change; the version is the sidecar's.
