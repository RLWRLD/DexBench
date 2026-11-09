# parts_bin_fc

Newest first. A version with no entry here is not a version.

## 2026-10-04 — 1.1.1

- Merged HF main's 1.0.2 (texture restored from its source delivery) into 1.1.0's runtime variants. The SimReady record is re-measured on the restored texture (Robotics-Prop 3.1.0).

## 2026-09-28 — 1.0.2

- Textures restored from the source delivery (deliveries/psl/e3_manufacturing/260827/fixtures/textures). The 1.0.0 packaging had taken the same-named `bin_*_2k.jpg` set from deliveries/psl/e4_hotel_fb/260831, so the mesh UVs sampled the wrong atlas. Geometry, UVs and materials unchanged.

## 2026-09-20 — 1.1.0

- Physics runtime variants `Newton` on the default prim (SimReady RV.001-RV.011), `Disabled` by default; `Enabled` composes `usd/runnables/physics/newton.usd`. The Newton payload annotates the colliders (`NewtonCollisionAPI`, `NewtonMeshCollisionAPI`) and the physics materials (`NewtonMaterialAPI`); contact gap and margin stay the engine's.

## 2026-09-14 — 1.0.1

- SimReady Core metadata (`SimReady_Metadata`, `asset_name`, `asset_type`, `source_file`, `usd_date_generated`) stamped into the root layer's customLayerData from the sidecar (NP.006/SR.001).

## 2026-09-15 — 1.0.0

Packaged as `object_library/parts_bin_fc/usd/parts_bin_fc.usd` from `object_library/fixtures/parts_bin_fc.usdc` (source: deliveries/psl/e3_manufacturing/260827). No content change; the version is the sidecar's.
