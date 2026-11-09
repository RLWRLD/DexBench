# gondola_fc

Newest first. A version with no entry here is not a version.

## 2026-10-04 — 1.2.1

- Merged HF main's 1.1.2 (texture restored from its source delivery) into 1.2.0's runtime variants. The SimReady record is re-measured on the restored texture (Robotics-Prop 3.1.0).

## 2026-09-28 — 1.1.2

- `surf_paint_normal_1k.jpg` restored from the source delivery (deliveries/psl/e1_retail/260914/fixtures/textures); packaging had taken the same-named file from deliveries/psl/e4_hotel_fb/260831. Geometry, UVs and materials unchanged.

## 2026-09-20 — 1.2.0

- Physics runtime variants `Newton` on the default prim (SimReady RV.001-RV.011), `Disabled` by default; `Enabled` composes `usd/runnables/physics/newton.usd`. The Newton payload annotates the colliders (`NewtonCollisionAPI`, `NewtonMeshCollisionAPI`) and the physics materials (`NewtonMaterialAPI`); contact gap and margin stay the engine's.

## 2026-09-14 — 1.1.1

- SimReady Core metadata (`SimReady_Metadata`, `asset_name`, `asset_type`, `source_file`, `usd_date_generated`) stamped into the root layer's customLayerData from the sidecar (NP.006/SR.001).

## 2026-09-15 — packaged

Moved to `object_library/gondola_fc/usd/gondola_fc.usd` from `object_library/fixtures/gondola_fc.usdc`. No content change.

## 2026-09-14 — PSL E1 260914 texture upgrade

### gondola_fc 1.1.0 · produce_display_fc 1.1.0
Re-exported by the vendor to carry a full 2k PBR set (base colour, metallic,
normal, roughness) in place of single maps. Geometry verified unchanged by
`scripts/tools/compare_usd_geometry.py`: every mesh has the same point and
face counts and extent; only material bindings differ. Drop-in for the scenes
that reference them.
