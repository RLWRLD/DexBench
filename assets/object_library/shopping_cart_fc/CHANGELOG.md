# shopping_cart_fc

Newest first. A version with no entry here is not a version.

## 2026-10-04 — 1.1.1

- Merged HF main's 1.0.3 (texture restored from its source delivery) into 1.1.0's runtime variants. The SimReady record is re-measured on the restored texture (Robotics-Prop 3.1.0).

## 2026-09-28 — 1.0.3

- `surf_paint_normal_1k.jpg` restored from the source delivery (deliveries/psl/e1_retail/260914/fixtures/textures); packaging had taken the same-named file from deliveries/psl/e4_hotel_fb/260831. Geometry, UVs and materials unchanged.

## 2026-09-20 — 1.1.0

- Physics runtime variants `Newton` on the default prim (SimReady RV.001-RV.011), `Disabled` by default; `Enabled` composes `usd/runnables/physics/newton.usd`. The Newton payload annotates the colliders (`NewtonCollisionAPI`, `NewtonMeshCollisionAPI`) and the physics materials (`NewtonMaterialAPI`); contact gap and margin stay the engine's.

## 2026-09-15 — 1.0.2

- 2 gprims with no material binding bound to a UsdPreviewSurface in their displayColor (VM.MAT.001); renders as before.

## 2026-09-14 — 1.0.1

- SimReady Core metadata (`SimReady_Metadata`, `asset_name`, `asset_type`, `source_file`, `usd_date_generated`) stamped into the root layer's customLayerData from the sidecar (NP.006/SR.001).

## 2026-09-15 — packaged

Moved to `object_library/shopping_cart_fc/usd/shopping_cart_fc.usd` from `object_library/fixtures/shopping_cart_fc.usdc`. No content change.

## 2026-09-14 — PSL E1 260914 texture upgrade

### shopping_basket_fc 1.0.0 · shopping_cart_fc 1.0.0 · tote_stack_fc 1.0.0
New in the pool from this delivery. Nothing loads them yet. `tote_stack_fc` is
a baked two-tier stack authoring ten collision proxies; the single `tote_fc`
(five proxies) stays the asset DB-R2 fills and DB-R3 stacks six high.
