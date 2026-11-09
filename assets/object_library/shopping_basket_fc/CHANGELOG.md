# shopping_basket_fc

Newest first. A version with no entry here is not a version.

## 2026-09-20 — 1.1.0

- Physics runtime variants `Newton` on the default prim (SimReady RV.001-RV.011), `Disabled` by default; `Enabled` composes `usd/runnables/physics/newton.usd`. The Newton payload annotates the colliders (`NewtonCollisionAPI`, `NewtonMeshCollisionAPI`) and the physics materials (`NewtonMaterialAPI`); contact gap and margin stay the engine's.

## 2026-09-15 — 1.0.2

- 13 gprims with no material binding bound to a UsdPreviewSurface in their displayColor (VM.MAT.001); renders as before.

## 2026-09-14 — 1.0.1

- SimReady Core metadata (`SimReady_Metadata`, `asset_name`, `asset_type`, `source_file`, `usd_date_generated`) stamped into the root layer's customLayerData from the sidecar (NP.006/SR.001).

## 2026-09-15 — packaged

Moved to `object_library/shopping_basket_fc/usd/shopping_basket_fc.usd` from `object_library/fixtures/shopping_basket_fc.usdc`. No content change.

## 2026-09-14 — PSL E1 260914 texture upgrade

### shopping_basket_fc 1.0.0 · shopping_cart_fc 1.0.0 · tote_stack_fc 1.0.0
New in the pool from this delivery. Nothing loads them yet. `tote_stack_fc` is
a baked two-tier stack authoring ten collision proxies; the single `tote_fc`
(five proxies) stays the asset DB-R2 fills and DB-R3 stacks six high.
