# fridge_minifit_fc

Newest first. A version with no entry here is not a version.

## 2026-09-20 — 1.2.0

- Physics runtime variants `Newton` on the default prim (SimReady RV.001-RV.011), `Disabled` by default; `Enabled` composes `usd/runnables/physics/newton.usd`. The Newton payload annotates the colliders (`NewtonCollisionAPI`, `NewtonMeshCollisionAPI`) and the physics materials (`NewtonMaterialAPI`); contact gap and margin stay the engine's.

## 2026-09-15 — 1.1.2

- Extents authored on 1 boundable non-gprims (lights) (VG.002).
- 48 gprims with no material binding bound to a UsdPreviewSurface in their displayColor (VM.MAT.001); renders as before.

## 2026-09-14 — 1.1.1

- SimReady Core metadata (`SimReady_Metadata`, `asset_name`, `asset_type`, `source_file`, `usd_date_generated`) stamped into the root layer's customLayerData from the sidecar (NP.006/SR.001).

## 2026-09-15 — packaged

Moved to `object_library/fridge_minifit_fc/usd/fridge_minifit_fc.usd` from `object_library/fixtures/fridge_minifit_fc.usdc`. No content change.

## 2026-09-14 — PSL E1 260914 texture upgrade

### fridge_minifit_fc 1.1.0
Same texture treatment, and 42 meshes added: the vendor now ships the
Shelf1/3/4 MiniFit tiers (`FrgFC_MiniFitS1A/S1B/S3A/S3B/S4A/S4B`) with their
`_colproxy` and `_visonly` children, which our spawner used to compose at
runtime from the Shelf2 prototypes. The original 31 meshes are unchanged.
`_spawn_fc_fridge_with_review_state` now skips its own composition when the
vendor tiers are present, so either generation composes once.
