# produce_display_fc

Newest first. A version with no entry here is not a version.

## 2026-10-30 — 2.0.0

- Tray 1 (the one with the two dividers) and tray 2 trade heights: `ProdFC_Tray1` now stands at 1.04 m (board
  0.974–1.089) and `ProdFC_Tray2`, with no dividers, at 0.68 m (0.614–0.729). Each tray moves as one Xform with
  its board, lip, walls, dividers and colliders; the posts, rails and tray 3 stay. DB-R1's five-variant piles sit
  in the divided tray's middle compartment at the second tier. Breaking for anything placed on a tray by
  height: DB-R1's fixed-SKU slot (tray 1's middle compartment at 0.64 m) now stands on the undivided tray.

## 2026-09-20 — 1.2.0

- Physics runtime variants `Newton` on the default prim (SimReady RV.001-RV.011), `Disabled` by default; `Enabled` composes `usd/runnables/physics/newton.usd`. The Newton payload annotates the colliders (`NewtonCollisionAPI`, `NewtonMeshCollisionAPI`) and the physics materials (`NewtonMaterialAPI`); contact gap and margin stay the engine's.

## 2026-09-14 — 1.1.1

- SimReady Core metadata (`SimReady_Metadata`, `asset_name`, `asset_type`, `source_file`, `usd_date_generated`) stamped into the root layer's customLayerData from the sidecar (NP.006/SR.001).

## 2026-09-15 — packaged

Moved to `object_library/produce_display_fc/usd/produce_display_fc.usd` from `object_library/fixtures/produce_display_fc.usdc`. No content change.

## 2026-09-14 — PSL E1 260914 texture upgrade

### gondola_fc 1.1.0 · produce_display_fc 1.1.0
Re-exported by the vendor to carry a full 2k PBR set (base colour, metallic,
normal, roughness) in place of single maps. Geometry verified unchanged by
`scripts/tools/compare_usd_geometry.py`: every mesh has the same point and
face counts and extent; only material bindings differ. Drop-in for the scenes
that reference them.
