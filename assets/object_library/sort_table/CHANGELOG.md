# sort_table

Newest first. A version with no entry here is not a version.

## 2026-10-21 — 2.0.1

- Textures capped at 1024 px on the longest side for rendering cost: the 4 textures in `usd/textures/` that were larger (2048 px) are downscaled in place (Lanczos, aspect kept, same name and format, so no layer changes; its normal map resampled as float and renormalised), 1.5 MB -> 0.6 MB. Geometry, UVs, materials and colliders are unchanged, and the SimReady record carries over: the files keep their paths, formats and colour spaces, only the pixel count changes. PATCH: nothing a reference composes changes.

## 2026-10-14 — 2.0.0

- Re-sourced from PSL's E2 delivery 261014 (`deliveries/psl/e2_logistics/261014`, `internal/world.usdc` `Slot_SortTable/Asset`): r02's table top and four legs under `MAT_FC_Meshy_sorttable`. MAJOR: geometry changed (`compare_usd_geometry.py` against 1.1.0, 2026-10-14).
- Every collider (5) binds the package's `_physics/PhysicsMaterial` (static 0.6, dynamic 0.5, restitution 0): r02 binds visual materials that carry no physics material, which an engine reads as its default.
- Textures: the 4 the layer uses are in `usd/textures/`.
- Physics runtime variant `Newton` re-authored for these colliders (`author_physics_variants.py`'s own functions): 5 colliders and 1 physics materials annotated in `usd/runnables/physics/newton.usd`, `Disabled` by default; no PhysX data, so no PhysX payload.
- SimReady validated with simready-validate 2026.7.1 (SimReady Foundation 2026.07.1, tier-core d5c4c95) on 2026-10-14: Robotics-Prop 3.1.0 **pass** (scope static).

## 2026-09-20 — 1.1.0

- Physics runtime variants `Newton` on the default prim (SimReady RV.001-RV.011), `Disabled` by default; `Enabled` composes `usd/runnables/physics/newton.usd`. The Newton payload annotates the colliders (`NewtonCollisionAPI`, `NewtonMeshCollisionAPI`) and the physics materials (`NewtonMaterialAPI`); contact gap and margin stay the engine's.

## 2026-09-17 — 1.0.1

- SimReady Core metadata (`SimReady_Metadata`, `asset_name`, `asset_type`, `source_file`, `usd_date_generated`) stamped into the root layer's customLayerData from the sidecar (NP.006/SR.001).

## 2026-09-17 — 1.0.0

Packaged as `object_library/sort_table/usd/sort_table.usd` from `env/e2_logistics/cell/fixtures/sort_table.usdc` (source: deliveries/psl/e2_logistics/260827). No content change; the version is the sidecar's.
