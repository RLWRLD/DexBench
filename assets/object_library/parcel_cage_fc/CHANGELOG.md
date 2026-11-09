# parcel_cage_fc

Newest first. A version with no entry here is not a version.

## 2026-10-21 — 2.0.1

- Textures capped at 1024 px on the longest side for rendering cost: the 52 textures in `usd/textures/` that were larger (2052-4096 px) are downscaled in place (Lanczos, aspect kept, same name and format, so no layer changes; the 13 normal maps resampled as float and renormalised), 67.4 MB -> 9.3 MB. Geometry, UVs, materials and colliders are unchanged, and the SimReady record carries over: the files keep their paths and formats, only the pixel count changes. The PNGs' ancillary chunks (sRGB, gAMA, cHRM, eXIf, oFFs, pHYs) are not carried; every binding to these files declares its colour space (`sourceColorSpace` or `colorSpace` metadata, none `auto`), so the renderer reads them as before. PATCH: nothing a reference composes changes.

## 2026-10-14 — 2.0.0

- Re-sourced from PSL's E2 delivery 261014 (`deliveries/psl/e2_logistics/261014`, `assets/fixtures/e2_parcel_cage_r02.usdc`): r02's cage, the same frame colliders (rails and uprights, the inner clear space 1.132 x 0.966 x 1.42 m unchanged) under r02's finishes (`BK_MAT_E2_FixSteelBlue_*`, `BK_MAT_E2_FixSteelBlack`, `BK_MAT_E2_FixSteelBlueWire_*`) and r02's doors, 25 mm panels folded flat against the sides (x +-0.6225..0.6475, were 0.600..0.670). r02 drops the seven collision proxies; they are put back from 1.1.0 (`CageFC_Plane_{Floor,Back,Right,Left,Top}_colproxy`, `CageFC_DoorL/R_colproxy`, guide purpose, convex hull): the mesh panels are visual only, so without them a hand or a box passes through the cage's sides, back and roof. The two door proxies are refitted to r02's door panels. MAJOR: geometry changed (`compare_usd_geometry.py` against 1.1.0, 2026-10-14).
- Every collider (19) binds the package's `_physics/PhysicsMaterial` (static 0.6, dynamic 0.5, restitution 0): r02 binds visual materials that carry no physics material, which an engine reads as its default.
- Textures: the 52 the layer uses are in `usd/textures/`; 6 the previous version used are removed.
- Physics runtime variant `Newton` re-authored for these colliders (`author_physics_variants.py`'s own functions): 19 colliders and 1 physics materials annotated in `usd/runnables/physics/newton.usd`, `Disabled` by default; no PhysX data, so no PhysX payload.
- SimReady validated with simready-validate 2026.7.1 (SimReady Foundation 2026.07.1, tier-core d5c4c95) on 2026-10-14: Robotics-Prop 3.1.0 **pass** (scope static).

## 2026-09-20 — 1.1.0

- Physics runtime variants `Newton` on the default prim (SimReady RV.001-RV.011), `Disabled` by default; `Enabled` composes `usd/runnables/physics/newton.usd`. The Newton payload annotates the colliders (`NewtonCollisionAPI`, `NewtonMeshCollisionAPI`) and the physics materials (`NewtonMaterialAPI`); contact gap and margin stay the engine's.

## 2026-09-14 — 1.0.1

- SimReady Core metadata (`SimReady_Metadata`, `asset_name`, `asset_type`, `source_file`, `usd_date_generated`) stamped into the root layer's customLayerData from the sidecar (NP.006/SR.001).

## 2026-09-15 — 1.0.0

Packaged as `object_library/parcel_cage_fc/usd/parcel_cage_fc.usd` from `object_library/fixtures/parcel_cage_fc.usdc` (source: deliveries/psl/e2_logistics/260827). No content change; the version is the sidecar's.
