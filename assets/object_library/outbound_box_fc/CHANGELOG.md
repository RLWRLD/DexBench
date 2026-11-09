# outbound_box_fc

Newest first. A version with no entry here is not a version.

## 2026-10-21 — 3.0.1

- Textures capped at 1024 px on the longest side for rendering cost: the 6 textures in `usd/textures/` that were larger (1956-3776 px) are downscaled in place (Lanczos, aspect kept, same name and format, so no layer changes; the 3 normal maps resampled as float and renormalised), 16.1 MB -> 2.2 MB. The other one, already at or below 1024 px, is unchanged. Geometry, UVs, materials and colliders are unchanged, and the SimReady record carries over: the files keep their paths and formats, only the pixel count changes. The PNGs' ancillary chunks (sRGB, gAMA, cHRM, eXIf, oFFs, pHYs) are not carried; every binding to these files declares its colour space (`sourceColorSpace` or `colorSpace` metadata, none `auto`), so the renderer reads them as before. PATCH: nothing a reference composes changes.

## 2026-10-14 — 3.0.0

- Re-sourced from PSL's E2 delivery 261014 (`deliveries/psl/e2_logistics/261014`, `internal/world.usdc` `Slot_PackingBench/OwnedDeco/outbound_box_fc`): r02's outbound box, walls re-skinned (`BK_MAT_E2_OutboxCarton`), four open flaps as their own visual-only meshes, the gift box, bubble fill and a FRAGILE sticker inside. r02 composes it as static bench decoration with one collision box; here it stays the handled prop it was: the walls, floor and fill are its colliders again (convex hull, as in 1.2.0), r02's `ColBox` is not taken, and the rigid body, the 3.118 kg mass and the grasp line `grasp_identifier_01` are carried over from 1.2.0 unchanged (its benchmark records are for 1.1.3 and were not re-run). MAJOR: prim paths and geometry changed (`compare_usd_geometry.py` against 1.2.0, 2026-10-14).
- No 2.0.0: internal PR 257 (PSL's r01 delivery, superseded by this version, not merged) re-sourced this package as 2.0.0 on its branch, which is still on origin, so 2.0.0 already names r01's package and reusing it would give two payloads one version.
- Every collider (11) binds the package's `_physics/PhysicsMaterial` (static 0.6, dynamic 0.5, restitution 0): r02 binds visual materials that carry no physics material, which an engine reads as its default.
- Textures: the 7 the layer uses are in `usd/textures/`; 3 the previous version used are removed.
- Physics runtime variant `Newton` re-authored for these colliders (`author_physics_variants.py`'s own functions): 11 colliders and 1 physics materials annotated in `usd/runnables/physics/newton.usd`, `Disabled` by default; no PhysX data, so no PhysX payload.
- Kit's OmniGlass.mdl bundled under `materials/` and 1 MDL shader source paths made relative in outbound_box_fc.usd (NP.008/VM.MDL.001/AA.001), the layout NVIDIA's sample packages use.
- SimReady validated with simready-validate 2026.7.1 (SimReady Foundation 2026.07.1, tier-core d5c4c95) on 2026-10-14: Robotics-Prop 3.1.0 **pass** (scope full).

## 2026-09-20 — 1.2.0

- Physics runtime variants `Newton` on the default prim (SimReady RV.001-RV.011), `Disabled` by default; `Enabled` composes `usd/runnables/physics/newton.usd`. The Newton payload annotates the colliders (`NewtonCollisionAPI`, `NewtonMeshCollisionAPI`) and the physics materials (`NewtonMaterialAPI`); contact gap and margin stay the engine's.

## 2026-09-16 — 1.1.4

- Newton 1.5.2 cert (standalone MuJoCo-Warp solver, not the Arena image): parse OK; ground_drop pass, slope_drop pass; grasp_and_lift PASS on `grasp_identifier_01` with the line riding on the body (line kept in the stage frame after the settle: pass; stock Newton contact defaults: fail). Certified package version 1.1.3. Settings and report: the maintainers' Newton 1.5 grasp benchmark.

## 2026-09-15 — 1.1.3

- Grasp line verified by NVIDIA's FET005 grasp-and-lift benchmark on Isaac Sim 6.0.1 (PhysX): a parallel-jaw gripper closed on `grasp_identifier_01`, lifted, held and shook the asset without losing it.

## 2026-09-15 — 1.1.2

- Grasp line `grasp_identifier_01` re-placed from [0.0, -0.236, 0.125] to [0.0, 0.236, 0.125] m in the default prim's frame (across-y): both endpoints now sit 2 cm clear of the surface, because NVIDIA's grasp-and-lift benchmark builds its jaw pads at the endpoints and a pad inside the collider launches the asset.

## 2026-09-15 — 1.1.1

- Grasp line re-shaped as an `Xform "grasp_identifier_01"` holding the `BasisCurves "grasp_identifier_01_line"` (same two points): NVIDIA's runtime grasp-and-lift benchmark looks for the Xform and reads its child, the static GSP.001 rule for a curve of that name; the plain curve satisfied only the latter.

## 2026-09-15 — 1.1.0

- Grasp line `grasp_identifier_01` authored from [0.0, -0.216, 0.125] to [0.0, 0.216, 0.125] m in the default prim's frame (across-y) as the closing line of a parallel-jaw grasp (GSP.001). Chosen from a projection preview of the mesh; not yet run through the FET005 grasp-and-lift benchmark.

## 2026-09-14 — 1.0.1

- SimReady Core metadata (`SimReady_Metadata`, `asset_name`, `asset_type`, `source_file`, `usd_date_generated`) stamped into the root layer's customLayerData from the sidecar (NP.006/SR.001).

## 2026-09-15 — 1.0.0

Packaged as `object_library/outbound_box_fc/usd/outbound_box_fc.usd` from `object_library/items/parcels/outbound_box_fc.usdc` (source: deliveries/psl/e2_logistics/260827). No content change; the version is the sidecar's.
