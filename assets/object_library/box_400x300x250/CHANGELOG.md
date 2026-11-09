# box_400x300x250

Newest first. A version with no entry here is not a version.

## 2026-11-05 — 3.1.2

- Record NdotLight as the creator and its confirmed CC BY 4.0 grant for NdotLight-owned contributions. No third-party elements were reported for this item in the confirmation. Bundled third-party material keeps its own terms.

## 2026-10-21 — 3.1.1

- Textures capped at 1024 px on the longest side for rendering cost: the 3 textures in `usd/textures/` that were larger (2048 px) are downscaled in place (Lanczos, aspect kept, same name and format, so no layer changes; its normal map resampled as float and renormalised), 9.9 MB -> 2.3 MB. Geometry, UVs, materials and colliders are unchanged, and the SimReady record carries over: the files keep their paths and formats, only the pixel count changes; the PNGs' pixel-density chunk (pHYs) is not carried; nothing reads it. PATCH: nothing a reference composes changes.

## 2026-10-04 — 3.1.0

- Physics runtime variants `PhysX`, `Newton` on the default prim (SimReady RV.001-RV.011), `Disabled` by default; `Enabled` composes `usd/runnables/physics/physx.usd` / `usd/runnables/physics/newton.usd`. The engine schemas and attributes the base used to author now live in those payloads (PhysX: 1 prims), so the base composes neutral OpenUSD physics until an engine is selected; DexBench spawners select the engine they run, scene layers select `PhysX`. The Newton payload annotates the colliders (`NewtonCollisionAPI`, `NewtonMeshCollisionAPI`) and the physics materials (`NewtonMaterialAPI`); contact gap and margin stay the engine's.
- The self-contained Newton package of 3.0.0 (`usd/variants/<name>_newton/`) is gone: it was this package with the same Cube collider, and the Newton payload now annotates that Cube.
- Re-validated with simready-validate 2026.7.1: Robotics-Prop 3.1.0 passes its scope, `FET_003_PHYSX` and `FET_003_NEWTON` pass.

## 2026-09-29 — 3.0.0

- Remove unused SDF/collision attributes from the non-colliding Newton render mesh. Use the shared `move_package_layer.py` packaging tool; geometry, mass and the active Cube collider are unchanged.
- Package the Newton variant under `usd/variants/box_400x300x250_newton/usd/box_400x300x250_newton.usd`, with local materials/textures and `./` dependency paths. The old variant path is removed; update consumers. Geometry, primitive Cube collider, body physics and dependency bytes are unchanged. Earlier grasp/drop certificates retain their original version scope.

## 2026-09-29 — 2.0.0

- Replace the base SDF mesh and Newton convex mesh colliders with one analytic USD Cube at the nominal box dimensions, centered above the bottom-face pivot. Visual meshes, textures, mass/inertia and grasp lines are unchanged. Collider prim paths change; recheck placements and contact behavior. Earlier grasp/drop certificates describe the previous collider versions.

## 2026-09-20 — 1.3.0

- Physics runtime variants `PhysX`, `Newton` on the default prim (SimReady RV.001-RV.011), `Disabled` by default; `Enabled` composes `usd/runnables/physics/physx.usd` / `usd/runnables/physics/newton.usd`. The engine schemas and attributes the base used to author now live in those payloads (PhysX: 2 prims), so the base composes neutral OpenUSD physics until an engine is selected; DexBench spawners select the engine they run, scene layers select `PhysX`. The Newton payload brings the 1 convex pieces of the former `usd/variants/` layer in under the original collider and switches that collider off (1); the flattened variant layer is gone, the same geometry now composes through the variant.

## 2026-09-16 — 1.2.1

- Newton 1.5.2 cert of the `newton` collision variant (standalone MuJoCo-Warp solver, not the Arena image): 1 colliders imported as authored (0 replaced, 0 thickened by the runner); ground_drop pass, slope_drop pass; grasp_and_lift PASS on `grasp_identifier_01` with the line riding on the body (stage frame: pass). Report: the maintainers' Newton 1.5 grasp benchmark.

## 2026-09-16 — 1.2.0

- Added the Newton collision variant `usd/variants/box_400x300x250_newton.usd`: a flattened copy of this package in which `box_400x300x250` (sdf, 1 piece, thickened to 1 mm) is replaced by pre-decomposed convex pieces (CoACD threshold 0.05, <= 64 vertices each), because Newton's solvers collide a mesh as one 64-vertex convex hull. Same render mesh, same physics material, same mass. Parsed back through Newton's importer: every collider convex, nothing replaced. The package USD is unchanged.

## 2026-09-16 — 1.1.4

- Newton 1.5.2 cert (standalone MuJoCo-Warp solver, not the Arena image): parse OK; 1 `sdf` collider(s) collided as 64-vertex convex hulls (MuJoCo-Warp keeps no concavity); ground_drop pass, slope_drop pass; grasp_and_lift PASS on `grasp_identifier_01` with the line riding on the body (line kept in the stage frame after the settle: pass; stock Newton contact defaults: fail). Certified package version 1.1.3. Settings and report: the maintainers' Newton 1.5 grasp benchmark.

## 2026-09-15 — 1.1.3

- Grasp line verified by NVIDIA's FET005 grasp-and-lift benchmark on Isaac Sim 6.0.1 (PhysX): a parallel-jaw gripper closed on `grasp_identifier_01`, lifted, held and shook the asset without losing it.

## 2026-09-15 — 1.1.2

- Grasp line `grasp_identifier_01` re-placed from [0.0, -0.17, 0.125] to [0.0, 0.17, 0.125] m in the default prim's frame (across-y): both endpoints now sit 2 cm clear of the surface, because NVIDIA's grasp-and-lift benchmark builds its jaw pads at the endpoints and a pad inside the collider launches the asset.

## 2026-09-15 — 1.1.1

- Grasp line re-shaped as an `Xform "grasp_identifier_01"` holding the `BasisCurves "grasp_identifier_01_line"` (same two points): NVIDIA's runtime grasp-and-lift benchmark looks for the Xform and reads its child, the static GSP.001 rule for a curve of that name; the plain curve satisfied only the latter.

## 2026-09-15 — 1.1.0

- Grasp line `grasp_identifier_01` authored from [0.0, -0.15, 0.125] to [0.0, 0.15, 0.125] m in the default prim's frame (across-y) as the closing line of a parallel-jaw grasp (GSP.001). Chosen from a projection preview of the mesh; not yet run through the FET005 grasp-and-lift benchmark.

## 2026-09-14 — 1.0.1

- SimReady Core metadata (`SimReady_Metadata`, `asset_name`, `asset_type`, `source_file`, `usd_date_generated`) stamped into the root layer's customLayerData from the sidecar (NP.006/SR.001).
- Kit's OmniPBR.mdl bundled under `materials/` and the 1 MDL shader source paths made relative (NP.008/VM.MDL.001), the layout NVIDIA's sample packages use.

## 2026-09-15 — 1.0.0

Packaged as `object_library/box_400x300x250/usd/box_400x300x250.usd` from `object_library/items/parcels/Box_400x300x250/Box_400x300x250.usd` (source: authored in this repo). No content change; the version is the sidecar's.
