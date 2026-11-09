# bowl_rounded_19cm

Newest first. A version with no entry here is not a version.

## 2026-11-05 — 3.0.0

- Rename the package and main USD from `ikea_365_bowl_rounded_19` to `bowl_rounded_19cm` to replace the product name with a generic name, as requested by NdotLight. USD, texture, material and runtime payload bytes, prim paths and historical certificates are preserved. The package address changes (MAJOR); existing DB-H1 recording/configuration keys remain compatible.
- Record NdotLight as the creator and its confirmed CC BY 4.0 grant for NdotLight-owned contributions. No third-party elements were reported for this item in the confirmation. Bundled third-party material keeps its own terms.

## 2026-10-12 — 2.0.0

- Replace the Newton payload's 32 CoACD pieces with 128 PhysX-cooked convex pieces, the recipe the other five DB-H1 dishes carry (the training dishes since 2.0.0, both godmiddag dishes since 2026-10-06): V-HACD at most 128 hulls of at most 64 vertices, error 0.5 %, voxel resolution 1e6, shrink-wrap, cooked from this package's render mesh by `scripts/tools/prepare_h1_newton_colliders.py`. Each piece is reframed so its AABB-centre collision seed sits inside the hull (body-space geometry within 0.1 micrometre of the cooked hull, minimum seed clearance 1.15 mm) and carries `newton:contactGap` 2 mm and `newton:contactMargin` 0; no hull needed the 1 mm floor. The pieces replace the old ones under `ikea_365_bowl_rounded_19_newton_collision` in `usd/runnables/physics/newton.usd`, by `scripts/tools/author_physics_variants.py --refold`, with the schemas it gives pieces; the original collider stays switched off there.
- Measured for DB-H1 on Newton (`smoke_hotel_fb_e4_db_h1.py`, 600 steps, Isaac Sim 6.1.0 / Newton 1.6.1, 2026-10-12): with the CoACD pieces the dishes did not nest, the table stack stood 12.8 mm higher than its spawn (over the 10 mm budget) and leaned up to 3.2 deg; with these the stack drift is 3.6 mm, the lean at most 0.5 deg and the rim offsets 1.1 / 0.6 mm, in line with `ikea_365_bowl_rounded_22` in the same run (3.7 mm, 0.4 deg, 0.4 / 0.2 mm). Render geometry, pivot, mass, centre of mass, inertia and the PhysX payload are unchanged.
- The recipe record (source and output SHA-256, cook parameters, geometry checks) is `usd/runnables/physics/newton.preparation.json`, as for the other five, and its `fold` block names the payload before and after the fold (SHA-256 `1f91af94…` with 32 pieces, `54660d6e…` with 128). The same run re-cooked `godmiddag_bowl_16` to identical hull statistics, so the recipe reproduces; the fold reproduces the payload byte for byte from the 1.4.0 one.
- Re-validated with simready-validate 2026.7.1 on this payload (2026-10-12): Robotics-Prop 3.1.0 passes its scope, `FET_003_PHYSX` and `FET_003_NEWTON` pass, every feature outcome as on 1.4.0.

## 2026-10-04 — 1.4.0

- The PhysX payload carries the DB-H1 collider tuning that DexBench used to apply at every spawn: `convexDecomposition` (V-HACD, at most 128 hulls of at most 64 vertices, error 0.5 %, voxel resolution 1e6, shrink-wrap) with `physxCollision:contactOffset` 2 mm and `restOffset` 0, in place of the delivered `sdf` collider at 0.2 mm. The SDF cooking attributes go with it. Measured for DB-H1: `sdf` lets these thin walls fall through each other, and at 0.2 mm a dish set down from 20 mm is inside the one below before PhysX sees the contact; under this recipe every one of the six nests. The same recipe cooked the Newton pieces of the three training dishes. Geometry, pivot, mass and the Newton payload are unchanged; a scene that selects `PhysX` now gets the nesting collider without re-cooking it.
- Re-validated with simready-validate 2026.7.1: Robotics-Prop 3.1.0 passes its scope, `FET_003_PHYSX` and `FET_003_NEWTON` pass.

## 2026-09-20 — 1.3.0

- Physics runtime variants `PhysX`, `Newton` on the default prim (SimReady RV.001-RV.011), `Disabled` by default; `Enabled` composes `usd/runnables/physics/physx.usd` / `usd/runnables/physics/newton.usd`. The engine schemas and attributes the base used to author now live in those payloads (PhysX: 2 prims), so the base composes neutral OpenUSD physics until an engine is selected; DexBench spawners select the engine they run, scene layers select `PhysX`. The Newton payload brings the 32 convex pieces of the former `usd/variants/` layer in under the original collider and switches that collider off (1); the flattened variant layer is gone, the same geometry now composes through the variant.

## 2026-09-16 — 1.2.1

- Newton 1.5.2 cert of the `newton` collision variant (standalone MuJoCo-Warp solver, not the Arena image): 32 colliders imported as authored (0 replaced, 0 thickened by the runner); ground_drop pass, slope_drop pass; grasp_and_lift PASS on `grasp_identifier_01` with the line riding on the body (stage frame: pass). Report: the maintainers' Newton 1.5 grasp benchmark.

## 2026-09-16 — 1.2.0

- Added the Newton collision variant `usd/variants/ikea_365_bowl_rounded_19_newton.usd`: a flattened copy of this package in which `ikea_365_bowl_rounded_19` (sdf, 32 pieces) is replaced by pre-decomposed convex pieces (CoACD threshold 0.05, <= 64 vertices each), because Newton's solvers collide a mesh as one 64-vertex convex hull. Same render mesh, same physics material, same mass. Parsed back through Newton's importer: every collider convex, nothing replaced. The package USD is unchanged.

## 2026-09-16 — 1.1.4

- Newton 1.5.2 cert (standalone MuJoCo-Warp solver, not the Arena image): parse OK; 1 `sdf` collider(s) collided as 64-vertex convex hulls (MuJoCo-Warp keeps no concavity); ground_drop pass, slope_drop pass; grasp_and_lift FAIL on `grasp_identifier_01` with the line riding on the body: Grasping failed: pads touched (no object) (line kept in the stage frame after the settle: fail; stock Newton contact defaults: fail). Certified package version 1.1.3. Settings and report: the maintainers' Newton 1.5 grasp benchmark.

## 2026-09-15 — 1.1.3

- Grasp line verified by NVIDIA's FET005 grasp-and-lift benchmark on Isaac Sim 6.0.1 (PhysX): a parallel-jaw gripper closed on `grasp_identifier_01`, lifted, held and shook the asset without losing it.

## 2026-09-15 — 1.1.2

- Grasp line `grasp_identifier_01` re-placed from [0.0449, 0.0, 0.0715] to [0.1199, 0.0, 0.0715] m in the default prim's frame (rim-pinch): both endpoints now sit 2 cm clear of the surface, because NVIDIA's grasp-and-lift benchmark builds its jaw pads at the endpoints and a pad inside the collider launches the asset.

## 2026-09-15 — 1.1.1

- Grasp line re-shaped as an `Xform "grasp_identifier_01"` holding the `BasisCurves "grasp_identifier_01_line"` (same two points): NVIDIA's runtime grasp-and-lift benchmark looks for the Xform and reads its child, the static GSP.001 rule for a curve of that name; the plain curve satisfied only the latter.

## 2026-09-15 — 1.1.0

- Grasp line `grasp_identifier_01` authored from [0.0649, 0.0, 0.0715] to [0.0999, 0.0, 0.0715] m in the default prim's frame (rim-pinch) as the closing line of a parallel-jaw grasp (GSP.001). Chosen from a projection preview of the mesh; not yet run through the FET005 grasp-and-lift benchmark.

## 2026-09-14 — 1.0.1

- SimReady Core metadata (`SimReady_Metadata`, `asset_name`, `asset_type`, `source_file`, `usd_date_generated`) stamped into the root layer's customLayerData from the sidecar (NP.006/SR.001).
- Kit's OmniPBR.mdl bundled under `materials/` and the 1 MDL shader source paths made relative (NP.008/VM.MDL.001), the layout NVIDIA's sample packages use.

## 2026-09-15 — 1.0.0

Packaged as `object_library/ikea_365_bowl_rounded_19/usd/ikea_365_bowl_rounded_19.usd` from `object_library/items/tableware/IKEA_365_Bowl_Rounded_19/IKEA_365_Bowl_Rounded_19.usd` (source: authored in this repo). No content change; the version is the sidecar's.
