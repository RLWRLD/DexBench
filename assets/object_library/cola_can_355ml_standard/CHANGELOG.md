# cola_can_355ml_standard

Newest first. A version with no entry here is not a version.

## 2026-11-08 — 2.0.2

- Record maintainer confirmation that NdotLight's written CC BY 4.0 approval covers the 2026-11-06 replacement labels. Credit NdotLight and RLWRLD modifications.
- Keep current license/source metadata with the package and preserve prior originality, trade-dress and appearance review records internally. The current license confirmation supersedes the historical pending-label status recorded in 2.0.1.
- PATCH for rights metadata only. USD layers, textures, physics and certificate results are unchanged.

## 2026-11-06 — 2.0.1

- Integrate the original NdotLight grant and the maintainer-confirmed replacement-label creator with the generic package name. Preserve the separate historical exclusions and unresolved replacement-artwork reviews. Runtime payloads match the reviewed label/name revision. The original CC BY 4.0 grant covers NdotLight-owned contributions; the 2026-11-06 author confirmation does not establish new-artwork permission, originality or trade-dress approval. Fold the provisional pre-rebase licensing notes into this actual 2.0.1 version; do not list them as separate released versions.

## 2026-11-06 — 2.0.0

- Rename the package and main USD/sidecar to generic product names. MAJOR because references to the previous package path must be updated. Keep recorded SKU keys, USD prim paths, geometry, physics, transforms and texture bytes unchanged.

## 2026-11-06 — 1.4.0

- Adopt the Cola_Can_355ml_Standard label appearance from `20261106_DexBench_Assets_R2_R3_Label.zip` (base color, normal and ORM textures). MINOR for the new material appearance; source mesh geometry, topology, UVs, normals, world transforms and shader settings match exactly.
- Keep all runtime USD layers byte-for-byte, including stable prim paths, bundled MDL, collision geometry, mass, grasp lines and engine variants. Source-to-package texture hashes are in the maintainers' delivery receipt.

## 2026-09-20 — 1.3.0

- Physics runtime variants `PhysX`, `Newton` on the default prim (SimReady RV.001-RV.011), `Disabled` by default; `Enabled` composes `usd/runnables/physics/physx.usd` / `usd/runnables/physics/newton.usd`. The engine schemas and attributes the base used to author now live in those payloads (PhysX: 2 prims), so the base composes neutral OpenUSD physics until an engine is selected; DexBench spawners select the engine they run, scene layers select `PhysX`. The Newton payload brings the 32 convex pieces of the former `usd/variants/` layer in under the original collider and switches that collider off (1); the flattened variant layer is gone, the same geometry now composes through the variant.

## 2026-09-16 — 1.2.1

- Newton 1.5.2 cert of the `newton` collision variant (standalone MuJoCo-Warp solver, not the Arena image): 32 colliders imported as authored (0 replaced, 0 thickened by the runner); ground_drop pass, slope_drop pass; grasp_and_lift PASS on `grasp_identifier_01` with the line riding on the body (stage frame: pass). Report: the maintainers' Newton 1.5 grasp benchmark.

## 2026-09-16 — 1.2.0

- Added the Newton collision variant `usd/variants/cocacola_355ml_standard_can_newton.usd`: a flattened copy of this package in which `cocacola_355ml_standard_can` (sdf, 32 pieces) is replaced by pre-decomposed convex pieces (CoACD threshold 0.05, <= 64 vertices each), because Newton's solvers collide a mesh as one 64-vertex convex hull. Same render mesh, same physics material, same mass. Parsed back through Newton's importer: every collider convex, nothing replaced. The package USD is unchanged.

## 2026-09-16 — 1.1.4

- Newton 1.5.2 cert (standalone MuJoCo-Warp solver, not the Arena image): parse OK; 1 `sdf` collider(s) collided as 64-vertex convex hulls (MuJoCo-Warp keeps no concavity); ground_drop pass, slope_drop pass; grasp_and_lift PASS on `grasp_identifier_01` with the line riding on the body (line kept in the stage frame after the settle: pass; stock Newton contact defaults: fail). Certified package version 1.1.3. Settings and report: the maintainers' Newton 1.5 grasp benchmark.

## 2026-09-15 — 1.1.3

- Grasp line verified by NVIDIA's FET005 grasp-and-lift benchmark on Isaac Sim 6.0.1 (PhysX): a parallel-jaw gripper closed on `grasp_identifier_01`, lifted, held and shook the asset without losing it.

## 2026-09-15 — 1.1.2

- Grasp line `grasp_identifier_01` re-placed from [-0.053, 0.0, 0.0621] to [0.053, 0.0, 0.0621] m in the default prim's frame (cylinder-diameter): both endpoints now sit 2 cm clear of the surface, because NVIDIA's grasp-and-lift benchmark builds its jaw pads at the endpoints and a pad inside the collider launches the asset.

## 2026-09-15 — 1.1.1

- Grasp line re-shaped as an `Xform "grasp_identifier_01"` holding the `BasisCurves "grasp_identifier_01_line"` (same two points): NVIDIA's runtime grasp-and-lift benchmark looks for the Xform and reads its child, the static GSP.001 rule for a curve of that name; the plain curve satisfied only the latter.

## 2026-09-15 — 1.1.0

- Grasp line `grasp_identifier_01` authored from [-0.033, 0.0, 0.0621] to [0.033, 0.0, 0.0621] m in the default prim's frame (cylinder-diameter) as the closing line of a parallel-jaw grasp (GSP.001). Chosen from a projection preview of the mesh; not yet run through the FET005 grasp-and-lift benchmark.

## 2026-09-14 — 1.0.1

- SimReady Core metadata (`SimReady_Metadata`, `asset_name`, `asset_type`, `source_file`, `usd_date_generated`) stamped into the root layer's customLayerData from the sidecar (NP.006/SR.001).
- Kit's OmniPBR.mdl bundled under `materials/` and the 1 MDL shader source paths made relative (NP.008/VM.MDL.001), the layout NVIDIA's sample packages use.
- Root prims `Render` beside the default prim removed (HI.001); a referencing scene never composed them.

## 2026-09-15 — 1.0.0

Packaged as `object_library/cocacola_355ml_standard_can/usd/cocacola_355ml_standard_can.usd` from `object_library/items/beverages/CocaCola_355ml_Standard_Can/CocaCola_355ml_Standard_Can.usd` (source: authored in this repo). No content change; the version is the sidecar's.
