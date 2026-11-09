# rs_pro_stepper_motor_535_0502_nema23_class

Newest first. A version with no entry here is not a version.

## 2026-11-05 — 1.4.1

- Record NdotLight as the creator and its confirmed CC BY 4.0 grant for NdotLight-owned contributions. No third-party elements were reported for this item in the confirmation. Bundled third-party material keeps its own terms.

## 2026-10-27 — 1.4.0

- Newton collider choice `NewtonColliders`, nested under the `Newton` variant's `Enabled`: `pieces` (the default) composes the Newton payload alone, as before; `primitives` adds `usd/runnables/physics/newton_primitives.usd`, which brings in 3 collision pieces (cube, cylinder, mesh), deactivates the payload's 64 pieces and switches off 6 of the base's 6 colliders (the ones the new layer does not keep). The pieces are dbclutter's `_prim5` collision primitives (at most 5 boxes, cylinders, capsules or spheres fitted to the visual mesh by db-m5-create-stable-clutter `scripts/fit_primitive_colliders.py`, a script not yet committed there, run on its 5f08c9a working tree). Between the choices the render mesh, mass properties and physics materials are the same; the recipe is in `runnables/physics/newton_primitives.preparation.json`. A consumer that does not select `primitives` composes what it did, and a Newton cert recorded for an earlier version still describes `pieces`.

## 2026-09-20 — 1.3.0

- Physics runtime variants `PhysX`, `Newton` on the default prim (SimReady RV.001-RV.011), `Disabled` by default; `Enabled` composes `usd/runnables/physics/physx.usd` / `usd/runnables/physics/newton.usd`. The engine schemas and attributes the base used to author now live in those payloads (PhysX: 7 prims), so the base composes neutral OpenUSD physics until an engine is selected; DexBench spawners select the engine they run, scene layers select `PhysX`. The Newton payload brings the 64 convex pieces of the former `usd/variants/` layer in under the original collider and switches that collider off (2); the flattened variant layer is gone, the same geometry now composes through the variant.

## 2026-09-16 — 1.2.1

- Newton 1.5.2 cert of the `newton` collision variant (standalone MuJoCo-Warp solver, not the Arena image): 68 colliders imported as authored (0 replaced, 0 thickened by the runner); ground_drop pass, slope_drop pass; grasp_and_lift PASS on `grasp_identifier_01` with the line riding on the body (stage frame: pass). Report: the maintainers' Newton 1.5 grasp benchmark.

## 2026-09-16 — 1.2.0

- Added the Newton collision variant `usd/variants/rs_pro_stepper_motor_535_0502_nema23_class_newton.usd`: a flattened copy of this package in which `motor_housing` (sdf, 32 pieces), `stator_stack` (sdf, 32 pieces) are replaced by pre-decomposed convex pieces (CoACD threshold 0.05, <= 64 vertices each), because Newton's solvers collide a mesh as one 64-vertex convex hull. Same render mesh, same physics material, same mass. Parsed back through Newton's importer: every collider convex, nothing replaced. The package USD is unchanged.

## 2026-09-16 — 1.1.4

- Newton 1.5.2 cert (standalone MuJoCo-Warp solver, not the Arena image): parse OK; 2 `sdf` collider(s) collided as 64-vertex convex hulls (MuJoCo-Warp keeps no concavity); ground_drop pass, slope_drop pass; grasp_and_lift PASS on `grasp_identifier_01` with the line riding on the body (line kept in the stage frame after the settle: pass; stock Newton contact defaults: fail). Certified package version 1.1.3. Settings and report: the maintainers' Newton 1.5 grasp benchmark.

## 2026-09-15 — 1.1.3

- Grasp line verified by NVIDIA's FET005 grasp-and-lift benchmark on Isaac Sim 6.0.1 (PhysX): a parallel-jaw gripper closed on `grasp_identifier_01`, lifted, held and shook the asset without losing it.

## 2026-09-15 — 1.1.2

- Grasp line `grasp_identifier_01` re-placed from [-0.0482, 0.0, 0.0378] to [0.0482, 0.0, 0.0378] m in the default prim's frame (cylinder-diameter): both endpoints now sit 2 cm clear of the surface, because NVIDIA's grasp-and-lift benchmark builds its jaw pads at the endpoints and a pad inside the collider launches the asset.

## 2026-09-15 — 1.1.1

- Grasp line re-shaped as an `Xform "grasp_identifier_01"` holding the `BasisCurves "grasp_identifier_01_line"` (same two points): NVIDIA's runtime grasp-and-lift benchmark looks for the Xform and reads its child, the static GSP.001 rule for a curve of that name; the plain curve satisfied only the latter.

## 2026-09-15 — 1.1.0

- Grasp line `grasp_identifier_01` authored from [-0.0282, 0.0, 0.0378] to [0.0282, 0.0, 0.0378] m in the default prim's frame (cylinder-diameter) as the closing line of a parallel-jaw grasp (GSP.001). Chosen from a projection preview of the mesh; not yet run through the FET005 grasp-and-lift benchmark.

## 2026-09-14 — 1.0.1

- SimReady Core metadata (`SimReady_Metadata`, `asset_name`, `asset_type`, `source_file`, `usd_date_generated`) stamped into the root layer's customLayerData from the sidecar (NP.006/SR.001).
- Kit's OmniPBR.mdl bundled under `materials/` and the 3 MDL shader source paths made relative (NP.008/VM.MDL.001), the layout NVIDIA's sample packages use.
- Root prims `Render` beside the default prim removed (HI.001); a referencing scene never composed them.

## 2026-09-15 — 1.0.0

Packaged as `object_library/rs_pro_stepper_motor_535_0502_nema23_class/usd/rs_pro_stepper_motor_535_0502_nema23_class.usd` from `object_library/items/industrial/RS_PRO_Stepper_Motor_535_0502_NEMA23_Class/RS_PRO_Stepper_Motor_535_0502_NEMA23_Class.usd` (source: authored in this repo). No content change; the version is the sidecar's.
