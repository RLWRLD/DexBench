# power_drill

Newest first. A version with no entry here is not a version.

## 2026-09-20 — 1.3.0 (power_drill_1.0.0_SimReady/Power_Drill.usd)

- Physics runtime variants `PhysX`, `Newton` on the default prim (SimReady RV.001-RV.011), `Disabled` by default; `Enabled` composes `power_drill_1.0.0_SimReady/runnables/physics/physx.usd` / `power_drill_1.0.0_SimReady/runnables/physics/newton.usd`. The engine schemas and attributes the base used to author now live in those payloads (PhysX: 13 prims), so the base composes neutral OpenUSD physics until an engine is selected; DexBench spawners select the engine they run, scene layers select `PhysX`. The Newton payload brings the 200 convex pieces of the former `usd/variants/` layer in under the original collider and switches that collider off (11); the flattened variant layer is gone, the same geometry now composes through the variant.

## 2026-09-16 — 1.2.1

- Newton 1.5.2 cert of the `newton` collision variant (standalone MuJoCo-Warp solver, not the Arena image): 200 colliders imported as authored (0 replaced, 0 thickened by the runner); ground_drop pass, slope_drop pass; grasp_and_lift PASS on `grasp_identifier_01` with the line riding on the body (stage frame: pass). Report: the maintainers' Newton 1.5 grasp benchmark.

## 2026-09-16 — 1.2.0

- Added the Newton collision variant `power_drill_1.0.0_SimReady/variants/Power_Drill_newton.usd`: a flattened copy of this package in which `body` (sdf, 32 pieces), `body_vent` (sdf, 32 pieces), `body_cap` (sdf, 1 piece, thickened to 1 mm), `grip` (sdf, 32 pieces), `battery_clip_r` (sdf, 2 pieces), `battery_clip_l` (sdf, 2 pieces), `battery` (sdf, 32 pieces), `chuck` (sdf, 32 pieces), `bit_tip` (sdf, 32 pieces), `bit` (sdf, 1 piece, thickened to 1 mm), `trigger` (sdf, 2 pieces) are replaced by pre-decomposed convex pieces (CoACD threshold 0.05, <= 64 vertices each, 2 thickened to 1 mm), because Newton's solvers collide a mesh as one 64-vertex convex hull. Same render mesh, same physics material, same mass. Parsed back through Newton's importer: every collider convex, nothing replaced. The package USD is unchanged.

## 2026-09-16 — 1.1.7

- Newton 1.5.2 cert (standalone MuJoCo-Warp solver, not the Arena image): parse OK; 11 `sdf` collider(s) collided as 64-vertex convex hulls (MuJoCo-Warp keeps no concavity); ground_drop pass, slope_drop pass; grasp_and_lift PASS on `grasp_identifier_01` with the line riding on the body (line kept in the stage frame after the settle: pass; stock Newton contact defaults: fail). Certified package version 1.1.6. Settings and report: the maintainers' Newton 1.5 grasp benchmark.

## 2026-09-16 — 1.1.6

- Grasp line verified by NVIDIA's FET005 grasp-and-lift benchmark on Isaac Sim 6.0.1 (PhysX): a parallel-jaw gripper closed on `grasp_identifier_01`, lifted, held and shook the asset without losing it.

## 2026-09-15 — 1.1.5

- Grasp line `grasp_identifier_01` re-placed from [-0.043, 0.067, 0.087] to [0.043, 0.067, 0.087] m in the default prim's frame (reviewed-grip: across the grip at mid height and mid depth (the earlier hand-placed line ran along the front edge of the grip and the pads met nothing); passes NVIDIA's floor-standing grasp-and-lift): both endpoints now sit 2 cm clear of the surface, because NVIDIA's grasp-and-lift benchmark builds its jaw pads at the endpoints and a pad inside the collider launches the asset.

## 2026-09-16 — 1.1.4

- Free-space variant of NVIDIA's grasp-and-lift benchmark: still FAILED on every line (id01: Grasping). Grasping failed: pads touched (no object)

## 2026-09-15 — 1.1.3

- Grasp line FAILED NVIDIA's FET005 grasp-and-lift benchmark on Isaac Sim 6.0.1 (PhysX): The authored line and 4 sweep candidates (rot90, h40, h50, h60) all lost the asset. Engine session kit_94_3 crashed before the test could complete (retries exhausted). Classified `tooling`: the engine session crashed before the test completed: Engine session kit_94_3 crashed before the test could complete (retries exhausted).

## 2026-09-15 — 1.1.2

- Grasp line `grasp_identifier_01` re-placed from [-0.0557, 0.0278, 0.0823] to [0.0557, 0.0278, 0.0823] m in the default prim's frame (reviewed: standing; jaws across the pistol grip under the barrel): both endpoints now sit 2 cm clear of the surface, because NVIDIA's grasp-and-lift benchmark builds its jaw pads at the endpoints and a pad inside the collider launches the asset.

## 2026-09-15 — 1.1.1

- Grasp line re-shaped as an `Xform "grasp_identifier_01"` holding the `BasisCurves "grasp_identifier_01_line"` (same two points): NVIDIA's runtime grasp-and-lift benchmark looks for the Xform and reads its child, the static GSP.001 rule for a curve of that name; the plain curve satisfied only the latter.

## 2026-09-15 — 1.1.0

- Grasp line `grasp_identifier_01` authored from [-0.0357, 0.0278, 0.0823] to [0.0357, 0.0278, 0.0823] m in the default prim's frame (reviewed: standing; jaws across the pistol grip under the barrel) as the closing line of a parallel-jaw grasp (GSP.001). Chosen from a projection preview of the mesh; not yet run through the FET005 grasp-and-lift benchmark.

## 2026-09-15 — 1.0.0

Packaged as `object_library/power_drill/power_drill_1.0.0_SimReady` from `object_library/items/tools/power_drill_1.0.0_SimReady` (source: authored in this repo). No content change; the version is the sidecar's.
