# apple_scan

Newest first. A version with no entry here is not a version.

## 2026-11-06 — 1.3.1

- `twin.deformable` is null: the deformable twin `object_library/apple_deformable` was removed from the pool, since no task loads it. The scan's geometry, physics and textures are unchanged.

## 2026-10-29 — 1.3.0

- Replace the Newton runtime's scan-mesh collider with one 64-vertex convex hull, bound directly to the package physics material. Keep the scan mesh and textures for rendering, and preserve authored mass. Inertia is derived from the hull. Maximum bbox change 1.11 mm; convex volume ratio 0.9112.

## 2026-09-20 — 1.2.0

- Physics runtime variants `Newton` on the default prim (SimReady RV.001-RV.011), `Disabled` by default; `Enabled` composes `usd/runnables/physics/newton.usd`. The Newton payload annotates the colliders (`NewtonCollisionAPI`, `NewtonMeshCollisionAPI`) and the physics materials (`NewtonMaterialAPI`); contact gap and margin stay the engine's.

## 2026-09-16 — 1.1.0

- Grasp line `grasp_identifier_01` authored from [-0.059, 0.0, 0.0381] to [0.059, 0.0, 0.0381] m in the default prim's frame (cylinder-diameter) as the closing line of a parallel-jaw grasp (GSP.001). Chosen from a projection preview of the mesh; not yet run through the FET005 grasp-and-lift benchmark.

## 2026-09-16 — 1.0.1

- SimReady Core metadata (`SimReady_Metadata`, `asset_name`, `asset_type`, `source_file`, `usd_date_generated`) stamped into the root layer's customLayerData from the sidecar (NP.006/SR.001).
- Area-weighted vertex normals authored on 1 meshes that had none, a mismatched count or normals against their windings, with subdivisionScheme = none (VG.027/VG.028/VG.029).

## 2026-09-16 — 1.0.0

- SpaceAI scanned apple (2026-09-14 delivery) as a rigid prop: the scan's render mesh and texture, a convex hull collider, the delivery's density. Render mesh 14998 vertices, bottom set at z 0 and centred in x/y (shift [0.00031, -0.00026, 0.03725] m), convex hull collider on the render mesh, mass 0.22883 kg (delivery density 840.0 kg/m3 x render-mesh volume 2.724e-04 m3), physics material 0.6/0.5. Deformable twin for the Newton backend: `object_library/apple_deformable`. Grasp line, normals, extents and SimReady stamp follow from the conform and grasp-line tools.

