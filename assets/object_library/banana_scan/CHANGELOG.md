# banana_scan

Newest first. A version with no entry here is not a version.

## 2026-09-20 — 1.2.0

- Physics runtime variants `Newton` on the default prim (SimReady RV.001-RV.011), `Disabled` by default; `Enabled` composes `usd/runnables/physics/newton.usd`. The Newton payload annotates the colliders (`NewtonCollisionAPI`, `NewtonMeshCollisionAPI`) and the physics materials (`NewtonMaterialAPI`); contact gap and margin stay the engine's.

## 2026-09-16 — 1.1.0

- Grasp line `grasp_identifier_01` authored from [-0.0583, -0.036, 0.0239] to [-0.0583, 0.036, 0.0239] m in the default prim's frame (elongated-narrowest) as the closing line of a parallel-jaw grasp (GSP.001). Chosen from a projection preview of the mesh; not yet run through the FET005 grasp-and-lift benchmark.

## 2026-09-16 — 1.0.1

- SimReady Core metadata (`SimReady_Metadata`, `asset_name`, `asset_type`, `source_file`, `usd_date_generated`) stamped into the root layer's customLayerData from the sidecar (NP.006/SR.001).
- Area-weighted vertex normals authored on 1 meshes that had none, a mismatched count or normals against their windings, with subdivisionScheme = none (VG.027/VG.028/VG.029).

## 2026-09-16 — 1.0.0

- SpaceAI scanned banana (2026-08-20 delivery, no newer scan) as a rigid prop: the scan's render mesh and textures, a convex hull collider, the delivery's density. Render mesh 17186 vertices, bottom set at z 0 and centred in x/y (shift [9e-05, -1e-05, 0.02397] m), convex hull collider on the render mesh, mass 0.09778 kg (delivery density 1010.0 kg/m3 x render-mesh volume 9.681e-05 m3), physics material 0.6/0.5. Grasp line, normals, extents and SimReady stamp follow from the conform and grasp-line tools.

