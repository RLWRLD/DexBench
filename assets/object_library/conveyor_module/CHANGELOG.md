# conveyor_module

Newest first. A version with no entry here is not a version.

## 2026-11-07 — 1.1.1

- `usd/textures/` removed: a byte-identical copy of `usd/deps/conveyor_mesh/textures/`, left when 1.0.2 moved the mesh under
  `deps/`. No layer references it; `conveyor_mesh.usd` uses the copy under `deps/`. No content change.

## 2026-09-20 — 1.1.0

- Physics runtime variants `PhysX`, `Newton` on the default prim (SimReady RV.001-RV.011), `Disabled` by default; `Enabled` composes `usd/runnables/physics/physx.usd` / `usd/runnables/physics/newton.usd`. The engine schemas and attributes the base used to author now live in those payloads (PhysX: 1 prims), so the base composes neutral OpenUSD physics until an engine is selected; DexBench spawners select the engine they run, scene layers select `PhysX`. The Newton payload annotates the colliders (`NewtonCollisionAPI`, `NewtonMeshCollisionAPI`) and the physics materials (`NewtonMaterialAPI`); contact gap and margin stay the engine's.

## 2026-09-15 — 1.0.2

- Sibling package(s) `conveyor_mesh` copied under `deps/` and the 1 references rewritten to them, so the package is self-contained (AA.001); the same LFS objects.

## 2026-09-14 — 1.0.1

- SimReady Core metadata (`SimReady_Metadata`, `asset_name`, `asset_type`, `source_file`, `usd_date_generated`) stamped into the root layer's customLayerData from the sidecar (NP.006/SR.001).

## 2026-09-15 — 1.0.0

Packaged as `object_library/conveyor_module/usd/conveyor_module.usda` from `object_library/fixtures/conveyor/conveyor_module.usda` (source: deliveries/psl/e3_manufacturing/260827). No content change; the version is the sidecar's.
