# logistics_conveyor

Newest first. A version with no entry here is not a version.

## 2026-10-21 — 2.0.1

- Textures capped at 1024 px on the longest side for rendering cost: the 22 textures in `usd/textures/` that were larger (2048 px) are downscaled in place (Lanczos, aspect kept, same name and format, so no layer changes; the 2 normal maps resampled as float and renormalised), 60.9 MB -> 15.1 MB. Geometry, UVs, materials and colliders are unchanged, and the SimReady record carries over: the files keep their paths, formats and colour spaces, only the pixel count changes. PATCH: nothing a reference composes changes.

## 2026-10-20 — 2.0.0

- Renamed to neutral names (breaking: the package path and every prim path change). The package is `object_library/logistics_conveyor` with `usd/logistics_conveyor.usda` and `usd/geometry/logistics_conveyor_geom.usdc`; the default prim is `/LogisticsConveyor`; the vendor's brand prefix is gone from every prim, look and material name (`Belts/HB2_Belt`, `Rollers/Roller1`, `Rollers/Joint_Roller1`, ...); the control inputs are `conveyor:control:{run, hb2_speed_mps, fb1_speed_mps}` and the control API `dexbench.conveyor_control/2`; the build provenance strings in the root layer's custom data use the same neutral names. Geometry, colliders, physics materials, looks, textures and both runtime variants are otherwise unchanged; the geometry layer was rewritten from its text form with the renamed prims only, so the recorded `geom_sha256` is the source build's, not this file's. The SimReady verdict below carries over: the rename changes names, not what the validator checks.

## 2026-10-14 — 1.0.0

- Packaged from PSL's E2 delivery (`deliveries/psl/e2_logistics/261014`), the conveyor module built from the real equipment's measured dimensions: HB2 infeed and FB1 outfeed belts, two free rollers, the work plate, frame, legs, covers and the control box, with their colliders, physics materials and the belt drive (`Control`: three control inputs driving each belt's surface velocity through OmniGraph, which needs Isaac Sim's `isaacsim.asset.gen.conveyor` extension). The delivery's `conveyor_module.usda` became this package's root layer; only its asset paths changed: the geometry it references moved from `_build/` to `usd/geometry/` and its 22 textures from `textures/looks/` to `usd/textures/`, all byte for byte. E3 261013 ships the same module, byte for byte.
- Physics runtime variants (`author_physics_variants.py`'s own functions): 47 colliders and 4 physics materials annotated in `usd/runnables/physics/newton.usd`; the PhysX schemas and attributes (the belts' surface velocity, the rollers' PhysX rigid-body settings) moved to `usd/runnables/physics/physx.usd`; both `Disabled` by default.
- SimReady validated with simready-validate 2026.7.1 (SimReady Foundation 2026.07.1, tier-core d5c4c95) on 2026-10-14: Robotics-Prop 3.1.0 **pass** (scope static).
