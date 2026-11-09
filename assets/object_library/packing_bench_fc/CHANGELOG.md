# packing_bench_fc

Newest first. A version with no entry here is not a version.

## 2026-10-21 — 3.0.1

- Textures capped at 1024 px on the longest side for rendering cost: the 28 textures in `usd/textures/` that were larger (1476-4096 px) are downscaled in place (Lanczos, aspect kept, same name and format, so no layer changes; the 8 normal maps resampled as float and renormalised), 42.6 MB -> 5.4 MB. The 8 already at or below 1024 px are unchanged. Geometry, UVs, materials and colliders are unchanged, and the SimReady record carries over: the files keep their paths and formats, only the pixel count changes. The PNGs' ancillary chunks (sRGB, gAMA, cHRM, eXIf, oFFs, pHYs) are not carried, nor the XMP packet of the four Leaking018A JPEGs; every binding to these files declares its colour space (`sourceColorSpace` or `colorSpace` metadata, none `auto`), so the renderer reads them as before. PATCH: nothing a reference composes changes.

## 2026-10-14 — 3.0.0

- Re-sourced from PSL's E2 delivery 261014 (`deliveries/psl/e2_logistics/261014`, `assets/fixtures/e2_packing_bench_r02.usdc`): r02's bench, with the monitor, posts, roll arms and pegboard remodelled and the upper cartons and the scale taken out of the bench mesh, under r02's finishes; plus the kitchen scale r02 stands on the worktop (`Deco/BG_E2_KitchenScale`, from `internal/world.usdc` `Slot_PackingBench/OwnedDeco/BG_E2_KitchenScale`, with its static collision box `ColBox`). The scale mesh is "Kitchen Scale 3D Model" by developrstore, CC BY 4.0, modified by PSL (credited in the sidecar). The rest of r02's bench dressing (binders, paper boxes, a crate) is bought material licensed for use inside the scene only and lives in the E2 cell's shell, not here; the outbound box is its own package. MAJOR: prim paths and geometry changed (`compare_usd_geometry.py` against 1.1.1, 2026-10-14).
- No 2.0.0: internal PR 257 (PSL's r01 delivery, superseded by this version, not merged) re-sourced this package as 2.0.0 on its branch, which is still on origin, so 2.0.0 already names r01's package and reusing it would give two payloads one version.
- Every collider (23) binds the package's `_physics/PhysicsMaterial` (static 0.6, dynamic 0.5, restitution 0): r02 binds visual materials that carry no physics material, which an engine reads as its default.
- Textures: the 36 the layer uses are in `usd/textures/`.
- Physics runtime variant `Newton` re-authored for these colliders (`author_physics_variants.py`'s own functions): 23 colliders and 1 physics materials annotated in `usd/runnables/physics/newton.usd`, `Disabled` by default; no PhysX data, so no PhysX payload.
- Kit's OmniGlass.mdl, OmniPBR.mdl bundled under `materials/` and 2 MDL shader source paths made relative in packing_bench_fc.usd (NP.008/VM.MDL.001/AA.001), the layout NVIDIA's sample packages use.
- SimReady validated with simready-validate 2026.7.1 (SimReady Foundation 2026.07.1, tier-core d5c4c95) on 2026-10-14: Robotics-Prop 3.1.0 **pass** (scope static).

## 2026-10-04 — 1.1.1

- Merged HF main's 1.0.2 (texture restored from its source delivery) into 1.1.0's runtime variants. The SimReady record is re-measured on the restored texture (Robotics-Prop 3.1.0).

## 2026-09-28 — 1.0.2

- Textures restored from the source delivery (deliveries/psl/e2_logistics/260827/fixtures/textures). The 1.0.0 packaging had taken the same-named `bench_*_2k.jpg` set from deliveries/psl/e3_manufacturing/260827 (the assembly workbench atlas), so the mesh UVs sampled the wrong atlas. Geometry, UVs and materials unchanged.

## 2026-09-20 — 1.1.0

- Physics runtime variants `Newton` on the default prim (SimReady RV.001-RV.011), `Disabled` by default; `Enabled` composes `usd/runnables/physics/newton.usd`. The Newton payload annotates the colliders (`NewtonCollisionAPI`, `NewtonMeshCollisionAPI`) and the physics materials (`NewtonMaterialAPI`); contact gap and margin stay the engine's.

## 2026-09-14 — 1.0.1

- SimReady Core metadata (`SimReady_Metadata`, `asset_name`, `asset_type`, `source_file`, `usd_date_generated`) stamped into the root layer's customLayerData from the sidecar (NP.006/SR.001).

## 2026-09-15 — 1.0.0

Packaged as `object_library/packing_bench_fc/usd/packing_bench_fc.usd` from `object_library/fixtures/packing_bench_fc.usdc` (source: deliveries/psl/e2_logistics/260827). No content change; the version is the sidecar's.
