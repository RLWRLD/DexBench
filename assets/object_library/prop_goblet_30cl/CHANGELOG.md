# prop_goblet_30cl

Newest first. A version with no entry here is not a version.

## 2026-11-02 — 1.4.0

Fluid only; the USD, its colliders, mass and looks are unchanged, so 1.3.0's SimReady record still describes it.

- `usd/prop_goblet_30cl_profile.npz`: the bowl's interior lathe profile (`z`, `r_inner`, `floor_z`, `rim_z`), 256 rows
  from the bowl floor at 65.4 mm to the rim at 130.0 mm, 179.6 mL below the rim. Measured from the visual bowl mesh
  by `dexbench_arena.fluids.goblet_cavity.extract` (`scripts/tools/measure_goblet_cavity.py`): rays from the cavity
  axis on 16 bearings, the smallest radius of each height kept (bearings agree within 0.85 mm). DB-H2's MPM water
  builds the goblet's water collider from it and keeps its rendered surface inside the glass, as it does with a
  jug's `*_profile.npz`. Until now it was measured from the USD at every run, and the water read
  a copy in DexBench; the file is byte-identical to both.
- MINOR: an addition; nothing a placement depends on moved.

## 2026-10-04 — 1.3.0

- Re-sourced from PSL's E4 r01 delivery (`deliveries/psl/e4_hotel_fb/260927`). The bowl mesh is bound to r01's clear glass (`MAT_E4_glass_goblet`, Kit's OmniGlass MDL, bundled under `materials/`) instead of the textured `MAT_E4_Gen_goblet_30cl_0`. Geometry, colliders, mass unchanged.
- Base frame unchanged against the previous version: same default prim, no root transform, the same prim paths, point counts and collider bounds (`compare_usd_geometry.py` and a frame/bounds comparison, 2026-10-02).
- Textures replaced by the ones the delivered layer uses (0 in `usd/textures/`); the 3 previous ones are removed.
- The reviewed grasp line `/Prop_Goblet30cl/grasp_identifier_01` is carried over unchanged (same points, same mesh); its benchmark records keep the package version they certified.
- Kit's OmniGlass.mdl bundled under `materials/OmniGlass/` together with the two modules it reaches from its own folder, as in Kit's `mdl/core/Base`: `OmniGlass_Opacity.mdl` (`using OmniGlass_Opacity import ...`) and `OmniPBRBase.mdl`, which `OmniGlass_Opacity` imports in turn (`import OmniPBRBase::*`). Without either, RTX fails to compile the module (`[EntityResolver] FAILED ... OmniGlass::OmniPBRBase`) and the glass renders as an error material. All three are byte-identical to Kit 110.3's. The MDL shader source path is relative (NP.008/VM.MDL.001/AA.001), the layout NVIDIA's sample packages use.
- MINOR: a new look (materials) or an addition; nothing a placement depends on moved.
- SimReady validated for this version with simready-validate 2026.7.1 on 2026-10-04: Robotics-Prop 3.1.0 **pass** (scope full); `FET_003_PHYSX` and `FET_003_NEWTON` pass.
- Physics runtime variants `Newton` on the default prim (SimReady RV.001-RV.011), `Disabled` by default; `Enabled` composes `usd/runnables/physics/newton.usd`. The Newton payload annotates the colliders (`NewtonCollisionAPI`, `NewtonMeshCollisionAPI`) and the physics materials (`NewtonMaterialAPI`); contact gap and margin stay the engine's.

## 2026-09-20 — 1.2.0

- Physics runtime variants `Newton` on the default prim (SimReady RV.001-RV.011), `Disabled` by default; `Enabled` composes `usd/runnables/physics/newton.usd`. The Newton payload annotates the colliders (`NewtonCollisionAPI`, `NewtonMeshCollisionAPI`) and the physics materials (`NewtonMaterialAPI`); contact gap and margin stay the engine's.

## 2026-09-16 — 1.1.4

- Newton 1.5.2 cert (standalone MuJoCo-Warp solver, not the Arena image): parse OK; ground_drop pass, slope_drop pass; grasp_and_lift PASS on `grasp_identifier_01` with the line riding on the body (line kept in the stage frame after the settle: pass; stock Newton contact defaults: fail). Certified package version 1.1.3. Settings and report: the maintainers' Newton 1.5 grasp benchmark.

## 2026-09-15 — 1.1.3

- Grasp line verified by NVIDIA's FET005 grasp-and-lift benchmark on Isaac Sim 6.0.1 (PhysX): a parallel-jaw gripper closed on `grasp_identifier_01`, lifted, held and shook the asset without losing it.

## 2026-09-15 — 1.1.2

- Grasp line `grasp_identifier_01` re-placed from [-0.0575, 0.0, 0.0936] to [0.0575, 0.0, 0.0936] m in the default prim's frame (reviewed: jaws across the bowl, not the stem): both endpoints now sit 2 cm clear of the surface, because NVIDIA's grasp-and-lift benchmark builds its jaw pads at the endpoints and a pad inside the collider launches the asset.

## 2026-09-15 — 1.1.1

- Grasp line re-shaped as an `Xform "grasp_identifier_01"` holding the `BasisCurves "grasp_identifier_01_line"` (same two points): NVIDIA's runtime grasp-and-lift benchmark looks for the Xform and reads its child, the static GSP.001 rule for a curve of that name; the plain curve satisfied only the latter.

## 2026-09-15 — 1.1.0

- Grasp line `grasp_identifier_01` authored from [-0.0375, 0.0, 0.0936] to [0.0375, 0.0, 0.0936] m in the default prim's frame (reviewed: jaws across the bowl, not the stem) as the closing line of a parallel-jaw grasp (GSP.001). Chosen from a projection preview of the mesh; not yet run through the FET005 grasp-and-lift benchmark.

## 2026-09-14 — 1.0.1

- SimReady Core metadata (`SimReady_Metadata`, `asset_name`, `asset_type`, `source_file`, `usd_date_generated`) stamped into the root layer's customLayerData from the sidecar (NP.006/SR.001).

## 2026-09-15 — 1.0.0

Packaged as `object_library/prop_goblet_30cl/usd/prop_goblet_30cl.usd` from `object_library/items/tableware/prop_goblet_30cl.usdc` (source: deliveries/psl/e4_hotel_fb/260831). No content change; the version is the sidecar's.
