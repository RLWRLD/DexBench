# polybag_cotton_tet_deformable

Newest first. A version with no entry here is not a version.

## 2026-11-07 — 2.0.0

- Correct the version for retiring `usd/textures/green_albedo.png`: removing a shipped address is a MAJOR change, matching the box package renames. The prior 1.2.0 entry records the history of internal PR 497.
- No USD, texture, geometry or physics bytes change relative to internal PR 497 at `8abd93886`. Its 2026-11-06 SimReady result still describes these same payloads; no new validation run is claimed. DB-L1 and DB-L4 continue selecting their own albedos.

## 2026-11-06 — 1.2.0

- `usd/textures/green_albedo.png`, the delivered albedo with its Korea Post waybill (a real recipient, address and barcode), removed; the package's material (`diffuseColorTex`) reads `usd/textures/green_film.png` instead, that albedo with the waybill painted out by the weave next to it (make_polybag_sort_variants.py `clean_base`). No task draws the package's own material: DB-L4 points each bag at its `green_albedo_waybill_<parcel>.jpg` and DB-L1's sort variants carry their own albedos, so nothing a task renders changes. The delivered albedo stays in the history (1.1.0), where scripts/tools/relabel_l4_parcels.py takes it from. Geometry, tets and physics unchanged.
- SimReady validated for this version with simready-validate 2026.7.1 (SimReady Foundation 2026.07.1) on 2026-11-06: Robotics-Prop 3.1.0 **pass**, the deformable scope (Core, Minimal, Materials); every feature's outcome as the previous version's.

## 2026-11-06 — 1.1.0

- 5 albedos `usd/textures/green_albedo_waybill_<parcel>.jpg`, one per DB-L4 bag: `green_albedo.png` with the delivered Korea Post waybill painted out (make_polybag_sort_variants.py `clean_base`) and the fictional DX EXPRESS landscape label in its place, turned as the delivered one, each bag with its own made-up recipient and codes (scripts/tools/relabel_l4_parcels.py; the sidecar's `waybill`). The package's material still reads `green_albedo.png`, which DB-L1's pool is built from; DB-L4 points its bags at these. Files are added, hence the minor. Geometry, tets and physics unchanged.
- SimReady validated for this version with simready-validate 2026.7.1 (SimReady Foundation 2026.07.1) on 2026-11-06: Robotics-Prop 3.1.0 **pass**, the deformable scope (Core, Minimal, Materials); every feature's outcome as the previous version's.

## 2026-10-09 — 1.0.1

- SimReady Core metadata (`SimReady_Metadata`, `asset_name`, `asset_type`, `source_file`, `usd_date_generated`) stamped into the root layer's customLayerData from the sidecar (NP.006/SR.001).
- 1 gprims with no material binding bound to a UsdPreviewSurface in their displayColor (VM.MAT.001); renders as before.
- Area-weighted vertex normals authored on 1 meshes that had none, a mismatched count or normals against their windings, with subdivisionScheme = none (VG.027/VG.028/VG.029).
- Validated with `simready-validate` 2026.7.1 (SimReady Foundation 2026.07.1), `Robotics-Prop` 3.1.0: the deformable scope (Core, Minimal, Materials) PASS. The rigid-body features fail on RB.001 and do not apply to a deformable.
- The render mesh's authored normals describe its rest shape; a scene that deforms it (DexBench draws it from the tets) blocks them so the renderer recomputes them every frame.

## 2026-10-09 — 1.0.0

- Packaged from `deliveries/spaceai/261009/261009_polybag_single_tet_spaceai/polybag_green_cotton.usdz`: the delivery's inner layer `polybag_green_cotton.usdc` copied byte for byte as `usd/polybag_cotton_tet_deformable.usd`, its textures to `usd/textures/` (green_albedo.png, green_normal.png); the shader inputs already read `./textures/…`. Geometry, tetrahedra and physics values are the vendor's.
- Deformable (tet_softbody): one TetMesh, 536 vertices, 1770 tetrahedra, 90 g (density 22.4 kg/m³, Young's modulus 15 kPa, Poisson's ratio 0.15, kDamp 30, particle radius 4 mm, friction 0.8), and a 16506-vertex render mesh with the parcel label that follows it. Meant for Newton's VBD solver, not for PhysX rigid-body profiles.
- Simulated on Newton 1.6.1 standalone (warp 1.17, SolverVBD), not the Arena image, with the delivery's loading code and solver settings (8 substeps × 5 iterations, self-contact radius 1 mm margin 2 mm, bag-frame truncation, full-surface rigid-soft contact): 20 cm drop onto a plane, rests at its particle radius, finite. PASS. The vendor's own scenario report is beside the delivery (`261009_polybag_single_tet_spaceai.html`).
- SimReady Core metadata and conformance follow in 1.0.1.
