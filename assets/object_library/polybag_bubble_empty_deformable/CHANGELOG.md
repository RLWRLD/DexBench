# polybag_bubble_empty_deformable

Newest first. A version with no entry here is not a version.

## 2026-10-21 — 1.0.3

- Textures capped at 1024 px on the longest side for rendering cost: the one texture in `usd/textures/` that was larger (2048 px) is downscaled in place (Lanczos, aspect kept, same name and format, so no layer changes), 1.6 MB -> 0.2 MB. The other one, already at or below 1024 px, is unchanged. Geometry, UVs, materials and colliders are unchanged, and the SimReady record carries over: the files keep their paths and formats, only the pixel count changes. The PNGs' ancillary chunks (sRGB, gAMA, cHRM) are not carried; every binding to these files declares its colour space (`sourceColorSpace` or `colorSpace` metadata, none `auto`), so the renderer reads them as before. PATCH: nothing a reference composes changes.

## 2026-09-17 — 1.0.2

- Area-weighted vertex normals authored on 3 meshes that had none, a mismatched count or normals against their windings, with subdivisionScheme = none (VG.027/VG.028/VG.029).
- 2 relative asset paths anchored with `./` (AA.001).

## 2026-09-16 — 1.0.1

- Simulated on Newton 1.5.2 standalone (warp 1.17, SolverVBD), not the Arena image: film as VBD cloth with the 292 seal springs, film self-contact off (the two flat layers lie 1.2 mm apart, closer than the vendor contact radius); the flat film starts 0.15 m up at zero g (a cube pad cannot get under a film lying on a table), two pads come in from the side at lip height and pinch the sealed lip, gravity returns, lift 0.36 m, hold, shake, hold, open, the lower pad drops away. Film lifted clear of the table (mean height 151 -> 390 mm), held through the shake, dropped back on release (9 mm). PASS.

## 2026-09-15 — 1.0.0

- SimReady Core metadata (`SimReady_Metadata`, `asset_name`, `asset_type`, `source_file`, `usd_date_generated`) stamped into the root layer's customLayerData from the sidecar (NP.006/SR.001).
- Packaged from `deliveries/spaceai/260911/polybag_green_bubble_empty.usdz`: the delivery's inner layer `asset.usdc` re-exported as `usd/polybag_bubble_empty_deformable.usd`, textures moved to `usd/textures/` (green_albedo.png, green_normal.png) and the shader inputs re-based. Geometry, tetrahedra and physics values are the vendor's. Deformable (vbd_cloth): meant for Newton's VBD solver, not for PhysX rigid-body profiles.
