# box_2

Newest first. A version with no entry here is not a version.

## 2026-11-06 — 2.0.0

- `usd/Textures/material_0.png`: every print of the scan (waybills, marks, logos, barcodes, text) painted out with the board around it, 18 % of the covered texels (scripts/tools/relabel_l4_parcels.py `clean_texture`), then a design of its own in the scan's colours, drawn on each face seen from outside and written back texel by texel, only on texels no other face reads (`l4_parcel_design.BoxFaces`): a round DEX MALL mark, the product name, a barcode and an information block on +z, a DEX MALL band with a barcode on +y and a contents form (품명, 수량 / 중량) on +x and -x, in a dark brown ink. Geometry, UVs, the material and physics unchanged.
- A `waybill` mesh under the rigid body, bound to its own `Materials/waybill_mat` (UsdPreviewSurface reading `usd/Textures/waybill.jpg`): the fictional DX EXPRESS landscape label, 104 x 72 mm (a header band, a sort-code block, a masked made-up recipient, a random barcode and a QR code for dxexpress.example; the sidecar's `waybill`), on the -z face, the barcode face DB-L4 reads, following the scanned surface 0.8 mm outside it, top toward +x. Visual only, no collider. The package address and root/body/visual/collider prim paths change, requiring a major version.
- Renamed from `dream_depot_box` to `box_2` (DB-L4's boxes are box_1..box_5): the folder, `usd/box_2.usd` (the main layer re-encoded from ASCII `.usda` as a crate, so it takes the `*.usd` LFS rule instead of a per-path one; no content change), the sidecar, the root and geometry prims, the material `box_2_mat`, the visual `box_2_visual` and the collider pieces `box_2_piece_NNN`, in the main layer and `runnables/physics/newton.usd`.
- SimReady validated for this version with simready-validate 2026.7.1 (SimReady Foundation 2026.07.1) on 2026-11-06: Robotics-Prop 3.1.0 **pass**, the full handled-prop scope; every feature's outcome as the previous version's.
- The sidecar's new `authored` records where the scan comes from: a scan of a real parcel made in-house by RLWRLD staff, who own it, not a vendor delivery. Metadata only.

## 2026-10-23 — 1.3.0

- `usd/Textures/material_0.png` brightened with a gamma of 1.4 on its sRGB values (out = 255 * (in / 255) ** (1 / 1.4), RGB only, alpha untouched): its mean luminance was 55 against 90-107 for box, styrofoam_box and post_office_package_1, the other DB-L4 boxes. Gamma, not a gain, so the shipping label's whites stay unclipped. Geometry, UVs, material and physics unchanged.

## 2026-09-20 — 1.2.0

- Physics runtime variant `Newton` on the default prim (SimReady RV.001-RV.011), `Disabled` by default; `Enabled` composes `usd/runnables/physics/newton.usd`. The engine schemas and attributes the base used to author now live in that payload (Newton: 22 prims; the converter's MuJoCo data is dropped, nothing runs MuJoCo), so the base composes neutral OpenUSD physics until an engine is selected; DexBench spawners select the engine they run, scene layers select `PhysX`. The Newton payload annotates the colliders (`NewtonCollisionAPI`, `NewtonMeshCollisionAPI`) and the physics materials (`NewtonMaterialAPI`); contact gap and margin stay the engine's.

## 2026-09-16 — 1.1.4

- Newton 1.5.2 cert (standalone MuJoCo-Warp solver, not the Arena image): parse OK; ground_drop pass, slope_drop pass; grasp_and_lift PASS on `grasp_identifier_01` with the line riding on the body (line kept in the stage frame after the settle: pass; stock Newton contact defaults: fail). Certified package version 1.1.3. Settings and report: the maintainers' Newton 1.5 grasp benchmark.

## 2026-09-15 — 1.1.3

- Grasp line verified by NVIDIA's FET005 grasp-and-lift benchmark on Isaac Sim 6.0.1 (PhysX): a parallel-jaw gripper closed on `grasp_identifier_01`, lifted, held and shook the asset without losing it.

## 2026-09-15 — 1.1.2

- Grasp line `grasp_identifier_01` re-placed from [0.0, -0.1553, 0.088] to [0.0, 0.1553, 0.088] m in the default prim's frame (across-y): both endpoints now sit 2 cm clear of the surface, because NVIDIA's grasp-and-lift benchmark builds its jaw pads at the endpoints and a pad inside the collider launches the asset.

## 2026-09-15 — 1.1.1

- Grasp line re-shaped as an `Xform "grasp_identifier_01"` holding the `BasisCurves "grasp_identifier_01_line"` (same two points): NVIDIA's runtime grasp-and-lift benchmark looks for the Xform and reads its child, the static GSP.001 rule for a curve of that name; the plain curve satisfied only the latter.

## 2026-09-15 — 1.1.0

- Grasp line `grasp_identifier_01` authored from [0.0, -0.1353, 0.088] to [0.0, 0.1353, 0.088] m in the default prim's frame (across-y) as the closing line of a parallel-jaw grasp (GSP.001). Chosen from a projection preview of the mesh; not yet run through the FET005 grasp-and-lift benchmark.

## 2026-09-15 — 1.0.2

- Area-weighted vertex normals authored on 11 meshes that had none or a mismatched count, with subdivisionScheme = none (VG.027/VG.028).

## 2026-09-14 — 1.0.1

- SimReady Core metadata (`SimReady_Metadata`, `asset_name`, `asset_type`, `source_file`, `usd_date_generated`) stamped into the root layer's customLayerData from the sidecar (NP.006/SR.001).
- Root prims `PhysicsScene` beside the default prim removed (HI.001); a referencing scene never composed them.

## 2026-09-15 — 1.0.0

Packaged as `object_library/dream_depot_box/usd/dream_depot_box.usda` from `object_library/meshes/dream_depot_box/dream_depot_box.usda` (source: authored in this repo). No content change; the version is the sidecar's.
