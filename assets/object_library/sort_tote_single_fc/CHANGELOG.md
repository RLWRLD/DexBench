# sort_tote_single_fc

Newest first. A version with no entry here is not a version.

## 2026-10-22 — 1.0.0

- First build: one sort tote for the E2 sort table in place of PSL's three (`Slot_SortTote_01..03`). sort_tote_fc 2.0.0's shell (PSL E2 261014) 9-sliced along x and y to 0.80 x 0.56 x 0.26 m outside, 0.78 x 0.54 x 0.25 m inside: the same 10 mm walls and floor, rim, `BK_MAT_E2_TotePlastic_*` material and textures (unscaled texel density; `usd/textures/` holds byte-for-byte copies of sort_tote_fc 2.0.1's four textures, capped at 1024 px, the same LFS objects, so the package resolves on its own and adds no new payload). Colliders: floor and 4 walls as boxes, guide purpose, bound to the package's physics material (static 0.6, dynamic 0.5, restitution 0), as sort_tote_fc 2.0.0. Text layers (`usd/sort_tote_single_fc.usda`, physics runtime variant `Newton` in `usd/runnables/physics/newton.usda`), `Disabled` by default.
- SimReady Core metadata (`SimReady_Metadata`, `asset_name`, `asset_type`, `source_file`, `usd_date_generated`) stamped into both layers' customLayerData from the sidecar (NP.006/SR.001), and vertex normals authored on the five collider meshes (VG.027), by `scripts/tools/conform_simready_basics.py --stamp --normals`.
