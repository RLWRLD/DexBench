# cutlery_set_table_fork

Newest first. A version with no entry here is not a version.

## 2026-11-09 — 1.0.0

First pooled: DB-H3's `ForgeCAD_cutlery_set_Table_fork`, until now distributed outside this tree in
an internal revision of DB-H3's evaluation inputs (`eval/db_h3_inputs/asset_payload/`).
“cutlery_set” by ForgeCAD (https://huggingface.co/datasets/ForgeCAD/simready-assets), CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/); the table fork selected from the set and prepared for DB-H3 by RLWRLD (rigid body on the selected utensil, the other four deactivated).

- **Composition unchanged.** The wrapper layer is the published one with two edits: its
  reference now points at `deps/cutlery_set/` inside this package, and its root
  `customLayerData` `h3_*` notes moved to the sidecar (`derived_from.layer_metadata`) in place of
  the SimReady Core stamp. Every composed prim, applied schema, variant selection and authored
  attribute value, physics and geometry included, is identical to the published wrapper's, with
  every asset path resolving to a file of the same SHA-256.
- **Dependencies.** `usd/deps/cutlery_set/` holds the upstream files the wrapper loads, from
  `ForgeCAD/simready-assets` at `0573053f`, byte-identical to that revision.
- **Known gaps:** Robotics-Prop 3.1.0 fails as recorded in the sidecar and listed in
  `scripts/tools/assets_lint_allowlist.json`; closing them edits collision or grasp prims, which would
  move DB-H3's recorded physics, so they wait for a decision.
