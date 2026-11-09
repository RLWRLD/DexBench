# fork_d00313

Newest first. A version with no entry here is not a version.

## 2026-11-09 — 1.0.0

First pooled: DB-H3's `PalatialSim_D00313`, until now distributed outside this tree in
an internal revision of DB-H3's evaluation inputs (`eval/db_h3_inputs/asset_payload/`).
Palatial Platforms — Palatial NV Asset Library, item D00313 (https://huggingface.co/datasets/PalatialSim/palatial-nv-asset-library), CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/); prepared for DB-H3 by RLWRLD (rigid body enabled, delivered grasp line deactivated).

- **Composition unchanged.** The wrapper layer is the published one with two edits: its
  reference now points at `deps/D00313/` inside this package, and its root
  `customLayerData` `h3_*` notes moved to the sidecar (`derived_from.layer_metadata`) in place of
  the SimReady Core stamp. Every composed prim, applied schema, variant selection and authored
  attribute value, physics and geometry included, is identical to the published wrapper's, with
  every asset path resolving to a file of the same SHA-256.
- **Dependencies.** `usd/deps/D00313/` holds the upstream files the wrapper loads, from
  `PalatialSim/palatial-nv-asset-library` at `69a87329`, byte-identical to that revision.
- **Known gaps:** Robotics-Prop 3.1.0 fails as recorded in the sidecar and listed in
  `scripts/tools/assets_lint_allowlist.json`; closing them edits collision or grasp prims, which would
  move DB-H3's recorded physics, so they wait for a decision.
