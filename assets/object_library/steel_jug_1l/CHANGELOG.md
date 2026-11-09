# steel_jug_1l

## 2026-11-08 — 1.2.1

- Rename the package, catalog ID and main USD from `ikea_idealisk_jug_1l` to `steel_jug_1l`; use an unbranded description and asset-name metadata.
- Geometry, textures, collision layers, fluid profile and physical parameters are unchanged. Keep the `/IDEALISK` prim, `idealisk` task key and scene/registration identifiers for existing recordings and frozen benchmark cells. Earlier entries below retain the historical source names.

Newest first. A version with no entry here is not a version.

## 2026-10-04 — 1.2.0

- Physics runtime variants `PhysX`, `Newton` on the default prim (SimReady RV.001-RV.011), `Disabled` by default; `Enabled` composes `usd/runnables/physics/physx.usd` / `usd/runnables/physics/newton.usd`. The engine schemas and attributes the base used to author now live in those payloads (PhysX: 1 prims), so the base composes neutral OpenUSD physics until an engine is selected; DexBench spawners select the engine they run, scene layers select `PhysX`. The Newton payload brings the 1 convex pieces of the former `usd/variants/` layer in under the original collider and switches that collider off (1); the flattened variant layer is gone, the same geometry now composes through the variant.
- The Newton payload references the collision mesh at `../../SubUSDs/idealisk_collision.usdc`, re-anchored from the variant layer's `../SubUSDs/`.
- Re-validated with simready-validate 2026.7.1: Robotics-Prop 3.1.0 passes its scope, `FET_003_PHYSX` and `FET_003_NEWTON` pass.

## 2026-09-30 — 1.1.0

- Grasp line `grasp_identifier_01` authored from [-0.022, -0.1, 0.065] to [-0.022, 0.1, 0.065] m in the default prim's frame (GSP.001): across Y through the cavity axis at half height, the same rule as `prop_jug_1l`; both jaws sit 4 cm clear of the tapered wall and the 16 mm handle strap is left to the hand. Hand-placed from the per-height extent profile of the meshes (the proposer's rule for this shape would be `container-squeeze` at (0, ±0.099, 0.080)) and pinned in DexBench's `grasp_line_overrides.json` so a pool-wide `--replace` keeps it; not yet run through the FET005 grasp-and-lift benchmark. The same prim is authored on the Newton variant, which composes its own `/IDEALISK`.
- Physics material `PhysicsMaterials/BrushedStainless` (static 0.6, dynamic 0.5, restitution 0.1; a ballpark for brushed stainless on laminate, not measured) bound with the `physics` purpose to the SDF collider on the main layer and to the watertight proxy on the Newton variant (PMT.001). Before this the `physics` binding resolved to the render material, which carries no `PhysicsMaterialAPI`.
- The baked `DomeLight` `/IDEALISK/env_light` is removed from `SubUSDs/idealisk_visual.usdc`. It came along from the Blender export and lit every scene the jug was spawned into with its own near-black environment; the E4 spike scripts had to deactivate it by hand. The helper layer is no longer byte-identical to the retired `import/idealisk` copy for that reason alone. `SubUSDs/textures/color_0C0C0C.exr`, which only that light referenced, goes with it (review of HF PR #8).
- Sidecars follow the pool's variant shape: the base lists `asset.variants.newton`, the variant carries `variant`, `variant_of`, `newton_collision` and its own `grasp_lines`; the `SimReady_Metadata` stamp on both layers says 1.1.0. MINOR: additions and the light removal, which no reference observes; geometry, pivot, units and the prim paths a scene addresses are unchanged.

## 2026-09-29 — 1.0.0

- First pooled. Authored in-house in Blender 5.2.0 (export dated 2026-09-02); previously carried in the DexBench repo under `import/idealisk` and `import/idealisk_newton`.
- Main layer (PhysX): rigid body, mass 0.22 kg, SDF triangle-mesh collider on the render mesh (`sdfResolution` 500 keeps the handle hole open, contact offset 5 mm, rest offset 0). Content unchanged from `import/idealisk/idealisk.usda`; only the visual reference moved to `./SubUSDs/`.
- Newton variant `usd/variants/ikea_idealisk_jug_1l_newton.usd`: same render mesh plus an invisible watertight collision proxy (`SubUSDs/idealisk_collision.usdc`, 5–6 mm wall, `physics:approximation = "none"`, Newton contact gap 1 mm).
- `usd/idealisk_profile.npz`: measured interior profile the MPM water sidecar lathes into the fluid container.
- Helper layers and textures are byte-identical to the repo copies.
- SimReady, recorded in the sidecar (`asset.simready`, `asset.simready_physx`): Prop-Robotics-Neutral 2.0.0 and Prop-Robotics-Physx 2.1.0 **fail** on FET005 only, with simready-validate 2026.7.1 (tier-core 2026.7.1). GSP.001: no grasp line authored yet. PMT.001: the collider carries no physics material. FET000/001/003/006 pass; FET004 is optional. Both gaps are left for a MINOR bump.
