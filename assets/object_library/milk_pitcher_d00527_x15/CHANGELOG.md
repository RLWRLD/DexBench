# milk_pitcher_d00527_x15

## 2026-11-08 — Attribution clarification

Add the Palatial Platforms D00527 attribution, CC BY 4.0 links and the 1.5x scaling/repackaging notice. Payloads and physical settings are unchanged.


Newest first. A version with no entry here is not a version.

## 2026-10-31 — 2.0.0

- **Bottom collider:** the three delivered bottom plates `Colliders_00/Collider_0`, `Collider_1` and `Collider_12` are replaced by one convex frustum, `Colliders_00/Collider_bottom` (64-gon, convexHull, bound to `part_00_material`, guide purpose): bottom radius 75.4 mm on the plates' bottom plane, top radius 84.2 mm at 10.17 mm (the plates' top), centre (-5.4, -0.3) mm. The plates' lowest vertices lay 0-0.05 mm off one plane, so Newton's contact manifold, which keeps several points only for coplanar support samples, kept one point per plate and the jug never came to rest standing (DB-H2 condition recording, 8 substeps: 22 of 90 Train starts settled; the tall Trinix jug's case, steel_water_jug_tall). With the frustum 90 of 90 settle (last-second angular speed at most 0.0035 rad/s, no drift), at the same rest height. MAJOR: three collider prim paths are removed. Visual mesh, mass, centre of mass, inertia, physics material and the water files are unchanged.

## 2026-10-30 — 1.0.0

First pooled: PalatialSim's D00527 Milk Pitcher (HF `PalatialSim/palatial-nv-asset-library` @ 69a87329, CC-BY-4.0) scaled 1.5x for the DB-H2 pour, the delivered jug being small for the RBY1-M Wuji2 hand's caging grasps.

- **Scale:** uniform 1.5 in x, y and z; mass 0.25 -> 0.5625 kg (x 1.5^2, a thin shell keeps its wall), diagonal inertia x 1.5^4, centre of mass scaled with the geometry. Everything else as delivered (default prim `/World`, rigid body `/World/part_00`, delivered convex collision meshes and physics material).
- **Water:** `usd/milk_pitcher_d00527_x15_profile.npz` (interior lathe profile, rim 0.192 m) and `usd/milk_pitcher_d00527_x15_water_shell.npz` (MPM water collider: closed shell ray-cast from the render mesh, 2-voxel wall, 8 mm corners, thin handle).
- **Known gaps:** no SimReady validation record, no grasp line and no Newton runtime variant yet (the delivered physics is neutral UsdPhysics only); listed in `scripts/tools/assets_lint_allowlist.json`.
