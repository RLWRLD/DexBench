# steel_pitcher_large

Newest first. A version with no entry here is not a version.

## 2026-11-08 — 1.1.1

- Credit NdotLight (Trinix), CC BY 4.0; modified by RLWRLD. PATCH for attribution metadata only; source delivery identifiers, payloads, physics and certificates are unchanged.

## 2026-10-26 — 1.1.0

Fluid only; the rigid body, colliders, mass and looks are unchanged. The MPM water sidecar (`dexbench_arena/fluids/water_pour_setup.py`) finds both files beside the USD by name.

- `usd/steel_pitcher_large_profile.npz`: interior lathe profile from the render vessel (`Geom/vessel`), 121 rows, floor 1.6 mm, rim 22.75 cm. It seeds the water and gives the rim and floor heights; the shell above holds the water.
- `usd/steel_pitcher_large_water_shell.npz`: the MPM water collider, baked by `scripts/tools/bake_water_vessel_shell.py` from `Geom/vessel`. The faces that line the cavity are kept as they are and closed by a 16 mm wall behind them (two 8 mm voxels; the delivered sheet is thinner than the MPM contact band and fast water tunnels through it), the floor is lifted 8 mm and the file carries a contact margin of one voxel, so the spout is what the water sees. 1576 verts, 3132 tris.

## 2026-10-21 — 1.0.0

First pooled: `waterjug2.usdz` of Trinix's 2026-10-20 delivery (cad2sim 0.3.0 from a STEP file; archive address, file and member sha256 in the sidecar). The delivery itself stays outside the tree.

- **Frame and hierarchy.** Default prim `/steel_pitcher_large` (kind `component`) is the rigid body, with no transform op. The render meshes sit under `Geom/` with their transforms baked in. The handle already faced +X. The base centre stays at the origin. Size 0.198 x 0.152 x 0.239 m. The delivery's `/World/Untitled_Project/Pitcher_Body` nesting, its `PhysicsScene` and its grasp curve are gone.
- **Look:** hammered stainless, so the pitcher reads apart from its siblings. UsdPreviewSurface, as on `ikea_idealisk_jug_1l`.
  - `Looks/steel_hammered` (vessel): metallic 1, base colour (0.62, 0.62, 0.63), roughness 0.27, and a hammered-dimple normal map, `textures/hammered_steel_normal.png`. The map is procedural: irregular round facets, 64 per tile, about 9-14 mm across, 1024 px, tileable.
  - `Looks/steel_satin_bright` (handle): same colour, flat roughness 0.27.
  - The delivery's unwrap UVs give the facets no consistent scale, so the walls' `st` is re-projected cylindrically round the jug axis (4 tiles round, 8 cm per tile up). Floors and rims are projected planar. Points, normals and topology are as delivered.
- **Colliders, `Collisions/` (guide purpose), 46 pieces: 38 for the vessel, 7 for the handle, 1 base disc.** CoACD 1.0.14 was run on each part (vessel threshold 0.08, handle 0.03 capped at 7, at most 64 vertices per hull). The delivery's two handle hulls filled most of the loop; these keep it open.
  - Handle loop clearance 52.0 mm against the render's 52.8 at its widest, and at most 3.3 mm short at any height.
  - Bore radius 62.2 mm against 64.5 (median); the worst direction, near the spout, is 7.3 mm short.
  - Fit to the render surface: p95 6.3 mm, max 9.4 mm. The roughly 40-piece budget was kept: 87 pieces gave p95 3.1 mm.
  - The CoACD hulls round the base edge about 2 mm inside the render footprint. The `base_disc` hull (the render vessel's points within 1.5 mm of the floor, 1.15 mm thick) brings the support out to the visible edge, r 51.6 mm.
  - Every collider binds `PhysicsMaterials/steel` on its own prim: friction 0.6 / 0.48 as delivered, restitution 0.1 instead of 0.3.
- **Mass, centre of mass and inertia** come from sheet stainless (7900 kg/m^3) laid on the render surface and integrated triangle by triangle, so the result is the full inertia tensor.
  - Side walls are 0.7 mm. The floor is an encapsulated 3.0 mm base plate.
  - The 3.0 mm base is a modelling choice (an encapsulated bottom), not taken from a product reference. The delivered CAD measures 1.5 mm at the base and 1.4-1.6 mm in the walls. The base is 0.227 kg, 27 % of the mass, and it is what holds the centre of mass at 37 % of the height.
  - The tilt and drop results below hold because of this choice. With a 0.7 mm single-sheet base (a re-run in the review of internal PR 343, base mass scaled by 0.7/3), the pitcher falls in one 20 deg tilted drop, away from the handle (toward -X), and its static tip angle drops to 16.0 deg.
  - The rolled rim is 52 g/m, a 3 mm bead of 0.7 mm sheet, along the rim faces.
  - Handles are hollow tubular or pressed sections at 0.9 mm, and mounts are 1.0 mm.
  - A part the CAD models as a wall with two faces counts half its surface area. A part the CAD models as a solid counts its visible surface once, as one skin.
  - Not weighed. A hand estimate (cylinder shell + base disc + handle surface + rim, from measured dimensions) agrees within 5 %.
  - Newton loads exactly the authored mass, centre of mass and principal inertia.
  - **Total 0.840 kg:** walls 0.524, base 0.227, rim 0.025, handle 0.064 (hollow pressed strap). That fits a 2.8 L pitcher.
  - Centre of mass (0.0069, 0, 0.0886) m: 7 mm toward the handle, 37 % of the height.
  - Principal inertia (0.00307, 0.00667, 0.00714) kg m^2.
  - Tip-over angle 25.9 deg: geometric, base only. It is where the centre of mass passes the edge of the base colliders, and it does not count the handle. Simulated static releases on Newton main (sidecar `newton_smoke`) fall at 28-31 deg toward the handle and the sides; the weakest direction, in Newton contacts, is the diagonal away from the handle, 21.7 deg toward -X+Y (33.3 deg in MuJoCo contacts).
- **Physics runtime variant `Newton`** (author_physics_variants.py): `Disabled` by default. There is no PhysX payload (internal issue 313).
- **Grasp line `grasp_identifier_01`** from (0, -0.0893, 0.12) to (0, 0.0893, 0.12) m: across Y through the vessel axis at half height. The proposer's line ran along X through the handle and the spout. It is pinned in `grasp_line_overrides.json` and has not been simulated.
- **SimReady:** Robotics-Prop 3.1.0 passes its full scope with simready-validate 2026.7.1, including `FET_003_NEWTON` and `FET_005`.
- **Newton smoke** (sidecar `newton_smoke`; the maintainers' `smoke_newton.py`, standalone, dt 1/480, an internal simulation image): Newton main 9c28c80c with MuJoCo 3.15, 2026-10-22. The cases are upright 2 cm drops, 2 cm drops leaning 10 and 20 deg toward eight directions, static releases at 10 and 15 deg toward the same eight, and a bisection of the static tip angle per direction.
  - Newton and MuJoCo contacts: 0 falls. Upright drops, all 16 tilted drops and all 16 static releases recover.
  - All three jugs and `prop_jug_1l` show lower Newton-contact than MuJoCo-contact tip angles toward the -X diagonals on Newton main (`prop_jug_1l` 24.8 against 30.1 deg); `ikea_idealisk_jug_1l` shows it toward +X-Y (29.4 against 34.3 deg). Tracked in internal issue 351.
  - Nothing goes NaN. No contact stiffness or damping is authored.
  - `ikea_idealisk_jug_1l` and `prop_jug_1l`: 0 falls in either contact mode in the same harness.
  - A 7 mm capsule finger through the handle loop lifted the pitcher 8.8 cm.
  - Newton 1.6.1 (secondary record, same harness and package): 0 falls in either contact mode; hook lift 8.8 cm. The 2026-10-21 run about three axes also had 0 falls.

Left as delivered, deliberately: the render meshes and the friction values.
