# steel_pitcher_footed

Newest first. A version with no entry here is not a version.

## 2026-11-08 — 1.1.1

- Credit NdotLight (Trinix), CC BY 4.0; modified by RLWRLD. PATCH for attribution metadata only; source delivery identifiers, payloads, physics and certificates are unchanged.

## 2026-10-26 — 1.1.0

Fluid only; the rigid body, colliders, mass and looks are unchanged. The MPM water sidecar (`dexbench_arena/fluids/water_pour_setup.py`) finds both files beside the USD by name.

- `usd/steel_pitcher_footed_profile.npz`: interior lathe profile from the render vessel and foot (`Geom/vessel`, `Geom/foot`; 61 levels, 96 rays per level), continued above the lowest full ring (17.05 cm) up to the spout lip (-X) at 18.08 cm, 130 rows. Floor 31.8 mm. It seeds the water and gives the rim and floor heights; the shell above holds the water.
- `usd/steel_pitcher_footed_water_shell.npz`: the MPM water collider, baked by `scripts/tools/bake_water_vessel_shell.py` from `Geom/vessel` and `Geom/foot`. The faces that line the cavity are kept as they are and closed by a 16 mm wall behind them (two 8 mm voxels; the delivered sheet is thinner than the MPM contact band and fast water tunnels through it), the floor is lifted 8 mm and the file carries a contact margin of one voxel, so the spout is what the water sees. 1388 verts, 2772 tris.

## 2026-10-21 — 1.0.0

First pooled: `waterjug3.usdz` of Trinix's 2026-10-20 delivery (cad2sim 0.3.0 from a STEP file; archive address, file and member sha256 in the sidecar). The delivery itself stays outside the tree.

- **Frame and hierarchy.** Default prim `/steel_pitcher_footed` (kind `component`) is the rigid body, with no transform op. The render meshes sit under `Geom/` with their transforms baked in, and the pitcher is turned 180 deg about Z so the handle faces +X. The base centre stays at the origin. Size 0.183 x 0.121 x 0.193 m. The delivery's `/World/Untitled_Project/Pitcher_Body` nesting, its `PhysicsScene` and its grasp curve are gone.
- **Look:** copper plating over the stainless body, so the pitcher reads apart from its siblings. UsdPreviewSurface, as on `ikea_idealisk_jug_1l`.
  - `Looks/copper_plated` (vessel and foot): metallic 1, base colour (0.95, 0.64, 0.54), roughness from a polishing-swirl map, `textures/plated_swirl_roughness.png` (0.25 ± 0.03, procedural rings, 1024 px, tileable).
  - `Looks/copper_plated_handle`: the same plating, flat roughness 0.25.
  - Copper read more like real serveware than a brass gold-tone on this footed shape.
  - The plating is a finish only: mass, centre of mass and inertia are the stainless body's below.
  - The walls' `st` is re-projected cylindrically round the jug axis (4 tiles round, 8 cm per tile up). Floors and rims are projected planar. Points, normals and topology are as delivered.
- **Colliders, `Collisions/` (guide purpose), 57 pieces: 46 for the vessel, 7 for the handle, 4 for the foot.** CoACD 1.0.14 was run on each part (vessel threshold 0.08, handle 0.03 capped at 7, foot 0.05 capped at 4, at most 64 vertices per hull). The delivery's handle hull filled the top of the loop; these keep it open.
  - Handle loop clearance 59.0 mm against the render's 60.5 at its widest, and at most 2.0 mm short.
  - Bore radius 47.2 mm against 49.9 (median); the worst direction is 5.5 mm short.
  - Fit of the vessel and handle pieces: p95 4.3 and 2.2 mm.
  - The foot hulls fill the hollow underside of the foot. Nothing reaches there, and the foot ring's outer edge (r 47.5 mm, the same as the render's) is the support.
  - The piece count is above the roughly 40 asked for. Merging the vessel down to 22 pieces cost 13 mm of bore.
  - Every collider binds `PhysicsMaterials/steel` on its own prim: friction 0.6 / 0.48 as delivered, restitution 0.1 instead of 0.3.
- **Mass, centre of mass and inertia** come from sheet stainless (7900 kg/m^3) laid on the render surface and integrated triangle by triangle, so the result is the full inertia tensor.
  - Side walls are 0.7 mm. The floor is an encapsulated 3.0 mm base plate.
  - The 3.0 mm base is a modelling choice (an encapsulated bottom), not taken from a product reference. The delivered CAD measures 1.5 mm at the base and 1.1-1.8 mm in the walls. The base is 0.152 kg, 27 % of the mass, and it is what holds the centre of mass at 41 % of the height.
  - The tilt and drop results below hold because of this choice. With a 0.7 mm single-sheet base (a re-run in the review of internal PR 343, base mass scaled by 0.7/3), the pitcher falls in one 20 deg tilted drop, away from the handle (toward -X), and its static tip angle drops to 19.5 deg.
  - The rolled rim is 52 g/m, a 3 mm bead of 0.7 mm sheet, along the rim faces.
  - Handles are hollow tubular or pressed sections at 0.9 mm, and mounts are 1.0 mm.
  - A part the CAD models as a wall with two faces counts half its surface area. A part the CAD models as a solid counts its visible surface once, as one skin.
  - Not weighed. A hand estimate (cylinder shell + base disc + handle surface + rim, from measured dimensions) agrees within 5 %.
  - Newton loads exactly the authored mass, centre of mass and principal inertia. The pressed foot is 0.8 mm.
  - **Total 0.562 kg:** walls 0.263, base 0.152, rim 0.020, foot 0.063, handle 0.064 (hollow pressed scroll; 11 % of the mass).
  - Centre of mass (0.0082, 0, 0.0784) m: 8 mm toward the handle, 41 % of the height.
  - Principal inertia (0.00140, 0.00217, 0.00261) kg m^2.
  - Tip-over angle 26.4 deg: geometric, base only. It is where the centre of mass passes the edge of the foot ring's colliders, and it does not count the handle. Simulated static releases on Newton main (sidecar `newton_smoke`) fall at 25.9-35.4 deg; the weakest directions are toward -X+Y in Newton contacts (25.9 deg, 32.9 in MuJoCo contacts) and toward the handle (26.6-26.9 deg).
- **Physics runtime variant `Newton`** (author_physics_variants.py): `Disabled` by default. There is no PhysX payload (internal issue 313).
- **Grasp line `grasp_identifier_01`** from (0, -0.0805, 0.08) to (0, 0.0805, 0.08) m: across Y through the vessel axis at the widest part of the belly. It is pinned in `grasp_line_overrides.json` and has not been simulated.
- **SimReady:** Robotics-Prop 3.1.0 passes its full scope with simready-validate 2026.7.1, including `FET_003_NEWTON` and `FET_005`.
- **Newton smoke** (sidecar `newton_smoke`; the maintainers' `smoke_newton.py`, standalone, dt 1/480, an internal simulation image): Newton main 9c28c80c with MuJoCo 3.15, 2026-10-22. The cases are upright 2 cm drops, 2 cm drops leaning 10 and 20 deg toward eight directions, static releases at 10 and 15 deg toward the same eight, and a bisection of the static tip angle per direction.
  - Newton and MuJoCo contacts: 0 falls. Upright drops, all 16 tilted drops and all 16 static releases recover.
  - All three jugs and `prop_jug_1l` show lower Newton-contact than MuJoCo-contact tip angles toward the -X diagonals on Newton main (`prop_jug_1l` 24.8 against 30.1 deg); `ikea_idealisk_jug_1l` shows it toward +X-Y (29.4 against 34.3 deg). Tracked in internal issue 351.
  - Nothing goes NaN. No contact stiffness or damping is authored.
  - `ikea_idealisk_jug_1l` and `prop_jug_1l`: 0 falls in either contact mode in the same harness.
  - A 7 mm capsule finger through the handle loop lifted the pitcher 7.4 cm.
  - Newton 1.6.1 (secondary record, same harness and package): 0 falls in either contact mode; hook lift 7.7 cm. The 2026-10-21 run about three axes also had 0 falls.

Left as delivered, deliberately: the render meshes and the friction values.
