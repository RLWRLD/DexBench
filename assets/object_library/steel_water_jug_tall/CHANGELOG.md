# steel_water_jug_tall

Newest first. A version with no entry here is not a version.

## 2026-11-08 — 1.1.1

- Credit NdotLight (Trinix), CC BY 4.0; modified by RLWRLD. PATCH for attribution metadata only; source delivery identifiers, payloads, physics and certificates are unchanged.
- Align the grasp-line note with the current `steel_jug_1l` control package name; line geometry and historical benchmark records are unchanged.

## 2026-10-26 — 1.1.0

Fluid only; the rigid body, colliders, mass and looks are unchanged. The MPM water sidecar (`dexbench_arena/fluids/water_pour_setup.py`) finds both files beside the USD by name.

- `usd/steel_water_jug_tall_profile.npz`: interior lathe profile from the render vessel (`Geom/vessel_shell`; 61 levels, 96 rays per level), continued above the lowest full ring (21.6 cm) up to the lip on the pour side (-X) at 27.6 cm, 155 rows. Floor 3.6 mm. It seeds the water and gives the rim and floor heights; the shell above holds the water.
- `usd/steel_water_jug_tall_water_shell.npz`: the MPM water collider, baked by `scripts/tools/bake_water_vessel_shell.py` from `Geom/vessel_shell`. The faces that line the cavity are kept as they are and closed by a 16 mm wall behind them (two 8 mm voxels; the delivered sheet is thinner than the MPM contact band and fast water tunnels through it), the floor is lifted 8 mm and the file carries a contact margin of one voxel, so the slanted rim and the guard notch are what the water sees (a 2-voxel margin closes the 2.8 cm notch). 178 verts, 352 tris.
- `usd/steel_water_jug_tall_water_lid.npz`: the half top guard (`Geom/top_guard`) as a separate collider: its top surface at the real height, extruded 16 mm down, notch kept, contact margin 0. 66 verts, 128 tris.

## 2026-10-21 — 1.0.0

First pooled: `Waterjug.usdz` of Trinix's 2026-10-20 delivery (cad2sim 0.3.0 from a STEP file; archive address, file and member sha256 in the sidecar). The delivery itself stays outside the tree.

- **Frame and hierarchy.** Default prim `/steel_water_jug_tall` (kind `component`) is the rigid body, with no transform op. The render meshes sit under `Geom/` with their transforms baked in, and the jug is turned +90 deg about Z so the tube handle faces +X, as on `ikea_idealisk_jug_1l`. The base centre stays at the origin. Size 0.146 x 0.100 x 0.276 m. The delivery's `/World/Untitled_Project/Jug_Body` nesting, its `PhysicsScene` and its grasp curve are gone.
- **Look:** brushed stainless in place of the delivered polished chrome, with UsdPreviewSurface as on `ikea_idealisk_jug_1l`. Metallic is 1 and base colour (0.56, 0.57, 0.58).
  - `Looks/steel_brushed` (vessel walls): roughness from `textures/brushed_steel_roughness.png` (0.35 ± 0.05, streaked) and a fine streak normal map, `textures/brushed_steel_normal.png`. Both are procedural, 1024 px, and tile in both directions.
  - `Looks/steel_satin` (handle parts): a flat roughness of 0.35.
  - The delivery's unwrap UVs give the streaks no direction, so the walls' `st` is re-projected cylindrically round the jug axis (4 tiles round, 8 cm per tile up). Floors and rims are projected planar. Points, normals and topology are as delivered. The delivered glossy normal map is not used: it read as polish, not grain.
- **Colliders, `Collisions/` (guide purpose), 28 pieces: 26 CoACD hulls, the handle cylinder and a base disc.** CoACD 1.0.14 was run on each render part (vessel shell threshold 0.03, top guard and mounts 0.05, mounts capped at 2 pieces, top guard at 3, at most 64 vertices per hull). The tube handle is a `Cylinder` refitted to the render tube (r 8.0 mm, h 0.168 m).
  - Bore radius 48.0 mm, the same as the render's 48.0.
  - Handle loop clearance 30.2 mm against the render's 30.5.
  - Fit to the render surface: p95 1.1 mm, max 3.8 mm.
  - The `base_disc` hull (the render vessel's points within 1.5 mm of the floor: the flat support ring at r 48.0 mm and the fillet above it, 1.13 mm thick) gives the jug one flat face to stand on. CoACD's base hull, `vessel_shell_00`, reaches the visible base edge (r 49 mm) but its bottom is faceted 0-0.35 mm off flat. Newton main (9c28c80c) builds a multi-point contact only from coplanar support samples, so on it that hull touched the floor at a single point: standing upright the jug never came to rest and sat 0.6 mm in the floor. With the disc it settles (0.1 mm). Newton 1.6.1 did not check planarity. The large pitcher has had the same disc from the start.
  - Every collider binds `PhysicsMaterials/steel` on its own prim: friction 0.6 / 0.48 as delivered, restitution 0.1 (the pool's stainless value) instead of 0.3.
- **Mass, centre of mass and inertia** come from sheet stainless (7900 kg/m^3) laid on the render surface and integrated triangle by triangle, so the result is the full inertia tensor.
  - Side walls are 0.7 mm. The floor is an encapsulated 3.0 mm base plate.
  - The 3.0 mm base is a modelling choice (an encapsulated bottom), not taken from a product reference. The delivered CAD measures 3.0 mm at the base and 1.8 mm in the walls. The base is 0.175 kg, 24 % of the mass, and it is what holds the centre of mass at 36 % of the height.
  - The tilt and drop results below hold because of this choice. With a 0.7 mm single-sheet base (a re-run in the review of internal PR 343, base mass scaled by 0.7/3), the jug falls in one 20 deg tilted drop, about the XY diagonal, and its static tip angle drops to 21.1 deg.
  - The rolled rim is 52 g/m, a 3 mm bead of 0.7 mm sheet, along the rim faces.
  - Handles are hollow tubular or pressed sections at 0.9 mm, and mounts are 1.0 mm.
  - A part the CAD models as a wall with two faces counts half its surface area. A part the CAD models as a solid counts its visible surface once, as one skin.
  - Not weighed. A hand estimate (cylinder shell + base disc + handle surface + rim, from measured dimensions) agrees within 5 %.
  - Newton loads exactly the authored mass, centre of mass and principal inertia.
  - **Total 0.719 kg:** walls 0.416, base 0.175, rim 0.018, top guard 0.022, tube handle 0.055, mounts 2 x 0.016.
  - Centre of mass (0.0072, 0, 0.0985) m: 7 mm toward the handle, 36 % of the height.
  - Principal inertia (0.00178, 0.00597, 0.00647) kg m^2.
  - Tip-over angle 22.7 deg toward the handle: geometric, base only. It is where the centre of mass passes the edge of the base colliders, and it does not count the handle. By geometry the handle cylinder's lower end touches the floor at about 16 deg of tilt toward the handle, before the centre of mass passes the base edge. Simulated static releases on Newton main (sidecar `newton_smoke`) fall at 46 deg toward the handle (41 deg in MuJoCo contacts) and 26 deg sideways. The weakest direction is the diagonal away from the handle, 18.9 deg toward -X+Y in Newton contacts against 27.7 deg in MuJoCo contacts.
- **Physics runtime variant `Newton`** (author_physics_variants.py): `Disabled` by default. There is no PhysX payload (internal issue 313).
- **Grasp line `grasp_identifier_01`** from (0, -0.0698, 0.12) to (0, 0.0698, 0.12) m: across Y through the vessel axis below the top guard. It is pinned in `grasp_line_overrides.json` and has not been simulated.
- **SimReady:** Robotics-Prop 3.1.0 passes its full scope with simready-validate 2026.7.1, including `FET_003_NEWTON` and `FET_005`.
- **Newton smoke** (sidecar `newton_smoke`; the maintainers' `smoke_newton.py`, standalone, dt 1/480, an internal simulation image): Newton main 9c28c80c with MuJoCo 3.15, 2026-10-22. The cases are upright 2 cm drops, 2 cm drops leaning 10 and 20 deg toward eight directions, static releases at 10 and 15 deg toward the same eight, and a bisection of the static tip angle per direction.
  - **Newton contacts: 2 of 16 tilted drops fall**, the 20 deg drops toward -X+Y and -X-Y. Everything else recovers: upright drops (0.11 mm into the floor, at rest), the other 14 tilted drops and all 16 static releases.
  - The fall: the jug lands on the far edge, swings back and slaps its base down at 0.28 s. It keeps 2.27 rad/s 50 ms after the slap, against 1.89 in MuJoCo contacts and 2.10 on Newton 1.6.1, and passes the 24.5 deg far-side tip angle (toward +X-Y) at 0.58 s. Contact stiffness and damping (40000 / 400), the contact gap (1 cm to 0.5 mm), the handle collider and the lowest wall hulls do not change it. Open in internal issue 351.
  - MuJoCo contacts: 0 falls.
  - All three jugs and `prop_jug_1l` show lower Newton-contact than MuJoCo-contact tip angles toward the -X diagonals on Newton main (`prop_jug_1l` 24.8 against 30.1 deg); `ikea_idealisk_jug_1l` shows it toward +X-Y (29.4 against 34.3 deg). Tracked in internal issue 351.
  - Nothing goes NaN. No contact stiffness or damping is authored.
  - `ikea_idealisk_jug_1l` and `prop_jug_1l`: 0 falls in either contact mode in the same harness.
  - A 7 mm capsule finger through the handle loop lifted the jug 9.5 cm (8.9-13.0 cm while it swung during the hold).
  - Newton 1.6.1 (secondary record, same harness and package): 0 falls in either contact mode; hook lift 9.2 cm. The 2026-10-21 run, about three axes and without the base disc, also had 0 falls.

Left as delivered, deliberately:
- The render meshes. Brushing comes from the material, not from geometry.
- The friction values. They are a ballpark, as ours are.
- The half top guard. It is the product's shape.
