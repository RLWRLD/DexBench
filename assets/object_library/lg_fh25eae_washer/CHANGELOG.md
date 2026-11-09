# lg_fh25eae_washer

Newest first. A version with no entry here is not a version.

## 2026-10-30 — 3.0.0

LG asked that no part number stay in the assets we use. LG's CAD export names every part `T_<Teamcenter id>_<part number>_<Description>_3DAsset__refset_MODEL[_n]`, and nests connector housings as `LIB_<library>_<maker's part number>_<Description>`. Every prim whose name carried one is renamed to its description and instance numbering (`Screw_Customized_1_005`, `Housing_001`); siblings that would then collide take `_2`, `_3`. MAJOR: 679 prim names change under `Body_Visual/Geometry`.

- `userProperties:blender:object_name`, the exporter's copy of the CAD name, is removed from the 679 prims that carried it.
- Part numbers dropped from the body's prim doc.
- `visual_trim.json` names the new paths. Its REMOVE rows, whose meshes were deleted before, name each part by its description.
- Nothing else changes. The composed stage under every physics selection (none, Newton) compares equal to 2.0.0 prim by prim, in order, apart from the names: types, applied schemas, every attribute value (points, masses, inertia, collider geometry and approximation, materials), relationship targets and prim metadata (4,002 prims, 358 colliders, one rigid body).
- SimReady re-validated: Robotics-Prop 3.1.0 with simready-validate 2026.7.1, static scope passes; every feature outcome is the same as before.
- Kept: the package name and the LG model code in it, and every non-CAD prim name (task frames, colliders the rig authored, looks).

## 2026-10-21 — 2.0.0

The visual is trimmed to what a camera can see or a task collider can touch ("option 1"): every mesh visible from
outside the cabinet, every see-through or named part (`*Glass*`, `*Window*`, transmissive beauty glass), and every
hidden mesh within 10 mm of a slot or recess task collider is kept; every other hidden mesh is deleted. MAJOR: 1,192
prim paths under `Body_Visual/Geometry` go away.

- Meshes 2,495 -> 1,303 (the list, `visual_trim.json` at the package root, keyed by path under the washer assembly; 1,192 deleted, 33 of
  them already deactivated); triangles of the active visual 2,692,918 -> 2,280,202. The one mesh the list does not
  name (a drain-pump cap, `Mesh_4335`) is kept.
- The 922 Xforms left holding no mesh are deleted, and with them the 84 looks nothing binds any more: 7 beauty
  looks only deleted meshes used, and 77 of the 78 old CAD looks the beauty bindings orphaned in 1.1.0 (the two
  `Baked*` among them). `Color__000000ff_001` stays bound. All six textured looks stay.
- Every kept mesh is unchanged against 1.1.0 (points, indices, normals, UVs, bindings, transforms of every
  ancestor). Physics is unchanged against 1.0.2 and 1.1.0: every layer spec outside `Body_Visual` is identical, the
  composed stage with `Newton` disabled and enabled compares equal (358 colliders), and
  `runnables/physics/newton.usd` is byte-identical.
- SimReady re-validated: Robotics-Prop 3.1.0 with simready-validate 2026.7.1, static scope passes.

## 2026-10-21 — 1.1.0

The look of N.Light's DB-M1 beauty delivery of 2026-10-13, ported from archive `DBM1_Washer_beauty_20261013.zip`,
member `DBM1_Washer_beauty/DBM1_Washer_FH25EAE.usd` (logical address `deliveries/nlight/261013`; sha256 of both in
the sidecar's `visuals_from`). That file is the 10-01 washer with render materials and printed details; its points,
face indices and transforms are identical to this package's, checked prim by prim.

- `Body_Visual/Looks/Beauty`: 74 materials on one MDL module, `materials/mdl/lg_gltf.mdl`, a flat front end for
  NVIDIA's glTF `pbr.mdl` (BSD-3-Clause, shipped beside it). Six of them read a texture under `textures/`: the
  cabinet panel print, the door glass print, the display and knob-ring windows (emissive) and the power and start
  buttons. `panel_print.png` is downscaled from 4115x5749 to 1466x2048 (Lanczos); the others are 1280 px or less and
  ship as delivered.
- Every `material:binding` under `Body_Visual/Geometry` is the beauty file's: 2,642 retargeted to the new looks,
  4 added (with `MaterialBindingAPI`) on PCB screw meshes that stay deactivated here.
- The six decal meshes (power and control buttons, cabinet cover, panel windows) take the beauty file's
  `primvars:st`, vertex-interpolated, and lose their `primvars:st:indices`; the safety switch mesh takes its
  recomputed face-varying `normals` and `primvars:normals`.
- Kept as they were: the 76 old looks, including the two `Baked*` ones the beauty file dropped (nothing binds them
  now; removing them would move a prim path), the 37 meshes this package deactivates and the beauty file does not,
  and everything outside `Body_Visual`. Physics is unchanged: every layer spec outside `Body_Visual` is identical,
  the composed stage with `Newton` disabled and enabled compares equal prim by prim (358 colliders), and
  `runnables/physics/newton.usd` is byte-identical.
- SimReady re-validated: Robotics-Prop 3.1.0 with simready-validate 2026.7.1, static scope passes; the feature
  outcomes are unchanged.

## 2026-10-20 — 1.0.2

The sidecar's `source_archive.from` no longer names the shared folder the archive came through; the
archive's file name and sha256 still identify it. Nothing in the package changes.

## 2026-10-15 — 1.0.1

The sidecar's `derived_from.rig` no longer names the host and directory the rig was built in; the rig
file's sha256 still identifies it. Nothing in the package changes.

## 2026-10-13 — 1.0.0

First pooled: the LG FH25EAE washer of N.Light's DB-M1 delivery (`Assets_M1_M2_20261001.zip`, member
`M1/robot_assets/DBM1_Washer_FH25EAE.usd`; sha256 of both in the sidecar), with the colliders of the EA rig
(`proxy_rg_s3`). The rig's file is the source; its sha256 is in the sidecar.

- One kinematic rigid body `Body`, 112 kg as N.Light authored it. Root at the N.Light washer origin, metres, Z up.
  The visual (2.7 M triangles) is the delivery's, instanced from the `Body_Visual` class; 28 empty meshes stay
  deactivated (they crash Newton's importer).
- Task frames `Frames/DrawerPreInsert`, `Frames/DrawerRailEntry`, `Frames/DrawerSeated` as delivered: the drawer
  origin's pose at each stage, identity rotation, insertion along +y (seated at (-0.2264, -0.3877, 0.9092) m).
- Colliders, all on `Body` (the rig's): `CollisionProxy/*`, 13 shapes (`recess_{left,right,top,bottom}_{a,b}`, the
  slot mouth in two stages from the CAD; `body_right`, `body_under_slot`, `body_behind_slot`, `panel_knob`; the door
  cylinder `door`) binding `PhysicsMaterials/PaintedSteel` (static 0.40, dynamic 0.35, restitution 0.1), and
  `CollisionSlot/slot_000..344`, convex hulls of the dispenser housing with their fronts on the cover face (y 47.5 mm
  in the seat frame) binding `PhysicsMaterials/PP` (0.35/0.30/0.1). Every collider binds its material on its own prim.
- The 148 vendor colliders under `Body/Collision`, which the rig had switched off, are removed: Newton loaded them
  as non-colliding shapes (flags 0).
- The two `Baked*` looks referenced `../textures/FX25_HP_*.png`, which the delivery does not contain; their texture
  readers are removed and the surfaces take a flat dark grey (0.15). Only two drain-pump meshes inside the cabinet
  bind them.
- Physics runtime variant `Newton` on the default prim (SimReady RV.001-RV.011), `Disabled` by default; the package
  has no PhysX data. The payload annotates the colliders and both materials (`NewtonMaterialAPI`, torsional 0.005,
  rolling 0.0001) and carries the rig's contact softness, MuJoCo solref (0.005 s, damping ratio 1) as
  `newton:contactStiffness = 40000` and `newton:contactDamping = 400` on both materials.
- Of nine pairs of coincident identical vendor visual meshes (filter-cover screws, damper, filter magnets, hose
  connector; same geometry, transform and material as their twins) one each is deactivated in the `Body_Visual` class
  (VG.008); nothing rendered changes.
- SimReady, recorded in the sidecar: Robotics-Prop 3.1.0 with simready-validate 2026.7.1, static scope passes;
  `FET_003_NEWTON` and `FET_003_PHYSX` pass.
- Extents and area-weighted normals authored on the hulls (VG.002, VG.027-029); SimReady Core metadata stamped
  (NP.006/SR.001).
- Checked against the rig file with Newton 1.6.1's importer: the same 358 colliding shapes with identical type,
  transform, scale, hull, margin, gap and friction (0.30 on the slot hulls, 0.35 on the proxy shapes), the same body
  mass and inertia and kinematic flag, and ke/kd 40000/400 where the rig file loads Newton's defaults that the rig
  then overwrote.
