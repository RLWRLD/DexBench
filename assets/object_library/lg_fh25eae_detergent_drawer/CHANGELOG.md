# lg_fh25eae_detergent_drawer

Newest first. A version with no entry here is not a version.

## 2026-10-30 — 3.0.0

LG asked that no part number stay in the assets we use. LG's CAD export names every part `T_<Teamcenter id>_<part number>_<Description>_3DAsset__refset_MODEL[_n]`, and nests connector housings as `LIB_<library>_<maker's part number>_<Description>`. Every prim whose name carried one is renamed to its description and instance numbering (`Screw_Customized_1_005`, `Housing_001`); siblings that would then collide take `_2`, `_3`. MAJOR: 32 prim names change under `Body/Geometry`.

- `userProperties:blender:object_name`, the exporter's copy of the CAD name, is removed from the 32 prims that carried it.
- Part numbers dropped from the body's prim doc.
- `visual_trim.json` names the new paths. Its REMOVE rows, whose meshes were deleted before, name each part by its description.
- Nothing else changes. The composed stage under every physics selection (none, Newton, PhysX) compares equal to 2.0.0 prim by prim, in order, apart from the names: types, applied schemas, every attribute value (points, masses, inertia, collider geometry and approximation, materials), relationship targets and prim metadata (342 prims, 72 colliders, one rigid body).
- SimReady re-validated: Robotics-Prop 3.1.0 with simready-validate 2026.7.1, full scope fails on GSP.001 only, as before (allowlisted); every feature outcome is the same as before.
- Kept: the package name and the LG model code in it, and every non-CAD prim name (task frames, colliders the rig authored, looks).

## 2026-10-21 — 2.0.0

The five visual meshes never visible from 9,600 viewpoints all round the free drawer (0.25-1.5 m, 10 G rays) are
deleted: the two springs and two lock holders of the detergent and softener cases and one inner face of the softener
case (`Case_Lower ... Mesh_5047`). MAJOR: their prim paths go away.

- The list, with each mesh's reason, is `visual_trim.json` at the package root.
- Meshes 39 -> 34, triangles of the active visual 84,643 -> 82,771; the 9 Xforms left holding no mesh and the two
  looks only these meshes bound (`Color__0099ffff_001`, `Color__999999ff_001`) are deleted.
- Every kept mesh is unchanged. Physics is unchanged: mass, centre of mass and inertia are authored on `Body`
  (MassAPI), every layer spec outside `Body/Geometry` and `Looks` is identical, the composed stage compares equal
  under every `Newton`/`PhysX` selection (72 colliders), and both `runnables/physics/*.usd` are byte-identical.
- The drawer has no beauty delivery; its looks stay the 10-01 CAD colours.
- SimReady re-validated: Robotics-Prop 3.1.0 with simready-validate 2026.7.1, full scope fails on GSP.001 only, as
  before (allowlisted).

## 2026-10-20 — 1.0.2

The sidecar's `source_archive.from` no longer names the shared folder the archive came through; the
archive's file name and sha256 still identify it. Nothing in the package changes.

## 2026-10-15 — 1.0.1

The sidecar's `derived_from.rig` no longer names the host and directory the rig was built in; the rig
file's sha256 still identifies it. Nothing in the package changes.

## 2026-10-13 — 1.0.0

First pooled: the LG FH25EAE detergent drawer of N.Light's DB-M1 delivery (`Assets_M1_M2_20261001.zip`, member
`M1/robot_assets/DBM1_DetergentDrawer.usd`; sha256 of both in the sidecar), with the colliders of the EA rig
(`proxy_rg_s3`). The rig's file is the source; its sha256 is in the sidecar.

- One dynamic rigid body `Body`, mass 1.2101 kg, centre of mass (-0.0002, 0.2569, -0.0065) m and principal inertia as
  N.Light authored them. Origin at the centre of the front face, x lateral, y into the washer, z up.
- Colliders, all on `Body` (the rig's): `CollisionProxy/*`, 12 boxes traced from the visual CAD's front panel and
  handle pocket (`front_wall`, `hollow_fin`, `top_shell`, `pocket_back`, `pocket_floor_0..3`, `bottom_plate`,
  `front_lip_low`, `panel_left`, `panel_right`; the last two trimmed to the panel's back face at y 47 mm), and
  `CollisionTray/tray_00..59`, convex hulls fitted to the tray's outer surface. Each binds `PhysicsMaterials/PP`
  (static 0.35, dynamic 0.30, restitution 0.1, N.Light's values) on its own prim, so Newton loads it (it reads no
  ancestor binding).
- The 5,003 vendor collider pieces under `Body/Collision`, which the rig had switched off
  (`physics:collisionEnabled = false`), are removed rather than kept switched off: Newton loaded them as 5,003
  non-colliding shapes (flags 0) that only slowed resets. Nothing else in the body changed.
- The two `Baked*` looks referenced `../textures/FX25_HP_*.png`, which the delivery does not contain; their texture
  readers are removed and the surfaces take a flat dark grey (0.15). No mesh of the drawer binds them.
- Physics runtime variants `PhysX` and `Newton` on the default prim (SimReady RV.001-RV.011), `Disabled` by default.
  The PhysX payload holds the body's solver iteration counts (16/4). The Newton payload annotates the colliders
  (`NewtonCollisionAPI`, `NewtonMeshCollisionAPI`) and the material (`NewtonMaterialAPI`: torsional 0.005, rolling
  0.0001, Newton's defaults) and carries the rig's contact softness, MuJoCo solref (0.005 s, damping ratio 1) as
  `newton:contactStiffness = 40000` and `newton:contactDamping = 400` (kd = 2/0.005, ke = (kd/2)^2). Newton 1.6.1
  reads those from the bound material into `Model.shape_material_ke/kd`, which the rig wrote after loading.
- One of two coincident identical vendor visual meshes (`Case_Upper ... Mesh_5061/Mesh_5067`, same geometry, transform
  and material as its twin) deactivated (VG.008); nothing rendered changes.
- SimReady, recorded in the sidecar: Robotics-Prop 3.1.0 with simready-validate 2026.7.1, full scope **fails on
  GSP.001 only** (no grasp line: the drawer is taken by a two-hand dexterous grasp, not a parallel-jaw line; listed in
  `scripts/tools/assets_lint_allowlist.json`). Core, Minimal, RBD physics, MDL materials, `FET_003_PHYSX` and
  `FET_003_NEWTON` pass.
- Extents and area-weighted normals authored on the 60 hulls (VG.002, VG.027-029); SimReady Core metadata stamped
  (NP.006/SR.001).
- Checked against the rig file with Newton 1.6.1's importer (Isaac Lab's Newton and PhysX resolvers): the same 72
  colliding shapes with identical type, transform, scale, hull, margin, gap and friction, the same body mass,
  centre of mass and inertia, and ke/kd 40000/400 where the rig file loads Newton's defaults (2500/100) that the rig
  then overwrote.
