# lg_fx25_leveling_leg

Newest first. A version with no entry here is not a version.

## 2026-10-30 — 2.0.0

LG asked that no part number stay in the assets we use. LG's CAD export names every part `T_<Teamcenter id>_<part number>_<Description>_3DAsset__refset_MODEL[_n]`, and nests connector housings as `LIB_<library>_<maker's part number>_<Description>`. Every prim whose name carried one is renamed to its description and instance numbering (`Screw_Customized_1_005`, `Housing_001`); siblings that would then collide take `_2`, `_3`. MAJOR: one prim name changes, under `Leg/Geometry/Leg_LS` (`LegLS_Leg_Assembly`).

- `userProperties:blender:object_name`, the exporter's copy of the CAD name, is removed from the one prim that carried it.
- Part numbers dropped from the docs of the five collider hulls; the leg's `lge` customData loses its `partNumber` key.
- Nothing else changes. The composed stage under every physics selection (none, Newton, PhysX) compares equal to 1.0.0 prim by prim, in order, apart from the names: types, applied schemas, every attribute value (points, masses, inertia, collider geometry and approximation, materials), relationship targets and prim metadata (31 prims, 5 colliders, one rigid body).
- SimReady re-validated: Robotics-Prop 3.1.0 with simready-validate 2026.7.1, full scope fails on GSP.001 only, as before (allowlisted); every feature outcome is the same as before.
- Kept: the package name and the LG model code in it, and every non-CAD prim name (task frames, colliders the rig authored, looks).
- The part numbers are also taken out of this changelog's earlier entries.

## 2026-10-14 — 1.0.0

First pooled: the LG FX25 leveling leg (M10 x 1.25 stud, lock nut and rubber pad) of N.Light's DB-M4
delivery (`DBM4_beauty_20261014.zip`, member `DBM4_newton_beauty/DBM4_LevelingLeg.usd`; sha256 of both in the sidecar).
N.Light's README calls the delivery an appearance pass over their Newton-validated set: render materials changed,
geometry, colliders, masses and physics materials did not.

- One dynamic rigid body `Leg`, 0.0696 kg (N.Light's estimate from the mesh volumes at steel 7850 and rubber 1200 kg/m³;
  the part ticket gives no mass). Its origin is the stud tip on the stud axis, stud pointing down (-Z), pad on top
  (z 0.0518 to 0.0585); its customData names the grasp point 57.3 mm up the axis and the thread (`M10 x 1.25 RH`).
- **`Leg` moved to the root origin.** The delivery placed the body at the cabinet's `Frames/Leg_Start` (-0.323, -0.310,
  0.0941), so that both files referenced at one placement line up. Isaac Lab spawns a rigid object's root prim at its
  initial state and writes that same state to the body at a reset (`AssetBase` spawn translation, Newton
  `RigidObject` default root pose), so a body offset under the root would move by that offset at the first reset.
  Nothing under `Leg` moved.
- Colliders under `Leg/Collision`, each binding its own physics material: `StudThread`, the M10 x 1.25 external thread
  from the tip to the hex (47 mm, ISO profile as the CAD), `ThreadSteel`; convex hulls `Stud_Hex` and `LockNut`
  (`Steel`) and `Pad` (`Rubber`). `Stud_Shaft`, the plain shaft hull the thread replaces, stays deactivated as delivered.
- Physics runtime variants `PhysX` and `Newton` on the default prim (SimReady RV.001-RV.011), `Disabled` by default
  (`author_physics_variants.py`). The PhysX payload holds what the delivery authored for PhysX: solver iterations 16/4,
  contact offset 0.1 mm, rest offset 0, the thread's `sdf` approximation at SDF resolution 1024. The Newton payload holds
  what it authored for Newton: `NewtonSDFCollisionAPI` on the thread (maximum resolution 1024, narrow band ±1 mm),
  `newton:contactGap` 0.1 mm and `newton:contactMargin` 0 on every collider, `newton:contactStiffness` 1e7 and
  `newton:contactDamping` 1e4 on the three physics materials; it adds `NewtonCollisionAPI` on every collider,
  `NewtonMeshCollisionAPI` on the hulls, and Newton's torsional (0.005) and rolling (0.0001) friction defaults on the
  materials.
- Dropped: `mjc:solimp` (0.99, 0.99, 1e-5, 0.5, 2) on `LockNut`, the hard lock-nut seat. The pool keeps no MuJoCo data
  (`author_physics_variants.py`); it acts only when the lock nut reaches the bracket boss at the nut runner's seating
  (see `lg_fx25_base_cabinet`).
- Area-weighted vertex normals authored on the 5 collider meshes (guide purpose, invisible), which had none
  (VG.027-VG.029); nothing rendered changes.
- `materials/mdl/lg_gltf.mdl` and `pbr.mdl` copied byte for byte (sha256 in the sidecar).
- Checked with Newton 1.6.1's importer against the delivered file: with `Newton` enabled, the same 7 shapes with
  identical type, shape transform, scale, SDF resolution and band, ke, kd, friction, restitution, gap and margin and the
  same body mass; the lock nut loads the default solimp.
- SimReady, recorded in the sidecar: Robotics-Prop 3.1.0 with simready-validate 2026.7.1, full scope **fails on GSP.001
  only** (no grasp line; listed in `scripts/tools/assets_lint_allowlist.json`). Core, Minimal, RBD physics, MDL
  materials, `FET_003_PHYSX` and `FET_003_NEWTON` pass.
