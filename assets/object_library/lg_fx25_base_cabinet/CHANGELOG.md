# lg_fx25_base_cabinet

Newest first. A version with no entry here is not a version.

## 2026-10-30 — 2.0.0

LG asked that no part number stay in the assets we use. LG's CAD export names every part `T_<Teamcenter id>_<part number>_<Description>_3DAsset__refset_MODEL[_n]`, and nests connector housings as `LIB_<library>_<maker's part number>_<Description>`. Every prim whose name carried one is renamed to its description and instance numbering (`Screw_Customized_1_005`, `Housing_001`); siblings that would then collide take `_2`, `_3`. MAJOR: 7 prim names change under `Body/Geometry/BaseRig_inverted` (the cabinet, the four brackets, the two installed legs).

- `userProperties:blender:object_name`, the exporter's copy of the CAD name, is removed from the 7 prims that carried it.
- Part numbers dropped from the prim docs of the 20 collider pieces that named a part, and the `lge.basis` customData of the cabinet's physics material.
- Nothing else changes. The composed stage under every physics selection (none, Newton, PhysX) compares equal to 1.0.0 prim by prim, in order, apart from the names: types, applied schemas, every attribute value (points, masses, inertia, collider geometry and approximation, materials), relationship targets and prim metadata (82 prims, 15 colliders, one rigid body).
- SimReady re-validated: Robotics-Prop 3.1.0 with simready-validate 2026.7.1, static scope passes; every feature outcome is the same as before.
- Kept: the package name and the LG model code in it, and every non-CAD prim name (task frames, colliders the rig authored, looks).
- The part numbers are also taken out of this changelog's earlier entries.

## 2026-10-14 — 1.0.0

First pooled: the LG FX25 base cabinet of N.Light's DB-M4 delivery (`DBM4_beauty_20261014.zip`, member
`DBM4_newton_beauty/DBM4_BaseCabinet.usd`; sha256 of both in the sidecar), upside down as the leveling-leg station has it,
with its four leg brackets and the two legs already installed at the +y corners. N.Light's README calls the
delivery an appearance pass over their Newton-validated set: render materials changed, geometry, colliders, masses and
physics materials did not.

- Root `DBM4_BaseCabinet` (N.Light's name, kept) at the centre of the cabinet's footprint on its bottom face, which faces
  up; metres, Z up. The cabinet occupies x ±0.341, y ±0.328, z 0 to 0.086.
- Frames as delivered: `Frames/Hole_0..3`, each bracket's hole axis on the bracket top face (z 0.0741), `Hole_1` at
  (-0.323, -0.310) being the hole the leg goes into; `Frames/Leg_Start`, `Leg_Entry`, `Leg_Seated`, the leg's origin (its
  stud tip) 20 mm above that hole's mouth, at the mouth and fully fastened (z 0.0311). `Leg_Seated` carries N.Light's
  thread data in its customData (`pitch_m` 0.00125, `gateTilt_deg` 2, `gateOffset_m` 0.0006, the run-down and seating
  torques).
- Colliders, all under `Body/Collision`, each binding its own physics material: the cabinet and the four brackets as
  triangle meshes with the hole bosses carved out (`Plastic`, `Steel`), the two installed legs as convex hulls (`Steel`),
  and on the two -y holes (`HoleThreads/Thread_0`, `Thread_1`) an M10 x 1.25 internal thread with 0.15 mm radial
  clearance and its lock-nut seat (`Seat_0`, `Seat_1`), bound to `ThreadSteel`. The +y holes carry no thread: their legs
  are part of the cabinet.
- `Body` is a kinematic rigid body (`PhysicsRigidBodyAPI`, `physics:kinematicEnabled`). The delivery left it static
  colliders without a body, because "a kinematic body makes Isaac re-sync every CAD prim under it each step". DexBench
  selects a package's physics runtime variant only on rigid and articulated objects (`physics_variants.py`), and the
  thread SDFs, the contact stiffness and the contact gap are in the Newton payload: a static cabinet would load its threads
  as plain triangles at Newton's default stiffness. With N.Light's leg turned at 0.5 turn/s for two turns in Newton 1.6.1
  (120 Hz collision, 20 substeps at 60 Hz) the cabinet static and the cabinet kinematic gave 1.2547 and 1.2539 mm per
  turn and 10.598 and 10.606 mm below the hole mouth.
- Physics runtime variants `PhysX` and `Newton` on the default prim (SimReady RV.001-RV.011), `Disabled` by default
  (`author_physics_variants.py`). The PhysX payload holds what the delivery authored for PhysX: contact offset 0.1 mm,
  rest offset 0, the threads' `sdf` approximation at SDF resolution 1024. The Newton payload holds what it authored for
  Newton: `NewtonSDFCollisionAPI` on the threads (maximum resolution 1024, narrow band ±1 mm), `newton:contactGap`
  0.1 mm and `newton:contactMargin` 0 on every collider, `newton:contactStiffness` 1e7 and `newton:contactDamping` 1e4
  on the four physics materials; it adds `NewtonCollisionAPI` on every collider, `NewtonMeshCollisionAPI` on every
  collider except the two threads (beside their SDF API Newton warns and ignores it), and Newton's torsional (0.005) and
  rolling (0.0001) friction defaults on the materials.
- Dropped: `mjc:solimp` (0.99, 0.99, 1e-5, 0.5, 2) on `Seat_0` and `Seat_1`, the hard lock-nut seat. The pool keeps no
  MuJoCo data (`author_physics_variants.py`). Newton 1.6.1 reads it only after `SolverMuJoCo.register_custom_attributes`
  has run on the builder, and it acts only once the lock nut reaches the boss, about 28 turns past where the thread
  catches, at the nut runner's seating; a run-down task has to bring it back in the Newton payload.
- Area-weighted vertex normals authored on the 15 collider meshes (guide purpose, invisible), which had none
  (VG.027-VG.029); nothing rendered changes.
- `materials/mdl/lg_gltf.mdl` and `pbr.mdl` copied byte for byte (sha256 in the sidecar); `lg_gltf.mdl` imports
  `pbr.mdl` from its own folder.
- Checked with Newton 1.6.1's importer against the delivered file: with `Newton` enabled, the same 26 shapes with
  identical type, transform, scale, SDF resolution and band, ke, kd, friction, restitution, gap and margin; they hang on
  the kinematic body instead of the world, and the two seats load the default solimp. With every engine disabled the
  threads load without an SDF, at ke 2500, kd 100 and a 0.1 m gap.
- SimReady, recorded in the sidecar: Robotics-Prop 3.1.0 with simready-validate 2026.7.1, the static scope passes;
  `FET_003_PHYSX` and `FET_003_NEWTON` pass.
