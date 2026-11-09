# lg_mnext3_liner_connector_task

Newest first. A version with no entry here is not a version.

## 2026-10-30 — 2.0.0

LG asked that no part number stay in the assets we use. LG's CAD export names every part `T_<Teamcenter id>_<part number>_<Description>_3DAsset__refset_MODEL[_n]`, and nests connector housings as `LIB_<library>_<maker's part number>_<Description>`. Every prim whose name carried one is renamed to its description and instance numbering (`Screw_Customized_1_005`, `Housing_001`); siblings that would then collide take `_2`, `_3`. MAJOR: 239 prim names change under `Liner_Visual/Geometry` and `Liner/Collision`, in the base layer and the Newton payload.

- Part numbers dropped from the docs of the seven mate colliders, which named the mates' housings.
- `visual_trim.json` names the new paths. Its REMOVE rows, whose meshes were deleted before, name each part by its description.
- Nothing else changes. The composed stage under every physics selection (none, Newton) compares equal to 1.0.0 prim by prim, in order, apart from the names: types, applied schemas, every attribute value (points, masses, inertia, collider geometry and approximation, materials), relationship targets and prim metadata (3,748 prims, 2,015 colliders, 42 rigid bodies, 39 joints).
- SimReady re-validated: Robotics-Prop 3.1.0 with simready-validate 2026.7.1, static scope passes; every feature outcome is the same as before.
- Kept: the package name and the LG model code in it, and every non-CAD prim name (task frames, colliders the rig authored, looks).
- The part numbers are also taken out of this changelog's earlier entries.

## 2026-10-21 — 1.0.0

First pooled: the LG M-Next3 refrigerator liner (Liner_2, pre-foam) with the harness-connector pocket from N.Light's
DB-M2 beauty delivery of 2026-10-13: archive `DBM2_beauty_20261013.zip`, member `DBM2_beauty/DBM2_Liner2_ConnectorTask.usd`
(sha256 of both in the sidecar), logical address `deliveries/nlight/261013` in the outside archive. Its physics
before the changes below is the 10-01 delivery's (`Assets_M1_M2_20261001.zip`, member
`M2/robot_assets/DBM2_Liner2_ConnectorTask.usd`, compared prim by prim); the beauty file adds render materials.
Metres, Z up, default prim `/DBM2_Liner2`. DB-M2 inserts connector S2.

- Bodies as N.Light authored them: the kinematic `Liner`, the dynamic plugs `Task/Connector_S1` (15.3 g)
  and `Task/Connector_S2` (17.4 g) with 5 convex wall colliders each, the `Mate_S*_Front/Rear` pockets on
  the liner, two wires as capsule chains (S1 19 links, S2 20: 39 bodies and 39 joints, D6 drives, no collision), task frames `Frames/S1_Seated`,
  `S1_Unplugged`, `S2_Seated`, `S2_Unplugged`. The file starts seated.
- Colliders: the liner's 28 triangle-mesh colliders are replaced by 2,001 convex pieces (CoACD), each mesh's prim
  becoming an Xform of the same name and transform that holds its pieces. Newton could not run the triangle meshes
  (with MuJoCo contacts each mesh becomes its convex hull, which swallowed the plugs; the 242k-face liner shell is the
  largest of them). The bracket and the liner patch around the pocket are fitted to 0.7 mm (p95) / 1.1 mm (max); 14
  bracket pieces that reached into a plug, seated or along a 22 mm pull-out, are pulled back by their depth + 0.1 mm,
  3 too small for that and 13 thinner than 0.1 mm are dropped, so no piece intersects either plug. The liner shell
  (365 pieces) fits to 9.7 mm (p95); the 25 parts 370 mm or more from the plugs (834 pieces) to 1.2-15.9 mm (p95).
  The four mates and the plug walls are N.Light's convex hulls, unchanged.
- Every collider binds its physics material on its own prim: 38 of the delivered 42 inherited it from their
  `Collision` scope, which Newton does not read (it loaded them at mu 1.0). The values are the inherited ones:
  LinerABS static 0.35 / dynamic 0.30 / restitution 0.1, PBT 0.30 / 0.25 / 0.1.
- Wires: each plug is an articulation root (the chain plug -> `Link_00` ... `Link_18`, or `Link_19` for S2, was a set of orphan joints to
  Newton), and the two `AtTape` fixed joints that closed each chain to the liner are deleted. As a loop closure Newton
  1.6.1 turns the whole chain NaN on the first step once the asset sits away from the world origin; without it each
  wire hangs from its plug, its far end free, which moves the plug by less than 0.1 mm.
- Physics runtime variant `Newton` only (internal issue 313), `Disabled` by default: it annotates the colliders and both materials
  and carries the M1 assets' contact softness, MuJoCo solref (0.005 s, damping ratio 1) as
  `newton:contactStiffness = 40000` and `newton:contactDamping = 400`. With MuJoCo contacts and no multi-CCD, S2 rests
  3.5 mm short of its seat without it. The PhysX attributes N.Light authored (contact and rest offsets, hull vertex
  limits, solver iterations, `disableGravity` on the wire links) are not carried.
- Visual trimmed to what a camera can see or a task collider can touch (`visual_trim.json`): of 605 visual meshes,
  14 hidden ones more than 10 mm from every task collider, mate and frame are deleted with the 17 Xforms they leave
  empty, and 8 coincident duplicates of a kept mesh (VG.008); 583 meshes, 522,561 triangles. No empty mesh.
- Render looks: `Looks/Beauty`, 35 materials on `materials/mdl/lg_gltf.mdl` (with NVIDIA's `pbr.mdl`, BSD-3-Clause,
  beside it). No textures.
- Checked in Newton 1.6.1 (mujoco_warp, the DB-M1 env's settings: 480 Hz, 100 iterations, elliptic cone, 1 cm gap),
  2 s settle then S2 pulled 17.7 mm out along its pull-out axis by a spring handle and pushed back, then released.
  Newton contacts: S1 0.26 mm and S2 0.34 mm from their seated frames, S2 back to 0.34 mm. MuJoCo contacts: S1 0.29,
  S2 0.48, S2 back to 0.48 mm; with multi-CCD 0.33 mm. The same with the asset moved 2.5 m from the origin. Finite
  throughout.
- SimReady housekeeping: extents and normals authored where missing (VG.002, VG.027), the wire capsules and guide
  colliders bound to a display-colour preview surface (VM.MAT.001), Core metadata stamped (NP.006/SR.001).
- SimReady, recorded in the sidecar: Robotics-Prop 3.1.0 with simready-validate 2026.7.1, static scope passes (Core,
  Minimal, MDL materials). The optional rigid-body features fail RB.007 only: the kinematic liner carries no mass, as
  delivered; it never moves, so none is invented here.
