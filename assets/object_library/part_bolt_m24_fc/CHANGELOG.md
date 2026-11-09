# part_bolt_m24_fc

Newest first. A version with no entry here is not a version.

## 2026-10-27 — 1.3.0

- Newton collider choice `NewtonColliders`, nested under the `Newton` variant's `Enabled`: `pieces` (the default) composes the Newton payload alone, as before; `primitives` adds `usd/runnables/physics/newton_primitives.usd`, which brings in 2 collision pieces (cylinder, mesh) and switches off 2 of the base's 2 colliders (the ones the new layer does not keep). The pieces are dbclutter's `_prim5` collision primitives (at most 5 boxes, cylinders, capsules or spheres fitted to the visual mesh by db-m5-create-stable-clutter `scripts/fit_primitive_colliders.py`, a script not yet committed there, run on its 5f08c9a working tree). Between the choices the render mesh, the authored mass and the physics materials are the same; the package authors no centre of mass or inertia, so Newton derives both from the colliders and they differ between the choices; the recipe is in `runnables/physics/newton_primitives.preparation.json`. A consumer that does not select `primitives` composes what it did, and a Newton cert recorded for an earlier version still describes `pieces`.

## 2026-09-20 — 1.2.0

- Physics runtime variants `Newton` on the default prim (SimReady RV.001-RV.011), `Disabled` by default; `Enabled` composes `usd/runnables/physics/newton.usd`. The Newton payload annotates the colliders (`NewtonCollisionAPI`, `NewtonMeshCollisionAPI`) and the physics materials (`NewtonMaterialAPI`); contact gap and margin stay the engine's.

## 2026-09-16 — 1.1.5

- Newton 1.5.2 cert (standalone MuJoCo-Warp solver, not the Arena image): parse OK; ground_drop pass, slope_drop pass; grasp_and_lift PASS on `grasp_identifier_01` with the line riding on the body (line kept in the stage frame after the settle: fail, Lifting failed: object dropped; stock Newton contact defaults: fail). Certified package version 1.1.3 (this package has changed since; the cert has not been re-run on 1.1.4). Settings and report: the maintainers' Newton 1.5 grasp benchmark.

## 2026-09-16 — 1.1.4

- Free-space variant of NVIDIA's grasp-and-lift benchmark (asset floated, jaws closed at zero g, gravity restored before the lift; `outputs/freespace` in DexBench): `grasp_identifier_01` PASSED, lifted, held and shook. The floor-standing verdict above stands; what defeats it there is the floor under the pads, not the line.

## 2026-09-15 — 1.1.3

- Grasp line FAILED NVIDIA's FET005 grasp-and-lift benchmark on Isaac Sim 6.0.1 (PhysX): Across-body lines all lost the asset; only the lengthwise pinch (0.15 m gap) held.  Classified `review`: every across line failed but the lengthwise line (jaws at the two ends, 0.15 m apart) lifted, held and shook it: mass, friction and collider are fine; the across pads close above the body: it rests lower than its bbox mid-height (resting z below the bbox centre), so a line at 40-60 % of the height passes over it

## 2026-09-15 — 1.1.2

- Grasp line `grasp_identifier_01` re-placed from [0.0, -0.0408, 0.0208] to [0.0, 0.0408, 0.0208] m in the default prim's frame (across-y): both endpoints now sit 2 cm clear of the surface, because NVIDIA's grasp-and-lift benchmark builds its jaw pads at the endpoints and a pad inside the collider launches the asset.

## 2026-09-15 — 1.1.1

- Grasp line re-shaped as an `Xform "grasp_identifier_01"` holding the `BasisCurves "grasp_identifier_01_line"` (same two points): NVIDIA's runtime grasp-and-lift benchmark looks for the Xform and reads its child, the static GSP.001 rule for a curve of that name; the plain curve satisfied only the latter.

## 2026-09-15 — 1.1.0

- Grasp line `grasp_identifier_01` authored from [0.0, -0.0208, 0.0208] to [0.0, 0.0208, 0.0208] m in the default prim's frame (across-y) as the closing line of a parallel-jaw grasp (GSP.001). Chosen from a projection preview of the mesh; not yet run through the FET005 grasp-and-lift benchmark.

## 2026-09-14 — 1.0.1

- SimReady Core metadata (`SimReady_Metadata`, `asset_name`, `asset_type`, `source_file`, `usd_date_generated`) stamped into the root layer's customLayerData from the sidecar (NP.006/SR.001).

## 2026-09-15 — 1.0.0

Packaged as `object_library/part_bolt_m24_fc/usd/part_bolt_m24_fc.usd` from `object_library/items/industrial/part_bolt_m24_fc.usdc` (source: deliveries/psl/e3_manufacturing/260827). No content change; the version is the sidecar's.
