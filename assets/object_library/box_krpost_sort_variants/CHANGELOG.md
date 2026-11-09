# box_krpost_sort_variants

Newest first. A version with no entry here is not a version.

## 2026-10-30 — 1.1.0

- Physics runtime variant `Newton` on each variant layer's default prim (SimReady RV.004-RV.006), `Disabled` by default; `Enabled` composes its box number's `over` in `usd/runnables/physics/newton.usd`: the convex piece switched off and annotated (`NewtonCollisionAPI`, `NewtonMeshCollisionAPI`), and one analytic box collider of the box's size (`collision_box`, a guide-purpose Cube standing on the origin) bound directly (physics purpose) to its own `CollisionBoxMaterial` (`NewtonMaterialAPI`): friction 1.0, restitution 0, torsional 0.005, rolling 0.0001, Newton's shape defaults, at which DB-L1 loaded the same collider when it added it unbound at spawn. Geometry, labels and the base physics unchanged. Added by scripts/tools/make_polybag_sort_variants.py --box-newton-runtime; a variant set is added, hence the minor.
- SimReady validated for this version with simready-validate 2026.7.1 on 2026-10-30: Robotics-Prop 3.1.0 **pass** (scope full); `FET_003_NEWTON` and `FET_000_NEWTON` pass (unmet in 1.0.0), `FET_003_PHYSX` pass.

## 2026-10-11 — 1.0.0

- First build: box_krpost_fc_01-03 flattened, each with two different fictional DX EXPRESS 4x6 in labels on the lid (sort codes 5K21-1A/2B/3C), under usd/variants/; the main file selects one by its `label` variant set. Built by scripts/tools/make_polybag_sort_variants.py.

