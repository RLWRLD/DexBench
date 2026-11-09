# plum_v04

Newest first. A version with no entry here is not a version.

## 2026-10-30 — 1.2.0

- Split the Newton runtime's collider into 4 overlapping convex hulls of at most 64 vertices each (was one): the scan surface now stands at most 0.7 mm outside them (was 1.6 mm), so a finger touching the collider sinks that much less into the drawn fruit. MuJoCo-Warp takes no hull over 64 vertices, so the closer fit comes from smaller pieces. Maximum bbox change 0.24 mm; convex volume ratio 0.9738.

## 2026-10-30 — 1.1.0

- Replace the Newton runtime's scan-mesh collider with one 64-vertex convex hull, bound directly to the package physics material. Keep the scan mesh and textures for rendering, and preserve authored mass. Inertia is derived from the hull. Maximum bbox change 0.70 mm; convex volume ratio 0.9106.

## 2026-10-29 — 1.0.0

- Rigid plum variant (lumpy) from object_library/plum_scan: render mesh deformed and recoloured by scripts/tools/make_fruit_variants.py, convex hull collider on the new mesh, realistic weight 56.6 g, plum skin friction 0.55/0.45. Parameters {"label": "lumpy", "size_mm": 50, "lobes": {"n": 6, "amp": 0.05, "kappa": 5.0}, "seed": 204}; size [47.2, 50.0, 44.9] mm (-3.8% of the 52 mm preset), 15000 vertices, mass 0.0566 kg (realistic plum weight: render-mesh volume x flesh density x (1 + seeded spread), clamped to the typical range; implied density 1060.9 kg/m3 over render-mesh volume 5.335e-05 m3), physics material static/dynamic friction 0.55/0.45, restitution 0.05, grasp line [[-0.0436, 0.0, 0.0225], [0.0436, 0.0, 0.0225]] m. Base quality: SpaceAI scan, 15k vertices, 2048 px albedo: good. Newton runtime payload as in plum_scan.
