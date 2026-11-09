# apple_v02

Newest first. A version with no entry here is not a version.

## 2026-10-30 — 1.2.0

- Split the Newton runtime's collider into 4 overlapping convex hulls of at most 64 vertices each (was one): the scan surface now stands at most 1.6 mm outside them (was 3.2 mm), so a finger touching the collider sinks that much less into the drawn fruit. MuJoCo-Warp takes no hull over 64 vertices, so the closer fit comes from smaller pieces. Maximum bbox change 0.45 mm; convex volume ratio 0.9755.

## 2026-10-30 — 1.1.0

- Replace the Newton runtime's scan-mesh collider with one 64-vertex convex hull, bound directly to the package physics material. Keep the scan mesh and textures for rendering, and preserve authored mass. Inertia is derived from the hull. Maximum bbox change 1.33 mm; convex volume ratio 0.9151.

## 2026-10-29 — 1.0.0

- Rigid apple variant (large flat, lobed) from object_library/apple_scan: render mesh deformed and recoloured by scripts/tools/make_fruit_variants.py, convex hull collider on the new mesh, realistic weight 262.2 g, apple skin friction 0.6/0.5. Parameters {"label": "large flat, lobed", "size_mm": 92, "aspect": 0.82, "lobes": {"n": 5, "amp": 0.05, "kappa": 6.0}, "seed": 102}; size [85.6, 92.0, 74.5] mm (+15.0% of the 80 mm preset), 14998 vertices, mass 0.2622 kg (realistic apple weight: render-mesh volume x flesh density x (1 + seeded spread), clamped to the typical range; implied density 800.5 kg/m3 over render-mesh volume 3.275e-04 m3), physics material static/dynamic friction 0.6/0.5, restitution 0.05, grasp line [[-0.0626, 0.0, 0.0372], [0.0626, 0.0, 0.0372]] m. Base quality: SpaceAI scan, 15k vertices, 2048 px albedo: good. Newton runtime payload as in apple_scan.
