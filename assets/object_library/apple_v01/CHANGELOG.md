# apple_v01

Newest first. A version with no entry here is not a version.

## 2026-10-30 — 1.2.0

- Split the Newton runtime's collider into 4 overlapping convex hulls of at most 64 vertices each (was one): the scan surface now stands at most 1.3 mm outside them (was 2.7 mm), so a finger touching the collider sinks that much less into the drawn fruit. MuJoCo-Warp takes no hull over 64 vertices, so the closer fit comes from smaller pieces. Maximum bbox change 0.24 mm; convex volume ratio 0.9732.

## 2026-10-30 — 1.1.0

- Replace the Newton runtime's scan-mesh collider with one 64-vertex convex hull, bound directly to the package physics material. Keep the scan mesh and textures for rendering, and preserve authored mass. Inertia is derived from the hull. Maximum bbox change 0.94 mm; convex volume ratio 0.9112.

## 2026-10-29 — 1.0.0

- Rigid apple variant (small round) from object_library/apple_scan: render mesh deformed and recoloured by scripts/tools/make_fruit_variants.py, convex hull collider on the new mesh, realistic weight 138.7 g, apple skin friction 0.6/0.5. Parameters {"label": "small round", "size_mm": 70, "aspect": 0.92, "color": {"hue_deg": -3, "sat": 1.06, "val": 0.95}, "seed": 101}; size [65.9, 70.0, 64.4] mm (-12.5% of the 80 mm preset), 14998 vertices, mass 0.1387 kg (realistic apple weight: render-mesh volume x flesh density x (1 + seeded spread), clamped to the typical range; implied density 844.4 kg/m3 over render-mesh volume 1.643e-04 m3), physics material static/dynamic friction 0.6/0.5, restitution 0.05, grasp line [[-0.0529, 0.0, 0.0322], [0.0529, 0.0, 0.0322]] m. Base quality: SpaceAI scan, 15k vertices, 2048 px albedo: good. Newton runtime payload as in apple_scan.
