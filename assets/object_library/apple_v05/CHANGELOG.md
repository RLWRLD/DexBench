# apple_v05

Newest first. A version with no entry here is not a version.

## 2026-10-30 — 1.2.0

- Split the Newton runtime's collider into 4 overlapping convex hulls of at most 64 vertices each (was one): the scan surface now stands at most 1.9 mm outside them (was 3.4 mm), so a finger touching the collider sinks that much less into the drawn fruit. MuJoCo-Warp takes no hull over 64 vertices, so the closer fit comes from smaller pieces. Maximum bbox change 0.60 mm; convex volume ratio 0.9741.

## 2026-10-30 — 1.1.0

- Replace the Newton runtime's scan-mesh collider with one 64-vertex convex hull, bound directly to the package physics material. Keep the scan mesh and textures for rendering, and preserve authored mass. Inertia is derived from the hull. Maximum bbox change 1.33 mm; convex volume ratio 0.9090.

## 2026-10-29 — 1.0.0

- Rigid apple variant (one side flat, tilted) from object_library/apple_scan: render mesh deformed and recoloured by scripts/tools/make_fruit_variants.py, convex hull collider on the new mesh, realistic weight 247.2 g, apple skin friction 0.6/0.5. Parameters {"label": "one side flat, tilted", "size_mm": 86, "flatten": {"depth": 0.12, "azimuth_deg": 210}, "tilt_deg": 7, "tilt_azimuth_deg": 30, "seed": 105}; size [79.6, 86.0, 79.2] mm (+7.5% of the 80 mm preset), 14998 vertices, mass 0.2472 kg (realistic apple weight: render-mesh volume x flesh density x (1 + seeded spread), clamped to the typical range; implied density 837.8 kg/m3 over render-mesh volume 2.951e-04 m3), physics material static/dynamic friction 0.6/0.5, restitution 0.05, grasp line [[-0.0598, 0.0, 0.0396], [0.0598, 0.0, 0.0396]] m. Base quality: SpaceAI scan, 15k vertices, 2048 px albedo: good. Newton runtime payload as in apple_scan.
