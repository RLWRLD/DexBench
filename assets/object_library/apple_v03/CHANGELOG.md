# apple_v03

Newest first. A version with no entry here is not a version.

## 2026-10-30 — 1.2.0

- Split the Newton runtime's collider into 4 overlapping convex hulls of at most 64 vertices each (was one): the scan surface now stands at most 1.7 mm outside them (was 3.3 mm), so a finger touching the collider sinks that much less into the drawn fruit. MuJoCo-Warp takes no hull over 64 vertices, so the closer fit comes from smaller pieces. Maximum bbox change 0.53 mm; convex volume ratio 0.9743.

## 2026-10-30 — 1.1.0

- Replace the Newton runtime's scan-mesh collider with one 64-vertex convex hull, bound directly to the package physics material. Keep the scan mesh and textures for rendering, and preserve authored mass. Inertia is derived from the hull. Maximum bbox change 1.40 mm; convex volume ratio 0.9114.

## 2026-10-29 — 1.0.0

- Rigid apple variant (tall conical, deeper red) from object_library/apple_scan: render mesh deformed and recoloured by scripts/tools/make_fruit_variants.py, convex hull collider on the new mesh, realistic weight 207.2 g, apple skin friction 0.6/0.5. Parameters {"label": "tall conical, deeper red", "size_mm": 80, "aspect": 1.0, "profile": {"taper": 0.12}, "lobes": {"n": 5, "amp": 0.04, "kappa": 7.0}, "color": {"hue_deg": -4, "sat": 1.08, "val": 0.9}, "seed": 103}; size [76.5, 80.0, 78.8] mm (+0.0% of the 80 mm preset), 14998 vertices, mass 0.2072 kg (realistic apple weight: render-mesh volume x flesh density x (1 + seeded spread), clamped to the typical range; implied density 807.1 kg/m3 over render-mesh volume 2.567e-04 m3), physics material static/dynamic friction 0.6/0.5, restitution 0.05, grasp line [[-0.0579, 0.0, 0.0394], [0.0579, 0.0, 0.0394]] m. Base quality: SpaceAI scan, 15k vertices, 2048 px albedo: good. Newton runtime payload as in apple_scan.
