# plum_v03

Newest first. A version with no entry here is not a version.

## 2026-10-30 — 1.2.0

- Split the Newton runtime's collider into 4 overlapping convex hulls of at most 64 vertices each (was one): the scan surface now stands at most 0.8 mm outside them (was 1.7 mm), so a finger touching the collider sinks that much less into the drawn fruit. MuJoCo-Warp takes no hull over 64 vertices, so the closer fit comes from smaller pieces. Maximum bbox change 0.15 mm; convex volume ratio 0.9730.

## 2026-10-30 — 1.1.0

- Replace the Newton runtime's scan-mesh collider with one 64-vertex convex hull, bound directly to the package physics material. Keep the scan mesh and textures for rendering, and preserve authored mass. Inertia is derived from the hull. Maximum bbox change 0.78 mm; convex volume ratio 0.9108.

## 2026-10-29 — 1.0.0

- Rigid plum variant (oval, suture) from object_library/plum_scan: render mesh deformed and recoloured by scripts/tools/make_fruit_variants.py, convex hull collider on the new mesh, realistic weight 73.2 g, plum skin friction 0.55/0.45. Parameters {"label": "oval, suture", "size_mm": 55, "ovality": 1.1, "aspect": 1.05, "lobes": {"n": 1, "amp": -0.06, "kappa": 12.0}, "seed": 203}; size [46.3, 52.7, 55.0] mm (+5.8% of the 52 mm preset), 15000 vertices, mass 0.0732 kg (realistic plum weight: render-mesh volume x flesh density x (1 + seeded spread), clamped to the typical range; implied density 1086.8 kg/m3 over render-mesh volume 6.735e-05 m3), physics material static/dynamic friction 0.55/0.45, restitution 0.05, grasp line [[-0.0432, 0.0, 0.0275], [0.0432, 0.0, 0.0275]] m. Base quality: SpaceAI scan, 15k vertices, 2048 px albedo: good. Newton runtime payload as in plum_scan.
