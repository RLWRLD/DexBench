# plum_v01

Newest first. A version with no entry here is not a version.

## 2026-10-30 — 1.2.0

- Split the Newton runtime's collider into 4 overlapping convex hulls of at most 64 vertices each (was one): the scan surface now stands at most 0.6 mm outside them (was 2.0 mm), so a finger touching the collider sinks that much less into the drawn fruit. MuJoCo-Warp takes no hull over 64 vertices, so the closer fit comes from smaller pieces. Maximum bbox change 0.37 mm; convex volume ratio 0.9738.

## 2026-10-30 — 1.1.0

- Replace the Newton runtime's scan-mesh collider with one 64-vertex convex hull, bound directly to the package physics material. Keep the scan mesh and textures for rendering, and preserve authored mass. Inertia is derived from the hull. Maximum bbox change 0.62 mm; convex volume ratio 0.9103.

## 2026-10-29 — 1.0.0

- Rigid plum variant (small round) from object_library/plum_scan: render mesh deformed and recoloured by scripts/tools/make_fruit_variants.py, convex hull collider on the new mesh, realistic weight 48.3 g, plum skin friction 0.55/0.45. Parameters {"label": "small round", "size_mm": 46, "aspect": 0.96, "seed": 201}; size [44.5, 46.0, 44.2] mm (-11.5% of the 52 mm preset), 15000 vertices, mass 0.0483 kg (realistic plum weight: render-mesh volume x flesh density x (1 + seeded spread), clamped to the typical range; implied density 1063.4 kg/m3 over render-mesh volume 4.542e-05 m3), physics material static/dynamic friction 0.55/0.45, restitution 0.05, grasp line [[-0.0423, 0.0, 0.0221], [0.0423, 0.0, 0.0221]] m. Base quality: SpaceAI scan, 15k vertices, 2048 px albedo: good. Newton runtime payload as in plum_scan.
