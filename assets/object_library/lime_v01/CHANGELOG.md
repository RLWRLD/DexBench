# lime_v01

Newest first. A version with no entry here is not a version.

## 2026-10-30 — 1.2.0

- Split the Newton runtime's collider into 4 overlapping convex hulls of at most 64 vertices each (was one): the scan surface now stands at most 1.4 mm outside them (was 1.8 mm), so a finger touching the collider sinks that much less into the drawn fruit. MuJoCo-Warp takes no hull over 64 vertices, so the closer fit comes from smaller pieces. Maximum bbox change 0.85 mm; convex volume ratio 0.9753.

## 2026-10-30 — 1.1.0

- Replace the Newton runtime's scan-mesh collider with one 64-vertex convex hull, bound directly to the package physics material. Keep the scan mesh and textures for rendering, and preserve authored mass. Inertia is derived from the hull. Maximum bbox change 0.63 mm; convex volume ratio 0.9141.

## 2026-10-29 — 1.0.0

- Rigid lime variant (small round) from SimReady food_produce_lime_d01 (nvidia/simready-assets@a666cb43, CC-BY-4.0) in apple_scan's package layout: render mesh deformed and recoloured by scripts/tools/make_fruit_variants.py, convex hull collider on the new mesh, realistic weight 63.3 g, lime skin friction 0.55/0.45. Parameters {"label": "small round", "size_mm": 53, "aspect": 0.95, "seed": 901}; size [46.2, 53.0, 50.3] mm (-8.6% of the 58 mm preset), 2500 vertices, mass 0.0633 kg (realistic lime weight: render-mesh volume x flesh density x (1 + seeded spread), clamped to the typical range; implied density 1030.7 kg/m3 over render-mesh volume 6.142e-05 m3), physics material static/dynamic friction 0.55/0.45, restitution 0.05, grasp line [[-0.0431, 0.0, 0.0252], [0.0431, 0.0, 0.0252]] m. Base quality: SimReady lime_d01: 5k triangles, 512 px albedo + normal map: fair (soft texture up close). Newton runtime payload as in apple_scan. Source: food_produce_lime_d01 from NVIDIA SimReady Assets (nvidia/simready-assets), CC BY 4.0; modified: render mesh re-scaled and deformed, albedo recoloured, re-packaged by scripts/tools/make_fruit_variants.py.
