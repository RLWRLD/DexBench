# lemon_v01

Newest first. A version with no entry here is not a version.

## 2026-10-30 — 1.2.0

- Split the Newton runtime's collider into 4 overlapping convex hulls of at most 64 vertices each (was one): the scan surface now stands at most 1.5 mm outside them (was 2.0 mm), so a finger touching the collider sinks that much less into the drawn fruit. MuJoCo-Warp takes no hull over 64 vertices, so the closer fit comes from smaller pieces. Maximum bbox change 0.23 mm; convex volume ratio 0.9732.

## 2026-10-30 — 1.1.0

- Replace the Newton runtime's scan-mesh collider with one 64-vertex convex hull, bound directly to the package physics material. Keep the scan mesh and textures for rendering, and preserve authored mass. Inertia is derived from the hull. Maximum bbox change 1.04 mm; convex volume ratio 0.9117.

## 2026-10-29 — 1.0.0

- Rigid lemon variant (small, rounder) from SimReady food_produce_lemon_a09 (nvidia/simready-assets@a666cb43, CC-BY-4.0) in apple_scan's package layout: render mesh deformed and recoloured by scripts/tools/make_fruit_variants.py, convex hull collider on the new mesh, realistic weight 74.3 g, lemon skin friction 0.55/0.45. Parameters {"label": "small, rounder", "size_mm": 66, "ovality": 0.92, "seed": 501}; size [52.8, 66.0, 47.8] mm (-12.0% of the 75 mm preset), 8516 vertices, mass 0.0743 kg (realistic lemon weight: render-mesh volume x flesh density x (1 + seeded spread), clamped to the typical range; implied density 911.9 kg/m3 over render-mesh volume 8.148e-05 m3), physics material static/dynamic friction 0.55/0.45, restitution 0.05, grasp line [[-0.0464, 0.0, 0.0239], [0.0464, 0.0, 0.0239]] m. Base quality: SimReady lemon_a09: 8.5k vertices, 2048 px albedo + normal map: good. Newton runtime payload as in apple_scan. Source: food_produce_lemon_a09 from NVIDIA SimReady Assets (nvidia/simready-assets), CC BY 4.0; modified: render mesh re-scaled and deformed, albedo recoloured, re-packaged by scripts/tools/make_fruit_variants.py.
