# mandarin_v03

Newest first. A version with no entry here is not a version.

## 2026-10-30 — 1.2.0

- Split the Newton runtime's collider into 4 overlapping convex hulls of at most 64 vertices each (was one): the scan surface now stands at most 1.0 mm outside them (was 2.3 mm), so a finger touching the collider sinks that much less into the drawn fruit. MuJoCo-Warp takes no hull over 64 vertices, so the closer fit comes from smaller pieces. Maximum bbox change 0.24 mm; convex volume ratio 0.9742.

## 2026-10-30 — 1.1.0

- Replace the Newton runtime's scan-mesh collider with one 64-vertex convex hull, bound directly to the package physics material. Keep the scan mesh and textures for rendering, and preserve authored mass. Inertia is derived from the hull. Maximum bbox change 1.21 mm; convex volume ratio 0.9095.

## 2026-10-29 — 1.0.0

- Rigid mandarin variant (lumpy loose peel) from SimReady food_produce_tangerine_a01 (nvidia/simready-assets@a666cb43, CC-BY-4.0) in apple_scan's package layout: render mesh deformed and recoloured by scripts/tools/make_fruit_variants.py, convex hull collider on the new mesh, realistic weight 94.7 g, mandarin skin friction 0.55/0.45. Parameters {"label": "lumpy loose peel", "size_mm": 64, "aspect": 0.78, "lobes": {"n": 7, "amp": 0.05, "kappa": 6.0}, "seed": 703}; size [64.0, 62.5, 49.0] mm (+3.2% of the 62 mm preset), 16794 vertices, mass 0.0947 kg (realistic mandarin weight: render-mesh volume x flesh density x (1 + seeded spread), clamped to the typical range; implied density 852.9 kg/m3 over render-mesh volume 1.110e-04 m3), physics material static/dynamic friction 0.55/0.45, restitution 0.05, grasp line [[-0.052, 0.0, 0.0245], [0.052, 0.0, 0.0245]] m. Base quality: SimReady tangerine_a01: 4.2k vertices Loop-subdivided x1, 512 px albedo + normal map: fair (soft texture up close). Newton runtime payload as in apple_scan. Source: food_produce_tangerine_a01 from NVIDIA SimReady Assets (nvidia/simready-assets), CC BY 4.0; modified: render mesh re-scaled and deformed, albedo recoloured, re-packaged by scripts/tools/make_fruit_variants.py.
