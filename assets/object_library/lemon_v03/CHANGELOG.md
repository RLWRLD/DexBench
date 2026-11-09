# lemon_v03

Newest first. A version with no entry here is not a version.

## 2026-10-30 — 1.2.0

- Split the Newton runtime's collider into 4 overlapping convex hulls of at most 64 vertices each (was one): the scan surface now stands at most 1.8 mm outside them (was 2.3 mm), so a finger touching the collider sinks that much less into the drawn fruit. MuJoCo-Warp takes no hull over 64 vertices, so the closer fit comes from smaller pieces. Maximum bbox change 0.30 mm; convex volume ratio 0.9735.

## 2026-10-30 — 1.1.0

- Replace the Newton runtime's scan-mesh collider with one 64-vertex convex hull, bound directly to the package physics material. Keep the scan mesh and textures for rendering, and preserve authored mass. Inertia is derived from the hull. Maximum bbox change 0.74 mm; convex volume ratio 0.9134.

## 2026-10-29 — 1.0.0

- Rigid lemon variant (a little deeper yellow) from SimReady food_produce_lemon_a09 (nvidia/simready-assets@a666cb43, CC-BY-4.0) in apple_scan's package layout: render mesh deformed and recoloured by scripts/tools/make_fruit_variants.py, convex hull collider on the new mesh, realistic weight 104.0 g, lemon skin friction 0.55/0.45. Parameters {"label": "a little deeper yellow", "size_mm": 75, "color": {"hue_deg": -3, "sat": 1.06, "val": 0.97}, "seed": 503}; size [55.2, 75.0, 52.1] mm (+0.0% of the 75 mm preset), 8516 vertices, mass 0.104 kg (realistic lemon weight: render-mesh volume x flesh density x (1 + seeded spread), clamped to the typical range; implied density 985.7 kg/m3 over render-mesh volume 1.055e-04 m3), physics material static/dynamic friction 0.55/0.45, restitution 0.05, grasp line [[-0.0476, 0.0, 0.0261], [0.0476, 0.0, 0.0261]] m. Base quality: SimReady lemon_a09: 8.5k vertices, 2048 px albedo + normal map: good. Newton runtime payload as in apple_scan. Source: food_produce_lemon_a09 from NVIDIA SimReady Assets (nvidia/simready-assets), CC BY 4.0; modified: render mesh re-scaled and deformed, albedo recoloured, re-packaged by scripts/tools/make_fruit_variants.py.
