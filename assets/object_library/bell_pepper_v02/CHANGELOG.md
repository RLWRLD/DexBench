# bell_pepper_v02

Newest first. A version with no entry here is not a version.

## 2026-10-30 — 1.2.0

- Split the Newton runtime's collider into 4 overlapping convex hulls of at most 64 vertices each (was one): the scan surface now stands at most 2.0 mm outside them (was 4.0 mm), so a finger touching the collider sinks that much less into the drawn fruit. MuJoCo-Warp takes no hull over 64 vertices, so the closer fit comes from smaller pieces. Maximum bbox change 0.75 mm; convex volume ratio 0.9780.

## 2026-10-30 — 1.1.0

- Replace the Newton runtime's scan-mesh collider with one 64-vertex convex hull, bound directly to the package physics material. Keep the scan mesh and textures for rendering, and preserve authored mass. Inertia is derived from the hull. Maximum bbox change 1.21 mm; convex volume ratio 0.9253.

## 2026-10-29 — 1.0.0

- Rigid bell_pepper variant (large, darker green) from SimReady food_produce_bellPepper_a01 (nvidia/simready-assets@a666cb43, CC-BY-4.0) in apple_scan's package layout: render mesh deformed and recoloured by scripts/tools/make_fruit_variants.py, convex hull collider on the new mesh, realistic weight 176.9 g, bell_pepper skin friction 0.55/0.45. Parameters {"label": "large, darker green", "size_mm": 118, "color": {"hue_deg": 3, "sat": 1.05, "val": 0.9}, "seed": 1102}; size [79.2, 118.0, 78.6] mm (+7.3% of the 110 mm preset), 19378 vertices, mass 0.1769 kg (realistic bell_pepper weight: render-mesh volume x flesh density x (1 + seeded spread), clamped to the typical range; implied density 521.2 kg/m3 over render-mesh volume 3.394e-04 m3), physics material static/dynamic friction 0.55/0.45, restitution 0.05, grasp line [[-0.0589, 0.0, 0.0393], [0.0589, 0.0, 0.0393]] m. Base quality: SimReady bellPepper_a01 (red) recoloured green: 1.2k quads Loop-subdivided x2, 2048 px albedo + normal map: good. Newton runtime payload as in apple_scan. Source: food_produce_bellPepper_a01 from NVIDIA SimReady Assets (nvidia/simready-assets), CC BY 4.0; modified: render mesh re-scaled and deformed, albedo recoloured, re-packaged by scripts/tools/make_fruit_variants.py.
