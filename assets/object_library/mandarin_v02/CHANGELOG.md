# mandarin_v02

Newest first. A version with no entry here is not a version.

## 2026-10-30 — 1.2.0

- Split the Newton runtime's collider into 4 overlapping convex hulls of at most 64 vertices each (was one): the scan surface now stands at most 0.9 mm outside them (was 2.5 mm), so a finger touching the collider sinks that much less into the drawn fruit. MuJoCo-Warp takes no hull over 64 vertices, so the closer fit comes from smaller pieces. Maximum bbox change 0.53 mm; convex volume ratio 0.9742.

## 2026-10-30 — 1.1.0

- Replace the Newton runtime's scan-mesh collider with one 64-vertex convex hull, bound directly to the package physics material. Keep the scan mesh and textures for rendering, and preserve authored mass. Inertia is derived from the hull. Maximum bbox change 1.34 mm; convex volume ratio 0.9087.

## 2026-10-29 — 1.0.0

- Rigid mandarin variant (large flat) from SimReady food_produce_tangerine_a01 (nvidia/simready-assets@a666cb43, CC-BY-4.0) in apple_scan's package layout: render mesh deformed and recoloured by scripts/tools/make_fruit_variants.py, convex hull collider on the new mesh, realistic weight 125.1 g, mandarin skin friction 0.55/0.45. Parameters {"label": "large flat", "size_mm": 72, "aspect": 0.72, "color": {"hue_deg": -3, "sat": 1.06, "val": 0.96}, "seed": 702}; size [72.0, 70.7, 51.8] mm (+16.1% of the 62 mm preset), 16794 vertices, mass 0.1251 kg (realistic mandarin weight: render-mesh volume x flesh density x (1 + seeded spread), clamped to the typical range; implied density 830.1 kg/m3 over render-mesh volume 1.507e-04 m3), physics material static/dynamic friction 0.55/0.45, restitution 0.05, grasp line [[-0.056, 0.0, 0.0259], [0.056, 0.0, 0.0259]] m. Base quality: SimReady tangerine_a01: 4.2k vertices Loop-subdivided x1, 512 px albedo + normal map: fair (soft texture up close). Newton runtime payload as in apple_scan. Source: food_produce_tangerine_a01 from NVIDIA SimReady Assets (nvidia/simready-assets), CC BY 4.0; modified: render mesh re-scaled and deformed, albedo recoloured, re-packaged by scripts/tools/make_fruit_variants.py.
