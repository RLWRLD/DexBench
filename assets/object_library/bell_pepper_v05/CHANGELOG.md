# bell_pepper_v05

Newest first. A version with no entry here is not a version.

## 2026-10-30 — 1.2.0

- Split the Newton runtime's collider into 4 overlapping convex hulls of at most 64 vertices each (was one): the scan surface now stands at most 2.3 mm outside them (was 3.7 mm), so a finger touching the collider sinks that much less into the drawn fruit. MuJoCo-Warp takes no hull over 64 vertices, so the closer fit comes from smaller pieces. Maximum bbox change 0.61 mm; convex volume ratio 0.9781.

## 2026-10-30 — 1.1.0

- Replace the Newton runtime's scan-mesh collider with one 64-vertex convex hull, bound directly to the package physics material. Keep the scan mesh and textures for rendering, and preserve authored mass. Inertia is derived from the hull. Maximum bbox change 1.45 mm; convex volume ratio 0.9240.

## 2026-10-29 — 1.0.0

- Rigid bell_pepper variant (one side flat, lighter green) from SimReady food_produce_bellPepper_a01 (nvidia/simready-assets@a666cb43, CC-BY-4.0) in apple_scan's package layout: render mesh deformed and recoloured by scripts/tools/make_fruit_variants.py, convex hull collider on the new mesh, realistic weight 180.0 g, bell_pepper skin friction 0.55/0.45. Parameters {"label": "one side flat, lighter green", "size_mm": 115, "flatten": {"depth": 0.1, "azimuth_deg": 90}, "color": {"hue_deg": -3, "sat": 0.95, "val": 1.08}, "seed": 1105}; size [81.2, 115.0, 80.6] mm (+4.5% of the 110 mm preset), 19378 vertices, mass 0.18 kg (realistic bell_pepper weight: render-mesh volume x flesh density x (1 + seeded spread), clamped to the typical range; implied density 512.3 kg/m3 over render-mesh volume 3.514e-04 m3), physics material static/dynamic friction 0.55/0.45, restitution 0.05, grasp line [[-0.0599, 0.0, 0.0403], [0.0599, 0.0, 0.0403]] m. Base quality: SimReady bellPepper_a01 (red) recoloured green: 1.2k quads Loop-subdivided x2, 2048 px albedo + normal map: good. Newton runtime payload as in apple_scan. Source: food_produce_bellPepper_a01 from NVIDIA SimReady Assets (nvidia/simready-assets), CC BY 4.0; modified: render mesh re-scaled and deformed, albedo recoloured, re-packaged by scripts/tools/make_fruit_variants.py.
