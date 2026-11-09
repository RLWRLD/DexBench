# bell_pepper_v03

Newest first. A version with no entry here is not a version.

## 2026-10-30 — 1.2.0

- Split the Newton runtime's collider into 4 overlapping convex hulls of at most 64 vertices each (was one): the scan surface now stands at most 1.8 mm outside them (was 3.2 mm), so a finger touching the collider sinks that much less into the drawn fruit. MuJoCo-Warp takes no hull over 64 vertices, so the closer fit comes from smaller pieces. Maximum bbox change 0.86 mm; convex volume ratio 0.9776.

## 2026-10-30 — 1.1.0

- Replace the Newton runtime's scan-mesh collider with one 64-vertex convex hull, bound directly to the package physics material. Keep the scan mesh and textures for rendering, and preserve authored mass. Inertia is derived from the hull. Maximum bbox change 0.70 mm; convex volume ratio 0.9219.

## 2026-10-29 — 1.0.0

- Rigid bell_pepper variant (squat, wide) from SimReady food_produce_bellPepper_a01 (nvidia/simready-assets@a666cb43, CC-BY-4.0) in apple_scan's package layout: render mesh deformed and recoloured by scripts/tools/make_fruit_variants.py, convex hull collider on the new mesh, realistic weight 159.2 g, bell_pepper skin friction 0.55/0.45. Parameters {"label": "squat, wide", "size_mm": 106, "ovality": 0.86, "seed": 1103}; size [82.7, 106.0, 76.1] mm (-3.6% of the 110 mm preset), 19378 vertices, mass 0.1592 kg (realistic bell_pepper weight: render-mesh volume x flesh density x (1 + seeded spread), clamped to the typical range; implied density 516.0 kg/m3 over render-mesh volume 3.085e-04 m3), physics material static/dynamic friction 0.55/0.45, restitution 0.05, grasp line [[-0.0606, 0.0, 0.0381], [0.0606, 0.0, 0.0381]] m. Base quality: SimReady bellPepper_a01 (red) recoloured green: 1.2k quads Loop-subdivided x2, 2048 px albedo + normal map: good. Newton runtime payload as in apple_scan. Source: food_produce_bellPepper_a01 from NVIDIA SimReady Assets (nvidia/simready-assets), CC BY 4.0; modified: render mesh re-scaled and deformed, albedo recoloured, re-packaged by scripts/tools/make_fruit_variants.py.
