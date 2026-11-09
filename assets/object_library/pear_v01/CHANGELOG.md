# pear_v01

Newest first. A version with no entry here is not a version.

## 2026-10-30 — 1.2.0

- Split the Newton runtime's collider into 4 overlapping convex hulls of at most 64 vertices each (was one): the scan surface now stands at most 1.7 mm outside them (was 1.9 mm), so a finger touching the collider sinks that much less into the drawn fruit. MuJoCo-Warp takes no hull over 64 vertices, so the closer fit comes from smaller pieces. Maximum bbox change 0.13 mm; convex volume ratio 0.9663.

## 2026-10-30 — 1.1.0

- Replace the Newton runtime's scan-mesh collider with one 64-vertex convex hull, bound directly to the package physics material. Keep the scan mesh and textures for rendering, and preserve authored mass. Inertia is derived from the hull. Maximum bbox change 0.80 mm; convex volume ratio 0.9344.

## 2026-10-29 — 1.0.0

- Rigid pear variant (small) from SimReady food_produce_pear_j01 (nvidia/simready-assets@a666cb43, CC-BY-4.0) in apple_scan's package layout: render mesh deformed and recoloured by scripts/tools/make_fruit_variants.py, convex hull collider on the new mesh, realistic weight 162.0 g, pear skin friction 0.6/0.5. Parameters {"label": "small", "size_mm": 93, "color": {"hue_deg": 2, "sat": 1.0, "val": 1.04}, "seed": 1001}; size [65.2, 93.0, 64.1] mm (-7.0% of the 100 mm preset), 5316 vertices, mass 0.162 kg (realistic pear weight: render-mesh volume x flesh density x (1 + seeded spread), clamped to the typical range; implied density 1087.8 kg/m3 over render-mesh volume 1.489e-04 m3), physics material static/dynamic friction 0.6/0.5, restitution 0.05, grasp line [[-0.0526, 0.0, 0.0321], [0.0526, 0.0, 0.0321]] m. Base quality: SimReady pear_j01 (light green European pear): 5.3k faces, 512 px albedo + normal map: fair. Newton runtime payload as in apple_scan. Source: food_produce_pear_j01 from NVIDIA SimReady Assets (nvidia/simready-assets), CC BY 4.0; modified: render mesh re-scaled and deformed, albedo recoloured, re-packaged by scripts/tools/make_fruit_variants.py.
