# pear_v04

Newest first. A version with no entry here is not a version.

## 2026-10-30 — 1.2.0

- Split the Newton runtime's collider into 4 overlapping convex hulls of at most 64 vertices each (was one): the scan surface now stands at most 1.3 mm outside them (was 1.9 mm), so a finger touching the collider sinks that much less into the drawn fruit. MuJoCo-Warp takes no hull over 64 vertices, so the closer fit comes from smaller pieces. Maximum bbox change 0.57 mm; convex volume ratio 0.9657.

## 2026-10-30 — 1.1.0

- Replace the Newton runtime's scan-mesh collider with one 64-vertex convex hull, bound directly to the package physics material. Keep the scan mesh and textures for rendering, and preserve authored mass. Inertia is derived from the hull. Maximum bbox change 0.83 mm; convex volume ratio 0.9380.

## 2026-10-29 — 1.0.0

- Rigid pear variant (plump, lumpy) from SimReady food_produce_pear_j01 (nvidia/simready-assets@a666cb43, CC-BY-4.0) in apple_scan's package layout: render mesh deformed and recoloured by scripts/tools/make_fruit_variants.py, convex hull collider on the new mesh, realistic weight 205.8 g, pear skin friction 0.6/0.5. Parameters {"label": "plump, lumpy", "size_mm": 96, "ovality": 0.92, "lobes": {"n": 4, "amp": 0.04, "kappa": 5.0}, "seed": 1004}; size [73.6, 96.0, 69.8] mm (-4.0% of the 100 mm preset), 5316 vertices, mass 0.2058 kg (realistic pear weight: render-mesh volume x flesh density x (1 + seeded spread), clamped to the typical range; implied density 1088.2 kg/m3 over render-mesh volume 1.891e-04 m3), physics material static/dynamic friction 0.6/0.5, restitution 0.05, grasp line [[-0.0568, 0.0, 0.0349], [0.0568, 0.0, 0.0349]] m. Base quality: SimReady pear_j01 (light green European pear): 5.3k faces, 512 px albedo + normal map: fair. Newton runtime payload as in apple_scan. Source: food_produce_pear_j01 from NVIDIA SimReady Assets (nvidia/simready-assets), CC BY 4.0; modified: render mesh re-scaled and deformed, albedo recoloured, re-packaged by scripts/tools/make_fruit_variants.py.
