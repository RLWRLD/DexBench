# peach_v05

Newest first. A version with no entry here is not a version.

## 2026-10-30 — 1.2.0

- Split the Newton runtime's collider into 4 overlapping convex hulls of at most 64 vertices each (was one): the scan surface now stands at most 1.1 mm outside them (was 3.3 mm), so a finger touching the collider sinks that much less into the drawn fruit. MuJoCo-Warp takes no hull over 64 vertices, so the closer fit comes from smaller pieces. Maximum bbox change 0.32 mm; convex volume ratio 0.9754.

## 2026-10-30 — 1.1.0

- Replace the Newton runtime's scan-mesh collider with one 64-vertex convex hull, bound directly to the package physics material. Keep the scan mesh and textures for rendering, and preserve authored mass. Inertia is derived from the hull. Maximum bbox change 1.04 mm; convex volume ratio 0.9108.

## 2026-10-29 — 1.0.0

- Rigid peach variant (flattened, tilted) from SimReady food_produce_peach_d01 (nvidia/simready-assets@a666cb43, CC-BY-4.0) in apple_scan's package layout: render mesh deformed and recoloured by scripts/tools/make_fruit_variants.py, convex hull collider on the new mesh, realistic weight 200.5 g, peach skin friction 0.75/0.65. Parameters {"label": "flattened, tilted", "size_mm": 78, "aspect": 0.86, "flatten": {"depth": 0.1, "azimuth_deg": 60}, "tilt_deg": 6, "tilt_azimuth_deg": 200, "seed": 305}; size [70.0, 78.0, 68.7] mm (+8.3% of the 72 mm preset), 11998 vertices, mass 0.2005 kg (realistic peach weight: render-mesh volume x flesh density x (1 + seeded spread), clamped to the typical range; implied density 929.6 kg/m3 over render-mesh volume 2.157e-04 m3), physics material static/dynamic friction 0.75/0.65, restitution 0.05, grasp line [[-0.055, 0.0, 0.0344], [0.055, 0.0, 0.0344]] m. Base quality: SimReady peach_d01: 3k vertices Loop-subdivided x1, 2048 px albedo + normal map: good. Newton runtime payload as in apple_scan. Source: food_produce_peach_d01 from NVIDIA SimReady Assets (nvidia/simready-assets), CC BY 4.0; modified: render mesh re-scaled and deformed, albedo recoloured, re-packaged by scripts/tools/make_fruit_variants.py.
