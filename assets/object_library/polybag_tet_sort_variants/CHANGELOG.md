# polybag_tet_sort_variants

Newest first. A version with no entry here is not a version.

## 2026-10-27 — 5.0.0

- Every bag takes bag shape SI, chosen on 2026-10-27 from the candidate shapes (first built for train_a's four bodies only), in all 33 variants. The body is a symmetric pillow instead of the parcel section: the seam at mid-height, 2 mm thick at the edge, a straight 45 mm wedge to the full thickness, 8 / 7.6 / 7.2 cm by label 1A / 2B / 3C (sort_variants.THICKNESS_LEVELS_M; until now 5.5 / 6.5 / 7.5 cm cycled over body and label). Its TetMesh is a structured pillow (15 mm columns, 3 layers, 20 mm corner radius; 1200-2900 vertices instead of 536) with per-tet materials authored on it (dexbench:tetYoungsModulus / tetPoissonsRatio / tetKDamp: core and bottom layers 2.5 kPa, the top skin 0.5 kPa, Poisson 0.05, kDamp 0.5 / 0.2 / 0.5; the bound material carries the core's, 2.5 kPa instead of 300) and a surface film of springs (dexbench:surfaceFilmMode springs, 2 kN/m, kd 0.1, the whole surface). Masses by size: 30x40 0.25 kg, 25x35 0.22, 20x30 0.20 (SI's), and for the held-out shapes by footprint between them, 35x45 0.30 and 20x40 0.21; the density takes the volume. The textured mesh keeps its vertices, UVs and wrinkles, moved onto the new section's top and bottom sheets; textures and labels are unchanged. Each layer's dexbench:sort_variant now says the variant's split (purple_25x35 and skyblue_20x30: eval_colour, as 4.1.1's sidecar did), thickness and modulus, and dexbench:pa_spec records the build spec.
- Built by scripts/tools/make_polybag_sort_variants.py over 4.1.1's parcel layers (its SI_SPEC, the parcel thickness and modulus PARCEL_THICKNESS_LEVELS_M / PARCEL_YOUNGS_MODULUS_PA), the textured mesh's sheets placed by this package's parcel section; each layer's dexbench:pa_spec records SI_SPEC with the variant's mass. DB-L1's collection (scripts/evaluation/experiment_configs/db_l1_collect.yaml) overrides the per-tet Poisson ratio and damping at run time (--polybag_tet_poisson 0.45, --polybag_tet_damping 10). Checked on CPU Newton without Kit: all 33 load through the tet template (finite, counts above, no inverted tets); grey_25x35__1a, grey_35x45__1a and black_20x40__3c rest on a floor with no inverted tet, settled under 1 mm/s, as thick as built within 2 mm. Geometry and material change (every path, prim and the label variant set stay), hence the major.

## 2026-10-27 — 4.1.1

- Sidecar only: purple_25x35 and skyblue_20x30 are DB-L1's held-out colours (split eval_colour, sort_variants.BAG_BODIES; decided 2026-10-27), and their six variants' rows say so. Their layers' `dexbench:sort_variant` data still say train: they are rewritten when the pool is next built (its bag shape is being chosen). No geometry, material or label changes, hence the patch.

## 2026-10-22 — 4.1.0

- A long body, black_20x40 (0.199 x 0.367 m, film as black_25x35's), with the three waybills: black_20x40__1a/2b/3c at 6.5/7.5/5.5 cm, with the others' section, 300 kPa and density. It and grey_35x45 are DB-L1's held-out shapes (split eval_shape, sort_variants.BAG_BODIES); purple_25x35 and skyblue_20x30, held out by colour until now, are training bodies (split train), which their variants' sidecar rows and layer data now say: those 6 and grey_35x45's 3 (eval_size until now) are rebuilt (--bodies), their geometry, material and label fields as in 4.0.0, the label print a few pixels off from newer tools; the other 21 variants' files are as shipped. Variants are added, hence the minor.

## 2026-10-13 — 4.0.0

- Thickness and stiffness per variant (sort_variants.THICKNESS_LEVELS_M, YOUNGS_MODULI_PA): 5.5, 6.5 or 7.5 cm in the middle instead of 4.5 (rim and underside unchanged), Young's modulus 300 kPa instead of 15 (newton:kMu / kLambda restated; Poisson ratio 0.15 and kDamp 30 unchanged); each body has all three thicknesses, one per label. At 15 kPa a 4.5 cm bag held by one end stretched into strings (worst tet 45x its length); these stay under 1.4x and need 16 substeps to be as stiff as set (DB-L1 runs the pool at 16 x 5). Mass unchanged, the density takes the volume. Geometry and material change, hence the major.

## 2026-10-11 — 3.0.0

- Every bag takes the parcel section (make_polybag_sort_variants.py SECTION): 4.5 cm thick in the middle instead of 8.5, an 8 mm seam band, the underside lifted to 24 mm at the rim and running down to the belt over 8 cm, so a palm-up hand on the belt slides under the bag instead of pushing it (the 261009 underside climbs ~40 deg within 3.5 cm of the edge). Footprint, mass, material and contact numbers unchanged; the density takes the volume change (22.4 -> 53.3 kg/m3). The textured mesh's normals now follow the x/y scale and the section (they were the 261009 bag's). Geometry changes, hence the major.

## 2026-10-11 — 2.0.0

- The 30 variant layers and their textures move from usd/ to usd/variants/, unchanged; usd/polybag_tet_sort_variants.usd is the main file, a `label` variant set over them (SimReady NP.005: one USD beside the sidecar). A reference to usd/<key>.usd breaks, hence the major.

## 2026-10-11 — 1.0.0

- First build: 30 variants of polybag_cotton_tet_deformable 1.0.1 (10 bodies x 3 waybills), the delivered Korea Post waybill painted out and replaced by fictional DX EXPRESS 4x6 in shipping labels (ISO 15394 zones, sort code 5K21-1A/2B/3C). Built by scripts/tools/make_polybag_sort_variants.py.
