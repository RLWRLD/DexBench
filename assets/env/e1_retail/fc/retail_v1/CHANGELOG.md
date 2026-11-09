# e1_retail/fc/retail_v1

Newest first. A version with no entry here is not a version.

## 2026-11-08 — 1.4.1

- Record public distribution approval by maintainer designation (v3 asset publication policy). Fab Standard License content ships only inside the integrated scene USDZ; see assets/ASSET_LIST.md.
- Keep generation service and texture provenance with the package; preserve the prior account-terms and appearance review records internally. This approval supersedes the historical distribution status below.
- PATCH for rights metadata only; USD layers, textures and scene behavior are unchanged.

## 2026-11-06 — 1.4.0

- Redesign all printed packaging on the five background product albedos. Replace the first-pass artwork with fictional MORUVA and TAVERO snacks, NIVORA tomato puree, SOVELA water and LOMIO/KORU FIZZ/TAVO/MIRELO drinks; remove the remaining original FRESH design, photographs, color schemes and label layouts.
- Adopt the five complete 1254×1254 RGB outputs from the built-in image generator without pixel edits or resampling. Preserve USD layers, UVs, material bindings, geometry, normal/roughness maps, lights and physics byte-for-byte. Existing build-input hash checks now select these new images.
- Preserve the first-pass textures and generation record in the internal archive. Full prompts, input/output hashes and pending scene review are in the maintainers' records.
- Retained PSL/Fab-derived material distribution remains `not_cleared`; fictional packaging is not a trademark or raw-asset redistribution clearance.

## 2026-11-06 — 1.3.0

- Replace five background product albedo atlases with whole AI-edited images: generic cola/orange soda/lemon tea, water, snack packaging and tomato sauce remove the identified real branding. These are the background stock in the room/gondola, separate from the NdotLight R2/R3 object-library labels.
- Use the generated 1254×1254 RGB PNGs unchanged; prior atlases were opaque 2048×2048 RGBA. Keep USD UVs, material bindings, geometry, normal/roughness maps and physics unchanged. The builder preserves the reviewed inputs and refuses mismatched hashes before rebuilding.
- The modification does not clear redistribution of retained PSL/Fab-derived material. Generation prompts, source/output hashes and scene comparisons are internal review evidence.

## 2026-11-06 — 1.2.0

- Replace the four `BK_MI_Floor_{BC,M,N,R}.png` maps with the independently generated
  `textures/retail_floor_generated_basecolor.png`. The generation prompt, tool and output
  hash are in the maintainers' generation record.
- Keep the floor mesh, UVs and material binding. Use metallic 0, a flat tangent-space normal,
  and roughness 0.2123584 (the previous top-surface map mean, excluding atlas padding).
  A material UV transform repeats the 1254-pixel image by 4096/1254 on both axes,
  retaining the previous texel density without resampling the generated image.
  Room fixtures, lights, colliders and task poses are unchanged.
- The look builder preserves and checks the generated PNG before rebuilding, including
  `--check`, and no longer copies the retired maps from the delivery.

## 2026-10-30 — 1.1.0

- `fixtures/produce_display.usdc` follows produce_display_fc 2.0.0: its `ProdFC_Tray1` (with the dividers) and
  `ProdFC_Tray2` trade heights, so the drawn trays stay on the colliders. The price tags stay with their tier.
  `build_retail_v1_look.py` now sets the look's trays to the FC package's tray heights.

## 2026-10-20 — 1.0.1

The sidecar's `source_archive` no longer names the person and the shared folder the delivery came
through; the archive name and the source files' sha256 still identify it. Nothing in the package changes.

## 2026-10-12 — 1.0.0

- The retail_v1 look, adopted from PSL's E1 r03 look-check delivery (2026-10-07,
  `DexBench_E1_RetailStore_r03_LookCheck_261007`) and drawn over the FC_260821 fixtures,
  built by `scripts/tools/build_retail_v1_look.py` from the delivery's `e1_r03_noproxy.usdc`
  and `e1_r03_ext_look.usda` (sha256 in `retail_v1.meta.json`). Nothing here collides: the
  environment draws these layers over the FC packages and room shell, whose colliders and poses
  it keeps.
- `fixtures/<kind>.usdc`: r03's render meshes and the decoration riding on each fixture, in the FC
  package's frame. Two edits from the delivery: r03's gondola is 220 mm longer than FC's at its
  west end, so the look's west posts and brackets move in and its shelves, back, deck, plinth and
  rails squeeze to FC's 0.80 m (tags and price cards keep their size and take their squeezed
  place); the fridge's door is a `Door` Xform at the FC hinge and its Shelf3 wire shelf and MiniFit
  modules sit under a `Shelf3` Xform, so the spawner can follow the FC door angle and Shelf3 raise.
  r03's tote crate is another mesh than FC's shell; its lip stands up to 11 mm proud of FC's.
- `room.usdc`: the rest of the background without its lights. The decorative east drink fridge
  and its stock move +60 mm in x and y, clear of FC's task fridge (100 mm east of r03's) and of
  its door open at 180 degrees; the task bulkhead and banner stretch 3.7 % east over FC's wider
  fridge pair. The track spot heads turn with their lights.
- `lighting.usda`: the delivery's 19 lights and its dome at FC coordinates, attributes as
  delivered except the task fridge's two interior LED-bar lights (`DECO_frgl_led_light_1_L/R`), at
  x0.4 (1336.9 -> 534.76): at the delivered value the open fridge's white liner and floor blew out
  the DB-R2 head camera (12 % of a 320x180 frame, 22 % of the target lane clipped; at x0.4, 6 % and
  15 %). The fridge light bars move with their fridge; the track spots re-aim at the point of their
  fixture they lit in r03 (gondola 24.7 deg, produce display 16.1 deg, task fridge 3.6 deg).
- `textures/`: the 143 delivered textures these layers use, bytes unchanged (one of the 144 is
  used only by what the look leaves out).
