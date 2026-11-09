# object_library

Pool-level history. Each package carries its own CHANGELOG.md.

## 2026-11-09 — DB-H3 utensils pooled

Added the twelve utensils DB-H3's runtime bundles load, until now shipped outside this tree in
DB-H3's evaluation inputs (`eval/db_h3_inputs/asset_payload/`): `cutlery_set_dessert_fork`,
`cutlery_set_table_fork`, `cutlery_set_table_knife`, `cutlery_set_tablespoon`,
`cutlery_set_teaspoon` (ForgeCAD), `butter_knife_d00052`, `fork_d00313` (Palatial Platforms)
and `kitchenware_fork_b01`, `kitchenware_fork_c01`, `kitchenware_fork_k01`,
`kitchenware_spoon_b01`, `kitchenware_spoon_d01` (NVIDIA), all CC BY 4.0 and prepared for DB-H3
by RLWRLD. Each package is the published wrapper layer with its upstream package under
`usd/deps/`; their composed stages are identical to the published ones. Robotics-Prop 3.1.0 gaps
are recorded and allowlisted (see each package CHANGELOG.md).

## 2026-11-08 — frame_prim removed

Removed `frame_prim`, an axis marker that nothing in this repository spawns any more. No environment, registered asset, tool or test names
it. The scene export plan no longer lists it as a reference input.

## 2026-11-08 — Unused produce proxies removed (internal PR 546)

Removed `r1_plum`, `r1_strawberry`, `fresh_red_apple_fc` and `green_banana_fc`:
no current benchmark task loads them. The optional legacy display-stock rows
now resolve to `PLUM_SCAN`, `STRAWBERRY_SCAN`, `APPLE_SCAN` and `BANANA_SCAN`
through the retired-SKU substitution table. Current settled
produce dressing and per-kind R1 task variants remain unchanged. The 2026-11-06
entry below records why these packages were retained at that time.

## 2026-11-06 — Remaining unloaded packages removed

Removed, since none of the 15 benchmark tasks loads them and no code path names them: the YCB
`005_tomato_soup_can`; the desk scene's `mesh_aaef117c`, with `env/desk/default`, and the
in-repo `desk` mesh, which no scene layer or code referenced; the first
retail scene's GLB conversions `shelving_depot` and `fruit_stand`, with `env/retail_store/default`;
`env/common` (a local ground plane nothing spawns); NdotLight's SimReady `container_box`,
`screwdriver` and `spanner` (DB-M3 keeps `power_drill`); SpaceAI's `apple_deformable`,
`plum_deformable` and `strawberry_deformable` (the rigid `*_scan` twins stay and record the twin as
null in 1.3.1); the disks `disk_100mm_5mm` and `disk_130mm_5mm`.

## 2026-11-06 — Unloaded E4 packages removed

Removed, since none of the 15 benchmark tasks loads them and no code path names them: the
delivery's own E4 props (`prop_bowl_16`, `prop_bowl_22`, `prop_cutlery_fork`, `prop_cutlery_knife`,
`prop_cutlery_spoon`, `prop_deep_plate_22`, `prop_jug_1l`, `prop_pot_holder`, `prop_rubber_mat`,
`prop_tray`; DB-H2 keeps `prop_goblet_30cl`), the r01 fixture variants (`round_table_fc_var_a`–`e`,
`round_table_fc_var_navy`, `serving_station_fc_var_a`–`e`, `utensil_bin_fc_var_a`–`e`), the
clearing station `clearing_station_fc` (its slot in `banquet.usda` was inactive and is gone), the
NdotLight DB-H3 cutlery `dragon_fork_19cm`, `dragon_knife_21cm`, `dragon_spoon_19cm` (DB-H3 loads
its utensils from its runtime bundle), the NVIDIA SimReady tableware and furniture of the first
hotel scene (`case_a01`, `dellwood_diningchair`, `dellwood_rounddiningtable`, `fork_big`,
`fork_small`, `plate_large`, `plate_small`, `serving_bowl`, `spoon_big`, `spoon_small`), the
in-house scans `lotte_dorim_plate` and `lotte_minibar_water_notice`, and `stainless_milk_pitcher`
(DB-H2 uses `milk_pitcher_d00527_x15`). The banquet hall's `items_default.usda` and
`variations/var_fixture_01.usda`, which nothing composed, and `env/e4_hotel_fb/legacy` went with
them.

Kept although no task loads it: `steel_water_jug_tall`, which DB-H2's jug axis declares as its
stand-in.

## 2026-11-06 — Unloaded E3 parts removed

Removed, since none of the 15 benchmark tasks loads them and no code path names them:
NdotLight's DB-M4 hex nuts (`hex_nut_m16`, `hex_nut_m20`, `hex_nut_m24`, `hex_nut_m30`,
`hex_nut_m36`; DB-M4 uses the leveling leg) and its DB-M5 parts outside the task's part table
(`l_bracket_804030_t3`, `mk_b7_stud_bolt_m16x80`,
`mk_electric_k3390_alm_blanking_plate`, `mk_high_strength_hex_bolt_m24x120`,
`navi_stainless_welded_elbow_50a_od60_5`, `ngk_bkr6eix_11_spark_plug`,
`ntn_608llbc3_1k_miniature_ball_bearing`, `ntn_6203llu_lp03_deep_groove_ball_bearing`,
`ntn_6206lluc4_5k_deep_groove_ball_bearing`, `pom_conveyor_roller`,
`rs_pro_stepper_motor_535_0372_nema17_class`, `shinjin_high_strength_hex_bolt_m12x60`,
`sus304_flat_bar`, `wieldy_whb11095_5504_nema42_stepper`), the M2 `lg_mnext3_connector_pocket`
(DB-M2 uses `lg_mnext3_liner_connector_task`, which carries its own pocket), and PSL's
`part_sus304_coupon_fc`, `parts_bin_fc_var_green`, `pedestal_bench_fc`, `pedestal_m5_fc` and
`pedestal_station_fc`. The
assembly line's `items_default.usda` and `variations/var_fixture_01.usda`, which nothing
composed and which placed two of these, went with them.

## 2026-11-06 — Unloaded E2 packages removed

Removed, since none of the 15 benchmark tasks loads them and no code path names them: the
260806 full room's leftovers (`env/e2_logistics/legacy`, `paper_stack_80`, `paper_stack_120`,
`polybag_t1`, `polybag_t2`, and `nucleus_mirror/logistics_center`, the NVIDIA Isaac conveyors and
SimReady props it mirrored), the 260827 cell's distractors and conveyor (`dist_longbox_fc`,
`dist_tote_fc`, `dist_wcbox_fc`, `conveyor_mesh`), PSL's three-tote `sort_tote_fc` (the cell
uses `sort_tote_single_fc`), the RoboCasa-derived `control_conveyor_a08`, and the frontier demo's
`conveyor_station_structure` and `polybag_1`–`polybag_5` (DB-L4's `polybag_1`–`polybag_5` parcel
kinds are tints of `polybag_cotton_tet_deformable`, not these packages).

## 2026-11-06 — Unloaded E1 packages removed

Removed, since none of the 15 benchmark tasks loads them and no code path names them:

- PSL's first E1 delivery (260729) — the environment layer `env/e1_retail/legacy` and its
  fixtures and items: `basket`, `fridge`, `fruit_display`, `gondola`, `tote`, `black_energy_can`,
  `blue_chips_bag`, `blue_energy_can`, `cracker_box`, `dark_plum`, `fresh_orange`,
  `fresh_red_apple`, `gold_energy_can`, `green_banana`, `green_energy_can`, `green_soda_bottle`,
  `macaroni_cheese_box`, `morning_crunch_box`, `oatmeal_box`, `orange_drink_carton`,
  `peanut_butter_jar`, `penne_pasta_box`, `plastic_soda_bottle`, `red_energy_can`, `ripe_mango`,
  `ripe_tomato`, `wholegrain_rice_pouch`, `yellow_lemon`;
- the FC delivery's per-SKU proxies `tc1`, `tc2`, `tp1`, `ec1`, `ec2`, `ep1`, `ep2`, `ep3`,
  `r3t1`–`r3t4`, `r3e1`–`r3e6` (every code resolves to NdotLight's product since 2026-09-15),
  and the fixtures only `env/e1_retail/fc/fc.usda` placed: `basket_fc`, `pallet`,
  `fridge_minifit_all_tiers_fc`, `tote_stack_4_fc`, `tote_stack_fc`;
- `refrigerator`, the retired single-mesh fridge.

Kept although no task loads them: `fresh_red_apple_fc`, `green_banana_fc`, `r1_plum`,
`r1_strawberry` (the vendor produce rows of the display stock table resolve to them) and
`ripe_peach` (`settle_produce_clutter.py` settles the clutter with it).

## 2026-11-06 — Legacy E3 shell props removed

Removed with the hand-written E3 shell (`dexbench_manufacturing_environment`), the only
environment that spawned them: `035_power_drill`, `051_large_clamp`, `blue_crate`, `case_c01`,
`computer_cart`, `container_h20`, `conveyor_belt_a05`, `conveyor_belt_a08`, `factory_bolt_m12`,
`factory_nut_m12`, `heavy_duty_packing_table`, `small_klt_bin`. None of the 15 benchmark tasks
loads them. Each package's last version stays in the history of this tree.


<!-- from object_library/fixtures/CHANGELOG.md -->
# object_library/fixtures

Newest first. A version with no entry here is not a version.

## 2026-09-14 — PSL E1 260914 texture upgrade

### gondola_fc 1.1.0 · produce_display_fc 1.1.0
Re-exported by the vendor to carry a full 2k PBR set (base colour, metallic,
normal, roughness) in place of single maps. Geometry verified unchanged by
`scripts/tools/compare_usd_geometry.py`: every mesh has the same point and
face counts and extent; only material bindings differ. Drop-in for the scenes
that reference them.

### fridge_minifit_fc 1.1.0
Same texture treatment, and 42 meshes added: the vendor now ships the
Shelf1/3/4 MiniFit tiers (`FrgFC_MiniFitS1A/S1B/S3A/S3B/S4A/S4B`) with their
`_colproxy` and `_visonly` children, which our spawner used to compose at
runtime from the Shelf2 prototypes. The original 31 meshes are unchanged.
`_spawn_fc_fridge_with_review_state` now skips its own composition when the
vendor tiers are present, so either generation composes once.

### shopping_basket_fc 1.0.0 · shopping_cart_fc 1.0.0 · tote_stack_fc 1.0.0
New in the pool from this delivery. Nothing loads them yet. `tote_stack_fc` is
a baked two-tier stack authoring ten collision proxies; the single `tote_fc`
(five proxies) stays the asset DB-R2 fills and DB-R3 stacks six high.

### textures
45 files from the delivery: 22 shared maps revised (walls, floor, ceiling,
fluorescents, tile sheets, and the tote's 2k set — which upgrades the single
`tote_fc` without touching its geometry) and 23 new. Textures do not affect
geometry or physics, so this changes how every E1 surface looks and nothing
a task measures.

Source: `deliveries/psl/e1_retail/260914`. The delivery's own layout and its
replacement of the single tote were not adopted; see the scene under
`env/e1_retail/fc/`.

## 2026-09-14 — PSL E4 banquet hall 260831

### round_table_fc 1.0.0 · round_table_fc_var_navy 1.0.0 · serving_station_fc 1.0.0 · utensil_bin_fc 1.0.0 · utensil_stand_fc 1.0.0 · clearing_station_fc 1.0.0
New in the pool from the first E4 delivery. The round table is Ø1220 per the
lab purchase decision of 2026-08-27, and the vendor retired the three robot
pedestals on 2026-08-28: the anchors `A_h1|A_h2|A_h3` are at `mount_z 0.000`,
the robot stands on the floor. `env/e4_hotel_fb/` is the scene; the eleven
`prop_*` items sit under `object_library/items/tableware/`.

Source: `deliveries/psl/e4_hotel_fb/260831`.

<!-- from object_library/meshes/manufacturing/CHANGELOG.md -->
# object_library/meshes/manufacturing

Newest first. A version with no entry here is not a version.

## 2026-09-15

### case_c01 1.1.0
Every material and texture the case names was referenced through the
absolute `/tmp/measure_assets/case_c01/` path of the authoring machine, so
on any other host the case rendered with fallback shading. Now
self-contained the SimReady way: `materials/` holds the five MDL modules the
looks bind, `textures/` the nine maps they sample, and every path is
relative (`./materials/`, `../textures/`).

- `Metal_Rough_A`, `Metal_Glossy_A` are Isaac's conveyor material library
  modules (copied from `object_library/nucleus_mirror/.../Conveyors/Material
  Library/Metal/Misc/`, texture paths lower-cased to `../textures/`);
  `Plastic_Black_A` is the SimReady `box_a09` module. Textures are
  byte-identical to those copies.
- `Chrome_A` and `Metal_Glossy_B` exist in no package we hold. They are
  authored here as constant-only OmniPBR stand-ins (metallic 1.0, roughness
  0.12 / 0.30) so the looks resolve; swap in the real modules if a delivery
  ever carries them. MINOR: materials changed, geometry did not.
