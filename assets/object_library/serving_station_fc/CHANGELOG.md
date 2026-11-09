# serving_station_fc

Newest first. A version with no entry here is not a version.

## 2026-10-04 — 1.2.0

- Re-sourced from PSL's E4 r01 delivery (`deliveries/psl/e4_hotel_fb/260927`). The cart is r01's: satin brushed-stainless finish (`MAT_E4_fx_brushed_serving`, PSL's 4th-edition look fix) and the dressing r01 gives it, `Deco/BG_food` -- two red velvet cakes and two plates on the lowest shelf, visual-only, no collider or body. The slot payload in r01 composes the cart with that dressing; here the two layers are flattened into the one package layer. Cart geometry, colliders and prim paths are unchanged.
- Base frame unchanged against the previous version: same default prim, no root transform; every previous prim path keeps its point count and the collider bounds are identical (`compare_usd_geometry.py`, 2026-10-02); `Deco` is the only new prim.
- Textures replaced by the ones the delivered layer uses (11 in `usd/textures/`); the 6 previous ones are removed.
- MINOR: a new look (materials) or an addition; nothing a placement depends on moved.
- SimReady validated for this version with simready-validate 2026.7.1 on 2026-10-04: Robotics-Prop 3.1.0 **pass** (scope static).
- Physics runtime variants `Newton` on the default prim (SimReady RV.001-RV.011), `Disabled` by default; `Enabled` composes `usd/runnables/physics/newton.usd`. The Newton payload annotates the colliders (`NewtonCollisionAPI`, `NewtonMeshCollisionAPI`) and the physics materials (`NewtonMaterialAPI`); contact gap and margin stay the engine's.

## 2026-09-20 — 1.1.0

- Physics runtime variants `Newton` on the default prim (SimReady RV.001-RV.011), `Disabled` by default; `Enabled` composes `usd/runnables/physics/newton.usd`. The Newton payload annotates the colliders (`NewtonCollisionAPI`, `NewtonMeshCollisionAPI`) and the physics materials (`NewtonMaterialAPI`); contact gap and margin stay the engine's.

## 2026-09-14 — 1.0.1

- SimReady Core metadata (`SimReady_Metadata`, `asset_name`, `asset_type`, `source_file`, `usd_date_generated`) stamped into the root layer's customLayerData from the sidecar (NP.006/SR.001).

## 2026-09-15 — packaged

Moved to `object_library/serving_station_fc/usd/serving_station_fc.usd` from `object_library/fixtures/serving_station_fc.usdc`. No content change.

## 2026-09-14 — PSL E4 banquet hall 260831

### round_table_fc 1.0.0 · round_table_fc_var_navy 1.0.0 · serving_station_fc 1.0.0 · utensil_bin_fc 1.0.0 · utensil_stand_fc 1.0.0 · clearing_station_fc 1.0.0
New in the pool from the first E4 delivery. The round table is Ø1220 per the
lab purchase decision of 2026-08-27, and the vendor retired the three robot
pedestals on 2026-08-28: the anchors `A_h1|A_h2|A_h3` are at `mount_z 0.000`,
the robot stands on the floor. `env/e4_hotel_fb/` is the scene; the eleven
`prop_*` items sit under `object_library/items/tableware/`.

Source: `deliveries/psl/e4_hotel_fb/260831`.
