# round_table_fc

Newest first. A version with no entry here is not a version.

## 2026-10-04 — 2.0.0

- Re-sourced from PSL's E4 r01 delivery (`deliveries/psl/e4_hotel_fb/260927`). The tablecloth mesh `TbFC_Tablecloth_visonly_001` moved: its top now lies flat 0.2 mm over the tabletop (PSL's 4th-edition fix, kept in r01; visual top z 0.7796 -> 0.7402, XY outline 1.278 x 1.267 -> 1.254 x 1.254 m). It stays visual-only and the tabletop collider is unchanged, so anything placed on the table still rests where it did.
- Base frame unchanged against the previous version: same default prim, no root transform, the same prim paths, point counts and collider bounds (`compare_usd_geometry.py` and a frame/bounds comparison, 2026-10-02).
- MAJOR because a mesh moved; placements and colliders a scene relies on are unchanged.
- Textures replaced by the ones the delivered layer uses (6 in `usd/textures/`); the 6 previous ones are removed.
- SimReady validated for this version with simready-validate 2026.7.1 on 2026-10-04: Robotics-Prop 3.1.0 **pass** (scope static).
- Physics runtime variants `Newton` on the default prim (SimReady RV.001-RV.011), `Disabled` by default; `Enabled` composes `usd/runnables/physics/newton.usd`. The Newton payload annotates the colliders (`NewtonCollisionAPI`, `NewtonMeshCollisionAPI`) and the physics materials (`NewtonMaterialAPI`); contact gap and margin stay the engine's.

## 2026-09-20 — 1.1.0

- Physics runtime variants `Newton` on the default prim (SimReady RV.001-RV.011), `Disabled` by default; `Enabled` composes `usd/runnables/physics/newton.usd`. The Newton payload annotates the colliders (`NewtonCollisionAPI`, `NewtonMeshCollisionAPI`) and the physics materials (`NewtonMaterialAPI`); contact gap and margin stay the engine's.

## 2026-09-14 — 1.0.1

- SimReady Core metadata (`SimReady_Metadata`, `asset_name`, `asset_type`, `source_file`, `usd_date_generated`) stamped into the root layer's customLayerData from the sidecar (NP.006/SR.001).

## 2026-09-15 — packaged

Moved to `object_library/round_table_fc/usd/round_table_fc.usd` from `object_library/fixtures/round_table_fc.usdc`. No content change.

## 2026-09-14 — PSL E4 banquet hall 260831

### round_table_fc 1.0.0 · round_table_fc_var_navy 1.0.0 · serving_station_fc 1.0.0 · utensil_bin_fc 1.0.0 · utensil_stand_fc 1.0.0 · clearing_station_fc 1.0.0
New in the pool from the first E4 delivery. The round table is Ø1220 per the
lab purchase decision of 2026-08-27, and the vendor retired the three robot
pedestals on 2026-08-28: the anchors `A_h1|A_h2|A_h3` are at `mount_z 0.000`,
the robot stands on the floor. `env/e4_hotel_fb/` is the scene; the eleven
`prop_*` items sit under `object_library/items/tableware/`.

Source: `deliveries/psl/e4_hotel_fb/260831`.
