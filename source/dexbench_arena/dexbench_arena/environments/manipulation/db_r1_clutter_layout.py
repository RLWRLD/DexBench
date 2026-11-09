# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Build variant instances from an R1 recipe, using existing registered assets."""

from dexbench_arena.tasks.r1_clutter import layout_set, load_recipe


def fruit_geometry(asset_registry, recipe):
    from pxr import Usd, UsdPhysics

    geometry = {}
    for row in recipe['objects']:
        key = (row['asset'], row['scale'])
        if key not in geometry:
            probe = asset_registry.get_asset_by_name(row['asset'])(
                instance_name='_r1_probe', scale=(row['scale'],) * 3)
            bbox = probe.get_bounding_box()
            center = tuple(float(v) for v in bbox.center[0].tolist())
            half = tuple(float(v) for v in ((bbox.max_point[0] - bbox.min_point[0]) * .5).tolist())
            stage = Usd.Stage.Open(probe.usd_path)
            masses = [UsdPhysics.MassAPI(p).GetMassAttr().Get() for p in stage.Traverse() if p.HasAPI(UsdPhysics.MassAPI)]
            masses = [m for m in masses if m is not None and m > 0]
            if len(masses) != 1 and 'mass_kg' not in row:
                raise ValueError(f"{row['asset']} needs one authored mass or an explicit mass_kg per variant")
            geometry[key] = (center, half, masses[0] * row['scale'] ** 3 if len(masses) == 1 else row['mass_kg'])
    return [{**row, 'center_local': geometry[row['asset'], row['scale']][0],
             'half_extents': geometry[row['asset'], row['scale']][1],
             'mass_kg': row.get('mass_kg', geometry[row['asset'], row['scale']][2])}
            for row in recipe['objects']]


def build_r1_clutter_assets(asset_registry, recipe_file, fruit_id, layouts=None):
    """The recipe's fruit at its poses; with ``layouts`` (a layout set's directory, whose first layout
    ``recipe_file`` is: ``r1_clutter.r1_recipe_file``) the first fruit carries the per-reset layout draw."""
    import isaaclab.sim as sim_utils
    from isaaclab_arena.utils.pose import Pose

    recipe = load_recipe(recipe_file, fruit_id)
    assets = []
    for row in fruit_geometry(asset_registry, recipe):
        asset = asset_registry.get_asset_by_name(row['asset'])(
            instance_name=row['name'], scale=(row['scale'],) * 3,
            initial_pose=Pose(position_xyz=tuple(row['position_xyz']), rotation_xyzw=tuple(row['rotation_xyzw'])))
        asset.object_cfg.spawn.mass_props = sim_utils.MassPropertiesCfg(mass=row['mass_kg'])
        assets.append(asset)
    if layouts is not None:
        from dexbench_arena.variations.db_r1_layout_variation import R1LayoutVariation

        assets[0].add_variation(R1LayoutVariation(layout_set(layouts, fruit_id), assets[0].name))
    return assets
