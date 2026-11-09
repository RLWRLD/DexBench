# Changelog

## 2.0.0

LG asked that no part number stay in the assets we use. LG's CAD export names every part `T_<Teamcenter id>_<part number>_<Description>_3DAsset__refset_MODEL[_n]`, and nests connector housings as `LIB_<library>_<maker's part number>_<Description>`. Every prim whose name carried one is renamed to its description and instance numbering (`Screw_Customized_1_005`, `Housing_001`); siblings that would then collide take `_2`, `_3`. MAJOR: 1,118 prim names change under `Fridge_M3_Root`.

- `userProperties:blender:object_name`, the exporter's copy of the CAD name, is removed from the 1,118 prims that carried it.
- `manifest.json`: the bolts' `part_number` field is gone, and `source_prim`, `excluded_internal_bolts` and `deactivated_duplicate_meshes` name the prims of this package (the source CAD's own paths carry the numbers). The bolt packages `b1`..`b6` carried none and are unchanged.
- `center_hinge_station.py` finds the freezer door assemblies and the parts it cuts tool access channels into by their new names (it matched part numbers before); the same prims are selected.
- Nothing else changes. The composed stage (the package has no physics runtime variant) compares equal to 1.0.0 prim by prim, in order, apart from the names: types, applied schemas, every attribute value (points, masses, inertia, collider geometry and approximation, materials), relationship targets and prim metadata (7,618 prims, 2,982 colliders).
- SimReady re-validated: Robotics-Prop 3.1.0 with simready-validate 2026.7.1, static scope passes; every feature outcome is the same as before.
- Kept: the package name and the LG model code in it, and every non-CAD prim name (task frames, colliders the rig authored, looks).

## 1.0.0

- Adopt the M3 open-door CAD as a static fixture with collision, six separate kinematic bolts and measured head sites. Keep the four internal hinge bolts outside the task. Preserve source geometry; use a distance/motor-turn surrogate rather than thread or torque physics.
