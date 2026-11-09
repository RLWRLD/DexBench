# Attribution and changes

The current `beverage_pet_500ml` geometry, convex collision geometry and
`usd/materials/forgecad_glass.mdl` and `usd/textures/water-label.png` derive from **ForgeCAD**'s
[screw_cap_bottle](https://huggingface.co/datasets/ForgeCAD/simready-assets/tree/0573053fa2c9174ca2a4678a462ec4dfd3035d59/screw_cap_bottle) at commit `0573053fa2c9174ca2a4678a462ec4dfd3035d59`.
The pinned repository README and main USD declare
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).

Attribution: **ForgeCAD, CC BY 4.0; modified by RLWRLD.**

RLWRLD scaled the bottle to the EP2 envelope, revolved simplified measured source
profiles for the body and liquid (including annular rather than helical hidden
threads), reduced cap geometry with QEM,
retained the original water-label artwork, mesh/UVs and selected material graphs,
fixed the cap/liner/tamper band in the
closed pose, combined the assembly into one rigid body, adapted mass/inertia,
and added grasp metadata and a Newton runtime payload.
See `CHANGELOG.md` and `usd/beverage_pet_500ml.meta.json` for the recorded changes.

Earlier versions used a different supplier and bottle shape. Their attribution
and third-party exclusions remain historical records in Git and the changelog;
the earlier geometry, textures and MDL modules are not included in this version.
The historical USD body name is retained only as a recording key.

CC BY 4.0 does not grant trademark rights or imply endorsement.
