# SPDX-License-Identifier: Apache-2.0
"""The deformable-asset loader the SpaceAI polybag was verified with, vendored unchanged.

Source: simready-foundation 2f32655cd41e80a218f533b917a08b5eb8e519ca (branch
asset_checks_cleanup_20260929), ``rlwrld/asset_checks/asset_checks/native/{loader,newton_scene,
usd_deformable,asset_properties,setups,stepping,drop_shape}.py``, as shipped in the 2026-09-29
deformable-asset delivery's ``loader/`` (Apache-2.0). The only edits are the imports between these
modules, made relative so they import as a package. Re-sync by copying the files again and
re-applying that one change.

Newton's USD importer and Isaac Lab's ``DeformableObject`` read the meshes and the moduli of these
assets and silently drop the rest: per-body particle radii, the film's damping, the seal springs,
the fold contact exclusions, the self-contact radius and margin, and the contact numbers the asset
states for itself (its README, "loader/"). This loader carries all of them and refuses an asset
that authors anything it did not read, so it is the one way the polybag is built here.
"""
