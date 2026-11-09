# DexBench assets

This is the ordinary `assets/` directory of the repository. Small text and LFS
pointer blobs live in the parent Git repository; payload bytes remain on the
Hugging Face dataset that `.lfsconfig` names. The parent commit and assets tree OID
are the version. No assets lock or separate snapshot pin is maintained.

The [asset rights guide](Asset_Rights.md), [per-path index](Asset_Rights_Index.md)
and [third-party license texts](THIRD_PARTY_LICENSES.md) accompany this tree when
read independently of the code. The [license statement](ASSET_LIST.md#license)
sets CC BY 4.0 as the default and identifies component exceptions. Audit findings
and generator review inputs remain in the source repository's internal documentation.

## Source and storage

Vendor deliveries remain byte-for-byte in the unfiltered HF source history and
an operator-owned archive outside this repository. Never edit, rename or partially
delete them, and never load a simulator path from deliveries. Sidecar
`source_delivery` values keep their historical logical archive addresses.
Only adopted/generated packages enter the shipped assets tree.

Binary wildcard formats use LFS regardless of size. Other payloads over
1,000,000 bytes use LFS; small text stays in Git. Reviewed large-text paths are
LFS across all historical versions, recorded in the parent trusted policy and
this subtree's attributes. Preserve original bytes: disable text normalization.
Control files remain ordinary Git. Do not weaken binary rules or add redundant
path rules for an extension already covered by a wildcard. For a new large text
path, add an exact path rule and request policy review for all-version treatment.

Pull from the parent repository root with `git lfs pull --include="assets/**"`.
Partial pulls use parent-relative `assets/...` include paths. The parent
`.lfsconfig` declares the existing HF LFS endpoint. Git/LFS operations run on the
host as the normal user. Generated output goes outside assets, under outputs or
datasets. Never leave temporary or untracked files in the mounted payload.

## Packages and conformance

| Path | Holds |
| --- | --- |
| `object_library/<name>/usd/<name>.usd` | One shipped package with sidecar, textures, materials and dependencies; CHANGELOG at the package root |
| `embodiments/` | Robots; SimReady builds retain `simready/<name>/<version>/` folders |
| `env/<environment>/<layout>/` | Composition layers and layout-specific room shells; handled objects reference the pool |
| `object_library/nucleus_mirror/` | Local mirror of NVIDIA content with its original paths (none at present) |

Paths are stable addresses. Use lowercase snake_case; no dates, task codes,
spaces or vendor names outside retained vendor/Nucleus internal package layouts.
Reclassification belongs in sidecar categories and the generated catalog.

Every shipped package needs a version, CHANGELOG entry and recorded SimReady
verdict with profile/spec version and scope. Props take `Robotics-Prop` 3.1.0;
robots take `Robot-Body-Neutral` 1.1.0 and `Robot-Body-Runnable` 1.1.0 when
simulated. Handled props also require the reviewed grasp line, materials and
Newton feature scope (`FET_003_NEWTON`; DexBench runs on Newton only, internal issue 313); fixture and
deformable scopes follow the parent guideline. Known gaps remain in
`scripts/tools/assets_lint_allowlist.json` with reasons; delete an entry when fixed.
Bump MAJOR for geometry/pivot/units/prim-path changes, MINOR for additions and
PATCH for changes a reference cannot observe. Versions stay out of paths except
for the existing SimReady robot folder convention.

Packages resolve independently with local dependencies, textures and materials.
No absolute path or remote URL in a USD. Author composition layers as `.usda`;
main crate files remain `.usd`. A handled prop carries a reviewed grasp line
and its simulation evidence. The parent asset pipeline guideline and existing
sidecar certificates define validation scopes and backend-specific variants.

Physics runtime variants use the `PhysX` and `Newton` variant sets, disabled by
default, with engine payloads under `usd/runnables/physics/`. DB-M5's fifteen
parts also offer a Newton collider choice, `NewtonColliders`, nested under
`Newton`'s `Enabled`: `pieces` (the default) is the Newton payload as it is,
`primitives` swaps in dbclutter's fitted collision primitives (`_prim5`,
`newton_primitives.usd`, recipe and source hashes in
`newton_primitives.preparation.json`). The package base
carries neutral physics; consumers select their engine. DexBench selects `Newton`
only; the `PhysX` payloads go in their own PR (internal issue 313). Do not restore the retired
flattened `usd/variants/*_newton.usd` layers. See the parent's
`docs/assets/Physics_Runtime_Variants.md` and `scripts/tools/author_physics_variants.py`.

## Commit and preserve

Work on a parent repository branch. Validate with the existing SimReady tools,
record the verdict, regenerate the catalog and run `python3 scripts/tools/lint_assets.py`.
Commit the changed packages and pointer blobs in the parent PR. The local
pre-push hook validates all new pointer trees and uploads payload to HF. The
trusted `assets / preserved` gate audits actual size/SHA-256 and preserves every
PR tree, including intermediate additions later reverted, on HF github-mirror.
After the history-preserving first import, normal merge, squash and rebase are
supported. Parent commit and assets tree OIDs remain the authoritative version.

## Rebuild the integrated scenes

The scene index records the source and output SHA-256 hashes. An edit to an
indexed source, texture, generated look or view fails ordinary asset lint until
the four packages and their fingerprints are rebuilt. Keep the original source
files while internal PRs 505 and 542 are pending. Do not edit a generated view or archive
in isolation, or waive the drift check.

Use the current DexBench extension on PYTHONPATH and install its benchmark
inputs. Run these maintained tools under `/isaac-sim/python.sh` from the repo
root. The refresh reuses `asset_closure.Builds` to construct current typed Train and held-out-object Runs;
it does not start Kit or physics. It keeps `scope.changed_from` as the baseline
before the replacement work and `scope.replacement_review_head` as its reviewed
upper bound. Pass `--changed-to <reviewed-asset-commit>` for a new replacement
batch. Geometry and catalog paths are always read from the current checkout;
Scene roots and explicitly indexed fixture packages enter the bundles. Task-owned numerical templates remain
external and are still listed as used inputs in the usage receipt; do not
delete them as unused files.

```bash
/isaac-sim/python.sh scripts/tools/refresh_scene_bundle_index.py \
  --output outputs/scene_bundle_index.json
/isaac-sim/python.sh scripts/tools/bundle_usd_scene.py \
  --runtime-index outputs/scene_bundle_index.json --version 2.1.1 \
  --search-path assets/env/e4_hotel_fb/banquet/shell/background/materials/OmniGlass \
  --output outputs/rebuilt_scenes
```

Choose new output paths and the next package version for each rebuild. Both
commands leave assets read-only. The refresh also writes a `.usage.json` receipt.
On the host, as the normal user, adopt only `env/scene_bundle_index.json` and the
four `integrated/` directories from the completed staging tree. Run
`python3 scripts/tools/lint_assets.py` and, with Kit and hydrated assets,
`/isaac-sim/python.sh scripts/tools/validate_scene_bundle_runtime.py --output
outputs/scene_validation.json --check-e2-tests`. Commit the packages, index and
validation record together. No internal script is a build input.

The drift guard records the catalog's complete fixture package set.
The three explicitly indexed E2 fixtures (`packing_bench_fc`, `parcel_cage_fc`,
`sort_tote_single_fc`) and their textures are embedded. Other fixtures remain
external. Keep these entries when refreshing the scene index. Unrelated catalog
metadata and new handled-model entries do not require archive rebuilds. Changing
fixture classification still fails the guard. Payload and view fingerprints stay
strict; no timestamp cache can hide an in-place source edit. Adding a new model
to the bundled replacement scope remains an explicit refresh/rebuild decision.

Three E1 views contain a copy of `OmniGlass.mdl` in place of the bare Kit core
module used by their sources. It is byte-identical in the qualified image. The
maintained Kit validator checks the installed core module before it binds the
source comparison's search context. An Isaac Sim bump must pass that check and
new rendering qualification; asset-only fingerprints cannot certify SDK bytes.
The pointer-only PR lint also checks the external export plan against committed
asset paths and Git/LFS object identities. Adding or editing a retained robot or
object can change that plan even when no scene archive needs rebuilding. With
hydrated internal assets, refresh only the omission/dependency plan without Kit:

```bash
/isaac-sim/python.sh scripts/tools/scene_export_plan.py --output outputs/scene_export_index.json
# Adopt on the host as the normal user, then commit with the asset change:
cp outputs/scene_export_index.json assets/env/scene_bundle_index.json
```

This command preserves every scene input/output fingerprint and archive byte.
It rejects changed scene inputs/outputs, which still require the rebuild above,
and rejects external snapshots whose packaged source files are absent.
