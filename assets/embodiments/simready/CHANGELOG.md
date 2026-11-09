# Changelog — assets/Embodiement/simready

All changes relative to the production USDs in `../rby1_wuji/` and `../rby1_wuji2_beta2/`
(both removed; the export that stayed in `../rby1_wuji2_beta2/` is `rby1m_wujihand2_beta2/source/`
since 2026-11-08). Dates are generation dates. Source repo for the converter direction:
RLWRLD/robot-assets#14.

## 2026-11-08 — one RBY1 asset: v0.3.0 and v0.4.0 removed, the export moved beside the build

`rby1m_wujihand2_beta2/` is now the only RBY1 asset: the build `v0.5.0/`, which DexBench
loads, and `source/`, the DEFT export it is made from. Every moved file keeps its bytes.
A reference to an old path breaks; v0.3.0, v0.4.0 and `../rby1_wuji2_beta2/` are on main at
f0a89ecca.

- Remove `v0.3.0` and `v0.4.0`. Nothing loads either build. A test read v0.3.0's hand
  material; it now checks the same MJCF-derived material against the MJCF and against
  v0.5.0's robot-body colliders, which still bind it. The robot constants read v0.4.0 only
  for its collision-sphere file.
- Move `v0.4.0/collision_spheres/right_axial_pinch.json` and its `.meta.json` to
  `v0.5.0/collision_spheres/`. The model keeps version 1.0.0 and SHA-256 `b0b43afc…`.
  `build_rby1_collision_spheres.py` now fits v0.5.0's hulls less the wrist-camera mounts,
  which are v0.4.0's hulls (the geometry layers are byte-identical, `90cbc54a…`). It
  reproduces the 207 spheres and the ignore list exactly. In the sidecar only the source
  USD's hash (now v0.5.0's) and the URDF path changed.
- Move `../rby1_wuji2_beta2/` to `rby1m_wujihand2_beta2/source/`: the URDF, MJCF,
  `scene.xml`, the meshes under `parts/`, the retarget URDFs under `urdf/` and
  `asset_manifest.json`. Its own version record (sidecar 2.0.0 and CHANGELOG) is folded into
  this entry, below. One asset carries one version, the build's.
- Remove `tools/appearance_overrides_v0.2.0.yaml`, the input of build v0.2.0 only (removed
  2026-11-06). The chain to v0.5.0 uses `appearance_overrides_v0.3.0.yaml` and
  `physics_overrides_v0.4.0.yaml`.
- Remove `source/parts/hands/wujihand2/metadata.yaml`, the part metadata of the beta1 hand
  (wuji-description `hand2/hand2_beta1`); the beta2 hand's is
  `source/parts/hands/wujihand2_beta2/metadata.yaml`. `asset_manifest.json` does not list
  it, and it has no `joint_limits`, so the converter's joint-limit loader skipped it. Kept:
  `source/scene.xml`, which nothing here reads but `asset_manifest.json` hashes as part of
  the export.
- v0.5.0's USD, geometry, optics and SimReady verdicts are unchanged, so the build stays
  v0.5.0. `env/scene_bundle_index.json`'s export plan is refreshed for the new paths. No
  digest changed and no scene input moved.

### History of the export folder (was `embodiments/rby1_wuji2_beta2/`)

**2026-10-06, 2.0.0.** The export's USD (`rby1m_wujihand2_beta2.usd`) is removed.
DexBench loads the robot from the SimReady build
(`embodiments/simready/rby1m_wujihand2_beta2/v0.3.0/`) since its cleanup-it1 5d67e984
(asset pin 83ab4b81), and internal PR 235 deletes the
`dexbench_rby1_wuji2_production` embodiment that still loaded this file. MAJOR because a
reference to the USD breaks. Kept: the URDF, MJCF, `scene.xml`, the meshes under `parts/`
and the retarget URDFs under `urdf/` (Pink IK, the Pinocchio model and hand retargeting
read them; the SimReady build is made from the MJCF), and `asset_manifest.json`, the
export record, whose `usd_sha256` names the removed file. The SimReady verdicts went with
the USD they described: the sidecar then recorded only the version. Recordings made with the
USD reproduce from the assets pin before this change.

**2026-09-20, 1.0.0.** First versioned record of the articulation as shipped; the SimReady
verdict lived in `rby1m_wujihand2_beta2.meta.json` (Robot-Body-Neutral and
Robot-Body-Runnable, SimReady Foundation 2026.07.1).

## 2026-11-06 — v0.1.0 and v0.2.0 removed

- Remove the `v0.1.0` and `v0.2.0` folders of `rby1m_wujihand2_beta2`: no environment, tool or
  test loads them. `v0.3.0` stays as the reference build tests compare the hand material with,
  `v0.4.0` for the collision-sphere file the robot constants read, `v0.5.0` as the default.

## 2026-10-22 — v0.5.0: wrist mount planning spheres 1.0.0

- Add 148 conservative spheres covering all 20 authored camera/mount hulls in
  their arm_6 frames, including connector and support interiors. A sphere
  encloses each complete bounding-box cell (maximum edge 12 mm).
- Reuse v0.4.0's original 207-sphere robot file instead of duplicating it. Generic
  cuRobo combines both models (355 spheres). The open-hand and moving-torso
  collision checks also include the mount model, even with RGB off.
- Keep USD geometry, contact materials, optical poses and SimReady verdicts
  unchanged. Planner spheres are conservative envelopes, not exact contact meshes.

## 2026-10-20 — v0.5.0: permanent bilateral palm GMSL2 mounts for GA

- Add the reviewed ZED X One S Wide (ZED-412012), angled FAKRA connector and compact
  palm-side wrist mount under each `link_{side}_arm_6/wrist_camera`.
- Preserve v0.4.0 rigid bodies, joints, mass and inertia; no attachment mass is added.
- Package ten offline convex meshes per wrist, each at most 64 vertices. Six use
  guide purpose and four simple supports share one render/contact surface.
- Record Robot-Body-Neutral and Robot-Body-Runnable 1.1.0 passes. Optical poses,
  nominal intrinsics and CAD provenance are in `wrist_cameras.json`.
- Use v0.5.0 as the GA physical default, including camera-free generation;
  retain v0.4.0 for prior results. This is excluded from the EA release.
  The reviewed L2 demo_0 post-state 6 still intersects the conveyor cover; this
  release does not certify existing trajectories as collision-free.

## 2026-10-16 — v0.4.0: cuRobo collision spheres 1.0.0 added (`rby1m_wujihand2_beta2/v0.4.0/collision_spheres/`); USD unchanged

A sidecar for motion planning, not a change to the robot: the USD, its geometry and its
SimReady records are untouched, so the build stays v0.4.0 and the sphere model carries its
own version (1.0.0, in `right_axial_pinch.meta.json`).

### Added

- `collision_spheres/right_axial_pinch.json`: the sphere review's final cuRobo model
  (2026-10-07), byte for byte as handed over (`rby1_selfcol_r4_m2_axial.json`, SHA-256
  `b0b43afc97ceeb5d6f4ca0e77a88701e8b2615a1dcdff2f373b1cdbd49cc5af7`): `collision_spheres`
  + `self_collision_ignore`, 207 spheres on 26 links (cuRobo's self-collision kernel takes
  < 1024), right hand in the axial pinch, left hand flat; the fingers ride on the wrist.
  DexBench declares it as the RBY1-Wuji2 embodiment's model
  (`RBY1_WUJI2_COLLISION_SPHERES`, SHA-256 checked on load), and its planners default to it.
- `collision_spheres/right_axial_pinch.meta.json`: provenance. DexBench
  `scripts/tools/build_rby1_collision_spheres.py` fits this build's collision hulls and
  reproduces the model's spheres and ignore list exactly (v0.4.0's hulls are v0.3.0's, which
  the review fitted); the sidecar records that fit: the SHA-256 of the USD and geometry, the
  finger joints, the base link's box (the base carries no spheres; footprint checks use the
  box) and the per-link fit. Its top-level key is `collision_spheres`, not `asset`: its
  version is the model's, not the build folder's.
- Ignore list: links within two joints, the torso/base/head among themselves, left vs right
  hand (83 ignored pairs, 242 checked).
- Fit (hull surface outside every sphere = uncovered, sphere surface outside the link's
  convex envelope = stick-out): hands p95 uncovered 3.4-3.7 mm, max 18-19 mm (finger webs),
  stick-out max 7.6 mm; arm links stick out 24-45 mm past their hulls (one to four balls per
  link, `arm_3` worst) and leave up to 50 mm of `arm_2`'s elbow housing uncovered; the torso
  is underfit on purpose (inscribed spheres, uncovered max 142 mm on `link_torso_1`, 78 mm on
  `link_torso_5`) so it does not bulge into arms that hang 2-3 cm beside it -- arm-torso
  contact shallower than that can go unseen.

## 2026-10-12 — v0.4.0: the hand carries the grasp contact material (folder `rby1m_wujihand2_beta2/v0.4.0/`)

Physics materials only, via `tools/physics_overrides_v0.4.0.yaml` (new converter option
`--physics-overrides`); geometry, joints, drives, colliders and visual materials are those
of v0.3.0 (the geometry layer differs from v0.3.0 only in the lines below; regenerating
v0.3.0 with the same converter reproduces it byte for byte).

### Why

Up to v0.3.0 every collider bound its body's MJCF-derived material, MuJoCo's default
sliding friction 1.0 written as static = dynamic = 1.0. DexBench put its grasp
material on the hand at run time instead, with a prestartup event that bound a new material
under `/World/Looks`. That event only reaches Newton when Isaac Lab builds the Newton model
after it runs, which DexBench forced by turning physics replication off; a loader that
replicates (the M1 rig, Isaac Lab's default on Newton) builds the model at clone time and
loaded the fingers at 1.0 (internal issue 254). The material now ships with the asset.

### Changed

- **Hand colliders bind `physics_materials/hand_contact`**: static 6.0, dynamic 5.0,
  restitution 0, `physxMaterial:frictionCombineMode = max` -- DexBench's grasp baseline
  (`WUJI_GRASP_CONTACT_MATERIAL`). Bodies: both palms (`<side>_wrist`), every finger link and
  the ten fingertip sensor pads (`<side>_<finger>_tip_sensor_frame`): 52 bodies, 56
  colliders. Each collider binds it directly (`material:binding:physics` on the collider
  prim): OpenUSD's physics parser, which Newton reads, takes a collider's material only from
  the collider prim itself.
- **Removed** the `physics_materials/friction_0` prim of those 52 bodies' collider prototypes
  (`hand_contact` replaces it). The other 29 bodies keep `friction_0` (1.0 / 1.0, max).
  In a 0.y version the minor number is the breaking one, hence 0.4.0 rather than a 1.0.0;
  nothing outside the build referenced the removed prims.
- `isaac:version = 0.4.0`.

### Validation

SimReady Robot-Body-Neutral and Robot-Body-Runnable v1.1.0 PASS (simready-validate 2026.7.1,
every feature of both profiles, same result as v0.3.0 in the same run). Thumbnail re-rendered
from the MJCF (`tools/make_thumbnail.py`), same picture.

## 2026-10-06 — `rby1m_wujihand` (v1 hand) builds removed; no version change to `rby1m_wujihand2_beta2`

DexBench deleted the Wuji Hand 1 embodiment (internal PR 232) and never wired
the v1 build in, so `rby1m_wujihand/` (v0.1.0–v0.3.0 and its MJCF copy) goes. The
production USDs the entries below compare against are removed in the same change
(`../rby1_wuji/`, `../rby1_wuji2_beta2/rby1m_wujihand2_beta2.usd`); the comparisons stay
as written, and the assets pin before this change still has both sides.
`rby1m_wujihand2_beta2/` is untouched and is now the only RBY1 the benchmark loads.

## 2026-09-06 — v0.3.0: anodized-aluminium look for the RBY1 (folders `<name>/v0.3.0/`)

Visual materials only, via `tools/appearance_overrides_v0.3.0.yaml`; physics, geometry
and joints identical to v0.1.0/v0.2.0. Requested: white parts and the non-hand black parts
should read as anodized aluminium.

### Changed

- **Clear-anodized aluminium** on the RBY1 light shells (CAD colours 244/254/255/242/216/
  192/191/190): base colour (0.80, 0.81, 0.83), metallic 0.85, roughness 0.40. Clear
  anodizing reads as satin silver rather than paper white, hence the darker base colour.
- **Black-anodized aluminium** on the RBY1 dark structural parts (CAD colours 0/10/25/65/
  69/70/79/80/90) and on the wrist link `link_{left,right}_arm_6`: base colour
  (0.05, 0.05, 0.055), metallic 0.85, roughness 0.40.
- **Hand unchanged**: Wuji glove links, in-house mount geoms and (beta2) fingertip pads stay
  matte BlackGlove (0.02, metallic 0, roughness 0.7).
- Everything else (mid greys, red button, yellow-green marker, ZED body) keeps its colour
  with the roughness floor 0.5 from v0.2.0.
- Converter: rules may now set `metallic`; a rule's explicit `roughness` wins over
  `min_roughness` (the floor only applies to unmatched materials).
- Materials are still OmniPBR + UsdPreviewSurface constants, not the vMaterials
  `Aluminum_Anodized` MDL: that library is served from NVIDIA's S3 content server, which
  breaks offline loading and the SimReady anchored-path rule (AA.001).
- `isaac:version = 0.3.0`.

### Validation

Both v0.3.0 assets: SimReady Robot-Body-Neutral and Robot-Body-Runnable v1.1.0 PASS, MJCF
parity PASS, headless RTX render checked.

## 2026-09-06 — v0.2.0: visual fixes (folders `<name>/v0.2.0/`)

Physics, geometry, joints and colliders are byte-for-byte the same policy as v0.1.0; only
visual materials changed, via `tools/appearance_overrides_v0.2.0.yaml`
(`--appearance-overrides`). The MJCF sources are untouched.

### Changed

- **Roughness floor 0.5** on every visual material. RBY1 vendor CAD materials with
  `shininess=1` (pure black `0,0,0`, `10,10,10`, the whites `244,244,244` /
  `254,254,255`, greys `65/69/70/90`) mapped to roughness 0 and rendered as mirrors.
  Reference: the vendor's own Isaac USD (rby1-sim-isaac `model_v_1_2_m.usd`) authors
  only diffuse colors, so it renders at OmniPBR's default roughness 0.5. 70 geoms per
  robot affected; materials already rougher than 0.5 (hand glove 0.7, 0.2-shininess CAD
  parts at 0.8) unchanged.
- **Wrist link `link_{left,right}_arm_6` is black** (BlackGlove 0.02, roughness 0.7)
  instead of a near-white shell (`244,244,244`) plus dark ring (`69,69,69`), so the wrist
  reads as one part with the hand mount. 12 geoms per robot.
- **Wuji Hand 2 Beta 2 fingertip tactile pads are black** (BlackGlove) instead of the
  vendor's light blue `(0.55, 0.78, 0.95)` (`*_sensor_pad` material), matching the other
  hand links. 20 geoms (beta2 only; v1 has no pads).
- `isaac:version = 0.2.0` on the default prim.

### Added

- Folder versioning: the first build moved from `<name>/usd/` to `<name>/v0.1.0/`
  (same files); v0.2.0 is a new folder. The converter gained `--subdir` and
  `--appearance-overrides`; `tools/view_in_isaac.py` defaults to the latest version.

### Validation

Both v0.2.0 assets: SimReady Robot-Body-Neutral and Robot-Body-Runnable v1.1.0 PASS,
MJCF parity PASS, headless RTX render shows the robot with the new materials.
v0.1.0 (moved) still passes Robot-Body-Runnable at its new path.

## 2026-09-06 — v0.1.0: first SimReady build of `rby1m_wujihand2_beta2` and `rby1m_wujihand`

### Why

The production USDs fail the NVIDIA SimReady Foundation validator (Robot-Body-Neutral:
missing `extent`, coincident visual/collision meshes; Robot-Body-Runnable: no Isaac
robot schema, no `isaac:robotType`). The vendor Wuji USD fails too, so the decision was
to make SimReady the truth for USD generation and re-generate from the MJCF with a
SimReady-targeted converter (`tools/mjcf_to_usd_simready.py`).

### Result

Both assets render in Isaac Sim RTX (headless render check `tools/render_check.py`), pass **Robot-Body-Neutral v1.1.0** and **Robot-Body-Runnable v1.1.0**, pass
element-wise MJCF parity (`tools/verify_usd_simready.py`), load in Isaac Lab as a
fixed-base articulation (81/66 and 73/66 bodies/joints) with no NaN and no GPU convex
cooking warnings.

### Added

- **Folder layout** `<name>/usd/<name>.usd` (now `<name>/v0.1.0/`) + `.thumbs/256x256/<name>.usd.png`
  (RC.003, RC.004, AA.001). Thumbnails are software-rendered from the MJCF
  (`tools/make_thumbnail.py`), no OpenGL needed.
- **`extent`** on every Gprim (VG.002).
- **Geometry layer** `<version>/geometry/<name>_geometry.usd` (Isaac Sim instanceable-asset
  layout): shared mesh data once per MuJoCo mesh asset under `/<name>_geometry/meshes/`,
  per-body prototypes `/visuals/<body>` and `/colliders/<body>` whose `geom_N` children
  reference the shared data internally and carry pose, displayColor, purpose, materials
  and CollisionAPI. In the robot file, `<body>/visuals` and `<body>/collisions` are
  `instanceable` Xforms referencing those prototypes with an anchored `./geometry/...`
  path (AA.001). Visual and physics materials are authored inside each prototype, because
  relationship targets outside the referenced subtree do not survive composition.
  (A first attempt made each Mesh prim itself instanceable via internal references to
  `class` prims: PhysX read it, the validator passed, but nothing rendered, since USD
  shares only the *descendants* of an instanceable prim and Hydra draws the prototype.)
- **Isaac robot schema** (RC.007/RC.008/RC.009, DJ.008–010): `IsaacRobotAPI` on the default
  prim with `isaac:robotType = Manipulator`, `isaac:namespace`, `isaac:description`,
  `isaac:license`, `isaac:source`, `isaac:version`, breadth-first
  `isaac:physics:robotLinks` and matching `isaac:physics:robotJoints` (first entry is the
  world fixed joint `joints/base_fixed`); `IsaacLinkAPI` on every link; `IsaacJointAPI`
  + `isaac:physics:DofOffsetOpOrder = [RotX]` on every joint.
- **`kind = component`** on the default prim.
- **Drive `maxForce`** (DJ.001) on all 62 driven joints: MJCF actuator `forcerange`,
  else joint `actuatorfrcrange`, else part metadata `joint_limits.effort`
  (RBY1 arm/torso/head from robot-assets `parts/arms/rby1m/metadata.yaml`).
- **`physxJoint:maxJointVelocity`** (DJ.005) on all 62 driven joints. Hand values come
  from the Wuji vendor URDFs (`tools/wuji_hand_joint_limits.yaml`, 8.1–13.6 rad/s), arm
  values from robot-assets metadata. Wheels (velocity actuators, no drive) carry neither
  limit.
- **`PhysicsJointStateAPI:angular`** with `state:angular:physics:position` = qpos0 (deg)
  and velocity 0 on every joint (DJ.002); `drive:angular:physics:targetPosition` = qpos0.
- **`PhysxJointAPI`** applied on every joint (armature where MJCF has it).
- **Visual materials** as UsdPreviewSurface **and** OmniPBR (MDL) shaders on one
  `Material` (`outputs:surface` + `outputs:mdl:surface`), 14 distinct appearances
  deduplicated from MJCF color/shininess. No textures: the MJCF carries none for these
  robots (the Wuji wrist logo needs UV-bearing meshes the vendor STLs do not have).

### Changed

- **Collision meshes are offline convex hulls with ≤ 64 vertices** (PhysX GPU convex
  limit) instead of the full visual mesh with `convexHull` approximation left to the
  runtime cooker.
  - Reduction: greedy max-error vertex insertion starting from the six axis extremes;
    the reported "inset" is the largest distance any true-hull vertex lies outside the
    reduced hull. Worst case 3.66 mm (205 mm base shell piece); every hand link ≤ 0.6 mm;
    16 of 574 hulls > 1 mm, none > 5 mm.
  - Near-flat pieces are padded to a 2 mm minimum thickness along their thin axis (two
    0.07 mm plates on `link_torso_5`; one of them previously failed GPU cooking and fell
    back to CPU collision).
  - Collider vertex data: 309,966 → 29,296 (beta2), 192,438 → 29,025 (v1).
  - The vendor's per-piece convex decomposition is preserved (e.g. 221 pieces on
    `link_torso_5`); no re-decomposition.
  - Visual and collision meshes therefore never coincide (VG.008 resolved); colliders keep
    `purpose = guide` where MuJoCo hides them.
- **Articulation root** is on the base's world fixed joint (`joints/base_fixed`), as in
  robot-assets since PR #9. The production beta2 USD had it on the `base` body.
- **Geom prim paths unchanged** (`<body>/visuals/geom_N`, `<body>/collisions/geom_N`),
  but they are now instance proxies (read-only on the stage; edit the geometry layer or
  the instance Xform instead). Two files per asset instead of one.

### Unchanged (deliberately)

- Body tree, joint names/axes/limits, masses, inertias, sites, contact excludes
  (FilteredPairs), drive stiffness/damping (MJCF values; DexBench overrides them via
  `ImplicitActuatorCfg`), friction physics material, epsilon mass/inertia for massless
  frame bodies, articulation self-collision OFF.
- The `base` world weld equality (`<weld>`) is still skipped by the converter (the base is
  jointless, so the fixed joint already pins it).

### Known gaps / not done

- **Robot-Body-Isaac** profile (physics in a separate layer, `_base.usd` payload,
  clean-folder rule) is not targeted; it conflicts with the single-file principle.
- `dexbench_arena/embodiments/rby1_wuji2/spawn.py` cannot load these files: it expects the
  articulation root on `base`, a fixed collider count (198/34), and edits collision prims
  that are now read-only instance proxies. Spawn with plain `spawn_from_usd` for now.
- Wheels have no drive/limits in the USD (MJCF velocity actuators are not mapped).
- Drive gains are the MJCF values, not the vendor's PhysX-tuned USD gains; hands sag
  ~2.7 deg under gravity when the USD drives are used unmodified.
- The converter still imports helpers from a robot-assets checkout
  (`../robot-assets/tools` relative to the script); upstreaming to robot-assets is the
  next step.
- `robotType = Manipulator` (not `Mobile Manipulators`) because the MJCF welds the base;
  RC.009 requires a pinned root for that type.

### Provenance

| Asset | MJCF | Parts metadata |
|---|---|---|
| rby1m_wujihand2_beta2 | `../rby1_wuji2_beta2/rby1m_wujihand2_beta2.xml` (robot-assets build of config `rby1m_wujihand2_beta2`, rev `2c7d933` dirty) | `../rby1_wuji2_beta2/parts/**/metadata.yaml` + robot-assets `ff62df1` `parts/arms/rby1m/metadata.yaml` + `tools/wuji_hand_joint_limits.yaml` |
| rby1m_wujihand | robot-assets `ff62df1` build of config `rby1m_wujihand` (copy in `rby1m_wujihand/rby1m_wujihand.mjcf.xml.txt`) | robot-assets `ff62df1` `parts/**/metadata.yaml` + `tools/wuji_hand_joint_limits.yaml` |

Tooling: mujoco 3.11.0, usd-core 26.8, scipy 1.17, SimReady Foundation validator
(`simready-validate`, `omniverse-usd-profiles`, spec repo NVIDIA/simready-foundation),
Isaac Lab from the project `.venv`.
