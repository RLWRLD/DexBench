# 0.5.0

Add the reviewed bilateral palm GMSL2 camera and compact mount for GA. Preserve v0.4.0 joints, bodies, mass and inertia. Author ten precomputed convex colliders per wrist (at most 64 vertices). Six are guide geometry; four simple supports share one render/contact mesh to avoid coincident duplicates. Camera-body/connector envelopes retain the reviewed 0.5 mm padding. L2 demo_0 post-state 6 retains a known body/collar overlap (2.272/3.326 mm); this is not a collision-free trajectory certificate.

Correct nominal wrist optics to the manufacturer rectified HD1080 focal length (737 px at 1920×1080, 245.666667 px at 640×360). Preserve original review intrinsics as provenance. This is not per-unit SDK calibration. Geometry, dynamics and collision profile verdicts are unchanged.

## 2026-10-22 — wrist mount planning spheres 1.0.0

Add the 148-sphere conservative mount model in `collision_spheres/wrist_mounts.json`.
Its SHA-256 and source hull hashes are in the sidecar. cuRobo uses it with the
existing robot and hand envelopes; the duplicate axial-pinch model is
replaced by a reference to v0.4.0. USD geometry and optics are unchanged.

## 2026-11-08 — robot planning spheres moved in

`collision_spheres/right_axial_pinch.json` and its sidecar move here from v0.4.0, which is
removed; the model is unchanged (1.0.0, SHA-256 `b0b43afc…`). Its builder fits this build's
hulls less the wrist mounts, which are v0.4.0's, and reproduces it exactly. USD geometry,
optics and verdicts are unchanged.

`.thumbs/256x256/rby1m_wujihand2_beta2.png`, a byte-identical copy of
`rby1m_wujihand2_beta2.usd.png`, is removed. The copy served RC.004 as worded before
SimReady foundation 2026.08.0; neither profile this build was validated against
(Robot-Body-Neutral and Robot-Body-Runnable 1.1.0) requires RC.004.
