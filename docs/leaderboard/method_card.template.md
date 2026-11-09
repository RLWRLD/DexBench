# Method card: MODEL / REVISION

Replace every prompt below. Include a paper link if available; a paper is not required.

## Identity and access

- Team, contact, model family and revision:
- Confirm one checkpoint and inference system across tasks; disclose routing or ensembles:
- Base models and immutable revisions; component and dataset licenses:
- Code repository and commit; checkpoint URI and SHA-256 (or weight-file manifest hash):
- Public/private availability and how reviewers obtain access:
- Related earlier submission and what changed:

## Training and selection

- Confirm standard-data training using the release's designated official training roots:
- Every dataset, immutable revision, per-task episodes and total training frames:
- External pretraining, known benchmark overlap and unavailable provenance:
- Confirm no additional demonstrations, synthetic trajectories or simulator interaction/RL for DexBench training:
- Any training on held-out objects or beyond-training ranges:
- Training configuration, compute and seeds:
- Checkpoint/hyperparameter selection rule and development conditions:
- Confirmation that scored starts, outcomes and row IDs were not used for training/selection:

## Inference contract

- Cameras, resolutions, image preprocessing and proprioception:
- Joint order, units, action representation and normalization:
- Prediction horizon, executed chunk, replanning, denoising and temporal ensembling:
- Policy RNG seed/reset, state/memory/cache reset at episode boundaries:
- Planner/tool/API dependencies and their fixed settings, routing and access needs:
- Confirmation that ground-truth state/success and evaluation IDs are not policy inputs:

## Reproduction and evidence

- Accepted benchmark release, commit, evaluator image digest and frozen hashes:
- Policy image digest, dependencies and exact launch/evaluation commands:
- GPU model/count, VRAM, latency measurement method and evaluation wall time:
- Submitted tasks/cells, v1/v1-plus, confirmation that all release tasks are complete:
- Bundle location/checksum and retention/access arrangement:
- Original result, log, trajectory and video locations; camera/episode association:
- Confirm `robot_pov_left_cam_rgb` (`camera_ego_left`) video for every successful and failed episode:
- Optional additional video cameras and the episodes they cover:
- HDF5 metadata/joint-column contract, state coordinate frame and timestep:
- Any extra raw images, prediction chunks or network traces and their alignment:
- Every failed/interrupted attempt, retry reason and operator approval if applicable:
- Known limitations and confirmation that no outcomes were cherry-picked:
