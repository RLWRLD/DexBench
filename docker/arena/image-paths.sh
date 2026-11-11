#!/usr/bin/env bash
# Prints what the DexBench image is built from, one git pathspec per line: the
# builder and the CloudXR profile the image carries, the inputs the recipe
# consumes (base-inputs.sh), the assets pin (the Hugging Face snapshot that
# scripts/download_assets.sh downloads; assets/ is not in Git) and the Arena tree
# the image holds, submodule/isaaclab_arena.
#
#   docker/arena/image-paths.sh
#
# A recording's provenance (dexbench_arena.recording.rerecord_metadata.capture_source) calls it
# dirty only when one of these is: a replay runs the image of the recording's commit, so a change
# anywhere else never reaches it. Run from the repository root.
set -euo pipefail
layer=$(printf '%s\n' docker/build.py docker/cloudxr_quest3.env)
inputs=$(docker/arena/base-inputs.sh)
printf '%s\n' "$layer" "$inputs" reproducibility/assets.lock.json submodule/isaaclab_arena | sed '/^$/d' | sort -u
