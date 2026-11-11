#!/usr/bin/env bash
# Download the assets into assets/: the snapshot of the RLWRLD/DexBench-assets Hugging Face dataset that
# reproducibility/assets.lock.json pins.
#
#   ./scripts/download_assets.sh                                  # every asset
#   ./scripts/download_assets.sh --include "env/e4_hotel_fb/**"  # part of them; paths are relative to assets/
#   ./scripts/download_assets.sh --prune                         # every asset, and delete files the snapshot does not have
#
# Requires the hf CLI (pip install -U huggingface_hub) and python3. The download fetches only the files that are
# missing or changed. With --include it also fetches every file that the receipt lists as stored in Git, not LFS
# (indexes, reference views, sidecars and documents; about 7 MB): the code reads some of them for every asset.
# Then the script checks what it downloaded: the snapshot's receipt
# (assets/asset-upload-manifest.json) must name the assets tree the lock pins, and every file must have the size
# and SHA-256 that the receipt records. Without --include it also refuses files that the snapshot does not have
# (an earlier snapshot can leave some after a git pull); --prune deletes them.
# Run it on the host as your own user, not from a container: files that a container writes are root-owned.
# Git ignores assets/.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)
ASSETS="${REPO_ROOT}/assets"
export PYTHONPATH="${REPO_ROOT}/source/dexbench_arena${PYTHONPATH:+:${PYTHONPATH}}"

INCLUDES=()
PRUNE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --include) INCLUDES+=("${2:?--include takes a path pattern}"); shift 2 ;;
    --prune) PRUNE=1; shift ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "download_assets.sh: unknown argument $1 (see --help)" >&2; exit 2 ;;
  esac
done
if [ "${PRUNE}" = 1 ] && [ "${#INCLUDES[@]}" -gt 0 ]; then
  echo "download_assets.sh: --prune needs the whole snapshot; it takes no --include" >&2
  exit 2
fi
if ! command -v hf >/dev/null 2>&1; then
  echo "hf CLI not found: pip install -U huggingface_hub" >&2
  exit 1
fi

LOCKED=$(python3 -c '
import sys
from dexbench_arena.assets.snapshot import read_lock
lock = read_lock(sys.argv[1])
if lock is None:
    sys.exit("download_assets.sh: this checkout has no reproducibility/assets.lock.json")
print(lock["repo_id"], lock["revision"])' "${REPO_ROOT}")
read -r REPO_ID REVISION <<< "${LOCKED}"

echo "Downloading ${REPO_ID}@${REVISION} into assets/"
mkdir -p "${ASSETS}"
# The options go before the repo ID, and the file names right after it: hf 0.x (argparse) takes the positional
# arguments only as one run, so "REPO --option value file" fails with "unrecognized arguments: file".
download() { hf download --repo-type dataset --revision "${REVISION}" --local-dir "${ASSETS}" "${REPO_ID}" "$@"; }
download asset-upload-manifest.json >/dev/null
if [ "${#INCLUDES[@]}" -eq 0 ]; then
  download >/dev/null
else
  # A Git checkout of assets/ holds these files before any LFS download, and asset registration reads some of them
  # for every package (env/scene_bundle_index.json, the reference views), so a part needs them all.
  SHARED=()
  while IFS= read -r path; do
    if [ -n "${path}" ]; then SHARED+=("${path}"); fi
  done < <(python3 -c '
import sys
from dexbench_arena.assets.snapshot import read_receipt, shared_files
print("\n".join(shared_files(read_receipt(sys.argv[1]) or {})))' "${ASSETS}")
  if [ "${#SHARED[@]}" -gt 0 ]; then
    download "${SHARED[@]}" >/dev/null
  fi
  # One call for each pattern: hf download takes repeated --include options in one version and nargs in another.
  for pattern in "${INCLUDES[@]}"; do
    download --include "${pattern}" >/dev/null
  done
fi

python3 - "${ASSETS}" "${REPO_ROOT}" "${PRUNE}" "${INCLUDES[@]}" <<'PY'
import sys
from pathlib import Path
from dexbench_arena.assets.snapshot import extra_files, read_lock, read_receipt, selected_files, snapshot_problems
assets, root, prune, include = Path(sys.argv[1]), sys.argv[2], sys.argv[3] == "1", sys.argv[4:]
lock = read_lock(root)
if prune:
    receipt = read_receipt(assets)
    for relative in extra_files(assets, receipt or {}) if receipt else ():
        (assets / relative).unlink()
        print(f"pruned assets/{relative}")
problems = snapshot_problems(assets, lock, include)
if problems:
    shown = "\n  ".join(problems[:20])
    more = f"\n  ... and {len(problems) - 20} more" if len(problems) > 20 else ""
    sys.exit(f"assets/ is not {lock['repo_id']}@{lock['revision']}:\n  {shown}{more}\n"
             "Run ./scripts/download_assets.sh again; --prune deletes the files that the snapshot does not have.")
checked = len(selected_files(read_receipt(assets), include))
print(f"assets/: {checked} files match {lock['repo_id']}@{lock['revision'][:12]} "
      f"(assets tree {lock['source_assets_tree'][:12]})")
PY
