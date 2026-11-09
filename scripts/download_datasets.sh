#!/usr/bin/env bash
# Download DexBench datasets at their repository's revision in
# reproducibility/datasets.lock.json. RLWRLD/DexBench-eval is the default.
#
#   ./scripts/download_datasets.sh                 # pinned revision, everything
#   ./scripts/download_datasets.sh --revision main # override the pin
#   ./scripts/download_datasets.sh --repo-id RLWRLD/DexBench-datasets \
#     --include "<path>/**"     # the multitask entry of the lock gives the multitask set's path
#   ./scripts/download_datasets.sh --eval-conditions # frozen rows and DB-H3's saved starts
#   ./scripts/download_datasets.sh --eval-conditions --force # replace conflicting local files
#
# Requires the hf CLI and authentication for the selected repository.
# --eval-conditions checks every frozen row, the start-state sets the cells read beside them
# (DB-R1's layouts, DB-M5's piles: benchmark/frozen/inputs.json), and that DB-H3's inputs
# reproduce its 240 rows, before installing outputs/db_conditions and outputs/db_h3_inputs.
# The check needs only python3; DB-H3's utensils are asset-pool packages under assets/, not
# part of the download.
# --h3-inputs is accepted and ignored: --eval-conditions always installs DB-H3's inputs now.
# Downloads stage in a fresh temporary directory. Verification failures leave
# outputs untouched; differing destination files are refused unless --force is set.
# Ordinary downloads use ${DEXBENCH_DATASETS_ROOT:-$HOME/datasets}/dexbench_arena.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)
LOCK="${REPO_ROOT}/reproducibility/datasets.lock.json"

DATASETS_ROOT="${DEXBENCH_DATASETS_ROOT:-${HOME}/datasets}"
LOCKED_REPO_ID=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["repo_id"])' "${LOCK}")
REPO_ID="${DEXBENCH_DATA_REPO_ID:-${LOCKED_REPO_ID}}"
REVISION="${DEXBENCH_DATA_REVISION:-}"

ARGS=()
EVAL_CONDITIONS=0
FORCE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --eval-conditions) EVAL_CONDITIONS=1; shift ;;
    --force) FORCE=1; shift ;;
    --h3-inputs) echo "--h3-inputs is no longer needed and is ignored: --eval-conditions installs DB-H3's inputs" >&2; shift ;;
    --revision) REVISION="$2"; shift 2 ;;
    --repo-id) REPO_ID="$2"; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

# Resolve the pin after CLI/environment repo selection. Unknown repositories
# retain the CLI's default-branch behavior; never borrow another repo's commit.
if [ -z "${REVISION}" ]; then
  REVISION=$(python3 - "${LOCK}" "${REPO_ID}" <<'PY'
import json, sys
lock = json.load(open(sys.argv[1]))
for entry in (lock, lock.get("multitask", {})):
    if entry.get("repo_id") == sys.argv[2]:
        print(entry["revision"])
        break
PY
  )
fi
REVISION_ARGS=()
if [ -n "${REVISION}" ]; then
  REVISION_ARGS=(--revision "${REVISION}")
fi

if [ "${EVAL_CONDITIONS}" = 1 ] && [ "${#ARGS[@]}" -gt 0 ]; then
  echo "--eval-conditions takes no other download arguments (got: ${ARGS[*]}); it fetches evaluation rows and DB-H3's inputs only" >&2
  exit 2
fi
if [ "${EVAL_CONDITIONS}" = 0 ] && [ "${FORCE}" = 1 ]; then
  echo "--force applies to --eval-conditions only" >&2
  exit 2
fi

if ! command -v hf >/dev/null 2>&1; then
  echo "hf CLI not found: pip install huggingface_hub, then hf auth login" >&2
  exit 1
fi

if [ "${EVAL_CONDITIONS}" = 0 ]; then
  mkdir -p "${DATASETS_ROOT}/dexbench_arena"
  echo "Downloading ${REPO_ID}@${REVISION:-<default branch>} to ${DATASETS_ROOT}/dexbench_arena"
  exec hf download "${REPO_ID}" --repo-type dataset "${REVISION_ARGS[@]}" \
    --local-dir "${DATASETS_ROOT}/dexbench_arena" "${ARGS[@]}"
fi

# A fresh directory per run, so the check below sees what this revision holds and
# not a folder an earlier download at another revision left behind.
DOWNLOAD_DIR=$(mktemp -d)
trap 'rm -rf "${DOWNLOAD_DIR}"' EXIT
echo "Downloading evaluation rows from ${REPO_ID}@${REVISION:-<default branch>}"
hf download "${REPO_ID}" --repo-type dataset "${REVISION_ARGS[@]}" \
  --local-dir "${DOWNLOAD_DIR}" --include "eval/db_conditions/**"
# Separate calls support both hub 0.36's nargs and newer repeated-option CLIs.
hf download "${REPO_ID}" --repo-type dataset "${REVISION_ARGS[@]}" \
  --local-dir "${DOWNLOAD_DIR}" --include "eval/db_h3_inputs/**"
SRC="${DOWNLOAD_DIR}/eval/db_conditions"
DST="${REPO_ROOT}/outputs/db_conditions"
if [ ! -d "${SRC}" ] || [ -z "$(ls -A "${SRC}")" ]; then
  echo "${REPO_ID}@${REVISION:-<default branch>} has no eval/db_conditions/: it needs revision" \
       "containing every checked-in frozen task. Check the revision and CLI download output; nothing was installed." >&2
  exit 1
fi
# Check the download itself before anything is installed: every pinned row file must be in
# it with the pinned SHA-256 and row count, so an incomplete or corrupt download can neither
# borrow files an earlier run left in outputs/db_conditions/ nor overwrite good ones there.
PYTHONPATH="${REPO_ROOT}/source/dexbench_arena${PYTHONPATH:+:${PYTHONPATH}}" python3 - "${SRC}" "${DOWNLOAD_DIR}/eval/db_h3_inputs" <<'PY'
import json, os, sys
from pathlib import Path
from dexbench_arena.benchmark.frozen import INPUTS_PIN, PINNED_DIR, digest_and_count, input_problems
src = Path(sys.argv[1])
bad = []
checked = 0
for frozen_file in sorted(PINNED_DIR.glob("db_*.json")):
    for name, record in json.loads(frozen_file.read_text()).items():
        rel = os.path.relpath(os.path.normpath(name), "outputs/db_conditions")
        downloaded = src / rel
        checked += 1
        if not downloaded.is_file():
            bad.append(f"{name}: not in the download")
        elif digest_and_count(downloaded) != (record["sha256"], record["rows"]):
            bad.append(f"{name}: differs from {frozen_file.name} ({record['rows']} rows, sha256 {record['sha256'][:12]})")
# The start-state sets the cells name beside their rows (r1_layouts, parts_bin_clutter_dir): a revision without
# them installs rows no build of those tasks can start from.
bad += input_problems(src)
if bad:
    sys.exit("The download does not match benchmark/frozen/; nothing was installed:\n  " + "\n  ".join(bad))
# DB-H3's rows are categorical draws of saved starts: its inputs must reproduce every one. The
# rows bind the bundles' recorded utensil digests, so the asset files (LFS payload in assets/)
# are not read here; the environment verifies them when it loads a bundle.
try:
    from dexbench_arena.evaluation.artifacts import jsonl_bytes, sha256_bytes
    from dexbench_arena.evaluation.h3_eval_inputs import SELECTION, cell_rows
    selection = json.loads(SELECTION.read_text())
    pins = json.loads((PINNED_DIR / "db_h3.json").read_text())
    for cell in ("train", "ga_os"):
        for key, rows in cell_rows(sys.argv[2], selection, cell, verify_assets=False).items():
            path = f"outputs/db_conditions/db_h3_{cell}/{key}.jsonl"
            if sha256_bytes(jsonl_bytes(rows)) != pins[path]["sha256"]:
                raise ValueError(f"{path}: H3 inputs do not reproduce frozen rows")
except (ImportError, OSError, ValueError, KeyError, TypeError) as exc:
    sys.exit(f"H3 input validation failed: {exc}; nothing was installed")
print("download: H3 runtime inputs reproduce all 240 frozen rows")
print(f"download: {checked} row files match benchmark/frozen/")
print(f"download: {len(json.loads(INPUTS_PIN.read_text()))} start-state sets match benchmark/frozen/{INPUTS_PIN.name}")
PY
SRC="${DOWNLOAD_DIR}/eval"
DST="${REPO_ROOT}/outputs"
CONFLICTS=()
while IFS= read -r -d '' file; do
  rel="${file#"${SRC}/"}"
  if [ -e "${DST}/${rel}" ] && ! cmp -s "${file}" "${DST}/${rel}"; then
    CONFLICTS+=("outputs/${rel}")
  fi
done < <(find "${SRC}" -type f -print0)
if [ "${#CONFLICTS[@]}" -gt 0 ] && [ "${FORCE}" = 0 ]; then
  echo "These files in this checkout differ from ${REPO_ID}@${REVISION:-<default branch>} and would be overwritten:" >&2
  printf '  %s\n' "${CONFLICTS[@]}" >&2
  echo "Move them aside, or pass --force to replace them." >&2
  exit 1
fi
mkdir -p "${DST}"
cp -R "${SRC}/." "${DST}/"
echo "Installed ${SRC#"${DOWNLOAD_DIR}/"} into outputs/"
