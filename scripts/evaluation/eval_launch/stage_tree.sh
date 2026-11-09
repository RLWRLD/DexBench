#!/usr/bin/env bash
# Build a tree an eval launch can run from, from the published sources, in one command:
#   scripts/evaluation/eval_launch/stage_tree.sh [--repo <url>] <commit> <dest>
# then point the config at it (backend.k8s.tree_dir: <dest>, backend.k8s.commit: the full commit; USER_GUIDE.md, "The
# tree a launch runs from"). What it does, in order, in <dest>.partial, renamed to <dest> only once every step passed:
#   1. clone the repository (--repo; default: the origin remote of the checkout this script is in) and check out
#      <commit>, detached;
#   2. submodule/isaaclab_arena at its gitlink, recursively (the evaluation needs no other submodule); LFS payloads are
#      not fetched for any of them;
#   3. the assets' LFS payloads (git lfs pull --include="assets/**" --exclude=""), then a check that no file under
#      assets/ is still a pointer (LFS files elsewhere, e.g. docs/, stay pointers and are not checked);
#   4. the benchmark's runtime inputs (EVAL_INPUTS below: every cell's frozen rows, checked against benchmark/frozen/,
#      and DB-H3's input package) with the tree's own scripts/download_datasets.sh;
#   5. .source-commit, the full commit the pods hold the tree to, and SHA256SUMS of every tracked file.
# The launcher's own input check (plan.missing_inputs) sees whether files are there, not what they hold: this script's
# steps are what make them right.
# Run it on a host, as your normal user, not inside a container or pod (a container writes assets/ as its own user).
# It refuses to run as root (files written into assets/ as root lock the next git lfs pull out, and on a shared volume
# a non-root reader may not open them), an existing <dest> or <dest>.partial, and a missing tool (git, git-lfs, the hf
# CLI, python3), before anything is written. Credentials are the ones git and the hf CLI already use (a git
# credential helper for GitHub and huggingface.co, HF_TOKEN or HF_TOKEN_PATH, hf auth login; an SSH origin, such as
# git@github.com:..., needs an SSH key for it on this host, or --repo https://...); this script never reads,
# writes or prints a token, and git is told not to prompt. Each step prints its time.
set -euo pipefail

# What step 4 installs: --eval-conditions brings every cell's frozen rows and DB-H3's input package (eval/db_h3_inputs).
EVAL_INPUTS=(--eval-conditions)

# The repository this script came from: a user of another copy of the benchmark stages from their own
repo=$(git -C "$(dirname "$0")" remote get-url origin 2>/dev/null || true)
if [ "${1:-}" = --repo ]; then
  repo=${2:?--repo takes a URL}
  shift 2
fi
[ -n "$repo" ] || { echo "stage_tree.sh: this checkout has no origin remote; name the repository with --repo <url>" >&2
                    exit 2; }
if [ $# -ne 2 ]; then
  sed -n '2,3p' "$0" >&2
  exit 2
fi
commit=$1
dest=$(realpath -m "$2")
partial=$dest.partial

[ "$(id -u)" != 0 ] || { echo "stage_tree.sh: refusing to run as root (assets/ must never be written as root)" >&2; exit 1; }
for path in "$dest" "$partial"; do
  [ ! -e "$path" ] || { echo "stage_tree.sh: $path exists; remove it or choose another destination" >&2; exit 1; }
done
for tool in git git-lfs hf python3; do
  command -v "$tool" >/dev/null || { echo "stage_tree.sh: $tool not found (hf: pip install huggingface_hub)" >&2; exit 1; }
done
# A user may have no writable home (a uid without a passwd entry, as a scheduler can set); git and hf keep their state
# there. Only then is HOME replaced, with a private temporary one, which carries no credential helper of the user's.
if [ ! -w "${HOME:-/nonexistent}" ]; then
  HOME=$(mktemp -d)
  export HOME
  trap 'rm -rf "$HOME"' EXIT
  echo "HOME was not writable: using $HOME (no ~/.gitconfig credential helper; use HF_TOKEN_PATH, GIT_ASKPASS or a" \
       "system helper)"
fi
export GIT_TERMINAL_PROMPT=0 HF_HUB_DISABLE_PROGRESS_BARS=1
umask 022  # readable by the launch's pods, whatever user they run as

started=$(date +%s)
step() { echo "[stage_tree] $1 done after $(( $(date +%s) - started )) s"; }

GIT_LFS_SKIP_SMUDGE=1 git clone --no-checkout "$repo" "$partial"
cd "$partial"
# Before the checkout: a host's global git-lfs filter would otherwise smudge every pointer of the commit through the
# LFS endpoint, assets and docs alike, and fail at the checkout without a huggingface.co credential
git lfs install --local --skip-smudge --skip-repo
GIT_LFS_SKIP_SMUDGE=1 git -c advice.detachedHead=false checkout --detach "$commit"
full=$(git rev-parse HEAD)
step "clone at $full"
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --recursive submodule/isaaclab_arena
step submodules
git lfs pull --include="assets/**" --exclude=""
# git lfs pull checks each object's sha256 as it downloads it, and fails itself (naming only the oid) on an object the
# endpoint does not have; this catches what is left after a pull that ran (ls-files marks a file whose payload is not in
# the tree with '-'). Not git lfs fsck: it checks every pointer of HEAD, docs/ included.
pointers=$(git lfs ls-files --include="assets/**" --exclude="" | awk '$2 == "-" { print $3 }')
[ -z "$pointers" ] || { echo "stage_tree.sh: these assets are still LFS pointers after the pull:" >&2
                       echo "$pointers" | head -20 >&2; exit 1; }
step "assets (LFS)"
./scripts/download_datasets.sh "${EVAL_INPUTS[@]}"
step "runtime inputs (${EVAL_INPUTS[*]})"
# These generated provenance files are not source modifications.
printf '\n/.source-commit\n/SHA256SUMS\n' >> .git/info/exclude
echo "$full" > .source-commit
git ls-files -z --stage | awk -v RS='\0' -v ORS='\0' '$1 != "160000" { sub(/^[^\t]*\t/, ""); print }' \
  | xargs -0 sha256sum > SHA256SUMS
step "SHA256SUMS of $(wc -l < SHA256SUMS) tracked files"
cd /
mv -T "$partial" "$dest"  # -T: never into a <dest> that appeared meanwhile (an empty directory there is replaced)
echo "[stage_tree] ready: $dest at $full ($(du -sh "$dest" | cut -f1)); set backend.k8s.tree_dir and commit to them"
