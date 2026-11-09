#!/usr/bin/env bash
# Run the eval launcher (python -m eval_launch) with any mode and exit with its status. It puts this package's parent and this tree's
# source/dexbench_arena (which the launcher imports) first on the path, saves this shell's environment for the policy
# servers before /isaac-sim/python.sh adds Isaac Sim's paths to it, and starts the launcher through signal_group.sh,
# so a SIGTERM to this script reaches it (python.sh passes none on). The saved
# environment holds the shell's tokens: the launcher reads it from a pipe on descriptor 3, never a file
# (server_env.py).
#   scripts/evaluation/eval_launch/launch.sh <config.yaml> --plan-only
#   scripts/evaluation/eval_launch/launch.sh <config.yaml> --local
#   scripts/evaluation/eval_launch/launch.sh <config.yaml> --resume --requeue-claims
#   scripts/evaluation/eval_launch/launch.sh <config.yaml> --score
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
# The saved environment never touches the disk: descriptor 3 is a pipe that env -0 writes and the launcher reads, so no
# failure, signal or exec error can leave it behind.
exec 3< <(env -0)
tree=$(cd "$here/../../.." && pwd)
export PYTHONPATH="$(dirname "$here"):$tree/source/dexbench_arena${PYTHONPATH:+:$PYTHONPATH}"
# The interpreter: ISAAC_PYTHON when set; else Isaac Sim's (the simulator image's), and on a host without Isaac Sim
# python3, which is enough for --plan-only, --k8s-job and --status (a Python with PyYAML). The modes that run the
# simulator, or score with scipy, are refused there with one line rather than a traceback.
python=${ISAAC_PYTHON:-/isaac-sim/python.sh}
if [ -z "${ISAAC_PYTHON:-}" ] && [ ! -x "$python" ]; then
  python=python3
  for argument in "$@"; do
    case "$argument" in
      --local|--resume|--smoke|--score)
        echo "launch.sh: $argument runs in the simulator image (/isaac-sim/python.sh is not here); set ISAAC_PYTHON to" \
             "a Python that has Isaac Sim's packages (and scipy for --score) to run it elsewhere" >&2
        exit 2 ;;
    esac
  done
fi
# /isaac-sim/python.sh turns every non-zero exit status into 1: the launcher (or the driver it becomes) writes its own
# status to $EVAL_LAUNCH_STATUS (eval_launch.run_main), which this script exits with. The wrapper runs as a job of its
# own (set -m: in a group of its own, and not ignoring INT as a background job would); this script passes it a TERM or
# an INT and waits for it.
status=$(mktemp "${TMPDIR:-/tmp}/eval-launch-status.XXXXXX")
trap 'rm -f "$status"' EXIT
export EVAL_LAUNCH_STATUS=$status
set -m
"$here/signal_group.sh" "$python" -m eval_launch "$@" --server-env /dev/fd/3 &
child=$!
exec 3<&-
trap 'kill -TERM "$child" 2> /dev/null || true' TERM
trap 'kill -INT "$child" 2> /dev/null || true' INT
rc=0
while :; do
  wait "$child" && rc=0 || rc=$?
  kill -0 "$child" 2> /dev/null || break  # a trap ended the wait, not the job
done
code=$(cat "$status" 2> /dev/null || true)
exit "${code:-$rc}"
