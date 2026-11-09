#!/usr/bin/env bash
# Run a command in its own process group, pass SIGTERM and SIGINT on to the whole group, and return once every
# process of the group has exited. /isaac-sim/python.sh runs python as its child and passes no signal on; as a pod's
# PID 1 it ignores SIGTERM altogether. A worker or launcher started through it directly never saw a deleted pod's
# SIGTERM and was killed at the end of the grace time with its claim still held (2026-11-02 rehearsal).
#   signal_group.sh /isaac-sim/python.sh -m eval_launch.eval_worker ...
set -m  # the command gets a process group of its own, so the whole wrapper chain can be signalled at once
# Traps first: a signal that arrives while the command starts is kept and passed on as soon as it has a group
# (2026-11-02: a pod deleted in its first second ran on, claim held, until its SIGKILL).
pass_on() { signal=$1; [ -z "${group:-}" ] || kill -"$1" -- "-$group" 2>/dev/null; }
trap 'pass_on TERM' TERM
trap 'pass_on INT' INT
"$@" &
group=$!
[ -z "${signal:-}" ] || kill -"$signal" -- "-$group" 2>/dev/null
while :; do
  wait "$group"
  rc=$?
  kill -0 "$group" 2>/dev/null || break  # a trap ended the wait, not the command
done
# The wrapper may exit before the python it started has finished stopping (handing its unit back, its children)
while kill -0 -- "-$group" 2>/dev/null; do sleep 1; done
exit "$rc"
