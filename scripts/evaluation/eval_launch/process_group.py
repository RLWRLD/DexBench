# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Stopping a process group whole. Everything the launcher starts that can leave processes behind (a simulator, a policy
server, a recipe step, a local worker) runs in its own session, and is stopped as a group: SIGTERM to the group, then
SIGKILL to whatever of it is still there after the grace time.

Waiting for the group's leader alone was not enough: a leader that exits on SIGTERM while a child
ignores it or takes longer to stop looked like a stopped group, and the child kept its GPU memory or its port while the
next attempt started.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from collections.abc import Sequence


def group_alive(pgid: int) -> bool:
    """Whether the group has a member that is not a zombie. A killed member stays a zombie until its parent (often
    init, for a member whose parent was the leader) reaps it, and signal 0 still finds a zombie, so the members are read
    from /proc (Linux, where the launcher runs); without /proc, signal 0."""
    if not os.path.isdir("/proc/self"):
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        try:
            with open(f"/proc/{entry.name}/stat") as stat:
                fields = stat.read().rsplit(")", 1)[1].split()
        except OSError:
            continue
        if int(fields[2]) == pgid and fields[0] != "Z":
            return True
    return False


def stop_groups(processes: Sequence[subprocess.Popen], grace_s: float) -> None:
    """SIGTERM the process group of each of ``processes`` (each started with ``start_new_session=True``), wait up to
    ``grace_s`` for every member of every group to exit, then SIGKILL the groups that still have one."""
    for process in processes:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + grace_s
    while True:
        for process in processes:
            process.poll()
        remaining = [process for process in processes if group_alive(process.pid)]
        if not remaining or time.monotonic() >= deadline:
            break
        time.sleep(0.2)
    for process in remaining:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for process in processes:
        process.wait()
    deadline = time.monotonic() + 10.0
    while any(group_alive(process.pid) for process in remaining) and time.monotonic() < deadline:
        time.sleep(0.05)


def stop_group(process: subprocess.Popen, grace_s: float) -> None:
    """:func:`stop_groups` for one process."""
    stop_groups([process], grace_s)


OWNER_VARIABLE = "EVAL_LAUNCH_OWNER"
"""Set in a worker's environment to ``<scope>|<worker pid>``; everything the worker starts (simulators, servers,
recipe steps, and their children) inherits it, which is how :func:`stop_orphans` finds what a dead worker left."""


def claim_ownership(scope: str) -> None:
    """Mark this process, and every process it starts from now on, as ``scope``'s (one worker of one launch)."""
    os.environ[OWNER_VARIABLE] = f"{scope}|{os.getpid()}"


def _running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def stop_orphans(scope: str, grace_s: float = 30.0) -> list[int]:
    """Stop every process a dead worker of ``scope`` left behind, and return their pids: each whose environment
    names ``scope`` and a worker pid that no longer runs. A worker stops its simulator and server itself, but not
    when it is killed (SIGKILL, a crash); on a host they then held the port and GPU memory the restarted worker
    needed. Only processes carrying the tag are touched, and none of a worker still running.
    Linux only (``/proc``); elsewhere, nothing."""
    orphans = []
    prefix = f"{OWNER_VARIABLE}={scope}|".encode()
    for entry in os.scandir("/proc") if os.path.isdir("/proc/self") else ():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            with open(f"/proc/{entry.name}/environ", "rb") as handle:
                tag = next((item for item in handle.read().split(b"\0") if item.startswith(prefix)), None)
        except OSError:
            continue
        owner = tag[len(prefix):].decode(errors="replace") if tag else ""
        if owner.isdigit() and not _running(int(owner)):
            orphans.append(int(entry.name))
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in orphans:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + (grace_s if sig == signal.SIGTERM else 10.0)
        while any(_member(pid) for pid in orphans) and time.monotonic() < deadline:
            time.sleep(0.2)
    return orphans


def _member(pid: int) -> bool:
    """Whether ``pid`` still runs (a zombie does not)."""
    try:
        with open(f"/proc/{pid}/stat") as stat:
            return stat.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False
