# SPDX-License-Identifier: Apache-2.0
"""What ``git status`` says about a checkout, for the provenance a recording or a reproducibility check records."""
from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path


class GitUnavailable(RuntimeError):
    """A git command this check depends on failed; the state is unknown, not clean."""


def git_dirty_paths(cwd: Path, pathspecs: Sequence[str] = (), *, all_untracked: bool = False) -> list[str]:
    """Paths ``git status`` reports, submodule HEAD moves and edits included,
    under ``pathspecs`` when given (git pathspecs: a directory, a glob).
    Asset audits request individual untracked files; recording metadata retains
    Git's default grouping of untracked directories."""
    try:
        output = subprocess.run(
            ["git", "status", "--porcelain", "-z", "--ignore-submodules=none",
             *(["--untracked-files=all"] if all_untracked else []),
             *(["--", *pathspecs] if pathspecs else [])],
            cwd=cwd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", "") or str(error)
        raise GitUnavailable(f"git status failed in {cwd}: {detail.strip()}") from error
    paths, entries = [], iter(output.split("\0"))
    for entry in entries:
        if len(entry) < 4:
            continue
        paths.append(entry[3:])
        if "R" in entry[:2] or "C" in entry[:2]:
            paths.append(next(entries, ""))
    return [path for path in paths if path]
