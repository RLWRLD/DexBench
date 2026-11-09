# SPDX-License-Identifier: Apache-2.0
"""Replay hand-marked subtask boundaries so ``annotate_demos.py --auto`` needs no GUI pass."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path

import torch

SubtaskBoundaries = list[dict[str, int] | None]


def load_subtask_boundaries(path: Path) -> SubtaskBoundaries:
    """Read and validate a boundaries table; raise ``ValueError`` on anything but the documented shape."""
    table = json.loads(Path(path).read_text())
    if not isinstance(table, list):
        raise ValueError(f"{path}: expected a JSON list indexed by source episode, got {type(table).__name__}")
    for index, row in enumerate(table):
        if row is None:
            continue
        if not isinstance(row, dict) or not all(isinstance(k, str) and isinstance(v, int) for k, v in row.items()):
            raise ValueError(f"{path}: episode {index} must be null or a {{signal: step}} object, got {row!r}")
    marked = sum(1 for row in table if row)
    print(f"[DexBench Mimic] replaying stored subtask boundaries from {path}: {marked}/{len(table)} episode(s) carry marks", flush=True)
    return table


_DEMO_NAME = re.compile(r"demo_(\d+)\Z")


def boundary_rows(names: Sequence[str]) -> dict[str, int]:
    """Each demo's row in a boundaries table: its position in numeric order of the ``demo_<int>`` names."""
    numbers = {}
    for name in names:
        match = _DEMO_NAME.match(name)
        if match is None:
            raise ValueError(f"demo {name!r} is not named demo_<int>; a boundaries table cannot be matched to it")
        numbers[name] = int(match.group(1))
    return {name: row for row, name in enumerate(sorted(names, key=numbers.__getitem__))}


class ReplaysSubtaskBoundaries:
    """Mimic env mixin: answer subtask signals from a stored table during ``--auto`` replay."""

    _subtask_boundaries: SubtaskBoundaries | None = None
    _subtask_boundaries_loaded: bool = False

    @property
    def subtask_boundaries(self) -> SubtaskBoundaries | None:
        if not self._subtask_boundaries_loaded:
            path = getattr(self.cfg, "subtask_boundaries_path", None)
            table = load_subtask_boundaries(Path(path)) if path else None
            if table is not None:
                names = getattr(self.cfg, "subtask_boundaries_episode_names", None)
                if names is None:
                    raise ValueError(
                        f"{path}: the replayed file's demo names are not on the env cfg "
                        "(subtask_boundaries_episode_names), so its rows cannot be matched to demos; "
                        "run annotate_demos.py through scripts/imitation_learning/annotate_demos.py"
                    )
                if len(names) != len(table):
                    raise ValueError(f"{path} has {len(table)} rows but the replayed file has {len(names)} demos")
                rows = boundary_rows(names)
                table = [table[rows[name]] for name in names]
            self._subtask_boundaries = table
            self._subtask_boundaries_loaded = True
        return self._subtask_boundaries

    def reset_to(self, state, env_ids=None, seed=None, is_relative=False):
        """Track which recorded episode is being replayed, for the stored boundaries."""
        self._replay_episode_index = getattr(self, "_replay_episode_index", -1) + 1
        self._replay_start_step = int(self.common_step_counter)
        return super().reset_to(state, env_ids, seed=seed, is_relative=is_relative)

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        table = self.subtask_boundaries
        if table is None:
            raise NotImplementedError(
                f"{type(self).__name__} answers subtask signals from a stored table: pass "
                "--mimic_boundaries <table.json> on the environment (recovered from an earlier annotation by the "
                "maintainers' boundary extraction), or drop --auto to mark by hand."
            )
        index = getattr(self, "_replay_episode_index", 0)
        step = int(self.common_step_counter) - getattr(self, "_replay_start_step", 0)
        boundaries = table[index] if 0 <= index < len(table) else None
        signals = {}
        for name in {signal for row in table if row for signal in row}:
            reached = boundaries is not None and step >= boundaries.get(name, 1 << 30)
            signals[name] = torch.full((self.num_envs,), bool(reached), dtype=torch.bool, device=self.device)
        return signals
