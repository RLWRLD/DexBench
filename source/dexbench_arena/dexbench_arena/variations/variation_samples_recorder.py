# SPDX-License-Identifier: Apache-2.0
"""Demo recorder term: the value each enabled variation drew for the episode, in the exported dataset."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.managers.manager_term_cfg import RecorderTermCfg
from isaaclab.managers.recorder_manager import RecorderTerm
from isaaclab.utils.configclass import configclass

from dexbench_arena.embodiments.recorder_terms import add_declared_recorder_term
from dexbench_arena.variations.variation_draws import episode_variation_draws


class VariationSamplesRecorder(RecorderTerm):
    """Record, after every reset, what each enabled variation drew for the episode that starts."""

    def record_post_reset(self, env_ids: Sequence[int] | None) -> tuple[str | None, dict | None]:
        if env_ids is None:
            env_ids = range(self._env.num_envs)
        values = {
            key: torch.tensor(rows, dtype=torch.float32)
            for key, rows in episode_variation_draws(self._env, env_ids).items()
            if len({len(row) for row in rows}) == 1
        }
        return ("variations", values) if values else (None, None)


@configclass
class VariationSamplesRecorderCfg(RecorderTermCfg):
    """Configuration for VariationSamplesRecorder."""

    class_type: type[RecorderTerm] = VariationSamplesRecorder


def with_declared_variation_samples(previous):
    """Chain an env cfg callback that declares this term on the demo and Mimic recorders, so ``copy()`` keeps it."""

    def callback(env_cfg):
        if previous is not None:
            env_cfg = previous(env_cfg) or env_cfg
        add_declared_recorder_term(env_cfg, "record_variation_samples", VariationSamplesRecorderCfg())
        return env_cfg

    return callback
