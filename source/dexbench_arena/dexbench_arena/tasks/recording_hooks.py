# SPDX-License-Identifier: Apache-2.0
"""The task methods DexBench's recording and replay tools call, declared on one Arena task base."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from isaaclab_arena.tasks.task_base import TaskBase


class DexBenchRecordingTask(TaskBase):
    """Arena's ``TaskBase`` with DexBench's recording and replay hooks (module docstring); each default does
    nothing."""

    def validate_recording_dataset(self, path) -> None:
        """Raise ValueError to refuse a recording session into ``path`` before anything is written; the default
        accepts every path."""

    def episode_attributes(self, env, env_id: int) -> dict[str, str]:
        """The attributes a recorded demo of env ``env_id``'s episode carries; the default none."""
        return {}

    def prepare_replay_episode(self, env, demo_attributes: Mapping[str, Any]) -> None:
        """Set, from a demo's attributes, what the reset to its initial state would otherwise draw; the default
        nothing."""


def recording_hooks(task) -> DexBenchRecordingTask | None:
    """``task`` when it declares the hooks, else None (an Arena task without them)."""
    return task if isinstance(task, DexBenchRecordingTask) else None


def with_episode_attributes(handler_class: type, episode_attrs: Callable[[Any], Mapping[str, str]]) -> type:
    """``handler_class`` (an Isaac Lab ``HDF5DatasetFileHandler``) writing ``episode_attrs(episode)`` as attributes
    of every demo group it writes, right after the demo is written and before the env resets: what one episode was,
    which the file-level ``env_args`` (one per file, and the first input's after a merge) cannot hold. The group is
    named as Isaac Lab's handler names it (``demo_<demo_id>``, else ``demo_<count>``). An error there is reported and
    the demo kept without them: the take is already written."""

    class EpisodeAttributesHDF5DatasetFileHandler(handler_class):
        def write_episode(self, episode, demo_id: int | None = None, *args, **kwargs):
            name = f"demo_{demo_id if demo_id is not None else self._demo_count}"
            result = super().write_episode(episode, demo_id, *args, **kwargs)
            if not episode.is_empty():
                try:
                    self._hdf5_data_group[name].attrs.update(dict(episode_attrs(episode)))
                except Exception as error:
                    print(f"\n{'!' * 78}\n[record_demos] {name} was written WITHOUT its per-demo attributes: "
                          f"{type(error).__name__}: {error}\n{'!' * 78}\n", flush=True)
            return result

    return EpisodeAttributesHDF5DatasetFileHandler
