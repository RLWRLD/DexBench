# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Add an entry to the episode metadata a recorder writes into the dataset (``env_args``)."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy


class _EpisodeMetadata:
    """One contributor: adds ``metadata``, or what it returns at each export, under ``key``."""

    def __init__(self, get_ep_meta, key: str, metadata: dict | Callable[[], dict]):
        self._get_ep_meta = get_ep_meta
        self.key = key
        self._metadata = metadata

    def __call__(self) -> dict:
        result = deepcopy(self._get_ep_meta())
        metadata = self._metadata() if callable(self._metadata) else self._metadata
        result[self.key] = deepcopy(metadata)
        return result


def install_episode_metadata(env, key: str, metadata: dict | Callable[[], dict]) -> None:
    """Add ``metadata`` under ``key`` to the recorder's episode metadata, once per recorder."""
    recorder = env.recorder_manager
    link = recorder.get_ep_meta
    while link is not None:
        if isinstance(link, _EpisodeMetadata) and link.key == key:
            return
        link = getattr(link, "_get_ep_meta", None)
    recorder.get_ep_meta = _EpisodeMetadata(recorder.get_ep_meta, key, metadata)
