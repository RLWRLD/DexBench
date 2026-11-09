# SPDX-License-Identifier: Apache-2.0
"""An Isaac Lab HDF5 dataset file handler that adds metadata to the file's env_args when it writes the first
episode."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

from isaaclab.utils.datasets import HDF5DatasetFileHandler


def make_dataset_file_handler(
    deferred_env_args: Callable[[], Mapping[str, Any]],
) -> type[HDF5DatasetFileHandler]:
    """Return a handler that parses live metadata at the first write boundary."""

    class DatasetContractHDF5DatasetFileHandler(HDF5DatasetFileHandler):
        def create(self, file_path: str, env_name: str | None = None):
            super().create(file_path, env_name=env_name)
            self._contract_written = False

        def write_episode(self, episode, demo_id=None, dataset_compression=True):
            if not self._contract_written:
                frozen = json.loads(json.dumps(dict(deferred_env_args())))
                self.add_env_args(frozen)
                self._contract_written = True
            return super().write_episode(
                episode,
                demo_id=demo_id,
                dataset_compression=dataset_compression,
            )

    return DatasetContractHDF5DatasetFileHandler
