#!/usr/bin/env python3
# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Record demonstrations with DexBench-native Arena environments.

Runs Isaac Lab's ``scripts/tools/record_demos.py``. Beside the dataset it writes
the ``<stem>.job.json`` replay sidecar the teleop preprocessing pipeline needs,
and the HDF5 carries the commit, the physics and the joint contract the pipeline
replays and converts it by (``_install_recording_provenance`` in
``_dexbench_arena_script.py``).

It also writes ``<stem>.episodes.jsonl``, Arena's per-episode records: one row
per take, discarded takes included, every enabled variation's draw, string draws
too (``_dexbench_arena_script._write_episode_records_of_recording``).

``--experiment <yaml> --run <name>`` in place of the environment name records in the
environment that experiment Run builds (``_dexbench_arena_script.expand_experiment_run``);
flags after the pair replace the Run's, e.g. for DB-R2 on the headset::

    record_demos.py --seed 101 --viz kit --device cuda:0 --dataset_file datasets/db_r2/tc1.hdf5 --num_demos 10 \
        --experiment db_r2_collect.yaml --run tc1 --embodiment dexbench_rby1_wuji2 --teleop_device openxr

``--collection_rows <rows.jsonl>`` collects from a recorded start-state dump (a task's
``db_<task>_collect_conditions.yaml``, filtered): each take starts from the current row, which
moves on only when a demo of it is kept, one demo per row; ``--collection_first_row <i>``
resumes a collection, and ``--collection_requeue_file <path>`` offers again, first, every row appended to that
file (a kept demo the operator threw away) (``dexbench_arena.variations.collection_rows``).
"""

from __future__ import annotations

from _dexbench_arena_script import run_upstream_imitation_script


if __name__ == "__main__":
    run_upstream_imitation_script("record_demos.py")
