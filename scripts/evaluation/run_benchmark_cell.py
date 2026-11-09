#!/usr/bin/env python3
# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Run one model on one benchmark cell (or a task's sweep) from the cell's checked-in file.

The cell file (``db_<task>_eval_<cell>.yaml``, or ``db_<task>_sensitivity.yaml`` with ``--sweep``) names no
model. This puts the model's policy block from ``dexbench_arena.benchmark.models`` in place of the
baseline's, checks the frozen rows and the cell's declaration, writes ``composed_experiment.yaml`` and
``run_manifest.json`` into ``--output``, and runs the unchanged experiment runner on the composed file into
``<output>/arena`` (``dexbench_arena.benchmark.model_run``). Arguments after ``--`` go to the experiment
runner. A benchmark run keeps every episode's per-step state (``record_dataset: true`` in the composed Runs'
environment, about 3.2 KB per step, no camera frames) and every rendered camera's mp4, head and palm (``--record_camera_video``);
``--no_record_dataset`` and ``--no_record_camera_video`` opt out. ``--submission`` requires both recordings
and enables upstream trajectory episode IDs for leaderboard validation. ``--version`` picks the cell's rows: ``v1`` (the
default) a fixed half of each cell, ``v1-plus`` every row (``protocol.EVAL_VERSIONS``). ``--shard I/N`` runs part of
those rows from a verified file of them, and ``--slices '{"<run>": [start, end]}'`` exactly those Runs' rows at the
version (an eval launch's unit); the scorer joins the parts. The scorer takes ``--results <cell>=<output>``::

    /isaac-sim/python.sh scripts/evaluation/run_benchmark_cell.py --task r2 --cell train --model gr00t \\
        --output outputs/db_r2_eval_train_gr00t --set remote_port=5812 --checkpoint s3://.../step_30000 -- --viz none

The composed file is checked before Kit starts, in this process; the runner is a child process.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

from dexbench_arena.benchmark.model_run import parse_slices, prepare
from dexbench_arena.benchmark.protocol import DEFAULT_EVAL_VERSION, EVAL_VERSIONS
from dexbench_arena.benchmark.run_manifest import ARENA_OUTPUT, COMPOSED

HERE = Path(__file__).resolve().parent
CONFIGS = HERE / "experiment_configs" / "benchmark"


def _deployment(items: list[str], parser: argparse.ArgumentParser) -> dict:
    fields = {}
    for item in items:
        key, separator, value = item.partition("=")
        if not separator or not key:
            parser.error(f"--set takes FIELD=VALUE, got {item!r}")
        fields[key] = yaml.safe_load(value)
    return fields


def _shard(text: str) -> tuple[int, int]:
    index, found, parts = text.partition("/")
    if not found:
        raise argparse.ArgumentTypeError(f"expected I/N, got {text!r}")
    return int(index), int(parts)


def _slices(text: str) -> dict:
    try:
        return parse_slices(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from None


def _model_entry(text: str) -> dict:
    try:
        entry = json.loads(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"not JSON: {error}") from None
    if not isinstance(entry, dict):
        raise argparse.ArgumentTypeError("a JSON object, the model's entry")
    return entry


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    own, runner_args = (argv[:argv.index("--")], argv[argv.index("--") + 1:]) if "--" in argv else (argv, [])
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--task", required=True)
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument("--cell")
    where.add_argument("--sweep", action="store_true", help="The task's sensitivity sweep instead of a cell.")
    parser.add_argument("--model", required=True,
                        help="A key of dexbench_arena.benchmark.models, or the key of the model --model-entry registers.")
    parser.add_argument("--model-entry", type=_model_entry, metavar="JSON",
                        help="A model an eval config registers instead of models.py (models.config_model): "
                             '{"key", "label", "note", "group", "policy"}, as JSON. The eval launcher passes its '
                             "config's model block; the run manifest records the whole entry.")
    parser.add_argument("--output", type=Path, required=True, help="The run's own directory (one run per directory).")
    parser.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE",
                        help="Change a deployment field of the model's policy block (model_run.DEPLOYMENT_FIELDS: server address, "
                             "contract file, device; the server's token comes from DEXBENCH_POLICY_API_TOKEN only); "
                             "never its type, policy_variant, horizon or task fields.")
    parser.add_argument("--checkpoint", help="What the policy server loaded, recorded in the run manifest.")
    parser.add_argument("--submission", action="store_true",
                        help="Require dataset and camera recording; include upstream trajectory episode identities for submission.")
    parser.add_argument("--manifest", type=Path, default=CONFIGS / "benchmark_manifest.json")
    parser.add_argument("--no_record_dataset", dest="record_dataset", action="store_false",
                        help="Do not keep the per-step state HDF5 (kept by default, about 3.2 KB per step).")
    parser.add_argument("--no_record_camera_video", dest="record_camera_video", action="store_false",
                        help="Do not keep the cameras' mp4s (kept by default).")
    part = parser.add_mutually_exclusive_group()
    part.add_argument("--shard", type=_shard, metavar="I/N",
                      help="Run the I-th of N contiguous parts of every Run's rows at --version (I from 0); the scorer joins the parts.")
    part.add_argument("--slices", type=_slices, metavar="JSON",
                      help='Run exactly these Runs\' rows at --version, {"<run>": [start, end], ...} (end exclusive); the scorer joins the parts.')
    parser.add_argument("--version", choices=sorted(EVAL_VERSIONS), default=DEFAULT_EVAL_VERSION,
                        help="The cell's rows: v1 a fixed half of each cell (the default), v1-plus every row.")
    parser.add_argument("--checkpoint-sha256", help="Immutable checkpoint digest; required for config-model submissions.")
    parser.add_argument("--prepare_only", action="store_true", help="Check and write the two files; do not run.")
    args = parser.parse_args(own)
    run_manifest = prepare(args.task, None if args.sweep else args.cell, args.model, args.output,
                           manifest_path=args.manifest, configs=CONFIGS,
                           deployment=_deployment(args.set, parser), checkpoint=args.checkpoint,
                           record_dataset=args.record_dataset, record_camera_video=args.record_camera_video,
                           shard=args.shard, slices=args.slices, version=args.version, argv=argv,
                           model_entry=args.model_entry, submission=args.submission, checkpoint_sha256=args.checkpoint_sha256)
    print(f"{args.output / COMPOSED}: {run_manifest['label']} on {args.task} {args.cell or 'sweep'} "
          f"(frozen set {run_manifest['frozen_sha256'][:12]})")
    if args.prepare_only:
        return 0
    command = [sys.executable, str(HERE / "experiment_runner.py"), "--experiment_output_directory",
               str(args.output / ARENA_OUTPUT), "--experiment_config", str(args.output / COMPOSED),
               *(["--record_camera_video"] if args.record_camera_video else []), *runner_args]
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
