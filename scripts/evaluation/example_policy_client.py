#!/usr/bin/env python3
# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Example policy client for the DexBench env server.

Connects to a running ``scripts/evaluation/env_server.py``, prints the served
environment's metadata, then rolls out a trivial hold-position policy (the
server-provided hold action plus optional Gaussian noise). Use this as the
template for a real policy process: replace the ``client.hold_action()`` call
with model inference on ``result["observation"]``.

Runs anywhere with pyzmq + torch — no Isaac Sim required:

    python scripts/evaluation/example_policy_client.py --port 5560 --episodes 2 --steps 100
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "source" / "dexbench_arena"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from dexbench_arena.evaluation.env_client import DexBenchEnvClient  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=5560)
    parser.add_argument("--timeout_ms", type=int, default=300_000, help="Per-request timeout (first reset is slow).")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=100, help="Max steps per episode.")
    parser.add_argument("--seed", type=int, default=None, help="Base reset seed; episode index is added per episode.")
    parser.add_argument(
        "--action_noise_std",
        type=float,
        default=0.0,
        help="Gaussian noise std added to the hold action, to make the rollout visibly move.",
    )
    parser.add_argument("--stop_server", action="store_true", help="Ask the server process to shut down at the end.")
    return parser.parse_args()


def _print_info(info: dict) -> None:
    print(f"Environment: {info.get('environment_name')}")
    if info.get("task_description"):
        print(f"Task description: {info['task_description']}")
    print(f"num_envs={info['num_envs']}, device={info['device']}, step_dt={info['step_dt']:.4f}s")
    print(f"Action dim: {info['action_dim']}")
    for term in info["action_terms"]:
        detail = ""
        if "frame_names" in term:
            detail = f", frames={term['frame_names']}, hand_joints={len(term['hand_joint_names'])}"
        elif "joint_names" in term:
            detail = f", joints={len(term['joint_names'])}"
        print(f"  - {term['name']} ({term['class']}): dim={term['dim']}{detail}")
    print("Observation structure:")
    for group, terms in info["observation_structure"].items():
        print(f"  [{group}]")
        if isinstance(terms, dict) and "shape" not in terms:
            for term_name, spec in terms.items():
                print(f"    {term_name}: {spec}")
        else:
            print(f"    {terms}")


def main() -> None:
    args = _parse_args()

    with DexBenchEnvClient(host=args.host, port=args.port, timeout_ms=args.timeout_ms) as client:
        print(f"Connected to tcp://{args.host}:{args.port}")
        info = client.info()
        _print_info(info)

        for episode in range(args.episodes):
            seed = args.seed + episode if args.seed is not None else None
            t0 = time.monotonic()
            client.reset(seed=seed)
            print(f"\n[episode {episode}] reset done in {time.monotonic() - t0:.1f}s (seed={seed})")

            successes = None
            for step_index in range(args.steps):
                action = client.hold_action()
                if args.action_noise_std > 0.0:
                    action = action + torch.randn_like(action) * args.action_noise_std

                result = client.step(action)
                step_info = result.get("info") or {}
                if "is_success" in step_info:
                    successes = step_info["is_success"]

                terminated = result["terminated"]
                truncated = result["truncated"]
                if bool(terminated.any()) or bool(truncated.any()):
                    print(
                        f"[episode {episode}] ended at step {step_index + 1} "
                        f"(terminated={terminated.tolist()}, truncated={truncated.tolist()})"
                    )
                    break
            else:
                print(f"[episode {episode}] ran the full {args.steps} steps")

            if successes is not None:
                print(f"[episode {episode}] is_success={successes.tolist()}")

        client.close(stop_server=args.stop_server)
        if args.stop_server:
            print("Requested server shutdown")


if __name__ == "__main__":
    main()
