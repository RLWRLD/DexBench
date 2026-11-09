# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Run the pack-invoice paper sidecar alongside an environment's step, reset and close."""

from __future__ import annotations

import os
import time
from pathlib import Path

from dexbench_arena.deformables.pack_invoice_scene_randomization import PAPER_PHYSICS_ARGS
from dexbench_arena.deformables.pack_invoice_setup import build_pack_invoice_sidecar
from dexbench_arena.demos.pack_invoice_video import PackInvoiceVideoRecorder
from dexbench_arena.demos.paper_demo_hdf5 import PaperDemoHdf5Recorder
from dexbench_arena.demos.paper_demo_robot_channels import paper_demo_robot_channels

REPO_ROOT = Path(__file__).resolve().parents[4]


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def attach_pack_invoice_sidecar(env, cfg=None, hand=None, paper_args=None, scene_args=None, paper_pile=None) -> None:
    """Wrap ``env`` (unwrapped) so the paper sidecar rides along its step, reset and close."""

    env = getattr(env, "unwrapped", env)
    if getattr(env, "_dexbench_pack_invoice_attached", False):
        return

    holder: dict[str, object] = {"sidecar": None, "recorder": None}
    original_step = env.step
    original_reset = env.reset
    original_close = env.close
    original_reset_idx = env._reset_idx

    paper_args = {**PAPER_PHYSICS_ARGS, **(paper_args or {})}
    if _env_flag("DEXBENCH_PAPER_DEMO"):
        demo_dir = Path(os.environ.get("DEXBENCH_PAPER_DEMO_DIR", str(REPO_ROOT / "outputs" / "paper_demos")))
        holder["recorder"] = PaperDemoHdf5Recorder(env, demo_dir, lambda: holder["sidecar"],
                                                   channels=paper_demo_robot_channels(env), scene_args=scene_args,
                                                   paper_coupling=paper_args["paper_coupling"])
    want_video = _env_flag("DEXBENCH_PAPER_VIDEO") and holder["recorder"] is not None
    perf_every = int(os.environ.get("DEXBENCH_PACK_INVOICE_PERF_EVERY", "300") or 0)
    perf = {"n": 0, "env_s": 0.0, "side_s": 0.0, "frame_s": 0.0, "last_t0": None}

    original_render = env.sim.render

    def render_with_paper(*args, **kwargs):
        sidecar = holder["sidecar"]
        if sidecar is not None:
            sidecar.stack.update_usd_meshes()
        return original_render(*args, **kwargs)

    env.sim.render = render_with_paper

    def _build() -> None:
        holder["sidecar"] = build_pack_invoice_sidecar(env, cfg, hand, paper_args=paper_args, paper_pile=paper_pile)

    video_state = {"tried": False}

    def _attach_video() -> None:
        try:
            holder["recorder"].video = PackInvoiceVideoRecorder(
                env, None, render=_env_flag("DEXBENCH_PAPER_VIDEO_RENDER"),
                camera_path="/World/dexbench_paper_episode_camera",
                fps=max(1, round(1.0 / float(env.step_dt))),
            )
            print("[paper_demo] recording an MP4 per episode next to each HDF5")
        except Exception as exc:
            print(f"[paper_demo] video off: {exc!r}")

    def sidecar_reset(*args, **kwargs):
        result = original_reset(*args, **kwargs)
        built_now = holder["sidecar"] is None
        try:
            if built_now:
                _build()
            holder["sidecar"].reset(env)
        except Exception as exc:
            raise RuntimeError("Pack-invoice paper sidecar build/reset failed") from exc
        if built_now:
            if env.has_rtx_sensors:
                for _ in range(max(1, env.cfg.num_rerenders_on_reset)):
                    env.sim.render()
            for sensor in env.scene.sensors.values():
                sensor.reset()
            env.observation_manager.reset()
            env.obs_buf = env.observation_manager.compute(update_history=True)
            result = (env.obs_buf, *result[1:])
        if holder["recorder"] is not None:
            holder["recorder"].on_reset("xr_reset")
        return result

    def sidecar_step(action, *args, **kwargs):
        t0 = time.perf_counter()
        if holder["sidecar"] is None:
            _build()
        recorder = holder["recorder"]
        if want_video and not video_state["tried"]:
            video_state["tried"] = True
            _attach_video()
        if recorder is not None:
            recorder.begin_step(action)
        result = original_step(action, *args, **kwargs)
        t1 = time.perf_counter()
        try:
            holder["sidecar"].step(float(env.step_dt))
        except Exception as exc:
            raise RuntimeError("Pack-invoice paper sidecar step failed") from exc
        if recorder is not None:
            recorder.end_step()
        t2 = time.perf_counter()
        if perf_every > 0:
            perf["env_s"] += t1 - t0
            perf["side_s"] += t2 - t1
            if perf["last_t0"] is not None:
                perf["frame_s"] += t0 - perf["last_t0"]
            perf["last_t0"] = t0
            perf["n"] += 1
            if perf["n"] >= perf_every:
                n = perf["n"]
                frame_ms = perf["frame_s"] / max(n - 1, 1) * 1000.0
                print(
                    f"[DexBench PackInvoice perf] frame {frame_ms:.1f} ms ({1000.0 / max(frame_ms, 1e-6):.1f} fps) | "
                    f"env.step {perf['env_s'] / n * 1000.0:.1f} ms | paper sidecar {perf['side_s'] / n * 1000.0:.1f} ms "
                    f"(avg over {n} frames)"
                )
                perf.update({"n": 0, "env_s": 0.0, "side_s": 0.0, "frame_s": 0.0})
        return result

    def sidecar_close():
        try:
            if holder["recorder"] is not None:
                holder["recorder"].close()
            if holder["sidecar"] is not None:
                holder["sidecar"].close()
        finally:
            original_close()

    def sidecar_reset_idx(env_ids, *args, **kwargs):
        recorder = holder["recorder"]
        reason = recorder.reason_from_terminations() if recorder is not None and len(env_ids) else None
        original_reset_idx(env_ids, *args, **kwargs)
        if holder["sidecar"] is not None and len(env_ids):
            holder["sidecar"].reset(env)
        if recorder is not None and len(env_ids):
            recorder.on_reset(reason)

    env._reset_idx = sidecar_reset_idx
    env.close = sidecar_close
    env.reset = sidecar_reset
    env.step = sidecar_step
    env._dexbench_pack_invoice_attached = True
    env._dexbench_pack_invoice_sidecar = holder
    print("[DexBench PackInvoice] VBD paper sidecar attached; it is built on the first reset")
