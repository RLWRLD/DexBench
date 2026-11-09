# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""One HDF5 file per pack-invoice teleop episode, robot and paper state included."""

from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import os
import shlex
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import h5py
import numpy as np
import warp as wp

from dexbench_arena.variations.variation_draws import episode_variation_draws

PAPER_LAYOUT_AT_EPISODE_START = "episode_start"
"""The ``paper_layout_at`` attr: ``paper_layout`` is the sidecar as the episode's first row saw it. A file
without the attr took it when the file was written, after the reset that ended the episode, so with a
varied box its carton is the next episode's."""


def _to_np(value: Any) -> np.ndarray:
    """Torch tensor, warp array or array-like -> host numpy."""

    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    try:
        if isinstance(value, wp.array):
            return value.numpy()
    except Exception:
        pass
    return np.asarray(value)


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {k: _jsonable(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


REPO_ROOT = Path(__file__).resolve().parents[4]
CONTAINER_REPO_ROOT = "/workspaces/dexbench_arena"
REPLAY_SCRIPT = "scripts/tools/replay_paper_demo_physics.py"
PACKAGES = ("isaacsim", "isaaclab", "isaaclab_arena", "newton", "warp-lang", "mujoco", "mujoco-warp", "torch")


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-c", "safe.directory=*", "-C", str(REPO_ROOT), *args],
            capture_output=True, text=True, timeout=10, check=True,
        )
    except Exception:
        return None
    return out.stdout.rstrip()


def _environment_args(argv: list[str]) -> list[str]:
    """The environment name and its arguments from a teleop command line, without the teleop device."""

    start = next((i for i, a in enumerate(argv) if a.startswith("dexbench_") and not argv[i - 1].startswith("--")), None)
    if start is None:
        return []
    out, rest = [], argv[start:]
    i = 0
    while i < len(rest):
        if rest[i] == "--teleop_device":
            i += 2
            continue
        if not rest[i].startswith("--teleop_device="):
            out.append(rest[i])
        i += 1
    return out


def collect_run_info() -> dict[str, Any]:
    """What this session runs in and how it was started, for ``episode.json``."""

    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    info: dict[str, Any] = {
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "in_container": Path("/.dockerenv").exists(),
        "launch": {
            "host_command": os.environ.get("DEXBENCH_LAUNCH_COMMAND"),
            "host_repo": os.environ.get("DEXBENCH_HOST_REPO"),
            "image": os.environ.get("DEXBENCH_LAUNCH_IMAGE"),
            "log": os.environ.get("DEXBENCH_LAUNCH_LOG"),
        },
        "command": shlex.join([sys.executable, *sys.argv]),
        "argv": list(sys.argv),
        "cwd": os.getcwd(),
        "git": {
            "head": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": dirty.splitlines() if dirty else [],
            "assets_pin": _git("rev-parse", "HEAD:assets"),
        },
        "python": sys.version.split()[0],
        "packages": packages,
        "env": {k: v for k, v in sorted(os.environ.items()) if k.startswith("DEXBENCH_")},
    }
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return info


def replay_commands(hdf5: Path, run_info: dict[str, Any]) -> dict[str, str] | None:
    """``replay_paper_demo_physics.py`` for one episode: in the container, from the host repo root, and where it writes.
    None in a checkout without that script."""

    if not (REPO_ROOT / REPLAY_SCRIPT).is_file():
        return None
    try:
        rel = hdf5.resolve().relative_to(REPO_ROOT)
    except ValueError:
        rel = hdf5.resolve()
    args = ["/isaac-sim/python.sh", REPLAY_SCRIPT, "--viz", "none", "--hdf5", str(rel),
            "--out", str(rel.parent / "replay"), "--video"]
    args += [a for a in _environment_args(run_info.get("argv", [])) if a.startswith("-") or "=" not in a]
    steps = [f"cd {CONTAINER_REPO_ROOT}", shlex.join(args)]
    inner = " && ".join(steps)
    return {
        "container": inner,
        "host": f"DEXBENCH_REPRODUCIBLE=false ./docker/run_docker.sh -s paper-replay {shlex.quote(inner)}",
        "output": str(rel.parent / "replay" / rel.stem),
    }


class PaperDemoHdf5Recorder:
    """Record (state, action, done) rows and cut a new file at every reset."""

    FORMAT_VERSION = 1

    def __init__(self, env, out_dir: str | Path, get_sidecar: Callable[[], Any], *, robot_name: str = "robot",
                 channels: Sequence[Any] = (), scene_args: dict | None = None, paper_coupling: str | None = None):
        self.env = env
        self.scene_args = dict(scene_args or {})
        """The box size the scene was built with, stored as the ``scene_args`` attr so a replay builds the same box."""
        self.paper_coupling = paper_coupling
        """The environment's ``--paper_coupling``, for ``episode.json``."""
        self.channels = list(channels)
        """Extra per-row groups (``paper_demo_robot_channels.PaperDemoChannel``): ``key``, ``metadata`` and
        ``read(env)``, the fields of env 0 stored as ``<key>/<field>``. One that fails is dropped with a line
        printed, and the rest of the recording goes on."""
        self.out_dir = Path(out_dir)
        self.get_sidecar = get_sidecar
        self.robot_name = robot_name
        self.episode_index = 0
        self.enabled = True
        self._rows: dict[str, list] = {}
        self._pending: dict[str, Any] | None = None
        self._pending_end: str | None = None
        self._t0: float | None = None
        self._session_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.next_episode_stem: str | None = None
        """Folder and file stem for the next episode written, used once (a replay names its file after the
        demo it replays); None names it ``paper_demo_<session>_epNNN``."""
        self.video = None
        """Optional per-episode MP4 (``pack_invoice_video.PackInvoiceVideoRecorder``)."""
        self._episode_variations: dict[str, list[float]] = {}
        """What each enabled variation drew for the episode being recorded (``episode_variation_draws``)."""
        self.variation_overrides: dict[str, list[float]] = {}
        """Draws a tool placed by hand after the reset (``augment_paper_demo_layout.py`` moves the pile and the
        base itself), recorded in place of the variation's own for the next episodes."""
        self._episode_layout: str | None = None
        """The ``paper_layout`` attr as the episode's first row saw it (``_paper_layout``)."""
        self._physics_targets: list[np.ndarray] | None = None
        """The joint targets written to the solver during the pending row's physics steps (``_hook_physics_targets``)."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._hook_physics_targets()
        try:
            self.run_info = collect_run_info()
        except Exception as exc:
            self.run_info = {"error": repr(exc)}
        print(f"[paper_demo] recording one folder per episode into {self.out_dir}")


    def begin_step(self, action) -> None:
        """Capture the state the action is about to act on."""

        if not self.enabled:
            return
        try:
            if not self._rows:
                draws = episode_variation_draws(self.env, [0])
                self._episode_variations = {key: rows[0] for key, rows in draws.items()}
                self._episode_variations.update({k: list(v) for k, v in self.variation_overrides.items()})
                self._episode_layout = self._paper_layout()
            self._pending = self._capture(action)
            self._capture_video_frame()
            if self._physics_targets is not None:
                self._physics_targets = []
        except Exception as exc:
            self._disable(f"capture failed: {exc!r}")

    def end_step(self) -> None:
        """Attach the step's termination flags, store the row, cut the file if a reset happened."""

        if not self.enabled or self._pending is None:
            return
        try:
            row = self._pending
            self._pending = None
            tm = getattr(self.env, "termination_manager", None)
            for name in ("success", "sheet_dropped", "time_out"):
                row[f"terminations/{name}"] = self._term(tm, name)
            if self._physics_targets:
                row["robot/joint_pos_target_physics"] = np.stack(self._physics_targets)
                self._physics_targets = []
            for key, value in row.items():
                self._rows.setdefault(key, []).append(value)
            if self._pending_end is not None:
                reason, self._pending_end = self._pending_end, None
                self._finalize(reason)
        except Exception as exc:
            self._disable(f"append failed: {exc!r}")

    def on_reset(self, reason: str | None = None) -> None:
        """Episode boundary. Inside a step it is applied after the row is stored."""

        if not self.enabled:
            return
        reason = reason or self.reason_from_terminations()
        if self._pending is not None:
            self._pending_end = reason
            return
        try:
            self._finalize(reason)
        except Exception as exc:
            self._disable(f"finalize failed: {exc!r}")

    def close(self) -> None:
        if not self.enabled:
            return
        try:
            self._finalize("close")
        except Exception as exc:
            self._disable(f"close failed: {exc!r}")

    def reason_from_terminations(self) -> str:
        """Name the task termination that is currently raised, else ``xr_reset``."""

        tm = getattr(self.env, "termination_manager", None)
        for name in ("success", "sheet_dropped", "time_out"):
            if self._term(tm, name):
                return name
        return "xr_reset"


    @staticmethod
    def _term(tm, name: str) -> bool:
        if tm is None or name not in getattr(tm, "active_terms", ()):
            return False
        return bool(_to_np(tm.get_term(name)).any())

    def _capture(self, action) -> dict[str, Any]:
        env = self.env
        now = time.perf_counter()
        if self._t0 is None:
            self._t0 = now
        row: dict[str, Any] = {
            "time_s": np.float64(now - self._t0),
            "sim_step": np.int64(getattr(env, "common_step_counter", 0)),
            "action": _to_np(action).reshape(-1).astype(np.float32),
        }
        robot = env.scene[self.robot_name] if self.robot_name in env.scene.keys() else None
        if robot is not None:
            data = robot.data
            row["robot/joint_pos"] = _to_np(data.joint_pos)[0].astype(np.float32)
            row["robot/joint_vel"] = _to_np(data.joint_vel)[0].astype(np.float32)
            row["robot/joint_pos_target"] = _to_np(data.joint_pos_target)[0].astype(np.float32)
            row["robot/body_pose_w"] = _to_np(data.body_link_pose_w)[0].astype(np.float32)
        sidecar = self.get_sidecar()
        if sidecar is not None:
            stack = sidecar.stack
            q = _to_np(stack.state_0.particle_q).astype(np.float32)
            qd = _to_np(stack.state_0.particle_qd).astype(np.float32)
            n = stack.num_sheets
            centroid = np.zeros((n, 3), dtype=np.float32)
            speed = np.zeros(n, dtype=np.float32)
            in_box = np.zeros(n, dtype=bool)
            for i in range(n):
                sl = stack.sheet_slice(i)
                centroid[i] = q[sl].mean(axis=0)
                speed[i] = np.linalg.norm(qd[sl], axis=1).max()
                in_box[i] = stack.sheet_in_box(i)
            row["paper/particle_q"] = q
            row["paper/particle_qd"] = qd
            row["paper/sheet_centroid"] = centroid
            row["paper/sheet_max_speed"] = speed
            row["paper/sheet_in_box"] = in_box
            coupled = getattr(sidecar, "coupled_solver", None)
            mapping = getattr(coupled, "body_mapping", None)
            if coupled is not None and mapping:
                ids = np.array([mapping[k] for k in sorted(mapping)], dtype=np.int64)
                row["coupling/wrench"] = _to_np(coupled.coupling_forces)[ids].astype(np.float32)
        for channel in list(self.channels):
            try:
                for field, value in channel.read(env).items():
                    row[f"{channel.key}/{field}"] = _to_np(value)[0].astype(np.float32)
            except Exception as exc:
                print(f"[paper_demo] {channel.key} not recorded from here on: {exc!r}")
                self.channels.remove(channel)
                for key in [k for k in row if k.startswith(f"{channel.key}/")]:
                    del row[key]
        return row

    def _capture_video_frame(self) -> None:
        if self.video is None:
            return
        try:
            if not self.video.is_open:
                self.video.open(self.out_dir / f".recording_{self._session_tag}.mp4")
            self.video.capture()
        except Exception as exc:
            print(f"[paper_demo] video disabled: {exc!r}")
            self.video = None

    def _finish_video(self, path: Path | None) -> None:
        """Close the episode's MP4 and name it after its HDF5 (``None``: an empty episode, discard)."""

        if self.video is None or not self.video.is_open:
            return
        try:
            tmp = self.video.output_path
            self.video.close()
            if path is None or self.video.frames == 0:
                tmp.unlink(missing_ok=True)
            else:
                tmp.replace(path)
                print(f"[paper_demo] wrote {path.name}: {self.video.frames} frames")
        except Exception as exc:
            print(f"[paper_demo] video disabled: {exc!r}")
            self.video = None


    def _finalize(self, reason: str) -> None:
        rows, self._rows = self._rows, {}
        t0, self._t0 = self._t0, None
        n = len(rows.get("time_s", []))
        if n == 0:
            self._finish_video(None)
            return

        self.episode_index += 1
        stem = self.next_episode_stem or f"paper_demo_{self._session_tag}_ep{self.episode_index:03d}"
        self.next_episode_stem = None
        episode_dir = self.out_dir / stem
        episode_dir.mkdir(parents=True, exist_ok=True)
        path = episode_dir / f"{stem}.hdf5"
        env = self.env
        with h5py.File(path, "w") as f:
            f.attrs["format_version"] = self.FORMAT_VERSION
            f.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
            f.attrs["episode_index"] = self.episode_index
            f.attrs["control_dt"] = float(getattr(env, "step_dt", 0.0))
            f.attrs["num_steps"] = n
            f.attrs["duration_s"] = float(rows["time_s"][-1] - rows["time_s"][0])
            f.attrs["end_reason"] = reason
            for name in ("success", "sheet_dropped", "time_out"):
                flags = rows.get(f"terminations/{name}")
                f.attrs[name] = bool(flags and flags[-1])
            self._write_metadata(f)
            if self.scene_args:
                f.attrs["scene_args"] = json.dumps(self.scene_args)
            for key, value in self._episode_variations.items():
                f.create_dataset(f"variations/{key}", data=np.asarray([value], dtype=np.float32))
            f.attrs["run_info"] = json.dumps(_jsonable(self.run_info))
            for channel in self.channels:
                f.attrs[channel.key] = json.dumps(_jsonable(channel.metadata))
            for key, values in rows.items():
                if len(values) != n:
                    print(f"[paper_demo] {key} left out of {stem}: {len(values)} of {n} rows")
                    continue
                try:
                    arr = np.stack([np.asarray(v) for v in values])
                except ValueError:
                    print(f"[paper_demo] {key} left out of {stem}: its rows differ in shape")
                    continue
                kwargs = {"compression": "gzip", "compression_opts": 4} if arr.ndim >= 2 and arr.size > 4096 else {}
                f.create_dataset(key, data=arr, **kwargs)
            sidecar = self.get_sidecar()
            if sidecar is not None:
                stack = sidecar.stack
                particle_sheet = np.concatenate(
                    [np.full(stack.sheet_slice(i).stop - stack.sheet_slice(i).start, i, dtype=np.int32)
                     for i in range(stack.num_sheets)]
                ) if stack.num_sheets else np.zeros(0, dtype=np.int32)
                f.create_dataset("paper/particle_sheet", data=particle_sheet)
        duration = float(rows["time_s"][-1] - rows["time_s"][0])
        print(f"[paper_demo] wrote {path.relative_to(self.out_dir)}: {n} steps, {duration:.1f} s, end={reason}")
        self._finish_video(path.with_suffix(".mp4"))
        self._write_episode_card(path, rows, reason, duration)

    def _write_episode_card(self, path: Path, rows: dict[str, list], reason: str, duration: float) -> None:
        """``episode.json`` and, where the checkout has the replay script, ``replay_physics.sh`` next to the episode's
        HDF5."""

        sidecar = self.get_sidecar()
        paper_cfg = _jsonable(sidecar.stack.cfg) if sidecar is not None else {}
        paper_cfg = paper_cfg if isinstance(paper_cfg, dict) else {}
        replay = replay_commands(path, self.run_info)
        card = {
            "episode": path.stem,
            "files": sorted(p.name for p in path.parent.iterdir() if p.is_file()),
            "outcome": {
                "end_reason": reason,
                **{name: bool(rows.get(f"terminations/{name}") and rows[f"terminations/{name}"][-1])
                   for name in ("success", "sheet_dropped", "time_out")},
                "num_steps": len(rows["time_s"]),
                "duration_s": round(duration, 3),
                "control_dt": float(getattr(self.env, "step_dt", 0.0)),
            },
            "paper": {
                "coupling": self.paper_coupling,
                "resolution": paper_cfg.get("resolution"),
                "num_sheets": getattr(getattr(sidecar, "stack", None), "num_sheets", None),
            },
            "variations": self._episode_variations,
            "run": self.run_info,
        }
        if replay is not None:
            card["replay"] = {
                **replay,
                "host_repo": self.run_info.get("launch", {}).get("host_repo"),
                "note": "replay_paper_demo_physics.py rebuilds the paper from the HDF5 attrs and drives the robot "
                        "with robot/joint_pos_target; summary.json in the output compares it with this recording.",
            }
        (path.parent / "episode.json").write_text(json.dumps(_jsonable(card), indent=2) + "\n")
        if replay is None:
            return
        script = path.parent / "replay_physics.sh"
        host_repo = card["replay"]["host_repo"]
        script.write_text(
            "#!/usr/bin/env bash\n"
            f"# Physics replay of {path.stem}; writes {replay['output']}/ (summary.json, HDF5, MP4).\n"
            "# Runs on the host, from the DexBench checkout (DEXBENCH_REPO, else the recording's).\n"
            "set -euo pipefail\n"
            f"cd \"${{DEXBENCH_REPO:-{host_repo or '.'}}}\"\n"
            f"{replay['host']}\n"
        )
        script.chmod(0o755)

    def _write_metadata(self, f) -> None:
        env = self.env
        robot = env.scene[self.robot_name] if self.robot_name in env.scene.keys() else None
        if robot is not None:
            f.attrs["robot_joint_names"] = json.dumps(list(robot.joint_names))
            f.attrs["robot_body_names"] = json.dumps(list(robot.body_names))
        sidecar = self.get_sidecar()
        if sidecar is None:
            return
        stack = sidecar.stack
        f.attrs["paper_cfg"] = json.dumps(_jsonable(stack.cfg))
        layout, self._episode_layout = self._episode_layout, None
        f.attrs["paper_layout"] = layout if layout is not None else self._paper_layout()
        f.attrs["paper_layout_at"] = PAPER_LAYOUT_AT_EPISODE_START
        coupled = getattr(sidecar, "coupled_solver", None)
        if coupled is not None:
            f.attrs["coupling_cfg"] = json.dumps(_jsonable(coupled.cfg))
            mapping = getattr(coupled, "body_mapping", {}) or {}
            f.attrs["coupling_body_names"] = json.dumps(sorted(mapping))

    def _hook_physics_targets(self) -> None:
        """Keep the joint target the robot hands the solver on each physics step of a recorded row."""
        env = self.env
        scene = getattr(env, "scene", None)
        decimation = int(getattr(getattr(env, "cfg", None), "decimation", 0) or 0)
        if scene is None or not hasattr(scene, "write_data_to_sim") or decimation < 1:
            return
        if getattr(env, "_physics_handles_decimation", False):
            print("[paper_demo] per-physics-step joint targets not recorded: Newton folds the decimation loop")
            return
        self._physics_targets = []
        write = scene.write_data_to_sim

        def write_and_keep(*args, **kwargs):
            out = write(*args, **kwargs)
            targets = self._physics_targets
            if self.enabled and self._pending is not None and targets is not None and len(targets) < decimation:
                try:
                    robot = scene[self.robot_name]
                    targets.append(_to_np(robot.data.joint_pos_target)[0].astype(np.float32))
                except Exception as exc:
                    print(f"[paper_demo] per-physics-step joint targets not recorded from here on: {exc!r}")
                    self._physics_targets = None
            return out

        scene.write_data_to_sim = write_and_keep

    def _paper_layout(self) -> str | None:
        """The sidecar's piles and carton now, as the ``paper_layout`` JSON attr."""

        sidecar = self.get_sidecar()
        if sidecar is None:
            return None
        stack = sidecar.stack
        return json.dumps(
            _jsonable(
                {
                    "sites": list(stack.sites),
                    "carton": sidecar.carton,
                    "num_sheets": stack.num_sheets,
                    "particles_per_sheet": stack.particles_per_sheet,
                }
            )
        )

    def _disable(self, why: str) -> None:
        print(f"[paper_demo] recording disabled: {why}")
        self.enabled = False
        self._rows = {}
        self._pending = None
