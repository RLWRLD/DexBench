# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""DB-H2 (Lift and Pour): lift the jug with the right hand and pour its 100 mL of water into the wine glass."""

from __future__ import annotations

import json
from dataclasses import MISSING
from pathlib import Path

import numpy as np
import torch
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import TerminationTermCfg
from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import matrix_from_quat
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.metric_term_cfg import MetricTermCfg
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg

from dexbench_arena import ASSETS_DIR
from dexbench_arena.fluids.pour_evaluation import Cavity, measure
from dexbench_arena.fluids.pour_jugs import GOBLET_PROFILE, jug as pour_jug
from dexbench_arena.fluids.water_pour_setup import WaterPourSidecar, WaterPourSidecarCfg, refuse_water_environment
from dexbench_arena.tasks.predicates.hold import sustained
from dexbench_arena.tasks.task_termination import task_termination_cfg

POUR_CONFIG = Path(__file__).with_name("hotel_pour_100ml.json")
"""The water: 100 mL, 4 mm particles, 8 mm voxels, 40 solver iterations, the goblet's 6 mm collider margin."""
POUR_MIN_ML = 90.0
"""The goblet's volume for success: at least this [mL]. No upper bound: the jug holds the 100 mL seed, whose particles
sum to a hair over 100 mL (100.000002), so an upper bound of 100 would fail a pour that puts every particle in the
goblet."""
POUR_HOLD_S = 5.0
"""How long the goblet holds it [s]."""
SPILL_LIMIT_ML = 5.0
"""Spilled water at or above this fails the episode [mL]."""
COUNTER_TOP_Z_M = 0.80
"""The serving trolley's top [m, world]: the surface the jug and the goblet stand on, and the water's catch shell
(the sidecar's default is a 0.90 m hotel counter). Fixed, not measured from the goblet, so the generator
and the replay build the same shell whatever the draw."""
JUG_DROPPED_Z_M = 0.70
"""The jug's root below this height is dropped (the counter top is about 0.80 m)."""
WATER_SUBSTEP_S = 1.0 / 180.0
"""The water's substep: 6 substeps per 1/30 s command."""
WATER_SETTLE_S = 2.0
"""Water-only settling after the seed, before the episode's first step (the collection's ``--water_settle_s``)."""
GOBLET = "pour_goblet"


def _pose(env, name: str) -> tuple[np.ndarray, np.ndarray]:
    """``name``'s world position and rotation matrix (env 0)."""
    pose = env.scene[name].data.root_pose_w
    pose = pose if isinstance(pose, torch.Tensor) else pose.torch
    pose = pose[0].detach().to(torch.float64)
    return pose[:3].cpu().numpy(), matrix_from_quat(pose[3:7].unsqueeze(0))[0].cpu().numpy()


class PourTaskWater:
    """The episode's water and its measured volumes (env 0: the pour runs one env)."""

    def __init__(self, env, jug_key: str):
        if env.num_envs != 1:
            raise ValueError(f"DB-H2's water runs one env, got {env.num_envs}")
        refuse_water_environment("DB-H2's task (lift_and_pour)")
        self.env, self.measure = env, measure
        self.jug = pour_jug(jug_key)
        self.axis, _, _ = self.jug.cavity()
        cfg = json.loads(POUR_CONFIG.read_text())
        goblet_profile = ASSETS_DIR / GOBLET_PROFILE
        self.cup, self.jug_cavity = Cavity.load(goblet_profile), Cavity.load(ASSETS_DIR / self.jug.profile)
        self.substeps = max(1, round(env.step_dt / WATER_SUBSTEP_S))
        sidecar_cfg = WaterPourSidecarCfg(
            vessel_asset=self.jug.scene_name, cup_asset=GOBLET, profile_npz=str(ASSETS_DIR / self.jug.profile),
            axis_xy_local=tuple(self.axis[:2]), voxel_size=cfg["voxel_m"],
            cup_profile_npz=str(goblet_profile), particles_per_cell=cfg["voxel_m"] / cfg["spacing_m"],
            max_iterations=cfg["max_iterations"], substeps=self.substeps, cup_wall_height=float(self.cup.z[-1]),
            cup_margin=float(cfg["cup_collider_margin_m"]), target_ml=float(cfg["initial_ml"]),
            counter_top_z=COUNTER_TOP_Z_M)
        self.sidecar = WaterPourSidecar(env, sidecar_cfg)
        if "camera_obs" in env.observation_manager.active_terms:
            self.sidecar.enable_render("surface")
        self.density = float(cfg["density_kg_m3"])
        self.masses = self.sidecar._nm._mpm_model.particle_mass.numpy().astype(float)
        self.last: dict = {}
        self.failure: str | None = None
        self.step_counter = -1
        self.episode = env.get_episode_index(0)
        self.settle()

    def settle(self) -> None:
        """Let the fresh water settle and take the episode's first measurement, checked for a failure as every later
        one is: ``reset()`` moves ``step_counter`` to the current step, so that step's ``update()`` measures nothing."""
        for _ in range(round(WATER_SETTLE_S / self.env.step_dt)):
            self.sidecar.step(self.env.step_dt)
        self.observe()
        self.latch_failure()

    def reset(self) -> None:
        """Seed the water anew in the jug where the episode's reset put it, and let it settle."""
        self.episode = self.env.get_episode_index(0)
        self.failure = None
        self.sidecar.reset()
        self.masses = self.sidecar._nm._mpm_model.particle_mass.numpy().astype(float)
        self.settle()
        self.step_counter = int(self.env.common_step_counter)

    def update(self) -> dict:
        """Step the water once per env step (a second call in the same step reads the same measurement); the volumes
        are printed once a second of simulated time, so a replay or an evaluation log says where the water went."""
        if int(self.env.common_step_counter) != self.step_counter:
            self.step_counter = int(self.env.common_step_counter)
            self.sidecar.step(self.env.step_dt)
            self.observe()
            self.latch_failure()
            step = int(self.env.episode_length_buf[0])
            if step % max(1, round(1.0 / self.env.step_dt)) == 0:
                print(f"[pour] t={step * self.env.step_dt:5.1f}s goblet {self.last['target_ml']:5.1f} jug {self.last['source_ml']:5.1f} "
                      f"spilled {self.last['spilled_ml']:4.1f} mL jug z {self.last['jug_z_m']:.3f}", flush=True)
        return self.last

    def latch_failure(self) -> None:
        """Remember the episode's first failure (the water spilled, the jug dropped), measured each env step: from then
        on the pour cannot count, whatever the goblet holds later."""
        if self.failure is None:
            if self.last["spilled_ml"] >= SPILL_LIMIT_ML:
                self.failure = "water_spilled"
            elif self.last["jug_z_m"] < JUG_DROPPED_Z_M:
                self.failure = "jug_dropped"

    def counts(self) -> bool:
        """The last measurement counts toward success: poured (``poured``) and no failure in the episode so far."""
        return self.failure is None and poured(self.last)

    def observe(self) -> None:
        cup_p, cup_r = _pose(self.env, GOBLET)
        jug_p, jug_r = _pose(self.env, self.jug.scene_name)
        positions = self.sidecar.positions() - cup_p
        row, _ = self.measure(positions, self.masses, self.density, self.cup, (np.zeros(3), cup_r),
                              self.jug_cavity, (jug_p + jug_r @ self.axis - cup_p, jug_r))
        self.last = dict(row, jug_z_m=float(jug_p[2]))


def episode_water(env, jug_key: str) -> PourTaskWater:
    """The episode's water: built at the first step, seeded anew at the first step of every later episode."""
    water = getattr(env, "dexbench_pour_water", None)
    if water is None:
        water = env.dexbench_pour_water = PourTaskWater(env, jug_key)
    elif water.episode != env.get_episode_index(0):
        water.reset()
    return water


def _all(env, value: bool) -> torch.Tensor:
    return torch.full((env.num_envs,), value, dtype=torch.bool, device=env.device)


def poured(volume: dict) -> bool:
    """A measurement that counts toward success: at least ``POUR_MIN_ML`` in the goblet, under ``SPILL_LIMIT_ML``
    spilled."""
    return volume["target_ml"] >= POUR_MIN_ML and volume["spilled_ml"] < SPILL_LIMIT_ML


def pour_held(env, jug_key: str, water: bool) -> torch.Tensor:
    """Success: at least 90 mL in the goblet held for 5 s with under 5 mL spilled, and no failure earlier in the
    episode (``PourTaskWater.counts``). ``water`` False: never (no water)."""
    if not water:
        return _all(env, False)
    pour = episode_water(env, jug_key)
    pour.update()
    return sustained(env, _all(env, pour.counts()), POUR_HOLD_S, key="pour_held")


def water_spilled(env, jug_key: str, water: bool) -> torch.Tensor:
    """Failure: the spilled water reached the limit."""
    return _all(env, water and episode_water(env, jug_key).update()["spilled_ml"] >= SPILL_LIMIT_ML)


def jug_dropped(env, jug_name: str) -> torch.Tensor:
    """Failure: the jug's root is below ``JUG_DROPPED_Z_M``."""
    return _all(env, bool(_pose(env, jug_name)[0][2] < JUG_DROPPED_Z_M))


@configclass
class LiftAndPourTerminationsCfg:
    """Success and failure terms; the episode limit is the task's ``timeout_s``."""

    success: TerminationTermCfg = MISSING
    water_spilled: TerminationTermCfg = MISSING
    jug_dropped: TerminationTermCfg = MISSING


class GobletVolumeRecorder(RecorderTerm):
    """The mL in the goblet at the episode's end, as the success term last measured it (0 without water)."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.name = cfg.name

    def record_pre_reset(self, env_ids):
        water = getattr(self._env, "dexbench_pour_water", None)
        ml = float(water.last["target_ml"]) if water is not None and water.last else 0.0
        return self.name, torch.full((len(env_ids),), ml, dtype=torch.float32, device=self._env.device)


@configclass
class GobletVolumeRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = GobletVolumeRecorder
    name: str = "goblet_ml"


def compute_mean_goblet_ml(recorded_metric_data: list[np.ndarray]) -> float:
    """The episodes' mean goblet volume at their end [mL]."""
    if not recorded_metric_data:
        return 0.0
    return float(np.mean(np.concatenate([np.asarray(v, dtype=float).reshape(-1) for v in recorded_metric_data])))


class GobletVolumeMetric(MetricBase):
    """The mean mL the goblet holds when an episode ends: how near a failed pour came, beside the success rate."""

    name = "goblet_ml"
    recorder_term_name = "goblet_ml"

    def get_recorder_term_cfg(self) -> RecorderTermCfg:
        return GobletVolumeRecorderCfg(name=self.recorder_term_name)

    def get_metric_term_cfg(self) -> MetricTermCfg:
        return MetricTermCfg(compute_metric_func=compute_mean_goblet_ml, params={},
                             recorder_term_name=self.recorder_term_name)


class DexBenchLiftAndPourTask(TaskBase):
    """DB-H2's task: the water, its success and failure terms, the time limit and its metrics (no scene or events of
    its own)."""

    name = "lift_and_pour"

    def __init__(self, jug_key: str, episode_length_s: float | None = None, water: bool = True):
        """``water`` False builds no water (a condition recording: the hold-pose policy checks that the start comes
        to rest, and nothing pours), so the success term never fires."""
        super().__init__(
            episode_length_s=episode_length_s,
            task_description="Lift the jug with the right hand and pour its 100 mL of water into the wine glass.",
        )
        self.jug_key = jug_key
        self.termination_cfg = LiftAndPourTerminationsCfg(
            success=TerminationTermCfg(func=pour_held, params={"jug_key": jug_key, "water": water}),
            water_spilled=TerminationTermCfg(func=water_spilled, params={"jug_key": jug_key, "water": water}),
            jug_dropped=TerminationTermCfg(func=jug_dropped, params={"jug_name": pour_jug(jug_key).scene_name}),
        )

    def get_termination_cfg(self) -> TaskTerminationCfg:
        return task_termination_cfg(self.episode_length_s, self.termination_cfg)

    def get_scene_cfg(self):
        return None

    def get_events_cfg(self):
        return None

    def get_mimic_env_cfg(self, arm_mode):
        return None

    def get_metrics(self) -> list[MetricBase]:
        """The success rate, as every DexBench task reports it, and the goblet's mean mL at the episodes' end."""
        return [SuccessRateMetric(), GobletVolumeMetric()]

    def get_viewer_cfg(self) -> ViewerCfg:
        return ViewerCfg(eye=(-1.5, -1.5, 1.5), lookat=(0.0, 0.0, 0.5))
