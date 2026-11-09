# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Sense the contact force on each fingertip; observe it, and record net, normal and tangential."""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import RecorderTerm, RecorderTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.configclass import configclass
from isaaclab_contrib.custom_coupling.newton_manager_cfg import CoupledMJWarpVBDSolverCfg

from dexbench_arena.embodiments.episode_metadata import install_episode_metadata
from dexbench_arena.embodiments.recorder_terms import add_declared_recorder_term

FINGERTIP_CONTACT_SENSOR_NAME = "fingertip_contact"
FINGERTIP_CONTACT_DATASET_KEY = "fingertip_contact"
FINGERTIP_CONTACT_SCHEMA_VERSION = 1


def fingertip_contact_sensor_cfg(robot_prim_path: str, body_names: Sequence[str]) -> ContactSensorCfg:
    """The one fingertip contact sensor: ``body_names`` under the robot, against everything they touch."""
    return ContactSensorCfg(prim_path=f"{robot_prim_path}/({'|'.join(body_names)})")


def fingertip_contact_force(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Normal contact force magnitude on each of ``sensor_cfg``'s bodies [N], (num_envs, bodies)."""
    sensor = env.scene.sensors[sensor_cfg.name]
    return sensor.data.net_normal_forces_w.torch[:, sensor_cfg.body_ids].norm(dim=-1)


def fingertip_contact_metadata(body_names: Sequence[str]) -> dict:
    """What the recorded arrays mean, for the dataset's episode metadata."""
    return {
        "schema_version": FINGERTIP_CONTACT_SCHEMA_VERSION,
        "body_names": list(body_names),
        "fields": ["net_force", "normal_force", "tangential_force"],
        "shape": "(steps, bodies, 3)",
        "unit": "N",
        "frame": "body link frame",
        "sampled": "instantaneous, at the last physics substep of each env step",
        "source": "Isaac Lab ContactSensor on Newton: SensorContact total_force and total_force_friction",
    }


def uses_coupled_vbd(env_cfg) -> bool:
    """Whether the env runs Isaac Lab's coupled MJWarp + VBD manager (deformable objects)."""
    physics = getattr(getattr(env_cfg, "sim", None), "physics", None)
    return isinstance(getattr(physics, "solver_cfg", None), CoupledMJWarpVBDSolverCfg)


class FingertipContactReader:
    """The net, normal and tangential contact force on each of ``body_names``, in its link frame."""

    def __init__(self, body_names: Sequence[str]):
        self.body_names = tuple(body_names)
        self._rows: torch.Tensor | None = None
        self._bodies: torch.Tensor | None = None

    def _resolve(self, sensor, robot, device) -> None:
        sensor_names = list(sensor.sensor_names)
        missing = [name for name in self.body_names if name not in sensor_names]
        if missing:
            raise RuntimeError(f"Fingertip contact sensor has no rows for {missing} (sensed: {sensor_names})")
        self._rows = torch.tensor([sensor_names.index(name) for name in self.body_names], device=device)
        body_ids, _ = robot.find_bodies(list(self.body_names), preserve_order=True)
        self._bodies = torch.tensor(body_ids, device=device)

    def read(self, sensor, robot) -> dict[str, torch.Tensor]:
        net_w = sensor.data.net_forces_w.torch
        if self._rows is None:
            self._resolve(sensor, robot, net_w.device)
        tangential_w = _friction_forces_w(sensor)
        net_w = net_w.index_select(1, self._rows)
        tangential_w = tangential_w.index_select(1, self._rows)
        quat_w = robot.data.body_link_quat_w.torch.index_select(1, self._bodies)
        net = math_utils.quat_apply_inverse(quat_w, net_w)
        tangential = math_utils.quat_apply_inverse(quat_w, tangential_w)
        return {"net_force": net, "normal_force": net - tangential, "tangential_force": tangential}


class FingertipContactRecorder(RecorderTerm):
    """Per-step fingertip contact force in each sensing body's link frame."""

    def __init__(self, cfg: FingertipContactRecorderCfg, env):
        super().__init__(cfg, env)
        self._reader = FingertipContactReader(cfg.body_names)
        self._metadata_installed = False

    def record_post_step(self):
        if not self._metadata_installed:
            install_episode_metadata(
                self._env, FINGERTIP_CONTACT_DATASET_KEY, fingertip_contact_metadata(self._reader.body_names)
            )
            self._metadata_installed = True
        sensor = self._env.scene.sensors[FINGERTIP_CONTACT_SENSOR_NAME]
        return FINGERTIP_CONTACT_DATASET_KEY, self._reader.read(sensor, self._env.scene["robot"])


def _friction_forces_w(sensor) -> torch.Tensor:
    """The friction (tangential) part of each sensed body's net contact force, world frame."""
    friction_w = sensor.data.net_friction_forces_w
    if friction_w is None:
        raise RuntimeError(
            "Fingertip contact recording needs the Newton contact sensor's friction sum "
            "(ContactSensorCfg(track_friction_forces=True)); this contact sensor does not provide it."
        )
    return friction_w.torch


@configclass
class FingertipContactRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = FingertipContactRecorder
    body_names: list[str] = []
    """Sensing bodies, in the order of the recorded rows."""


class FingertipContactMixin:
    """Embodiment capability: a fingertip contact sensor, and an opt-out recorder on it."""

    fingertip_contact_body_names: tuple[str, ...] = ()
    record_fingertip_contact: bool = False
    fingertip_contact_required: bool = False

    def enable_fingertip_contact_recording(self, required: bool = False) -> None:
        """Record this embodiment's fingertip contact sensor in its datasets."""
        if not self.fingertip_contact_body_names:
            raise ValueError(f"{self.name} declares no fingertip contact bodies")
        if getattr(self.scene_config, FINGERTIP_CONTACT_SENSOR_NAME, None) is None:
            raise ValueError(f"{self.name}'s scene declares no {FINGERTIP_CONTACT_SENSOR_NAME} sensor")
        self.record_fingertip_contact = True
        self.fingertip_contact_required = bool(required)

    def configure_fingertip_contact_recorders(self, env_cfg) -> None:
        """Put the recorder term next to the embodiment's other per-step terms (``add_declared_recorder_term``)."""
        coupled_vbd = uses_coupled_vbd(env_cfg)
        if coupled_vbd:
            _drop_fingertip_contact_sensor(env_cfg)
        if not self.record_fingertip_contact:
            return
        if coupled_vbd:
            message = (
                "fingertip contact force is not recorded with deformables (coupled MJWarp + VBD): that manager "
                "refuses contact sensors, and VBD's soft-contact forces need a converged solve to mean a force"
                ""
            )
            if self.fingertip_contact_required:
                raise ValueError(f"--record_fingertip_contact: {message}")
            print(f"[dexbench] --record_fingertip_contact is off: {message}", flush=True)
            return
        getattr(env_cfg.scene, FINGERTIP_CONTACT_SENSOR_NAME).track_friction_forces = True
        term = FingertipContactRecorderCfg(body_names=list(self.fingertip_contact_body_names))
        add_declared_recorder_term(env_cfg, "record_post_step_fingertip_contact", term)


def _drop_fingertip_contact_sensor(env_cfg) -> None:
    """Remove the sensor and the observation terms that read it (an observation group skips a None term)."""
    setattr(env_cfg.scene, FINGERTIP_CONTACT_SENSOR_NAME, None)
    observations = getattr(env_cfg, "observations", None)
    for group in vars(observations).values() if observations is not None else ():
        for name, term in list(vars(group).items()):
            params = getattr(term, "params", None)
            if isinstance(params, dict) and getattr(params.get("sensor_cfg"), "name", None) == FINGERTIP_CONTACT_SENSOR_NAME:
                setattr(group, name, None)
