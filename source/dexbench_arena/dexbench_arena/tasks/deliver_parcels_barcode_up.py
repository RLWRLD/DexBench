# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Deliver-the-parcels-barcode-up task for the DexBench E2 logistics centre (DB-L4)."""

from __future__ import annotations

from dataclasses import MISSING
from pathlib import Path
from typing import Any

import numpy as np
import torch
import warp as wp

from isaaclab.managers import EventTermCfg, ManagerTermBase, TerminationTermCfg
from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply

from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.metric_term_cfg import MetricTermCfg
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg
from isaaclab_arena.utils.configclass import make_configclass

from dexbench_arena.assets.e2_logistics_cell import E2_L4_PARCELS_BY_NAME, polybag_size_scale
from dexbench_arena.assets.polybag.polybag_object_cfg import PolybagObjectCfg, PolybagVisualCfg
from dexbench_arena.assets.polybag.template import load_cfg_template, polybag_particle_masses
from dexbench_arena.contact_materials import HandContactMaterialCfg
from dexbench_arena.layouts.e2_logistics_l4_layout import AXIS_BY_NAME, EPISODE_LENGTH_S
from dexbench_arena.evaluation.replay_guards import deterministic_reset
from dexbench_arena.tasks.parcel_relay import DELIVERED, FAILED, PENDING, ParcelRelay
from dexbench_arena.tasks.task_termination import task_termination_cfg


VERDICT_WORDS = {DELIVERED: "ok", FAILED: "FAILED", PENDING: "not put down"}


class L4Parcel:
    """One parcel of the episode as the relay moves it: its entity, where it waits, where it is put down, and what
    it weighs."""

    def __init__(self, entity_name: str, name: str, park_pose: tuple, stage_pose: tuple, mass_kg: float):
        parcel = E2_L4_PARCELS_BY_NAME[name]
        self.entity_name = entity_name
        self.name = name
        self.is_polybag = parcel.is_polybag
        self.mass_kg = float(mass_kg)
        """[kg]. A bag's is set on its cfg here; a box's on its scene asset, which the environment spawns."""
        self.barcode_axis = AXIS_BY_NAME[parcel.barcode_axis]
        self.collider_size_xyz_m = parcel.collider_size_xyz_m
        self.park_pose = tuple(float(v) for v in park_pose)
        """(x, y, z, qx, qy, qz, qw) in the environment frame."""
        self.stage_pose = tuple(float(v) for v in stage_pose)


def _box_corners(size_xyz_m, device) -> torch.Tensor:
    """(8, 3) corners of a box parcel's collider in its own frame (centred in XY, standing on its origin)."""
    sx, sy, sz = size_xyz_m
    xs, ys, zs = (-sx / 2, sx / 2), (-sy / 2, sy / 2), (0.0, sz)
    return torch.tensor([(x, y, z) for x in xs for y in ys for z in zs], dtype=torch.float32, device=device)


def best_fit_rotation(rest: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """(n, 3, 3) the rotations that best carry ``rest`` (k, 3) onto each of ``points`` (n, k, 3) (Kabsch)."""
    a = rest - rest.mean(dim=0)
    b = points - points.mean(dim=1, keepdim=True)
    u, _, vt = torch.linalg.svd(a.T[None] @ b)
    d = torch.sign(torch.linalg.det(vt.transpose(1, 2) @ u.transpose(1, 2)))
    fix = torch.diag_embed(torch.stack((torch.ones_like(d), torch.ones_like(d), d), dim=-1))
    return vt.transpose(1, 2) @ fix @ u.transpose(1, 2)


@deterministic_reset
class relay_parcels(ManagerTermBase):
    """Every step: judge the parcel that is out, take judged ones off, put the next one down."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.parcels: list[L4Parcel] = cfg.params["parcels"]
        self.relay = ParcelRelay(env.num_envs, len(self.parcels), cfg.params["belt_speed_m_s"], device=env.device)
        self.stage_poses = torch.tensor([p.stage_pose for p in self.parcels], dtype=torch.float32,
                                        device=env.device)[None].repeat(env.num_envs, 1, 1)
        self.box_corners = {p.entity_name: _box_corners(p.collider_size_xyz_m, env.device)
                            for p in self.parcels if not p.is_polybag}
        """A box's collider corners in its own frame, by entity: what its geometry is read from."""
        self.rest_points = {p.entity_name: torch.as_tensor(env.scene[p.entity_name].template.rest, dtype=torch.float32,
                                                           device=env.device)
                            for p in self.parcels if p.is_polybag}
        """A bag's rest particles in its own frame, by entity: what its rotation is fitted to."""

    def reset(self, env_ids=None) -> None:
        self.relay.reset(env_ids)
        for parcel in self.parcels:
            if parcel.is_polybag:
                self._env.scene[parcel.entity_name].write_active_to_sim_index(False, env_ids)

    def _geometry(self, env, parcel: L4Parcel) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """(max x, lowest z, centre y, barcode-up cosine) of one parcel in every environment, environment frame: a
        box's from its collider's corners, a bag's from its particles."""
        origins = env.scene.env_origins
        asset = env.scene[parcel.entity_name]
        axis = torch.tensor(parcel.barcode_axis, dtype=torch.float32, device=env.device)
        if parcel.is_polybag:
            points = asset.data.particle_pos_w.torch - origins[:, None, :]
            rotation = best_fit_rotation(self.rest_points[parcel.entity_name], points)
            barcode = rotation @ axis
        else:
            pose = wp.to_torch(asset.data.root_pose_w)
            pos, quat = pose[:, :3] - origins, pose[:, 3:7]
            corners = self.box_corners[parcel.entity_name]
            n, k = len(pos), len(corners)
            points = quat_apply(quat[:, None, :].expand(n, k, 4), corners[None].expand(n, k, 3)) + pos[:, None, :]
            barcode = quat_apply(quat, axis[None].expand(n, 3))
        return (points[..., 0].max(dim=1).values, points[..., 2].min(dim=1).values, points[..., 1].mean(dim=1),
                barcode[:, 2])

    def _report(
        self, judged: torch.Tensor, now: torch.Tensor, parcels: list[L4Parcel], end_on_parcel_failure: bool
    ) -> None:
        """Print the verdicts given this step, and the episode of every environment they ended."""
        verdict = self.relay.verdict.tolist()
        for env_id, i in judged.nonzero().tolist():
            word = "delivered" if verdict[env_id][i] == DELIVERED else "FAILED"
            print(f"[dexbench] DB-L4 relay: env {env_id} {parcels[i].entity_name} {parcels[i].name} {word} "
                  f"at {float(now[env_id]):.2f} s", flush=True)
        for env_id in judged.any(dim=1).nonzero().flatten().tolist():
            failed = bool((judged[env_id] & self.relay.failed[env_id]).any())
            if not (bool(self.relay.finished[env_id]) or (end_on_parcel_failure and failed)):
                continue
            delivered = sum(v == DELIVERED for v in verdict[env_id])
            print(f"[dexbench] DB-L4 relay: env {env_id} episode done, {delivered} of {len(parcels)} delivered "
                  f"({', '.join(f'{p.name} {VERDICT_WORDS[v]}' for p, v in zip(parcels, verdict[env_id]))})",
                  flush=True)

    def move(self, env, parcel: L4Parcel, env_ids: torch.Tensor, poses: torch.Tensor) -> None:
        """Put ``parcel`` at ``poses`` (len(env_ids), 7), environment frame, in those environments."""
        target = poses.clone()
        target[:, :3] += env.scene.env_origins[env_ids]
        asset = env.scene[parcel.entity_name]
        if parcel.is_polybag:
            asset.write_rest_pose_to_sim_index(target[:, :3], target[:, 3:], env_ids)
        else:
            asset.write_root_pose_to_sim_index(root_pose=target, env_ids=env_ids)
            asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros(len(env_ids), 6, device=env.device),
                                                   env_ids=env_ids)

    def __call__(
        self, env, env_ids, parcels: list[L4Parcel], belt_speed_m_s: float, end_on_parcel_failure: bool
    ) -> None:
        geometry = [self._geometry(env, parcel) for parcel in parcels]
        max_x, bottom_z, centre_y, barcode_up = (torch.stack(values, dim=1) for values in zip(*geometry))
        now = env.episode_length_buf.to(torch.float32) * env.step_dt
        judged, stage = self.relay.advance(now, max_x, bottom_z, centre_y, barcode_up)
        if judged.any():
            self._report(judged, now, parcels, end_on_parcel_failure)
        for i, parcel in enumerate(parcels):
            off = judged[:, i].nonzero().flatten()
            if len(off):
                park = torch.tensor(parcel.park_pose, dtype=torch.float32, device=env.device)
                self.move(env, parcel, off, park[None].repeat(len(off), 1))
                if parcel.is_polybag:
                    env.scene[parcel.entity_name].write_active_to_sim_index(False, off)
            on = stage[:, i].nonzero().flatten()
            if len(on):
                self.move(env, parcel, on, self.stage_poses[on, i])
                if parcel.is_polybag:
                    env.scene[parcel.entity_name].write_active_to_sim_index(True, on)


def relay_term(env) -> relay_parcels:
    """The DB-L4 relay of ``env`` (the task's ``relay`` event term), whose ``relay`` and ``stage_poses`` hold its
    state."""
    return env.event_manager.get_term_cfg("relay").func


def all_parcels_delivered(env) -> torch.Tensor:
    """Success: every parcel of the episode delivered barcode-up."""
    return relay_term(env).relay.delivered.all(dim=1)


def relay_finished(env) -> torch.Tensor:
    """Every parcel judged, some failed: nothing is left to do."""
    relay = relay_term(env).relay
    return relay.finished & ~relay.delivered.all(dim=1)


class ParcelsDeliveredRecorder(RecorderTerm):
    """Records how many of the episode's parcels were delivered, just before the environment resets (the relay resets
    with the events, after the recorders)."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.name = cfg.name

    def record_pre_reset(self, env_ids):
        return self.name, relay_term(self._env).relay.delivered[env_ids].sum(dim=1)


@configclass
class ParcelsDeliveredRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = ParcelsDeliveredRecorder
    name: str = "parcels_delivered"


def compute_mean_parcels_delivered(recorded_metric_data: list[np.ndarray]) -> float:
    """The mean number of parcels delivered per episode (0 to 5); 0.0 for no episodes."""
    if not recorded_metric_data:
        return 0.0
    counts = np.concatenate(recorded_metric_data)
    assert counts.ndim == 1 and counts.shape[0] == len(recorded_metric_data)
    return float(np.mean(counts))


class ParcelsDeliveredMetric(MetricBase):
    """The mean number of parcels an episode delivered barcode-up, reported beside the success rate and never scored
    (docs/Eval_Protocol.md scores success). With ``end_on_parcel_failure`` (the default, every benchmark cell) it is
    the parcels delivered before the first failure; a time-out counts the ones delivered by then."""

    name = "parcels_delivered"
    recorder_term_name = "parcels_delivered"

    def get_recorder_term_cfg(self) -> RecorderTermCfg:
        return ParcelsDeliveredRecorderCfg(name=self.recorder_term_name)

    def get_metric_term_cfg(self) -> MetricTermCfg:
        return MetricTermCfg(compute_metric_func=compute_mean_parcels_delivered, params={},
                             recorder_term_name=self.recorder_term_name)


def parcel_failed(env) -> torch.Tensor:
    """A parcel of the episode failed: success is out of reach."""
    return relay_term(env).relay.failed.any(dim=1)


@configclass
class DeliverParcelsBarcodeUpTerminationsCfg:
    """Success and one failure, the task's ``end_on_parcel_failure`` choosing which; the episode limit is the task's
    ``timeout_s``. The verdict is given in the relay's interval event, after the step's termination check, so the
    episode ends on the control step after it."""

    success: TerminationTermCfg = TerminationTermCfg(func=all_parcels_delivered)
    parcel_failed: TerminationTermCfg | None = TerminationTermCfg(func=parcel_failed)
    relay_finished: TerminationTermCfg | None = None


@configclass
class DeliverParcelsBarcodeUpEventsCfg:
    relay: EventTermCfg = MISSING


class DexBenchDeliverParcelsBarcodeUpTask(TaskBase):
    """Turn each parcel barcode-up and put it on the belt, five parcels one at a time (DB-L4)."""

    def __init__(
        self,
        parcels: list[L4Parcel],
        belt_speed_m_s: float,
        polybag_usd: str,
        vbd_substeps: int,
        vbd_iterations: int,
        mjwarp_substeps: int,
        full_surface_contact: bool,
        frame_truncation: str | None,
        coupling_mode: str = "one_way",
        box_colliders: tuple[str, ...] = (),
        end_on_parcel_failure: bool = True,
        episode_length_s: float = EPISODE_LENGTH_S,
        task_description: str | None = None,
        robot_joint_pos_overrides: dict[str, float] | None = None,
        hand_contact_material: HandContactMaterialCfg | None = None,
        tet_youngs_modulus: float | None = None,
        tet_poisson_ratio: float | None = None,
        tet_k_damp: float | None = None,
        tet_contact_ke: float | None = None,
        tet_contact_kd: float | None = None,
        tet_contact_mu: float | None = None,
        tet_collider_ke: float | None = None,
        tet_collider_kd: float | None = None,
        tet_collider_mu: float | None = None,
        tet_collider_mu_by_path: tuple[tuple[str, float], ...] | None = None,
    ):
        super().__init__(
            episode_length_s=episode_length_s,
            task_description=task_description
            or "Bring the package over, orient barcode up, then place it on the conveyor.",
        )
        if frame_truncation not in ("init", "all", None):
            raise ValueError(f"frame_truncation {frame_truncation!r}: one of 'init', 'all', None")
        if coupling_mode not in ("one_way", "two_way"):
            raise ValueError(f"coupling_mode {coupling_mode!r}: one of 'one_way', 'two_way'")
        if not isinstance(end_on_parcel_failure, bool):
            raise ValueError(f"end_on_parcel_failure {end_on_parcel_failure!r}: True or False")
        self.parcels = list(parcels)
        self.hand_contact_material = hand_contact_material
        self.robot_joint_pos_overrides = dict(robot_joint_pos_overrides) if robot_joint_pos_overrides else None
        bags = [p for p in self.parcels if p.is_polybag]
        fields = []
        for parcel in bags:
            row = E2_L4_PARCELS_BY_NAME[parcel.name]
            cfg = PolybagObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{parcel.entity_name}",
                spawn=PolybagVisualCfg(usd_path=str(polybag_usd),
                                       diffuse_texture=str(Path(polybag_usd).parent / row.waybill_albedo),
                                       diffuse_tint_rgb=row.tint_rgb),
                init_state=PolybagObjectCfg.InitialStateCfg(pos=parcel.park_pose[:3], rot=parcel.park_pose[3:]),
                cotton_youngs_modulus=None,
                cotton_poisson_ratio=None,
                film_rest_bend_angles=None,
                tet_youngs_modulus=tet_youngs_modulus,
                tet_poisson_ratio=tet_poisson_ratio,
                tet_k_damp=tet_k_damp,
                tet_contact_ke=tet_contact_ke,
                tet_contact_kd=tet_contact_kd,
                tet_contact_mu=tet_contact_mu,
                tet_collider_ke=tet_collider_ke,
                tet_collider_kd=tet_collider_kd,
                tet_collider_mu=tet_collider_mu,
                tet_collider_mu_by_path=tet_collider_mu_by_path,
                tet_size_scale=polybag_size_scale(parcel.name),
                mass_kg=parcel.mass_kg,
                vbd_substeps=int(vbd_substeps),
                vbd_iterations=int(vbd_iterations),
                mjwarp_substeps=int(mjwarp_substeps),
                frame_truncation=frame_truncation,
                full_surface_contact=bool(full_surface_contact),
                coupling_mode=coupling_mode,
                box_colliders=tuple(box_colliders),
            )
            fields.append((parcel.entity_name, PolybagObjectCfg, cfg))
        if bags:
            for parcel, (_, _, cfg) in zip(bags, fields):
                polybag_particle_masses(load_cfg_template(cfg), parcel.mass_kg)
        scene_cls = make_configclass("DeliverParcelsBarcodeUpSceneCfg", fields)
        self.scene_cfg = scene_cls()
        self.events_cfg = DeliverParcelsBarcodeUpEventsCfg(
            relay=EventTermCfg(func=relay_parcels, mode="interval", interval_range_s=(0.0, 0.0),
                               params={"parcels": self.parcels, "belt_speed_m_s": float(belt_speed_m_s),
                                       "end_on_parcel_failure": end_on_parcel_failure}),
        )
        self.termination_cfg = DeliverParcelsBarcodeUpTerminationsCfg()
        if not end_on_parcel_failure:
            self.termination_cfg.parcel_failed = None
            self.termination_cfg.relay_finished = TerminationTermCfg(func=relay_finished)

    def get_scene_cfg(self) -> Any:
        return self.scene_cfg

    def get_termination_cfg(self) -> TaskTerminationCfg:
        return task_termination_cfg(self.episode_length_s, self.termination_cfg)

    def get_events_cfg(self) -> Any:
        return self.events_cfg

    def get_mimic_env_cfg(self, arm_mode: Any) -> Any:
        raise NotImplementedError("deliver_parcels_barcode_up has no Mimic configuration yet")

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric(), ParcelsDeliveredMetric()]
