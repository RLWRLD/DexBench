# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Observe actual screw joints and command only the driver's spindle motor."""

import math

import torch
from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import quat_apply

from dexbench_arena.layouts.db_m3_layout import (
    BOLT_NAMES,
    SEATED_HEADS,
    TOOL_AXIS_LOCAL,
    TOOL_NAME,
    TOOL_TIP_LOCAL,
)
from dexbench_arena.evaluation.replay_guards import deterministic_reset
from dexbench_arena.tasks.power_driver_model import FasteningCfg, FasteningModel


class TorqueFasteningState(FasteningModel):
    """Six-slot task contract populated by measured joints, never motor time."""

    def __init__(self, seated_heads, num_envs, cfg=FasteningCfg(), device="cpu"):
        super().__init__(seated_heads, num_envs, cfg, device)
        self.angle_rad = torch.zeros_like(self.turns)
        self.travel_m = torch.zeros_like(self.turns)
        self.positions = super().head_positions
        self.pitch_error_m = torch.zeros_like(self.turns)

    @property
    def head_positions(self):
        return self.positions

    @property
    def remaining_m(self):
        return (self.cfg.travel_m - self.travel_m).clamp_min(0) * self.active

    @property
    def seated(self):
        return self.joints_seated(self.angle_rad, self.travel_m)

    def joints_seated(self, angles_rad, travel_m):
        """Evaluate measured thread joints without advancing or caching state."""
        lead_m_per_rad = self.cfg.travel_m / (2 * math.pi * self.cfg.turns_to_seat)
        return (
            torch.isfinite(angles_rad) & torch.isfinite(travel_m)
            & (angles_rad >= 2 * math.pi * self.cfg.turns_to_seat - math.radians(1.0))
            & ((self.cfg.travel_m - travel_m).clamp_min(0) <= 0.0001)
            & ((travel_m - lead_m_per_rad * angles_rad).abs() <= 0.0001)
        )

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self.angle_rad[ids] = 0
        self.travel_m[ids] = 0
        self.pitch_error_m[ids] = 0
        positions = self.seated_heads.clone()
        positions[self.active, 1] -= self.cfg.travel_m
        self.positions[ids] = positions

    def observe(self, angles_rad, travel_m, positions, forces_n, tip_xyz, axis_xyz):
        if (
            angles_rad.shape != self.turns.shape
            or travel_m.shape != self.turns.shape
            or positions.shape != self.positions.shape
        ):
            raise ValueError("Expected six physical screw slots per environment")
        finite = torch.isfinite(angles_rad) & torch.isfinite(travel_m) & torch.isfinite(positions).all(-1)
        if not finite.all():
            raise ValueError("Nonfinite physical screw state")
        self.angle_rad.copy_(angles_rad * self.active)
        self.travel_m.copy_(travel_m * self.active)
        self.positions.copy_(positions)
        lead_m_per_rad = self.cfg.travel_m / (2 * math.pi * self.cfg.turns_to_seat)
        self.pitch_error_m.copy_((self.travel_m - lead_m_per_rad * self.angle_rad) * self.active)
        self.turns.copy_((self.angle_rad / (2 * math.pi)).clamp(0, self.cfg.turns_to_seat))
        self.turns.copy_(torch.where(self.seated & self.active, self.cfg.turns_to_seat, self.turns))
        delta = tip_xyz[:, None] - positions
        alignment = axis_xyz[:, 1] / axis_xyz.norm(dim=-1).clamp_min(1e-9)
        self.engaged.copy_(
            self.active
            & (forces_n >= self.cfg.contact_threshold_n)
            & (delta[..., (0, 2)].norm(dim=-1) <= self.cfg.lateral_tolerance_m)
            & (delta[..., 1].abs() <= self.cfg.axial_tolerance_m)
            & (alignment[:, None] >= math.cos(math.radians(self.cfg.alignment_tolerance_deg)))
        )

    def advance(self, *args, **kwargs):
        raise RuntimeError("Torque fastening requires measured screw joints; elapsed motor time cannot advance it")


@deterministic_reset
class ContactTorqueFastening(ManagerTermBase):
    """Gate the spindle once per control step and observe solver-driven joints."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.model = TorqueFasteningState(
            cfg.params.get("seated_heads", SEATED_HEADS),
            env.num_envs,
            FasteningCfg(**cfg.params["fastening_cfg"]),
            env.device,
        )
        self.tool = env.scene[TOOL_NAME]
        self.bolts = [env.scene[name] for name in BOLT_NAMES]
        self.motor_id = self.tool.joint_names.index("SpindleMotor")
        self.trigger_id = self.tool.joint_names.index("TriggerPress")
        self.screw_ids = {
            i: (
                self.bolts[i].joint_names.index("ScrewRotation"),
                self.bolts[i].joint_names.index("ThreadTravel"),
                self.bolts[i].body_names.index("Screw"),
            )
            for i in self.model.cfg.active_bolt_indices
        }
        tip = (TOOL_TIP_LOCAL[0], TOOL_TIP_LOCAL[1] - cfg.params["bit_extension_m"], TOOL_TIP_LOCAL[2])
        self.tip_local = torch.tensor(tip, device=env.device).expand(env.num_envs, -1)
        self.axis_local = torch.tensor(TOOL_AXIS_LOCAL, device=env.device).expand(env.num_envs, -1)
        self.last_step = -1
        self.spindle_angle_rad = torch.zeros(env.num_envs, device=env.device)
        self.spindle_speed_rad_s = torch.zeros_like(self.spindle_angle_rad)
        self.motor_effort_nm = torch.zeros_like(self.spindle_angle_rad)
        self.motor_available = cfg.params["motor_enabled"]
        self.trigger_travel_m = torch.zeros(env.num_envs, device=env.device)
        self.trigger_index_force_n = torch.zeros_like(self.trigger_travel_m)
        self.trigger_pressed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.motor_contact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.motor_target_speed_rad_s = torch.zeros_like(self.spindle_angle_rad)

    @property
    def success(self):
        """Require seated joints both before and after the latest physics step."""
        success = self.model.success.clone()
        for i, (turn_id, slide_id, _) in self.screw_ids.items():
            joints = self.bolts[i].data.joint_pos.torch
            success &= self.model.joints_seated(joints[:, turn_id], joints[:, slide_id])
        return success

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for i in self.model.cfg.active_bolt_indices:
            bolt = self.bolts[i]
            zero = torch.zeros_like(bolt.data.joint_pos.torch[ids])
            bolt.write_joint_state_to_sim_index(position=zero, velocity=zero, env_ids=env_ids)
        zero = torch.zeros_like(self.tool.data.joint_pos.torch[ids])
        self.tool.write_joint_state_to_sim_index(position=zero, velocity=zero, env_ids=env_ids)
        self.model.reset(env_ids)
        self.spindle_angle_rad[ids] = 0
        self.spindle_speed_rad_s[ids] = 0
        self.motor_effort_nm[ids] = 0
        self.motor_contact[ids] = False
        self.trigger_travel_m[ids] = 0
        self.trigger_index_force_n[ids] = 0
        self.trigger_pressed[ids] = False
        self.motor_target_speed_rad_s[ids] = 0
        if self.motor_available:
            self.tool.actuators.target_command.set_velocity_index(
                value=self.motor_target_speed_rad_s[ids, None], joint_ids=[self.motor_id], env_ids=env_ids
            )

    def __call__(
        self, env, env_ids, fastening_cfg, motor_enabled=True, bit_extension_m=0.12,
        motor_direction=1.0, seated_heads=SEATED_HEADS, trigger_press_m=.002,
    ):
        del env_ids, fastening_cfg, bit_extension_m, seated_heads
        if self.last_step == env.common_step_counter:
            return
        self.last_step = env.common_step_counter
        contact_history = env.scene["driver_bolt_contact"].data.normal_force_matrix_w_history.torch
        latest_forces = contact_history[:, 0, 0].norm(dim=-1)
        self.motor_contact.copy_(
            (
                self.model.active
                & torch.isfinite(latest_forces)
                & (latest_forces >= self.model.cfg.contact_threshold_n)
            ).any(dim=-1)
        )
        speed_rad_s = (
            2 * math.pi * self.model.cfg.motor_rps * motor_direction if motor_enabled and self.motor_available else 0.0
        )
        self.trigger_travel_m.copy_(self.tool.data.joint_pos.torch[:, self.trigger_id])
        index_forces = env.scene["driver_trigger_contact"].data.normal_force_matrix_w_history.torch[:, 0]
        self.trigger_index_force_n.copy_(index_forces.norm(dim=-1).flatten(1).amax(dim=1))
        self.trigger_pressed.copy_(
            torch.isfinite(self.trigger_travel_m) & torch.isfinite(self.trigger_index_force_n)
            & (self.trigger_travel_m >= trigger_press_m)
            & (self.trigger_index_force_n >= self.model.cfg.contact_threshold_n)
        )
        self.motor_target_speed_rad_s.copy_(self.motor_contact * self.trigger_pressed * speed_rad_s)
        if self.motor_available:
            self.tool.actuators.target_command.set_velocity_index(
                value=-self.motor_target_speed_rad_s[:, None], joint_ids=[self.motor_id]
            )
        self.spindle_angle_rad.copy_(-self.tool.data.joint_pos.torch[:, self.motor_id])
        self.spindle_speed_rad_s.copy_(-self.tool.data.joint_vel.torch[:, self.motor_id])
        self.motor_effort_nm.copy_(-self.tool.data.applied_torque.torch[:, self.motor_id])

        angles = torch.zeros_like(self.model.turns)
        travel = torch.zeros_like(angles)
        positions = self.model.seated_heads.expand(env.num_envs, -1, -1).clone()
        for i, (turn_id, slide_id, body_id) in self.screw_ids.items():
            bolt = self.bolts[i]
            angles[:, i] = bolt.data.joint_pos.torch[:, turn_id]
            travel[:, i] = bolt.data.joint_pos.torch[:, slide_id]
            positions[:, i] = bolt.data.body_link_pose_w.torch[:, body_id, :3] - env.scene.env_origins
        pose = self.tool.data.root_link_pose_w.torch
        tip = pose[:, :3] + quat_apply(pose[:, 3:], self.tip_local) - env.scene.env_origins
        axis = quat_apply(pose[:, 3:], self.axis_local)
        peak_forces = contact_history[:, :, 0].norm(dim=-1).amax(dim=1)
        self.model.observe(angles, travel, positions, peak_forces, tip, axis)
