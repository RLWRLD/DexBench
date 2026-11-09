"""DB-M1 live grasp at reset, driven only by an Arena continuous variation draw."""

from __future__ import annotations

import math
import time
from dataclasses import field
from typing import TYPE_CHECKING

import torch
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import (
    combine_frame_transforms,
    quat_apply,
    quat_from_angle_axis,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    subtract_frame_transforms,
)
from isaaclab_arena.variations.continuous_sampler import ContinuousSampler
from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg
from isaaclab_arena.variations.variation_base import (
    RunTimeVariationBase,
    VariationBaseCfg,
)

from dexbench_arena.assets.newton_episode_reset import (
    synchronize_newton_episode_reset,
    uses_newton,
)
from dexbench_arena.embodiments.rby1_wuji2.constants import RBY1_WUJI2_HEAD_JOINT_NAMES
from dexbench_arena.environments.manipulation.db_m1_grasp import (
    load_grasp_plan,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _rig_sampler_cfg() -> UniformSamplerCfg:
    """The rig's draw ranges, as the frozen grasp plan records them (``controller.base_noise``, ``grasp_tilt_deg``)."""
    plan = load_grasp_plan()
    xy_m, yaw_deg = plan.controller.base_noise
    tilt_lo, tilt_hi = plan.controller.grasp_tilt_deg
    return UniformSamplerCfg(
        low=[-xy_m, -xy_m, -yaw_deg, tilt_lo, -math.pi, -1.0],
        high=[xy_m, xy_m, yaw_deg, tilt_hi, math.pi, 1.0],
    )


@configclass
class M1GraspDisturbanceCfg(VariationBaseCfg):
    """The draw ranges of :class:`M1GraspDisturbance`, tuned only through Arena's variation cfg."""

    enabled: bool = True
    sampler_cfg: UniformSamplerCfg = field(default_factory=_rig_sampler_cfg)
    head_joint_positions: dict[str, float] = field(
        default_factory=lambda: dict(zip(RBY1_WUJI2_HEAD_JOINT_NAMES, (-0.15, 0.7), strict=True))
    )


class M1GraspDisturbance(RunTimeVariationBase):
    """A six-coordinate reset draw that forms the DB-M1 grasp (:class:`M1GraspDisturbanceCfg` has the coordinates)."""

    cfg: M1GraspDisturbanceCfg

    def __init__(
        self,
        object_name: str,
        *,
        cfg: M1GraspDisturbanceCfg | None = None,
        diagnostics: bool = False,
    ):
        super().__init__(cfg if cfg is not None else M1GraspDisturbanceCfg(), name="grasp_disturbance")
        self.object_name = object_name
        self.diagnostics = diagnostics

    def apply_cfg(self, cfg: M1GraspDisturbanceCfg) -> None:
        """Refuse a draw range the grasp cannot be formed from, then apply it (Arena's override path calls this too)."""
        low, high = list(cfg.sampler_cfg.low), list(cfg.sampler_cfg.high)
        if len(low) != 6 or len(high) != 6:
            raise ValueError("DB-M1 grasp disturbance draws six coordinates")
        if not all(math.isfinite(v) for v in low + high) or any(lo > hi for lo, hi in zip(low, high)):
            raise ValueError("DB-M1 disturbance bounds must be finite and ordered (low <= high)")
        if not (0 <= low[3] and high[3] <= 180 and -math.pi <= low[4] and high[4] <= math.pi and -1 <= low[5] and high[5] <= 1):
            raise ValueError("DB-M1 tilt must lie in [0, 180] deg, azimuth in [-pi, pi], cos-polar in [-1, 1]")
        if not all(math.isfinite(v) for v in cfg.head_joint_positions.values()):
            raise ValueError("DB-M1 head joint positions must be finite")
        super().apply_cfg(cfg)

    def build_event_cfg(self) -> tuple[str, EventTermCfg]:
        if tuple(self.sampler.shape_per_sample) != (6,):
            raise ValueError("DB-M1 grasp disturbance needs exactly six coordinates")
        return (
            grasp_event_term_name(self.object_name),
            EventTermCfg(
                func=FormM1Grasp,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg(self.object_name),
                    "sampler": self.sampler,
                    "head_joint_positions": dict(self.cfg.head_joint_positions),
                    "diagnostics": self.diagnostics,
                },
            ),
        )


def grasp_event_term_name(object_name: str) -> str:
    """The reset event term that forms the grasp around ``object_name``."""
    return f"{object_name}_grasp_disturbance_variation"


def formed_grasp(env: ManagerBasedEnv, object_name: str) -> FormM1Grasp:
    """The event manager's grasp-forming term for ``object_name``, which holds what the last reset formed."""
    term = env.event_manager.get_term_cfg(grasp_event_term_name(object_name)).func
    if not isinstance(term, FormM1Grasp):
        raise ValueError(f"{grasp_event_term_name(object_name)!r} is not the DB-M1 grasp-forming term")
    return term


class FormM1Grasp(ManagerTermBase):
    """Port of the frozen rig's calibration, wrap ramp, hook touch, thumb and hold."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        if env.scene.num_envs != 1:
            raise ValueError("The DB-M1 live grasp steps the whole scene and requires num_envs == 1")
        if not uses_newton(env.cfg):
            raise ValueError("The DB-M1 live grasp requires Newton")
        self.plan = load_grasp_plan()
        self.c = self.plan.controller
        if not math.isclose(env.sim.get_physics_dt(), self.c.physics_dt_s, abs_tol=1e-9):
            raise ValueError("DB-M1 rig parity requires physics dt = 1/60 s")
        self.robot = env.scene["robot"]
        self.drawer = env.scene[cfg.params["asset_cfg"].name]
        self.jn = self.robot.joint_names
        self.bn = self.robot.body_names
        missing = set(self.plan.keyframe_joint_names) - set(self.jn)
        if missing:
            raise ValueError(f"DB-M1 grasp keyframe requires missing joints: {sorted(missing)}")
        head = cfg.params["head_joint_positions"]
        missing = set(head) - set(self.jn)
        if missing:
            raise ValueError(f"DB-M1 head joint positions name joints the robot does not have: {sorted(missing)}")
        self.head = [(self.jn.index(name), float(value)) for name, value in head.items()]
        self.hands = {h.side: h for h in self.plan.hands}
        self.wrist = {s: self.bn.index(h.wrist_body) for s, h in self.hands.items()}
        self.arm = {s: [self.jn.index(n) for n in h.arm_joint_names] for s, h in self.hands.items()}
        self.pads = {p.key: p for h in self.plan.hands for p in h.pads}
        self.tip = {k: self.bn.index(p.body) for k, p in self.pads.items()}
        self.finger = {k: [self.jn.index(n) for n in p.joint_names] for k, p in self.pads.items() if k.startswith("L_")}
        self.left_sensor = env.scene["left_cs"]
        self.right_sensor = env.scene["right_cs"]
        self.pad_groups = {}
        for k, p in self.pads.items():
            sensor = self.left_sensor if k.startswith("L_") else self.right_sensor
            ids = [i for i, name in enumerate(sensor.body_names) if name in p.contact_bodies]
            if len(ids) != len(p.contact_bodies):
                raise ValueError(f"DB-M1 pad {k} is missing filtered sensing bodies {p.contact_bodies}")
            self.pad_groups[k] = ids
        self.ik = {
            s: DifferentialIKController(
                DifferentialIKControllerCfg(
                    command_type="pose",
                    use_relative_mode=False,
                    ik_method="dls",
                    ik_params={"lambda_val": self.c.wrist_dls_lambda},
                ),
                num_envs=1,
                device=self.device,
            )
            for s in self.hands
        }
        self.wrist_in_drawer = {s: self.tensor(h.wrist_in_drawer)[None] for s, h in self.hands.items()}
        self.nbd = 0 if self.robot.is_fixed_base else 6
        self.eye = torch.eye(3, device=self.device)
        self.zero_velocity = torch.zeros(1, 6, device=self.device)
        self.wrap = [
            (self.jn.index(n), v)
            for n, v in self.hands["right"].fingers
            if "thumb" not in n and n.endswith(("_pip", "_dip"))
        ]
        self.hooks = [f"L_{f}" for f in self.plan.hook_fingers]
        self.pad_target: dict[str, torch.Tensor] | None = None
        self.reset_result: dict | None = None

    def tensor(self, values):
        return torch.tensor(values, dtype=torch.float32, device=self.device)

    def jac_body(self, body):
        return body - 1 if self.robot.is_fixed_base else body

    def forces(self):
        result = {}
        for sensor, prefix in ((self.left_sensor, "L_"), (self.right_sensor, "R_")):
            f = sensor.data.force_matrix_w.torch[0, :, 0]
            for key, ids in self.pad_groups.items():
                if key.startswith(prefix):
                    result[key] = float(f[ids].sum(0).norm())
        return result

    def step_held(self, env, pose):
        self.drawer.write_root_pose_to_sim_index(root_pose=pose)
        self.drawer.write_root_velocity_to_sim_index(root_velocity=self.zero_velocity)
        self.robot.set_joint_position_target_index(target=self.q_tgt)
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(env.sim.get_physics_dt())

    def place_wrists(self):
        bp = self.robot.data.body_link_pose_w.torch
        jac = self.robot.data.body_link_jacobian_w.torch
        for side in self.hands:
            pose = self.untilted if side == "left" else self.held
            wr = self.wrist_in_drawer[side]
            tp, tq = combine_frame_transforms(pose[:, :3], pose[:, 3:], wr[:, :3], wr[:, 3:])
            self.ik[side].set_command(torch.cat((tp, tq), -1))
            joints = self.arm[side]
            cols = [self.nbd + j for j in joints]
            q = self.q_tgt[:, joints]
            b = self.wrist[side]
            new = self.ik[side].compute(bp[:, b, :3], bp[:, b, 3:], jac[:, self.jac_body(b), :, cols], q)
            self.q_tgt[:, joints] = q + (new - q).clamp(-self.c.arm_step, self.c.arm_step)

    def pad_servo(self, keys):
        bp = self.robot.data.body_link_pose_w.torch[0]
        jac = self.robot.data.body_link_jacobian_w.torch[0]
        dp, dq = self.drawer.data.root_pos_w.torch, self.drawer.data.root_quat_w.torch
        for k in keys:
            target, _ = combine_frame_transforms(dp, dq, self.pad_target[k][None])
            error = target[0] - bp[self.tip[k], :3]
            joints = self.finger[k]
            j = jac[self.jac_body(self.tip[k]), :3][:, [self.nbd + i for i in joints]]
            delta = j.T @ torch.linalg.solve(j @ j.T + self.c.pad_dls_squared_lambda * self.eye, error)
            self.q_tgt[0, joints] += delta.clamp(-self.c.pad_step, self.c.pad_step)
        limits = self.robot.data.soft_joint_pos_limits.torch[0]
        self.q_tgt[0] = torch.maximum(torch.minimum(self.q_tgt[0], limits[:, 1]), limits[:, 0])

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        sampler: ContinuousSampler,
        head_joint_positions: dict[str, float],
        diagnostics: bool,
    ):
        del head_joint_positions
        manager = env.sim.physics_manager
        manager.set_decimation(1)
        try:
            self._form(env, env_ids, asset_cfg, sampler, diagnostics)
        finally:
            manager.set_decimation(env.cfg.decimation)

    def _form(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        sampler: ContinuousSampler,
        diagnostics: bool,
    ):
        del asset_cfg
        ids = (
            torch.tensor([0], dtype=torch.int32, device=self.device)
            if env_ids is None
            else torch.as_tensor(env_ids, device=self.device, dtype=torch.int32).reshape(-1)
        )
        if not ids.numel():
            return
        if ids.tolist() != [0]:
            raise ValueError("DB-M1 live grasp requires resetting the whole single-env scene")
        self.pad_target = None
        self.reset_result = None
        started = time.perf_counter()
        sample = sampler.sample(num_samples=1, env_ids=ids).to(self.device)
        if sample.shape != (1, 6) or not torch.isfinite(sample).all() or abs(float(sample[0, 5])) > 1:
            raise ValueError("DB-M1 grasp draw must be six finite values with cos-polar in [-1, 1]")
        dx, dy, dyaw, tilt, azimuth, cos_polar = sample[0]
        axis = torch.stack(
            (
                torch.sqrt(1 - cos_polar * cos_polar) * torch.cos(azimuth),
                torch.sqrt(1 - cos_polar * cos_polar) * torch.sin(azimuth),
                cos_polar,
            )
        )[None]
        zeros = torch.zeros(1, device=self.device)
        yaw0 = self.tensor([math.radians(self.plan.base_yaw_deg)])
        q_nom = quat_from_euler_xyz(zeros, zeros, yaw0)
        q_base = quat_from_euler_xyz(zeros, zeros, yaw0 + torch.deg2rad(dyaw[None]))
        p_nom = self.tensor(self.plan.base_position_m)[None]
        p_base = p_nom + torch.stack((dx, dy, zeros[0]))[None]
        delta_q = quat_mul(q_base, quat_inv(q_nom))
        delta_p = p_base - quat_apply(delta_q, p_nom)
        dp = quat_apply(delta_q, self.tensor(self.plan.drawer_position_m)[None]) + delta_p
        clamp = self.tensor(self.plan.clamp_point_m)[None]
        center = dp + quat_apply(delta_q, clamp)
        tilted_q = quat_mul(delta_q, quat_from_angle_axis(torch.deg2rad(tilt[None]), axis))
        tilted_p = center - quat_apply(tilted_q, clamp)
        origin = env.scene.env_origins[0:1]
        self.untilted = torch.cat((dp + origin, delta_q), -1)
        self.held = torch.cat((tilted_p + origin, tilted_q), -1)
        self.robot.write_root_pose_to_sim_index(root_pose=torch.cat((p_base + origin, q_base), -1))
        self.robot.write_root_velocity_to_sim_index(root_velocity=self.zero_velocity)
        self.q_tgt = self.robot.data.default_joint_pos.torch.clone()
        for name, value in self.plan.joint_positions:
            if name in self.jn:
                self.q_tgt[0, self.jn.index(name)] = value
        for joint, value in self.head:
            self.q_tgt[0, joint] = value
        for hand in self.plan.hands:
            for name, value in hand.fingers:
                self.q_tgt[0, self.jn.index(name)] = value
        for joint, value in self.wrap:
            self.q_tgt[0, joint] = value - self.c.wrap_open
        self.robot.write_joint_position_to_sim_index(position=self.q_tgt)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(self.q_tgt))
        for controller in self.ik.values():
            controller.reset()
        self.hook_touch = dict.fromkeys(self.hooks, False)
        self.thumb_loaded = False
        far = self.held.clone()
        far[:, 2] += 1.0
        self.drawer.write_root_pose_to_sim_index(root_pose=far)
        self.drawer.write_root_velocity_to_sim_index(root_velocity=self.zero_velocity)
        synchronize_newton_episode_reset(ids)
        self.step_held(env, far)
        bp = self.robot.data.body_link_pose_w.torch[0]
        pad_target = {}
        self.pad_site = {}
        sites = {s.name: s for s in self.plan.drawer_sites}
        for key, body in self.tip.items():
            side = "left" if key.startswith("L_") else "right"
            wrist = self.wrist[side]
            tip_in_wrist, _ = subtract_frame_transforms(bp[None, wrist, :3], bp[None, wrist, 3:], bp[None, body, :3])
            wr = self.wrist_in_drawer[side]
            fitted, _ = combine_frame_transforms(wr[:, :3], wr[:, 3:], tip_in_wrist)
            if key.startswith("L_"):
                site = sites["thumb_face" if key == "L_thumb" else "handle_pocket"]
                direction = site.normal if key == "L_thumb" else self.c.hook_push
                press = self.c.thumb_press_m if key == "L_thumb" else site.press_m
                self.pad_site[key] = fitted[0]
                pad_target[key] = fitted[0] + self.tensor(direction) * press
        self.pad_target = pad_target
        self.robot.write_joint_position_to_sim_index(position=self.q_tgt)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(self.q_tgt))
        synchronize_newton_episode_reset(ids)
        self.step_held(env, self.held)
        nc = int(self.c.close_s * 60)
        for i in range(nc):
            close = min(1.0, (i + 1) / (0.8 * nc))
            for joint, value in self.wrap:
                self.q_tgt[0, joint] = value - self.c.wrap_open * (1 - close)
            forces = self.forces()
            todo = [k for k in self.hooks if not self.hook_touch[k]]
            for k in todo:
                if forces[k] > self.c.f_hook_on_n:
                    self.hook_touch[k] = True
                    if self.c.force_band:
                        self.q_tgt[0, self.finger[k]] = self.robot.data.joint_pos.torch[0, self.finger[k]]
            if todo:
                self.pad_servo(todo)
            self.place_wrists()
            self.step_held(env, self.held)
        thumb_steps = 0
        while thumb_steps < self.c.thumb_steps:
            if self.forces()["L_thumb"] > self.c.f_thumb_on_n:
                self.thumb_loaded = True
                if self.c.force_band:
                    js = self.finger["L_thumb"]
                    measured = self.robot.data.joint_pos.torch[0, js]
                    self.q_tgt[0, js] = measured + (self.q_tgt[0, js] - measured).clamp(-0.05, 0.05)
                break
            self.pad_servo(["L_thumb"])
            self.place_wrists()
            self.step_held(env, self.held)
            thumb_steps += 1
        for _ in range(self.c.hold_steps):
            self.place_wrists()
            self.step_held(env, self.held)
        self.robot.set_joint_position_target_index(target=self.q_tgt)
        env.scene.write_data_to_sim()
        if diagnostics:
            bp = self.robot.data.body_link_pose_w.torch
            errors = {}
            target_errors = {}
            for side, b in self.wrist.items():
                wr = self.wrist_in_drawer[side]
                target, _ = combine_frame_transforms(self.untilted[:, :3], self.untilted[:, 3:], wr[:, :3], wr[:, 3:])
                errors[side] = float(1000 * (target - bp[:, b, :3]).norm())
                pose = self.untilted if side == "left" else self.held
                commanded, _ = combine_frame_transforms(pose[:, :3], pose[:, 3:], wr[:, :3], wr[:, 3:])
                target_errors[side] = float(1000 * (commanded - bp[:, b, :3]).norm())
            drawer_pose = torch.cat(
                (self.drawer.data.root_pos_w.torch, self.drawer.data.root_quat_w.torch),
                -1,
            ).clone()
            lw = bp[:, self.wrist["left"]]
            rel_p, rel_q = subtract_frame_transforms(lw[:, :3], lw[:, 3:], drawer_pose[:, :3], drawer_pose[:, 3:])
            self.reset_result = {
                "sample": sample[0].tolist(),
                "hook_touch": dict(self.hook_touch),
                "thumb_loaded": self.thumb_loaded,
                "thumb_steps": thumb_steps,
                "wrist_error_mm": errors,
                "wrist_target_error_mm": target_errors,
                "drawer_in_left_wrist": torch.cat((rel_p, rel_q), -1)[0].tolist(),
                "drawer_world_pose": drawer_pose[0].tolist(),
                "joint_position": self.robot.data.joint_pos.torch[0].tolist(),
                "joint_target": self.q_tgt[0].tolist(),
                "pad_forces_n": self.forces(),
                "reset_event_wall_s": time.perf_counter() - started,
            }
