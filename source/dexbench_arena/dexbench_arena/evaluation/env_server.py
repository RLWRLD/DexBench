# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""ZeroMQ environment server that rolls out remote policy actions in-sim."""

from __future__ import annotations

import json
import logging
import math
import sys
from dataclasses import dataclass
from typing import Any

import torch
import zmq

from dexbench_arena.evaluation.serialization import TorchSerializer, tree_to_cpu
from dexbench_arena.policy.image_preprocessing import image_preprocessing_record, resize_rgb_frames

_RECV_POLL_TIMEOUT_MS = 1000


def episode_cap_steps(env_cfg) -> int:
    """The env's own episode cap in control steps: what ``ManagerBasedRLEnv.max_episode_length`` is."""
    return math.ceil(env_cfg.episode_length_s / (env_cfg.sim.dt * env_cfg.decimation))


def _as_torch(array) -> torch.Tensor:
    """Convert warp-backed articulation data to torch; torch tensors pass through."""
    if isinstance(array, torch.Tensor):
        return array
    import warp as wp

    return wp.to_torch(array)


@dataclass
class DexBenchEnvServerConfig:
    """Configuration for the DexBench env server."""

    port: int = 5560
    bind_address: str = "*"
    max_episode_steps: int = 300
    """Steps per episode before the server reports truncated=True (stabilization excluded)."""
    reset_stabilization_steps: int = 10
    """Physics-settling steps executed inside reset() before the first obs is returned."""
    stabilization_mode: str = "hold"
    """'hold' repeats the current-state hold action, 'zero' a zero action, 'none' skips."""
    terminate_on_success: bool = False
    """Fold the success termination term (when the env has one) into terminated."""
    render_on_idle: bool = False
    """Render viewport frames while waiting for client requests (keeps the Kit GUI responsive)."""
    wire_image_resize_height: int | None = None
    """Optional camera height used only for observations sent over the wire (area downscale of
    uint8 RGB(A) terms, same aspect ratio only; see :mod:`dexbench_arena.policy.image_preprocessing`)."""
    wire_image_resize_width: int | None = None
    """Optional camera width used only for observations sent over the wire."""


class DexBenchEnvServer:
    """Serve a built DexBench environment to remote policy clients."""

    def __init__(self, config: DexBenchEnvServerConfig, env, simulation_app, success_term=None):
        self.config = config
        self.env = env
        self.simulation_app = simulation_app
        self.success_term = success_term
        self.logger = self._setup_logger()

        self._episode_step_count = 0
        self._running = False
        self._wire_image_record: dict[str, Any] | None = None

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{config.bind_address}:{config.port}")

        self._last_obs, _ = self.env.reset()

        self.logger.info(f"DexBench env server listening on tcp://{config.bind_address}:{config.port}")
        self.logger.info(
            f"num_envs={self.env.num_envs}, device={self.env.device}, "
            f"action_dim={self.env.action_manager.total_action_dim}"
        )
        self.logger.info(
            f"max_episode_steps={config.max_episode_steps}, "
            f"reset_stabilization_steps={config.reset_stabilization_steps} (mode={config.stabilization_mode})"
        )
        if self.success_term is None:
            self.logger.info("No success termination term: info['is_success'] will not be reported")

    @staticmethod
    def _setup_logger() -> logging.Logger:
        """Logger with an explicit stdout handler to bypass Isaac Sim's logging."""
        logger = logging.getLogger("DexBenchEnvServer")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
        return logger


    def build_hold_action(self) -> torch.Tensor:
        """Build a full action tensor that keeps the robot at its current state."""
        robot = self.env.scene["robot"]
        pieces = []
        for term_name in self.env.action_manager.active_terms:
            term = self.env.action_manager.get_term(term_name)
            pieces.append(self._hold_action_for_term(term_name, term, robot))
        action = torch.cat(pieces, dim=1)
        expected_dim = self.env.action_manager.total_action_dim
        assert action.shape == (self.env.num_envs, expected_dim), (
            f"Hold action shape {tuple(action.shape)} does not match ({self.env.num_envs}, {expected_dim})"
        )
        return action

    def _hold_action_for_term(self, term_name: str, term, robot) -> torch.Tensor:
        try:
            controller_cfg = getattr(getattr(term, "cfg", None), "controller", None)
            if controller_cfg is not None and hasattr(term, "_hand_joint_ids"):
                return self._hold_action_for_pink_ik_term(term, controller_cfg, robot)
            if getattr(term, "_joint_ids", None) is not None:
                return self._hold_action_for_joint_term(term, robot)
        except Exception as exc:
            self.logger.warning(f"Hold action failed for term '{term_name}' ({exc}); falling back to zeros")
            return torch.zeros(self.env.num_envs, term.action_dim, device=self.env.device)

        self.logger.warning(f"No hold-action rule for term '{term_name}' ({type(term).__name__}); using zeros")
        return torch.zeros(self.env.num_envs, term.action_dim, device=self.env.device)

    def _hold_action_for_pink_ik_term(self, term, controller_cfg, robot) -> torch.Tensor:
        """Hold action for ``PinkInverseKinematicsAction``-style terms."""
        import isaaclab.utils.math as math_utils

        env_origins = _as_torch(self.env.scene.env_origins)
        body_pos_w = _as_torch(robot.data.body_pos_w)
        body_quat_w = _as_torch(robot.data.body_quat_w)
        joint_pos = _as_torch(robot.data.joint_pos)
        pieces = []
        frame_names = [
            getattr(task, "frame", None)
            for task in controller_cfg.variable_input_tasks
            if getattr(task, "frame", None)
        ]
        for frame_name in frame_names:
            if frame_name in robot.body_names:
                body_index = robot.body_names.index(frame_name)
                pieces.append(body_pos_w[:, body_index] - env_origins)
                pieces.append(body_quat_w[:, body_index])
                continue
            frame_pos, frame_quat = self._ik_frame_pose_from_pinocchio(
                term, frame_name, joint_pos, body_pos_w, body_quat_w, env_origins, robot
            )
            pieces.append(frame_pos)
            pieces.append(frame_quat)
        pieces.append(joint_pos[:, term._hand_joint_ids])
        action = torch.cat(pieces, dim=1)
        assert action.shape[1] == term.action_dim, (
            f"Pink IK hold action dim {action.shape[1]} != term action_dim {term.action_dim} "
            f"(frames={frame_names})"
        )
        return action

    def _ik_frame_pose_from_pinocchio(
        self, term, frame_name: str, joint_pos, body_pos_w, body_quat_w, env_origins, robot
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Current env-local world pose of a URDF-only IK frame, via the term's own pinocchio FK."""
        import numpy as np

        import isaaclab.utils.math as math_utils

        anchor = self._resolve_ik_frame_anchor(term, frame_name, robot)
        anchor_body_index, anchor_urdf_link = anchor
        anchor_pos = body_pos_w[:, anchor_body_index] - env_origins
        anchor_quat = body_quat_w[:, anchor_body_index]

        joint_pos_np = joint_pos.detach().cpu().numpy()
        rel_positions, rel_rotations = [], []
        for env_index, controller in enumerate(term._ik_controllers):
            configuration = controller.pink_configuration
            configuration.update(joint_pos_np[env_index][controller.isaac_lab_to_pink_ordering])
            transform_frame = configuration.get_transform_frame_to_world(frame_name)
            transform_anchor = configuration.get_transform_frame_to_world(anchor_urdf_link)
            relative = transform_anchor.inverse() * transform_frame
            rel_positions.append(np.asarray(relative.translation, dtype=np.float32))
            rel_rotations.append(np.asarray(relative.rotation, dtype=np.float32))

        device = anchor_pos.device
        rel_pos = torch.from_numpy(np.stack(rel_positions)).to(device)
        rel_quat = math_utils.quat_from_matrix(torch.from_numpy(np.stack(rel_rotations)).to(device))
        world_pos = anchor_pos + math_utils.quat_apply(anchor_quat, rel_pos)
        world_quat = math_utils.quat_mul(anchor_quat, rel_quat)
        return world_pos, world_quat

    def _resolve_ik_frame_anchor(self, term, frame_name: str, robot) -> tuple[int, str]:
        """Find (USD body index, URDF link name) anchoring a URDF-only IK frame, cached per term."""
        cache = getattr(term, "_dexbench_anchor_cache", None)
        if cache is None:
            cache = term._dexbench_anchor_cache = {}
        if frame_name in cache:
            return cache[frame_name]

        body_names = list(robot.body_names)

        def _match_body(urdf_link: str) -> str | None:
            if urdf_link in body_names:
                return urdf_link
            candidates = [body for body in body_names if urdf_link.endswith(f"_{body}")]
            return candidates[0] if len(candidates) == 1 else None

        anchor: tuple[int, str] | None = None
        urdf_path = getattr(getattr(term.cfg, "controller", None), "urdf_path", None)
        if urdf_path:
            try:
                import xml.etree.ElementTree as ET

                root = ET.parse(urdf_path).getroot()
                parent_by_child = {
                    joint.find("child").get("link"): joint.find("parent").get("link")
                    for joint in root.iter("joint")
                    if joint.find("child") is not None and joint.find("parent") is not None
                }
                current = frame_name
                for _ in range(32):
                    matched = _match_body(current)
                    if matched is not None:
                        anchor = (body_names.index(matched), current)
                        break
                    current = parent_by_child.get(current)
                    if current is None:
                        break
            except (OSError, ET.ParseError) as exc:
                self.logger.warning(f"Could not parse IK URDF for frame anchoring: {exc}")

        if anchor is None:
            base_link = term.cfg.controller.base_link_name
            anchor = (body_names.index(base_link), base_link)
        self.logger.info(f"IK frame '{frame_name}' anchored at USD body '{body_names[anchor[0]]}' (URDF '{anchor[1]}')")
        cache[frame_name] = anchor
        return anchor

    def _hold_action_for_joint_term(self, term, robot) -> torch.Tensor:
        """Hold action for ``JointAction``-style terms."""
        target = _as_torch(robot.data.joint_pos)[:, term._joint_ids]
        scale = getattr(term, "_scale", 1.0)
        offset = getattr(term, "_offset", 0.0)
        return (target - offset) / scale


    def info(self) -> dict[str, Any]:
        try:
            env = self.env
            robot = env.scene["robot"]

            action_terms = []
            for term_name in env.action_manager.active_terms:
                term = env.action_manager.get_term(term_name)
                entry: dict[str, Any] = {
                    "name": term_name,
                    "dim": term.action_dim,
                    "class": type(term).__name__,
                }
                controller_cfg = getattr(getattr(term, "cfg", None), "controller", None)
                if controller_cfg is not None and hasattr(term, "_hand_joint_names"):
                    entry["frame_names"] = [
                        getattr(task, "frame", None)
                        for task in controller_cfg.variable_input_tasks
                        if getattr(task, "frame", None)
                    ]
                    entry["hand_joint_names"] = list(term._hand_joint_names)
                    entry["layout"] = "[pos_xyz, quat_xyzw] per frame (env-local), then hand joints"
                elif getattr(term, "_joint_names", None) is not None:
                    entry["joint_names"] = list(term._joint_names)
                action_terms.append(entry)

            packaged = self._package_observation(self._last_obs)
            wire_image_preprocessing = self._wire_image_record
            return {
                "status": "ok",
                "environment_name": getattr(env.cfg, "env_name", None) or type(env).__name__,
                "task_description": getattr(env.cfg, "task_description", None),
                "num_envs": env.num_envs,
                "device": str(env.device),
                "action_dim": env.action_manager.total_action_dim,
                "action_terms": action_terms,
                "observation_structure": _describe_tree(packaged),
                "robot_joint_names": list(robot.joint_names),
                "robot_body_names": list(robot.body_names),
                "physics_dt": float(env.physics_dt),
                "step_dt": float(env.step_dt),
                "max_episode_steps": self.config.max_episode_steps,
                "reset_stabilization_steps": self.config.reset_stabilization_steps,
                "stabilization_mode": self.config.stabilization_mode,
                "has_success_term": self.success_term is not None,
                "wire_image_preprocessing": wire_image_preprocessing,
            }
        except Exception as exc:
            self.logger.error(f"Error building info: {exc}", exc_info=True)
            return {"status": "error", "message": f"Error building info: {exc}"}

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        try:
            if seed is not None:
                self.logger.info(f"Resetting environment (seed={seed})")
                obs, info = self.env.reset(seed=seed)
            else:
                self.logger.info("Resetting environment")
                obs, info = self.env.reset()

            self._write_default_joint_targets()
            obs, info = self._stabilize(obs, info)
            self._episode_step_count = 0
            self._last_obs = obs
            return {
                "status": "ok",
                "observation": self._package_observation(obs),
                "info": self._package_info(info),
            }
        except Exception as exc:
            self.logger.error(f"Error resetting environment: {exc}", exc_info=True)
            return {"status": "error", "message": f"Error resetting environment: {exc}"}

    def step(self, action) -> dict[str, Any]:
        try:
            action = torch.as_tensor(action, dtype=torch.float32)
            expected_shape = (self.env.num_envs, self.env.action_manager.total_action_dim)
            if tuple(action.shape) != expected_shape:
                return {
                    "status": "error",
                    "message": f"Invalid action shape {tuple(action.shape)}; expected {expected_shape}",
                }
            action = action.to(self.env.device)

            obs, reward, terminated, truncated, info = self.env.step(action)

            self._episode_step_count += 1
            if self._episode_step_count >= self.config.max_episode_steps:
                truncated = torch.ones_like(truncated, dtype=torch.bool)

            if self.success_term is not None:
                success = self.success_term.func(self.env, **self.success_term.params)
                success = success.flatten().bool()
                info = dict(info) if info is not None else {}
                info["is_success"] = success.cpu()
                if self.config.terminate_on_success:
                    terminated = terminated | success.to(terminated.device)

            self._last_obs = obs
            return {
                "status": "ok",
                "observation": self._package_observation(obs),
                "reward": tree_to_cpu(reward),
                "terminated": tree_to_cpu(terminated),
                "truncated": tree_to_cpu(truncated),
                "info": self._package_info(info),
            }
        except Exception as exc:
            self.logger.error(f"Error executing step: {exc}", exc_info=True)
            return {"status": "error", "message": f"Error executing step: {exc}"}

    def hold_action(self) -> dict[str, Any]:
        try:
            return {"status": "ok", "action": self.build_hold_action().cpu()}
        except Exception as exc:
            self.logger.error(f"Error building hold action: {exc}", exc_info=True)
            return {"status": "error", "message": f"Error building hold action: {exc}"}

    def _write_default_joint_targets(self) -> None:
        """Pin all joint position targets to the reset pose after env.reset()."""
        try:
            robot = self.env.scene["robot"]
            default_joint_pos = _as_torch(robot.data.default_joint_pos).clone()
            robot.set_joint_position_target_index(
                target=default_joint_pos, joint_ids=list(range(default_joint_pos.shape[1]))
            )
        except Exception as exc:
            self.logger.warning(f"Could not pin default joint targets after reset: {exc}")


    def _package_observation(self, obs):
        """Convert an env observation into its wire form, optionally resizing cameras."""
        height = self.config.wire_image_resize_height
        width = self.config.wire_image_resize_width
        if height is None or width is None:
            return tree_to_cpu(obs)

        packaged = dict(obs)
        camera_obs = dict(obs.get("camera_obs") or {})
        record = {}
        for name, value in camera_obs.items():
            image = _as_torch(value)
            if image.ndim != 4 or image.shape[-1] not in (3, 4) or image.dtype != torch.uint8:
                continue
            camera_obs[name] = torch.from_numpy(resize_rgb_frames(image.cpu().numpy(), height, width))
            record[name] = image_preprocessing_record(tuple(image.shape[1:3]), (height, width))
        packaged["camera_obs"] = camera_obs
        if record != self._wire_image_record:
            self._wire_image_record = record
            self.logger.info(f"wire_image_preprocessing={json.dumps(record)}")
        return tree_to_cpu(packaged)

    def _package_info(self, info):
        """Convert an env info dict into its wire form (default: CPU passthrough)."""
        return tree_to_cpu(info)


    def _stabilize(self, obs, info):
        """Step the sim with a fixed action so contacts settle after reset."""
        steps = self.config.reset_stabilization_steps
        mode = self.config.stabilization_mode
        if steps <= 0 or mode == "none":
            return obs, info

        if mode == "hold":
            action = self.build_hold_action()
        elif mode == "zero":
            action = torch.zeros(
                self.env.num_envs, self.env.action_manager.total_action_dim, device=self.env.device
            )
        else:
            raise ValueError(f"Unknown stabilization mode: {mode!r}")

        self.logger.info(f"Stabilizing for {steps} steps (mode={mode})")
        for _ in range(steps):
            obs, _, terminated, truncated, info = self.env.step(action)
            if bool(terminated.any()) or bool(truncated.any()):
                self.logger.warning("Episode ended during stabilization; resetting again")
                obs, info = self.env.reset()
                break
        return obs, info


    def _dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        endpoint = request.get("endpoint", "")
        data = request.get("data") or {}
        self.logger.debug(f"Received request: {endpoint}")

        if endpoint == "ping":
            return {"status": "ok"}
        if endpoint == "info":
            return self.info()
        if endpoint == "reset":
            return self.reset(seed=data.get("seed"))
        if endpoint == "step":
            action = data.get("action")
            if action is None:
                return {"status": "error", "message": "No action data provided"}
            return self.step(action)
        if endpoint == "hold_action":
            return self.hold_action()
        if endpoint == "close":
            self.logger.info("Close requested by client")
            self._running = False
            return {"status": "ok", "message": "Server stopping"}
        return {"status": "error", "message": f"Unknown endpoint: {endpoint}"}

    def run(self) -> None:
        """Serve requests until a client sends ``close`` or the sim app exits."""
        self._running = True
        self.logger.info("DexBench env server is ready to accept requests")
        try:
            while self._running and self.simulation_app.is_running():
                if self.socket.poll(_RECV_POLL_TIMEOUT_MS if not self.config.render_on_idle else 50) == 0:
                    if self.config.render_on_idle:
                        self.env.sim.render()
                    continue
                message = self.socket.recv()
                try:
                    request = TorchSerializer.from_bytes(message)
                    response = self._dispatch(request)
                except Exception as exc:
                    self.logger.error(f"Error processing request: {exc}", exc_info=True)
                    response = {"status": "error", "message": f"Error processing request: {exc}"}
                self.socket.send(TorchSerializer.to_bytes(response))
        except KeyboardInterrupt:
            self.logger.info("Server interrupted by user")
        finally:
            self.socket.close()
            self.context.term()
            self.logger.info("DexBench env server stopped")


def _describe_tree(tree: Any) -> Any:
    """Summarize a nested observation dict as ``{key: {shape, dtype}}``."""
    if isinstance(tree, dict):
        return {key: _describe_tree(value) for key, value in tree.items()}
    if isinstance(tree, torch.Tensor):
        return {"shape": tuple(tree.shape), "dtype": str(tree.dtype)}
    return {"type": type(tree).__name__}
