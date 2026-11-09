# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""IsaacTeleop pipeline builder shared by the RBY1-M hand generations."""

from __future__ import annotations

import os
from pathlib import Path

def _read_hand_scaling_factor(config_path: Path) -> float:
    """Return the DexPilot ``scaling_factor`` from a hand retargeting YAML (default 1.0)."""
    try:
        import yaml

        with open(config_path) as handle:
            data = yaml.safe_load(handle) or {}
        return float(data.get("retargeting", {}).get("scaling_factor", 1.0))
    except Exception as exc:  # noqa: BLE001
        print(f"[RBY1 Wuji OpenXR Debug] Failed to read scaling_factor from {config_path}: {exc}")
        return 1.0

def build_rby1m_openxr_pipeline(
    arm_mode: object | None = None,
    idle_wrist_poses: dict[str, tuple[float, ...]] | None = None,
    *,
    action_order: tuple[str, ...],
    left_hand_config: Path,
    right_hand_config: Path,
    left_hand_urdf: Path,
    right_hand_urdf: Path,
    default_left_idle_pose: tuple[float, ...],
    default_right_idle_pose: tuple[float, ...],
    operator_to_mano_left: tuple[float, ...],
    operator_to_mano_right: tuple[float, ...],
    wrist_offset_rpy_left: tuple[float, float, float],
    wrist_offset_rpy_right: tuple[float, float, float],
):
    """Build the RBY1-M OpenXR teleoperation pipeline for one hand generation."""

    try:
        from isaacteleop.retargeters import (
            DexHandRetargeter,
            DexHandRetargeterConfig,
            Se3AbsRetargeter,
            Se3RetargeterConfig,
            TensorReorderer,
        )
        from isaacteleop.retargeters.dex_hand_retargeter import BaseRetargeter
        from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource, HandsSource
        from isaacteleop.retargeting_engine.interface import (
            OptionalType,
            OutputCombiner,
            TensorGroupType,
            ValueInput,
        )
        from isaacteleop.retargeting_engine.tensor_types import (
            DLDataType,
            ControllerInput,
            ControllerInputIndex,
            FloatType,
            HandInput,
            HandInputIndex,
            NDArrayType,
            TransformMatrix,
        )
        import numpy as np
        from dexbench_arena.diagnostics import teleop_trace
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "RBY1-M OpenXR teleop requires the IsaacTeleop runtime available in the "
            "IsaacLab-Arena Docker container."
        ) from exc

    hand_action_names = tuple(action_order[14:])
    left_hand_action_names = tuple(name for name in hand_action_names if name.startswith("left_"))
    right_hand_action_names = tuple(name for name in hand_action_names if name.startswith("right_"))

    class ConstantArrayRetargeter(BaseRetargeter):
        """Retargeter leaf that emits a fixed float32 vector."""

        def __init__(self, value: tuple[float, ...], name: str, output_name: str) -> None:
            self._value = np.asarray(value, dtype=np.float32)
            self._output_name = output_name
            super().__init__(name=name)

        def input_spec(self):
            return {}

        def output_spec(self):
            return {
                self._output_name: TensorGroupType(
                    self._output_name,
                    [
                        NDArrayType(
                            "value",
                            shape=(len(self._value),),
                            dtype=DLDataType.FLOAT,
                            dtype_bits=32,
                        )
                    ],
                )
            }

        def _compute_fn(self, inputs, outputs, context) -> None:
            outputs[self._output_name][0] = self._value

    class ConstantScalarsRetargeter(BaseRetargeter):
        """Retargeter leaf that emits fixed scalar values with named elements."""

        def __init__(
            self,
            element_names: tuple[str, ...],
            value: float,
            name: str,
            output_name: str,
        ) -> None:
            self._element_names = element_names
            self._value = float(value)
            self._output_name = output_name
            super().__init__(name=name)

        def input_spec(self):
            return {}

        def output_spec(self):
            return {
                self._output_name: TensorGroupType(
                    self._output_name,
                    [FloatType(name) for name in self._element_names],
                )
            }

        def _compute_fn(self, inputs, outputs, context) -> None:
            for index in range(len(self._element_names)):
                outputs[self._output_name][index] = self._value

    class Rby1mOpenXrDebugMarkers:
        """VisualizationMarkers-based RBY1 Wuji OpenXR debug markers."""

        def __init__(self, *, base_prim_path: str | None, enabled: bool) -> None:
            self._base_prim_path = base_prim_path
            self._enabled = enabled
            self._stage = None
            self._left_hand_markers = None
            self._right_hand_markers = None
            self._left_wrist_frame_marker = None
            self._right_wrist_frame_marker = None
            self._left_controller_grip_marker = None
            self._right_controller_grip_marker = None
            self._left_controller_aim_marker = None
            self._right_controller_aim_marker = None

        def _get_stage(self):
            if not self._enabled:
                return None
            if self._stage is None:
                import omni.usd

                self._stage = omni.usd.get_context().get_stage()
            return self._stage

        def _base_to_world(self, positions: np.ndarray) -> np.ndarray:
            if self._base_prim_path is None:
                return positions

            stage = self._get_stage()
            if stage is None:
                return positions

            try:
                from pxr import Gf, Usd, UsdGeom

                base_prim = stage.GetPrimAtPath(self._base_prim_path)
                if not base_prim.IsValid():
                    return positions
                mat = UsdGeom.Xformable(base_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                world_positions = []
                for pos in positions:
                    world_pos = mat.Transform(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
                    world_positions.append([float(world_pos[0]), float(world_pos[1]), float(world_pos[2])])
                return np.asarray(world_positions, dtype=np.float32)
            except Exception:
                return positions

        def _ensure_markers(self) -> bool:
            if not self._enabled:
                return False
            if (
                self._left_hand_markers is not None
                and self._right_hand_markers is not None
                and self._left_wrist_frame_marker is not None
                and self._right_wrist_frame_marker is not None
                and self._left_controller_grip_marker is not None
                and self._right_controller_grip_marker is not None
                and self._left_controller_aim_marker is not None
                and self._right_controller_aim_marker is not None
            ):
                return True

            try:
                import isaaclab.sim as sim_utils
                from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
                from isaaclab.markers.config import FRAME_MARKER_CFG

                def hand_marker_cfg(prim_path: str) -> VisualizationMarkersCfg:
                    return VisualizationMarkersCfg(
                        prim_path=prim_path,
                        markers={
                            "joint": sim_utils.SphereCfg(
                                radius=0.003,
                                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                            ),
                        },
                    )

                self._left_hand_markers = VisualizationMarkers(
                    hand_marker_cfg("/Visuals/dexbench_xr_left_hand_markers")
                )
                self._right_hand_markers = VisualizationMarkers(
                    hand_marker_cfg("/Visuals/dexbench_xr_right_hand_markers")
                )

                wrist_frame_cfg = FRAME_MARKER_CFG.copy()
                wrist_frame_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
                self._left_wrist_frame_marker = VisualizationMarkers(
                    wrist_frame_cfg.replace(prim_path="/Visuals/dexbench_xr_left_wrist_frame")
                )
                self._right_wrist_frame_marker = VisualizationMarkers(
                    wrist_frame_cfg.replace(prim_path="/Visuals/dexbench_xr_right_wrist_frame")
                )

                controller_grip_cfg = FRAME_MARKER_CFG.copy()
                controller_grip_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)
                controller_aim_cfg = FRAME_MARKER_CFG.copy()
                controller_aim_cfg.markers["frame"].scale = (0.045, 0.045, 0.045)
                self._left_controller_grip_marker = VisualizationMarkers(
                    controller_grip_cfg.replace(prim_path="/Visuals/dexbench_xr_left_controller_grip_frame")
                )
                self._right_controller_grip_marker = VisualizationMarkers(
                    controller_grip_cfg.replace(prim_path="/Visuals/dexbench_xr_right_controller_grip_frame")
                )
                self._left_controller_aim_marker = VisualizationMarkers(
                    controller_aim_cfg.replace(prim_path="/Visuals/dexbench_xr_left_controller_aim_frame")
                )
                self._right_controller_aim_marker = VisualizationMarkers(
                    controller_aim_cfg.replace(prim_path="/Visuals/dexbench_xr_right_controller_aim_frame")
                )
                for marker in (
                    self._left_controller_grip_marker,
                    self._right_controller_grip_marker,
                    self._left_controller_aim_marker,
                    self._right_controller_aim_marker,
                ):
                    marker.set_visibility(False)
                return True
            except Exception as exc:
                self._enabled = False
                print(f"[RBY1 Wuji OpenXR Debug] Disabling VisualizationMarkers after error: {exc}")
                return False

        def draw_wrist(self, side: str, pose: np.ndarray) -> None:
            if not self._ensure_markers():
                return

            world_pos = self._base_to_world(np.asarray([pose[:3]], dtype=np.float32))[0]
            orientation = np.asarray([pose[3:7]], dtype=np.float32)
            marker = self._left_wrist_frame_marker if side == "left" else self._right_wrist_frame_marker
            marker.visualize(translations=np.asarray([world_pos], dtype=np.float32), orientations=orientation)

        def draw_hand(self, side: str, joint_positions: np.ndarray, valid: np.ndarray) -> None:
            if not self._ensure_markers():
                return

            world_positions = self._base_to_world(joint_positions.astype(np.float32))
            valid_mask = valid.astype(bool)
            if not np.any(valid_mask):
                return
            marker = self._left_hand_markers if side == "left" else self._right_hand_markers
            marker.visualize(translations=world_positions[valid_mask].astype(np.float32))

        def hide_controller(self, side: str) -> None:
            if not self._ensure_markers():
                return
            for marker in self._controller_markers(side):
                marker.set_visibility(False)

        def draw_controller_pose(
            self,
            side: str,
            pose_kind: str,
            position: np.ndarray,
            orientation: np.ndarray,
            is_valid: bool,
        ) -> None:
            if not self._ensure_markers():
                return

            marker = self._controller_marker(side, pose_kind)
            position = np.asarray(position, dtype=np.float32).reshape(-1)
            orientation = np.asarray(orientation, dtype=np.float32).reshape(-1)
            if (
                not is_valid
                or position.size < 3
                or orientation.size < 4
                or not np.all(np.isfinite(position[:3]))
                or not np.all(np.isfinite(orientation[:4]))
                or np.linalg.norm(orientation[:4]) < 1.0e-6
            ):
                marker.set_visibility(False)
                return

            marker.set_visibility(True)
            world_pos = self._base_to_world(np.asarray([position[:3]], dtype=np.float32))[0]
            marker.visualize(
                translations=np.asarray([world_pos], dtype=np.float32),
                orientations=np.asarray([orientation[:4]], dtype=np.float32),
            )

        def _controller_markers(self, side: str):
            if side == "left":
                return (self._left_controller_grip_marker, self._left_controller_aim_marker)
            return (self._right_controller_grip_marker, self._right_controller_aim_marker)

        def _controller_marker(self, side: str, pose_kind: str):
            if side == "left":
                return self._left_controller_grip_marker if pose_kind == "grip" else self._left_controller_aim_marker
            return self._right_controller_grip_marker if pose_kind == "grip" else self._right_controller_aim_marker

    class WristDebugTapRetargeter(BaseRetargeter):
        """Pass-through wrist pose tap that draws the target pose."""

        def __init__(
            self, name: str, side: str, debug_markers: Rby1mOpenXrDebugMarkers, *, log_every: int = 0
        ) -> None:
            self._side = side
            self._debug_markers = debug_markers
            self._log_every = max(0, log_every)
            self._tick = 0
            self._first_pose: np.ndarray | None = None
            super().__init__(name=name)

        def input_spec(self):
            return {
                "ee_pose": TensorGroupType(
                    "ee_pose",
                    [
                        NDArrayType(
                            "pose",
                            shape=(7,),
                            dtype=DLDataType.FLOAT,
                            dtype_bits=32,
                        )
                    ],
                )
            }

        def output_spec(self):
            return self.input_spec()

        def _compute_fn(self, inputs, outputs, context) -> None:
            pose = np.asarray(inputs["ee_pose"][0], dtype=np.float32)
            outputs["ee_pose"][0] = pose
            teleop_trace.record_event("xr_wrist_target", {"side": self._side, "pose": pose})
            self._debug_markers.draw_wrist(self._side, pose)
            if self._log_every:
                if self._first_pose is None:
                    self._first_pose = pose.copy()
                self._tick += 1
                if self._tick % self._log_every == 0:
                    drift = float(np.linalg.norm(pose[:3] - self._first_pose[:3]))
                    print(
                        f"[RBY1 Wuji OpenXR Debug] {self._side}_wrist target "
                        f"pos={np.round(pose[:3], 3).tolist()} quat={np.round(pose[3:], 3).tolist()} "
                        f"drift_from_start={drift:.3f} m"
                    )

    class HandDebugTapRetargeter(BaseRetargeter):
        """Pass-through hand joint tap that draws tracked XR hand points and logs commands."""

        def __init__(
            self,
            name: str,
            side: str,
            element_names: tuple[str, ...],
            debug_markers: Rby1mOpenXrDebugMarkers,
            *,
            log_every: int,
            hand_scaling: float = 1.0,
        ) -> None:
            self._side = side
            self._element_names = element_names
            self._debug_markers = debug_markers
            self._log_every = max(0, log_every)
            self._hand_scaling = hand_scaling
            self._counter = 0
            super().__init__(name=name)

        def _scaled(self, positions: np.ndarray) -> np.ndarray:
            """Scale tracked hand joints about the wrist by the DexPilot ``scaling_factor``."""
            if self._hand_scaling == 1.0 or positions.shape[0] == 0:
                return positions
            origin = positions[1] if positions.shape[0] > 1 else positions[0]
            return (origin + self._hand_scaling * (positions - origin)).astype(np.float32)

        def input_spec(self):
            return {
                f"hand_{self._side}": OptionalType(HandInput()),
                f"marker_hand_{self._side}": OptionalType(HandInput()),
                "hand_joints": TensorGroupType(
                    "hand_joints",
                    [FloatType(name) for name in self._element_names],
                ),
            }

        def output_spec(self):
            return {
                "hand_joints": TensorGroupType(
                    "hand_joints",
                    [FloatType(name) for name in self._element_names],
                )
            }

        def _compute_fn(self, inputs, outputs, context) -> None:
            values = np.zeros(len(self._element_names), dtype=np.float32)
            hand_joint_group = inputs["hand_joints"]
            for index in range(len(self._element_names)):
                value = float(hand_joint_group[index])
                values[index] = value
                outputs["hand_joints"][index] = value

            self._counter += 1
            if self._log_every and self._counter % self._log_every == 0:
                print(
                    "[RBY1 Wuji OpenXR Debug] "
                    f"{self._side}_hand action min={values.min():.4f} "
                    f"max={values.max():.4f} mean={values.mean():.4f} "
                    f"values={np.array2string(values, precision=3, suppress_small=True)}"
                )

            hand_group = inputs[f"hand_{self._side}"]
            if hand_group.is_none:
                teleop_trace.record_event(
                    "xr_hand",
                    {
                        "side": self._side,
                        "tracking": False,
                        "joint_names": self._element_names,
                        "command": values,
                    },
                )
                if self._log_every and self._counter % self._log_every == 0:
                    print(f"[RBY1 Wuji OpenXR Debug] {self._side}_hand tracking is none")
                return

            joint_positions = np.asarray(np.from_dlpack(hand_group[HandInputIndex.JOINT_POSITIONS]), dtype=np.float32)
            joint_orientations = np.asarray(
                np.from_dlpack(hand_group[HandInputIndex.JOINT_ORIENTATIONS]), dtype=np.float32
            )
            joint_valid = np.asarray(np.from_dlpack(hand_group[HandInputIndex.JOINT_VALID]), dtype=np.uint8)
            teleop_trace.record_event(
                "xr_hand",
                {
                    "side": self._side,
                    "tracking": True,
                    "joint_names": self._element_names,
                    "command": values,
                    "joint_positions": joint_positions,
                    "joint_orientations": joint_orientations,
                    "joint_valid": joint_valid,
                },
            )
            marker_hand_group = inputs[f"marker_hand_{self._side}"]
            if marker_hand_group.is_none:
                self._debug_markers.draw_hand(self._side, self._scaled(joint_positions), joint_valid)
                return

            marker_joint_positions = np.asarray(
                np.from_dlpack(marker_hand_group[HandInputIndex.JOINT_POSITIONS]), dtype=np.float32
            )
            marker_joint_valid = np.asarray(
                np.from_dlpack(marker_hand_group[HandInputIndex.JOINT_VALID]), dtype=np.uint8
            )
            self._debug_markers.draw_hand(self._side, self._scaled(marker_joint_positions), marker_joint_valid)

    class ControllerDebugTapRetargeter(BaseRetargeter):
        """Pass-through action tap that draws and traces XR controller poses."""

        def __init__(
            self,
            name: str,
            debug_markers: Rby1mOpenXrDebugMarkers,
            *,
            log_every: int,
        ) -> None:
            self._debug_markers = debug_markers
            self._log_every = max(0, log_every)
            self._counter = 0
            super().__init__(name=name)

        @staticmethod
        def _action_type():
            return TensorGroupType(
                "output",
                [
                    NDArrayType(
                        "action_flat",
                        shape=(len(action_order),),
                        dtype=DLDataType.FLOAT,
                        dtype_bits=32,
                    )
                ],
            )

        def input_spec(self):
            return {
                "action": self._action_type(),
                ControllersSource.LEFT: OptionalType(ControllerInput()),
                ControllersSource.RIGHT: OptionalType(ControllerInput()),
            }

        def output_spec(self):
            return {"output": self._action_type()}

        def _compute_fn(self, inputs, outputs, context) -> None:
            action = np.asarray(inputs["action"][0], dtype=np.float32)
            outputs["output"][0] = action

            self._counter += 1
            self._record_controller("left", inputs[ControllersSource.LEFT])
            self._record_controller("right", inputs[ControllersSource.RIGHT])

        def _record_controller(self, side: str, controller_group) -> None:
            if controller_group.is_none:
                self._debug_markers.hide_controller(side)
                teleop_trace.record_event("xr_controller", {"side": side, "tracking": False})
                if self._log_every and self._counter % self._log_every == 0:
                    print(f"[RBY1 Wuji OpenXR Debug] {side}_controller tracking is none")
                return

            grip_position = self._array(controller_group[ControllerInputIndex.GRIP_POSITION])
            grip_orientation = self._array(controller_group[ControllerInputIndex.GRIP_ORIENTATION])
            grip_is_valid = self._bool(controller_group[ControllerInputIndex.GRIP_IS_VALID])
            aim_position = self._array(controller_group[ControllerInputIndex.AIM_POSITION])
            aim_orientation = self._array(controller_group[ControllerInputIndex.AIM_ORIENTATION])
            aim_is_valid = self._bool(controller_group[ControllerInputIndex.AIM_IS_VALID])
            inputs_payload = {
                "primary_click": self._float(controller_group[ControllerInputIndex.PRIMARY_CLICK]),
                "secondary_click": self._float(controller_group[ControllerInputIndex.SECONDARY_CLICK]),
                "thumbstick_x": self._float(controller_group[ControllerInputIndex.THUMBSTICK_X]),
                "thumbstick_y": self._float(controller_group[ControllerInputIndex.THUMBSTICK_Y]),
                "thumbstick_click": self._float(controller_group[ControllerInputIndex.THUMBSTICK_CLICK]),
                "squeeze_value": self._float(controller_group[ControllerInputIndex.SQUEEZE_VALUE]),
                "trigger_value": self._float(controller_group[ControllerInputIndex.TRIGGER_VALUE]),
            }

            self._debug_markers.draw_controller_pose(side, "grip", grip_position, grip_orientation, grip_is_valid)
            self._debug_markers.draw_controller_pose(side, "aim", aim_position, aim_orientation, aim_is_valid)
            teleop_trace.record_event(
                "xr_controller",
                {
                    "side": side,
                    "tracking": True,
                    "grip_valid": grip_is_valid,
                    "aim_valid": aim_is_valid,
                    "grip_pose": np.concatenate([grip_position[:3], grip_orientation[:4]]),
                    "aim_pose": np.concatenate([aim_position[:3], aim_orientation[:4]]),
                    **inputs_payload,
                },
            )

            if self._log_every and self._counter % self._log_every == 0:
                print(
                    "[RBY1 Wuji OpenXR Debug] "
                    f"{side}_controller grip_valid={grip_is_valid} "
                    f"aim_valid={aim_is_valid} "
                    f"grip_pos={np.array2string(grip_position[:3], precision=3, suppress_small=True)} "
                    f"aim_pos={np.array2string(aim_position[:3], precision=3, suppress_small=True)} "
                    f"trigger={inputs_payload['trigger_value']:.3f} "
                    f"squeeze={inputs_payload['squeeze_value']:.3f}"
                )

        @staticmethod
        def _array(value) -> np.ndarray:
            try:
                return np.asarray(np.from_dlpack(value), dtype=np.float32)
            except (AttributeError, TypeError):
                return np.asarray(value, dtype=np.float32)

        @classmethod
        def _float(cls, value) -> float:
            return float(cls._array(value).reshape(-1)[0])

        @classmethod
        def _bool(cls, value) -> bool:
            return bool(cls._float(value))

    def _seed_retargeter_open_hand(retargeter) -> None:
        dex_hand = getattr(retargeter, "_dex_hand", None)
        if dex_hand is None:
            return

        robot = dex_hand.optimizer.robot
        open_qpos = np.zeros(robot.dof, dtype=np.float32)
        open_qpos = np.clip(open_qpos, robot.joint_limits[:, 0], robot.joint_limits[:, 1])
        dex_hand.set_qpos(open_qpos)

    def _build_dex_hand_retargeter(config: DexHandRetargeterConfig, name: str) -> DexHandRetargeter:
        retargeter = DexHandRetargeter(config, name=name)
        _seed_retargeter_open_hand(retargeter)
        return retargeter

    mode = getattr(arm_mode, "value", arm_mode) or "dual_arm"
    idle_wrist_poses = idle_wrist_poses or {}
    left_idle_pose = tuple(idle_wrist_poses.get("left", default_left_idle_pose))
    right_idle_pose = tuple(idle_wrist_poses.get("right", default_right_idle_pose))
    debug_enabled = os.environ.get("DEXBENCH_RBY1M_OPENXR_DEBUG", "").lower() in {"1", "true", "yes", "on"}
    trace_enabled = teleop_trace.is_enabled()
    debug_log_every = int(os.environ.get("DEXBENCH_RBY1M_OPENXR_DEBUG_LOG_EVERY", "120"))
    left_hand_scaling = _read_hand_scaling_factor(left_hand_config)
    right_hand_scaling = _read_hand_scaling_factor(right_hand_config)
    debug_markers = Rby1mOpenXrDebugMarkers(
        base_prim_path=None,
        enabled=debug_enabled,
    )

    hands = HandsSource(name="hands")
    transform_input = ValueInput("world_T_anchor", TransformMatrix())
    transformed_hands = hands.transformed(transform_input.output(ValueInput.VALUE))

    if mode == "right":
        left_se3 = ConstantArrayRetargeter(left_idle_pose, "left_wrist_idle", "ee_pose")
    else:
        left_se3 = Se3AbsRetargeter(
            Se3RetargeterConfig(
                input_device=HandsSource.LEFT,
                zero_out_xy_rotation=False,
                use_wrist_rotation=True,
                use_wrist_position=True,
                target_offset_roll=float(wrist_offset_rpy_left[0]),
                target_offset_pitch=float(wrist_offset_rpy_left[1]),
                target_offset_yaw=float(wrist_offset_rpy_left[2]),
            ),
            name="left_wrist_pose",
        ).connect({HandsSource.LEFT: transformed_hands.output(HandsSource.LEFT)})

    if mode == "left":
        right_se3 = ConstantArrayRetargeter(right_idle_pose, "right_wrist_idle", "ee_pose")
    else:
        right_se3 = Se3AbsRetargeter(
            Se3RetargeterConfig(
                input_device=HandsSource.RIGHT,
                zero_out_xy_rotation=False,
                use_wrist_rotation=True,
                use_wrist_position=True,
                target_offset_roll=float(wrist_offset_rpy_right[0]),
                target_offset_pitch=float(wrist_offset_rpy_right[1]),
                target_offset_yaw=float(wrist_offset_rpy_right[2]),
            ),
            name="right_wrist_pose",
        ).connect({HandsSource.RIGHT: transformed_hands.output(HandsSource.RIGHT)})

    if mode == "right":
        left_hand = ConstantScalarsRetargeter(left_hand_action_names, 0.0, "left_hand_idle", "hand_joints")
    else:
        left_hand = _build_dex_hand_retargeter(
            DexHandRetargeterConfig(
                hand_retargeting_config=str(left_hand_config),
                hand_urdf=str(left_hand_urdf),
                hand_joint_names=list(left_hand_action_names),
                handtracking_to_baselink_frame_transform=operator_to_mano_left,
                hand_side="left",
            ),
            name="left_hand",
        ).connect({HandsSource.LEFT: hands.output(HandsSource.LEFT)})

    if mode == "left":
        right_hand = ConstantScalarsRetargeter(right_hand_action_names, 0.0, "right_hand_idle", "hand_joints")
    else:
        right_hand = _build_dex_hand_retargeter(
            DexHandRetargeterConfig(
                hand_retargeting_config=str(right_hand_config),
                hand_urdf=str(right_hand_urdf),
                hand_joint_names=list(right_hand_action_names),
                handtracking_to_baselink_frame_transform=operator_to_mano_right,
                hand_side="right",
            ),
            name="right_hand",
        ).connect({HandsSource.RIGHT: hands.output(HandsSource.RIGHT)})

    left_wrist_output = left_se3.output("ee_pose")
    right_wrist_output = right_se3.output("ee_pose")

    from dexbench_arena.embodiments.wrist_pose_latch import make_wrist_pose_latch

    if mode != "right":
        left_wrist_output = (
            make_wrist_pose_latch("left", left_idle_pose)
            .connect({"ee_pose": left_wrist_output, "hand_left": transformed_hands.output(HandsSource.LEFT)})
            .output("ee_pose")
        )
    if mode != "left":
        right_wrist_output = (
            make_wrist_pose_latch("right", right_idle_pose)
            .connect({"ee_pose": right_wrist_output, "hand_right": transformed_hands.output(HandsSource.RIGHT)})
            .output("ee_pose")
        )

    left_hand_output = left_hand.output("hand_joints")
    right_hand_output = right_hand.output("hand_joints")

    if debug_enabled or trace_enabled:
        left_wrist_output = WristDebugTapRetargeter(
            "left_wrist_debug_tap",
            "left",
            debug_markers,
            log_every=debug_log_every if debug_enabled else 0,
        ).connect({"ee_pose": left_wrist_output}).output("ee_pose")
        right_wrist_output = WristDebugTapRetargeter(
            "right_wrist_debug_tap",
            "right",
            debug_markers,
            log_every=debug_log_every if debug_enabled else 0,
        ).connect({"ee_pose": right_wrist_output}).output("ee_pose")
        left_hand_output = HandDebugTapRetargeter(
            "left_hand_debug_tap",
            "left",
            left_hand_action_names,
            debug_markers,
            log_every=debug_log_every if debug_enabled else 0,
            hand_scaling=left_hand_scaling,
        ).connect(
            {
                "hand_left": hands.output(HandsSource.LEFT),
                "marker_hand_left": transformed_hands.output(HandsSource.LEFT),
                "hand_joints": left_hand_output,
            }
        ).output("hand_joints")
        right_hand_output = HandDebugTapRetargeter(
            "right_hand_debug_tap",
            "right",
            right_hand_action_names,
            debug_markers,
            log_every=debug_log_every if debug_enabled else 0,
            hand_scaling=right_hand_scaling,
        ).connect(
            {
                "hand_right": hands.output(HandsSource.RIGHT),
                "marker_hand_right": transformed_hands.output(HandsSource.RIGHT),
                "hand_joints": right_hand_output,
            }
        ).output("hand_joints")

    left_wrist_elements = (
        "left_wrist_pos_x",
        "left_wrist_pos_y",
        "left_wrist_pos_z",
        "left_wrist_quat_x",
        "left_wrist_quat_y",
        "left_wrist_quat_z",
        "left_wrist_quat_w",
    )
    right_wrist_elements = (
        "right_wrist_pos_x",
        "right_wrist_pos_y",
        "right_wrist_pos_z",
        "right_wrist_quat_x",
        "right_wrist_quat_y",
        "right_wrist_quat_z",
        "right_wrist_quat_w",
    )

    reorderer = TensorReorderer(
        input_config={
            "left_wrist_pose": left_wrist_elements,
            "right_wrist_pose": right_wrist_elements,
            "left_hand_joints": left_hand_action_names,
            "right_hand_joints": right_hand_action_names,
        },
        output_order=action_order,
        name="rby1_wuji_action_reorderer",
        input_types={
            "left_wrist_pose": "array",
            "right_wrist_pose": "array",
            "left_hand_joints": "scalar",
            "right_hand_joints": "scalar",
        },
    )
    connected_reorderer = reorderer.connect(
        {
            "left_wrist_pose": left_wrist_output,
            "right_wrist_pose": right_wrist_output,
            "left_hand_joints": left_hand_output,
            "right_hand_joints": right_hand_output,
        }
    )
    action_output = connected_reorderer.output("output")

    if debug_enabled or trace_enabled:
        controllers = ControllersSource(name="controllers")
        transformed_controllers = controllers.transformed(transform_input.output(ValueInput.VALUE))
        action_output = ControllerDebugTapRetargeter(
            "controller_debug_tap",
            debug_markers,
            log_every=debug_log_every if debug_enabled else 0,
        ).connect(
            {
                "action": action_output,
                ControllersSource.LEFT: transformed_controllers.output(ControllersSource.LEFT),
                ControllersSource.RIGHT: transformed_controllers.output(ControllersSource.RIGHT),
            }
        ).output("output")

    return OutputCombiner({"action": action_output})
