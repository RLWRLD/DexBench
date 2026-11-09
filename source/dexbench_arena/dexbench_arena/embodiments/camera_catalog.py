# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Camera catalog: a registry of camera MODELS composed onto per-robot MOUNT points."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from typing import Any


class _Unset:
    """Sentinel for 'do not pass this keyword at all', so callee defaults apply."""

    _instance: _Unset | None = None

    def __new__(cls) -> _Unset:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "UNSET"

    def __bool__(self) -> bool:
        return False


UNSET = _Unset()

@dataclass(frozen=True)
class CameraModel:
    """One camera MODEL -- a part number at one capture mode, not one physical unit."""

    name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    baseline_m: float | None = None
    distortion_k: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
    provenance: str = ""

    def scaled(self, scale: float) -> CameraModel:
        """Return this model rescaled by ``scale`` (size and intrinsics together)."""
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Camera scale must be finite and positive")
        width, height = round(self.width * scale), round(self.height * scale)
        if width < 1 or height < 1 or width * self.height != height * self.width:
            raise ValueError("Camera scale must produce positive dimensions with the original aspect ratio")
        if scale == 1.0:
            return self
        scale = width / self.width
        return replace(
            self,
            width=width,
            height=height,
            fx=self.fx * scale,
            fy=self.fy * scale,
            cx=self.cx * scale,
            cy=self.cy * scale,
        )

    def intrinsic_matrix(self) -> list[float]:
        """Row-major 3x3 pinhole K, flattened -- the shape IsaacLab wants."""
        return [self.fx, 0.0, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0]

    def pinhole_cfg(
        self,
        *,
        clipping_range: tuple[float, float] | _Unset = UNSET,
        focal_length: float | None = None,
    ) -> Any:
        """Build the closest RTX ``PinholeCameraCfg`` for this model."""
        import isaaclab.sim as sim_utils

        kwargs: dict[str, Any] = {}
        if not isinstance(clipping_range, _Unset):
            kwargs["clipping_range"] = clipping_range
        if focal_length is not None:
            kwargs["focal_length"] = focal_length
        return sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=self.intrinsic_matrix(),
            width=self.width,
            height=self.height,
            **kwargs,
        )


@dataclass(frozen=True)
class UnitCalibration:
    """One physical unit's measured optical calibration."""

    unit: str
    """Which physical unit this is, e.g. a serial number or the robot that carries it."""
    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None
    baseline_m: float | None = None
    distortion_k: tuple[float, ...] | None = None
    provenance: str = ""
    """When and how it was measured. Required reading before anyone edits the numbers."""

    def applied_to(self, model: CameraModel) -> CameraModel:
        """Return ``model`` with this unit's measured fields substituted in."""
        overrides = {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name not in ("unit", "provenance") and getattr(self, f.name) is not None
        }
        if not overrides:
            return model
        return replace(model, name=f"{model.name}@{self.unit}", **overrides)

    def deltas(self, model: CameraModel) -> dict[str, tuple[object, object]]:
        """``{field: (model_value, unit_value)}`` for every field this unit moves."""
        return {
            f.name: (getattr(model, f.name), getattr(self, f.name))
            for f in fields(self)
            if f.name not in ("unit", "provenance")
            and getattr(self, f.name) is not None
            and getattr(self, f.name) != getattr(model, f.name)
        }



ZED_MINI_SPEC_BASELINE_M = 0.063
"""Nominal mechanical baseline from the ZED Mini specification, not a calibration."""


ZED_MINI_HD720 = CameraModel(
    name="zed_mini_hd720",
    width=1280,
    height=720,
    fx=736.0,
    fy=736.0,
    cx=640.0,
    cy=360.0,
    baseline_m=None,
    provenance=(
        "ZED Mini HD720 nominal metadata. Stereolabs lists 1280x720 output and a 63 mm "
        "baseline at https://store.stereolabs.com/products/zed-mini, and lists 736 px as "
        "the ZED Mini HD720 focal length at https://support.stereolabs.com/hc/en-us/articles/"
        "360007395634-What-is-the-camera-focal-length-and-field-of-view while noting that "
        "exact values vary by physical calibration. The centred principal point is nominal. "
        "Stereo geometry is deliberately not inherited from the model: each stereo rig must "
        "explicitly select the product specification or apply its physical camera's measured "
        "UnitCalibration."
    ),
)

ZED_X_ONE_S_WIDE = CameraModel(
    name="zed_x_one_s_wide_nominal_rectified_16_9",
    width=1280, height=720,
    fx=737.0 * (1280.0 / 1920.0),
    fy=737.0 * (1280.0 / 1920.0),
    cx=640.0, cy=360.0,
    provenance=(
        "Stereolabs ZED X One S Wide (ZED-412012), 2.2 mm, GMSL2 mono RGB. "
        "Manufacturer nominal rectified HD1080 (1920x1080): fx=fy=737 px; resized "
        "to 1280x720, with an assumed centred principal point and zero distortion. "
        "640x360 at scale 0.5 uses fx=fy=245.6666667 px. Not SDK/per-unit calibration; "
        "the product's 110 degree optical FOV is not the rectified pinhole FOV. "
        "https://support.stereolabs.com/articles/8809264540-what-is-the-camera-focal-length-and-field-of-view"
    ),
)

CAMERA_MODELS: dict[str, CameraModel] = {
    model.name: model for model in (ZED_MINI_HD720, ZED_X_ONE_S_WIDE)
}



RBY1_HEAD_ZED_MINI = UnitCalibration(
    unit="rby1_head",
    fx=732.0785522460938,
    fy=732.0785522460938,
    cx=633.6272583007812,
    cy=356.186279296875,
    baseline_m=0.0628986191,
    provenance=(
        "The physical RBY1 robot's head ZED Mini, measured with simple-easyhec joint stereo "
        "calibration; HD rectified intrinsics (the ROS CameraInfo dumps report the same "
        "rectified K for both eyes and zero distortion), baseline from the right-optical offset "
        "of that same solve. RBY1-1 and RBY1-2 are two hand configurations of this same robot "
        "and must both reference this single calibration. An optional override; the RBY1 rig "
        "renders the ideal model instead."
    ),
)

RBY1_1_HEAD_ZED_MINI = RBY1_HEAD_ZED_MINI
RBY1_2_HEAD_ZED_MINI = RBY1_HEAD_ZED_MINI


def get_camera_model(name: str) -> CameraModel:
    if name not in CAMERA_MODELS:
        raise KeyError(f"Unknown camera model '{name}'. Known: {sorted(CAMERA_MODELS)}")
    return CAMERA_MODELS[name]




def _rotate_x_axis(rot_xyzw: tuple[float, float, float, float], distance: float) -> tuple[float, float, float]:
    """Rotate ``(distance, 0, 0)`` by the unit quaternion ``rot_xyzw``."""
    x, y, z, w = rot_xyzw
    return (
        (1.0 - 2.0 * (y * y + z * z)) * distance,
        (2.0 * (x * y + z * w)) * distance,
        (2.0 * (x * z - y * w)) * distance,
    )


@dataclass(frozen=True)
class CameraMount:
    """One mount point: a camera bolted to a named robot link at a fixed offset."""

    obs_key: str
    prim_name: str
    parent_link: str
    pos: tuple[float, float, float]
    rot: tuple[float, float, float, float]
    """Orientation as xyzw, in the ``convention`` frame."""
    convention: str = "ros"


@dataclass(frozen=True)
class StereoRightEye:
    """The right eye of a stereo pair, specified as an offset from the left one."""

    obs_key: str
    prim_name: str
    baseline_m: float | None = None
    """Ideal metres along local +x; a measured unit-calibration baseline takes precedence."""
    pos_override: tuple[float, float, float] | None = None
    """The as-recorded right-eye position, kept verbatim in place of the derivation."""
    override_tolerance_m: float = 1.0e-9
    """How far ``pos_override`` may sit from the derivation before import fails."""
    note: str = ""


@dataclass(frozen=True)
class CameraRig:
    """A camera model plus the mount(s) that carry it on one embodiment."""

    model: str
    left: CameraMount
    right: StereoRightEye | None = None
    calibration: UnitCalibration | None = None
    robot_prim: str = "Robot"
    clipping_range: tuple[float, float] | _Unset = UNSET
    focal_length: float | None = None
    update_period: float = 0.0
    data_types: tuple[str, ...] = ("rgb",)
    update_latest_camera_pose: bool | None = None
    """``None`` omits the keyword entirely (IsaacLab's own default applies)."""
    default_scale: float = 1.0
    """Raster scale relative to the calibrated model; optics and mounts stay fixed."""
    antialiasing_mode: str | None = None
    """RTX anti-aliasing for both eyes; ``None`` retains the renderer default."""
    def __post_init__(self) -> None:
        self.calibrated_model()
        self.mounts()
        if self.antialiasing_mode not in (None, "Off", "FXAA", "DLSS", "TAA", "DLAA"):
            raise ValueError(f"Unknown camera anti-aliasing mode: {self.antialiasing_mode!r}")

    def calibrated_model(self) -> CameraModel:
        """The rig's model with this robot's unit override applied, unscaled."""
        model = get_camera_model(self.model)
        return model if self.calibration is None else self.calibration.applied_to(model)

    def resolved_model(self, scale: float | None = None) -> CameraModel:
        """The rig's calibrated model with any per-run scaling also applied."""
        return self.calibrated_model().scaled(self.default_scale if scale is None else scale)

    def baseline_m(self) -> float:
        """The stereo baseline this rig mounts its eyes at, in metres."""
        if self.right is None:
            raise ValueError(f"rig on {self.left.parent_link} is mono; it has no baseline")
        if self.calibration is not None and self.calibration.baseline_m is not None:
            return self.calibration.baseline_m
        if self.right.baseline_m is not None:
            return self.right.baseline_m
        baseline = self.calibrated_model().baseline_m
        if baseline is None:
            raise ValueError(
                f"stereo rig on {self.left.parent_link} needs a baseline: model "
                f"'{self.model}' declares none, so StereoRightEye.baseline_m is required"
            )
        return baseline

    def derived_right_pos(self) -> tuple[float, float, float]:
        """Where the right eye lands as a pure baseline translation of the left one."""
        offset = _rotate_x_axis(self.left.rot, self.baseline_m())
        return (
            self.left.pos[0] + offset[0],
            self.left.pos[1] + offset[1],
            self.left.pos[2] + offset[2],
        )

    def right_override_residual_m(self) -> float:
        """Distance between a declared ``pos_override`` and the pure-baseline derivation."""
        if self.right is None or self.right.pos_override is None:
            return 0.0
        derived = self.derived_right_pos()
        return sum((a - b) ** 2 for a, b in zip(self.right.pos_override, derived)) ** 0.5

    def right_mount(self) -> CameraMount:
        """The right eye as a full mount: derived pose, or the declared override."""
        if self.right is None:
            raise ValueError(f"rig on {self.left.parent_link} is mono; it has no right eye")
        pos = self.right.pos_override
        if pos is None:
            pos = self.derived_right_pos()
        else:
            residual = self.right_override_residual_m()
            if residual > self.right.override_tolerance_m:
                raise ValueError(
                    f"right-eye pos_override on {self.left.parent_link} is {residual:.3e} m from "
                    f"the pure-baseline derivation {self.derived_right_pos()}, above the declared "
                    f"tolerance {self.right.override_tolerance_m:.3e} m. Either the baseline or the "
                    "override is wrong; if the asymmetry is real, raise the tolerance and say why "
                    "in StereoRightEye.note."
                )
        return CameraMount(
            obs_key=self.right.obs_key,
            prim_name=self.right.prim_name,
            parent_link=self.left.parent_link,
            pos=pos,
            rot=self.left.rot,
            convention=self.left.convention,
        )

    def mounts(self) -> tuple[CameraMount, ...]:
        """Every mount on this rig, left eye first."""
        if self.right is None:
            return (self.left,)
        return (self.left, self.right_mount())

    def prim_path(self, mount: CameraMount) -> str:
        return f"{{ENV_REGEX_NS}}/{self.robot_prim}/{mount.parent_link}/{mount.prim_name}"

    def build(self, scale: float | None = None) -> dict[str, Any]:
        """Resolve this rig into plain ``CameraCfg`` declarations."""
        from isaaclab.sensors import CameraCfg
        from isaaclab_physx.renderers.isaac_rtx_renderer_cfg import (
            IsaacRtxRendererCfg,
            IsaacRtxRendererGlobalSettingsCfg,
        )

        model = self.resolved_model(scale)
        spawn_kwargs: dict[str, Any] = {"focal_length": self.focal_length}
        if not isinstance(self.clipping_range, _Unset):
            spawn_kwargs["clipping_range"] = self.clipping_range

        cameras: dict[str, Any] = {}
        for mount in self.mounts():
            kwargs: dict[str, Any] = {
                "prim_path": self.prim_path(mount),
                "update_period": self.update_period,
                "height": model.height,
                "width": model.width,
                "data_types": list(self.data_types),
                "spawn": model.pinhole_cfg(**spawn_kwargs),
                "offset": CameraCfg.OffsetCfg(pos=mount.pos, rot=mount.rot, convention=mount.convention),
            }
            if self.update_latest_camera_pose is not None:
                kwargs["update_latest_camera_pose"] = self.update_latest_camera_pose
            if self.antialiasing_mode is not None:
                kwargs["renderer_cfg"] = IsaacRtxRendererCfg(
                    global_settings=IsaacRtxRendererGlobalSettingsCfg(antialiasing_mode=self.antialiasing_mode)
                )
            cameras[mount.obs_key] = CameraCfg(**kwargs)
        return cameras


def parse_runtime_camera_calibration(env) -> dict[str, dict[str, Any]]:
    """Parse camera models actually constructed by Isaac Lab."""
    sensors = getattr(getattr(env, "scene", None), "sensors", None) or {}
    calibration: dict[str, dict[str, Any]] = {}
    positions: dict[str, tuple[float, ...]] = {}
    for name, sensor in sensors.items():
        cfg = getattr(sensor, "cfg", None)
        if not hasattr(cfg, "width") or not hasattr(cfg, "height"):
            continue
        matrices = getattr(
            sensor.data.intrinsic_matrices, "torch", sensor.data.intrinsic_matrices
        )
        matrix = matrices[0]
        if hasattr(matrix, "detach"):
            matrix = matrix.detach().cpu()
        values = matrix.tolist()
        source = f"{name}_rgb"
        calibration[source] = {
            "model": "pinhole",
            "width": int(cfg.width),
            "height": int(cfg.height),
            "fx": float(values[0][0]),
            "fy": float(values[1][1]),
            "cx": float(values[0][2]),
            "cy": float(values[1][2]),
        }
        positions[source] = tuple(float(value) for value in cfg.offset.pos)

    left = "robot_pov_left_cam_rgb"
    right = "robot_pov_right_cam_rgb"
    if left in positions and right in positions:
        baseline = math.sqrt(
            sum(
                (lhs - rhs) ** 2
                for lhs, rhs in zip(positions[left], positions[right], strict=True)
            )
        )
        calibration[left]["baseline_m"] = baseline
        calibration[right]["baseline_m"] = baseline
    return calibration
