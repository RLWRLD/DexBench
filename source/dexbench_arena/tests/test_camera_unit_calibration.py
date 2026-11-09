# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Keep physical camera calibrations separate from ideal model metadata."""

from __future__ import annotations

import importlib.util
import pytest

from dexbench_arena.embodiments.camera_catalog import (
    ZED_MINI_HD720,
    ZED_MINI_SPEC_BASELINE_M,
    CameraModel,
    CameraMount,
    CameraRig,
    StereoRightEye,
    UnitCalibration,
)

SYNTHETIC_MODEL = CameraModel(
    name="synthetic_stereo",
    width=100,
    height=50,
    fx=10.0,
    fy=10.0,
    cx=50.0,
    cy=25.0,
    baseline_m=0.10,
    provenance="test fixture; not a real part",
)

SYNTHETIC_UNIT = UnitCalibration(
    unit="synthetic_unit",
    fx=11.0,
    baseline_m=0.20,
    provenance="test fixture; deviates on exactly two fields",
)


def _rig(model: CameraModel, calibration: UnitCalibration | None) -> CameraRig:
    """A stereo rig with an explicit ideal baseline, identity-rotated onto world +x."""
    return CameraRig(
        model=model.name,
        calibration=calibration,
        left=CameraMount(
            obs_key="robot_pov_left_cam",
            prim_name="left_cam",
            parent_link="link",
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        right=StereoRightEye(
            obs_key="robot_pov_right_cam",
            prim_name="right_cam",
            baseline_m=model.baseline_m,
        ),
    )


@pytest.fixture
def registered_synthetic_model(monkeypatch):
    """Register SYNTHETIC_MODEL for the duration of one test."""
    from dexbench_arena.embodiments import camera_catalog

    models = dict(camera_catalog.CAMERA_MODELS)
    models[SYNTHETIC_MODEL.name] = SYNTHETIC_MODEL
    monkeypatch.setattr(camera_catalog, "CAMERA_MODELS", models)
    return SYNTHETIC_MODEL


def test_applied_to_substitutes_only_the_measured_fields():
    calibrated = SYNTHETIC_UNIT.applied_to(SYNTHETIC_MODEL)

    assert calibrated.fx == 11.0
    assert calibrated.baseline_m == 0.20
    assert calibrated.fy == SYNTHETIC_MODEL.fy
    assert calibrated.cx == SYNTHETIC_MODEL.cx
    assert calibrated.cy == SYNTHETIC_MODEL.cy
    assert calibrated.width == SYNTHETIC_MODEL.width
    assert calibrated.height == SYNTHETIC_MODEL.height
    assert calibrated.name == "synthetic_stereo@synthetic_unit"
    assert SYNTHETIC_MODEL.fx == 10.0


def test_deltas_report_exactly_what_moved():
    assert SYNTHETIC_UNIT.deltas(SYNTHETIC_MODEL) == {
        "fx": (10.0, 11.0),
        "baseline_m": (0.10, 0.20),
    }


def test_calibration_with_no_overrides_returns_the_model_unchanged():
    empty = UnitCalibration(unit="stock_unit")
    assert empty.applied_to(SYNTHETIC_MODEL) is SYNTHETIC_MODEL
    assert empty.deltas(SYNTHETIC_MODEL) == {}


def test_rig_without_calibration_renders_the_model(registered_synthetic_model):
    rig = _rig(SYNTHETIC_MODEL, None)

    assert rig.calibrated_model() is SYNTHETIC_MODEL
    assert rig.baseline_m() == 0.10
    assert rig.derived_right_pos() == pytest.approx((0.10, 0.0, 0.0))


def test_rig_with_calibration_moves_intrinsics_and_the_derived_right_eye(
    registered_synthetic_model,
):
    """The whole point of the mechanism: one keyword moves K *and* the stereo geometry."""
    rig = _rig(SYNTHETIC_MODEL, SYNTHETIC_UNIT)

    assert rig.calibrated_model().fx == 11.0
    assert rig.baseline_m() == 0.20
    assert rig.derived_right_pos() == pytest.approx((0.20, 0.0, 0.0))
    assert rig.calibrated_model().scaled(0.5).fx == 5.5


def test_right_eye_override_is_preserved_only_within_its_declared_tolerance(
    registered_synthetic_model,
):
    derived_x = SYNTHETIC_MODEL.baseline_m
    accepted_pos = (derived_x + 5.0e-10, 0.0, 0.0)
    accepted = CameraRig(
        model=SYNTHETIC_MODEL.name,
        left=CameraMount(
            obs_key="robot_pov_left_cam",
            prim_name="left_cam",
            parent_link="link",
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        right=StereoRightEye(
            obs_key="robot_pov_right_cam",
            prim_name="right_cam",
            pos_override=accepted_pos,
            override_tolerance_m=1.0e-9,
            note="Synthetic rounding residual.",
        ),
    )
    assert accepted.right_mount().pos == accepted_pos
    assert accepted.right_override_residual_m() == pytest.approx(5.0e-10)

    with pytest.raises(ValueError, match="above the declared tolerance"):
        CameraRig(
            model=SYNTHETIC_MODEL.name,
            left=accepted.left,
            right=StereoRightEye(
                obs_key="robot_pov_right_cam",
                prim_name="right_cam",
                pos_override=(derived_x + 2.0e-9, 0.0, 0.0),
                override_tolerance_m=1.0e-9,
                note="Synthetic invalid residual.",
            ),
        )


def test_model_without_nominal_or_calibrated_baseline_cannot_define_stereo_geometry(
    registered_synthetic_model,
):
    nominal = CameraModel(
        name="nominal_without_baseline",
        width=1280,
        height=720,
        fx=736.0,
        fy=736.0,
        cx=640.0,
        cy=360.0,
    )
    from dexbench_arena.embodiments import camera_catalog

    camera_catalog.CAMERA_MODELS[nominal.name] = nominal
    with pytest.raises(ValueError, match="needs a baseline"):
        _rig(nominal, None)


@pytest.mark.skipif(
    importlib.util.find_spec("pxr") is None,
    reason="importing the embodiments needs the Isaac Sim runtime (pxr)",
)
def test_real_rigs_render_ideal_models_without_unit_overrides():
    from dexbench_arena.embodiments.rby1_wuji2.rby1_wuji2 import RBY1_WUJI2_HEAD_RIG

    assert RBY1_WUJI2_HEAD_RIG.calibration is None
    assert ZED_MINI_HD720.baseline_m is None
    assert RBY1_WUJI2_HEAD_RIG.right.baseline_m == 0.063
    assert RBY1_WUJI2_HEAD_RIG.baseline_m() == ZED_MINI_SPEC_BASELINE_M == 0.063
    assert RBY1_WUJI2_HEAD_RIG.right_mount().pos == pytest.approx(
        (0.026574731000910637, -0.047150079560382044, 0.05652762057118554)
    )


@pytest.mark.skipif(
    importlib.util.find_spec("pxr") is None,
    reason="importing the embodiments needs the Isaac Sim runtime (pxr)",
)
def test_camera_configs_follow_arena_runtime_selection_and_preserve_offsets():
    """Keep camera representation policy in Arena and data in declared fields."""
    from isaaclab.sensors import CameraCfg

    from dexbench_arena.embodiments.rby1_wuji2.rby1_wuji2 import Rby1Wuji2CameraCfg

    for cfg_type in (Rby1Wuji2CameraCfg,):
        declared = cfg_type()
        camera_names = [
            name
            for name in ("robot_pov_left_cam", "robot_pov_right_cam")
            if hasattr(declared, name)
        ]
        assert camera_names
        assert all(type(getattr(declared, name)) is CameraCfg for name in camera_names)

        expected_offset = (0.123, -0.456, 0.789)
        declared.robot_pov_left_cam.offset.pos = expected_offset

        declared.set_use_tiled_camera(True)
        tiled = declared.get_cfg()
        assert all(isinstance(getattr(tiled, name), CameraCfg) for name in camera_names)
        assert tiled.robot_pov_left_cam.offset.pos == expected_offset

        declared.set_use_tiled_camera(False)
        plain = declared.get_cfg()
        assert all(type(getattr(plain, name)) is CameraCfg for name in camera_names)
        assert plain.robot_pov_left_cam.offset.pos == expected_offset


@pytest.mark.skipif(
    importlib.util.find_spec("pxr") is None,
    reason="importing the environment base needs the Isaac Sim runtime (pxr)",
)
def test_production_camera_parallelism_tracks_environment_count():
    from argparse import Namespace

    from dexbench_arena.environments.manipulation.scene_environment_base import (
        _set_camera_parallelism,
    )

    class RecordingCameraCfg:
        selected: bool | None = None

        def set_use_tiled_camera(self, selected: bool) -> None:
            self.selected = selected

    class RecordingEmbodiment:
        camera_config = RecordingCameraCfg()

    embodiment = RecordingEmbodiment()
    _set_camera_parallelism(embodiment, Namespace(num_envs=1))
    assert embodiment.camera_config.selected is False

    _set_camera_parallelism(embodiment, Namespace(num_envs=8))
    assert embodiment.camera_config.selected is True

    _set_camera_parallelism(embodiment, Namespace(num_envs=None))
    assert embodiment.camera_config.selected is False


def test_resolution_scale_preserves_rays_and_stereo_mounts():
    from dataclasses import replace

    rig = replace(
        _rig(replace(ZED_MINI_HD720, baseline_m=ZED_MINI_SPEC_BASELINE_M), None),
        default_scale=0.25,
    )
    low, high = rig.resolved_model(), rig.resolved_model(1.0)
    assert (low.width, low.height) == (320, 180)
    assert (high.width, high.height) == (1280, 720)
    for model in (low, high):
        assert model.fx / model.width == pytest.approx(high.fx / high.width)
        assert model.fy / model.height == pytest.approx(high.fy / high.height)
        assert model.cx / model.width == high.cx / high.width
        assert model.cy / model.height == high.cy / high.height
        assert model.baseline_m == high.baseline_m
    mounts = rig.mounts()
    assert rig.resolved_model(0.25) == low
    assert rig.mounts() == mounts


@pytest.mark.parametrize('scale', [0, -1, float('inf'), float('nan'), 0.00001, 0.333])
def test_resolution_scale_refuses_invalid_rasters(scale):
    with pytest.raises(ValueError, match='Camera scale'):
        ZED_MINI_HD720.scaled(scale)
