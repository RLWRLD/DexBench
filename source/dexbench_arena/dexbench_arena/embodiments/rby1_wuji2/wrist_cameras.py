# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""GA palm-camera rig matching the versioned Wuji2 SimReady attachment."""

from dexbench_arena.embodiments.camera_catalog import (
    ZED_X_ONE_S_WIDE,
    CameraMount,
    CameraRig,
)

PALM_CAMERA_RIGS = tuple(
    CameraRig(
        model=ZED_X_ONE_S_WIDE.name,
        left=CameraMount(
            obs_key=f"wrist_{side}_cam", prim_name="rgb",
            parent_link=f"link_{side}_arm_6/wrist_camera",
            pos=pos, rot=rot,
        ),
        clipping_range=(0.002, 100.0),
        update_latest_camera_pose=True,
        default_scale=0.5,
        antialiasing_mode="DLAA",
    )
    for side, pos, rot in (
        ("left", (-0.005, -0.065, 0.059),
         (0.998909613036363, -0.02790495766417661, -0.029241530230314918, 0.02336303129076632)),
        ("right", (-0.005, 0.065, 0.059),
         (0.02790495766417666, -0.9989096130363632, -0.023363031290766365, 0.029241530230314918)),
    )
)
