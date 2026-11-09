# SPDX-License-Identifier: Apache-2.0
"""The XR anchor of the RBY1-M family: the operator faces the base link's heading."""

from __future__ import annotations

import numpy as np
import torch
from isaaclab.utils.math import quat_mul, yaw_quat
from isaaclab_teleop import XrCfg
from isaaclab_teleop.xr_cfg import XrAnchorRotationMode

from dexbench_arena.embodiments.rby1m.constants import RBY1M_XR_ANCHOR_POS, RBY1M_XR_ANCHOR_ROT


def rby1m_anchor_rotation(_previous_headpose: np.ndarray, anchor_prim_pose: np.ndarray) -> np.ndarray:
    """``RBY1M_XR_ANCHOR_ROT`` turned by the base link's current world yaw (xyzw)."""
    base = torch.as_tensor(np.asarray(anchor_prim_pose[3:7], dtype=np.float64))
    anchor = quat_mul(yaw_quat(base), torch.tensor(RBY1M_XR_ANCHOR_ROT, dtype=torch.float64))
    return anchor.numpy()


def rby1m_xr_cfg(base_link_prim_path: str) -> XrCfg:
    """The family's ``XrCfg``: anchored to the base link, facing its heading, at a fixed height."""
    return XrCfg(
        anchor_pos=RBY1M_XR_ANCHOR_POS,
        anchor_rot=RBY1M_XR_ANCHOR_ROT,
        anchor_prim_path=base_link_prim_path,
        anchor_rotation_mode=XrAnchorRotationMode.CUSTOM,
        anchor_rotation_custom_func=rby1m_anchor_rotation,
        fixed_anchor_height=True,
    )
