# SPDX-License-Identifier: Apache-2.0
"""MJWarp stepped every few of the bag's substeps: what the substeps between carry (Warp and Newton only)."""

from __future__ import annotations

import warp as wp
from newton import Model


@wp.kernel
def interpolate_body_q(start: wp.array(dtype=wp.transform), end: wp.array(dtype=wp.transform), t: float,
                       out: wp.array(dtype=wp.transform)):
    """``out = start + t (end - start)`` per body: positions linearly, rotations by slerp."""
    i = wp.tid()
    a = start[i]
    b = end[i]
    out[i] = wp.transform(wp.lerp(wp.transform_get_translation(a), wp.transform_get_translation(b), t),
                          wp.quat_slerp(wp.transform_get_rotation(a), wp.transform_get_rotation(b), t))


def rigid_arrays(state) -> dict:
    """The arrays MJWarp writes into a state, by name: the body and joint state, and namespaced
    attributes (``mujoco.qfrc_actuator``). Not the particles (VBD's) and not ``body_f`` (an input)."""
    arrays = {}
    for name, value in vars(state).items():
        if isinstance(value, wp.array) and name.startswith(("body_", "joint_")) and name != "body_f":
            arrays[name] = value
        elif isinstance(value, Model.AttributeNamespace):
            arrays.update({f"{name}.{k}": v for k, v in vars(value).items() if isinstance(v, wp.array)})
    return arrays
