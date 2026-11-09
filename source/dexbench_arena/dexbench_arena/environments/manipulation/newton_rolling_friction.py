# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Give selected Newton shapes rolling friction that MuJoCo applies, picked by shape label."""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass

from dexbench_arena.environments.manipulation.newton_build_scope import register_build_cleanup

logger = logging.getLogger(__name__)

COLLIDE_SHAPES = 1 << 1
CONDIM_ATTRIBUTE = "mujoco:condim"
ROLLING_CONDIM = 6


@dataclass(frozen=True)
class RollingFrictionRequest:
    name: str
    shape_pattern: str
    mu_rolling: float


@dataclass(frozen=True)
class RollingFrictionResult:
    request: str
    index: int
    label: str
    previous: float
    mu_rolling: float


_REQUESTS: dict[str, RollingFrictionRequest] = {}
_HANDLE = None


def _check_mu_rolling(mu_rolling: float) -> float:
    mu_rolling = float(mu_rolling)
    if not math.isfinite(mu_rolling) or mu_rolling <= 0.0:
        raise ValueError(f"rolling friction must be finite and positive, got {mu_rolling!r}")
    return mu_rolling


def set_rolling_friction(
    builder, shape_pattern: str, mu_rolling: float, request_name: str = ""
) -> list[RollingFrictionResult]:
    """Give every colliding shape whose label matches ``shape_pattern`` ``mu_rolling`` and condim 6."""

    mu_rolling = _check_mu_rolling(mu_rolling)
    condim = builder.custom_attributes.get(CONDIM_ATTRIBUTE)
    if condim is None:
        raise RuntimeError(
            f"Rolling friction {request_name!r}: the builder has no {CONDIM_ATTRIBUTE!r} attribute "
            "(SolverMuJoCo.register_custom_attributes was not called), so MuJoCo would ignore it."
        )
    if condim.values is None:
        condim.values = {}
    pattern = re.compile(shape_pattern)
    results = []
    for index, label in enumerate(builder.shape_label):
        if not (int(builder.shape_flags[index]) & COLLIDE_SHAPES) or not pattern.search(label or ""):
            continue
        previous = float(builder.shape_material_mu_rolling[index])
        builder.shape_material_mu_rolling[index] = mu_rolling
        condim.values[index] = ROLLING_CONDIM
        results.append(RollingFrictionResult(request_name, index, label, previous, mu_rolling))
    if not results:
        raise RuntimeError(f"Rolling friction {request_name!r} found no colliding shape matching {shape_pattern!r}.")
    return results


def apply_requests(builder, requests) -> list[RollingFrictionResult]:
    """Apply several requests to one builder; two requests touching the same shape is an error."""

    applied: list[RollingFrictionResult] = []
    owner: dict[int, str] = {}
    for request in requests:
        results = set_rolling_friction(builder, request.shape_pattern, request.mu_rolling, request.name)
        for result in results:
            if result.index in owner:
                raise RuntimeError(
                    f"Rolling friction requests {owner[result.index]!r} and {request.name!r} both match "
                    f"{result.label!r}."
                )
            owner[result.index] = request.name
        applied.extend(results)
    return applied


def _on_model_init(_event=None) -> None:
    from isaaclab_newton.physics.newton_manager import NewtonManager

    applied = apply_requests(NewtonManager._builder, _REQUESTS.values())
    for request in _REQUESTS.values():
        mine = [result for result in applied if result.request == request.name]
        logger.info(
            "Rolling friction %r: %d shapes matching %r -> %g m, condim %d (was %s m).",
            request.name,
            len(mine),
            request.shape_pattern,
            request.mu_rolling,
            ROLLING_CONDIM,
            sorted({result.previous for result in mine}),
        )


_BUILD_NAMES: set[str] = set()


def _remove_build_requests() -> None:
    """Drop the rolling friction requests the previous build installed."""
    for name in tuple(_BUILD_NAMES):
        _REQUESTS.pop(name, None)
    _BUILD_NAMES.clear()


def install_rolling_friction(name: str, shape_pattern: str, mu_rolling: float) -> None:
    """Give colliding shapes matching ``shape_pattern`` this rolling friction [m] and condim 6, from the next model on."""

    global _HANDLE
    from isaaclab.physics.physics_manager import PhysicsEvent
    from isaaclab_newton.physics.newton_manager import NewtonManager

    re.compile(shape_pattern)
    _REQUESTS[name] = RollingFrictionRequest(name, shape_pattern, _check_mu_rolling(mu_rolling))
    _BUILD_NAMES.add(name)
    register_build_cleanup(_remove_build_requests)
    if _HANDLE is not None:
        _HANDLE.deregister()
    _HANDLE = NewtonManager.register_callback(
        _on_model_init,
        PhysicsEvent.MODEL_INIT,
        name="dexbench_rolling_friction",
        wrap_weak_ref=False,
    )
