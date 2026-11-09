# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Give selected Newton shapes their own contact stiffness, picked by the shape's or its body's label."""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass

from dexbench_arena.environments.manipulation.newton_build_scope import register_build_cleanup

logger = logging.getLogger(__name__)

COLLIDE_SHAPES = 1 << 1
SOLMIX_ATTRIBUTE = "mujoco:geom_solmix"
MATCH_CHOICES = ("shape", "body")


@dataclass(frozen=True)
class ShapeContactRequest:
    name: str
    pattern: str
    ke: float
    kd: float
    solmix: float
    match: str = "shape"
    """What ``pattern`` is searched in: ``"shape"``, the shape's own label, or ``"body"``, its body's label."""
    friction: float | None = None
    """Optional sliding friction override; None preserves the parsed material."""


@dataclass(frozen=True)
class ShapeContactResult:
    request: str
    index: int
    label: str
    ke: float
    kd: float
    solmix: float


_REQUESTS: dict[str, ShapeContactRequest] = {}
_HANDLE = None


def _check(request: ShapeContactRequest) -> ShapeContactRequest:
    for field in ("ke", "kd", "solmix"):
        value = float(getattr(request, field))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"shape contact {request.name!r}: {field} must be finite and positive, got {value!r}")
    if request.match not in MATCH_CHOICES:
        raise ValueError(f"shape contact {request.name!r}: match must be one of {MATCH_CHOICES}, got {request.match!r}")
    if request.friction is not None and (not math.isfinite(request.friction) or request.friction < 0):
        raise ValueError("shape contact friction must be finite and nonnegative")
    re.compile(request.pattern)
    return request


def set_shape_contact(builder, request: ShapeContactRequest) -> list[ShapeContactResult]:
    """Apply ``request`` to every colliding shape whose label (or body's label) matches; raises when none does."""

    _check(request)
    pattern = re.compile(request.pattern)
    solmix = builder.custom_attributes.get(SOLMIX_ATTRIBUTE)
    if solmix is None:
        raise RuntimeError(f"the Newton builder has no {SOLMIX_ATTRIBUTE!r}; is the MuJoCo solver selected?")
    if solmix.values is None:
        solmix.values = {}
    results = []
    for index, label in enumerate(builder.shape_label):
        if not (int(builder.shape_flags[index]) & COLLIDE_SHAPES):
            continue
        if request.match == "body":
            body = builder.shape_body[index]
            searched = builder.body_label[body] if body >= 0 else ""
        else:
            searched = label
        if not pattern.search(searched or ""):
            continue
        builder.shape_material_ke[index] = request.ke
        builder.shape_material_kd[index] = request.kd
        if request.friction is not None:
            builder.shape_material_mu[index] = request.friction
        solmix.values[index] = request.solmix
        label = label or searched
        results.append(ShapeContactResult(request.name, index, label, request.ke, request.kd, request.solmix))
    if not results:
        raise RuntimeError(
            f"Shape contact {request.name!r} found no colliding shape whose {request.match} label "
            f"matches {request.pattern!r}."
        )
    return results


def _on_model_init(_event=None) -> None:
    from isaaclab_newton.physics.newton_manager import NewtonManager

    for request in _REQUESTS.values():
        results = set_shape_contact(NewtonManager._builder, request)
        logger.info("Shape contact %r: %d shapes -> ke %g, kd %g, solmix %g.", request.name, len(results),
                    request.ke, request.kd, request.solmix)


_BUILD_NAMES: set[str] = set()


def _remove_build_requests() -> None:
    """Drop the shape contact requests the previous build installed."""
    for name in tuple(_BUILD_NAMES):
        _REQUESTS.pop(name, None)
    _BUILD_NAMES.clear()


def install_shape_contact(request: ShapeContactRequest) -> None:
    """Apply ``request`` from the next model on; installing the same name again replaces it."""

    global _HANDLE
    from isaaclab.physics.physics_manager import PhysicsEvent
    from isaaclab_newton.physics.newton_manager import NewtonManager

    _REQUESTS[request.name] = _check(request)
    _BUILD_NAMES.add(request.name)
    register_build_cleanup(_remove_build_requests)
    if _HANDLE is not None:
        _HANDLE.deregister()
    _HANDLE = NewtonManager.register_callback(
        _on_model_init, PhysicsEvent.MODEL_INIT, name="dexbench_shape_contact", wrap_weak_ref=False
    )
