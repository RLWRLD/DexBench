# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Make the inertia of selected Newton bodies isotropic, picked by body label."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from dexbench_arena.environments.manipulation.newton_build_scope import register_build_cleanup


logger = logging.getLogger(__name__)

ANISOTROPY_WARN_RATIO = 20.0


@dataclass(frozen=True)
class IsotropicInertiaRequest:
    name: str
    label_tokens: tuple[str, ...]


@dataclass(frozen=True)
class IsotropicInertiaResult:
    request: str
    index: int
    label: str
    principal: tuple[float, float, float]
    value: float


_REQUESTS: dict[str, IsotropicInertiaRequest] = {}
_HANDLE = None
_APPLIED: list[IsotropicInertiaResult] = []


def make_inertia_isotropic(builder, label_tokens, request_name: str = "") -> list[IsotropicInertiaResult]:
    """Set every massive body labelled with any of ``label_tokens`` to its largest principal moment on every axis."""

    import numpy as np
    import warp as wp

    tokens = tuple(label_tokens)
    results = []
    for index, label in enumerate(builder.body_label):
        if not label or not any(token in label for token in tokens):
            continue
        if float(builder.body_mass[index]) <= 0.0:
            continue
        inertia = np.array(builder.body_inertia[index], dtype=np.float64).reshape(3, 3)
        principal = np.linalg.eigvalsh(0.5 * (inertia + inertia.T))
        value = float(principal[-1])
        if not value > 0.0:
            raise ValueError(f"isotropic inertia for {label!r}: principal moments {principal.tolist()} give {value}")
        if principal[0] > 0.0 and principal[-1] / principal[0] > ANISOTROPY_WARN_RATIO:
            logger.warning(
                "Isotropic inertia %r: %s principal moments %s span %.1fx; every axis set to the largest, %.4g.",
                request_name,
                label,
                principal.tolist(),
                principal[-1] / principal[0],
                value,
            )
        builder.body_inertia[index] = wp.mat33(*(np.eye(3) * value).flatten().tolist())
        builder.body_inv_inertia[index] = wp.mat33(*(np.eye(3) / value).flatten().tolist())
        results.append(IsotropicInertiaResult(request_name, index, label, tuple(float(p) for p in principal), value))
    if not results:
        raise RuntimeError(
            f"Isotropic inertia {request_name!r} found no body with mass labelled like any of {list(tokens)}."
        )
    return results


def apply_requests(builder, requests) -> list[IsotropicInertiaResult]:
    """Apply several requests to one builder; two requests touching the same body is an error."""

    applied: list[IsotropicInertiaResult] = []
    owner: dict[int, str] = {}
    for request in requests:
        results = make_inertia_isotropic(builder, request.label_tokens, request.name)
        for result in results:
            if result.index in owner:
                raise RuntimeError(
                    f"Isotropic inertia requests {owner[result.index]!r} and {request.name!r} both match "
                    f"{result.label!r}."
                )
            owner[result.index] = request.name
        applied.extend(results)
    return applied


def _on_model_init(_event=None) -> None:
    from isaaclab_newton.physics.newton_manager import NewtonManager

    applied = apply_requests(NewtonManager._builder, _REQUESTS.values())
    _APPLIED[:] = applied
    for request in _REQUESTS.values():
        mine = [result for result in applied if result.request == request.name]
        logger.info(
            "Isotropic inertia %r: %d bodies (%r), e.g. %s principal %s -> %.4g.",
            request.name,
            len(mine),
            list(request.label_tokens),
            mine[0].label,
            [f"{p:.4g}" for p in mine[0].principal],
            mine[0].value,
        )


_BUILD_NAMES: set[str] = set()


def _remove_build_requests() -> None:
    """Drop the isotropic inertia requests the previous build installed."""
    for name in tuple(_BUILD_NAMES):
        _REQUESTS.pop(name, None)
    _BUILD_NAMES.clear()


def install_isotropic_inertia(name: str, label_tokens, mode: str = "max") -> None:
    """Make bodies whose labels contain any of ``label_tokens`` isotropic, from the next model on."""

    global _HANDLE
    if mode != "max":
        raise ValueError(f"isotropic inertia mode {mode!r} is not supported; 'max' is the only mode")
    from isaaclab.physics.physics_manager import PhysicsEvent
    from isaaclab_newton.physics.newton_manager import NewtonManager

    tokens = (label_tokens,) if isinstance(label_tokens, str) else tuple(label_tokens)
    _REQUESTS[name] = IsotropicInertiaRequest(name, tokens)
    _BUILD_NAMES.add(name)
    register_build_cleanup(_remove_build_requests)
    if _HANDLE is not None:
        _HANDLE.deregister()
    _HANDLE = NewtonManager.register_callback(
        _on_model_init,
        PhysicsEvent.MODEL_INIT,
        name="dexbench_isotropic_inertia",
        wrap_weak_ref=False,
    )


def applied_isotropic_inertia() -> list[IsotropicInertiaResult]:
    """What the last model build changed, one entry per body."""

    return list(_APPLIED)
