# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Task-selected hand contact materials, independent of the simulation runtime."""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, replace
from typing import Literal

FrictionCombineMode = Literal["average", "min", "multiply", "max"]


@dataclass(frozen=True)
class HandContactMaterialCfg:
    """Contact coefficients for a hand."""

    static_friction: float
    dynamic_friction: float
    friction_combine_mode: FrictionCombineMode = "max"
    enabled: bool = True

    def __post_init__(self) -> None:
        for name in ("static_friction", "dynamic_friction"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative, got {value!r}")
        if self.dynamic_friction > self.static_friction:
            raise ValueError("dynamic_friction must not exceed static_friction")
        if self.friction_combine_mode not in ("average", "min", "multiply", "max"):
            raise ValueError(f"Unsupported friction_combine_mode: {self.friction_combine_mode!r}")
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a bool")


WUJI_GRASP_CONTACT_MATERIAL = HandContactMaterialCfg(6.0, 5.0)


def validate_legacy_hand_friction_environment() -> None:
    """Reject the hand friction ENV variables before launching Kit, including empty values."""
    legacy_names = ("DEXBENCH_HAND_STATIC_FRICTION", "DEXBENCH_HAND_DYNAMIC_FRICTION")
    present = [name for name in legacy_names if name in os.environ]
    if present:
        raise ValueError(
            f"Hand friction ENV variables are set: {', '.join(present)}; DexBench does not read them. "
            "Unset them and use task hand_contact_material settings or "
            "--hand_static_friction / --hand_dynamic_friction for run overrides; "
            "use --disable_hand_contact_material to put the robot model's own contact material "
            "on the hand (Wuji v2: its MJCF's 1.0 / 1.0, max)."
        )


def resolve_hand_contact_material(
    task_material: HandContactMaterialCfg | None,
    embodiment_material: HandContactMaterialCfg,
    *,
    static_friction: float | None = None,
    dynamic_friction: float | None = None,
    friction_combine_mode: FrictionCombineMode | None = None,
    enabled: bool | None = None,
) -> tuple[HandContactMaterialCfg, str]:
    """Resolve explicit run fields over the task profile and embodiment fallback."""
    material = task_material if task_material is not None else embodiment_material
    overrides = {
        name: value
        for name, value in {
            "static_friction": static_friction,
            "dynamic_friction": dynamic_friction,
            "friction_combine_mode": friction_combine_mode,
            "enabled": enabled,
        }.items()
        if value is not None
    }
    base_source = "task" if task_material is not None else "embodiment"
    if overrides:
        try:
            return replace(material, **overrides), "override"
        except ValueError as error:
            resolved = {**asdict(material), **overrides}
            details = ", ".join(
                f"{name}={value!r} ({'override' if name in overrides else base_source})"
                for name, value in resolved.items()
            )
            raise ValueError(f"{error}. Resolved hand contact material: {details}") from error
    return material, base_source
