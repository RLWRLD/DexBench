# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""What an embodiment class declares, looked up from the name a run selects."""

from __future__ import annotations

import argparse
from collections.abc import Mapping

from isaaclab_arena.assets.registries import AssetRegistry

from dexbench_arena.embodiments.common.collision_spheres import CollisionSphereModel
from dexbench_arena.mimic_plugin import register_datagen_embodiments


def register_dexbench_embodiments() -> None:
    """Import the DexBench embodiment modules so their ``@register_asset`` decorators run, and the data-generation
    ones (the E2 Mimic hybrid) through the Mimic plugin where the tree has them
    (``mimic_plugin.register_datagen_embodiments``)."""

    from dexbench_arena.assets import retargeter_library as _retargeter_library  # noqa: F401
    from dexbench_arena.embodiments.rby1_wuji2 import rby1_wuji2 as _rby1_wuji2  # noqa: F401

    register_datagen_embodiments()


def embodiment_class(name: str | None) -> type | None:
    """The registered embodiment class for ``name``; None when none is selected or registered."""

    if not name:
        return None
    register_dexbench_embodiments()
    registry = AssetRegistry()
    return registry.get_asset_by_name(name) if registry.is_registered(name) else None


def base_model(name: str | None) -> str | None:
    """The base the named embodiment stands on (e.g. ``RBY1M_BASE_MODEL``), or None."""

    return getattr(embodiment_class(name), "base_model", None)


def start_poses(name: str | None) -> Mapping[str, Mapping[str, float]]:
    """The named joint poses of the named embodiment; empty when it declares none."""

    return getattr(embodiment_class(name), "start_poses", None) or {}


def collision_spheres(name: str | None) -> CollisionSphereModel | None:
    """The cuRobo collision-sphere model of the named embodiment, or None."""

    return getattr(embodiment_class(name), "collision_spheres", None)


def apply_rerecord_camera_defaults(args_cli: argparse.Namespace) -> None:
    """Fill unset camera options from the selected embodiment after Kit launches."""
    if not args_cli.enable_cameras:
        return
    cls = embodiment_class(getattr(args_cli, "embodiment", None))
    for name, value in getattr(cls, "rerecord_camera_defaults", {}).items():
        if getattr(args_cli, name, None) is None:
            setattr(args_cli, name, value)


def ik_profile_for_source_origin(embodiment_cls: type, origin: str, evidence: str) -> str | None:
    """The IK profile ``embodiment_cls`` drives sources of ``origin`` with; None if it chooses none."""

    by_origin = getattr(embodiment_cls, "ik_profile_by_source_origin", None)
    if by_origin is None:
        return None
    if origin not in by_origin:
        profiles = ", ".join(sorted(set(by_origin.values())))
        raise ValueError(
            f"the source demos are {origin} ({evidence}), so {embodiment_cls.name}'s IK profile cannot be "
            f"chosen from how they were recorded; name it with --ik_profile ({profiles})"
        )
    return by_origin[origin]
