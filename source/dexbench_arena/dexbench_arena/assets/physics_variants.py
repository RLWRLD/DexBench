# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Physics runtime variants of pooled props: a scene selects the engine it runs on every rigid object it spawns."""

from __future__ import annotations

import json
import os

from dexbench_arena.assets.scene_bundles import source_usd

NEWTON_VARIANT = "Newton"
"""The runtime variant set DexBench selects on a prop (its Newton payload)."""
PHYSX_VARIANT = "PhysX"
"""The packages' PhysX variant set: never selected, and dropped from a spawn config that selects it."""
NEWTON_COLLIDERS_VARIANT = "NewtonColliders"
"""A package's choice among its Newton colliders, nested under its ``Newton`` variant's ``Enabled``."""


def make_static(prim) -> None:
    """Remove ``prim``'s rigid body and mass schemas, so its colliders are static on PhysX and Newton alike."""

    from pxr import UsdPhysics

    for schema in (UsdPhysics.RigidBodyAPI, UsdPhysics.MassAPI):
        if prim.HasAPI(schema):
            prim.RemoveAPI(schema)


def _sidecar_asset_block(usd_path: str | os.PathLike[str]) -> dict:
    """The ``asset`` block of the ``*.meta.json`` sidecar beside ``usd_path``; empty when there is none."""
    usd = source_usd(usd_path)
    sidecar = usd.parent / f"{usd.stem}.meta.json"
    if not sidecar.is_file():
        return {}
    try:
        block = json.loads(sidecar.read_text()).get("asset", {})
    except (OSError, ValueError):
        return {}
    return block if isinstance(block, dict) else {}


def asset_package_version(usd_path: str | os.PathLike[str]) -> str | None:
    """The package version the sidecar beside ``usd_path`` declares (``asset.version``); None without one."""
    version = _sidecar_asset_block(usd_path).get("version")
    return str(version) if version is not None else None


def physics_variants_of(usd_path: str | os.PathLike[str]) -> dict[str, str]:
    """``asset.variants.physics`` of the sidecar beside ``usd_path``: variant set name -> payload; empty when none."""
    block = _sidecar_asset_block(usd_path)
    physics = (block.get("variants") or {}).get("physics") or {}
    return dict(physics) if isinstance(physics, dict) else {}


def newton_colliders_of(usd_path: str | os.PathLike[str]) -> dict[str, str]:
    """``asset.variants.newton_colliders`` of the sidecar beside ``usd_path``: option -> payload; empty when none."""
    block = _sidecar_asset_block(usd_path)
    choice = (block.get("variants") or {}).get("newton_colliders") or {}
    return dict(choice) if isinstance(choice, dict) else {}


def select_newton_colliders(spawn, option: str) -> None:
    """Select ``option`` of :data:`NEWTON_COLLIDERS_VARIANT` on a spawn config whose every USD offers it."""
    paths = spawn.usd_path if isinstance(spawn.usd_path, (list, tuple)) else [spawn.usd_path]
    missing = [str(p) for p in paths if option not in newton_colliders_of(p)]
    if missing:
        raise ValueError(f"Newton collider choice {option!r} is not offered by {missing} (asset.variants.newton_colliders)")
    current = getattr(spawn, "variants", None)
    selection = dict(current) if isinstance(current, dict) else {}
    selection[NEWTON_COLLIDERS_VARIANT] = option
    spawn.variants = selection


def physics_variants_enabled(args_cli) -> bool:
    """Whether the run selects the backend's variant (``--physics_variants``, on unless switched off)."""
    return getattr(args_cli, "physics_variants", True) is not False


def _select(spawn, variant: str) -> bool:
    """Select ``variant``'s ``Enabled`` option on a spawn config, when every USD it spawns carries the set."""
    paths = spawn.usd_path if isinstance(spawn.usd_path, (list, tuple)) else [spawn.usd_path]
    if not paths or not all(variant in physics_variants_of(p) for p in paths):
        return False
    current = getattr(spawn, "variants", None)
    selection = dict(current) if isinstance(current, dict) else {}
    others = [name for name in (NEWTON_VARIANT, PHYSX_VARIANT) if name != variant and name in selection]
    if selection.get(variant) == "Enabled" and not others and next(iter(selection)) == variant:
        return False
    for name in (*others, variant):
        selection.pop(name, None)
    spawn.variants = {variant: "Enabled", **selection}
    return True


def apply_physics_variants(scene) -> list[tuple[str, str]]:
    """Select the ``Newton`` variant on every rigid / articulated object of ``scene`` that carries it."""
    selected: list[tuple[str, str]] = []
    for name, asset in getattr(scene, "assets", {}).items():
        object_type = getattr(getattr(asset, "object_type", None), "name", "")
        if object_type not in ("RIGID", "ARTICULATION") and not getattr(asset, "physics_variant_on_base", False):
            continue
        spawn = getattr(getattr(asset, "object_cfg", None), "spawn", None)
        if spawn is None or not hasattr(spawn, "usd_path") or not hasattr(spawn, "variants"):
            continue
        if _select(spawn, NEWTON_VARIANT):
            selected.append((name, NEWTON_VARIANT))
    return selected


NEWTON_USD_PARSE_GUARD_ENV = "PXR_WORK_THREAD_LIMIT"


def apply_newton_usd_parse_guard() -> None:
    """Single-thread USD's work pool before Kit starts: every DexBench scene runs on Newton."""

    if NEWTON_USD_PARSE_GUARD_ENV not in os.environ:
        os.environ[NEWTON_USD_PARSE_GUARD_ENV] = "1"
        print(f"[dexbench] Newton backend: {NEWTON_USD_PARSE_GUARD_ENV}=1 (USD physics parse races on many-collider bodies)")
