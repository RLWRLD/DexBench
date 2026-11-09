# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Draw a fixture with another look and keep its physics: the spawner side of the retail_v1 look."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from isaaclab.sim.utils import clone as _clone_spawner
from dexbench_arena.assets.scene_bundles import runtime_usd

if TYPE_CHECKING:
    from pxr import Usd

LOOK_CHILD = "Look"
"""Name of the prim the look is referenced on, under the fixture."""


def _authored_weakest(attr):
    """The value the weakest layer (the fixture's package) authors on ``attr``."""
    for spec in reversed(attr.GetPropertyStack()):
        if spec.HasDefaultValue():
            return spec.default
    return None


def apply_fixture_look(prim: Usd.Prim, look_usd: str | None) -> dict[str, int | float]:
    """Stop drawing ``prim``'s meshes and lights and reference ``look_usd`` under it."""
    from pxr import Gf, Usd, UsdGeom, UsdLux

    lights, drawn = [], []
    for part in Usd.PrimRange(prim):
        if part.HasAPI(UsdLux.LightAPI):
            lights.append(part)
        elif part.IsA(UsdGeom.Gprim):
            imageable = UsdGeom.Imageable(part)
            if imageable.ComputePurpose() in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render) and (
                imageable.ComputeVisibility() != UsdGeom.Tokens.invisible
            ):
                drawn.append(imageable)
    for light in lights:
        light.SetActive(False)
    for imageable in drawn:
        imageable.GetPurposeAttr().Set(UsdGeom.Tokens.proxy)
    report: dict[str, int | float] = {"hidden": len(drawn), "lights_off": len(lights)}
    if look_usd is None:
        return report

    stage = prim.GetStage()
    look = stage.DefinePrim(prim.GetPath().AppendChild(LOOK_CHILD), "Xform")
    assert look.GetReferences().AddReference(runtime_usd(look_usd)), f"could not reference {look_usd} at {look.GetPath()}"

    look_door, fixture_door = look.GetChild("Door"), prim.GetChild("Door")
    if look_door and fixture_door:
        angle = fixture_door.GetAttribute("xformOp:rotateXYZ").Get()
        look_door.GetAttribute("xformOp:rotateXYZ").Set(Gf.Vec3f(angle))
        report["door_deg"] = float(angle[2])

    look_shelf3 = look.GetChild("Shelf3")
    fixture_shelf3 = prim.GetPrimAtPath("Geom/FrgFC_Shelf3")
    if look_shelf3 and fixture_shelf3:
        translate = fixture_shelf3.GetAttribute("xformOp:translate")
        nominal = _authored_weakest(translate)
        raise_m = float(translate.Get()[2] - nominal[2]) if nominal is not None else 0.0
        look_shelf3.GetAttribute("xformOp:translate").Set(Gf.Vec3d(0.0, 0.0, raise_m))
        report["shelf3_raise_m"] = round(raise_m, 6)
    return report


@functools.lru_cache(maxsize=None)
def fixture_look_spawner(spawn_func, look_usd: str | None):
    """``spawn_func`` (a cloned Isaac Lab spawner), followed by :func:`apply_fixture_look`."""
    inner = getattr(spawn_func, "__wrapped__", spawn_func)

    def spawn_fixture_with_look(prim_path, cfg, translation=None, orientation=None, **kwargs):
        prim = inner(prim_path, cfg, translation, orientation, **kwargs)
        report = apply_fixture_look(prim, look_usd)
        print(f"[dexbench] look {look_usd.rsplit('/', 1)[-1] if look_usd else '(hidden)'} on {prim.GetPath()}: {report}")
        return prim

    return _clone_spawner(spawn_fixture_with_look)
