# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Which DB-H1 dish is which: the Isaac-free half of ``e4_hotel_fb_dishes``."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from dexbench_arena import ASSETS_DIR
from dexbench_arena.assets.scene_bundles import runtime_usd
from dexbench_arena.tasks.object_axis import ObjectAxis, TaskObject

DISH_PACKAGES_DIR = ASSETS_DIR / "object_library"

DISH_TABLE_PATH = Path(__file__).with_name("e4_hotel_fb_dishes.json")
_TABLE = json.loads(DISH_TABLE_PATH.read_text())

DISH_GEOMETRY: dict[str, dict] = _TABLE["dishes"]
DEFAULT_DISH_VARIANT: str = _TABLE["default_variant"]
DISH_CONTACT_OFFSET_M: float = _TABLE["contact_offset_m"]
DISH_COLLIDER_PARAMS: dict = _TABLE["vhacd"]


@dataclass(frozen=True)
class DishSpec:
    """One ``--plate_variant``: which asset spawns and the geometry that goes with it."""

    name: str
    asset_name: str
    usd_path: Path
    kind: str
    """``"plate"`` or ``"bowl"``: the dish class, and the noun in the task instruction."""
    diameter_m: float
    height_m: float
    nest_pitch_m: float

    @property
    def radius_m(self) -> float:
        return self.diameter_m / 2.0

    def scale_for(self, diameter_m: float) -> float:
        """Uniform scale that renders the dish at ``diameter_m`` across the rim (the GA-O knob)."""
        return float(diameter_m) / self.diameter_m

    def nest_pitch_for(self, diameter_m: float) -> float:
        """Settled nest pitch at ``diameter_m``; the pitch scales with the asset."""
        return self.nest_pitch_m * self.scale_for(diameter_m)


DISH_VARIANTS: dict[str, DishSpec] = {
    key: DishSpec(
        name=key,
        asset_name=f"dexbench_hotel_fb_e4_{key}",
        usd_path=DISH_PACKAGES_DIR / geometry["folder"] / "usd" / f"{geometry['folder']}.usd",
        kind=geometry["kind"],
        diameter_m=geometry["diameter_m"],
        height_m=geometry["height_m"],
        nest_pitch_m=geometry["nest_pitch_m"],
    )
    for key, geometry in DISH_GEOMETRY.items()
}


def dish_variant_names() -> tuple[str, ...]:
    """Every ``--plate_variant`` value, in table order (the default first)."""
    return tuple(DISH_GEOMETRY)


def dish_spec(name: str | None) -> DishSpec:
    """The spec for ``--plate_variant name`` (None = the default); refuses unknown names."""
    key = name or DEFAULT_DISH_VARIANT
    try:
        spec = DISH_VARIANTS[key]
    except KeyError:
        raise ValueError(f"--plate_variant {key!r} unknown; expected one of {dish_variant_names()}") from None
    if not Path(runtime_usd(spec.usd_path)).exists():
        raise FileNotFoundError(
            f"--plate_variant {key!r}: {spec.usd_path} is missing. The DB-H1 dishes live at "
            f"object_library/<dish>/usd/ in the pinned assets repository."
        )
    return spec


def dish_fit(task_object: TaskObject, cfg) -> dict:
    """What DB-H1 derives from its dish: the uniform scale ``plate_diameter_m`` asks for and the nest pitch at it."""
    spec = DISH_VARIANTS[task_object.key]
    diameter = float(getattr(cfg, "plate_diameter_m", None) or spec.diameter_m)
    return {
        "diameter_m": round(diameter, 6),
        "scale": round(spec.scale_for(diameter), 6),
        "nest_pitch_m": round(spec.nest_pitch_for(diameter), 6),
    }


DISH_OBJECT_AXIS = ObjectAxis(
    field="plate_variant",
    default=DEFAULT_DISH_VARIANT,
    objects=tuple(
        TaskObject(
            key=key,
            asset=DISH_VARIANTS[key].asset_name,
            split=geometry["split"],
            label=geometry["kind"],
            size_m=(geometry["diameter_m"], geometry["height_m"]),
            benchmark_asset_major_renames=((2, 3, f"assets/object_library/{geometry['folder']}/CHANGELOG.md"),),
        )
        for key, geometry in DISH_GEOMETRY.items()
    ),
    size_axes="rim diameter, height",
    fixture_profile={"plate_diameter_m": None},
    fit=dish_fit,
    unlabelled_default="godmiddag_deep_plate_23",
)

