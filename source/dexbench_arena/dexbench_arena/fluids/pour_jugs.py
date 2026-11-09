# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""The jugs the H2 pour collects with: scene instance, cavity profile, cavity axis and pour lip; and the target
goblet's cavity profile."""

from dataclasses import dataclass
import numpy as np

from dexbench_arena import ASSETS_DIR

IDEALISK_AXIS_XY = (-0.022, 0.0)


@dataclass(frozen=True)
class PourJug:
    pour_jug: str
    """The environment's ``--pour_jug`` choice."""
    scene_name: str
    """The jug's instance name in the scene and in the recordings (``rigid_object/<scene_name>``)."""
    profile: str
    """The cavity profile, relative to ``assets/``."""

    def cavity(self) -> tuple[np.ndarray, np.ndarray, float]:
        """(cavity axis, pour lip) in the jug frame, and the rim height."""
        with np.load(ASSETS_DIR / self.profile) as prof:
            axis_xy = prof["axis_xy"] if "axis_xy" in prof.files else np.array(IDEALISK_AXIS_XY)
            rim_r, rim_z = float(prof["r_inner"][-1]), float(prof["z"][-1])
        axis = np.array([float(axis_xy[0]), float(axis_xy[1]), 0.0])
        return axis, axis + np.array([-rim_r, 0.0, rim_z]), rim_z


JUGS = {
    "d00527_x15": PourJug("d00527_x15", "pour_jug", "object_library/milk_pitcher_d00527_x15/usd/milk_pitcher_d00527_x15_profile.npz"),
    "idealisk": PourJug("idealisk", "idealisk_jug",
                        "object_library/steel_jug_1l/usd/idealisk_profile.npz"),
    "steel_water_jug_tall": PourJug("steel_water_jug_tall", "pour_jug",
                                    "object_library/steel_water_jug_tall/usd/steel_water_jug_tall_profile.npz"),
    "steel_pitcher_large": PourJug("steel_pitcher_large", "pour_jug",
                                   "object_library/steel_pitcher_large/usd/steel_pitcher_large_profile.npz"),
    "steel_pitcher_footed": PourJug("steel_pitcher_footed", "pour_jug",
                                    "object_library/steel_pitcher_footed/usd/steel_pitcher_footed_profile.npz"),
}


GOBLET_PROFILE = "object_library/prop_goblet_30cl/usd/prop_goblet_30cl_profile.npz"
"""The target goblet's bowl cavity (``z``, ``r_inner``, ``floor_z``, ``rim_z``), relative to ``assets/``: beside its USD,
as each jug's profile is, declared as the package's ``fluid_profile``. ``goblet_cavity.extract`` measured it from the
visual bowl mesh (a maintainers' tool measures it again); the water builds the goblet's collider
from it and keeps its rendered surface inside the glass."""


def jug(name: str) -> PourJug:
    if name not in JUGS:
        raise ValueError(f"unknown pour jug {name!r}; one of {sorted(JUGS)}")
    return JUGS[name]
