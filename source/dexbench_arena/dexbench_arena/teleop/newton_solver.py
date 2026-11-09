# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""MuJoCo-Warp solver options a Newton teleop profile may set."""

from __future__ import annotations

import math

SOLVER_OPTIONS = (
    "use_mujoco_contacts", "ccd_iterations", "enable_multiccd", "solver", "nconmax", "njmax", "num_substeps", "tolerance",
    "collision_decimation", "bvh_constructor_geometry", "ls_iterations", "iterations",
)
PHYSICS_OPTIONS = ("num_substeps", "collision_decimation", "bvh_constructor_geometry")
BVH_CONSTRUCTORS = ("lbvh", "sah", "cubql")


def validate_solver_options(options: dict, profile_name: str) -> None:
    unknown = sorted(set(options) - set(SOLVER_OPTIONS))
    if unknown:
        raise ValueError(f"Profile {profile_name}: unknown physics options {unknown}; choose from {SOLVER_OPTIONS}")
    for key in ("use_mujoco_contacts", "enable_multiccd"):
        if key in options and not isinstance(options[key], bool):
            raise ValueError(f"Profile {profile_name}: physics.{key} must be true or false")
    for key in ("ccd_iterations", "nconmax", "njmax", "num_substeps", "collision_decimation", "ls_iterations", "iterations"):
        if key in options:
            value = options[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"Profile {profile_name}: physics.{key} must be a positive integer")
    if "solver" in options and options["solver"] not in ("cg", "newton"):
        raise ValueError(f"Profile {profile_name}: physics.solver must be cg or newton")
    if "bvh_constructor_geometry" in options and options["bvh_constructor_geometry"] not in BVH_CONSTRUCTORS:
        raise ValueError(f"Profile {profile_name}: physics.bvh_constructor_geometry must be one of {BVH_CONSTRUCTORS}")
    if "tolerance" in options:
        value = options["tolerance"]
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Profile {profile_name}: physics.tolerance must be a positive number")


def mjwarp_solver_cfg(physics):
    """The MJWarp solver config of a Newton physics config: the solver itself, or the rigid half of a coupled
    MJWarp + deformable solver (``rigid_solver_cfg``, e.g. the polybag's MJWarp + VBD)."""
    solver = physics.solver_cfg
    return getattr(solver, "rigid_solver_cfg", solver)


def apply_solver_options(physics, options: dict) -> dict:
    """Apply a profile's ``physics`` block to a Newton MJWarp physics config; returns what was set."""
    validate_solver_options(options, "runtime")
    coupled = hasattr(physics.solver_cfg, "rigid_solver_cfg")
    solver = mjwarp_solver_cfg(physics)
    missing = [key for key in options if not hasattr(physics if key in PHYSICS_OPTIONS else solver, key)]
    if missing:
        raise ValueError(f"{type(solver).__name__} has no {missing}: these physics options would not take effect")
    num_substeps = options.get("num_substeps", getattr(physics, "num_substeps", None))
    decimation = options.get("collision_decimation", getattr(physics, "collision_decimation", 0))
    if decimation > 0 and num_substeps is not None and decimation >= num_substeps:
        raise ValueError(
            f"Newton collision_decimation {decimation} must be below num_substeps {num_substeps} "
            "(the environment's newton_solver_options, then the run's --newton_num_substeps)"
        )
    for key, value in options.items():
        setattr(physics if key in PHYSICS_OPTIONS else solver, key, value)
    if options.get("use_mujoco_contacts") and not coupled:
        physics.collision_cfg = None
    return {key: getattr(physics if key in PHYSICS_OPTIONS else solver, key) for key in options}


NEWTON_SOLVER_MIN_NEWTON = "1.8.0.dev0"


def refusing_newton_solver_before_newton_main(previous, who: str):
    """An env_cfg_callback that runs ``previous``, then refuses MJWarp's newton solver on a Newton older than main."""

    def callback(env_cfg):
        if previous is not None:
            env_cfg = previous(env_cfg) or env_cfg
        import newton
        from packaging.version import Version

        solver = getattr(mjwarp_solver_cfg(env_cfg.sim.physics), "solver", None)
        if solver == "newton" and Version(newton.__version__) < Version(NEWTON_SOLVER_MIN_NEWTON):
            raise ValueError(
                f"{who} runs MJWarp's newton solver, which needs Newton main ({NEWTON_SOLVER_MIN_NEWTON} or later); "
                f"this is Newton {newton.__version__}. Run in the image docker/build.py builds (Newton 646cece4), "
                "or with --newton_solver cg"
            )
        return env_cfg

    return callback
