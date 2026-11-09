# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""The plain-MJWarp manager that drives :class:`~dexbench_arena.assets.newton_conveyor.BeltDriveCfg` belts."""

from __future__ import annotations

import dataclasses

import newton
from isaaclab.utils.configclass import configclass
from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import NewtonArenaMJWarpManager
from isaaclab_newton.physics import MJWarpSolverCfg

from dexbench_arena.assets.newton_conveyor import (
    BeltDriveCfg,
    BeltTreadmill,
    build_belt_treadmill,
    resolve_belt_textures,
    scroll_belt_textures,
)


class NewtonConveyorMJWarpManager(NewtonArenaMJWarpManager):
    """Arena's MJWarp manager (the one its newton preset selects), with ``ConveyorMJWarpSolverCfg.belts`` moving."""

    _belt_treadmill: BeltTreadmill | None = None
    _belt_cfgs: tuple = ()
    _belt_textures: list | None = None

    @classmethod
    def _build_solver(cls, model, solver_cfg: ConveyorMJWarpSolverCfg) -> None:
        super()._build_solver(model, solver_cfg)
        NewtonConveyorMJWarpManager._belt_cfgs = tuple(solver_cfg.belts)

    @classmethod
    def _initialize_contacts(cls) -> None:
        super()._initialize_contacts()
        treadmill, friction = build_belt_treadmill(cls._model, cls._state_0, cls._belt_cfgs)
        NewtonConveyorMJWarpManager._belt_treadmill = treadmill
        if friction:
            cls.add_model_change(newton.ModelFlags.SHAPE_PROPERTIES)

    @classmethod
    def _step_solver(cls, state_0, state_1, control, contacts, substep_dt: float) -> None:
        if cls._belt_treadmill is not None:
            cls._belt_treadmill.apply(state_0)
        super()._step_solver(state_0, state_1, control, contacts, substep_dt)

    @classmethod
    def pre_render(cls) -> None:
        super().pre_render()
        if not cls._belt_cfgs:
            return
        if cls._belt_textures is None:
            NewtonConveyorMJWarpManager._belt_textures = resolve_belt_textures(cls._belt_cfgs)
        scroll_belt_textures(cls._belt_textures, cls.get_simulation_time())

    @classmethod
    def _solver_specific_clear(cls) -> None:
        super()._solver_specific_clear()
        NewtonConveyorMJWarpManager._belt_treadmill = None
        NewtonConveyorMJWarpManager._belt_cfgs = ()
        NewtonConveyorMJWarpManager._belt_textures = None


@configclass
class ConveyorMJWarpSolverCfg(MJWarpSolverCfg):
    """MJWarp with conveyor belts: ``NewtonConveyorMJWarpManager`` drives ``belts``."""

    class_type: type = NewtonConveyorMJWarpManager

    belts: tuple[BeltDriveCfg, ...] = ()
    """The scene's belts; empty: none moves."""


def drive_conveyor_belts(physics, belts):
    """``physics`` (a ``NewtonCfg`` on MJWarp) with its solver cfg made a ``ConveyorMJWarpSolverCfg`` that keeps every
    MJWarp setting and drives ``belts``; for an environment's ``apply_newton_sim_settings``. TypeError for another
    solver (whose manager takes ``belts`` itself)."""

    solver = physics.solver_cfg
    if not isinstance(solver, MJWarpSolverCfg):
        raise TypeError(f"conveyor belts on Newton need MJWarp's solver cfg, not {type(solver).__name__}")
    kept = {f.name: getattr(solver, f.name) for f in dataclasses.fields(solver)
            if f.init and f.name not in ("class_type", "belts")}
    return physics.replace(class_type=None, solver_cfg=ConveyorMJWarpSolverCfg(**kept, belts=tuple(belts)))
