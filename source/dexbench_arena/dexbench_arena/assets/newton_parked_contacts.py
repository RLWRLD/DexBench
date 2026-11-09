# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Isaac Lab solver-factory extension installing a per-world parked-body filter."""

from isaaclab.utils import configclass
from isaaclab_newton.physics.mjwarp_manager_cfg import MJWarpSolverCfg
from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import NewtonArenaMJWarpManager

from dexbench_arena.assets.parked_contacts import ParkedBodyContactFilter


class NewtonParkedContactsMJWarpManager(NewtonArenaMJWarpManager):
    """Preserve Arena's passive-tendon handling while adding the contact callback."""

    @classmethod
    def _create_solver(cls, model, solver_cfg):
        solver = super()._create_solver(model, solver_cfg)
        solver.mjw_model.callback.contactfilter = ParkedBodyContactFilter(solver, solver_cfg.parked_body_offsets)
        return solver


@configclass
class ParkedContactsMJWarpSolverCfg(MJWarpSolverCfg):
    """Native MuJoCo-Warp with parked-body subtree expressions and initial-to-park offsets."""

    class_type: type = NewtonParkedContactsMJWarpManager
    parked_body_offsets: tuple[tuple[str, float, float, float], ...] = ()
