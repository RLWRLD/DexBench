# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""One episode lifetime on Newton: keep the model, sync the saved state."""

from numbers import Integral


def synchronize_newton_episode_reset(env_ids=None) -> bool:
    """Copy Newton's joint state into MuJoCo after an episode reset; do not reset gains or history."""
    import warp as wp
    from isaaclab_newton.physics import NewtonManager
    from newton.solvers import SolverMuJoCo

    solver = NewtonManager._solver
    if not isinstance(solver, SolverMuJoCo) and isinstance(getattr(solver, "rigid_solver", None), SolverMuJoCo):
        solver = solver.rigid_solver
    if not isinstance(solver, SolverMuJoCo):
        manager = getattr(getattr(NewtonManager, "_sim", None), "physics_manager", None)
        solver = getattr(manager, "_rigid_solver", None)
    if not isinstance(solver, SolverMuJoCo):
        return False
    model = NewtonManager.get_model()
    world_mask = None
    if env_ids is not None:
        selected = getattr(env_ids, "warp", env_ids)
        if isinstance(selected, wp.array):
            selected = selected.numpy().tolist()
        elif hasattr(selected, "detach"):
            selected = selected.detach().cpu().tolist()
        else:
            selected = list(selected)
        if any(isinstance(i, bool) or not isinstance(i, Integral) or not 0 <= i < model.world_count
               for i in selected):
            raise ValueError("Newton episode reset requires valid environment indices")
        if not selected or (solver.use_mujoco_cpu and 0 not in selected):
            return False
        mask = [False] * (model.world_count + 1)
        for index in selected:
            mask[index] = True
        world_mask = wp.array(mask, dtype=wp.bool, device=model.device)

    state = NewtonManager.get_state_0()
    data = solver.mj_data if solver.use_mujoco_cpu else solver.mjw_data
    with wp.ScopedDevice(model.device):
        solver._update_mjc_data(data, model, state, world_mask=world_mask)
    return True


def uses_newton(env_cfg) -> bool:
    """Whether an environment cfg runs on the Newton backend."""
    from isaaclab_newton.physics.newton_manager_cfg import NewtonCfg

    return isinstance(env_cfg.sim.physics, NewtonCfg)


def reset_sim_for_episode(env, requested: bool | None = None) -> bool:
    """Hard-reset the simulation before an episode only when the caller asks for it."""
    if requested:
        env.sim.reset()
    return bool(requested)


class SyncsNewtonSavedState:
    """Newton env mixin: copy the reset state into MuJoCo after every episode reset."""

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        synchronize_newton_episode_reset(env_ids)

    def reset_to(self, state, env_ids=None, seed=None, is_relative=False):
        for name in self.action_manager.active_terms:
            guard = getattr(self.action_manager.get_term(name), "validate_scene_state_restore", None)
            if guard is not None:
                guard(state, env_ids=env_ids, is_relative=is_relative)
        result = super().reset_to(state, env_ids, seed=seed, is_relative=is_relative)
        synchronize_newton_episode_reset(env_ids)
        return result


def newton_entry_point(entry_point):
    """The gym entry point of a Newton environment: ``entry_point``'s class with the saved-state sync."""
    from gymnasium.envs.registration import load_env_creator

    base = load_env_creator(entry_point) if isinstance(entry_point, str) else entry_point
    if issubclass(base, SyncsNewtonSavedState):
        return base
    return type(base.__name__, (SyncsNewtonSavedState, base), {"__module__": __name__, "__doc__": base.__doc__})
