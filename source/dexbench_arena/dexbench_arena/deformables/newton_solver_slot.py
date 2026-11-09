# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Every use DexBench's paper sidecar makes of ``NewtonManager``'s private state, in one place."""

_PRIVATE = ("_solver", "_solver_dt", "_graph", "_capture_or_defer_graph", "_graph_capture_pending")
_CAPTURE_METHOD = None


def _manager():
    from isaaclab_newton.physics import NewtonManager

    _require(NewtonManager)
    return NewtonManager


def _require(manager) -> None:
    missing = [name for name in _PRIVATE if not hasattr(manager, name)]
    if missing:
        raise RuntimeError(f"NewtonManager has no {missing}: the paper coupler's private hooks do not fit "
                           "this Isaac Lab; port deformables/newton_solver_slot.py to the new pin")


def defer_initial_graph_capture() -> None:
    """Hold the rigid step's CUDA-graph capture until :func:`install_solver` or :func:`capture_without_coupler`."""
    global _CAPTURE_METHOD
    manager = _manager()
    if _CAPTURE_METHOD is not None:
        return
    _CAPTURE_METHOD = manager.__dict__["_capture_or_defer_graph"]

    def defer(cls):
        manager._graph = None
        manager._graph_capture_pending = False

    manager._capture_or_defer_graph = classmethod(defer)


def _restore_capture() -> bool:
    global _CAPTURE_METHOD
    manager = _manager()
    was_deferred = _CAPTURE_METHOD is not None
    if was_deferred:
        manager._capture_or_defer_graph = _CAPTURE_METHOD
        _CAPTURE_METHOD = None
    return was_deferred


def main_mujoco_solver():
    """The main Newton MuJoCo solver, or None (PhysX, or Newton not initialised)."""

    try:
        import newton
    except ImportError:
        return None
    try:
        manager = _manager()
    except ImportError:
        return None
    rigid = manager._solver
    return rigid if isinstance(rigid, newton.solvers.SolverMuJoCo) else None


def solver_and_dt():
    """The manager's solver and its step [s]: what a coupler wraps."""

    manager = _manager()
    return manager._solver, float(manager._solver_dt)


def install_solver(coupled) -> None:
    """Put ``coupled`` in the solver's place and capture one graph with it."""
    manager = _manager()
    coupled._previous_graph = manager._graph
    manager._solver = coupled
    manager._graph = None
    _restore_capture()
    manager._capture_or_defer_graph()


def capture_without_coupler() -> None:
    """No coupler (the camera render's sidecar, ``hand_coupling`` off): capture the rigid step's graph as Isaac Lab
    would have."""

    if _restore_capture():
        _manager()._capture_or_defer_graph()
