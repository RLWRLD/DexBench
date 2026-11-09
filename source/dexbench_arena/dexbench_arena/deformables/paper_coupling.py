# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""One-way proxy coupling between Arena's MuJoCo solver and the VBD paper world."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import warp as wp

import newton

from .newton_solver_slot import install_solver, solver_and_dt


@dataclass
class ProxyCouplingCfg:
    """Knobs of the proxy pass; names follow ``SolverCoupledProxy.Proxy``."""

    mass_scale: float = 1.0
    """Proxy body mass/inertia = this times the source body's. Lighter proxies
    yield more to the cloth within a substep; they are re-synced every substep
    either way."""


@wp.kernel
def _sync_proxy_kernel(
    local_to_main: wp.array(dtype=wp.int32),
    start: wp.array(dtype=wp.transform),
    end: wp.array(dtype=wp.transform),
    end_qd: wp.array(dtype=wp.spatial_vector),
    main_com: wp.array(dtype=wp.vec3),
    alpha: float,
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
):
    """Pose the proxy on the source's begin->end trajectory; carry the end velocity."""

    i = wp.tid()
    j = local_to_main[i]
    if j < 0:
        return
    p0 = wp.transform_get_translation(start[j])
    p1 = wp.transform_get_translation(end[j])
    r0 = wp.transform_get_rotation(start[j])
    r1 = wp.transform_get_rotation(end[j])
    body_q[i] = wp.transform(wp.lerp(p0, p1, alpha), wp.quat_slerp(r0, r1, alpha))
    w = wp.spatial_bottom(end_qd[j])
    v = wp.spatial_top(end_qd[j]) - wp.cross(w, wp.quat_rotate(r1, main_com[j]))
    body_qd[i] = wp.spatial_vector(v, w)


class PaperProxyCoupler:
    """One-way proxy pass between the main rigid solver and the paper VBD world."""

    def __init__(self, rigid_solver, stack, body_mapping: dict[str, int], rigid_dt: float,
                 frame_dt: float, cfg: ProxyCouplingCfg | None = None):
        self.cfg = cfg or ProxyCouplingCfg()
        self.rigid_solver = rigid_solver
        self.stack = stack
        self.model = rigid_solver.model
        self.vbd = stack.solver
        self.rigid_dt = float(rigid_dt)
        if self.vbd.integrate_with_external_rigid_solver:
            raise ValueError("Proxy coupling needs VBD to integrate the proxy bodies itself")
        if not body_mapping:
            raise ValueError("Proxy coupling needs at least one mirrored hand body")

        main = self.model
        paper = stack.model
        device = paper.device
        self.body_mapping = dict(body_mapping)
        """Mirror-body name -> main-solver body index (kept for telemetry/recording)."""
        local_to_main = np.full(paper.body_count, -1, dtype=np.int32)
        for mirror, j in body_mapping.items():
            local_to_main[stack._mirror_body_ids[mirror]] = j
        self.local_to_main = wp.array(local_to_main, dtype=wp.int32, device=device)

        src_mass = main.body_mass.numpy()
        src_inertia = main.body_inertia.numpy()
        mass = paper.body_mass.numpy()
        inv_mass = paper.body_inv_mass.numpy()
        inertia = paper.body_inertia.numpy()
        inv_inertia = paper.body_inv_inertia.numpy()
        for local, j in enumerate(local_to_main):
            if j < 0:
                continue
            m = float(src_mass[j]) * self.cfg.mass_scale
            I = np.asarray(src_inertia[j], dtype=np.float32) * self.cfg.mass_scale
            mass[local] = m
            inv_mass[local] = 1.0 / m if m > 0.0 else 0.0
            inertia[local] = I
            inv_inertia[local] = np.linalg.inv(I) if m > 0.0 and np.linalg.det(I) > 0.0 else 0.0
        paper.body_mass.assign(mass)
        paper.body_inv_mass.assign(inv_mass)
        paper.body_inertia.assign(inertia)
        paper.body_inv_inertia.assign(inv_inertia)
        self.vbd.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)

        self.coupling_forces = wp.zeros(main.body_count, dtype=wp.spatial_vector, device=device)
        self._gravity_acc = wp.zeros(paper.body_count, dtype=wp.vec3, device=device)
        self.vbd.coupling_eval_gravity_acceleration(self._gravity_acc, None)
        self.start = wp.clone(main.body_q)

        desired = stack.cfg.substeps * self.rigid_dt / float(frame_dt)
        self.paper_substeps = max(1, math.ceil(desired - 1e-9))

    def __getattr__(self, name):
        return getattr(self.rigid_solver, name)


    def step(self, state_in, state_out, control, contacts, dt):
        if abs(dt - self.rigid_dt) > 1e-9:
            raise ValueError("Rigid timestep changed; rebuild the paper coupler")
        wp.copy(self.start, state_in.body_q)
        self.rigid_solver.step(state_in, state_out, control, contacts, dt)
        self.paper_pass(self.start, state_out.body_q, state_out.body_qd, dt)

    def paper_pass(self, start, end, end_qd, dt):
        """Steps 1-2 of the proxy pass over ``paper_substeps`` slices of ``dt``."""

        stack, vbd = self.stack, self.vbd
        paper = stack.model
        sub_dt = dt / self.paper_substeps
        n = self.paper_substeps
        for i in range(n):
            a, b = stack.state_0, stack.state_1
            wp.launch(
                _sync_proxy_kernel,
                dim=paper.body_count,
                inputs=[self.local_to_main, start, end, end_qd, self.model.body_com, i / n, a.body_q, a.body_qd],
                device=paper.device,
            )
            vbd.coupling_notify_input_state_update(a, newton.StateFlags.BODY_Q | newton.StateFlags.BODY_QD, dt=sub_dt)
            a.clear_forces()
            vbd.coupling_rewind_proxy_body(self.local_to_main, a, self.coupling_forces, self._gravity_acc, sub_dt)
            vbd.coupling_notify_input_state_update(a, newton.StateFlags.BODY_QD | newton.StateFlags.BODY_F, dt=sub_dt)
            stack.collision_pipeline.collide(a, stack.contacts)
            c = vbd.coupling_prepare_proxy_contacts(a, stack.contacts, contacts_freshly_detected=True)
            vbd.step(a, b, stack.control, c, sub_dt)
            a.particle_q, b.particle_q = b.particle_q, a.particle_q
            a.particle_qd, b.particle_qd = b.particle_qd, a.particle_qd
            a.body_q, b.body_q = b.body_q, a.body_q
            a.body_qd, b.body_qd = b.body_qd, a.body_qd
        if n % 2:
            a, b = stack.state_0, stack.state_1
            for name in ("particle_q", "particle_qd", "body_q", "body_qd"):
                wp.copy(getattr(b, name), getattr(a, name))
                src, dst = getattr(a, name), getattr(b, name)
                setattr(a, name, dst)
                setattr(b, name, src)

    def reset_feedback(self) -> None:
        """Zero the rewind's feedback buffer (call with the paper reset)."""

        self.coupling_forces.zero_()




def install_paper_coupling(env, stack, body_mapping: dict[str, int],
                           cfg: ProxyCouplingCfg | None = None) -> PaperProxyCoupler:
    """Wrap the main solver so the proxy pass runs inside its step (and its CUDA graph)."""

    from isaaclab_newton.physics import NewtonManager

    rigid, rigid_dt = solver_and_dt()
    coupled = PaperProxyCoupler(rigid, stack, body_mapping, rigid_dt, float(env.step_dt), cfg)
    state = NewtonManager.get_state_0()
    coupled.paper_pass(state.body_q, state.body_q, state.body_qd, coupled.rigid_dt)
    stack.reset()
    coupled.reset_feedback()
    install_solver(coupled)
    c = coupled.cfg
    print(f"[pack_invoice] PROXY COUPLING one-way: "
          f"{len(body_mapping)} proxies (hand links and box flaps), {coupled.paper_substeps} paper substeps/rigid step, "
          f"mass_scale {c.mass_scale:g}, full-surface contact on")
    return coupled
