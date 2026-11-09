# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""MuJoCo-Warp contact filtering for kinematic bodies at their parking poses."""

from __future__ import annotations

from collections.abc import Sequence
import re

import numpy as np
import warp as wp


@wp.kernel
def _filter_parked_contacts(
    count: wp.array[int],
    world: wp.array[int],
    geoms: wp.array[wp.vec2i],
    geom_body: wp.array[int],
    body_position: wp.array2d[wp.vec3],
    managed: wp.array2d[wp.bool],
    parked: wp.array2d[wp.vec3],
    contact_type: wp.array[int],
    efc_address: wp.array2d[int],
):
    contact = wp.tid()
    if contact >= count[0]:
        return
    env = world[contact]
    disabled = False
    for side in range(2):
        geom = geoms[contact][side]
        if geom >= 0:
            body = geom_body[geom]
            if managed[env, body]:
                delta = body_position[env, body] - parked[env, body]
                if wp.dot(delta, delta) < 1.0e-6:
                    disabled = True
    if disabled:
        contact_type[contact] = 0
        for axis in range(efc_address.shape[1]):
            efc_address[contact, axis] = -1


class ParkedBodyContactFilter:
    """Disable contacts at named bodies' parked positions, independently per world."""

    def __init__(self, solver, body_offsets: Sequence[tuple[str, float, float, float]]) -> None:
        if solver.use_mujoco_cpu or not solver.mjw_model.opt.run_collision_detection:
            raise ValueError("Parked-body filtering requires native MuJoCo-Warp contacts (use_mujoco_cpu=False)")
        model = solver.model
        mapping = solver.mjc_body_to_newton.numpy()
        positions = model.body_q.numpy()[:, :3]
        managed = np.zeros(mapping.shape, dtype=bool)
        parked = np.zeros((*mapping.shape, 3), dtype=np.float32)
        for expression, dx, dy, dz in body_offsets:
            matches = {i for i, label in enumerate(model.body_label) if re.fullmatch(expression, label)}
            if not matches:
                raise ValueError(f"No Newton body matches parked-body filter {expression!r}")
            for world, body in zip(*np.where(np.isin(mapping, list(matches)))):
                if managed[world, body]:
                    raise ValueError(f"Overlapping parked-body filters for {model.body_label[mapping[world, body]]}")
                managed[world, body] = True
                parked[world, body] = positions[mapping[world, body]] + (dx, dy, dz)
        self.managed = wp.array(managed, dtype=wp.bool, device=model.device)
        self.parked = wp.array(parked, dtype=wp.vec3, device=model.device)
        self._previous = solver.mjw_model.callback.contactfilter

    def __call__(self, model, data) -> None:
        if self._previous is not None:
            self._previous(model, data)
        wp.launch(
            _filter_parked_contacts,
            dim=data.naconmax,
            inputs=[
                data.nacon,
                data.contact.worldid,
                data.contact.geom,
                model.geom_bodyid,
                data.xpos,
                self.managed,
                self.parked,
                data.contact.type,
                data.contact.efc_address,
            ],
            device=self.parked.device,
        )
