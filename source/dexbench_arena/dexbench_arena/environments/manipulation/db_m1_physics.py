# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in DB-M1 contact damping with DB-M5's contact solref, independent of the DB-M5 task."""

from __future__ import annotations

from dexbench_arena.evaluation.evaluation_signature import NEWTON_SHAPE_OVERRIDES_ATTRIBUTE

from .newton_shape_contact import ShapeContactRequest, install_shape_contact

M1_PHYSICS_PROFILES = ("default", "damped_contacts")
M1_HAND_CONTACT_REQUEST = "m1_damped_hand_contacts"


def apply_m1_physics_profile(env_cfg, profile: str = "default", *, hand_body_pattern: str | None = None) -> None:
    """Apply after M1's normal Newton settings, before model initialization."""
    if profile not in M1_PHYSICS_PROFILES:
        raise ValueError(f"m1_physics_profile must be one of {M1_PHYSICS_PROFILES}, got {profile!r}")
    recorded = dict(getattr(env_cfg, NEWTON_SHAPE_OVERRIDES_ATTRIBUTE, None) or {})
    contacts = dict(recorded.get("shape_contact", {}))
    contacts.pop(M1_HAND_CONTACT_REQUEST, None)
    if profile == "default":
        if contacts:
            recorded["shape_contact"] = contacts
        else:
            recorded.pop("shape_contact", None)
        setattr(env_cfg, NEWTON_SHAPE_OVERRIDES_ATTRIBUTE, recorded)
        return
    if not hand_body_pattern:
        raise ValueError("damped_contacts needs the embodiment's hand_body_label_pattern")
    request = ShapeContactRequest(
        M1_HAND_CONTACT_REQUEST, hand_body_pattern, ke=2500.0, kd=100.0, solmix=1.0, match="body",
    )
    install_shape_contact(request)
    physics = env_cfg.sim.physics
    physics.collision_decimation = 1
    physics.solver_cfg.ls_iterations = 50
    physics.default_shape_cfg.ke = 625.0
    physics.default_shape_cfg.kd = 100.0
    contacts[M1_HAND_CONTACT_REQUEST] = {
        "pattern": hand_body_pattern, "match": "body", "ke": 2500.0, "kd": 100.0, "solmix": 1.0,
    }
    recorded["shape_contact"] = contacts
    setattr(env_cfg, NEWTON_SHAPE_OVERRIDES_ATTRIBUTE, recorded)
