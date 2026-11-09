# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Tensor-only DB-R1 placement rules, shared by the task's terms and the collection station's status."""

import torch


GRAVITY_M_S2 = 9.81


def evaluate(centers, half_extents, lin_vel, ang_vel, hand_forces, eligible, masses_kg, *, source_bounds,
             basket_bounds, workspace_xy_bounds, dropped_min_z_m, basket_margin_m, released_max_force_n,
             released_max_weight_fraction, still_gates):
    """Per-fruit masks of the pile. Inputs are (env, fruit, xyz), forces (env, fruit), eligible and masses_kg
    (fruit,)."""
    source = centers.new_tensor(source_bounds)
    basket = centers.new_tensor(basket_bounds)
    workspace = centers.new_tensor(workspace_xy_bounds)
    finite = (torch.isfinite(centers).all(-1) & torch.isfinite(lin_vel).all(-1)
              & torch.isfinite(ang_vel).all(-1) & torch.isfinite(hand_forces))
    in_basket = ((centers >= basket[0]) & (centers <= basket[1])).all(-1)
    inside = (in_basket
              & ((centers[..., :2] - half_extents[..., :2] >= basket[0, :2] - basket_margin_m)
                 & (centers[..., :2] + half_extents[..., :2] <= basket[1, :2] + basket_margin_m)).all(-1)
              & (centers[..., 2] + half_extents[..., 2] <= basket[1, 2] + basket_margin_m))
    on_source = ((centers >= source[0]) & (centers <= source[1])).all(-1)
    settled = (lin_vel.norm(dim=-1) <= still_gates[0]) & (ang_vel.norm(dim=-1) <= still_gates[1])
    weight_n = masses_kg.to(hand_forces) * GRAVITY_M_S2
    released = hand_forces <= torch.clamp(released_max_weight_fraction * weight_n, max=released_max_force_n)
    outside = ((centers[..., :2] < workspace[0]) | (centers[..., :2] > workspace[1])).any(-1)
    dropped = (centers[..., 2] < dropped_min_z_m) | (~on_source & ~in_basket & released & settled)
    wrong_kind = (in_basket & ~eligible).any(-1)
    multiple = in_basket.sum(-1) > 1
    failed = (~finite | outside | dropped).any(-1) | wrong_kind | multiple
    others_on_source = (on_source.sum(-1, keepdim=True) - on_source.long()) == centers.shape[1] - 1
    candidates = inside & settled & released & eligible & others_on_source & ~failed[:, None]
    return {"candidates": candidates, "failed": failed, "dropped": dropped.any(-1), "in_basket": in_basket,
            "inside": inside, "on_source": on_source, "wrong_kind": wrong_kind, "multiple": multiple,
            "settled": settled, "released": released}


def operator_status(masks, held, held_steps, step_dt, hold_time_s):
    """The success term's parts for the collection station's "This take" (``predicates.status``), per env."""
    inside = masks["inside"]
    held_s = (held_steps.max(-1).values * step_dt if held_steps is not None
              else torch.zeros(inside.shape[0], device=inside.device))
    return {"in_basket": masks["in_basket"].sum(-1), "right_kind": ~masks["wrong_kind"],
            "released": (inside & masks["released"]).any(-1), "still": (inside & masks["settled"]).any(-1),
            "others_on_tray": (masks["on_source"] | masks["in_basket"]).all(-1), "dropped": masks["dropped"],
            "held_s": held_s, "hold_s": hold_time_s, "success": held}
