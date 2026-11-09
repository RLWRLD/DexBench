# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""DB-L4's relay judges a parcel the belt carries past the judging line, not before it and not one held still there."""

import math

import pytest

torch = pytest.importorskip("torch")

from dexbench_arena.layouts.e2_logistics_l4_layout import (  # noqa: E402
    JUDGE_DWELL_S,
    FALL_CLEARANCE_M,
    L4_BELT_TOP_Z_M,
    L4_BELT_Y_RANGE_M,
    L4_JUDGE_LINE_X_M,
    L4_PLATE_X_RANGE_M,
)
from dexbench_arena.tasks.parcel_relay import DELIVERED, FAILED, PENDING, ParcelRelay  # noqa: E402

DT = 1.0 / 30.0
BELT_SPEED_M_S = 0.45
LENGTH_M = 0.271
CENTRE_Y = torch.full((1, 2), sum(L4_BELT_Y_RANGE_M) / 2.0)


def _staged_relay(barcode_up_cos: float):
    relay = ParcelRelay(num_envs=1, num_parcels=2, belt_speed_m_s=BELT_SPEED_M_S)
    bottom = torch.tensor([[L4_BELT_TOP_Z_M, L4_BELT_TOP_Z_M]])
    barcode = torch.tensor([[barcode_up_cos, 1.0]])
    _, stage = relay.advance(torch.tensor([0.0]), torch.zeros(1, 2), bottom, CENTRE_Y, barcode)
    assert stage.tolist() == [[True, False]]
    return relay, bottom, barcode


@pytest.mark.parametrize("barcode_up_cos, verdict", [(1.0, DELIVERED), (-1.0, FAILED)])
def test_parcel_the_belt_carries_past_the_line_is_judged(barcode_up_cos, verdict):
    relay, bottom, barcode = _staged_relay(barcode_up_cos)
    t = 0.0
    lead_x = L4_PLATE_X_RANGE_M[1] + 0.01 + LENGTH_M
    while lead_x < L4_JUDGE_LINE_X_M - 1e-3:
        t += DT
        judged, _ = relay.advance(torch.tensor([t]), torch.tensor([[lead_x, 0.0]]), bottom, CENTRE_Y, barcode)
        assert not judged.any(), f"judged with its leading edge at x {lead_x:.3f}, short of the line"
        lead_x += BELT_SPEED_M_S * DT
    assert relay.verdict[0, 0] == PENDING

    for _ in range(math.ceil(JUDGE_DWELL_S / DT) + 1):
        t += DT
        lead_x += BELT_SPEED_M_S * DT
        relay.advance(torch.tensor([t]), torch.tensor([[lead_x, 0.0]]), bottom, CENTRE_Y, barcode)
    assert relay.verdict[0, 0] == verdict


def test_parcel_held_still_at_the_line_is_not_judged():
    relay, bottom, barcode = _staged_relay(1.0)
    t = 0.0
    for _ in range(30):
        t += DT
        judged, _ = relay.advance(torch.tensor([t]), torch.tensor([[L4_JUDGE_LINE_X_M + 0.01, 0.0]]), bottom, CENTRE_Y,
                                  barcode)
        assert not judged.any(), f"judged at {t:.2f} s while lying still at the line"
    assert relay.verdict[0, 0] == PENDING


@pytest.mark.parametrize("where", ["off_the_end", "off_the_side"])
def test_a_barcode_up_parcel_that_leaves_the_belt_past_the_line_fails(where):
    """Moving +X past the line, barcode up, but no longer on FB1: dropping under the belt top off its end, or lying
    beside it at belt height. Neither is delivered; the drop fails once it is FALL_CLEARANCE below the belt."""
    relay, bottom, barcode = _staged_relay(1.0)
    t, lead_x = 0.0, L4_JUDGE_LINE_X_M + 0.01
    z = bottom.clone()
    y = CENTRE_Y.clone()
    if where == "off_the_side":
        y[0, 0] = L4_BELT_Y_RANGE_M[1] + 0.05
    for _ in range(30):
        t += DT
        lead_x += BELT_SPEED_M_S * DT
        if where == "off_the_end":
            z[0, 0] -= 0.01
        relay.advance(torch.tensor([t]), torch.tensor([[lead_x, 0.0]]), z, y, barcode)
        assert relay.verdict[0, 0] != DELIVERED, f"delivered at {t:.2f} s, its lowest point at z {float(z[0, 0]):.3f}"
    if where == "off_the_end":
        assert float(z[0, 0]) < L4_BELT_TOP_Z_M - FALL_CLEARANCE_M
        assert relay.verdict[0, 0] == FAILED
    else:
        assert relay.verdict[0, 0] == PENDING


@pytest.mark.parametrize("end_on_parcel_failure, ends", [(True, True), (False, False)])
def test_a_failed_parcel_ends_the_episode_unless_told_to_try_every_parcel(end_on_parcel_failure, ends):
    pytest.importorskip("isaaclab_arena.tasks.task_base")
    from types import SimpleNamespace

    from dexbench_arena.tasks.deliver_parcels_barcode_up import DexBenchDeliverParcelsBarcodeUpTask, L4Parcel

    pose = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    task = DexBenchDeliverParcelsBarcodeUpTask(
        parcels=[L4Parcel(f"l4_parcel_{i}", "box_1", pose, pose, 1.0) for i in range(2)],
        belt_speed_m_s=BELT_SPEED_M_S, polybag_usd="", vbd_substeps=8, vbd_iterations=5, mjwarp_substeps=8,
        full_surface_contact=True, frame_truncation="init", end_on_parcel_failure=end_on_parcel_failure,
    )
    relay, bottom, barcode = _staged_relay(1.0)
    relay.advance(torch.tensor([DT]), torch.zeros(1, 2), bottom - 1.0, CENTRE_Y, barcode)
    assert relay.verdict.tolist() == [[FAILED, PENDING]]

    term = SimpleNamespace(relay=relay)
    env = SimpleNamespace(event_manager=SimpleNamespace(get_term_cfg=lambda name: SimpleNamespace(func=term)))
    fired = [name for name, term in task.get_termination_cfg().failures.items() if bool(term.func(env)[0])]
    assert fired == (["parcel_failed"] if ends else [])


def test_parcels_delivered_records_each_resetting_episodes_count_and_averages_them():
    """The metric reported beside success: the relay's delivered count of each resetting environment, read before the
    relay resets, and its mean over the episodes."""
    np = pytest.importorskip("numpy")
    task = pytest.importorskip("dexbench_arena.tasks.deliver_parcels_barcode_up", reason="needs Isaac Lab importable")
    from types import SimpleNamespace

    relay = ParcelRelay(3, 5, 0.1, device="cpu")
    relay.verdict[0] = torch.tensor([DELIVERED, DELIVERED, FAILED, PENDING, PENDING])
    relay.verdict[2] = torch.tensor([DELIVERED] * 5)
    term = SimpleNamespace(func=SimpleNamespace(relay=relay))
    env = SimpleNamespace(event_manager=SimpleNamespace(get_term_cfg=lambda name: term))
    metric = task.ParcelsDeliveredMetric()
    recorder_cfg = metric.get_recorder_term_cfg()
    recorder = recorder_cfg.class_type.__new__(recorder_cfg.class_type)
    recorder._env, recorder.name = env, recorder_cfg.name
    name, counts = recorder.record_pre_reset(torch.tensor([0, 2]))
    assert (name, counts.tolist()) == ("parcels_delivered", [2, 5])
    metric_cfg = metric.get_metric_term_cfg()
    assert metric_cfg.recorder_term_name == name
    assert metric_cfg.compute_metric_func([np.array([2]), np.array([5]), np.array([0])]) == pytest.approx(7 / 3)
