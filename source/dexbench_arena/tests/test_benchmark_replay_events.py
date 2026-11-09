# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""A benchmark replay admits an event declared deterministic, of the variations' events only those its replay bound, and
an event that draws through a variation only while that variation is bound (needs Isaac Lab and Arena)."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("isaaclab.managers", reason="needs Isaac Lab (the DexBench container)")
pytest.importorskip("isaaclab_arena.variations.recorded_variation_replay", reason="needs IsaacLab-Arena on the path")

import dexbench_arena.environments  # noqa: E402, F401  -- registers the DexBench environment types the spec names
from isaaclab.managers import EventTermCfg  # noqa: E402
from isaaclab_arena.assets.registries import EnvironmentRegistry  # noqa: E402
from isaaclab_arena.utils.configclass import make_configclass  # noqa: E402
from isaaclab_arena.variations.object_mass_variation import ApplyObjectMassFromSampler, ObjectMassVariation  # noqa: E402
from isaaclab_arena.variations.recorded_variation_replay import configure_recorded_variation_replay  # noqa: E402

from dexbench_arena import ASSETS_DIR  # noqa: E402
from dexbench_arena.assets.polybag.sort_variants import variant_usd_relpath  # noqa: E402
from dexbench_arena.benchmark.spec import DB_L1, DB_M3  # noqa: E402
from dexbench_arena.environments.manipulation.e2_logistics_l1_environment import (  # noqa: E402
    L1_PARK_SLAB_ENTITY,
    l1_bag_variants,
    l1_settings,
    l1_stream_layout_draw_size,
)
from dexbench_arena.evaluation import replay_guards  # noqa: E402
from dexbench_arena.tasks.deliver_parcels_barcode_up import DexBenchDeliverParcelsBarcodeUpTask  # noqa: E402
from dexbench_arena.evaluation.replay_guards import ReplayGuardError, check_replayed_reset_events  # noqa: E402
from dexbench_arena.variations.db_l1_stream_layout_variation import L1StreamLayoutVariation  # noqa: E402


def _bound_mass_variation(tmp_path, draws):
    """Arena's mass variation on ``box``, enabled and bound to rows recording ``draws``."""
    rows = tmp_path / "rows.jsonl"
    rows.write_text("".join(json.dumps({"variations": {"box.mass": [draw]}}) + "\n" for draw in draws))
    variation = ObjectMassVariation("box")
    variation.enable()
    variations = {"box": [variation]}
    configure_recorded_variation_replay(rows, variations)
    return variations


def _composed_events(variations, **extra):
    """The variation events as the env config holds them: built the way ``ArenaEnvBuilder`` composes them,
    then deep-copied by configclass when the config is instantiated."""
    fields = [(name, EventTermCfg, cfg) for hosted in variations.values() for variation in hosted
              for name, cfg in (variation.build_event_cfg(),)]
    fields += [(name, EventTermCfg, cfg) for name, cfg in extra.items()]
    return make_configclass("VariationsEventCfg", fields)()


def test_a_benchmark_replay_exempts_only_the_events_its_replay_bound(tmp_path):
    """An event counts as a variation's only when it is a bound variation's term drawing through that replay.
    The env config holds deep copies of the variations' samplers, so the match cannot be by identity; a
    sampler created outside the catalogue (that no recorded row drives) draws live and is refused, and so
    is a copy of a bound sampler under another event's name."""
    variations = _bound_mass_variation(tmp_path, [0.5, 0.7])
    (variation,) = variations["box"]
    stray = ObjectMassVariation("clutter")
    events = _composed_events(variations)
    assert events.box_mass_variation.params["sampler"] is not variation.sampler
    bound = replay_guards.bound_variation_events(variations)
    assert set(bound) == {"box_mass_variation"}
    check_replayed_reset_events(events, what="Benchmark replay", strict=True, bound_events=bound)

    def term(sampler):
        return EventTermCfg(func=ApplyObjectMassFromSampler, mode="reset", params={"sampler": sampler})

    for name, sampler, match in (("clutter_mass_variation", stray.sampler, "clutter_mass_variation .*no recorded variation binds"),
                                 ("box_mass_twin", variation.sampler, "box_mass_twin .*no recorded variation binds")):
        events = _composed_events(variations, **{name: term(sampler)})
        assert replay_guards.undeclared_reset_events(events) == []
        with pytest.raises(ReplayGuardError, match=f"(?s){match}"):
            check_replayed_reset_events(events, what="Benchmark replay", strict=True, bound_events=bound)
    variation.set_replay_sampler(None)
    with pytest.raises(ReplayGuardError, match=r"(?s)box_mass_variation .*no recorded variation binds"):
        check_replayed_reset_events(_composed_events(variations), what="Benchmark replay", strict=True,
                                    bound_events=replay_guards.bound_variation_events(variations))


def _spec_environment(spec, **overrides):
    """``spec``'s environment factory and its typed config, through the registry."""
    registry = EnvironmentRegistry()
    fields = {**spec.environment, **overrides}
    factory = registry.get_component_by_name(fields.pop("type"))
    return factory, registry.get_environment_cfg_type(factory)(**fields)


def _task_events(factory, cfg):
    return factory.available_tasks[cfg.task](factory(), cfg).get_events_cfg()


def test_db_m3s_fastening_event_passes_a_benchmark_replay():
    """DB-M3's task, built from its spec, holds one interval event: the fastening term, which gates the spindle from
    contact and joint readings and draws nothing. The strict guard admits it as declared deterministic."""
    events = _task_events(*_spec_environment(DB_M3))
    assert "power_driver_fastening" in vars(events)
    check_replayed_reset_events(events, what="DB-M3 benchmark replay", strict=True)


def test_db_l1s_stream_passes_a_benchmark_replay_only_with_its_layout_variation_bound(tmp_path):
    """DB-L1's stream term draws an episode's layout, but in a benchmark replay only through the park slab's layout
    variation (``draws_through_variation``): the strict guard admits it with that variation bound to the rows, and
    refuses it with the variation left live or not composed at all. Its task is built from the spec with
    ``benchmark_replay``, so the stream it holds refuses a draw of its own."""
    pytest.importorskip("newton", reason="the bags' templates are built on Newton")
    factory, cfg = _spec_environment(DB_L1, benchmark_replay=True)
    missing = [key for key in l1_bag_variants(l1_settings(cfg))
               if (ASSETS_DIR / "object_library" / variant_usd_relpath(key)).stat().st_size < 1024]
    if missing:
        pytest.skip(f"the pool's payload is not pulled: {missing}")
    events = _task_events(factory, cfg)
    assert events.l1_stream.params["require_layout_draw"] is True
    size = l1_stream_layout_draw_size(l1_settings(cfg))
    rows = tmp_path / "rows.jsonl"
    rows.write_text("".join(json.dumps({"variations": {f"{L1_PARK_SLAB_ENTITY}.stream_layout": [[0.5] * size]}}) + "\n"
                            for _ in range(2)))
    layout = L1StreamLayoutVariation(size)
    variations = {L1_PARK_SLAB_ENTITY: [layout]}
    configure_recorded_variation_replay(rows, variations)
    bound = replay_guards.bound_variation_events(variations)

    def composed(task_events):
        return _composed_events(variations, **{name: term for name, term in vars(task_events).items()})

    check_replayed_reset_events(composed(events), what="DB-L1 benchmark replay", strict=True, bound_events=bound)
    layout.set_replay_sampler(None)
    with pytest.raises(ReplayGuardError, match=r"(?s)l1_stream .*draws through the variation event "
                                               r"l1_stream_layout_variation, which no recorded variation binds"):
        check_replayed_reset_events(composed(events), what="DB-L1 benchmark replay", strict=True, bound_events=bound)
    with pytest.raises(ReplayGuardError, match=r"(?s)l1_stream .*l1_stream_layout_variation, which is not composed"):
        check_replayed_reset_events(events, what="DB-L1 benchmark replay", strict=True, bound_events={})


def test_db_l4s_relay_passes_a_benchmark_replay():
    """DB-L4's task holds one interval event, the parcel relay, which judges and moves the parcels from the scene and
    the stage-pose table and draws nothing. The strict guard admits it as declared deterministic."""
    task = DexBenchDeliverParcelsBarcodeUpTask(parcels=[], belt_speed_m_s=0.45, polybag_usd="", vbd_substeps=1,
                                               vbd_iterations=1, mjwarp_substeps=1, full_surface_contact=False,
                                               frame_truncation=None)
    events = task.get_events_cfg()
    assert events.relay.mode == "interval"
    check_replayed_reset_events(events, what="DB-L4 benchmark replay", strict=True)
