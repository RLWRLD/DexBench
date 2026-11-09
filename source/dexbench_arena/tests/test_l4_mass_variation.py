# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""DB-L4's mass draw weighs a slot's box or bag at its nominal mass times the drawn multiplier, from the build's masses,
and records the multiplier."""

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("isaaclab_arena.variations.variations_hydra")

import torch  # noqa: E402
import warp as wp  # noqa: E402
from isaaclab.managers import EventTermCfg  # noqa: E402
from isaaclab_arena.utils.configclass import make_configclass  # noqa: E402
from isaaclab_arena.variations import variations_hydra  # noqa: E402
from isaaclab_arena.variations.recorded_variation_replay import configure_recorded_variation_replay  # noqa: E402

from dexbench_arena.layouts.e2_logistics_l4_layout import MASS_MULTIPLIERS  # noqa: E402
from dexbench_arena.evaluation import replay_guards  # noqa: E402
from dexbench_arena.variations.db_l4_mass_variation import L4MassVariation  # noqa: E402


class _Box:
    """A one-body rigid object built at 1 kg: what the term reads and writes."""

    num_bodies = 1
    device = "cpu"

    def __init__(self, num_envs: int):
        self.data = SimpleNamespace(body_mass=wp.array(torch.ones(num_envs, 1).numpy(), device="cpu"),
                                    body_inertia=wp.array(torch.full((num_envs, 1, 9), 0.01).numpy(), device="cpu"))
        self.masses, self.inertias = {}, {}

    def set_masses_index(self, masses, body_ids, env_ids):
        self.masses.update(zip(env_ids.tolist(), masses[:, 0].tolist()))

    def set_inertias_index(self, inertias, body_ids, env_ids):
        self.inertias.update(zip(env_ids.tolist(), inertias[:, 0, 0].tolist()))


def _term(variation, asset, num_envs):
    env = SimpleNamespace(num_envs=num_envs, device="cpu", scene={variation.entity_name: asset})
    _, cfg = variation.build_event_cfg()
    return cfg.func(EventTermCfg(func=cfg.func, params=cfg.params), env), env, cfg


def test_a_run_config_cannot_disable_the_draw():
    variation = L4MassVariation(2, "l4_parcel_02", 1.0, False, MASS_MULTIPLIERS["ood"])
    with pytest.raises(ValueError, match="l4_mass_draw"):
        variations_hydra.apply_overrides({"l4_park_slab": [variation]}, ["l4_park_slab.mass_slot_2.enabled=false"])


def test_a_box_weighs_its_nominal_mass_times_the_draw_with_its_inertia_scaled():
    variation = L4MassVariation(0, "l4_parcel_00", 2.0, False, MASS_MULTIPLIERS["ood"])
    drawn = []
    variation.add_sample_listener(lambda sample, env_ids: drawn.append(list(sample)))
    box = _Box(3)
    term, env, cfg = _term(variation, box, 3)
    for _ in range(2):
        term(env, torch.tensor([0, 2]), **cfg.params)
        assert set(box.masses) == {0, 2}
        assert all(m in MASS_MULTIPLIERS["ood"] for m in drawn[-1])
        assert [box.masses[i] for i in (0, 2)] == pytest.approx([2.0 * m for m in drawn[-1]])
        assert all(box.inertias[i] == pytest.approx(0.01 * box.masses[i]) for i in box.masses)


def test_a_bag_s_particles_are_scaled_to_its_nominal_mass_times_the_draw(monkeypatch):
    from isaaclab_newton.physics import NewtonManager

    count, num_envs = 4, 2
    model = SimpleNamespace(particle_mass=wp.array(torch.full((count * num_envs,), 0.025).numpy(), device="cpu"),
                            particle_inv_mass=wp.array(torch.full((count * num_envs,), 40.0).numpy(), device="cpu"))
    monkeypatch.setattr(NewtonManager, "get_model", classmethod(lambda cls: model))
    bag = SimpleNamespace(particles_per_object=count, particle_offsets=[0, count])
    variation = L4MassVariation(1, "l4_parcel_01", 0.3, True, MASS_MULTIPLIERS["id"])
    term, env, cfg = _term(variation, bag, num_envs)
    term(env, torch.tensor([1]), **cfg.params)
    mass, inverse = wp.to_torch(model.particle_mass), wp.to_torch(model.particle_inv_mass)
    assert float(mass[:count].sum()) == pytest.approx(0.1)
    total = float(mass[count:].sum())
    assert any(total == pytest.approx(0.3 * m) for m in MASS_MULTIPLIERS["id"])
    assert torch.allclose(inverse[count:], 1.0 / mass[count:])


def test_a_benchmark_replay_weighs_the_recorded_draw_and_admits_the_mass_events(tmp_path):
    """Bound to recorded rows, a box's and a bag's mass events draw the recorded multipliers through the copies the
    env config holds, and the strict replay guard admits both as their variations' events."""
    rows = tmp_path / "rows.jsonl"
    rows.write_text(json.dumps({"variations": {"l4_park_slab.mass_slot_0": 4.0, "l4_park_slab.mass_slot_1": 0.5}}) + "\n")
    box = L4MassVariation(0, "l4_parcel_00", 1.0, False, MASS_MULTIPLIERS["ood"])
    bag = L4MassVariation(1, "l4_parcel_01", 0.3, True, MASS_MULTIPLIERS["id"])
    variations = {"l4_park_slab": [box, bag]}
    scheduler = configure_recorded_variation_replay(rows, variations)
    events = make_configclass("VariationsEventCfg", [(name, EventTermCfg, cfg) for name, cfg in
                                                     (v.build_event_cfg() for v in (box, bag))])()
    replay_guards.check_replayed_reset_events(events, what="DB-L4 benchmark replay", strict=True,
                                              bound_events=replay_guards.bound_variation_events(variations))
    scheduler.assign_new_episodes([0])
    env_ids = torch.tensor([0])
    assert float(events.l4_mass_slot_0_variation.params["sampler"].sample(1, env_ids)[0, 0]) == pytest.approx(4.0)
    assert float(events.l4_mass_slot_1_variation.params["sampler"].sample(1, env_ids)[0, 0]) == pytest.approx(0.15)
