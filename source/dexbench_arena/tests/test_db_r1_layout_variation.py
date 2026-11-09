# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""DB-R1 layout sets: a teleop session holds a layout until a demo of it is kept, in one running scene."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("isaaclab_arena.variations.recorded_variation_replay", reason="needs IsaacLab-Arena importable")

import torch  # noqa: E402
from isaaclab_arena.variations import variations_hydra  # noqa: E402
from isaaclab_arena.variations.recorded_variation_replay import configure_recorded_variation_replay  # noqa: E402

from dexbench_arena.environments.manipulation.e1_retail_environment import (  # noqa: E402
    DexBenchRetailStoreE1EnvironmentCfg,
)
from dexbench_arena.evaluation.artifacts import jsonl_bytes  # noqa: E402
from dexbench_arena.evaluation.replay_guards import ObjectIdentity, benchmark_replay_problems, read_episode_records  # noqa: E402
from dexbench_arena.tasks.r1_clutter import R1_OBJECT_AXIS, layout_rows, layout_set, make_recipe  # noqa: E402
from dexbench_arena.variations.collection_rows import CollectionRows, hold_rows  # noqa: E402
from dexbench_arena.variations.db_r1_layout_variation import ApplyR1LayoutFromSampler, R1LayoutVariation  # noqa: E402

VARIANTS = [{"variant_id": f"plum_v0{i}", "asset": f"dexbench_retail_e1_fc_plum_v0{i}", "scale": 1.0,
             "size_m": [0.05, 0.05, 0.045], "mass_kg": 0.06, "asset_version": "1.1.0"} for i in range(1, 6)]
SEEDS = (10004, 10000, 10002)


def _write_set(directory, seeds=SEEDS, count=12):
    directory.mkdir(exist_ok=True)
    for seed in seeds:
        recipe = make_recipe("plum_pile", count, seed, variants=VARIANTS, exact=True) | {"status": "settled"}
        (directory / f"plum_pile_collection_s{seed}.json").write_text(json.dumps(recipe))
    (directory / "index_10000.json").write_text("{}")
    return directory


def test_a_benchmark_replay_refuses_rows_whose_layout_files_changed(tmp_path):
    """The rows' fit carries their layout set's size and digest (R1_OBJECT_AXIS.source_fit), so a layout edited or
    lost after the rows were written is refused before the replay starts, not at scoring."""
    directory = _write_set(tmp_path / "plum")
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_bytes(jsonl_bytes(layout_rows(directory, "plum_pile")))
    build = DexBenchRetailStoreE1EnvironmentCfg(task="pick_fruit_from_display", scene_layout="fc_260821",
                                                r1_target_sku="plum_pile", r1_layouts=str(directory))

    def problems():
        return benchmark_replay_problems(ObjectIdentity.of(R1_OBJECT_AXIS, build), read_episode_records(rows_path),
                                         {"r1_fruit_000.layout"})

    assert problems() == []
    path = directory / "plum_pile_collection_s10002.json"
    recipe = json.loads(path.read_text())
    recipe["objects"][0]["position_xyz"][0] += 0.01
    path.write_text(json.dumps(recipe))
    (found,) = problems()
    assert "layouts_sha256=" in found and found.startswith("3 of 3 rows"), found
    path.unlink()
    assert any("was made with layouts=3, this build is 2" in line for line in problems())


class _Fruit:
    def __init__(self):
        self.pose = self.velocity = None

    def write_root_pose_to_sim_index(self, root_pose, env_ids):
        self.pose = root_pose.clone()

    def write_root_velocity_to_sim_index(self, root_velocity, env_ids):
        self.velocity = root_velocity.clone()


def test_a_layout_is_held_through_discarded_takes_and_the_next_follows_a_kept_demo(tmp_path):
    directory = _write_set(tmp_path / "plum")
    layouts = layout_set(directory, "plum_pile")
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_bytes(jsonl_bytes(layout_rows(directory, "plum_pile")))
    rows = CollectionRows.read(rows_path)
    build = DexBenchRetailStoreE1EnvironmentCfg(task="pick_fruit_from_display", scene_layout="fc_260821",
                                                r1_target_sku="plum_pile", r1_layouts=str(directory))
    identity = ObjectIdentity.of(R1_OBJECT_AXIS, build)
    assert benchmark_replay_problems(identity, read_episode_records(rows_path), {"r1_fruit_000.layout"}) == []
    variation = R1LayoutVariation(layouts, "r1_fruit_000")
    assert variation.cfg.dataset_values == {label: recipe["seed"] for label, recipe in layouts.items()}
    scheduler = hold_rows(configure_recorded_variation_replay(rows_path, {"r1_fruit_000": [variation]}), rows)
    _, event = variation.build_event_cfg()
    fruit = {o["name"]: _Fruit() for o in next(iter(layouts.values()))["objects"]}
    origin = torch.tensor([[10.0, -2.0, 0.0]])
    env = SimpleNamespace(scene=type("Scene", (dict,), {"num_envs": 1, "env_origins": origin})(fruit), device="cpu")
    term = ApplyR1LayoutFromSampler(event, env)

    def written(recipe) -> bool:
        """Every fruit at the recipe's pose in env 0 (env-local position plus the env origin, xyzw), at rest."""
        return all(torch.allclose(fruit[o["name"]].pose, torch.tensor([[*o["position_xyz"], *o["rotation_xyzw"]]])
                                  + torch.cat((origin, torch.zeros(1, 4)), dim=1))
                   and not fruit[o["name"]].velocity.any() for o in recipe["objects"])

    def take() -> str:
        """Arena's _reset_idx for env 0: the finishing take's row is released, the next take assigned, events run."""
        if scheduler.num_assignments_started:
            scheduler.complete_episodes([0])
        scheduler.assign_new_episodes([0])
        term(env, torch.tensor([0]), **event.params)
        (label,) = [label for label, recipe in layouts.items() if written(recipe)]
        return label

    first, second, _ = layouts
    assert take() == first
    assert take() == first
    scheduler.advance("demo_0")
    assert take() == second
    with pytest.raises(ValueError, match="r1_layouts"):
        variations_hydra.apply_overrides({"r1_fruit_000": [variation]}, ["r1_fruit_000.layout.enabled=false"])

