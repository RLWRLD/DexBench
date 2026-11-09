# SPDX-License-Identifier: Apache-2.0
"""What DB-L1's task hands Arena and record_demos.py: success and the failing end, both read from the stream, each
bag built in its park slot and meeting the task's box colliders as boxes; every demo written carries the stream's
record, with the bags' stepping and material, as its ``dexbench_l1_stream`` attribute; and a recording session is
refused on a stream seed that demos beside its file were drawn on. The task is built from two pool bags (their
templates, on Newton without Kit); the env and the stream are stand-ins, Isaac Lab's own recorder manager writes the
HDF5."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")
h5py = pytest.importorskip("h5py")
pytest.importorskip("isaaclab.managers", exc_type=ImportError)
pytest.importorskip("isaaclab_arena", reason="the task needs isaaclab_arena on the path")
pytest.importorskip("newton", reason="the bags' templates are built on Newton")

from isaaclab.managers import (  # noqa: E402
    DatasetExportMode,
    RecorderManager,
    RecorderManagerBaseCfg,
    RecorderTerm,
    RecorderTermCfg,
)
from isaaclab.utils.datasets import HDF5DatasetFileHandler  # noqa: E402

from dexbench_arena import ASSETS_DIR  # noqa: E402
from dexbench_arena.assets.polybag.sort_variants import SORT_VARIANTS_BY_KEY, variant_usd_relpath  # noqa: E402
from dexbench_arena.tasks.pick_polybag_from_conveyor import DexBenchPickPolybagFromConveyorTask, L1Bag  # noqa: E402
from dexbench_arena.tasks.polybag_stream import STREAM_TERM  # noqa: E402
from dexbench_arena.tasks.polybag_stream_record import EPISODE_RECORD_ATTR  # noqa: E402
from dexbench_arena.tasks.recording_hooks import with_episode_attributes  # noqa: E402

KEYS = ("grey_20x30__1a", "pink_20x30__2b")


@pytest.fixture(scope="module")
def task():
    bags = [L1Bag(f"l1_bag_{i}", key, str(ASSETS_DIR / "object_library" / variant_usd_relpath(key)),
                  SORT_VARIANTS_BY_KEY[key].body.footprint_m, (2.3 + 0.5 * i, 1.6)) for i, key in enumerate(KEYS)]
    if not Path(bags[0].usd_path).is_file():
        pytest.skip(f"the pool's payload is not pulled: {bags[0].usd_path}")
    return DexBenchPickPolybagFromConveyorTask(
        bags, [], {"lead": "bag", "seed": 1100}, park_surface_z=-0.6, park_clearance=0.006, vbd_substeps=8, vbd_iterations=5,
        mjwarp_substeps=8, full_surface_contact=True, frame_truncation="init", coupling_mode="two_way",
        material={"tet_contact_ke": 3000.0, "tet_collider_ke": 3000.0}, box_colliders=(r"/Wall_N$",))


def test_success_and_the_failing_end_come_from_the_stream_and_each_bag_waits_in_its_slot(task):
    terms = task.get_termination_cfg()
    assert terms.timeout_s == 600.0
    assert terms.success[0].predicate_sequence[0].func.__name__ == "stream_success"
    assert list(terms.failures) == ["stream_failed"]
    pos = task.scene_cfg.l1_bag_1.init_state.pos
    assert pos[0] == pytest.approx(2.8, abs=0.02) and pos[1] == pytest.approx(1.6, abs=0.02) and pos[2] < -0.5
    assert task.scene_cfg.l1_bag_0.box_colliders == task.scene_cfg.l1_bag_1.box_colliders == (r"/Wall_N$",)


def test_every_demo_written_carries_its_stream_record_with_the_bags_stepping_and_material(task, tmp_path):
    stream = NS(episode_record=lambda env_id: {"session": {}, "placed": 1})
    env = NS(num_envs=1, device="cpu", event_manager=NS(get_term_cfg=lambda name: NS(func={STREAM_TERM: stream}[name])),
             sim=NS(is_playing=lambda: True),
             cfg=NS(env_name="stub", decimation=1, sim=NS(dt=1 / 60, render_interval=1), scene=NS(num_envs=1)))
    cfg = RecorderManagerBaseCfg(dataset_export_dir_path=str(tmp_path), dataset_filename="demos",
                                 dataset_export_mode=DatasetExportMode.EXPORT_SUCCEEDED_ONLY)
    cfg.dataset_file_handler_class_type = with_episode_attributes(
        HDF5DatasetFileHandler, lambda episode: task.episode_attributes(env, 0))
    cfg.any_term = RecorderTermCfg(class_type=RecorderTerm)
    manager = env.recorder_manager = RecorderManager(cfg, env)
    for _ in range(5):
        manager.add_to_episodes("actions", torch.zeros(1, 3))
    manager.set_success_to_episodes([0], torch.tensor([[True]]))
    manager.export_episodes([0])
    manager.close()
    with h5py.File(tmp_path / "demos.hdf5", "r") as recording:
        demo = recording["data/demo_0"]
        assert demo.attrs["success"] and demo["actions"].shape == (5, 3)
        record = json.loads(demo.attrs[EPISODE_RECORD_ATTR])
    assert record["placed"] == 1
    assert record["session"]["polybag_stepping"]["coupling_mode"] == "two_way"
    assert record["session"]["polybag_material"] == {"tet_contact_ke": 3000.0, "tet_collider_ke": 3000.0}


def test_a_session_on_a_seed_that_demos_beside_its_file_were_drawn_on_is_refused(task, tmp_path):
    def recording(name: str, seed: int, source: str) -> None:
        with h5py.File(tmp_path / name, "w") as f:
            f.create_group("data/demo_0").attrs[EPISODE_RECORD_ATTR] = json.dumps(
                {"session": {"seed": seed}, "plan": {"source": source}})

    recording("train_a_s1101.hdf5", 1101, "drawn")
    recording("plan.hdf5", 1100, "plan")
    task.validate_recording_dataset(tmp_path / "train_a_s1100.hdf5")
    recording("train_a_s1100.hdf5", 1100, "drawn")
    with pytest.raises(ValueError, match="record on --l1_stream_seed 1102"):
        task.validate_recording_dataset(tmp_path / "train_a_s1100.hdf5")


def test_a_run_without_the_yaml_has_the_benchmarks_bag_physics():
    """The environment's defaults, on the cfg and on the command line, are the bags' physics the spec states for
    every DB-L1 run (T19D, the collection's): a run without the YAML builds the same bags and stepping."""
    import argparse

    from dexbench_arena.benchmark.spec import DB_L1
    from dexbench_arena.environments.manipulation import e2_logistics_l1_environment as l1

    fields = {name for name in l1.DexBenchLogisticsE2L1EnvironmentCfg.__dataclass_fields__}
    stated = {key: value for key, value in DB_L1.environment.items() if key in fields and key.startswith(("polybag_", "l1_bag_wrist"))}
    assert len(stated) == 15
    parser = argparse.ArgumentParser()
    l1.DexBenchLogisticsE2L1Environment.add_cli_args(parser)
    flags, _ = parser.parse_known_args([])
    for cfg in (l1.DexBenchLogisticsE2L1EnvironmentCfg(),
                l1.DexBenchLogisticsE2L1EnvironmentCfg(**{key: getattr(flags, key) for key in stated})):
        assert {key: getattr(cfg, key) for key in stated} == stated
        settings, spec = l1.l1_settings(cfg), l1.l1_settings(l1.DexBenchLogisticsE2L1EnvironmentCfg(**stated))
        assert (settings.stepping, settings.material) == (spec.stepping, spec.material)


def test_the_benchmark_records_the_two_bag_headline_and_a_collection_keeps_every_bag():
    """The spec's l1_headline_bags (set on a model's run) reaches the stream; the collection's file sets none (every
    bag placed is its success, which replays keep too); a continuous stream or more bags than the
    episode's are refused."""
    import yaml

    from dexbench_arena.benchmark.spec import DB_L1
    from dexbench_arena.environments.manipulation import e2_logistics_l1_environment as l1

    benchmark = l1.DexBenchLogisticsE2L1EnvironmentCfg(l1_headline_bags=DB_L1.scoring_environment["l1_headline_bags"])
    assert l1.l1_stream_params(l1.l1_settings(benchmark))["headline_bags"] == 2
    collect = Path(__file__).resolve().parents[3] / "scripts/evaluation/experiment_configs/db_l1_collect.yaml"
    environment = yaml.safe_load(collect.read_text())["shared"]["environment"]
    assert "l1_headline_bags" not in environment
    assert l1.l1_stream_params(l1.l1_settings(l1.DexBenchLogisticsE2L1EnvironmentCfg()))["headline_bags"] == 0
    for refused in ({"l1_stream_mode": "continuous"}, {"l1_headline_bags": 4}):
        with pytest.raises(ValueError, match="l1_headline_bags"):
            l1.l1_settings(l1.DexBenchLogisticsE2L1EnvironmentCfg(**{"l1_headline_bags": 2, **refused}))
