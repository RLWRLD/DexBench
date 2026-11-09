# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""A replay or Mimic source made with another object is reported, or refused when the guard is set to."""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from dexbench_arena.assets.e2_logistics_cell import E2_LIFT_OBJECT_AXIS
from dexbench_arena.assets.e4_hotel_fb_dish_specs import DEFAULT_DISH_VARIANT, DISH_OBJECT_AXIS
from dexbench_arena.evaluation import replay_guards
from dexbench_arena.evaluation.replay_guards import (
    ObjectIdentity,
    ReplayGuardError,
    check_replay_source,
    check_replayed_reset_events,
    deterministic_reset,
)

from dexbench_arena.layouts.db_r2_layout import FC_260821_R2_OBJECT_AXIS as R2_AXIS


@dataclass
class _Cfg:
    """The fields of the E1 config the R2 axis reads, with the config's defaults."""

    r2_target_sku: str = "TC2"
    fridge_shelf3_raise_m: float = 0.0
    r2_stock_rule: str = "authored"
    r2_stock_fillers: str = "TC2,TC1"


TC1 = ObjectIdentity.of(R2_AXIS, _Cfg("TC1", 0.065))
TC1_OBJECT = {"axis": "r2_target_sku", "key": "TC1", "fit": dict(TC1.fixture)}
BASE = "dexbench_rby1_wuji2_joint.base_pose"


@pytest.mark.parametrize("source_version,built_version,refused", [
    ("3.0.0", "3.2.1", False),
    ("1.4.0", "3.0.0", True),
    (None, "3.0.0", True),
    ("3.0.0", "1.4.0", True),
    ("3.0.0", None, True),
])
def test_explicit_physical_major_contract_blocks_old_ep2_even_in_warning_mode(tmp_path, monkeypatch,
                                                                            source_version, built_version, refused):
    build = ObjectIdentity.of(R2_AXIS, _Cfg("EP2", .065), asset_version=built_version)
    assert build.benchmark_asset_major == 3
    row = {"task_object": {"axis": build.field, "key": build.key, "fit": dict(build.fixture),
                           "asset_version": source_version}}
    source = _rows(tmp_path / "ep2.jsonl", row)
    monkeypatch.setattr(replay_guards, "REPLAY_GUARD_MODE", "warn")
    if refused:
        with pytest.raises(ReplayGuardError, match="asset MAJOR"):
            replay_guards.check_benchmark_replay(build, source, set(), what="Benchmark replay")
    else:
        replay_guards.check_benchmark_replay(build, source, set(), what="Benchmark replay")
    unchanged = ObjectIdentity.of(R2_AXIS, _Cfg("TC1", .065), asset_version="2.0.0")
    assert unchanged.benchmark_asset_major is None
    old_row = {"task_object": {"axis": unchanged.field, "key": unchanged.key, "fit": dict(unchanged.fixture),
                               "asset_version": "1.4.0"}}
    replay_guards.check_benchmark_replay(unchanged, _rows(tmp_path / "tc1.jsonl", old_row), set(), what="Benchmark replay")


@pytest.mark.parametrize("source_version,current_version,refused", [
    ("1.4.0", "2.0.1", False),
    ("2.0.0", "2.1.0", False),
    ("1.4.0", "3.0.0", True),
    ("2.0.0", "3.0.0", True),
    ("2.0.0", None, True),
    (None, "2.0.1", False),
])
def test_all_known_runtime_majors_are_checked_without_opt_in(tmp_path, monkeypatch,
                                                           source_version, current_version, refused):
    identity = ObjectIdentity.of(R2_AXIS, _Cfg("TC1", .065), asset_version=current_version)
    assert identity.benchmark_asset_major is None
    row = {"task_object": {"axis": identity.field, "key": identity.key, "fit": dict(identity.fixture),
                           "asset_version": source_version}}
    source = _rows(tmp_path / "generic.jsonl", row)
    monkeypatch.setattr(replay_guards, "REPLAY_GUARD_MODE", "warn")
    if refused:
        with pytest.raises(ReplayGuardError, match="asset MAJOR"):
            replay_guards.check_benchmark_replay(identity, source, set(), what="Benchmark replay")
    else:
        replay_guards.check_benchmark_replay(identity, source, set(), what="Benchmark replay")


def _rows(path, *rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_a_source_made_with_another_object_or_fixture_profile_is_reported_then_refused(tmp_path, monkeypatch, capsys):
    build = ObjectIdentity.of(R2_AXIS, _Cfg("TC1", 0.065))
    matching = {"task_object": {"axis": "r2_target_sku", "key": "TC1", "fit": {"fridge_shelf3_raise_m": 0.065}}}
    conditions = _rows(
        tmp_path / "episode_results_rebuild0.jsonl",
        matching,
        {"environment_cfg": {"r2_target_sku": "TC1", "fridge_shelf3_raise_m": 0.065}},
        {"environment_cfg": {"r2_target_sku": "TC1"}},
        {"core": 1},
    )

    monkeypatch.setattr(replay_guards, "REPLAY_GUARD_MODE", "warn")
    check_replay_source(build, conditions, what="Replaying")
    report = capsys.readouterr().err
    assert "1 of 4 source records (first: episode_results_rebuild0.jsonl:3) were made with r2_target_sku=TC1, fridge_shelf3_raise_m=0.0" in report
    assert "1 of 4 source records (first: episode_results_rebuild0.jsonl:4) were made with r2_target_sku=TC2" in report
    assert "this build is r2_target_sku=TC1, fridge_shelf3_raise_m=0.065" in report

    monkeypatch.setattr(replay_guards, "REPLAY_GUARD_MODE", "refuse")
    with pytest.raises(ReplayGuardError, match="r2_target_sku=TC2"):
        check_replay_source(build, conditions, what="Replaying")
    check_replay_source(build, _rows(tmp_path / "same.jsonl", matching, matching), what="Replaying")
    other_lane = {"task_object": {**matching["task_object"], "fit": {"fridge_shelf3_raise_m": 0.065, "r2_target_lane": 3}}}
    check_replay_source(build, _rows(tmp_path / "lane3.jsonl", other_lane), what="Replaying")

    h5py = pytest.importorskip("h5py")
    dataset = tmp_path / "demos.hdf5"
    with h5py.File(dataset, "w") as file:
        file.create_group("data").attrs["env_args"] = json.dumps({"sim_args": {}})
    (tmp_path / "demos.job.json").write_text(json.dumps({"env": "e1", "args": ["--r2_target_sku", "TC1"]}))
    with pytest.raises(ReplayGuardError, match="demos.job.json.*fridge_shelf3_raise_m=0.0"):
        check_replay_source(build, dataset, what="Generating Mimic data")
    with h5py.File(dataset, "a") as file:
        file["data"].attrs["env_args"] = json.dumps({"task_object": matching["task_object"]})
    check_replay_source(build, dataset, what="Generating Mimic data")


def test_a_db_l2_source_from_the_260827_layout_is_reported_then_refused(tmp_path, monkeypatch, capsys):
    @dataclass
    class Cfg:
        object_box: str = "train_0"

    build = ObjectIdentity.of(E2_LIFT_OBJECT_AXIS, Cfg())
    _task_object, fit = E2_LIFT_OBJECT_AXIS.resolve(Cfg())
    stamped = {"axis": "object_box", "key": "train_0", "fit": {**fit, "seat_xyz_m": [0.736106, 1.6922, 0.871]}}
    old_cage = {"axis": "object_box", "key": "train_0", "fit": {"seat_xyz_m": [1.211, 1.8, 0.871]}}
    conditions = _rows(
        tmp_path / "train_0.jsonl",
        {"environment_cfg": {"object_box": "train_0", "anchor": "A_cage"}},
        {"environment_cfg": {"object_box": "train_0", "anchor": "A_cage"}, "task_object": old_cage},
        {"environment_cfg": {"object_box": "train_0", "anchor": "A1_CAGE"}, "task_object": stamped},
    )

    monkeypatch.setattr(replay_guards, "REPLAY_GUARD_MODE", "warn")
    check_replay_source(build, conditions, what="Replaying recorded variation samples")
    report = capsys.readouterr().err
    assert ("2 of 3 source records (first: train_0.jsonl:1) were made with object_box=train_0, scene_layout=260827; "
            "this build is object_box=train_0, scene_layout=261014") in report

    monkeypatch.setattr(replay_guards, "REPLAY_GUARD_MODE", "refuse")
    with pytest.raises(ReplayGuardError, match="scene_layout=260827"):
        check_replay_source(build, conditions, what="Replaying recorded variation samples")
    check_replay_source(build, _rows(tmp_path / "new.jsonl", {"task_object": stamped}), what="Replaying")

    h5py = pytest.importorskip("h5py")
    dataset = tmp_path / "e2_mimic.hdf5"
    with h5py.File(dataset, "w") as file:
        file.create_group("data").attrs["env_args"] = json.dumps({"task_object": old_cage})
    with pytest.raises(ReplayGuardError, match="e2_mimic.hdf5 env_args.*scene_layout=260827"):
        check_replay_source(build, dataset, what="Generating Mimic data")


@deterministic_reset
def _clear_hold(env, env_ids):
    del env, env_ids


def _jitter_pose(env, env_ids):
    del env, env_ids


def _solve_and_place_objects(env, env_ids, placement_sampler):
    del env, env_ids, placement_sampler


class _Sampler:
    def set_replay_sampler(self, replay_sampler):
        del replay_sampler


def _events():
    recorded, live = SimpleNamespace(sampler=object(), pool=None), SimpleNamespace(sampler=object(), pool=object())
    return SimpleNamespace(
        clear_hold=SimpleNamespace(func=_clear_hold, mode="reset", params={}),
        target_pose_variation=SimpleNamespace(func=_jitter_pose, mode="reset", params={"sampler": _Sampler()}),
        replayed_placement=SimpleNamespace(func=_solve_and_place_objects, mode="reset", params={"placement_sampler": recorded}),
        jitter=SimpleNamespace(func=_jitter_pose, mode="reset", params={}),
        solved_placement=SimpleNamespace(func=_solve_and_place_objects, mode="reset", params={"placement_sampler": live}),
        hide_camera=SimpleNamespace(func=_jitter_pose, mode="startup", params={}),
    )


@pytest.mark.parametrize("mode, strict, refused", [
    ("warn", False, None),
    ("refuse", False, "not declared deterministic"),
    ("warn", True, r"(?s)benchmark replay.*jitter .*solved_placement .*hide_camera .*mode startup"),
])
def test_a_replayed_env_with_an_event_that_may_draw_is_reported_or_refused(monkeypatch, capsys, mode, strict, refused):
    monkeypatch.setattr(replay_guards, "REPLAY_GUARD_MODE", mode)
    check_replayed_reset_events(None, what="Replaying", strict=strict)
    if refused is not None:
        with pytest.raises(ReplayGuardError, match=refused):
            check_replayed_reset_events(_events(), what="Replaying", strict=strict)
        return
    check_replayed_reset_events(_events(), what="Replaying", strict=strict)
    report = capsys.readouterr().err
    assert "reset event jitter (" in report and "reset event solved_placement (" in report
    assert not any(name in report for name in ("clear_hold", "target_pose", "replayed_placement", "hide_camera"))


def test_an_unlabelled_h1_source_is_the_dish_it_was_recorded_on():
    @dataclass
    class Cfg:
        plate_variant: str = DEFAULT_DISH_VARIANT

    assert ObjectIdentity.of(DISH_OBJECT_AXIS, Cfg()).source_identity({})[0] == "godmiddag_deep_plate_23"


def test_a_benchmark_replay_refuses_a_live_factor_or_an_unlabelled_row_whatever_the_mode(tmp_path, monkeypatch):
    """A benchmark cell's rows must record every enabled variation (Arena leaves an unrecorded one
    drawing live) and the build's object; the check refuses even under REPLAY_GUARD_MODE "warn"."""
    monkeypatch.setattr(replay_guards, "REPLAY_GUARD_MODE", "warn")
    light = "dexbench_retail_v1_lighting.intensity_scale"
    row = {"task_object": TC1_OBJECT, "variations": {BASE: [0.01, -0.01, 0.0]}}
    rows = _rows(tmp_path / "tc1.jsonl", row, row)

    def check(path, enabled=(BASE,), build=TC1):
        replay_guards.check_benchmark_replay(build, path, set(enabled), what="Benchmark replay")

    check(rows)
    for path, enabled, match in (
        (rows, (BASE, light), rf"2 of 2 rows .* does not record the enabled variation\(s\) \['{light}'\]"),
        (_rows(tmp_path / "old.jsonl", row, {"variations": {BASE: [0.0, 0.0, 0.0]}}), (BASE,),
         r"1 of 2 rows \(first: old.jsonl:2\) records no task_object"),
        (_rows(tmp_path / "unfit.jsonl", {**row, "task_object": {**TC1_OBJECT, "fit": {}}}), (BASE,),
         "records no fridge_shelf3_raise_m"),
        (_rows(tmp_path / "nan.jsonl", {**row, "variations": {BASE: [float("nan"), 0.0, 0.0]}}), (BASE,), "non-finite"),
        (_rows(tmp_path / "placed.jsonl", {**row, "placement": {"box": [0.0, 0.0, 0.0]}}), (BASE,), "relation placement"),
    ):
        with pytest.raises(ReplayGuardError, match=match):
            check(path, enabled)
    with pytest.raises(ReplayGuardError, match="no object axis"):
        check(rows, build=None)


def test_a_benchmark_replay_refuses_a_non_numeric_or_out_of_range_draw(tmp_path):
    """A recorded draw is replayed as it is, so a benchmark row's draw must be numbers of the variation's
    dimension inside its declared range (which values are numbers: test_benchmark_scoring.py)."""
    ranges = {BASE: ([-0.02, -0.02, 0.0], [0.02, 0.02, 0.0])}

    def check(draw):
        rows = _rows(tmp_path / "rows.jsonl", {"task_object": TC1_OBJECT, "variations": {BASE: draw}})
        replay_guards.check_benchmark_replay(TC1, rows, {BASE}, what="Benchmark replay", ranges=ranges)

    check([0.01, -0.02, 0.0])
    for draw, match in (("0.01", "not 3 finite numbers"), ([0.03, 0.0, 0.0], "outside its declared range")):
        with pytest.raises(ReplayGuardError, match=match):
            check(draw)


def test_a_fixture_value_the_fit_normalises_matches_its_own_recording():
    """``--r2_stock_fillers "TC2 TC1"`` builds the scene "TC2,TC1" does; the build and every source form of the
    value (the fit, environment_cfg, a sidecar's command line) are read in the fit's form."""
    recorded = {"task_object": {"axis": "r2_target_sku", "key": "TC1", "fit": {"r2_stock_fillers": "TC2,TC1"}}}
    in_cfg = {"task_object": {"axis": "r2_target_sku", "key": "TC1", "fit": {}},
              "environment_cfg": {"r2_stock_fillers": "TC2 TC1"}}
    in_args = {"args": ["--r2_target_sku", "TC1", "--r2_stock_fillers", "tc2, tc1"]}
    for spelling in ("TC2 TC1", "tc2,tc1", "TC2, TC1"):
        build = ObjectIdentity.of(R2_AXIS, _Cfg("TC1", r2_stock_fillers=spelling))
        assert build.differences([("row", recorded), ("cfg", in_cfg), ("sidecar", in_args)]) == [], spelling
    other = {"args": ["--r2_target_sku", "TC1", "--r2_stock_fillers", "TC1 EC1"]}
    assert ObjectIdentity.of(R2_AXIS, _Cfg("TC1")).differences([("sidecar", other)])


def test_a_benchmark_row_with_its_fixture_in_the_config_snapshot_reads_it_in_the_fit_form():
    build = ObjectIdentity.of(R2_AXIS, _Cfg("TC1", 0.065, r2_stock_fillers="TC2 TC1"))
    fit = {name: value for name, value in build.fixture.items() if name != "r2_stock_fillers"}
    row = {"task_object": {"axis": "r2_target_sku", "key": "TC1", "fit": fit},
           "environment_cfg": {"r2_stock_fillers": "tc2, tc1"}, "variations": {}}
    assert replay_guards.benchmark_replay_problems(build, [("row", row)], set()) == []
