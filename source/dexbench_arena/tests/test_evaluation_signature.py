# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""What a row was evaluated under, beyond its scene, is recorded in full and hashed (Kit-free)."""

from __future__ import annotations

import ast
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from dexbench_arena.evaluation.evaluation_signature import (
    MOVED_DEFINITIONS,
    code_revision,
    declaration_problems,
    evaluation_settings,
    evaluation_signature,
    record_evaluation_settings,
    settings_problems,
    stamp_policy_settings,
)

CODE = {"revision": "abc123", "assets": "def456"}


@dataclass
class _Solver:
    njmax: int = 6000
    num_substeps: int = 8


@dataclass
class _Physics:
    solver_cfg: _Solver = field(default_factory=_Solver)


class _Sampler:
    pass


def form_grasp(env, env_ids, sampler, head_joint_positions):
    del env, env_ids, sampler, head_joint_positions


def form_grasp_differently(env, env_ids, sampler, head_joint_positions):
    del env, env_ids, sampler, head_joint_positions


def time_out(env):
    del env


def task_success_from_progress(env):
    del env


def object_on_surface(env, surface_xy_rects):
    del env, surface_xy_rects


def _env_cfg(head=(-0.15, 0.7), substeps=8, func=form_grasp, interval=None, rects=(((0.0, 0.0), (0.4, 0.3)),)):
    grasp = SimpleNamespace(func=func, mode="interval" if interval else "reset",
                            params={"sampler": _Sampler(), "head_joint_positions": {"head_0": head[0], "head_1": head[1]}},
                            **({"interval_range_s": interval} if interval else {}))
    return SimpleNamespace(
        sim=SimpleNamespace(physics=_Physics(_Solver(num_substeps=substeps)), dt=1 / 240),
        decimation=8,
        episode_length_s=70.0,
        events=SimpleNamespace(drawer_grasp=grasp),
        terminations=SimpleNamespace(time_out=SimpleNamespace(func=time_out, params={}, time_out=True),
                                     success=SimpleNamespace(func=task_success_from_progress, params={})),
        task_progress=SimpleNamespace(
            success_criteria=[SimpleNamespace(
                name="success", predicate_sequence=[SimpleNamespace(func=object_on_surface, params={"surface_xy_rects": rects})],
                predicate_sequences=None, score=1.0, logical="all", K=None)],
            subtasks_are_sequential=False, desired_subtask_success_state=[True]),
    )


def test_the_settings_are_recorded_once_per_env_with_the_policy_stamped():
    env = SimpleNamespace(cfg=_env_cfg())
    first = record_evaluation_settings(env, 0)
    assert first["evaluation_settings"]["policy"] is None
    env.cfg = _env_cfg(head=(0.0, 0.9))
    assert record_evaluation_settings(env, 1)["evaluation_settings"]["events"] == first["evaluation_settings"]["events"]
    stamp_policy_settings(env, {"policy_variant": "gr00t", "action_chunk_length": 16})
    assert record_evaluation_settings(env, 0)["evaluation_settings"]["policy"] == {"policy_variant": "gr00t", "action_chunk_length": 16}


def test_the_signature_separates_settings_and_ignores_the_object_the_render_size_and_the_checkout():
    def row(alignment="rim", head=(-0.15, 0.7), substeps=8, func=form_grasp, interval=None, code=CODE,
            rects=(((0.0, 0.0), (0.4, 0.3)),), **cfg):
        return {
            "task_object": {"axis": "plate_variant", "key": cfg.pop("plate_variant", "ikea_365_bowl_16")},
            "environment_cfg": {"stack_alignment_point": alignment, "episode_length_s": 40.0, "hide_replay_prims": True,
                                "camera_scale": cfg.pop("camera_scale", 0.5), **cfg},
            "evaluation_settings": evaluation_settings(_env_cfg(head, substeps, func, interval, rects), code),
        }

    reference = evaluation_signature(row())
    assert evaluation_signature(row(camera_scale=1.0)) == reference
    assert evaluation_signature(row(code={"revision": "other", "assets": "other"})) == reference
    assert evaluation_signature({**row(), "seed": 9273569}) == evaluation_signature({**row(), "seed": 9200}) == reference
    assert evaluation_signature(row(alignment="root")) != reference
    assert evaluation_signature(row(hide_replay_prims=False)) != reference
    assert evaluation_signature(row(head=(0.0, 0.9))) != reference
    assert evaluation_signature(row(substeps=4)) != reference
    assert evaluation_signature(row(rects=(((0.0, 0.0), (0.4, 0.35)),))) != reference
    assert evaluation_signature(row(func=form_grasp_differently)) != reference
    assert evaluation_signature(row(interval=(1.0, 2.0))) != evaluation_signature(row(interval=(1.0, 3.0)))
    with_policy = row()
    with_policy["evaluation_settings"]["policy"] = {"policy_variant": "cosmos", "action_chunk_length": 16}
    assert evaluation_signature(with_policy) == reference


def test_rows_with_other_success_criteria_differ_and_rows_recorded_without_them_still_compare():
    """DB-H1's rim budget is a constant of the E4 factory, in no snapshot: only the composed criteria hold it.
    Built as the pinned builder composes it (Arena's TaskProgressCfg from the task's TaskTerminationCfg), a
    10 mm and a 50 mm budget sign differently; a row that does not carry the criteria compares with either once
    they are left out."""
    pytest.importorskip("isaaclab.managers", exc_type=ImportError)
    pytest.importorskip("isaaclab_arena")
    from isaaclab_arena.progress_tracking.task_progress_cfg import TaskProgressCfg

    from dexbench_arena.tasks.stack_plates import DexBenchStackPlatesTask

    def row(max_rim_offset_mm, swap_subtasks=False):
        task = DexBenchStackPlatesTask(
            plate_names=["table_plate_00", "clearing_plate_00"], stack_center_xy=(1.48, 1.9), surface_top_z_m=0.8,
            region_radius_m=0.1, expected_count=2, max_rim_offset_mm=max_rim_offset_mm, max_lean_deg=10.0,
            nest_pitch_m=0.03, plate_dropped_min_z_m=0.5, max_seat_height_error_mm=15.0,
            source_plate_names=["table_plate_00"])
        termination = task.get_termination_cfg()
        if swap_subtasks:
            for criteria, index in zip(termination.success, (1, 0)):
                criteria.parent_subtask_idx = index
        env_cfg = SimpleNamespace(
            sim=SimpleNamespace(physics=_Physics(), dt=1 / 240), decimation=8, episode_length_s=termination.timeout_s,
            events=SimpleNamespace(),
            terminations=SimpleNamespace(time_out=SimpleNamespace(func=time_out, params={}, time_out=True),
                                         success=SimpleNamespace(func=task_success_from_progress, params={})),
            task_progress=TaskProgressCfg(success_criteria=termination.success,
                                          subtasks_are_sequential=termination.subtasks_are_sequential,
                                          desired_subtask_success_state=termination.desired_subtask_success_state))
        return {"task_object": {"axis": "plate_variant", "key": "ikea_365_bowl_16"},
                "environment_cfg": {"stack_alignment_point": "rim"},
                "evaluation_settings": evaluation_settings(env_cfg, CODE)}

    tight, loose = row(10.0), row(50.0)
    assert tight["evaluation_settings"]["terminations"]["success"]["params"] == {}
    seated = tight["evaluation_settings"]["success_criteria"]["criteria"][0]["sequences"]["default_sequence"][0]
    assert seated["func"].endswith("plates_seated_in_stack") and seated["params"]["max_rim_offset_m"] == 0.01
    assert evaluation_signature(tight) != evaluation_signature(loose)
    recorded = tight["evaluation_settings"]["success_criteria"]["criteria"]
    assert [(c["name"], c["parent_subtask_idx"]) for c in recorded] == [("success", 0), ("stack_plates", 1)]
    assert evaluation_signature(row(10.0, swap_subtasks=True)) != evaluation_signature(tight)
    before = {**tight, "evaluation_settings": {key: value for key, value in tight["evaluation_settings"].items()
                                               if key != "success_criteria"}}
    assert evaluation_signature(before) == evaluation_signature(tight, with_success_criteria=False)
    assert evaluation_signature(before) == evaluation_signature(loose, with_success_criteria=False)


@dataclass
class _Above:
    min_height_m: float

    def __call__(self, env):
        del env


class _AboveObject:
    def __init__(self, min_height_m):
        self.min_height_m = min_height_m

    def __call__(self, env):
        del env


def test_a_predicate_is_recorded_with_its_parameters_or_its_row_is_not_scored():
    def settings(predicate):
        env_cfg = _env_cfg()
        env_cfg.task_progress.success_criteria[0].predicate_sequence = [predicate]
        return evaluation_settings(env_cfg, CODE)

    def row(predicate):
        return {"environment_cfg": {}, "evaluation_settings": settings(predicate)}

    assert evaluation_signature(row(_Above(0.1))) != evaluation_signature(row(_Above(0.2)))
    assert settings_problems({**settings(time_out), "policy": None}) == []
    assert settings_problems({**settings(_Above(0.1)), "policy": None}) == []
    assert "its own state" in settings_problems({**settings(_AboveObject(0.1)), "policy": None})[0]


def test_a_tensor_parameter_is_recorded_as_its_numbers():
    """R2's and R3's partial-credit chains hold their xy_bounds as tensors; two regions must sign apart."""
    torch = pytest.importorskip("torch")
    from functools import partial

    def settings(bounds):
        env_cfg = _env_cfg()
        env_cfg.task_progress.success_criteria[0].predicate_sequence = [partial(object_on_surface, surface_xy_rects=bounds)]
        return evaluation_settings(env_cfg, CODE)

    recorded = settings(torch.tensor([[0.0, 0.5], [0.25, 1.0]], dtype=torch.float64))["success_criteria"]["criteria"][0]
    assert recorded["sequences"]["default_sequence"][0]["params"] == {"surface_xy_rects": [[0.0, 0.5], [0.25, 1.0]]}
    wider = settings(torch.tensor([[0.0, 0.5], [0.3, 1.0]], dtype=torch.float64))
    assert evaluation_signature({"evaluation_settings": wider}) != evaluation_signature(
        {"evaluation_settings": settings(torch.tensor([[0.0, 0.5], [0.25, 1.0]], dtype=torch.float64))})


def test_server_managed_episodes_accept_empty_and_populated_manager_dicts():
    cfg = _env_cfg()
    expected = evaluation_settings(cfg, CODE)
    cfg.events = vars(cfg.events)
    cfg.terminations = vars(cfg.terminations)
    assert evaluation_settings(cfg, CODE) == expected
    cfg.terminations = {}
    assert evaluation_settings(cfg, CODE)["terminations"] == {}


def test_a_run_is_held_to_its_tasks_policy_rate_and_a_trial_to_its_own_control_step():
    ga = {"policy": {"policy_hz": 30, "policy_action_interpolation": "linear"},
          "environment": {"rby1_control_hz": 60, "camera_hz": 30}}
    declared = {"rby1_control_hz": 60, "camera_hz": 30}
    stamped = {"policy_variant": "m", "action_chunk_length": 16, "policy_hz": 30, "policy_action_interpolation": "linear"}
    legacy = {"policy_variant": "m", "action_chunk_length": 16}
    assert declaration_problems(declared, declared, stamped, "m", 16, ga) == []
    assert any("its cell asks" in problem for problem in declaration_problems(declared, declared, legacy, "m", 16, ga))
    assert declaration_problems({}, {}, legacy, "m", 16, None) == []
    assert declaration_problems({}, {}, stamped, "m", 16, None) != []
    assert declaration_problems({"rby1_control_hz": 30}, declared, {"policy_variant": "t"}, "t", None, ga) == []


@pytest.mark.parametrize("old, current", sorted(MOVED_DEFINITIONS.items()))
def test_a_row_that_names_a_moved_definition_by_its_old_path_scores_with_rows_recorded_after_the_move(old, current):
    """A row can name DB-L2's reset event or success predicate by its old path in ``dexbench_arena.db_l2_mimic_utils``,
    as a term's callable or as a callable parameter: it signs as a row that names the current path."""

    def row(name):
        settings = evaluation_settings(_env_cfg(), CODE)
        settings["events"]["drawer_grasp"]["func"] = name
        settings["events"]["drawer_grasp"]["params"]["sampler"] = f"<callable:{name}>"
        return {"environment_cfg": {"episode_length_s": 40.0}, "evaluation_settings": settings}

    assert evaluation_signature(row(old)) == evaluation_signature(row(current))
    assert evaluation_signature(row(current)) != evaluation_signature(row(f"{current}_changed"))


def test_each_moved_definition_is_defined_at_its_current_path_and_not_at_its_old_one():
    package_root = Path(__file__).resolve().parents[1]

    def defined(dotted: str) -> bool:
        module, name = dotted.rsplit(".", 1)
        path = package_root / f"{module.replace('.', '/')}.py"
        return path.exists() and any(
            isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name
            for node in ast.parse(path.read_text()).body
        )

    for old, current in MOVED_DEFINITIONS.items():
        assert defined(current), current
        assert not defined(old), old


@pytest.mark.parametrize("name", ["package_clear_of_cage", "reset_cage_clear"])
def test_rows_that_name_any_path_the_cage_clear_terms_had_score_together(name):
    """Rows name the cage-clear terms under ``dexbench_arena.db_l2_mimic_utils``, ``scripted.db_l2.mimic_utils`` or
    the current ``tasks.cage_clear``. Listed here, not read from the map, so a path the map leaves out fails."""

    def row(path):
        settings = evaluation_settings(_env_cfg(), CODE)
        settings["events"]["drawer_grasp"]["func"] = path
        return {"environment_cfg": {"episode_length_s": 40.0}, "evaluation_settings": settings}

    modules = ("dexbench_arena.db_l2_mimic_utils", "dexbench_arena.scripted.db_l2.mimic_utils", "dexbench_arena.tasks.cage_clear")
    assert len({evaluation_signature(row(f"{module}.{name}")) for module in modules}) == 1


@pytest.mark.parametrize("different_owner", [False, True])
def test_code_revision_records_clean_dirty_and_unknown_trees(tmp_path, monkeypatch, different_owner):
    assert code_revision(tmp_path, {})["dirty"] is None
    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=Test", "-c", "user.email=test@example.org", *args], check=True, capture_output=True)
    git("init", "-b", "duke/test")
    source = tmp_path / "source.py"
    source.write_text("original")
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "fixture.txt").write_text("fixture")
    git("add", "source.py", "assets")
    git("commit", "-m", "Fixture")
    expected = code_revision(tmp_path, {})
    assert expected["dirty"] is False
    if different_owner:
        monkeypatch.setenv("GIT_TEST_ASSUME_DIFFERENT_OWNER", "1")
        refused = subprocess.run(["git", "-c", "safe.directory=", "-C", str(tmp_path), "status"],
                                 capture_output=True, text=True)
        assert refused.returncode == 128
        assert "dubious ownership" in refused.stderr
    assert code_revision(tmp_path, {}) == expected
    source.write_text("modified")
    assert code_revision(tmp_path, {})["dirty"] is True


def test_a_checkout_that_downloads_its_assets_records_the_tree_its_lock_names(tmp_path):
    """RLWRLD/DexBench has no assets/ in Git: it records the assets tree its lock names, the same id a checkout
    with assets/ in Git records, and the downloaded assets (which Git ignores) leave it clean."""
    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=Test", "-c", "user.email=test@example.org", *args],
                       check=True, capture_output=True)
    git("init", "-b", "duke/test")
    (tmp_path / "reproducibility").mkdir()
    (tmp_path / "reproducibility/assets.lock.json").write_text(json.dumps(
        {"repo_id": "RLWRLD/DexBench-assets", "revision": "a" * 40, "source_assets_tree": "b" * 40}))
    (tmp_path / ".gitignore").write_text("/assets/\n")
    git("add", ".")
    git("commit", "-m", "Fixture")
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets/scene.usd").write_text("downloaded")
    recorded = code_revision(tmp_path, {})
    assert recorded["assets"] == "b" * 40 and recorded["dirty"] is False


def test_default_source_snapshot_is_reused_but_submission_refreshes_it(tmp_path, monkeypatch):
    monkeypatch.setattr("dexbench_arena.evaluation.evaluation_signature.REPO_ROOT", tmp_path)
    subprocess.run(["git", "init", "-b", "duke/test", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=Test", "-c", "user.email=test@example.org",
                    "commit", "--allow-empty", "-m", "Fixture"], check=True, capture_output=True)
    with patch("dexbench_arena.evaluation.evaluation_signature.subprocess.run", wraps=subprocess.run) as run:
        first = code_revision()
        assert first["dirty"] is False
        initial_calls = run.call_count
        assert initial_calls > 0
        first["dirty"] = "caller mutation"
        (tmp_path / "untracked.py").write_text("changed")
        assert code_revision()["dirty"] is False
        assert run.call_count == initial_calls
        assert code_revision(refresh=True)["dirty"] is True
        refreshed_calls = run.call_count
        assert refreshed_calls > initial_calls
        assert code_revision()["dirty"] is True
        assert run.call_count == refreshed_calls
        (tmp_path / "untracked.py").unlink()
        monkeypatch.setattr("dexbench_arena.evaluation.evaluation_signature.os.getpid", lambda: -1)
        assert code_revision()["dirty"] is False
        assert run.call_count > refreshed_calls
