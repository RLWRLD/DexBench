# Copyright (c) 2026, RLWRLD Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The command line the wrapper hands to an Isaac Lab script."""

from __future__ import annotations

from pathlib import Path
import re
import runpy
import shlex

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
WRAPPER_HELPER = REPO_ROOT / "scripts" / "imitation_learning" / "_dexbench_arena_script.py"
ISAACLAB_ROOT = REPO_ROOT / "submodule" / "isaaclab_arena" / "submodules" / "IsaacLab"
ARENA_CLI = REPO_ROOT / "submodule" / "isaaclab_arena" / "isaaclab_arena" / "cli" / "isaaclab_arena_cli.py"
APP_LAUNCHER = ISAACLAB_ROOT / "source" / "isaaclab" / "isaaclab" / "app" / "app_launcher.py"

CALLBACK = "dexbench_arena.isaaclab_interop.environment_registration_callback"

TELEOP_FLAGS = ("--teleop_device", "--cloudxr_env", "--auto_launch_cloudxr", "--disable_external_cameras")


@pytest.fixture(scope="module")
def helper() -> dict:
    return runpy.run_path(str(WRAPPER_HELPER))


@pytest.fixture(scope="module")
def compute(helper) -> object:
    return helper["_compute_lab_argv"]


def _lab_argv(compute, script_name: str, command: str) -> list[str]:
    return compute(script_name, shlex.split(command))[0]


def _value_after(argv: list[str], flag: str) -> str | None:
    return argv[argv.index(flag) + 1] if flag in argv else None




def test_the_documented_openxr_command_selects_the_isaacteleop_stack(compute):
    """README's teleop command, end to end."""

    lab_argv = _lab_argv(
        compute,
        "teleop.py",
        "--viz kit --device cpu dexbench_retail_v1_environment"
        " --task retrieve_beverage_from_clutter --anchor Fridge --teleop_device openxr"
        " --embodiment dexbench_rby1_wuji2 --arm_mode dual_arm",
    )

    assert "--teleop_device" not in lab_argv
    assert "--xr" in lab_argv
    assert _value_after(lab_argv, "--cloudxr_env") == "none"
    assert "--disable_external_cameras" in lab_argv
    assert _value_after(lab_argv, "--task") == "dexbench_retail_v1_environment"
    assert _value_after(lab_argv, "--external_callback") == CALLBACK
    assert lab_argv[:4] == ["--viz", "kit", "--device", "cpu"]


@pytest.mark.parametrize("given", ["", " --kit_args '--/app/foo=1'", " --kit_args=--/app/foo=1"])
def test_an_openxr_run_streams_textures_and_keeps_the_users_kit_args_last(compute, given):
    """Geometry streaming goes off for every Lab script; the XR experience turns texture streaming off and an OpenXR run
    turns it back on; any Kit setting the user gave comes last."""

    def settings(script, command):
        kit_args = [token for token in _lab_argv(compute, script, command) if token.startswith("--kit_args")]
        assert len(kit_args) == 1
        return kit_args[0].removeprefix("--kit_args=").split()

    openxr = settings("teleop.py", f"--viz kit{given} dexbench_retail_v1_environment --teleop_device openxr")
    assert openxr[:3] == ["--/persistent/UJITSO/geometry=false",
                          "--/rtx-transient/resourcemanager/enableTextureStreaming=true",
                          "--/rtx-transient/resourcemanager/texturestreaming/memoryBudget=0.25"]
    assert openxr[3:] == (["--/app/foo=1"] if given else [])
    keyboard = settings("teleop.py", f"--viz kit{given} dexbench_retail_v1_environment --teleop_device keyboard")
    assert keyboard == ["--/persistent/UJITSO/geometry=false", *(["--/app/foo=1"] if given else [])]


def test_the_environment_tail_never_reaches_the_lab_script(compute):
    """The tail's --task is the task variant and would be read as a gym id."""

    lab_argv, head, env_name, tail = compute(
        "teleop.py",
        shlex.split(
            "--viz kit dexbench_retail_v1_environment --task place_box_in_slot"
            " --anchor Gondola --teleop_device openxr"
        ),
    )

    assert env_name == "dexbench_retail_v1_environment"
    assert head == ["--viz", "kit"]
    assert "--anchor" in tail and "--anchor" not in lab_argv
    assert lab_argv.count("--task") == 1
    assert _value_after(lab_argv, "--task") == env_name


def test_recording_camera_scale_is_forwarded_to_the_environment_callback(compute):
    lab_argv, _, _, tail = compute(
        "record_demos.py",
        ["--viz", "none", "dexbench_hotel_fb_e4_environment", "--teleop_device", "openxr",
         "--enable_cameras", "--camera_scale", "1.0"],
    )
    assert _value_after(tail, "--camera_scale") == "1.0"
    assert "--camera_scale" not in lab_argv
    assert "--disable_external_cameras" not in lab_argv




def test_a_head_placed_device_is_stripped_from_the_lab_argv_but_kept_for_the_builder(compute):
    """The builder composes isaac_teleop from this flag, so the head keeps it."""

    lab_argv, head, _, _ = compute(
        "teleop.py",
        shlex.split("--teleop_device openxr --viz kit dexbench_retail_v1_environment --embodiment x"),
    )

    assert "--teleop_device" not in lab_argv
    assert head == ["--teleop_device", "openxr", "--viz", "kit"]
    assert "--xr" in lab_argv


@pytest.mark.parametrize(
    ("command", "expect_xr"),
    [
        ("--teleop_device keyboard dexbench_x_environment --teleop_device openxr", True),
        ("--teleop_device openxr dexbench_x_environment --teleop_device keyboard", False),
        ("dexbench_x_environment --teleop_device keyboard --teleop_device openxr", True),
        ("dexbench_x_environment --teleop_device=openxr", True),
    ],
)
def test_the_effective_device_follows_the_callbacks_merge_order(compute, command, expect_xr):
    lab_argv = _lab_argv(compute, "teleop.py", command)

    assert ("--xr" in lab_argv) is expect_xr
    if expect_xr:
        assert "--teleop_device" not in lab_argv
    else:
        assert _value_after(lab_argv, "--teleop_device") == "keyboard"
        assert "--cloudxr_env" not in lab_argv


def test_a_missing_device_fails_loudly(compute):
    """Silently it would run: an SE(3) keyboard against a full-joint action term."""

    with pytest.raises(SystemExit) as failure:
        compute("teleop.py", shlex.split("--viz kit dexbench_x_environment --embodiment x"))

    assert "--teleop_device" in str(failure.value)


def test_a_missing_environment_names_what_to_add(compute):
    with pytest.raises(SystemExit) as failure:
        compute("teleop.py", shlex.split("--viz kit --teleop_device openxr"))

    assert "positional" in str(failure.value)




def test_an_explicit_cloudxr_choice_wins(compute):
    """DexBench's sibling container is the default, not a mandate."""

    lab_argv = _lab_argv(
        compute, "teleop.py", "dexbench_x_environment --teleop_device openxr --cloudxr_env cloudxrjs"
    )

    assert _value_after(lab_argv, "--cloudxr_env") == "cloudxrjs"
    assert lab_argv.count("--cloudxr_env") == 1


def test_an_auto_launch_choice_is_forwarded_from_the_tail(compute):
    """The tail is otherwise never forwarded, so this needs lifting out of it."""

    lab_argv = _lab_argv(
        compute, "record_demos.py", "dexbench_x_environment --teleop_device openxr --no-auto_launch_cloudxr"
    )

    assert lab_argv.count("--no-auto_launch_cloudxr") == 1


def test_a_tail_placed_xr_does_not_suppress_the_injection(compute):
    """The tail never reaches the Lab side, so treating it as present would disable XR."""

    lab_argv = _lab_argv(compute, "teleop.py", "dexbench_x_environment --teleop_device openxr --xr")

    assert lab_argv.count("--xr") == 1


def test_a_head_placed_xr_is_not_duplicated(compute):
    lab_argv = _lab_argv(compute, "teleop.py", "--xr dexbench_x_environment --teleop_device openxr")

    assert lab_argv.count("--xr") == 1




@pytest.mark.parametrize("placement", ["head", "tail"])
def test_enable_cameras_turns_off_the_opt_out_from_either_side(compute, placement):
    """Cameras render by default upstream; --enable_cameras means keep them."""

    command = (
        "--enable_cameras dexbench_x_environment --teleop_device openxr"
        if placement == "head"
        else "dexbench_x_environment --teleop_device openxr --enable_cameras"
    )
    lab_argv = _lab_argv(compute, "record_demos.py", command)

    assert "--disable_external_cameras" not in lab_argv
    assert "--enable_cameras" not in lab_argv


def test_without_enable_cameras_the_lab_side_is_told_to_skip_them(compute):
    lab_argv = _lab_argv(compute, "record_demos.py", "dexbench_x_environment --teleop_device openxr")

    assert "--disable_external_cameras" in lab_argv




def test_scripts_without_the_teleop_flags_get_no_injections(compute):
    lab_argv = _lab_argv(
        compute,
        "replay_demos.py",
        "--dataset_file d.hdf5 dexbench_x_environment --teleop_device openxr --enable_cameras",
    )

    assert lab_argv == [
        "--dataset_file",
        "d.hdf5",
        "--kit_args=--/persistent/UJITSO/geometry=false",
        "--task",
        "dexbench_x_environment",
        "--external_callback",
        CALLBACK,
    ]



@pytest.mark.parametrize(
    ("head", "headless"),
    [
        (["--viz", "none", "--device", "cpu"], True),
        (["--viz=none"], True),
        (["--visualizer", "none"], True),
        (["--viz", "kit"], False),
        (["--viz", "none", "--viz", "kit"], False),
        (["--viz", "kit,newton"], False),
        (["--device", "cpu"], False),
        (["--viz"], False),
        (["--headless"], False),
    ],
)
def test_the_headless_replay_patch_follows_the_viz_flag(helper, head, headless):
    """IsaacLab has no --headless; `--viz none` is what a headless replay passes."""

    assert helper["_head_runs_headless"](head) is headless


def _declared_flags(path: Path) -> set[str]:
    """Flags a script declares, read from its ``add_argument`` calls."""

    text = path.read_text()
    flags: set[str] = set()
    for match in re.finditer(r"add_argument\(\s*((?:\"--[\w-]+\"\s*,\s*)*\"--[\w-]+\")", text):
        flags.update(re.findall(r"\"(--[\w-]+)\"", match.group(1)))
    return flags


def test_teleop_capable_scripts_match_the_upstream_flags(helper):
    """A pin bump that moves these flags must not pass silently."""

    if not ISAACLAB_ROOT.exists():
        pytest.skip("IsaacLab submodule is not checked out")

    declares = set()
    for script_name, relpath in helper["ISAACLAB_SCRIPT_RELPATHS"].items():
        flags = _declared_flags(ISAACLAB_ROOT / relpath)
        if all(flag in flags for flag in TELEOP_FLAGS):
            declares.add(script_name)
        else:
            assert not any(flag in flags for flag in TELEOP_FLAGS), (
                f"{script_name} declares only part of {TELEOP_FLAGS}; the wrapper assumes all or nothing"
            )

    assert declares == set(helper["TELEOP_CAPABLE_LAB_SCRIPTS"])


@pytest.mark.parametrize(
    "before, after",
    [
        ("--dataset_file d.hdf5", "--teleop_device openxr"),
        ("--dataset_file d.hdf5 dexbench_retail_v1_environment --teleop_device openxr", ""),
        ("--dataset_file d.hdf5 dexbench_hotel_fb_e4_environment", ""),
        ("--dataset_file d.hdf5", "--presets newton"),
        ("--dataset_file dexbench_r2.hdf5", "--teleop_device openxr"),
    ],
    ids=["documented", "environment-named-first", "another-environment", "presets-after", "dexbench-file-name"],
)
def test_an_experiment_run_keeps_its_physics_backend(compute, helper, before, after):
    """The Run's --presets reaches the launcher wherever the environment is named, or the line is refused."""

    command = f"{before} --experiment db_r2_collect.yaml --run tc1 {after}"
    if "hotel" in before or "--presets" in after:
        with pytest.raises(SystemExit):
            helper["expand_experiment_run"](shlex.split(command))
        return
    lab_argv, head, env_name, tail = compute("record_demos.py", helper["expand_experiment_run"](shlex.split(command)))
    assert _value_after(head, "--presets") == "newton"
    assert env_name == "dexbench_retail_v1_environment"
    assert "dexbench_retail_v1_environment" not in tail and "--presets" not in tail
    assert _value_after(tail, "--teleop_device") == "openxr"
    assert "--no-reset_sim_buffer_each_episode" in lab_argv




RECORD = (
    "--viz kit --device cuda:0 --dataset_file /tmp/d.hdf5"
    " dexbench_retail_v1_environment --task retrieve_beverage_from_clutter"
    " --teleop_device openxr --embodiment dexbench_rby1_wuji2"
)


@pytest.mark.parametrize("presets", ["", "--presets newton "], ids=["default", "presets-newton"])
def test_recording_skips_the_hard_sim_reset_without_reading_presets(compute, presets):
    """A hard reset re-finalizes the Newton model and drops the Lab actuator gains. Every DexBench
    environment runs on Newton, so the rule does not read --presets."""

    assert "--no-reset_sim_buffer_each_episode" in _lab_argv(compute, "record_demos.py", presets + RECORD)


def test_every_environment_the_wrapper_takes_builds_on_newton_only(helper):
    """The wrapper takes only a dexbench_*_environment name (_compute_lab_argv) and drops the Isaac Lab
    scripts' hard reset for it without reading the backend: record_demos.py's flag here, annotate_demos.py's
    sim.reset() once the environment is made. Every registered environment such a name reaches is a DexBench
    scene environment, which builds on Newton and refuses any other backend; an upstream one is not
    reachable through the wrapper."""
    pytest.importorskip("isaaclab_arena.assets.registries")
    from isaaclab_arena.assets.registries import EnvironmentRegistry

    from dexbench_arena.cli import ensure_dexbench_environments_registered
    from dexbench_arena.environments.manipulation.scene_environment_base import DexBenchSceneEnvironmentBase

    ensure_dexbench_environments_registered()
    registry = EnvironmentRegistry()
    taken = [name for name in registry.get_all_keys() if helper["_ENVIRONMENT_NAME"].fullmatch(name)]
    assert taken, "no DexBench environment registered"
    assert not [name for name in taken if not issubclass(registry.get_component_by_name(name), DexBenchSceneEnvironmentBase)]
    with pytest.raises(SystemExit, match="no DexBench environment named"):
        helper["_compute_lab_argv"]("record_demos.py", ["--dataset_file", "d.hdf5", "kitchen_pick_and_place"])


def test_the_sim_reset_is_left_alone_elsewhere(compute):
    assert "--no-reset_sim_buffer_each_episode" not in _lab_argv(
        compute, "teleop.py", RECORD.replace("--dataset_file /tmp/d.hdf5 ", "")
    )


def _lab_reset_choice(lab_argv):
    """What record_demos.py's BooleanOptionalAction (default True) would resolve to."""
    choice = True
    for token in lab_argv:
        if token == "--reset_sim_buffer_each_episode":
            choice = True
        elif token == "--no-reset_sim_buffer_each_episode":
            choice = False
    return choice


@pytest.mark.parametrize(
    "head_flag, tail_flag, expected",
    [
        (None, None, False),
        ("--no-reset_sim_buffer_each_episode", None, False),
        (None, "--no-reset_sim_buffer_each_episode", False),
        ("--reset_sim_buffer_each_episode", None, True),
        (None, "--reset_sim_buffer_each_episode", True),
        ("--no-reset_sim_buffer_each_episode", "--reset_sim_buffer_each_episode", True),
        ("--reset_sim_buffer_each_episode", "--no-reset_sim_buffer_each_episode", False),
    ],
)
def test_an_explicit_sim_reset_choice_wins_wherever_it_is_written(compute, head_flag, tail_flag, expected):
    head, env, tail = RECORD.partition("dexbench_retail_v1_environment")
    command = f"{head_flag or ''} {head}{env}{tail} {tail_flag or ''}"
    lab_argv, _, _, env_tail = compute("record_demos.py", shlex.split(command))

    assert _lab_reset_choice(lab_argv) is expected
    reset_flags = ("--reset_sim_buffer_each_episode", "--no-reset_sim_buffer_each_episode")
    assert sum(token in reset_flags for token in lab_argv) == 1
    assert not any(token in reset_flags for token in env_tail)


def test_record_demos_still_declares_the_sim_reset_flag(helper):
    if not ISAACLAB_ROOT.exists():
        pytest.skip("IsaacLab submodule is not checked out")

    flags = _declared_flags(ISAACLAB_ROOT / helper["ISAACLAB_SCRIPT_RELPATHS"]["record_demos.py"])
    assert "--reset_sim_buffer_each_episode" in flags




_COMMAND_START = re.compile(r"scripts/imitation_learning/(teleop|record_demos)\.py")


def _documented_commands() -> list[tuple[str, str, list[str]]]:
    """Every runnable ``teleop.py``/``record_demos.py`` command line in the docs."""

    found: list[tuple[str, str, list[str]]] = []
    paths = [REPO_ROOT / "README.md", *sorted((REPO_ROOT / "docs").rglob("*.md"))]
    for path in paths:
        lines = path.read_text().splitlines()
        for number, line in enumerate(lines, start=1):
            match = _COMMAND_START.search(line)
            if match is None:
                continue
            command = line[match.end() :]
            index = number
            while command.rstrip().endswith("\\") and index < len(lines):
                command = command.rstrip().rstrip("\\") + " " + lines[index].lstrip("> ").strip()
                index += 1
            command = command.replace("\\", " ").strip().rstrip("'\"")
            try:
                tokens = shlex.split(command)
            except ValueError:
                continue
            if not any(token.startswith("dexbench_") for token in tokens):
                continue
            found.append((f"{path.relative_to(REPO_ROOT)}:{number}", f"{match.group(1)}.py", tokens))
    return found


def test_the_docs_show_commands_that_survive_the_rewrite(compute, helper):
    """A documented command must reach the Lab parser without unknown flags."""

    if not ISAACLAB_ROOT.exists():
        pytest.skip("IsaacLab submodule is not checked out")

    commands = _documented_commands()
    assert commands, "no documented commands found -- the scraper stopped matching"

    known_to_arena = _declared_flags(ARENA_CLI) | _declared_flags(APP_LAUNCHER)
    problems = []
    for location, script_name, tokens in commands:
        expanded = helper["strip_collection_flags"](helper["expand_experiment_run"](tokens))[0]
        argv, _ = helper["_take_episode_video_dir"](script_name, expanded)
        _, head, _, _ = compute(script_name, argv)
        known_to_lab = _declared_flags(ISAACLAB_ROOT / helper["ISAACLAB_SCRIPT_RELPATHS"][script_name])
        for token in head:
            if not token.startswith("--"):
                continue
            flag = token.split("=", 1)[0]
            if flag.startswith("--no-"):
                flag = "--" + flag[len("--no-") :]
            if flag not in known_to_lab and flag not in known_to_arena:
                problems.append(f"{location}: {flag} is declared by neither parser")

    assert not problems, "\n".join(problems)




@pytest.mark.parametrize("script", ["teleop.py", "record_demos.py", "replay_demos.py", "generate_dataset.py"])
def test_waist_options_reach_the_environment_for_every_entrypoint(compute, script):
    options = ["--waist_ik", "--waist_ik_max_lead_rad", "0.075"]
    lab, _, _, tail = compute(script, [
        "--viz", "none", "dexbench_logistics_center_e2_environment",
        "--embodiment", "dexbench_rby1_wuji2", "--teleop_device", "openxr", *options,
    ])
    assert tail[-3:] == options
    assert "--waist_ik" not in lab and "--waist_ik_max_lead_rad" not in lab
