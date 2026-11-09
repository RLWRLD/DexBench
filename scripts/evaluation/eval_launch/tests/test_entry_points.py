"""Where the launcher is: the pods run it from the staged tree at ``k8s_backend.PACKAGE_PARENT``, and ``launch.sh``
puts that directory and its tree's ``source/dexbench_arena`` on the path itself, so neither depends on the caller's
PYTHONPATH or working directory."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from dexbench_arena import REPO_ROOT

from eval_launch import PACKAGE_DIR, k8s_backend


def test_the_pods_find_the_package_where_it_is_in_the_tree():
    assert PACKAGE_DIR == REPO_ROOT / k8s_backend.PACKAGE_PARENT / "eval_launch"


def test_launch_sh_runs_the_launcher_without_its_parent_on_the_path(tmp_path):
    path = [entry for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep)
            if entry and os.path.isabs(entry) and os.path.realpath(entry) not in
            (os.path.realpath(PACKAGE_DIR.parent), os.path.realpath(REPO_ROOT / "source/dexbench_arena"))]
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join(path), "ISAAC_PYTHON": sys.executable,
                   "TMPDIR": str(tmp_path)}
    result = subprocess.run([str(PACKAGE_DIR / "launch.sh"), str(PACKAGE_DIR / "examples/kubernetes.yaml"), "--plan-only"],
                            cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    assert "GPU-hours" in result.stdout
    assert not list(tmp_path.iterdir()), "the saved environment is removed"


def test_launch_sh_leaves_no_saved_environment_when_the_launcher_never_reads_it(tmp_path):
    """The saved environment holds the shell's tokens. An interpreter that fails before the launcher reads it (a wrong
    ISAAC_PYTHON, a crash at import) used to leave the file in TMPDIR; launch.sh now writes no file, and the interpreter
    finds the environment on descriptor 3."""
    temporary, seen = tmp_path / "tmp", tmp_path / "seen"
    temporary.mkdir()
    interpreter = tmp_path / "failing_python"
    interpreter.write_text(f'#!/bin/bash\nls -A "$TMPDIR" | grep -v "^eval-launch-status" > {seen}.files\n'
                           f'for arg; do last=$arg; done; tr "\\0" "\\n" < "$last" | grep -c "^MARKER=" > {seen}\nexit 7\n')
    interpreter.chmod(0o755)
    environment = {**os.environ, "ISAAC_PYTHON": str(interpreter), "TMPDIR": str(temporary), "MARKER": "x"}
    result = subprocess.run([str(PACKAGE_DIR / "launch.sh"), "config.yaml", "--status"], cwd=tmp_path, env=environment,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 7, result.stderr
    assert seen.read_text().split() == ["1"], "the interpreter reads the environment from the descriptor it is given"
    assert (tmp_path / "seen.files").read_text() == "", "the file was gone before the interpreter started"
    assert not list(temporary.iterdir())


def test_a_driver_that_cannot_import_ray_leaves_no_saved_environment(tmp_path):
    """--local on a Ray config execs the Ray driver, which imports Ray before it reads the servers' environment. Without
    Ray (the default for an image that does not install it) it exits at that import; the environment it was handed,
    tokens included, must not be left in TMPDIR."""
    import yaml

    temporary, no_ray = tmp_path / "tmp", tmp_path / "no_ray"
    temporary.mkdir()
    (no_ray / "ray").mkdir(parents=True)
    (no_ray / "ray/__init__.py").write_text("raise ModuleNotFoundError(\"No module named 'ray'\", name='ray')\n")
    config = yaml.safe_load((PACKAGE_DIR / "examples/single_task_local.yaml").read_text())
    config["backend"] = {"kind": "ray", "gpus": 1}
    config["output"] = str(tmp_path / "launch")
    path = tmp_path / "eval.yaml"
    path.write_text(yaml.safe_dump(config))
    environment = {**os.environ, "ISAAC_PYTHON": sys.executable, "TMPDIR": str(temporary),
                   "PYTHONPATH": os.pathsep.join(filter(None, (str(no_ray), os.environ.get("PYTHONPATH")))),
                   "FAKE_SECRET_TOKEN": "s3cret"}
    result = subprocess.run([str(PACKAGE_DIR / "launch.sh"), str(path), "--local"], cwd=tmp_path, env=environment,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode != 0 and "No module named 'ray'" in result.stderr, result.stderr
    assert not list(temporary.iterdir()), "the saved environment is removed"


def test_a_handed_over_environment_reads_back_from_its_descriptor_alone(tmp_path, monkeypatch):
    from eval_launch import server_env

    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(server_env.tempfile, "tempdir", None)
    path = server_env.handover({"HF_TOKEN": "s3cret", "PATH": "/bin"})
    assert path.startswith("/dev/fd/") and not list(tmp_path.iterdir()), "only the descriptor holds it"
    assert server_env.read(path) == {"HF_TOKEN": "s3cret", "PATH": "/bin"}


@pytest.mark.skipif(os.path.exists("/isaac-sim/python.sh"), reason="this host has Isaac Sim's Python, which launch.sh uses")
def test_launch_sh_plans_on_a_host_without_isaac_sim_with_python3(tmp_path):
    """Planning and rendering need no Isaac Sim: without ISAAC_PYTHON and without /isaac-sim/python.sh, launch.sh runs
    python3 (here the test's own interpreter, first on PATH)."""
    tools = tmp_path / "bin"
    tools.mkdir()
    (tools / "python3").write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    (tools / "python3").chmod(0o755)
    environment = {key: value for key, value in os.environ.items() if key != "ISAAC_PYTHON"}
    environment.update(PATH=f"{tools}{os.pathsep}{environment['PATH']}", TMPDIR=str(tmp_path))
    result = subprocess.run([str(PACKAGE_DIR / "launch.sh"), str(PACKAGE_DIR / "examples/kubernetes.yaml"), "--plan-only"],
                            cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0 and "GPU-hours" in result.stdout, result.stderr
    for mode in ("--local", "--score"):
        refused = subprocess.run([str(PACKAGE_DIR / "launch.sh"), str(PACKAGE_DIR / "examples/kubernetes.yaml"), mode],
                                 cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=60)
        assert refused.returncode == 2 and "runs in the simulator image" in refused.stderr, refused.stderr


def test_launch_sh_exits_with_the_launchers_status_through_isaac_sims_wrapper(tmp_path):
    """/isaac-sim/python.sh ends in `$python_exe "$@" || error_exit` (exit 1): every non-zero status became 1, so a
    smoke check that could not run (2) read like one that failed (1). launch.sh now exits with the status the launcher
    writes for it. A stand-in for python.sh that flattens the same way."""
    wrapper = tmp_path / "python.sh"
    wrapper.write_text(f'#!/bin/bash\n"{sys.executable}" "$@" || {{ echo "There was an error running python"; exit 1; }}\n')
    wrapper.chmod(0o755)
    config = tmp_path / "eval.yaml"
    config.write_text((PACKAGE_DIR / "examples/kubernetes.yaml").read_text())
    environment = {**os.environ, "ISAAC_PYTHON": str(wrapper), "TMPDIR": str(tmp_path)}
    result = subprocess.run([str(PACKAGE_DIR / "launch.sh"), str(config), "--smoke"], cwd=tmp_path, env=environment,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 2, (result.returncode, result.stderr)
    assert not list(tmp_path.glob("eval-launch-status*"))


def test_only_the_launcher_writes_its_status_file_not_a_child(tmp_path):
    """The smoke starts its driver as a child, whose own run_main used to write the status file too: a launcher that
    then died with an uncaught exception left the child's 0 there, and launch.sh exited 0. The reviewer's harness: a
    child run_main(lambda: 0), then the parent raises; the file must say 1."""
    status = tmp_path / "status"
    parent = (
        "import subprocess, sys\n"
        "from eval_launch import run_main\n"
        "def main():\n"
        "    subprocess.run([sys.executable, '-c', 'from eval_launch import run_main; run_main(lambda: 0)'], check=True)\n"
        "    raise RuntimeError('the launcher dies after its child')\n"
        "run_main(main)\n")
    environment = {**os.environ, "EVAL_LAUNCH_STATUS": str(status),
                   "PYTHONPATH": os.pathsep.join(filter(None, (str(PACKAGE_DIR.parent), os.environ.get("PYTHONPATH"))))}
    result = subprocess.run([sys.executable, "-c", parent], env=environment, capture_output=True, text=True, timeout=60)
    assert result.returncode == 1 and "the launcher dies" in result.stderr
    assert status.read_text().strip() == "1"
