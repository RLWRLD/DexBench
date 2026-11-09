"""The server kit against what the worker expects of a policy server: started from the example config's recipe by
``eval_worker.PolicyServer`` (its token file, its ready line), it answers the worker's token and refuses any other."""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from dexbench_arena import REPO_ROOT

from eval_launch import PACKAGE_DIR, eval_worker
from eval_launch.plan import load_config

pytest.importorskip("PIL", reason="the server kit needs Pillow")


def _get(port, path, token):
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, None


def test_the_worker_starts_the_kit_from_the_example_recipe_and_reaches_it_with_its_token(tmp_path):
    recipe = load_config(PACKAGE_DIR / "examples/own_policy_local.yaml")["servers"]["recipe"]
    recipe = {**recipe, "command": recipe["command"].replace("/isaac-sim/python.sh", sys.executable)}
    server = eval_worker.PolicyServer(recipe, 5893, tmp_path / "logs", ready_timeout_s=60.0)
    try:
        server.ensure(str(tmp_path))
        status, info = _get(5893, "/info", server.token)
        assert status == 200 and len(info["joint_names"]) == 62 and info["action_horizon"] >= 16
        names = REPO_ROOT / "source/dexbench_policy_server/dexbench_policy_server/rby1_wuji2_joint_names.json"
        assert info["joint_names"] == json.loads(names.read_text())
        assert _get(5893, "/info", "not-the-workers")[0] == 401
    finally:
        server.stop()


@pytest.mark.parametrize("example", ["own_policy_local.yaml", "ray_local.yaml"])
def test_the_example_recipe_runs_from_a_relative_output_and_checkpoint(tmp_path, monkeypatch, example):
    """The examples' output is relative (outputs/eval/...) and their command cds into the kit before it reads
    {token_file}: the worker hands the recipe absolute paths, so the server finds its token whatever the cwd."""
    recipe = load_config(PACKAGE_DIR / "examples" / example)["servers"]["recipe"]
    recipe = {**recipe, "command": recipe["command"].replace("/isaac-sim/python.sh", sys.executable)}
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ckpt").mkdir()
    server = eval_worker.PolicyServer(recipe, 5894, Path("outputs/eval/run/logs/s0"), ready_timeout_s=60.0)
    try:
        server.ensure("ckpt")
        assert server.fields["token_file"].startswith("/") and str(tmp_path) in server.fields["token_file"]
        assert server.fields["checkpoint"] == str(tmp_path / "ckpt")
        assert _get(5894, "/info", server.token)[0] == 200
    finally:
        server.stop()
