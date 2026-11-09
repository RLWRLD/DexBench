"""The download entry point selects its repository's pin and preserves explicit overrides."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
LOCK = json.loads((ROOT / "reproducibility/datasets.lock.json").read_text())
PRIMARY = LOCK["repo_id"]
MULTITASK = LOCK["multitask"]["repo_id"]
PATTERN = LOCK["multitask"]["path"] + "/66d_320x180_4cam/**"


@pytest.mark.parametrize("args,overrides,has_multitask,repo,revision", [
    ([], {}, True, PRIMARY, LOCK["revision"]),
    (["--repo-id", MULTITASK], {}, True, MULTITASK, LOCK["multitask"]["revision"]),
    ([], {"DEXBENCH_DATA_REPO_ID": MULTITASK}, True, MULTITASK, LOCK["multitask"]["revision"]),
    (["--repo-id", PRIMARY], {"DEXBENCH_DATA_REPO_ID": MULTITASK}, True, PRIMARY, LOCK["revision"]),
    (["--repo-id", MULTITASK, "--revision", "explicit"], {"DEXBENCH_DATA_REVISION": "from-env"}, True, MULTITASK, "explicit"),
    (["--repo-id", MULTITASK], {"DEXBENCH_DATA_REVISION": "from-env"}, True, MULTITASK, "from-env"),
    (["--repo-id", "example/unlocked"], {}, True, "example/unlocked", None),
    ([], {}, False, PRIMARY, LOCK["revision"]),
    (["--repo-id", MULTITASK], {}, False, MULTITASK, None),
])
def test_download_selects_the_requested_repository_pin(tmp_path, args, overrides, has_multitask, repo, revision):
    checkout = tmp_path / "checkout"
    (checkout / "scripts").mkdir(parents=True)
    (checkout / "reproducibility").mkdir()
    script = checkout / "scripts/download_datasets.sh"
    shutil.copy2(ROOT / "scripts/download_datasets.sh", script)
    lock = dict(LOCK)
    if not has_multitask:
        lock.pop("multitask")
    (checkout / "reproducibility/datasets.lock.json").write_text(json.dumps(lock))

    binary = tmp_path / "bin"
    binary.mkdir()
    hf = binary / "hf"
    hf.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "with open(os.environ['DOWNLOAD_TEST_ARGUMENTS'], 'w') as stream:\n"
        "    json.dump(sys.argv[1:], stream)\n"
    )
    hf.chmod(0o755)
    output = tmp_path / "hf-arguments.json"
    env = {key: value for key, value in os.environ.items()
           if key not in {"DEXBENCH_DATA_REPO_ID", "DEXBENCH_DATA_REVISION", "DEXBENCH_DATASETS_ROOT"}}
    env.update({"PATH": str(binary) + os.pathsep + os.environ["PATH"],
                "DEXBENCH_DATASETS_ROOT": str(tmp_path / "datasets"),
                "DOWNLOAD_TEST_ARGUMENTS": str(output), **overrides})
    subprocess.run(["bash", str(script), *args, "--include", PATTERN], env=env,
                   check=True, capture_output=True, text=True)

    expected = ["download", repo, "--repo-type", "dataset"]
    if revision is not None:
        expected.extend(["--revision", revision])
    expected.extend(["--local-dir", str(tmp_path / "datasets/dexbench_arena"), "--include", PATTERN])
    assert json.loads(output.read_text()) == expected


def _named_paths(node):
    if isinstance(node, dict):
        for value in node.values():
            yield from _named_paths(value)
    elif isinstance(node, list):
        for value in node:
            yield from _named_paths(value)
    elif isinstance(node, str):
        yield node


def test_every_start_state_set_a_cell_reads_is_pinned_for_eval_conditions():
    """A cell file names its rows and, for some tasks, a set of start states beside them (DB-R1's layouts, DB-M5's
    piles). --eval-conditions installs only what benchmark/frozen pins, so a set no pin names is one the release
    never ships and the cell cannot start from."""
    yaml = pytest.importorskip("yaml")
    from dexbench_arena.benchmark.frozen import INPUTS_PIN
    from dexbench_arena.benchmark.protocol import CONDITIONS_ROOT

    configs = ROOT / "scripts/evaluation/experiment_configs/benchmark"
    manifest = json.loads((configs / "benchmark_manifest.json").read_text())
    files = {cell["eval"] for task in manifest["tasks"].values() for cell in task["cells"].values()}
    files |= {task["sensitivity"]["eval"] for task in manifest["tasks"].values() if task.get("sensitivity")}
    named = {os.path.normpath(value) for name in files for value in _named_paths(yaml.safe_load((configs / name).read_text()))
             if value.startswith(CONDITIONS_ROOT + "/") and not value.endswith(".jsonl")}
    assert named == set(json.loads(INPUTS_PIN.read_text()))


def test_eval_conditions_refuses_a_missing_or_changed_start_state_set(tmp_path):
    from dexbench_arena.benchmark.frozen import input_problems, tree_digest

    layouts = tmp_path / "db_r1_layouts/eval"
    layouts.mkdir(parents=True)
    (layouts / "plum_pile_evaluation_s10000.json").write_text("{}")
    sha256, files = tree_digest(layouts)
    pins = {"outputs/db_conditions/db_r1_layouts/eval": {"sha256": sha256, "files": files}}
    assert input_problems(tmp_path, pins) == []

    (layouts / "plum_pile_evaluation_s10001.json").write_text("{}")
    assert [problem.split(":")[0] for problem in input_problems(tmp_path, pins)] == list(pins)
    assert input_problems(tmp_path / "elsewhere", pins) == [f"{next(iter(pins))}: not in {tmp_path / 'elsewhere'}"]
