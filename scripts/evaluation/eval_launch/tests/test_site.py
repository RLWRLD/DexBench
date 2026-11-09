from __future__ import annotations

import pytest
import yaml

from eval_launch import k8s_backend
from eval_launch.plan import load_config

PROFILE = {"name": "example-cluster", "image": "registry.example/arena@sha256:abc",
           "k8s": {"namespace": "evals", "volume_claim": "shared", "volume_mount": "/data", "cpu": 11.5, "memory": "90Gi",
                   "shm_size": "16Gi", "labels": {"cost-center": "evals", "team": "a"},
                   "required_labels": {"cost-center": "evals"},
                   "tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}],
                   "annotations": {"provenance/kind": "eval"}, "prepull": False}}


def _config(tmp_path, k8s=None):
    (tmp_path / "site.yaml").write_text(yaml.safe_dump(PROFILE))
    config = {"model": {"key": "gr00t", "checkpoint": "/ckpt"}, "tasks": ["r1"], "output": str(tmp_path / "out"),
              "backend": {"kind": "k8s", "gpus": 2, "site": "site.yaml",
                          "k8s": {"job_name": "me-eval", "tree_dir": "/data/t", "commit": "c" * 40, "labels": {"team": "b"},
                                  **(k8s or {})}}}
    path = tmp_path / "eval.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_a_site_profile_fills_in_the_cluster_and_the_config_wins_where_it_speaks(tmp_path):
    config = load_config(_config(tmp_path, {"cpu": 23}))
    backend, k8s = config["backend"], config["backend"]["k8s"]
    assert backend["image"] == PROFILE["image"] and k8s["namespace"] == "evals" and k8s["prepull"] is False
    assert k8s["cpu"] == 23, "the config's own value wins"
    assert k8s["labels"] == {"cost-center": "evals", "team": "b"}, "mappings merge key by key"
    assert backend["site"]["name"] == "example-cluster" and len(backend["site"]["sha256"]) == 64
    job = k8s_backend.render_job(config, "/x/eval.yaml")
    assert job["metadata"]["namespace"] == "evals" and job["metadata"]["annotations"] == {"provenance/kind": "eval"}
    written = tmp_path / "written.yaml"
    written.write_text(yaml.safe_dump(config))
    (tmp_path / "site.yaml").unlink()
    assert load_config(written) == config


def test_a_profiles_label_rule_holds_against_the_config(tmp_path):
    config = load_config(_config(tmp_path, {"labels": {"cost-center": "mine"}}))
    with pytest.raises(ValueError, match="required_labels"):
        k8s_backend.render_job(config, "/x/eval.yaml")


def test_a_profile_that_sets_a_launchs_own_field_is_refused(tmp_path):
    path = _config(tmp_path)
    PROFILE_BAD = {**PROFILE, "k8s": {**PROFILE["k8s"], "job_name": "theirs"}}
    (tmp_path / "site.yaml").write_text(yaml.safe_dump(PROFILE_BAD))
    with pytest.raises(ValueError, match="job_name"):
        load_config(path)


def test_a_relative_profile_is_found_from_a_copied_config_and_a_missing_one_names_where_it_looked(tmp_path, monkeypatch):
    """A config copied out of the package (the guide's configs/local/) names its profile relative to the package: it is
    found there. A profile in neither place is refused with both paths, before anything else."""
    from eval_launch import site

    monkeypatch.setattr(site, "PACKAGE_DIR", tmp_path / "package")
    (tmp_path / "package/profiles").mkdir(parents=True)
    (tmp_path / "package/profiles/site.yaml").write_text(yaml.safe_dump(PROFILE))
    config = yaml.safe_load(_config(tmp_path).read_text())
    config["backend"]["site"] = "profiles/site.yaml"
    (tmp_path / "local").mkdir()
    copied = tmp_path / "local/eval.yaml"
    copied.write_text(yaml.safe_dump(config))
    loaded = load_config(copied)
    assert loaded["backend"]["k8s"]["namespace"] == "evals"
    assert loaded["backend"]["site"]["profile"] == str((tmp_path / "package/profiles/site.yaml").resolve())
    config["backend"]["site"] = "profiles/missing.yaml"
    copied.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="local/profiles/missing.yaml or .*package/profiles/missing.yaml"):
        load_config(copied)


@pytest.mark.parametrize("k8s, message", [({"prepull": "false"}, "prepull is 'false' .*bool"),
                                           ({"labels": ["a"]}, "labels is .*mapping"),
                                           ({"ray": {"head_tolerations": None}}, "head_tolerations is None"),
                                           ({"retries_per_worker": True}, "retries_per_worker is True")])
def test_a_profile_value_of_the_wrong_type_is_refused_by_name(tmp_path, k8s, message):
    """A quoted "false" would render the pre-pull it means to turn off; a list of labels or a null toleration list would
    be a traceback in the renderer. Each is a ValueError naming the key."""
    path = _config(tmp_path)
    (tmp_path / "site.yaml").write_text(yaml.safe_dump({**PROFILE, "k8s": {**PROFILE["k8s"], **k8s}}))
    with pytest.raises(ValueError, match=message):
        load_config(path)


@pytest.mark.parametrize("example", ["kubernetes.yaml", "ray_kubernetes.yaml"])
def test_an_example_copied_to_configs_local_still_finds_its_profile(tmp_path, example):
    """The guide's workflow: copy an example to configs/local/ and edit it there."""
    from eval_launch import PACKAGE_DIR

    copied = tmp_path / "configs/local" / example
    copied.parent.mkdir(parents=True)
    copied.write_bytes((PACKAGE_DIR / "examples" / example).read_bytes())
    assert load_config(copied)["backend"]["site"]["profile"] == str(PACKAGE_DIR / "examples/sites/kubernetes.yaml")
    assert load_config(PACKAGE_DIR / "examples" / example)["backend"]["site"]["name"]


def test_a_config_or_profile_that_is_not_yaml_is_a_usage_error(tmp_path, capsys):
    from eval_launch import launch

    path = _config(tmp_path)
    (tmp_path / "site.yaml").write_text("k8s: {namespace: evals\n")
    with pytest.raises(SystemExit) as stopped:
        launch.main([str(path), "--plan-only"])
    assert stopped.value.code == 2 and "site.yaml" in capsys.readouterr().err
