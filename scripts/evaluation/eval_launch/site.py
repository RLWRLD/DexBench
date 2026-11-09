"""Site profiles: a cluster's settings in a file of their own, so a launch config names the model, the cells and the
GPUs and points at its site (``backend.site: <profile.yaml>``) instead of repeating the cluster's namespace, image,
storage, node rules, per-role pod sizes, annotations and security contexts.

A profile holds only keys of :data:`SITE_KEYS` (``image``, ``env``, ``k8s`` with the keys of :data:`SITE_K8S_KEYS`),
each the field of the same name under ``backend``. The config's own values win: a profile fills in what the config
does not say, mappings (labels, annotations, env, ``k8s.ray``) merge key by key, lists and scalars are the config's when
it has them. A relative profile path is looked for in the config's directory, then in this package's (so a config
copied elsewhere, e.g. to ``configs/local/``, still finds a profile it names from the package); neither, and the
config is refused naming both. Once applied, ``backend.site`` records which profile it was (absolute path, name, sha256): ``--k8s-job`` writes that merged config to the shared volume, so the pods
read the cluster's settings without the profile file.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from eval_launch import PACKAGE_DIR


SITE_KEYS = {"name", "note", "image", "env", "k8s"}
SITE_K8S_KEYS = {"namespace", "volume_claim", "volume_mount", "node_selector", "tolerations", "labels", "required_labels",
                 "annotations",
                 "security_context", "pod_security_context", "prepull", "cpu", "memory", "shm_size", "python",
                 "retries_per_worker", "termination_grace_s", "deadline_h", "secret_env", "env", "ray"}
"""What a cluster decides. Not here: what a launch decides (``job_name``, ``tree_dir``, ``commit``, ``output``, GPUs)."""

_TEXT, _NUMBER, _QUANTITY = (str,), (int, float), (str, int, float)
SITE_TYPES: dict[str, tuple[type, ...]] = {
    "name": _TEXT, "note": _TEXT, "image": _TEXT, "env": (Mapping,), "k8s": (Mapping,)}
SITE_K8S_TYPES: dict[str, tuple[type, ...]] = {
    "namespace": _TEXT, "volume_claim": _TEXT, "volume_mount": _TEXT, "node_selector": (Mapping,), "tolerations": (list,),
    "labels": (Mapping,), "required_labels": (Mapping,), "annotations": (Mapping,), "security_context": (Mapping,),
    "pod_security_context": (Mapping,), "prepull": (bool,), "cpu": _QUANTITY, "memory": _QUANTITY, "shm_size": _QUANTITY,
    "python": _TEXT, "retries_per_worker": (int,), "termination_grace_s": _NUMBER, "deadline_h": _NUMBER,
    "secret_env": (Mapping,), "env": (Mapping,), "ray": (Mapping,)}
"""The type each profile value must have. A quoted ``"false"`` is a string, and a string is not a ``prepull`` flag;
``bool`` counts as none of the numbers."""


def _check_types(values: Mapping[str, Any], types: Mapping[str, tuple[type, ...]], where: str) -> None:
    for key, value in values.items():
        expected = types.get(key)
        if expected is None:
            continue
        wrong_bool = isinstance(value, bool) and bool not in expected
        if wrong_bool or not isinstance(value, expected):
            names = " or ".join({Mapping: "a mapping", int: "an int", float: "a float", str: "a string",
                                 bool: "a bool", list: "a list"}[kind] for kind in expected)
            raise ValueError(f"{where}{key} is {value!r} ({type(value).__name__}); it takes {names} (a key left empty "
                             "is null, not an empty mapping: leave it out instead)")


def _merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """``base`` (the profile) under ``override`` (the config): mappings merge, anything else is the config's."""
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_profile(path: Path) -> dict[str, Any]:
    """The profile at ``path``, refused if it names a key no cluster decides."""
    try:
        profile = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as error:
        raise ValueError(f"site profile {path} is not YAML: {error}") from None
    if not isinstance(profile, Mapping):
        raise ValueError(f"site profile {path}: a mapping of {sorted(SITE_KEYS)}, not {type(profile).__name__}")
    unknown = sorted(set(profile) - SITE_KEYS)
    if unknown:
        raise ValueError(f"site profile {path}: unknown fields {unknown}; a profile has {sorted(SITE_KEYS)}")
    _check_types(profile, SITE_TYPES, f"site profile {path}: ")
    unknown = sorted(set(profile.get("k8s") or {}) - SITE_K8S_KEYS)
    if unknown:
        raise ValueError(f"site profile {path}: k8s has {sorted(SITE_K8S_KEYS)}, not {unknown} (a launch's own "
                         "fields, such as job_name, tree_dir, commit, stay in the config)")
    _check_types(profile.get("k8s") or {}, SITE_K8S_TYPES, f"site profile {path}: k8s.")
    for key, value in (profile.get("k8s", {}).get("ray") or {}).items():
        if key == "head_tolerations" and not isinstance(value, list):
            raise ValueError(f"site profile {path}: k8s.ray.head_tolerations is {value!r}; it takes a list ([] for none)")
    return profile


def _locate(site: Path, config_path: Path) -> Path:
    """The profile ``backend.site`` names: as given when absolute; else the first of the config's directory and this
    package's directory that holds it."""
    if site.is_absolute():
        candidates = [site]
    else:
        candidates = [config_path.parent / site, PACKAGE_DIR / site]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise ValueError(f"backend.site {site}: no such profile at {' or '.join(str(path) for path in candidates)}")


def apply_site(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    """``config`` with its ``backend.site`` profile applied (a config without one, or with one already applied, as is)."""
    backend = config.get("backend") or {}
    site = backend.get("site")
    if site is None or isinstance(site, Mapping):
        return config
    path = _locate(Path(str(site)), config_path)
    profile = load_profile(path)
    merged = _merge({key: value for key, value in profile.items() if key not in ("name", "note")},
                    {key: value for key, value in backend.items() if key != "site"})
    merged["site"] = {"profile": str(path), "name": str(profile.get("name", path.stem)),
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return {**config, "backend": merged}
