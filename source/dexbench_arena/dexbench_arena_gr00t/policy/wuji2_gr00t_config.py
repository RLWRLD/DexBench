"""Emit the N1.7 Wuji2 GR00T contract for task-specific manual YAML review."""

from pathlib import Path
import yaml

from dexbench_arena.embodiments.rby1_wuji2.constants import (
    RBY1_WUJI2_JOINT_ACTION_NAMES,
    RBY1_WUJI2_WHEEL_JOINT_NAMES,
)
from dexbench_arena_gr00t.policy.wuji2_gr00t_contract import ACTION_HORIZON, TARGET_IMAGE_SIZE
from dexbench_arena.embodiments.rby1_wuji2.joint_groups import WUJI2_JOINT_GROUPS
from dexbench_arena.embodiments.rby1_wuji2.camera_contract import (
    CAMERA_OBSERVATIONS,
    EGO_CAMERAS,
    WRIST_CAMERAS,
    camera_mapping,
)

MODALITY_MODULES = {
    tuple(EGO_CAMERAS): "wuji2_gr00t_modality.py",
    ("camera_ego_left",): "wuji2_gr00t_ego_left_modality.py",
    ("camera_ego_left", *WRIST_CAMERAS): "wuji2_gr00t_ego_left_wrists_modality.py",
    tuple(CAMERA_OBSERVATIONS): "wuji2_gr00t_wrist_modality.py",
}
"""The module beside this one that registers each camera set, keyed by the checkpoint's video keys in order."""

LEGACY_MODALITY_DIR = Path(__file__).parents[2] / "dexbench_arena" / "policy"
"""``dexbench_arena/policy/``: a file there of a modality module's name registers the same modality config, so a
contract that names a path there works too."""


def _same_contract(name: str, existing: str, text: str) -> bool:
    """Whether ``existing`` is the contract ``text`` describes: the same text, or a ``policy.yaml`` that differs only in
    naming the same module in :data:`LEGACY_MODALITY_DIR` of this checkout."""
    if existing == text:
        return True
    if name != "policy.yaml":
        return False
    old, new = yaml.safe_load(existing), yaml.safe_load(text)
    if not isinstance(old, dict):
        return False
    path = new["modality_config_path"]
    legacy = str(LEGACY_MODALITY_DIR / Path(path).name)
    return old.get("modality_config_path") == legacy and {**old, "modality_config_path": path} == new


def write_config(directory: Path, *, camera_keys=tuple(EGO_CAMERAS)) -> Path:
    cameras = camera_mapping(camera_keys)
    if tuple(cameras) not in MODALITY_MODULES:
        raise ValueError(f"No GR00T modality module for cameras {list(cameras)}; "
                         f"the supported sets are {[list(keys) for keys in MODALITY_MODULES]}")
    directory.mkdir(parents=True, exist_ok=True)
    directory = directory.resolve()
    action_names = list(RBY1_WUJI2_JOINT_ACTION_NAMES)
    state_names = action_names + list(RBY1_WUJI2_WHEEL_JOINT_NAMES)
    documents = {
        "policy_joints.yaml": {
            "joints": {k: list(v) for k, v in WUJI2_JOINT_GROUPS.items()}
        },
        "action_joints.yaml": {"joints": {n: i for i, n in enumerate(action_names)}},
        "state_joints.yaml": {"joints": {n: i for i, n in enumerate(state_names)}},
        "policy.yaml": {
            "embodiment_tag": "NEW_EMBODIMENT",
            "action_horizon": ACTION_HORIZON,
            "action_chunk_length": ACTION_HORIZON,
            "target_image_size": list(TARGET_IMAGE_SIZE),
            "pov_cam_name_sim": list(cameras.values()),
            "modality_config_path": str(Path(__file__).with_name(MODALITY_MODULES[tuple(cameras)])),
            "policy_joints_config_path": str(directory / "policy_joints.yaml"),
            "action_joints_config_path": str(directory / "action_joints.yaml"),
            "state_joints_config_path": str(directory / "state_joints.yaml"),
        },
    }
    texts = {name: yaml.safe_dump(document, sort_keys=False) for name, document in documents.items()}
    for name, text in texts.items():
        if (directory / name).exists() and not _same_contract(name, (directory / name).read_text(), text):
            raise FileExistsError(f"{directory / name} exists and differs from this contract; move it aside to write a new one")
    for name, text in texts.items():
        (directory / name).write_text(text)
    return directory / "policy.yaml"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    camera_set = parser.add_mutually_exclusive_group()
    camera_set.add_argument(
        "--cameras", nargs="+", choices=list(CAMERA_OBSERVATIONS), default=list(EGO_CAMERAS),
        help="the checkpoint's video keys, in its order; one of: "
        + "; ".join(" ".join(keys) for keys in MODALITY_MODULES),
    )
    camera_set.add_argument("--wrist-cameras", action="store_true", help="GA four-camera checkpoint contract")
    args = parser.parse_args()
    camera_keys = tuple(CAMERA_OBSERVATIONS) if args.wrist_cameras else tuple(args.cameras)
    print(write_config(args.output, camera_keys=camera_keys))
