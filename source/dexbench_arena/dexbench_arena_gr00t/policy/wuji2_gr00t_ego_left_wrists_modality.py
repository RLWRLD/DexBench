"""Wuji2 modality with the left ego camera and both wrist cameras, for checkpoints trained on those three views."""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag

from dexbench_arena.embodiments.rby1_wuji2.camera_contract import WRIST_CAMERAS
from dexbench_arena_gr00t.policy.wuji2_gr00t_modalities import modality_config

register_modality_config(
    modality_config(("camera_ego_left", *WRIST_CAMERAS)), embodiment_tag=EmbodimentTag.NEW_EMBODIMENT
)
