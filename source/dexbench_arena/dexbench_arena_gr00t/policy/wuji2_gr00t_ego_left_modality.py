"""Wuji2 modality with the left ego camera only, for checkpoints trained on that one view."""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag

from dexbench_arena_gr00t.policy.wuji2_gr00t_modalities import modality_config

register_modality_config(modality_config(("camera_ego_left",)), embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
