"""Shared ego-only Wuji2 modality for GR00T N1.7 training and rollout."""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag

from dexbench_arena_gr00t.policy.wuji2_gr00t_modalities import modality_config

register_modality_config(modality_config(), embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
