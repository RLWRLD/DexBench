"""GA four-camera Wuji2 modality, for checkpoints trained with bilateral wrists."""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag

from dexbench_arena.embodiments.rby1_wuji2.camera_contract import CAMERA_OBSERVATIONS
from dexbench_arena_gr00t.policy.wuji2_gr00t_modalities import modality_config

register_modality_config(modality_config(tuple(CAMERA_OBSERVATIONS)), embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
