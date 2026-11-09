# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""An alias of ``dexbench_arena_gr00t/policy/wuji2_gr00t_ego_left_modality.py``."""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag

from dexbench_arena_gr00t.policy.wuji2_gr00t_modalities import modality_config

register_modality_config(modality_config(("camera_ego_left",)), embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
