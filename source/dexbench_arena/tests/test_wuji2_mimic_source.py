# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Generation refuses achieved-pose labels only on Wuji2 joint-space sources, without Kit."""

from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from dexbench_arena import isaaclab_interop as interop
from dexbench_arena.embodiments.rby1_wuji2.mimic_source import validate_mimic_source_dataset


@pytest.mark.parametrize("action_dim", [62, 56, 54])
@pytest.mark.parametrize("different_target", [None, "left", "right"])
def test_generation_checks_stored_joint_space_targets(tmp_path, monkeypatch, action_dim, different_target):
    path = tmp_path / "annotated.hdf5"
    with h5py.File(path, "w") as handle:
        for index in range(2):
            demo = handle.create_group(f"data/demo_{index}")
            demo.create_dataset("actions", data=np.zeros((3, action_dim)))
            info = demo.create_group("obs/datagen_info")
            for side in ("left", "right"):
                pose = np.tile(np.eye(4), (3, 1, 1))
                info.create_dataset(f"eef_pose/{side}", data=pose)
                if index == 0 or side == different_target:
                    pose[-1, 0, 3] += 0.001
                info.create_dataset(f"target_eef_pose/{side}", data=pose)

    class MimicEnv:
        validate_mimic_source_dataset = staticmethod(validate_mimic_source_dataset)

    embodiment = SimpleNamespace(get_mimic_env=lambda: MimicEnv)
    arena = SimpleNamespace(embodiment=embodiment)
    monkeypatch.setattr(interop, "HEAD_ARGV", ["--input_file", str(path)])
    monkeypatch.setattr(interop, "IM_SCRIPT_NAME", "generate_dataset.py")
    if action_dim in (62, 56) and different_target is None:
        with pytest.raises(ValueError, match="demo_1.*target_eef_pose equal to eef_pose") as error:
            interop._check_source_dataset(arena)
        assert "the maintainers' boundary extraction, then scripts/imitation_learning/annotate_demos.py" in str(error.value)
        assert "scripts/imitation_learning/annotate_demos.py --auto --mimic_boundaries" in str(error.value)
    else:
        interop._check_source_dataset(arena)

    monkeypatch.setattr(interop, "IM_SCRIPT_NAME", "annotate_demos.py")
    interop._check_source_dataset(arena)
    monkeypatch.setattr(interop, "IM_SCRIPT_NAME", "generate_dataset.py")
    embodiment.get_mimic_env = lambda: object
    interop._check_source_dataset(arena)
