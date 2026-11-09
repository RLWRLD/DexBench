# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""The task/variation recorders survive the copy made by dataset collection entry points."""
from types import SimpleNamespace

import pytest

pytest.importorskip("isaaclab_arena")
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg  # noqa: E402
from dexbench_arena.environments.manipulation.scene_environment_base import _with_variation_samples  # noqa: E402
from dexbench_arena.tasks.task_obs_recorder import TaskObservationsRecorder, with_task_obs_recording  # noqa: E402
from dexbench_arena.variations.variation_samples_recorder import (  # noqa: E402
    VariationSamplesRecorder, with_declared_variation_samples,
)


def test_collection_copy_preserves_task_and_variation_terms_for_demo_and_mimic():
    shared = ActionStateRecorderManagerCfg()
    cfg = SimpleNamespace(demo_recorder_config=shared, mimic_recorder_config=shared, recorders=None)
    configured = with_declared_variation_samples(with_task_obs_recording(_with_variation_samples(None)))(cfg)
    assert configured.demo_recorder_config is configured.mimic_recorder_config
    for recorder in (configured.demo_recorder_config, configured.mimic_recorder_config):
        copied = recorder.copy()
        assert copied.record_task_obs.class_type is TaskObservationsRecorder
        assert copied.record_variation_samples.class_type is VariationSamplesRecorder
        assert copied.record_post_step_processed_actions is not None
    assert configured.recorders is None


def test_the_scene_default_still_leaves_the_variation_term_out_of_a_copy():
    """Only an environment that declares the term (DB-M2) records it through a copy; rerecord_demos.py's
    finalizer copies the source's variations/ and refuses an output demo that has its own."""
    cfg = SimpleNamespace(demo_recorder_config=ActionStateRecorderManagerCfg(), mimic_recorder_config=None,
                          recorders=None)
    configured = _with_variation_samples(None)(cfg)
    assert configured.demo_recorder_config.record_variation_samples.class_type is VariationSamplesRecorder
    assert not hasattr(configured.demo_recorder_config.copy(), "record_variation_samples")

