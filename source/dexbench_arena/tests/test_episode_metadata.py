# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Episode metadata contributors: one entry per key, chained with other wrappers, copied on export."""

from types import SimpleNamespace


def test_contributors_chain_once_per_key():
    from dexbench_arena.embodiments.episode_metadata import install_episode_metadata

    recorder = SimpleNamespace(get_ep_meta=lambda: {"sim_args": {"dt": 0.01}})
    env = SimpleNamespace(recorder_manager=recorder)
    install_episode_metadata(env, "fingertip_contact", {"body_names": ["a", "b"]})

    class Other:
        def __init__(self, inner):
            self._get_ep_meta = inner

        def __call__(self):
            return {**self._get_ep_meta(), "other": 1}

    recorder.get_ep_meta = Other(recorder.get_ep_meta)
    install_episode_metadata(env, "fingertip_contact", {"body_names": ["x"]})
    assert isinstance(recorder.get_ep_meta, Other)
    live = {"static_friction": 0.5}
    install_episode_metadata(env, "hand_contact_material", lambda: live)
    live["static_friction"] = 6.0
    meta = recorder.get_ep_meta()
    assert meta["sim_args"] == {"dt": 0.01} and meta["other"] == 1
    assert meta["fingertip_contact"] == {"body_names": ["a", "b"]}
    assert meta["hand_contact_material"] == {"static_friction": 6.0}
    meta["fingertip_contact"]["body_names"].append("c")
    assert recorder.get_ep_meta()["fingertip_contact"] == {"body_names": ["a", "b"]}
