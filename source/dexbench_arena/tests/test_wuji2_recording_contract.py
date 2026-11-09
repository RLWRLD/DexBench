# SPDX-License-Identifier: Apache-2.0
"""The joint names a Wuji2 recording carries, and the rerecord check that reads them."""

from __future__ import annotations

import json
from types import SimpleNamespace

import h5py
import pytest

pytest.importorskip("isaaclab.managers", reason="the recorder cfgs import Isaac Lab")

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg  # noqa: E402
from isaaclab.managers import RecorderManagerBaseCfg  # noqa: E402

from dexbench_arena.embodiments.hand_contact import install_hand_contact_recording_metadata  # noqa: E402
from dexbench_arena.embodiments.rby1_wuji2.constants import RBY1_WUJI2_JOINT_ACTION_NAMES  # noqa: E402
from dexbench_arena.embodiments.rby1_wuji2.recorders import JOINT_NAMES_KEY, configure_wuji2_recorders  # noqa: E402

from dexbench_arena.recording import rerecord_metadata as rerecord  # noqa: E402

TARGETS = list(RBY1_WUJI2_JOINT_ACTION_NAMES)
WHEELS = ["wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"]
PHYSX_ORDER = WHEELS + [name for pair in zip(TARGETS[::2], TARGETS[1::2]) for name in reversed(pair)]
NEWTON_ORDER = TARGETS + WHEELS


class Scene:
    def __init__(self, joint_names):
        self.sensors = {}
        self.robot = SimpleNamespace(joint_names=list(joint_names))

    def __getitem__(self, name):
        assert name == "robot"
        return self.robot


def _env(joint_names, *, recorder="demo", terms=(), embodiment="dexbench_rby1_wuji2"):
    """An env after startup: the embodiment's recorder cfg and startup events applied. ``recorder=None`` is
    render_demos_from_states.py's env, whose recorders are empty."""
    cfg = SimpleNamespace(demo_recorder_config=None, mimic_recorder_config=None, events=SimpleNamespace(),
                          dexbench_hand_contact_material={"physics_backend": "PhysxCfg"})
    configure_wuji2_recorders(cfg, embodiment=embodiment)
    cfg.recorders = (RecorderManagerBaseCfg() if recorder is None
                     else getattr(cfg, f"{recorder}_recorder_config", None) or ActionStateRecorderManagerCfg())
    env = SimpleNamespace(
        cfg=cfg,
        scene=Scene(joint_names),
        recorder_manager=SimpleNamespace(get_ep_meta=lambda: {"sim_args": {"dt": 0.01}}),
        action_manager=SimpleNamespace(
            active_terms=[f"term_{i}" for i in range(len(terms))],
            get_term=lambda name: terms[int(name.split("_")[1])],
        ),
    )
    install_hand_contact_recording_metadata(env)
    event = cfg.events.recording_joint_names
    event.func(env, None, **event.params)
    return env


@pytest.mark.parametrize("recorder", ["demo", "mimic"])
@pytest.mark.parametrize("order", [PHYSX_ORDER, NEWTON_ORDER], ids=["physx", "newton"])
def test_exported_metadata_names_the_live_robot_and_embodiment(order, recorder):
    meta = _env(order, recorder=recorder).recorder_manager.get_ep_meta()
    assert meta[JOINT_NAMES_KEY] == {
        "schema_version": 1, "robot": "rby1_wuji2", "profile": "rby1_wuji2",
        "embodiment": "dexbench_rby1_wuji2", "state_joint_names": order, "action_joint_names": TARGETS,
    }
    assert meta["sim_args"] == {"dt": 0.01} and meta["hand_contact_material"] == {"physics_backend": "PhysxCfg"}


def test_a_recorder_without_the_wuji2_targets_gets_no_names():
    assert JOINT_NAMES_KEY not in _env(NEWTON_ORDER, recorder="metrics").recorder_manager.get_ep_meta()


def _named(order, actions=TARGETS):
    return {JOINT_NAMES_KEY: {"state_joint_names": list(order), "action_joint_names": list(actions)}}


def test_named_joint_source_stays_strict():
    joint_term = SimpleNamespace(_joint_ids=[NEWTON_ORDER.index(name) for name in TARGETS])
    env = _env(NEWTON_ORDER, terms=[joint_term], embodiment="dexbench_rby1_wuji2_joint")
    rerecord.check_replay_joint_names(env, _named(NEWTON_ORDER), "NewtonCfg")
    with pytest.raises(ValueError, match="action joint order does not match rerecord action term"):
        rerecord.check_replay_joint_names(env, _named(NEWTON_ORDER, list(reversed(TARGETS))), "NewtonCfg")
    with pytest.raises(ValueError, match="state joint order"):
        rerecord.check_replay_joint_names(env, _named(PHYSX_ORDER), "NewtonCfg")


def test_named_ik_source_must_match_the_recorded_targets():
    env = _env(PHYSX_ORDER, terms=[SimpleNamespace()])
    rerecord.check_replay_joint_names(env, _named(PHYSX_ORDER), "PhysxCfg")
    with pytest.raises(ValueError, match="neither a direct joint action term"):
        rerecord.check_replay_joint_names(env, _named(PHYSX_ORDER, list(reversed(TARGETS))), "PhysxCfg")


def test_the_state_render_takes_a_named_ik_recording_and_still_checks_its_state_order():
    """A teleop session on the IK embodiment names Pink IK's joint targets, and the state
    render, which records nothing and copies them with their names, has no live targets to match them with: its check
    (``records_joint_targets=False``) compares only the state order it writes by index. The rerecord check refuses the
    same env, the console's "Source action joint names match neither ..."."""
    env = _env(NEWTON_ORDER, recorder=None, terms=[SimpleNamespace()])
    assert JOINT_NAMES_KEY not in env.recorder_manager.get_ep_meta()
    assert rerecord.check_replay_joint_names(env, _named(NEWTON_ORDER), "NewtonCfg",
                                             records_joint_targets=False) is None
    with pytest.raises(ValueError, match="state joint order"):
        rerecord.check_replay_joint_names(env, _named(PHYSX_ORDER), "NewtonCfg", records_joint_targets=False)
    with pytest.raises(ValueError, match="neither a direct joint action term"):
        rerecord.check_replay_joint_names(env, _named(NEWTON_ORDER), "NewtonCfg")


def test_unnamed_source_is_named_only_on_its_own_backend():
    env = _env(PHYSX_ORDER, terms=[SimpleNamespace()])
    assert rerecord.check_replay_joint_names(env, {"hand_contact_material": {"physics_backend": "PhysxCfg"}}, "PhysxCfg") is None
    with pytest.raises(ValueError, match="re-record it on that backend"):
        rerecord.check_replay_joint_names(env, {"dexbench_recording": {"physics_backend": "NewtonCfg"}}, "PhysxCfg")


def test_unnamed_source_without_a_backend_is_rerecorded_unnamed(tmp_path):
    source, output = tmp_path / "source.hdf5", tmp_path / "output.hdf5"
    source_metadata = {"hand_contact_material": {}}
    reason = rerecord.check_replay_joint_names(_env(PHYSX_ORDER, terms=[SimpleNamespace()]), source_metadata, "PhysxCfg")
    assert reason
    for path, env_args in ((source, source_metadata), (output, _env(PHYSX_ORDER).recorder_manager.get_ep_meta())):
        with h5py.File(path, "w") as f:
            f.create_group("data").attrs["env_args"] = json.dumps(env_args)
            f["data"].create_group("demo_0")
    rerecord.preserve_rerecord_metadata(source, output, source_metadata=source_metadata, calibration={},
                                        episodes={"demo_0": {}}, git_sha="0", runtime={"unnamed_joints": reason})
    with h5py.File(output) as f:
        assert JOINT_NAMES_KEY not in json.loads(f["data"].attrs["env_args"])
