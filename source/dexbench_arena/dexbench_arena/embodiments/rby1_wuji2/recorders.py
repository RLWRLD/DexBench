# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Record the complete Wuji2 joint command, including joints holding a target."""

from copy import deepcopy

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import EventTermCfg, RecorderTerm, RecorderTermCfg
from isaaclab.utils.configclass import configclass

from dexbench_arena.embodiments.camera_catalog import parse_runtime_camera_calibration
from dexbench_arena.embodiments.common.clean_action_labels import CleanActionsRecorderCfg, mimic_draws_noise
from dexbench_arena.embodiments.joint_state import resolve_joint_index
from dexbench_arena.embodiments.common.recorder_ready_events import RECORDER_READY_MODE
from dexbench_arena.embodiments.rby1_wuji2.constants import RBY1_WUJI2_DATASET_ROBOT, RBY1_WUJI2_JOINT_ACTION_NAMES


class Wuji2JointTargetsRecorder(RecorderTerm):
    """Snapshot all 62 position targets in the public direct-joint action order."""

    joint_names = RBY1_WUJI2_JOINT_ACTION_NAMES

    def record_post_step(self):
        robot = self._env.scene["robot"]
        targets = robot.actuators.target_command.position.torch
        index = resolve_joint_index(robot, self.joint_names, targets.device)
        return "processed_actions", targets.index_select(1, index)


@configclass
class Wuji2JointTargetsRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = Wuji2JointTargetsRecorder


class Wuji2CleanJointTargetsRecorder(Wuji2JointTargetsRecorder):
    """``processed_actions_clean``: the 62 targets with Pink's joints solved for the clean target."""

    def record_post_step(self):
        if not mimic_draws_noise(self._env.cfg):
            return None, None
        manager = self._env.action_manager
        terms = [manager.get_term(name) for name in manager.active_terms]
        term = next((term for term in terms if getattr(term, "clean_processed_actions", None) is not None), None)
        if term is None:
            return None, None
        robot = self._env.scene["robot"]
        targets = robot.actuators.target_command.position.torch.clone()
        targets[:, term._controlled_joint_ids] = term.clean_processed_actions.to(targets.device)
        index = resolve_joint_index(robot, self.joint_names, targets.device)
        return "processed_actions_clean", targets.index_select(1, index)


@configclass
class Wuji2CleanJointTargetsRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = Wuji2CleanJointTargetsRecorder


JOINT_NAMES_KEY = "dexbench_rrc_dataset"


def joint_names_metadata(env, embodiment: str) -> dict:
    """Name the columns ``Wuji2JointTargetsRecorder`` and the state observation record."""
    metadata = {
        "schema_version": 1,
        "robot": RBY1_WUJI2_DATASET_ROBOT,
        "profile": RBY1_WUJI2_DATASET_ROBOT,
        "embodiment": embodiment,
        "state_joint_names": list(env.scene["robot"].joint_names),
        "action_joint_names": list(Wuji2JointTargetsRecorder.joint_names),
    }
    calibration = parse_runtime_camera_calibration(env)
    if calibration:
        metadata["camera_calibration"] = calibration
    return {JOINT_NAMES_KEY: metadata}


class _JointNamesEpisodeMetadata:
    """Add the joint names to what the recorder writes into ``env_args`` on export."""

    def __init__(self, get_ep_meta, env, embodiment: str):
        self._get_ep_meta = get_ep_meta
        self._env = env
        self._embodiment = embodiment

    def __call__(self) -> dict:
        result = deepcopy(self._get_ep_meta())
        result.update(joint_names_metadata(self._env, self._embodiment))
        return result


def install_joint_names_recording_metadata(env, env_ids, embodiment: str) -> None:
    """Recorder-ready event: ``RecorderManager.export_episodes`` merges ``get_ep_meta()`` into the file."""
    del env_ids
    recorders = env.cfg.recorders
    if not isinstance(getattr(recorders, "record_post_step_processed_actions", None), Wuji2JointTargetsRecorderCfg):
        return
    recorder = env.recorder_manager
    if not isinstance(recorder.get_ep_meta, _JointNamesEpisodeMetadata):
        recorder.get_ep_meta = _JointNamesEpisodeMetadata(recorder.get_ep_meta, env, embodiment)


def configure_wuji2_recorders(env_cfg, embodiment: str):
    """Set the demo, Mimic and evaluation recording contracts without enabling recording."""
    for attr in ("demo_recorder_config", "mimic_recorder_config"):
        if not hasattr(env_cfg, attr):
            continue
        recorder_cfg = getattr(env_cfg, attr)
        if recorder_cfg is None:
            recorder_cfg = ActionStateRecorderManagerCfg()
            setattr(env_cfg, attr, recorder_cfg)
        recorder_cfg.record_post_step_processed_actions = Wuji2JointTargetsRecorderCfg()
        recorder_cfg.record_pre_step_actions_clean = CleanActionsRecorderCfg()
        recorder_cfg.record_post_step_processed_actions_clean = Wuji2CleanJointTargetsRecorderCfg()
    recorders = getattr(env_cfg, "recorders", None)
    if recorders is not None and hasattr(recorders, "record_post_step_processed_actions"):
        recorders.record_post_step_processed_actions = Wuji2JointTargetsRecorderCfg()
    if getattr(env_cfg, "events", None) is not None:
        env_cfg.events.recording_joint_names = EventTermCfg(
            func=install_joint_names_recording_metadata, mode=RECORDER_READY_MODE, params={"embodiment": embodiment}
        )
