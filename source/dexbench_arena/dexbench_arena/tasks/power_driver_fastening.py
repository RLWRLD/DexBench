# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""DB-M3 Arena task: two right-side task bolts and four seated inactive CAD bolts, native contact torque and a physical trigger."""

from dataclasses import MISSING, asdict

from isaaclab.managers import (
    EventTermCfg, ObservationGroupCfg, ObservationTermCfg, TerminationTermCfg,
)
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.configclass import configclass
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_base import TaskBase

from dexbench_arena.layouts.db_m3_layout import (
    BOLT_NAMES, SEATED_HEADS, TOOL_NAME,
)
from dexbench_arena.tasks.power_driver_model import FasteningCfg
from dexbench_arena.tasks.power_driver_torque import ContactTorqueFastening
from dexbench_arena.tasks.task_termination import task_termination_cfg

EVENT_NAME = "power_driver_fastening"


def fastening_state(env):
    return env.event_manager.get_term_cfg(EVENT_NAME).func.model


def all_bolts_seated(env):
    return env.event_manager.get_term_cfg(EVENT_NAME).func.success


def bolt_turns(env):
    return fastening_state(env).turns.clone()


def bolt_remaining_travel(env):
    return fastening_state(env).remaining_m


def bolt_engagement(env):
    return fastening_state(env).engaged.float()


@configclass
class FasteningEventsCfg:
    power_driver_fastening: EventTermCfg = MISSING


@configclass
class FasteningSceneCfg:
    driver_trigger_contact: ContactSensorCfg | None = None
    driver_bolt_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{TOOL_NAME}",
        filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/{name}" for name in BOLT_NAMES],
        update_period=0.0,
        history_length=2,
    )


@configclass
class FasteningTerminationsCfg:
    success: TerminationTermCfg = TerminationTermCfg(func=all_bolts_seated)


@configclass
class FasteningObservationsCfg:
    @configclass
    class TaskObsCfg(ObservationGroupCfg):
        motor_turns = ObservationTermCfg(func=bolt_turns)
        remaining_travel_m = ObservationTermCfg(func=bolt_remaining_travel)
        engaged = ObservationTermCfg(func=bolt_engagement)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    task_obs: TaskObsCfg = TaskObsCfg()


class DexBenchPowerDriverFasteningTask(TaskBase):
    def __init__(self, fastening_cfg: FasteningCfg = FasteningCfg(), motor_enabled: bool = True,
                 control_substeps: int = 2, bit_extension_m: float = 0.02,
                 fastening_mode: str = "contact_torque", motor_direction: float = 1., seated_heads: tuple = SEATED_HEADS,
                 trigger_press_m: float = .002, trigger_body_expr: str | None = None,
                 robot_joint_pos_overrides: dict | None = None):
        if fastening_mode != "contact_torque":
            raise ValueError("Only native contact_torque fastening is supported")
        super().__init__(
            episode_length_s=210.,
            task_description="Use the power driver to fasten " + ", ".join(
                BOLT_NAMES[i] for i in fastening_cfg.active_bolt_indices) + ". Other CAD bolts are excluded.",
        )
        self.robot_joint_pos_overrides = robot_joint_pos_overrides
        self.trigger_body_expr = trigger_body_expr
        self.object_name = TOOL_NAME
        self.fastening_mode = fastening_mode
        self.active_bolt_indices = fastening_cfg.active_bolt_indices
        self.contact_history_length = control_substeps
        self.events = FasteningEventsCfg(power_driver_fastening=EventTermCfg(
            func=ContactTorqueFastening, mode="interval", interval_range_s=(0.0, 0.0),
            params={"fastening_cfg": asdict(fastening_cfg), "motor_enabled": motor_enabled,
                    "bit_extension_m": bit_extension_m, "seated_heads": seated_heads, "motor_direction": motor_direction,
                    "trigger_press_m": trigger_press_m},
        ))

    def get_scene_cfg(self):
        scene = FasteningSceneCfg()
        scene.driver_bolt_contact.history_length = self.contact_history_length
        if not self.trigger_body_expr:
            raise ValueError("Contact torque requires the embodiment trigger body expression")
        scene.driver_trigger_contact = ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/{self.trigger_body_expr}",
            filter_prim_paths_expr=["{ENV_REGEX_NS}/power_driver/Trigger"],
            update_period=0., history_length=self.contact_history_length,
        )
        scene.driver_bolt_contact.prim_path += "/Spindle"
        scene.driver_bolt_contact.filter_prim_paths_expr = [
            f"{{ENV_REGEX_NS}}/{name}" + ("/Screw" if i in self.active_bolt_indices else "")
            for i, name in enumerate(BOLT_NAMES)]
        return scene

    def get_events_cfg(self):
        return self.events

    def get_observation_cfg(self):
        return FasteningObservationsCfg()

    def get_termination_cfg(self):
        return task_termination_cfg(self.episode_length_s, FasteningTerminationsCfg())

    def get_metrics(self):
        return [SuccessRateMetric()]

    def get_mimic_env_cfg(self, arm_mode):
        return None
