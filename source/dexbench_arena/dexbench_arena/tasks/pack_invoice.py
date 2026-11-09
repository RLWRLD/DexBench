# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Pack-invoice task: put one sheet from the paper stack into the open outbound box."""

from __future__ import annotations

from dataclasses import MISSING
from typing import Any

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg, TerminationTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg

from dexbench_arena.contact_materials import HandContactMaterialCfg

from dexbench_arena.deformables.newton_solver_slot import defer_initial_graph_capture
from dexbench_arena.deformables.pack_invoice_runtime import attach_pack_invoice_sidecar
from dexbench_arena.deformables.pack_invoice_scene_randomization import PACK_INVOICE_INSTRUCTION
from dexbench_arena.deformables.paper_stack import get_active_paper_stack
from dexbench_arena.evaluation.replay_guards import deterministic_reset
from dexbench_arena.tasks.task_termination import task_termination_cfg


def _active_paper_stack():
    stack = get_active_paper_stack()
    if stack is None:
        raise RuntimeError("pack_invoice: no paper sidecar is active; the env was built without attach_paper_sidecar")
    return stack


@deterministic_reset
def attach_paper_sidecar(env: ManagerBasedRLEnv, env_ids, paper_hand=None, paper_args=None, scene_args=None,
                         paper_pile=None) -> None:
    """Startup event: refuse more than one env and attach the paper sidecar, which builds it at the first reset."""

    if env.num_envs != 1:
        raise ValueError(f"pack_invoice simulates the paper of env 0 only; run it with --num_envs 1, not {env.num_envs}")
    attach_pack_invoice_sidecar(env, hand=paper_hand, paper_args=paper_args, scene_args=scene_args,
                                paper_pile=paper_pile)


def invoice_sheet_packed(
    env: ManagerBasedRLEnv,
    min_inside_frac: float = 0.9,
    settle_speed_m_s: float = 0.30,
) -> torch.Tensor:
    """True when any paper sheet rests inside the open box's interior volume."""

    num_envs = env.unwrapped.num_envs
    device = env.unwrapped.device
    stack = _active_paper_stack()
    packed = stack.any_sheet_in_box(min_inside_frac=min_inside_frac, settle_speed_m_s=settle_speed_m_s)
    return torch.full((num_envs,), packed, dtype=torch.bool, device=device)


def invoice_sheet_dropped(
    env: ManagerBasedRLEnv,
    drop_height_m: float = 0.15,
) -> torch.Tensor:
    """True when any sheet centroid fell ``drop_height_m`` below its pile's base."""

    num_envs = env.unwrapped.num_envs
    device = env.unwrapped.device
    stack = _active_paper_stack()
    dropped = stack.any_sheet_dropped(drop_height_m)
    return torch.full((num_envs,), dropped, dtype=torch.bool, device=device)


@configclass
class PackInvoiceTerminationsCfg:
    """Success and failure terms for the pack-invoice task; the episode limit is the task's ``timeout_s``."""

    success: TerminationTermCfg = MISSING

    sheet_dropped: TerminationTermCfg = MISSING


@configclass
class PackInvoiceEventsCfg:
    """The paper sidecar's attachment, at startup on every construction path."""

    attach_paper: EventTermCfg = MISSING


class DexBenchPackInvoiceTask(TaskBase):
    """Success when a sheet from the invoice stack rests inside the open box."""

    def __init__(
        self,
        episode_length_s: float | None = 120.0,
        task_description: str | None = None,
        min_inside_frac: float = 0.9,
        settle_speed_m_s: float = 0.30,
        drop_height_m: float = 0.15,
        hand_contact_material: HandContactMaterialCfg | None = None,
        robot_joint_pos_overrides: dict[str, float] | None = None,
        paper_args: dict[str, Any] | None = None,
        scene_args: dict[str, float] | None = None,
        paper_pile: Any = None,
    ):
        if task_description is None:
            task_description = PACK_INVOICE_INSTRUCTION
        elif not task_description.strip():
            raise ValueError("task_description is empty; leave it unset for the task's own instruction")
        super().__init__(episode_length_s=episode_length_s, task_description=task_description)
        self.hand_contact_material = hand_contact_material
        self.robot_joint_pos_overrides = dict(robot_joint_pos_overrides) if robot_joint_pos_overrides else None
        self.termination_cfg = PackInvoiceTerminationsCfg(
            success=TerminationTermCfg(
                func=invoice_sheet_packed,
                params={
                    "min_inside_frac": min_inside_frac,
                    "settle_speed_m_s": settle_speed_m_s,
                },
            ),
            sheet_dropped=TerminationTermCfg(
                func=invoice_sheet_dropped,
                params={"drop_height_m": drop_height_m},
            ),
        )
        self.events_cfg = PackInvoiceEventsCfg(attach_paper=EventTermCfg(
            func=attach_paper_sidecar, mode="startup",
            params={"paper_args": dict(paper_args or {}), "scene_args": dict(scene_args or {}), "paper_pile": paper_pile},
        ))

    def configure_for_embodiment(self, embodiment: Any) -> None:
        """Hand the embodiment's ``paper_hand`` to the sidecar; a robot that declares none is refused here."""

        defer_initial_graph_capture()
        if getattr(getattr(embodiment, "scene_config", None), "robot", None) is None:
            return
        hand = getattr(embodiment, "paper_hand", None)
        if hand is None:
            raise ValueError(f"pack_invoice needs the embodiment to declare paper_hand; "
                             f"{getattr(embodiment, 'name', type(embodiment).__name__)} declares none")
        self.events_cfg.attach_paper.params["paper_hand"] = hand

    def get_scene_cfg(self) -> Any:
        return None

    def get_termination_cfg(self) -> TaskTerminationCfg:
        return task_termination_cfg(self.episode_length_s, self.termination_cfg)

    def get_events_cfg(self) -> Any:
        return self.events_cfg

    def get_mimic_env_cfg(self, arm_mode: Any) -> Any:
        return None

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]
