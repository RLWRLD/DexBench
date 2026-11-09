# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""IsaacLab-Arena#1479's goal-commanded VLM agent policy on RBY1-M Wuji Hand 2, with DexBench's adapters and backend."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydantic import ValidationError

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase

from dexbench_arena.evaluation.dataset_recording import current_run_output_dir
from dexbench_arena.evaluation.evaluation_signature import stamp_policy_settings
from dexbench_arena_vlm_agent.adapters import (
    EEF_CONTROLLERS,
    REFERENCE_FRAMES,
    EefGoalAdapter,
    JointGoalAdapter,
    RobotStateObservationAdapter,
)
from dexbench_arena_vlm_agent.backend import OpenAIBackend
from dexbench_arena_vlm_agent.cost_ledger import CostLedger
from dexbench_arena_vlm_agent.upstream.action_adapter import InvalidAgentResponse
from dexbench_arena_vlm_agent.upstream.vlm_agent_policy import AgentCommandedGoalPolicy, VLMAgentPolicyCfg

OPENAI_URL = "https://api.openai.com/v1"
TASK_RECIPES = Path(__file__).resolve().parent / "task_recipes"
"""What an operator is told about each task and the robot's start, never an object pose: ``<task_recipe>.txt``."""


@dataclass
class VlmGoalPolicyCfg(VLMAgentPolicyCfg):
    system_prompt: str = "built from the harness and the task recipe when the policy binds"
    """Upstream requires one; the policy replaces it with its rung's prompt, bounds and recipe."""
    image_max_width_or_height: int = 640
    """The 640x360 render as it is (never upscaled)."""
    policy_variant: str | None = None
    """The model this Run scores; set, the run is a scored one (:meth:`scored_problems`)."""
    harness: str = "eef"
    """``joints`` (L0) or ``eef`` (L1). A ``str`` checked here, not a ``Literal``: Arena's typed loader refuses those."""
    hand: str = "shapes"
    """L1's hand: ``shapes`` (the five EA demo grasp shapes) or ``generic`` (one task-neutral eigengrasp)."""
    api: str = "responses"
    """``responses`` (OpenAI, paid) or ``chat`` (an OpenAI-compatible local server, for debugging; never scored)."""
    model: str = "gpt-6-astra"
    base_url: str = OPENAI_URL
    api_key_env: str = "OPENAI_API_KEY"
    """The environment variable holding the key; empty sends none (a local server). Never a key itself."""
    reasoning_effort: str = "medium"
    conversation: str = "none"
    """``none``: every request stands alone. ``server``: the episode's turns stay on OpenAI's side (Responses only)."""
    max_output_tokens: int = 16000
    chat_enable_thinking: bool | None = None
    """``api=chat`` only: the local model's thinking switch (``chat_template_kwargs.enable_thinking``), e.g. False
    for Qwen3.5, which thinks by default; None sends nothing."""
    request_timeout_s: float = 300.0
    max_model_calls: int = 170
    goal_timeout_s: float = 5.0
    max_linear_speed: float = 0.10
    max_angular_speed_deg: float = 90.0
    max_joint_speed: float = 0.6
    grasp_s: float = 1.5
    ik_profile: str = "generation"
    eef_reference_frame: str = "base"
    """L1's hands in the robot base frame (as registered) or the torso frame (``torso``), which turns and bends with the
    torso (:mod:`.adapters`). L0 has no hand goals and takes only ``base``."""
    eef_controller: str = "eef_abs"
    """L1's hand poses as absolute targets (as registered) or as changes from where each hand is measured when the goal
    arrives (``eef_delta``). L0 takes only ``eef_abs``."""
    task_recipe: str = ""
    cost_budget_usd: float | None = None
    """The cap on what every run sharing ``cost_ledger_path`` spends together; a request whose worst case would pass
    it is refused and the run stops. None: no cap (unscored runs only)."""
    cost_ledger_path: str = ""
    output_dir: str = "outputs/vlm_agent"
    """Where the logs go outside a benchmark run; inside one they go under the Run's own directory."""
    policy_device: str = "cpu"

    def __post_init__(self):
        super().__post_init__()
        assert self.harness in ("joints", "eef"), f"harness {self.harness!r}: joints or eef"
        assert self.hand in ("shapes", "generic"), f"hand {self.hand!r}: shapes or generic"
        assert self.eef_reference_frame in REFERENCE_FRAMES, \
            f"eef_reference_frame {self.eef_reference_frame!r}: one of {REFERENCE_FRAMES}"
        assert self.eef_controller in EEF_CONTROLLERS, f"eef_controller {self.eef_controller!r}: one of {EEF_CONTROLLERS}"
        assert self.harness == "eef" or (self.eef_reference_frame, self.eef_controller) == ("base", "eef_abs"), \
            "eef_reference_frame and eef_controller are L1's (harness eef); L0 commands joints"
        assert self.api in ("responses", "chat"), f"api {self.api!r}: responses or chat"
        assert self.conversation in ("none", "server"), f"conversation {self.conversation!r}: none or server"
        assert self.conversation == "none" or self.api == "responses", "a server-side conversation needs the Responses API"
        assert self.chat_enable_thinking is None or self.api == "chat", "chat_enable_thinking is for a local chat server"
        assert not self.task_recipe or (TASK_RECIPES / f"{self.task_recipe}.txt").is_file(), \
            f"no packaged task recipe {self.task_recipe!r} in {TASK_RECIPES}"
        problems = self.scored_problems()
        assert not problems, f"a scored run (policy_variant {self.policy_variant!r}) refuses: " + "; ".join(problems)

    def scored_problems(self) -> list[str]:
        """Why this cfg cannot be a scored run's; empty when it is not one (no ``policy_variant``) or it can be."""
        if self.policy_variant is None:
            return []
        problems = []
        if self.api != "responses" or self.base_url != OPENAI_URL:
            problems.append(f"a scored run plans through the OpenAI Responses API at {OPENAI_URL}")
        if not self.task_recipe:
            problems.append("a scored run names a packaged task recipe (task_recipe)")
        if self.cost_budget_usd is None or not self.cost_ledger_path:
            problems.append("a scored run is paid: --set cost_budget_usd=... --set cost_ledger_path=...")
        return problems


@register_policy
class VlmGoalPolicy(AgentCommandedGoalPolicy, PolicyBase[VlmGoalPolicyCfg]):
    name = "vlm_goal"

    def __init__(self, config: VlmGoalPolicyCfg):
        run_dir = current_run_output_dir()
        log_dir = Path(run_dir) / "vlm_agent" if run_dir is not None else Path(config.output_dir) / time.strftime("%Y%m%d_%H%M%S")
        log_dir.mkdir(parents=True, exist_ok=True)
        ledger = CostLedger(config.cost_ledger_path or log_dir / "cost_ledger.jsonl", config.cost_budget_usd)
        backend = OpenAIBackend(api=config.api, model=config.model, base_url=config.base_url, api_key_env=config.api_key_env,
                               reasoning_effort=config.reasoning_effort, conversation=config.conversation,
                               chat_enable_thinking=config.chat_enable_thinking,
                               max_output_tokens=config.max_output_tokens, timeout_s=config.request_timeout_s,
                               ledger=ledger, log_dir=log_dir)
        if config.harness == "joints":
            action_adapter = JointGoalAdapter(max_steps=1, joint_speed=config.max_joint_speed)
        else:
            action_adapter = EefGoalAdapter(max_steps=1, hand=config.hand, ik_profile=config.ik_profile, linear_speed=config.max_linear_speed,
                                            angular_speed_deg=config.max_angular_speed_deg,
                                            joint_speed=config.max_joint_speed, grasp_s=config.grasp_s,
                                            reference_frame=config.eef_reference_frame, controller=config.eef_controller)
        super().__init__(config, inference_backend=backend, observation_adapter=RobotStateObservationAdapter(action_adapter),
                         action_adapter=action_adapter)
        self.backend, self.log_dir = backend, log_dir
        self._bound = False

    def get_action(self, env, observation) -> torch.Tensor:
        if not self._bound:
            self._bind(env)
        return super().get_action(env, observation)

    def _bind(self, env) -> None:
        assert self.action_adapter.is_robot_action_space_supported(env), \
            "The VLM goal policy drives one env of the joint embodiment the frozen rows record (dexbench_rby1_wuji2_joint)"
        unwrapped = env.unwrapped
        self.action_adapter.max_steps = max(1, round(self.config.goal_timeout_s / float(unwrapped.step_dt)))
        recipe = (TASK_RECIPES / f"{self.config.task_recipe}.txt").read_text().strip() if self.config.task_recipe else ""
        self.config.system_prompt = self.action_adapter.system_prompt(recipe, self.config.goal_timeout_s)
        (self.log_dir / "system_prompt.txt").write_text(self.config.system_prompt + "\n")
        stamp_policy_settings(unwrapped, self.policy_settings())
        self._bound = True

    def policy_settings(self) -> dict[str, Any]:
        """The stamp (``evaluation_settings["policy"]``) the episode rows record: the model and how it ran."""
        fields = ("policy_variant", "harness", "hand", "api", "model", "reasoning_effort", "conversation", "max_output_tokens",
                  "image_max_width_or_height", "max_attempts", "max_model_calls", "goal_timeout_s", "max_linear_speed",
                  "max_angular_speed_deg", "max_joint_speed", "grasp_s", "ik_profile", "eef_reference_frame",
                  "eef_controller", "task_recipe")
        controller = getattr(self.action_adapter, "controller", None)
        return {"policy": self.name, **{name: getattr(self.config, name) for name in fields},
                "controller": None if controller is None else controller.settings.as_dict()}

    def _infer(self, content: list[dict], state: dict) -> np.ndarray:
        if self.backend.calls >= self.config.max_model_calls:
            return self._hold(state, "call budget spent")
        try:
            return super()._infer(content, state)
        except (InvalidAgentResponse, ValidationError, AssertionError) as rejection:
            return self._hold(state, f"{self.config.max_attempts} unusable answers in a row: {rejection}")

    def _hold(self, state: dict, why: str) -> np.ndarray:
        """A goal of where the robot is now, held for ``goal_timeout_s`` before the policy asks again."""
        with open(self.log_dir / "held.jsonl", "a") as log:
            log.write(json.dumps({"episode": self.backend.episode, "call": self.backend.calls, "why": why[:500]}) + "\n")
        return self.action_adapter.hold_goal(state)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        super().reset(env_ids)
        self.backend.reset()
        self.action_adapter.reset()
