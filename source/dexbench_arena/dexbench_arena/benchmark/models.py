# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""The models the benchmark requires, defined once."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from dexbench_arena.benchmark.protocol import EXECUTION_HORIZON, ROWS_PER_CELL
from dexbench_arena.evaluation.artifacts import API_TOKEN_FROM_ENVIRONMENT as _API_TOKEN

MODEL_ENVIRONMENT = MappingProxyType({"enable_cameras": True, "camera_scale": 0.5, "wrist_cameras": True})
"""What every model sees, over each Run's environment: the two head cameras and the two palm cameras, each at
640x360, the training render. Every cell renders all four and each model's server declares the ones it reads; an ego-only model leaves the palm images unread."""


def plain(value: Any) -> Any:
    """``value`` with every mapping a dict and every sequence a list: a read-only block as YAML and JSON write it."""
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


@dataclass(frozen=True)
class ModelEntry:
    key: str
    """``run_benchmark_cell.py --model``, the run manifest's ``model``, the policy cfg's ``policy_variant`` and the scorer's column."""
    label: str
    note: str
    """How the server is started (denoising settings live on the server side), or why there is no adapter yet."""
    policy: Mapping[str, Any] | None = None
    """The Run's ``policy`` block; None while the model has no adapter in the repo (required, unavailable)."""
    group: str = "headline"
    """How the scorer reports the model (:data:`GROUPS`). ``headline``: a model the benchmark requires; a missing
    one makes the report partial. ``ablation``: a study's variant of a headline model. ``trial``: a model whose
    execution the headline's equal-horizon comparison does not cover (a tool-calling planner). An ablation or a
    trial runs its cells like any other model, but the scorer reports it apart (``ablations``, ``trials``),
    never as a headline model, and a missing one never makes a report partial."""
    cells: Mapping[str, tuple[str, ...]] | None = None
    """``{task: (cell, ...)}`` the model runs, or None for every task's every cell and sweep. Only a non-headline
    model is scoped: a headline model runs everything."""
    task_policy: Mapping[str, Mapping[str, Any]] = field(default_factory=lambda: MappingProxyType({}))
    """Per task, fields set over :attr:`policy` for that task's runs (Astra's task recipe); a deployment changes none."""
    rows: int | None = None
    """A pilot's rows per cell it runs (:func:`~dexbench_arena.benchmark.protocol.cell_rows`: the first of each Run's
    v1 rows), or None for every row. Only a scoped non-headline model runs a subset."""

    @property
    def available(self) -> bool:
        return self.policy is not None

    @property
    def ablation(self) -> bool:
        return self.group == "ablation"

    @property
    def horizon(self) -> int | None:
        """The chunk length its runs are held to: the protocol's, or None for a trial (no chunk)."""
        return None if self.group == "trial" else EXECUTION_HORIZON

    def runs(self, task: str, cell: str | None) -> bool:
        """Whether the model runs ``task``'s ``cell`` (None: its sweep)."""
        return self.cells is None or (task in self.cells and cell in self.cells[task])

    def runs_task(self, task: str) -> bool:
        return self.cells is None or task in self.cells


BASELINE_POLICY = MappingProxyType({
    "type": "dexbench_arena.policy.hold_pose_policy.HoldPosePolicy",
})

GROUPS = ("headline", "ablation", "trial", "community")
"""How a model is reported; every group but ``headline`` is reported apart (the scorer's ``NON_HEADLINE_SECTIONS``).
``community``: a model an eval config registers (:func:`config_model`), and only such a model: no entry here takes it."""

MODELS: tuple[ModelEntry, ...] = (
    ModelEntry(
        key="gr00t",
        label="GR00T N1.7",
        policy=MappingProxyType({
            "type": "dexbench_arena_gr00t.policy.gr00t_joint_policy.Gr00tJointPolicy",
            "policy_variant": "gr00t",
            "policy_config_yaml_path": "/tmp/dexbench_gr00t_n17_contract/policy.yaml",
            "action_chunk_length": EXECUTION_HORIZON,
            "policy_device": "cpu",
            "remote_host": "127.0.0.1",
            "remote_port": 5810,
            "remote_api_token": _API_TOKEN,
            "scheduler": "chunk",
            "relative_action_groups": [],
        }),
        note=(
            "Multitask checkpoint trained on absolute joint actions with RLWRLD/DexBench-Isaac-GR00T's "
            "examples/DexBench/train.sh (the GA 15-task set: ego left and both palm cameras, the 62 policy joints "
            "in the wuji2_v1 groups), served next to the simulator by the training commit's run_gr00t_server. The "
            "run's host writes the joint contract with "
            "dexbench_arena_gr00t.policy.wuji2_gr00t_config --output /tmp/dexbench_gr00t_n17_contract (320x180 "
            "target, 40-step model horizon); the client area-resizes the 640x360 render to 320x180. 4 "
            "denoising steps, uniform timesteps (the checkpoint's num_inference_timesteps, server side)."
        ),
    ),
    ModelEntry(
        key="cosmos",
        label="Cosmos3-Nano",
        policy=MappingProxyType({
            "type": "dexbench_arena_cosmos.policy.cosmos3_joint_policy.Cosmos3JointPolicy",
            "policy_variant": "cosmos",
            "remote_url": "http://127.0.0.1:5811",
            "remote_api_token": _API_TOKEN,
            "action_chunk_length": EXECUTION_HORIZON,
            "policy_device": "cpu",
        }),
        note=(
            "Multitask checkpoint (DexBench-cosmos-framework action_policy_server_dexbench) served next to "
            "the simulator. The GA 15-task multitask checkpoint is trained with RLWRLD/DexBench-cosmos-framework "
            "(ego left and both palm cameras on the 320x256 ego_wrists canvas, the 62 policy joints selected by name "
            "from the 66-D set in the wuji2_v1 order, absolute actions, 32-step chunks) and served by the training "
            "commit's server at 10 UniPC denoising steps, shift 5 (--num-steps 10 --shift 5 --sampler unipc; the "
            "server's default is 4 steps). The "
            "server states its sampling settings and its checkpoint's frame size at /info; the client area-resizes "
            "the 640x360 render to that frame (320x180 for the GA checkpoints; a 256-tier checkpoint declares 144x256)."
        ),
    ),
    ModelEntry(
        key="rldx11",
        label="RLDX-1.1",
        policy=MappingProxyType({
            "type": "dexbench_arena_rldx.policy.rby1_wuji_joint_policy.RldxRby1WujiJointRemotePolicy",
            "policy_variant": "rldx11",
            "remote_host": "127.0.0.1",
            "remote_port": 5813,
            "remote_api_token": _API_TOKEN,
            "action_chunk_length": EXECUTION_HORIZON,
            "policy_device": "cpu",
            "scheduler": "synced_batch",
            "joint_layout": "wuji2_rrc_v1",
            "target_image_size": [360, 640, 3],
            "memory_sessions": False,
            "denoising_timesteps": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        }),
        note=(
            "The GA recipe: one multitask checkpoint over the GA 15-task set, fine-tuned with RLDX "
            "from the RLDX-1.1-PT-IMG pre-training checkpoint, padded-action noise mask on, on the ego left and both "
            "palm cameras with a 40-step action chunk (--action-horizon 40), absolute joint positions in the six-group wuji2_rrc_v1 layout (the 62 policy "
            "joints; the wheels are not trained). Served next to the simulator by the training commit's "
            "rldx.eval.run_rldx_server (--embodiment-tag GENERAL_EMBODIMENT --runtime-profile eager, no RTC): a "
            "server of another commit can refuse the checkpoint (its config.json has no normalization_type). The "
            "policy is stateless, so the client "
            "sends no memory session options. The served "
            "contract names the cameras the client sends, each the raw 640x360 render; the native processor "
            "resizes it to 320x192 on the server. Served at 10 uniform denoising steps (--denoising-timesteps 0 "
            "0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9), which the rows record."
        ),
    ),
    ModelEntry(
        key="pi05",
        label="openpi pi0.5",
        policy=MappingProxyType({
            "type": "dexbench_arena_openpi.policy.rby1_wuji_joint_policy.Pi0Rby1WujiJointRemotePolicy",
            "policy_variant": "pi05",
            "remote_host": "127.0.0.1",
            "remote_port": 5814,
            "action_chunk_length": EXECUTION_HORIZON,
            "policy_device": "cpu",
            "scheduler": "chunk",
        }),
        note=(
            "Multitask checkpoint fine-tuned from pi05_base with RLWRLD/openpi ("
            "config rby1_wujiv2_ga15_multitask_wrist_pi05_50: the GA 15-task set, "
            "ego left / palm left / palm right in the base / left-wrist / right-wrist slots, the 62 policy joints, "
            "50-step model horizon, each episode's task as the prompt), served next to the simulator by the training "
            "commit's scripts/serve_policy.py (policy:checkpoint with that config and the step directory). The "
            "server advertises its cameras as camera_keys and the client sends exactly those, each 640x360 render "
            "letterboxed to 224x224 with openpi's resize_with_pad. The EA checkpoints (rby1_wujiv2_ea_*_pi05_50, "
            "the two ego cameras) are served the same way, by their own training commit. 10 denoising "
            "steps (openpi's default), uniform."
        ),
    ),
    ModelEntry(
        key="fastwam",
        label="FastWAM Joint",
        policy=MappingProxyType({
            "type": "dexbench_arena_wam.policy.wam_joint_policy.WamJointPolicy",
            "policy_variant": "fastwam",
            "model_family": "fastwam",
            "remote_url": "http://127.0.0.1:5815",
            "remote_api_token": _API_TOKEN,
            "action_chunk_length": EXECUTION_HORIZON,
            "policy_device": "cpu",
        }),
        note=(
            "GA wrist-camera checkpoint fine-tuned with RLWRLD/DexBench-FastWAM experiments/dexbench (FastWAM Joint "
            "on Wan2.2-TI2V-5B; ego left on top of a 384x320 canvas, both wrist cameras below; 32-step model "
            "horizon), served next to the simulator by experiments/dexbench/serve.py --api-token-file on port "
            "5815: 10 denoising steps, the model's flow shifts (video 5, action 1), noise seed 42 for every "
            "request. The server states its cameras and frame size at /info; the client area-resizes each "
            "640x360 render to that frame (320x180). The GA 15-task multitask checkpoint is trained with the Cosmos "
            "recipe's colour jitter (task dexbench_ga_wrist_joint_3cam_384_1e-4) on the 62 policy joints selected by "
            "name from the 66-D set in the wuji2_v1 order, and served by the training commit's serve.py from the "
            "run's config.yaml and dataset_stats.json."
        ),
    ),
    ModelEntry(
        key="openwam",
        label="OpenWAM-alpha",
        policy=MappingProxyType({
            "type": "dexbench_arena_wam.policy.wam_joint_policy.WamJointPolicy",
            "policy_variant": "openwam",
            "model_family": "openwam",
            "remote_url": "http://127.0.0.1:5816",
            "remote_api_token": _API_TOKEN,
            "action_chunk_length": EXECUTION_HORIZON,
            "policy_device": "cpu",
        }),
        note=(
            "GA wrist-camera checkpoint fine-tuned from OpenWAM-alpha with RLWRLD/DexBench-OpenWAM (ego left in "
            "the 384x320 pre-training canvas's head slot, both wrist cameras in its wrist slots; arm joints in "
            "the 80-D action's end-effector slots; 32-step model horizon), served next to the simulator by "
            "benchmarks/dexbench/server.py --api-token-file on port 5816 at 10 denoising steps. The server states "
            "its cameras and frame size at /info; the client area-resizes each 640x360 render to that frame "
            "(320x180)."
        ),
    ),
)

_ASTRA_TASKS = ("r2", "l2", "m1", "h1")
_ASTRA_POLICY = MappingProxyType({
    "type": "dexbench_arena_vlm_agent.policy.VlmGoalPolicy",
    "api": "responses",
    "model": "gpt-6-astra",
    "reasoning_effort": "medium",
    "conversation": "none",
    "max_model_calls": 170,
    "policy_device": "cpu",
    "cost_budget_usd": None,
    "cost_ledger_path": "",
})


def _astra(harness: str, hand: str | None, level: str, rows: int, what: str) -> ModelEntry:
    key = "astra_" + level.lower().replace(" ", "_")
    return ModelEntry(
        key=key,
        label=f"GPT-6 Astra, {level} ({what})",
        group="trial",
        cells=MappingProxyType({task: ("train",) for task in _ASTRA_TASKS}),
        rows=rows,
        policy=MappingProxyType({**_ASTRA_POLICY, "policy_variant": key, "harness": harness,
                                 **({"hand": hand, "eef_reference_frame": "torso"} if hand else {})}),
        task_policy=MappingProxyType({task: MappingProxyType({"task_recipe": f"db_{task}"}) for task in _ASTRA_TASKS}),
        note=(
            f"Harness-ladder pilot, {level}: {what}. VlmGoalPolicy (upstream's VLM agent policy, IsaacLab-Arena#1479) "
            "asking the OpenAI Responses API for goals, statelessly "
            "(OPENAI_API_KEY from the environment; no server, no checkpoint) on the joint embodiment, with the task's "
            f"packaged recipe, on the first {rows} frozen Train rows of each EA task. A goal-commanding planner chooses its "
            "own re-plan points, so it is held to no chunk horizon and is reported apart from the headline. "
            "run_benchmark_cell.py --checkpoint names the model snapshot it ran."
        ),
    )


MODELS += (
    _astra("joints", None, "L0", 10, "the 62 absolute joint targets, no harness"),
    _astra("eef", "generic", "L1 generic", 20, "wrist poses through arm IK, one task-neutral eigengrasp, torso and neck joints"),
    _astra("eef", "shapes", "L1 shapes", 20, "wrist poses through arm IK, five demo grasp shapes, torso and neck joints"),
)

_GR00T = next(model for model in MODELS if model.key == "gr00t")
_COSMOS = next(model for model in MODELS if model.key == "cosmos")
MODELS += (
    ModelEntry(
        key="gr00t_rel",
        label="GR00T N1.7 (arm-relative actions, ablation)",
        group="ablation",
        policy=MappingProxyType({**_GR00T.policy, "policy_variant": "gr00t_rel",
                                 "relative_action_groups": ["left_arm_joints", "right_arm_joints"]}),
        note=(
            "Action-representation ablation: the GR00T recipe trained with both arms relative to the state, "
            "served like gr00t from its own checkpoint, whose modality config states each group's rep."
        ),
    ),
    ModelEntry(
        key="cosmos_rel",
        label="Cosmos3-Nano (arm-relative actions, ablation)",
        group="ablation",
        policy=MappingProxyType({**_COSMOS.policy, "policy_variant": "cosmos_rel", "relative_joint_slices": [[0, 14]]}),
        note=(
            "Action-representation ablation: the Cosmos recipe trained with both arms relative to the window's "
            "first state. Must be served by a cosmos-framework server that "
            "states action_representation at /info and adds the state back; a second server on one host takes "
            "another port (--set remote_url=...)."
        ),
    ),
)

for _model in MODELS:
    assert _model.group in GROUPS and _model.group != "community", _model.key
    assert _model.cells is None or _model.group != "headline", _model.key
    assert _model.cells is not None or not _model.task_policy, _model.key
    assert _model.rows is None or (_model.cells is not None and 0 < _model.rows <= ROWS_PER_CELL), _model.key
    assert set(_model.task_policy) <= set(_model.cells or ()), _model.key
    if _model.available:
        assert _model.horizon is None or _model.policy["action_chunk_length"] == _model.horizon, _model.key
        assert _model.policy["policy_variant"] == _model.key, _model.key
assert len({model.key for model in MODELS}) == len(MODELS)

AVAILABLE_MODELS = tuple(model for model in MODELS if model.available)
REQUIRED_MODELS = tuple(model for model in MODELS if model.group == "headline")
"""The headline's models: every one is reported, a missing one makes the report partial."""
MODELS_BY_KEY = {model.key: model for model in MODELS}

CONFIG_MODEL_FIELDS = ("key", "label", "note", "group", "policy")
"""What an eval config's model block may state of a model it registers (the launcher's ``model``, beside its
checkpoint; ``run_benchmark_cell.py --model-entry``)."""
CONFIG_GROUPS = ("community",)
"""The one group a config model takes: ``headline``, ``ablation`` and ``trial`` are entries in :data:`MODELS`, reviewed
with the manifest (the maintainers' rule, 2026-11-09)."""


def config_model(block: Mapping[str, Any]) -> ModelEntry:
    """The model an eval config registers: ``key`` (a key of its own, never one of :data:`MODELS`), ``label``, ``note``,
    ``group`` (``community``, the only one, which may be left out) and ``policy``, its client's block: ``type`` (the policy
    class) and its parameters. The block's ``policy_variant`` is the key (filled in when it is left out), and a chunk
    model's ``action_chunk_length`` the protocol's horizon (filled in likewise); a block that states another value is
    refused, as is a field of the block this function does not know."""
    unknown = sorted(set(block) - set(CONFIG_MODEL_FIELDS))
    if unknown:
        raise ValueError(f"a config model has {list(CONFIG_MODEL_FIELDS)}, not {unknown}")
    key = block.get("key")
    if not isinstance(key, str) or not key:
        raise ValueError("a config model needs a key: its name in the run manifests and the score")
    if key in MODELS_BY_KEY:
        raise ValueError(f"model {key!r} is registered in benchmark/models.py: a config model takes a key of its own, "
                         "or the config names the registered model by its key alone")
    group = block.get("group", "community")
    if group not in CONFIG_GROUPS:
        raise ValueError(f"a config model's group is community (leave it out), not {group!r}: a headline, ablation or "
                         "trial model is an entry in benchmark/models.py, reviewed with the manifest "
                         "(write_benchmark_experiments.py --check)")
    if not isinstance(block.get("policy") or {}, Mapping):
        raise ValueError(f"model {key}: policy is the client's block (a mapping: type and its parameters), not "
                         f"{type(block['policy']).__name__}")
    policy = dict(block.get("policy") or {})
    token = policy.get("remote_api_token")
    if token is not None and token != _API_TOKEN:
        raise ValueError(f"model {key}: policy.remote_api_token is written to plan.json, the launch records and the "
                         f"workers' arguments, so it takes no value: leave it out, or {_API_TOKEN} (the worker sets "
                         "the token itself)")
    if not isinstance(policy.get("type"), str) or not policy["type"]:
        raise ValueError(f"model {key}: policy.type names its client's policy class (e.g. "
                         "dexbench_arena_cosmos.policy.cosmos3_joint_policy.Cosmos3JointPolicy)")
    if policy.setdefault("policy_variant", key) != key:
        raise ValueError(f"model {key}: policy.policy_variant is the model's key, which the scorer holds every Run to, "
                         f"not {policy['policy_variant']!r}")
    model = ModelEntry(key=key, label=str(block.get("label") or key), note=str(block.get("note") or ""),
                       policy=MappingProxyType(policy), group=group)
    if model.horizon is not None and policy.setdefault("action_chunk_length", model.horizon) != model.horizon:
        raise ValueError(f"model {key}: a chunk model runs the protocol's horizon, action_chunk_length "
                         f"{model.horizon}, not {policy['action_chunk_length']!r}")
    return model


def resolve_model(model: str | Mapping[str, Any]) -> ModelEntry:
    """A model by its key in :data:`MODELS`, or the config model a mapping registers (:func:`config_model`)."""
    if isinstance(model, Mapping):
        return config_model(model)
    if model not in MODELS_BY_KEY:
        raise ValueError(f"no model {model!r}; the registry's are {sorted(MODELS_BY_KEY)}, and a model of your own is "
                         "registered by its eval config (a label, a policy block and a group beside its key)")
    return MODELS_BY_KEY[model]


def model_record(model: ModelEntry) -> dict[str, Any]:
    """The whole entry, as a run manifest records it: what the model was, whether this file or an eval config
    registered it (``registration``: ``code`` or ``config``)."""
    return {"key": model.key, "label": model.label, "note": model.note, "group": model.group,
            "policy": plain(model.policy), "cells": plain(model.cells), "task_policy": plain(model.task_policy),
            "rows": model.rows, "registration": "code" if MODELS_BY_KEY.get(model.key) is model else "config"}
