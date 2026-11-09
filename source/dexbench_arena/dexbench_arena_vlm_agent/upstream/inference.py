# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
# Copied from upstream IsaacLab-Arena main (d6699ca44, IsaacLab-Arena#1414) inference/backend.py: the request type and
# the strict-schema helpers IsaacLab-Arena#1479's policy uses. InferenceBackend here is only the interface the policy
# calls (upstream's is a Chat Completions client; ours is backend.OpenAIBackend).
# Pin workaround: the pinned Arena (3bd75b23) has neither the request type nor the helpers. Delete with this package
# once the pin includes them.

"""Structured-output request types for the VLM agent policy."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Protocol

from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel


@dataclass(frozen=True)
class InferenceRequest:
    """Request structured output from text and optional OpenAI-compatible image messages."""

    messages: list[ChatCompletionMessageParam]
    response_schema: dict[str, Any]
    schema_name: str = "agent_command"
    retry_label: str = "inference"


class InferenceBackend(Protocol):
    def infer(self, request: InferenceRequest) -> dict[str, Any]: ...


def build_strict_schema(model_cls: type[BaseModel]) -> dict[str, Any]:
    """Return ``model_cls``'s JSON schema munged for OpenAI strict mode."""
    schema = copy.deepcopy(model_cls.model_json_schema())
    _apply_strict_constraints(schema)
    return schema


def _apply_strict_constraints(node: dict | list) -> None:
    """Recursively apply OpenAI strict-mode constraints to a JSON-schema node."""
    if isinstance(node, dict):
        if node.get("type") == "object" and "properties" in node:
            node["additionalProperties"] = False
            node["required"] = list(node["properties"].keys())
        # Strict mode forbids ``default`` keys (every field is required, so
        # defaults can never apply). Drop them defensively at every level.
        node.pop("default", None)
        for v in node.values():
            _apply_strict_constraints(v)
    elif isinstance(node, list):
        for v in node:
            _apply_strict_constraints(v)
