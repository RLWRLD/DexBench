# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""The inference backend the VLM goal policy calls: OpenAI's Responses API, or a local server."""

from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
from typing import Any

import time

from openai import OpenAI, RateLimitError
from PIL import Image

from dexbench_arena_vlm_agent.cost_ledger import CostLedger
from dexbench_arena_vlm_agent.upstream.inference import InferenceRequest

PRICE_PER_MTOK = {"gpt-6-astra": (10.0, 50.0)}
"""Standard per-million-token prices (input, output): what the ledger reserves and records."""


def image_tokens_bound(width: int, height: int) -> int:
    """The larger of OpenAI's two published high-detail image formulas for a ``width`` x ``height`` image: 512 px
    tiles after fitting 2048 px and a 768 px short side (85 + 170 per tile), and 32 px patches counted twice."""
    scale = min(1.0, 2048 / max(width, height))
    w, h = width * scale, height * scale
    scale = min(1.0, 768 / min(w, h))
    tiles = math.ceil(w * scale / 512) * math.ceil(h * scale / 512)
    return max(85 + 170 * tiles, 2 * math.ceil(width / 32) * math.ceil(height / 32))


def _image_urls(value: Any) -> list[str]:
    """Every image data URL in a request, in either API's message format."""
    if isinstance(value, str):
        return [value] if value.startswith("data:image/") else []
    items = value.values() if isinstance(value, dict) else value if isinstance(value, list) else ()
    return [url for item in items for url in _image_urls(item)]


def _responses_message(message: dict[str, Any]) -> dict[str, Any]:
    """A Chat Completions message (what the policy builds) as a Responses API input message."""
    content = message["content"]
    if isinstance(content, str):
        return {"role": message["role"], "content": content}
    parts = [{"type": "input_text", "text": part["text"]} if part["type"] == "text" else
             {"type": "input_image", "image_url": part["image_url"]["url"], "detail": "high"} for part in content]
    return {"role": message["role"], "content": parts}


class OpenAIBackend:
    """``infer(InferenceRequest) -> dict``, what :class:`~dexbench_arena_vlm_agent.upstream.vlm_agent_policy.VLMAgentPolicy`
    calls; one per policy, one env."""

    def __init__(self, *, api: str, model: str, base_url: str, api_key_env: str, reasoning_effort: str,
                 conversation: str, max_output_tokens: int, timeout_s: float, ledger: CostLedger, log_dir: Path,
                 chat_enable_thinking: bool | None = None):
        api_key = os.environ.get(api_key_env) if api_key_env else None
        if api == "responses" and not api_key:
            raise RuntimeError(f"${api_key_env} is not set; the Responses API needs a key")
        self._client = OpenAI(api_key=api_key or "unused", base_url=base_url, timeout=timeout_s, max_retries=2)
        self.api, self.model, self.reasoning_effort = api, model, reasoning_effort
        self.conversation, self.max_output_tokens = conversation, max_output_tokens
        self._ledger, self._log_dir = ledger, log_dir
        self.chat_enable_thinking = chat_enable_thinking
        self.rate_limit_wait_s = 0.0
        """Seconds this backend spent waiting out rate limits (429), for the run's record."""
        self.episode = len(list((log_dir / "calls").glob("episode_*")))
        self.calls, self._previous_id, self._context_tokens = 0, None, 0

    def reset(self) -> None:
        """A new episode: its calls are counted and logged apart, and the server-side conversation starts over."""
        if self.calls:
            self.episode += 1
        self.calls, self._previous_id, self._context_tokens = 0, None, 0

    def infer(self, request: InferenceRequest) -> dict[str, Any]:
        system, messages = request.messages[0]["content"], list(request.messages[1:])
        server = self.api == "responses" and self.conversation == "server"
        if server and self._previous_id is not None:
            messages = messages[-1:]
        reservation = self._ledger.reserve(self._bound_usd(system, messages, request.response_schema),
                                           episode=self.episode, call=self.calls + 1)
        self.calls += 1
        body: dict[str, Any] = {"model": self.model}
        response = None
        try:
            if self.api == "responses":
                body.update(instructions=system, input=[_responses_message(m) for m in messages],
                            text={"format": {"type": "json_schema", "name": request.schema_name,
                                             "schema": request.response_schema, "strict": True}},
                            reasoning={"effort": self.reasoning_effort}, max_output_tokens=self.max_output_tokens,
                            store=server, **({"previous_response_id": self._previous_id} if server and self._previous_id else {}))
                response = self._rate_limited(self._client.responses.create, body)
                usage, text = response.usage, response.output_text
                if server:
                    self._previous_id = response.id
            else:
                body.update(messages=[{"role": "system", "content": system}, *messages],
                            response_format={"type": "json_schema", "json_schema": {
                                "name": request.schema_name, "schema": request.response_schema, "strict": True}},
                            max_tokens=self.max_output_tokens)
                if self.chat_enable_thinking is not None:
                    body["extra_body"] = {"chat_template_kwargs": {"enable_thinking": self.chat_enable_thinking}}
                response = self._rate_limited(self._client.chat.completions.create, body)
                usage, text = response.usage, response.choices[0].message.content
            input_tokens = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", 0)
            output_tokens = getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", 0)
            self._context_tokens = input_tokens + output_tokens
            cost = self._cost_usd(input_tokens, output_tokens)
            self._ledger.settle(reservation, cost, input_tokens=input_tokens, output_tokens=output_tokens)
            try:
                return json.loads(text or "")
            except ValueError:
                return {"unparsable_answer": (text or "")[:500]}
        finally:
            self._save(body, response)

    def _rate_limited(self, create, body: dict[str, Any], patience_s: float = 900.0) -> Any:
        """``create(**body)``, waiting out rate limits (the SDK's own retries spent) for up to ``patience_s``: the
        simulation waits with the request, so a limit only slows a run, where an error would end it. An account out
        of credits answers with the same 429, but no wait ends that, so it raises at once."""
        start = time.monotonic()
        for attempt in range(100):
            try:
                return create(**body)
            except RateLimitError as error:
                if error.type == "insufficient_quota":
                    raise
                waited = time.monotonic() - start
                if waited > patience_s:
                    raise
                pause = min(60.0, 5.0 * 2 ** attempt)
                print(f"[vlm_agent] rate limited; waiting {pause:.0f} s ({waited:.0f} s so far)", flush=True)
                time.sleep(pause)
                self.rate_limit_wait_s += pause
        raise RuntimeError("unreachable")

    def _cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        price_in, price_out = PRICE_PER_MTOK.get(self.model, (0.0, 0.0))
        return (input_tokens * price_in + output_tokens * price_out) / 1e6

    def _bound_usd(self, system: str, messages: list[dict[str, Any]], schema: dict[str, Any]) -> float:
        """The request's worst case: every byte of its text as a token, each image at its token bound, what the server
        already holds of the conversation, and ``max_output_tokens`` of output."""
        price_in, price_out = PRICE_PER_MTOK.get(self.model, (0.0, 0.0))
        if not price_in and self.api == "responses":
            raise ValueError(f"no price for {self.model!r}: an unpriced model is never counted as free")
        urls = _image_urls(messages)
        text = json.dumps([system, messages, schema])
        for url in urls:
            text = text.replace(url, "")
        image_tokens = 0
        for url in urls:
            width, height = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).size
            image_tokens += image_tokens_bound(width, height)
        return ((len(text.encode()) + image_tokens + self._context_tokens) * price_in + self.max_output_tokens * price_out) / 1e6

    def _save(self, body: dict[str, Any], response: Any) -> None:
        directory = self._log_dir / "calls" / f"episode_{self.episode:03d}"
        images = self._log_dir / "images"
        images.mkdir(parents=True, exist_ok=True)
        directory.mkdir(parents=True, exist_ok=True)
        sent = copy.deepcopy(body)
        text = json.dumps(sent)
        for url in set(_image_urls(sent)):
            data = base64.b64decode(url.split(",", 1)[1])
            digest = hashlib.sha256(data).hexdigest()
            (images / f"{digest}.jpg").write_bytes(data)
            text = text.replace(url, f"images/{digest}.jpg")
        record = {"request": json.loads(text), "response": None if response is None else response.model_dump(mode="json")}
        (directory / f"call_{self.calls:03d}.json").write_text(json.dumps(record, indent=1) + "\n")
