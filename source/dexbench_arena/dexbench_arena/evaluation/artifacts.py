# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""Reading, hashing and publishing the benchmark's files: one place for the byte-level rules."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: str | Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_jsonl(path: str | Path) -> tuple[str, list[dict]]:
    """``(sha256, rows)`` of a JSONL file, both from one read of its bytes."""
    data = Path(path).read_bytes()
    rows = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
    return sha256_bytes(data), rows


def jsonl_bytes(rows: list[dict]) -> bytes:
    return "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows).encode()


def _write_temporary(path: Path, data: bytes) -> str:
    """``data`` in a unique, synced temporary file beside ``path``; returns its name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    return temporary


def publish_new(path: str | Path, data: bytes) -> bool:
    """Publish ``data`` at ``path`` once: True when written, False when it already holds exactly ``data``."""
    path = Path(path)
    temporary = _write_temporary(path, data)
    try:
        os.link(temporary, path)
    except FileExistsError:
        if path.read_bytes() == data:
            return False
        raise FileExistsError(f"{path} exists with other content; refusing to replace it") from None
    finally:
        os.unlink(temporary)
    return True


def replace_atomic(path: str | Path, data: bytes) -> None:
    """Write ``data`` to ``path`` through a temporary file and an atomic rename (for files meant to be rewritten)."""
    path = Path(path)
    temporary = _write_temporary(path, data)
    try:
        os.replace(temporary, path)
    except BaseException:
        os.unlink(temporary)
        raise


def recorded(row: Mapping[str, Any], field: str, default: Any = None) -> Any:
    """A DexBench recorder term's field of an episode row, in either row layout."""
    value = row.get(field, default)
    if isinstance(value, Mapping) and set(value) == {field}:
        return value[field]
    return value

API_TOKEN_FROM_ENVIRONMENT = "${oc.env:DEXBENCH_POLICY_API_TOKEN,null}"
"""Where a policy block's ``remote_api_token`` comes from: the environment the runner starts in, never a file."""


CREDENTIAL_ENVIRONMENT = re.compile(r"DEXBENCH_(.*_)?(TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIALS?|KEY|AUTH)(_.*)?\Z")
"""A DEXBENCH_* variable named like a credential, whatever it is for: a file that records the environment
keeps its name and never its value (recordings, sidecars and dataset cards ship with the data)."""


def redact_api_tokens(value: Any) -> Any:
    """``value`` with every non-null ``*api_token*`` field replaced by :data:`API_TOKEN_FROM_ENVIRONMENT`."""
    if isinstance(value, Mapping):
        return {key: API_TOKEN_FROM_ENVIRONMENT if "api_token" in str(key) and item is not None
                else redact_api_tokens(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        redacted, after_flag = [], False
        for item in value:
            if after_flag and isinstance(item, str):
                item = API_TOKEN_FROM_ENVIRONMENT
            elif isinstance(item, str):
                item = _redact_argument(item)
            else:
                item = redact_api_tokens(item)
            after_flag = isinstance(item, str) and item.startswith("-") and "=" not in item and "api_token" in item
            redacted.append(item)
        return redacted
    return value


def _redact_json(text: str) -> str | None:
    """``text`` redacted inside when it is a JSON object, else None."""
    if not text.lstrip().startswith("{"):
        return None
    try:
        return json.dumps(redact_api_tokens(json.loads(text)))
    except ValueError:
        return None


def _redact_argument(argument: str) -> str:
    """One command-line argument with an api token in it redacted (see :func:`redact_api_tokens`)."""
    if "api_token" not in argument:
        return argument
    whole = _redact_json(argument)
    if whole is not None:
        return whole
    key, sep, value = argument.partition("=")
    if not sep:
        return argument if argument.startswith("-") else API_TOKEN_FROM_ENVIRONMENT
    if "api_token" not in key:
        inside = _redact_json(value)
        if inside is not None:
            return f"{key}={inside}"
    return f"{key}={API_TOKEN_FROM_ENVIRONMENT}"


def is_finite(value: Any) -> bool:
    """False when ``value`` holds a NaN or an infinity anywhere; other values (text, flags, None) pass."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(is_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(is_finite(item) for item in value)
    return True


def numeric_draw(value: Any) -> list[float] | None:
    """A recorded draw as its finite numbers (a scalar is one); None for anything else: null, text, a flag, NaN."""
    values = value if isinstance(value, (list, tuple)) else [value]
    if not values or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in values):
        return None
    numbers = [float(item) for item in values]
    return numbers if all(math.isfinite(item) for item in numbers) else None


__all__ = [
    "API_TOKEN_FROM_ENVIRONMENT",
    "CREDENTIAL_ENVIRONMENT",
    "file_sha256",
    "is_finite",
    "jsonl_bytes",
    "numeric_draw",
    "publish_new",
    "read_jsonl",
    "recorded",
    "redact_api_tokens",
    "replace_atomic",
    "sha256_bytes",
]
