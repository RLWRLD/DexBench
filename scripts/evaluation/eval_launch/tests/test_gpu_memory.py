"""gpu_memory.query against what nvidia-smi prints: numbers, or the bracketed words it writes where it has none."""

from __future__ import annotations

import os

import pytest

from eval_launch import gpu_memory


@pytest.mark.parametrize("line, reading", [("30000, 46068", (30000.0, 46068.0)), ("[N/A], 46068", None),
                                           ("[Insufficient Permissions], [Insufficient Permissions]", None)])
def test_a_reading_is_numbers_or_nothing(tmp_path, monkeypatch, line, reading):
    """A field nvidia-smi cannot read is no reading (None), as its docstring says; it used to raise ValueError in the
    sampling thread of every unit on such a host."""
    smi = tmp_path / "nvidia-smi"
    smi.write_text(f"#!/bin/sh\necho '{line}'\n")
    smi.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    assert gpu_memory.query("0") == reading
