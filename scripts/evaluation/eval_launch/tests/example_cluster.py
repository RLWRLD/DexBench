"""Cluster values for the shipped Kubernetes examples, whose site profile holds <...> placeholders that --k8s-job
refuses (launch.placeholders). Not a test module."""

from __future__ import annotations

from typing import Any

IMAGE = "registry.example/dexbench-arena@sha256:" + "0" * 64


def filled(config: dict[str, Any]) -> dict[str, Any]:
    """``config`` with the example profile's placeholders (image, namespace, volume claim) given real-looking values."""
    config["backend"]["image"] = IMAGE
    config["backend"]["k8s"].update(namespace="evals", volume_claim="shared")
    return config
