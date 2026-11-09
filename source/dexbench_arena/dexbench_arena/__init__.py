# Copyright (c) 2026, RLWRLD Inc. and contributors.
# SPDX-License-Identifier: Apache-2.0
"""DexBench's IsaacLab-Arena extension package."""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
EXTENSION_ROOT = PACKAGE_DIR.parent
REPO_ROOT = EXTENSION_ROOT.parents[1]
ASSETS_DIR = REPO_ROOT / "assets"
RBY1_WUJI2_BETA2_ASSETS_DIR = ASSETS_DIR / "embodiments" / "simready" / "rby1m_wujihand2_beta2" / "source"


def register_components() -> None:
    """Import DexBench environment modules that register Arena components."""

    from .assets import retargeter_library as _retargeter_library  # noqa: F401
    from . import environments as _environments  # noqa: F401


__all__ = [
    "ASSETS_DIR",
    "EXTENSION_ROOT",
    "PACKAGE_DIR",
    "RBY1_WUJI2_BETA2_ASSETS_DIR",
    "REPO_ROOT",
    "register_components",
]
