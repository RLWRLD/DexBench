# SPDX-License-Identifier: Apache-2.0
"""Whether Kit was launched with its XR runtime (``--xr``), read from carb settings."""

from __future__ import annotations


def xr_enabled() -> bool:
    """True when the running Kit has ``/app/xr/enabled`` set; False outside Kit."""
    try:
        import carb

        return bool(carb.settings.get_settings().get("/app/xr/enabled"))
    except Exception:  # noqa: BLE001 - no Kit, or no XR extension: not enabled
        return False
