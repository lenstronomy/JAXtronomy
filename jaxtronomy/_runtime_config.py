"""Runtime configuration helpers for platform-specific JAX settings."""

from __future__ import annotations

import os
import sys
from typing import Optional

import jax

_CONFIGURED: bool = False
_LAST_VALUE: Optional[bool] = None


def _parse_env_override(raw: Optional[str]) -> Optional[bool]:
    """Parse JAXTRONOMY_ENABLE_X64 values.

    Returns:
        - True for explicit enable values
        - False for explicit disable values
        - None for unset/auto/unknown values
    """

    if raw is None:
        return None
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    if value in {"", "auto"}:
        return None
    return None


def is_macos_metal_backend() -> bool:
    """Return True when running on macOS with JAX Metal backend."""

    if sys.platform != "darwin":
        return False
    try:
        if str(jax.default_backend()).strip().lower() == "metal":
            return True
    except Exception:
        pass
    try:
        return any(
            str(getattr(device, "platform", "")).strip().lower() == "metal"
            for device in jax.devices()
        )
    except Exception:
        return False


def configure_jax_precision_for_runtime() -> bool:
    """Set jax_enable_x64 with a Mac/Metal-specific default.

    Default behavior:
      - macOS + Metal backend: disable x64
      - all other backends/platforms: enable x64

    Env override:
      - JAXTRONOMY_ENABLE_X64 in {1,true,yes,on} forces enable
      - JAXTRONOMY_ENABLE_X64 in {0,false,no,off} forces disable
      - unset/auto uses default behavior
    """

    global _CONFIGURED, _LAST_VALUE
    if _CONFIGURED and _LAST_VALUE is not None:
        return bool(_LAST_VALUE)

    override = _parse_env_override(os.environ.get("JAXTRONOMY_ENABLE_X64"))
    if override is None:
        enable_x64 = not is_macos_metal_backend()
    else:
        enable_x64 = bool(override)

    jax.config.update("jax_enable_x64", bool(enable_x64))
    _LAST_VALUE = bool(enable_x64)
    _CONFIGURED = True
    return bool(enable_x64)


def _reset_runtime_config_cache_for_tests() -> None:
    """Reset helper cache to make runtime policy tests deterministic."""

    global _CONFIGURED, _LAST_VALUE
    _CONFIGURED = False
    _LAST_VALUE = None
