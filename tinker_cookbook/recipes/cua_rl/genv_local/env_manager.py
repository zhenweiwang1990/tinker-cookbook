from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# genv LocalEnv is heavy (docker + emulator). We strictly serialize access globally.
_global_env_semaphore: asyncio.Semaphore = asyncio.Semaphore(1)


def _require_genv() -> Any:
    """
    Import genv lazily with a clear error message.

    We keep this local so cua_rl can still import without genv installed.
    """
    try:
        import genv  # noqa: F401
        from genv.sdk.envs import LocalEnv  # type: ignore

        return LocalEnv
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "genv is required for env_mode='genv_local'. "
            "Install it with: pip install -e /Users/zhenwei/workspace/genv-umetrip/rl"
        ) from e


def _parse_port_from_endpoint(endpoint: str) -> int:
    parsed = urlparse(endpoint)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid backendEndpoint: {endpoint!r}")
    if parsed.port is None:
        raise ValueError(f"backendEndpoint missing port: {endpoint!r}")
    return int(parsed.port)


@dataclass(frozen=True)
class PreparedGenvEnv:
    env: Any
    backend_endpoint: str
    app_endpoint: str


async def acquire_env_slot() -> asyncio.Semaphore:
    """
    Return the global semaphore used to serialize genv env usage.

    Callers should use:
      async with (await acquire_env_slot()):
         ...
    """
    return _global_env_semaphore


def prepare_local_env(env_config: dict[str, Any]) -> PreparedGenvEnv:
    """
    Create a genv LocalEnv and fix the Android app endpoint for emulator networking.

    genv's LocalEnv sets backendEndpoint to something like http://localhost:<port>.
    The Android emulator cannot reach the host via localhost, so we rewrite the app's
    global setting `genv_umetrip_api_endpoint` to http://10.0.2.2:<port>.
    """
    LocalEnv = _require_genv()

    env = LocalEnv.create(env_config=env_config)

    backend_endpoint: Optional[str] = None
    try:
        resolved = getattr(env, "_resolved_vars", {}) or {}
        backend_endpoint = resolved.get("backendEndpoint")
    except Exception:
        backend_endpoint = None

    if not backend_endpoint:
        raise RuntimeError(
            "genv env did not expose _resolved_vars['backendEndpoint']; cannot configure app endpoint."
        )

    port = _parse_port_from_endpoint(str(backend_endpoint))
    app_endpoint = f"http://10.0.2.2:{port}"

    adb_device = getattr(env, "adb_device", None)
    if adb_device is None:
        raise RuntimeError("genv env has no adb_device; cannot set global setting for app endpoint.")

    # Ensure the app reads the correct endpoint from Android Global Settings.
    adb_device.set_global_setting("genv_umetrip_api_endpoint", app_endpoint)

    # Best-effort verification.
    try:
        actual = adb_device.get_global_setting("genv_umetrip_api_endpoint")
        if actual != app_endpoint:
            logger.warning(
                "[genv_local] global setting mismatch: expected=%s actual=%s",
                app_endpoint,
                actual,
            )
    except Exception:
        pass

    logger.info("[genv_local] backend_endpoint=%s app_endpoint=%s", backend_endpoint, app_endpoint)
    return PreparedGenvEnv(env=env, backend_endpoint=str(backend_endpoint), app_endpoint=app_endpoint)


def close_env(env: Any) -> None:
    try:
        env.close()
    except Exception as e:
        logger.warning("[genv_local] env.close() failed: %s", e)

