"""Backward-compatible config entrypoint for task modules.

Historically many task validators / prehooks did:

    from ... import config

from within a task subpackage like:
`tinker_cookbook.recipes.cua_rl.tasks.airbnb.<task_id>.validator`.

Because those files are nested two+ levels deep, `from ... import config` resolves
to `tinker_cookbook.recipes.cua_rl.tasks.config` (this file), *not* to the task
category config (e.g. `tasks.airbnb.config`).

We therefore keep this module as a thin dispatcher that resolves the correct
category (`demo`, `airbnb`, …) from the call stack and forwards to that
category's `config.py`.

New code should prefer importing category configs directly:

    from tinker_cookbook.recipes.cua_rl.tasks.airbnb import config
"""

from __future__ import annotations

import importlib
import inspect
import os
from functools import lru_cache
from typing import Optional

from ..executor.base import ApkConfig


def _maybe_env_category() -> Optional[str]:
    """Allow explicit override in unusual execution contexts (tests, scripts)."""
    for key in ("CUA_TASK_CATEGORY", "CUA_APP_NAME", "CUA_TASK_APP"):
        value = os.environ.get(key)
        if value:
            return value.strip()
    return None


def _category_from_module_name(module_name: str) -> Optional[str]:
    parts = module_name.split(".")
    if "tasks" not in parts:
        return None
    tasks_idx = parts.index("tasks")
    if tasks_idx + 1 >= len(parts):
        return None
    category = parts[tasks_idx + 1]
    # Guard against recursively resolving ourselves.
    if category == "config":
        return None
    return category


def _resolve_task_category_from_stack() -> Optional[str]:
    """Best-effort resolve category from caller stack."""
    # Env override wins.
    env = _maybe_env_category()
    if env:
        return env

    for frame_info in inspect.stack():
        module_name = frame_info.frame.f_globals.get("__name__")
        if not isinstance(module_name, str):
            continue
        category = _category_from_module_name(module_name)
        if category:
            return category
    return None


@lru_cache(maxsize=None)
def _import_category_config(category: str):
    return importlib.import_module(
        f"tinker_cookbook.recipes.cua_rl.tasks.{category}.config"
    )


def get_apk_config() -> ApkConfig:
    """Return APK config for the calling task category."""
    category = _resolve_task_category_from_stack()
    if not category:
        raise AttributeError(
            "Could not resolve task category for get_apk_config(). "
            "Import the category config directly (e.g. tasks.airbnb.config), "
            "or set env CUA_TASK_CATEGORY."
        )
    module = _import_category_config(category)
    if not hasattr(module, "get_apk_config"):
        raise AttributeError(
            f"Task category '{category}' has no get_apk_config() in its config module."
        )
    return module.get_apk_config()


def get_package_name() -> str:
    """Return app package name for the calling task category."""
    category = _resolve_task_category_from_stack()
    if not category:
        raise AttributeError(
            "Could not resolve task category for get_package_name(). "
            "Import the category config directly (e.g. tasks.airbnb.config), "
            "or set env CUA_TASK_CATEGORY."
        )
    module = _import_category_config(category)
    if not hasattr(module, "get_package_name"):
        # Fall back to ApkConfig if category only implemented get_apk_config().
        apk_config = module.get_apk_config() if hasattr(module, "get_apk_config") else None
        if apk_config and apk_config.package_name:
            return apk_config.package_name
        raise AttributeError(
            f"Task category '{category}' has no get_package_name() in its config module."
        )
    return module.get_package_name()


__all__ = ["get_apk_config", "get_package_name", "ApkConfig"]

