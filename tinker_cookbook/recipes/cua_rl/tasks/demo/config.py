"""Configuration module for Android Demo tasks."""
from __future__ import annotations

from ...executor.base import ApkConfig


def get_package_name() -> str:
    """Get the package name for Android Settings app."""
    return "com.android.settings"


def get_apk_config() -> ApkConfig:
    """Get APK configuration for demo tasks.
    
    Demo tasks use the built-in Android Settings app, so no APK installation is needed.
    """
    return ApkConfig(
        app_name="demo",
        cua_guide="You need to operate the android device to complete the task.",
        requires_apk=False,
        package_name=get_package_name(),
        launch_after_install=False,  # Not applicable since no APK
    )
