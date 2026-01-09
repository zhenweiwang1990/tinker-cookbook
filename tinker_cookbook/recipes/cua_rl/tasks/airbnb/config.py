"""Configuration module for Airbnb app."""
from __future__ import annotations

from ...executor.base import ApkConfig


def get_package_name() -> str:
    """Get the package name for the Airbnb app."""
    return "com.airbnb.clone"


def get_apk_config() -> ApkConfig:
    """Get APK configuration for Airbnb tasks.
    
    Airbnb tasks require the Airbnb clone APK to be installed.
    """
    return ApkConfig(
        app_name="airbnb",
        cua_guide="You need to operate the Airbnb app to complete the task.",
        requires_apk=True,
        apk_url="https://activate2-gbox-staging-public-assets.s3.us-west-2.amazonaws.com/test/airbnb.apk",
        package_name=get_package_name(),
        launch_after_install=True,
        main_activity=None,  # Will use default launcher activity
    )

