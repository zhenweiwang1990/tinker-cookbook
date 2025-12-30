"""ADB client module - re-export from tasks.adb for backward compatibility."""
from __future__ import annotations

# Re-export everything from tasks.adb
from tinker_cookbook.recipes.cua_rl.tasks.adb import (
    AdbClient,
    AdbError,
    Device,
    list_connected_devices,
)

__all__ = ["AdbClient", "AdbError", "Device", "list_connected_devices"]

